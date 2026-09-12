"""予算を決めたうえでの買い目の組み立て。

単勝オッズしか手元にない状況で馬連・3連複・ワイドの妥当性を判断するため、
  1. 単勝オッズを市場の勝率推定とみなして正規化する
  2. その市場勝率から Harville で各券種の的中率を出す
  3. 控除率を戻して「市場が付けるであろう配当」を推定する
  4. AI の推定確率と突き合わせて期待値を出す
という順で組む。

賭け金は分数ケリーで配分する。ケリーをそのまま使うと推定確率の誤差に対して
破滅的に脆いので、係数を掛けて縮める。
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from .betting import LAMBDA2, LAMBDA3, exacta_pair_prob, trio_prob
from .engine import Assessment

# JRAの控除率
TAKEOUT = {"単勝": 0.20, "馬連": 0.225, "ワイド": 0.235, "3連複": 0.25}

# 単勝オッズだけから複勝系の配当を当てるのは原理的に無理がある。
#
# Harville に Lo-Bacon-Shone のべき乗補正を入れても、実勢の経験則
# (馬連 ≒ 単勝オッズの積 / 3) に対してなお1.5〜1.9倍高く出る。
# どちらが正しいとも言い切れないので、両者の中間よりやや保守側に置く。
# 配当を低く見積もる誤りは「買わない」方向に効くので、買う判断をする側としては安全。
# 実際の購入前には必ず本物のオッズを確認すること。
PAYOUT_HAIRCUT = {"単勝": 1.0, "馬連": 0.75, "ワイド": 0.70, "3連複": 0.70}

# ケリー係数。1.0が理論最適だが、推定確率の誤差に極めて弱いので大きく縮める。
KELLY_FRACTION = 0.25

# AI の推定確率を市場(単勝オッズ)へどれだけ寄せるか。1.0=AIを全面的に信じる。
#
# これは飾りではなく必須の処理。AI はこのレースでグランヴィノスの勝率を23.8%と見るが
# 市場は4.0%で、6倍の乖離がある。この乖離を馬連・ワイド・3連複と複数脚に
# 掛け合わせると、期待回収率が690%といった到底あり得ない数字が出てしまう。
# JRAの重賞市場はそれなりに効率的で、こちらは過去データによる検証を一度もしていない。
# バックテストで優位性を実証するまでは、市場を主、AIを従として扱うのが正しい。
AI_WEIGHT = 0.35


@dataclass
class Ticket:
    kind: str
    legs: tuple[int, ...]
    names: tuple[str, ...]
    ai_prob: float
    payout: float          # 推定配当(倍)
    stake: int = 0

    @property
    def ev(self) -> float:
        return self.ai_prob * self.payout

    @property
    def label(self) -> str:
        return "-".join(str(n) for n in self.legs)


def market_probs(assessments: list[Assessment]) -> dict[int, float]:
    """単勝オッズを正規化して市場の勝率推定にする(控除率を割り戻す)。"""
    raw = {a.horse.num: 1.0 / a.horse.odds for a in assessments if a.horse.odds}
    z = sum(raw.values())
    return {k: v / z for k, v in raw.items()}


class _View:
    """Harville 計算を任意の勝率で回すための薄いラッパ。"""

    def __init__(self, num: int, prob: float):
        self.horse = type("H", (), {"num": num})()
        self.win_prob = prob


def _market_views(assessments: list[Assessment]) -> list[_View]:
    mp = market_probs(assessments)
    return [_View(a.horse.num, mp[a.horse.num]) for a in assessments if a.horse.num in mp]


def blended_views(assessments: list[Assessment], w: float = AI_WEIGHT) -> list[_View]:
    """AI勝率と市場勝率を w : (1-w) で混ぜた勝率。以降の判断はすべてこれで行う。"""
    mp = market_probs(assessments)
    out = []
    for a in assessments:
        if a.horse.num not in mp:
            continue
        out.append(_View(a.horse.num, w * a.win_prob + (1 - w) * mp[a.horse.num]))
    z = sum(v.win_prob for v in out)
    for v in out:
        v.win_prob /= z
    return out


def _pair_in_top3(views, a, b) -> float:
    """A と B がともに3着以内に入る確率(=ワイドの的中率)。"""
    return sum(trio_prob(a, b, c, views) for c in views if c is not a and c is not b)


def build_tickets(
    assessments: list[Assessment], pool_size: int = 8, min_ev: float = 1.15,
    min_prob: float = 0.02, w: float = AI_WEIGHT,
) -> list[Ticket]:
    """買える候補券をすべて列挙し、期待値で絞る。

    的中率は「AIを市場へ寄せた勝率」から、配当は「市場の勝率」から出す。
    """
    views = _market_views(assessments)
    vmap = {v.horse.num: v for v in views}
    bl = blended_views(assessments, w)
    bmap = {v.horse.num: v for v in bl}
    ai = {a.horse.num: a for a in assessments}
    pool_nums = [a.horse.num for a in assessments[:pool_size]]
    pool = [bmap[n] for n in pool_nums if n in bmap]
    tickets: list[Ticket] = []

    def payout(kind: str, mkt_prob: float) -> float:
        if mkt_prob <= 0:
            return 0.0
        return (1.0 - TAKEOUT[kind]) / mkt_prob * PAYOUT_HAIRCUT[kind]

    # 単勝は実オッズがそのまま使える
    for x in pool:
        n = x.horse.num
        if ai[n].horse.odds:
            tickets.append(Ticket("単勝", (n,), (ai[n].horse.name,), x.win_prob, ai[n].horse.odds))

    for x, y in combinations(pool, 2):
        nx, ny = x.horse.num, y.horse.num
        legs = tuple(sorted((nx, ny)))
        names = tuple(ai[n].horse.name for n in legs)

        mkt = exacta_pair_prob(vmap[nx], vmap[ny], views)
        tickets.append(Ticket("馬連", legs, names, exacta_pair_prob(x, y, bl), payout("馬連", mkt)))

        mkt_w = _pair_in_top3(views, vmap[nx], vmap[ny])
        tickets.append(Ticket("ワイド", legs, names, _pair_in_top3(bl, x, y), payout("ワイド", mkt_w)))

    for x, y, z in combinations(pool, 3):
        legs = tuple(sorted((x.horse.num, y.horse.num, z.horse.num)))
        names = tuple(ai[n].horse.name for n in legs)
        mkt = trio_prob(vmap[legs[0]], vmap[legs[1]], vmap[legs[2]], views)
        tickets.append(Ticket("3連複", legs, names, trio_prob(x, y, z, bl), payout("3連複", mkt)))

    good = [t for t in tickets if t.ev >= min_ev and t.ai_prob >= min_prob]
    return sorted(good, key=lambda t: -t.ev)


def allocate(
    tickets: list[Ticket], budget: int, unit: int = 100, max_lines: int = 8,
    max_horse_share: float = 1.0, min_units: int = 0,
) -> list[Ticket]:
    """分数ケリーで賭け金を配分し、単位金額に丸める。

    max_horse_share を1未満にすると、1頭に依存する買い目の合計額に上限をかける。
    ケリーはエッジのある1頭に集中させるのが最適解だが、モデルが読み違えていた場合に
    全額が飛ぶ。休み明けのような不確実性が大きい馬が軸のときの保険。
    """
    n_units = budget // unit
    if n_units <= 0 or not tickets:
        return []

    scored = []
    for t in tickets:
        b = t.payout - 1.0
        if b <= 0:
            continue
        kelly = (t.ai_prob * b - (1 - t.ai_prob)) / b     # 最適賭け金比率
        if kelly <= 0:
            # 期待値が1を下回る券。自動選択なら買わないが、
            # 人が「この買い目で」と指定した場合は配分対象に残す
            # (本来は買うべきでないことは呼び出し側で警告する)
            if min_units <= 0:
                continue
            kelly = 0.0
        scored.append((t, kelly * KELLY_FRACTION))

    if min_units > 0 and all(w <= 0 for _, w in scored):
        # 全点が期待値マイナスのときは期待値の大小で按分する
        scored = [(t, max(t.ev, 1e-6)) for t, _ in scored]

    scored.sort(key=lambda x: -x[1])
    scored = scored[:max_lines]
    if not scored:
        return []

    # 買い目を人が指定した場合は、全点に最低1単位を確保してから残りを配分する
    floor = 0
    if min_units > 0:
        floor = min_units * len(scored)
        if floor > n_units:
            raise ValueError(
                f"予算が足りません: {len(scored)}点 × {min_units}単位 に "
                f"{floor * unit:,}円 必要ですが予算は {budget:,}円です"
            )
        n_units -= floor

    if max_horse_share < 1.0:
        scored = _cap_concentration(scored, max_horse_share)

    total = sum(w for _, w in scored)
    # 最大剰余法で単位数を割り振る(丸め落ちを取りこぼさない)
    exact = [(t, w / total * n_units) for t, w in scored]
    alloc = [(t, int(x)) for t, x in exact]
    rest = n_units - sum(u for _, u in alloc)
    order = sorted(range(len(exact)), key=lambda i: -(exact[i][1] - int(exact[i][1])))
    for i in order[:rest]:
        alloc[i] = (alloc[i][0], alloc[i][1] + 1)

    out = []
    for t, u in alloc:
        total_u = u + min_units
        if total_u > 0:
            t.stake = total_u * unit
            out.append(t)
    return sorted(out, key=lambda t: (-t.stake, -t.ev))


def expand_formation(groups: list[list[int]], kind: str, field: list[int]) -> list[tuple[int, ...]]:
    """フォーメーションを組合せに展開する。

    例 3連複 [14] - [10,15] - [全] なら、14と(10か15)を含む3頭の組合せすべて。
    3連複・ワイド・馬連は順不同なので、重複を除いた集合として扱う。
    """
    size = {"3連複": 3, "馬連": 2, "ワイド": 2, "単勝": 1}[kind]
    if len(groups) != size:
        raise ValueError(f"{kind}は{size}列で指定してください(指定は{len(groups)}列)")

    seen: set[tuple[int, ...]] = set()
    out: list[tuple[int, ...]] = []

    def rec(i: int, chosen: list[int]):
        if i == size:
            key = tuple(sorted(chosen))
            if len(set(chosen)) == size and key not in seen:
                seen.add(key)
                out.append(key)
            return
        for n in groups[i]:
            if n in chosen:
                continue
            rec(i + 1, chosen + [n])

    rec(0, [])
    return out


def _normalize_spec(spec: str) -> str:
    """券の区切りを ";" に統一する。

    "," は列内の区切り(フォーメーション)にも使うので、単純に置換できない。
    "," で割った断片が全て券種指定(":"を含む)のときだけ、券の区切りとみなす。
    """
    if ";" in spec:
        return spec
    parts = [x.strip() for x in spec.split(",") if x.strip()]
    if len(parts) > 1 and all(":" in x for x in parts):
        return ";".join(parts)
    return spec


def parse_spec(spec: str, field: list[int]) -> list[tuple[str, tuple[int, ...]]]:
    """買い目文字列を (券種, 馬番の組) のリストに展開する。

    単点     : "3連複:4-9-13"
    フォーメ : "3連複:14-10,15-全"   (列は - 区切り、列内は , 区切り、全=全馬)
    軸流し   : "ワイド:13-全"
    """
    out: list[tuple[str, tuple[int, ...]]] = []
    for item in _normalize_spec(spec).split(";"):
        item = item.strip()
        if not item:
            continue
        kind, _, body = item.partition(":")
        kind = kind.strip()
        body, _, _odds = body.partition("@")      # 実オッズ指定は tickets_from_spec 側で拾う
        groups = []
        for g in body.split("-"):
            g = g.strip()
            if g in ("全", "*"):
                groups.append(list(field))
            else:
                groups.append([int(x) for x in g.split(",")])
        if all(len(g) == 1 for g in groups):
            # 単点指定でも列数は検証する("3連複:13-15" のような指定を弾く)
            legs = tuple(sorted(g[0] for g in groups))
            expected = {"3連複": 3, "馬連": 2, "ワイド": 2, "単勝": 1}.get(kind)
            if expected is None:
                raise ValueError(f"未知の券種: {kind}")
            if len(legs) != expected or len(set(legs)) != expected:
                raise ValueError(
                    f"{kind}は{expected}頭で指定してください(指定は{len(set(legs))}頭)")
            out.append((kind, legs))
        else:
            for legs in expand_formation(groups, kind, field):
                out.append((kind, legs))
    return out


def tickets_from_spec(spec: str, assessments: list[Assessment], w: float = AI_WEIGHT) -> list[Ticket]:
    """買い目を明示指定して組む。例 "単勝:13,馬連:9-13,3連複:4-9-13"

    build_tickets が期待値で自動的に絞るのに対し、こちらは買う券を人が決める。
    的中率と推定配当の計算は build_tickets と同じ経路を通す。
    """
    views = _market_views(assessments)
    vmap = {v.horse.num: v for v in views}
    bl = blended_views(assessments, w)
    bmap = {v.horse.num: v for v in bl}
    ai = {a.horse.num: a for a in assessments}

    def payout(kind: str, mkt_prob: float) -> float:
        if mkt_prob <= 0:
            return 0.0
        return (1.0 - TAKEOUT[kind]) / mkt_prob * PAYOUT_HAIRCUT[kind]

    field = [a.horse.num for a in assessments]

    # "3連複:9-13-15@42.9" のように実オッズを指定できる。
    # 推定配当より実オッズが分かっているならそちらを使うべき。
    real: dict[tuple[str, tuple[int, ...]], float] = {}
    for item in _normalize_spec(spec).split(";"):
        if "@" not in item:
            continue
        head, _, od = item.partition("@")
        k, _, body = head.partition(":")
        for kk, legs in parse_spec(f"{k.strip()}:{body}", field):
            real[(kk, legs)] = float(od)

    out: list[Ticket] = []
    for kind, legs in parse_spec(spec, field):
        if kind not in TAKEOUT:
            raise ValueError(f"未知の券種: {kind}")
        for n in legs:
            if n not in ai:
                raise ValueError(f"馬番{n}はこのレースにいません")
        names = tuple(ai[n].horse.name for n in legs)

        if kind == "単勝":
            od = real.get((kind, legs), ai[legs[0]].horse.odds)
            out.append(Ticket(kind, legs, names, bmap[legs[0]].win_prob, od))
        elif kind == "馬連":
            mkt = exacta_pair_prob(vmap[legs[0]], vmap[legs[1]], views)
            prob = exacta_pair_prob(bmap[legs[0]], bmap[legs[1]], bl)
            out.append(Ticket(kind, legs, names, prob,
                               real.get((kind, legs), payout(kind, mkt))))
        elif kind == "ワイド":
            mkt = _pair_in_top3(views, vmap[legs[0]], vmap[legs[1]])
            prob = _pair_in_top3(bl, bmap[legs[0]], bmap[legs[1]])
            out.append(Ticket(kind, legs, names, prob,
                               real.get((kind, legs), payout(kind, mkt))))
        else:  # 3連複
            mkt = trio_prob(*(vmap[n] for n in legs), field=views)
            prob = trio_prob(*(bmap[n] for n in legs), field=bl)
            out.append(Ticket(kind, legs, names, prob,
                               real.get((kind, legs), payout(kind, mkt))))
    return out


# ---------------------------------------------------------------- 収支の検証
def simulate(
    picks: list[Ticket], assessments: list[Assessment], n: int = 40000,
    w: float = AI_WEIGHT, seed: int = 20260912,
) -> dict:
    """買い目全体の収支分布をモンテカルロで出す。

    馬券同士は強く相関する(同じ馬を含む)ので、的中率を独立近似で掛け合わせては
    いけない。Plackett-Luce で着順そのものを繰り返し生成して評価する。
    """
    import random

    rng = random.Random(seed)
    views = blended_views(assessments, w)
    nums = [v.horse.num for v in views]
    weights = [v.win_prob for v in views]
    spent = sum(t.stake for t in picks)

    returns: list[float] = []
    hits = 0
    for _ in range(n):
        # 着順を上位3着まで逐次サンプリング
        pool_n, pool_w = list(nums), list(weights)
        top: list[int] = []
        for step in range(3):
            # 2着・3着の抽出はべき乗補正を掛ける(確率計算側と揃える)
            lam = (1.0, LAMBDA2, LAMBDA3)[step]
            ws = [w_ ** lam for w_ in pool_w]
            total = sum(ws)
            r = rng.random() * total
            acc = 0.0
            for i, wt in enumerate(ws):
                acc += wt
                if r <= acc:
                    top.append(pool_n[i])
                    pool_n.pop(i); pool_w.pop(i)
                    break
        ret = 0.0
        for t in picks:
            if t.kind == "単勝":
                ok = top[0] == t.legs[0]
            elif t.kind == "馬連":
                ok = set(t.legs) <= set(top[:2])
            elif t.kind == "ワイド":
                ok = set(t.legs) <= set(top)
            else:
                ok = set(t.legs) == set(top)
            if ok:
                ret += t.stake * t.payout
        returns.append(ret)
        if ret > 0:
            hits += 1

    returns.sort()
    mean = sum(returns) / n
    return {
        "spent": spent,
        "hit_rate": hits / n,
        "profit_rate": sum(1 for r in returns if r > spent) / n,
        "mean_return": mean,
        "roi": mean / spent if spent else 0.0,
        "median": returns[n // 2],
        "p90": returns[int(n * 0.9)],
        "max": returns[-1],
    }


def _cap_concentration(scored, max_share: float):
    """特定の1頭に賭け金が偏りすぎないよう重みを削る。"""
    total = sum(w for _, w in scored)
    if total <= 0:
        return scored
    counts: dict[int, float] = {}
    for t, w in scored:
        for n in t.legs:
            counts[n] = counts.get(n, 0.0) + w
    worst = max(counts.items(), key=lambda kv: kv[1]) if counts else None
    if worst is None or worst[1] / total <= max_share:
        return scored
    num, share = worst
    factor = (max_share * total - (total - share)) / share if share else 1.0
    factor = max(0.0, min(factor, 1.0))
    return [(t, w * factor if num in t.legs else w) for t, w in scored]
