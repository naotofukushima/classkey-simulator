"""買い目の組み立て。

AI の推定確率と実オッズの乖離(=期待値)から、券種ごとに妙味のある組合せを選ぶ。
単勝オッズしか手元にない前提で、馬連/3連複は
  - 馬連: P(A,B が1,2着) を Harville 近似
  - 3連複: P(3頭が上位3着を占める)
から理論オッズを出し、実際の配当見込みと比較する。
"""
from __future__ import annotations

from itertools import combinations

from .engine import Assessment

# JRA の控除率(単勝約20%, 馬連約22.5%, 3連複約25%)
TAKEOUT = {"win": 0.20, "quinella": 0.225, "trio": 0.25}


# Lo-Bacon-Shone のべき乗補正。
#
# 素の Harville モデル(勝率をそのまま2着・3着の条件付き確率に使う)は、
# 穴馬が2着・3着に来る確率を系統的に過小評価する。その結果、穴絡みの
# 馬連・3連複の「推定配当」が実勢よりはっきり高く出てしまう。
# 実測では 2着は勝率の0.76乗、3着は0.62乗に比例させると当てはまりが良い。
LAMBDA2 = 0.76
LAMBDA3 = 0.62


def _cond(probs: list[float], used: list[int], target: int, lam: float) -> float:
    """既に決着した馬を除いた集合の中で、target が次に来る条件付き確率。"""
    denom = sum(p ** lam for i, p in enumerate(probs) if i not in used)
    if denom <= 0:
        return 0.0
    return probs[target] ** lam / denom


def exacta_pair_prob(a: Assessment, b: Assessment, field: list[Assessment] | None = None) -> float:
    """a と b が1,2着を占める確率(順不同)。"""
    if field is None:
        pa, pb = a.win_prob, b.win_prob
        t1 = pa * (pb / (1 - pa)) if pa < 1 else 0.0
        t2 = pb * (pa / (1 - pb)) if pb < 1 else 0.0
        return t1 + t2

    probs = [x.win_prob for x in field]
    ia, ib = field.index(a), field.index(b)
    return (probs[ia] * _cond(probs, [ia], ib, LAMBDA2)
            + probs[ib] * _cond(probs, [ib], ia, LAMBDA2))


def trio_prob(a: Assessment, b: Assessment, c: Assessment,
              field: list[Assessment] | None = None) -> float:
    """3頭が上位3着を占める確率。"""
    if field is None:
        total = 0.0
        trio = [a, b, c]
        for i in range(3):
            for j in range(3):
                if j == i:
                    continue
                k = 3 - i - j
                x, y, z = trio[i], trio[j], trio[k]
                d1 = 1 - x.win_prob
                d2 = d1 - y.win_prob
                if d1 <= 0 or d2 <= 0:
                    continue
                total += x.win_prob * (y.win_prob / d1) * (z.win_prob / d2)
        return total

    probs = [x.win_prob for x in field]
    idx = [field.index(a), field.index(b), field.index(c)]
    total = 0.0
    for i in range(3):
        for j in range(3):
            if j == i:
                continue
            k = 3 - i - j
            x, y, z = idx[i], idx[j], idx[k]
            total += (probs[x]
                      * _cond(probs, [x], y, LAMBDA2)
                      * _cond(probs, [x, y], z, LAMBDA3))
    return total


def fair_payout(prob: float, kind: str) -> float:
    """控除率込みの「これ以上なら買える」最低オッズ。"""
    if prob <= 0:
        return float("inf")
    return 1.0 / prob


def value_win_bets(
    assessments: list[Assessment], min_ev: float = 1.15, min_prob: float = 0.03
) -> list[tuple[Assessment, float]]:
    """期待値プラスの単勝候補。

    min_prob を設けているのは、推定勝率が 1% 未満の馬は指数のわずかな誤差で
    期待値が簡単に 1 を超えてしまい、「大穴の期待値プラス」が
    モデルのノイズでしかなくなるため。
    """
    out = []
    for a in assessments:
        ev = a.ev
        if ev is not None and ev >= min_ev and a.win_prob >= min_prob:
            out.append((a, ev))
    return sorted(out, key=lambda x: -x[1])


def quinella_candidates(assessments: list[Assessment], axis: Assessment, partners: list[Assessment]):
    rows = []
    for p in partners:
        prob = exacta_pair_prob(axis, p)
        rows.append((axis, p, prob, fair_payout(prob, "quinella")))
    return sorted(rows, key=lambda r: -r[2])


def trio_candidates(assessments: list[Assessment], pool: list[Assessment], top: int = 12):
    rows = []
    for a, b, c in combinations(pool, 3):
        prob = trio_prob(a, b, c)
        rows.append((a, b, c, prob, fair_payout(prob, "trio")))
    rows.sort(key=lambda r: -r[3])
    return rows[:top]
