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


def exacta_pair_prob(a: Assessment, b: Assessment) -> float:
    """a と b が1,2着を占める確率(順不同)。"""
    pa, pb = a.win_prob, b.win_prob
    t1 = pa * (pb / (1 - pa)) if pa < 1 else 0.0
    t2 = pb * (pa / (1 - pb)) if pb < 1 else 0.0
    return t1 + t2


def trio_prob(a: Assessment, b: Assessment, c: Assessment) -> float:
    """3頭が上位3着を占める確率。"""
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
