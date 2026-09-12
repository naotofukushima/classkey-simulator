"""能力指数(パフォーマンス・レーティング)の算出。

1走ごとに「クラス格 × 着差 × 斤量 × 頭数」を合成した指数を作り、
直近走への重み付け平均でその馬の素の能力を推定する。
"""
from __future__ import annotations

import math
from datetime import date

from .models import (
    CLASS_BASE,
    SURFACE_MISMATCH_DISCOUNT,
    THREE_YO_ONLY_DISCOUNT,
    Horse,
    PastRun,
    RaceConditions,
)

# 着差1秒あたりの指数換算(芝2000m級)。0.2秒≒1馬身強。
SEC_TO_POINTS = 8.0
# 斤量1kgあたりの指数換算。定説の「1kg≒0.2秒」に近い値。
KG_TO_POINTS = 1.6
MARGIN_CAP = 4.0


def run_rating(run: PastRun, target_surface: str = "芝", sex: str = "牡") -> float:
    """1走のパフォーマンス指数。"""
    base = CLASS_BASE.get(run.grade, 70.0)
    if run.three_yo_only:
        base -= THREE_YO_ONLY_DISCOUNT

    margin = max(-1.0, min(run.margin, MARGIN_CAP))
    pts = base - margin * SEC_TO_POINTS

    # 背負って好走した価値を加点 / 軽ハンデでの好走は割り引く
    pts += (run.carried - run.reference_weight(sex)) * KG_TO_POINTS

    # 少頭数のレースは価値を割り引く
    pts += (min(run.field_size, 18) - 12) * 0.12

    # 芝のレースを予想するのにダート実績は直接は使えない
    if run.surface != target_surface:
        pts -= SURFACE_MISMATCH_DISCOUNT

    return pts


def recency_weight(run_date: date, race_date: date, half_life_days: float = 150.0) -> float:
    """直近走を重く見る指数減衰。"""
    days = max((race_date - run_date).days, 0)
    return 0.5 ** (days / half_life_days)


def ability(horse: Horse, race: RaceConditions, top_n: int = 3) -> tuple[float, list[tuple[PastRun, float]]]:
    """馬の能力指数と、その根拠となった走の内訳を返す。

    「直近かつ上位のパフォーマンス」を重視し、凡走1回で評価を落としすぎない
    よう上位 top_n 走の加重平均 + ベストパフォーマンスのブレンドを取る。
    """
    scored = [
        (r, run_rating(r, race.surface, horse.sex), recency_weight(r.date, race.date))
        for r in horse.runs
    ]
    if not scored:
        return 0.0, []

    # 重み付き指数で降順に並べ、上位 top_n を採用
    ranked = sorted(scored, key=lambda x: x[1] + math.log(max(x[2], 1e-6)) * 1.5, reverse=True)
    top = ranked[:top_n]

    wsum = sum(w for _, _, w in top)
    if wsum <= 0:
        weighted = sum(p for _, p, _ in top) / len(top)
    else:
        weighted = sum(p * w for _, p, w in top) / wsum

    best = max(p for _, p, _ in top)
    value = 0.72 * weighted + 0.28 * best

    breakdown = [(r, p) for r, p, _ in top]
    return value, breakdown


def improving_form(horse: Horse) -> tuple[float, str]:
    """上昇度。クラスを上げながら勝ち上がっている馬は指数が実力に追いつかない。"""
    turf = sorted([r for r in horse.runs if r.is_turf], key=lambda r: r.date, reverse=True)
    recent = turf[:3]
    if len(recent) < 2:
        return 0.0, ""
    wins = sum(1 for r in recent if r.finish == 1)
    bases = [CLASS_BASE.get(r.grade, 70.0) for r in recent]
    rising = bases[0] >= max(bases[1:])
    if wins >= 2 and rising:
        return 1.2, "直近3走で2勝以上かつ昇級中"
    if wins >= 1 and rising and recent[0].finish <= 3:
        return 0.5, "昇級後も崩れず"
    return 0.0, ""
