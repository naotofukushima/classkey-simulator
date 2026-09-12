"""ハンデ(斤量)評価。

ハンデ戦の本質は「絶対斤量」ではなく
  (1) 馬体に対する負担率  (2) 前走からの増減  (3) 実績に対する斤量の妥当性
の3点。ここを分離して評価する。
"""
from __future__ import annotations

from statistics import mean

from .models import Horse

# 負担率の差1ポイント(=1%)あたりの指数換算
RATIO_TO_POINTS = 250.0
# 前走比の斤量増減1kgあたり
DELTA_TO_POINTS = 0.55


def burden_points(horse: Horse, all_horses: list[Horse]) -> tuple[float, dict]:
    """負担率ベースのハンデ評価。"""
    ratios = [h.burden_ratio for h in all_horses]
    avg = mean(ratios)
    ratio = horse.burden_ratio
    pts = (avg - ratio) * RATIO_TO_POINTS

    detail = {
        "burden_ratio_pct": ratio * 100,
        "field_avg_pct": avg * 100,
        "points": pts,
    }
    return pts, detail


def weight_change_points(horse: Horse) -> tuple[float, float | None]:
    """前走からの斤量増減。増えた馬は割引、据え置き・減は加点。"""
    turf_runs = [r for r in horse.runs if r.is_turf]
    if not turf_runs:
        return 0.0, None
    last = max(turf_runs, key=lambda r: r.date)
    delta = horse.carried - last.carried

    # 3歳馬が春の定量戦(57kg)から秋のハンデ戦に来る際の減量は
    # 年齢定量の産物であって、ハンデキャッパーの評価ではない。
    # 絶対的な軽さは burden_points 側で評価済みなので、ここでは半分に割る。
    scale = 0.5 if (horse.age == 3 and last.three_yo_only) else 1.0
    return -delta * DELTA_TO_POINTS * scale, delta


def layoff_points(horse: Horse, race_date) -> tuple[float, int | None]:
    """休み明け補正。間隔と、その馬/厩舎の休み明け実績(freshness)で調整。"""
    if not horse.runs:
        return 0.0, None
    last = max(horse.runs, key=lambda r: r.date)
    days = (race_date - last.date).days

    if days <= 14:
        base = -0.3          # 詰めすぎ
    elif days <= 60:
        base = 0.0
    elif days <= 110:
        base = -0.3
    elif days <= 180:
        base = -1.0
    elif days <= 280:
        base = -2.2
    else:
        base = -3.2
    return base + horse.freshness, days
