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


# 適正体重から1kg外れるごとの減点
WEIGHT_DEV_POINTS = 0.035
WEIGHT_DEV_CAP = 1.2


def condition_points(horse: Horse, ratings) -> tuple[float, str]:
    """当日馬体重の評価。

    増減の絶対値だけを見るのは誤り。大事なのは「その馬が good な走りをしたときの
    体重に対して、今日はどうか」。
    たとえばグランヴィノスは 520kg で1着、524kg で2着、532kg で6着(1番人気)。
    9ヵ月ぶりで -12kg の 520kg は「減った」のではなく「best の体重に戻った」。
    """
    scored = [(r, p) for r, p in ratings if r.body_weight]
    if not scored or not horse.body_weight:
        return 0.0, ""

    # 「good な体重」は指数ではなく着差で測る。
    # 指数は時計が速ければ人気を裏切った凡走でも高く出るので、体調の指標には向かない。
    # (グランヴィノスは532kgの鳴尾記念が1番人気6着。指数は高いが体調面では
    #  評価できない走で、これを好走時体重に混ぜると読みが逆になる)
    runs = sorted(scored, key=lambda x: max(x[0].margin, 0.0))[:3]
    wts = [1.0 / (max(r.margin, 0.0) + 0.4) for r, _ in runs]
    best_w = sum(r.body_weight * w for (r, _), w in zip(runs, wts)) / sum(wts)

    dev = horse.body_weight - best_w
    scale = 1.0
    note_extra = ""
    if horse.age == 3 and dev > 0:
        # 3歳秋の増加は成長分が含まれるので割り引いて見る
        scale = 0.5
        note_extra = " ※3歳の増加は成長分として割引"

    pts = max(-WEIGHT_DEV_CAP, -abs(dev) * WEIGHT_DEV_POINTS * scale)
    note = (f"今回{horse.body_weight}kg / 好走時の体重 約{best_w:.0f}kg "
            f"({dev:+.0f}kg){note_extra}")
    return pts, note
