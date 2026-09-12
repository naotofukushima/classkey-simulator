"""総合スコアリングと確率・期待値の算出。"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .handicap import burden_points, layoff_points, weight_change_points
from .jockey import jockey_points, workout_points
from .models import Horse, RaceConditions
from .pace import classify_pace, draw_points, pace_points
from .pedigree import pedigree_score
from .ratings import ability, improving_form

# ---------------------------------------------------------------- 確率変換の根拠
# 同じ馬でも走るたびにパフォーマンスはブレる。実測ではそのブレ幅(標準偏差)は
# 2000m でおよそ 0.5〜0.6 秒 = 4.0〜4.8 点(SEC_TO_POINTS=8.0 換算)。
# 各馬の当日のパフォーマンスを Gumbel 分布(位置=スコア, 尺度=T)と見なすと
# 「最大値が誰になるか」は softmax(score / T) になり、
#   標準偏差 = T * pi / sqrt(6) = 1.283 * T
# したがって SD 4.5 点に対応する温度は T = 4.5 / 1.283 ≒ 3.5。
PERFORMANCE_SD = 4.5
TEMPERATURE = PERFORMANCE_SD / 1.283


@dataclass
class Assessment:
    horse: Horse
    base: float
    factors: dict[str, float] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)
    win_prob: float = 0.0
    place3_prob: float = 0.0

    @property
    def total(self) -> float:
        return self.base + sum(self.factors.values())

    @property
    def fair_odds(self) -> float:
        return 1 / self.win_prob if self.win_prob > 0 else float("inf")

    @property
    def ev(self) -> float | None:
        """単勝期待値。1.0 を超えれば理論上プラス。"""
        if self.horse.odds is None:
            return None
        return self.win_prob * self.horse.odds


def evaluate(horses: list[Horse], race: RaceConditions) -> tuple[list[Assessment], dict]:
    pace, pace_why = classify_pace(horses)
    n = len(horses)

    assessments: list[Assessment] = []
    for h in horses:
        base, breakdown = ability(h, race)
        a = Assessment(horse=h, base=base)

        a.factors["ハンデ(負担率)"], burden_detail = burden_points(h, horses)
        a.factors["斤量増減"], delta = weight_change_points(h)
        a.factors["展開/脚質"] = pace_points(h, pace, race)
        a.factors["枠順"] = draw_points(h, race, n)
        ped_pts, ped_detail = pedigree_score(h.sire, h.broodmare_sire, race.moisture)
        a.factors["血統適性"] = ped_pts
        a.factors["コース適性"] = h.course_bonus
        jk_pts, jk_detail = jockey_points(h.jockey, h.trainer)
        a.factors["騎手/厩舎"] = jk_pts
        a.factors["追い切り"] = workout_points(h.workout)
        a.factors["ローテ"], days = layoff_points(h, race.date)
        imp_pts, imp_why = improving_form(h)
        a.factors["上昇度"] = imp_pts
        # 3歳馬は古馬より秋にかけて伸びる余地がある
        a.factors["成長度"] = 1.0 if h.age == 3 else (-0.4 if h.age >= 7 else 0.0)

        a.notes["上昇度"] = imp_why
        a.notes["血統"] = ped_detail
        a.notes["騎手"] = jk_detail
        a.notes["負担率"] = (
            f"{burden_detail['burden_ratio_pct']:.2f}% "
            f"(出走馬平均 {burden_detail['field_avg_pct']:.2f}%)"
        )
        a.notes["斤量増減"] = "前走比 なし" if delta is None else f"前走比 {delta:+.1f}kg"
        a.notes["間隔"] = "不明" if days is None else f"中{days}日"
        a.notes["実績"] = " / ".join(
            f"{r.race}{r.finish}着[{p:.1f}]" for r, p in breakdown
        )
        assessments.append(a)

    _assign_probabilities(assessments)
    assessments.sort(key=lambda a: a.total, reverse=True)

    meta = {"pace": pace, "pace_reason": pace_why, "field_size": n}
    return assessments, meta


def _assign_probabilities(assessments: list[Assessment]) -> None:
    """スコアを softmax で勝率に、Harville 近似で複勝率に変換する。"""
    scores = [a.total for a in assessments]
    m = max(scores)
    exps = [math.exp((s - m) / TEMPERATURE) for s in scores]
    z = sum(exps)
    for a, e in zip(assessments, exps):
        a.win_prob = e / z

    # Harville モデル: 3着以内に入る確率 = P(1着) + P(2着) + P(3着)
    strengths = {id(a): a.win_prob for a in assessments}
    for a in assessments:
        pa = strengths[id(a)]
        p2 = 0.0   # a が2着
        p3 = 0.0   # a が3着
        for b in assessments:
            if b is a:
                continue
            pb = strengths[id(b)]
            rem_b = 1 - pb
            if rem_b <= 0:
                continue
            p2 += pb * (pa / rem_b)
            for c in assessments:
                if c is a or c is b:
                    continue
                pc = strengths[id(c)]
                rem_c = rem_b - pc
                if rem_c <= 0:
                    continue
                p3 += pb * (pc / rem_b) * (pa / rem_c)
        a.place3_prob = min(pa + p2 + p3, 1.0)
