"""コース固有のラップ特性と、当日のラップ予測、各馬のラップ適性。

「速い/遅い」よりも**どういうラップを刻むか**の方が着順を決めることが多い。
同じ1:58.0でも、前半58.4-後半59.6の消耗戦と、60.0-58.0の瞬発戦では
求められる資質が別物になる。

ここでは3つをやる。
  1. コースごとの par ラップ(その条件で標準的に刻まれる200mごとの区間)
  2. 出走馬の脚質構成と馬場から、当日のラップを予測する
  3. 各馬が「どういうラップで走ったとき走れたか」を測り、予測ラップと突き合わせる
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .models import Horse, PastRun, RaceConditions


@dataclass
class CourseProfile:
    """コースのラップ特性。par は良馬場・重賞水準の200mごとの区間タイム。"""

    par: list[float]
    description: str
    source: str = ""

    @property
    def par_time(self) -> float:
        return sum(self.par)


# 阪神芝2000m内回りの par は、2025年チャレンジカップ(同一コース・同一開催時期)の
# 実ラップ 12.6-11.5-11.9-11.3-11.1-11.8-12.1-12.1-11.7-11.9 (1:58.0) を
# 基準タイム117.8秒に合わせて微調整したもの。教科書値ではなく実測由来。
COURSE_PROFILES: dict[str, CourseProfile] = {
    "阪神芝2000": CourseProfile(
        par=[12.6, 11.5, 11.9, 11.3, 11.1, 11.8, 12.1, 12.0, 11.6, 11.9],
        description=(
            "正面スタンド前発走で1角まで約325mと短く、位置取り争いが早々に決着する。"
            "1-2角を回ると向正面が下りで、ここで一度ラップが11秒台前半まで速くなる。"
            "3角手前から4角にかけて12秒台に緩み(中弛み)、直線入口で再び11秒台に加速、"
            "最後は残り200mの急坂(高低差1.8m)でラップが落ちる。"
            "つまり『速い向正面 → 3-4角の緩み → 直線での二段加速 → 坂』という"
            "2段ギアチェンジ型。単発の瞬発力より、緩んだところで置かれずに"
            "4角までポジションを取れる機動力と、坂を踏ん張る持続力が要る。"
            "レースの上がり3Fは35秒台が標準で、極端な上がり勝負にはなりにくい。"
        ),
        source="2025年チャレンジカップ(阪神芝2000m 1:58.0)の実ラップ",
    ),
}


@dataclass
class LapPrediction:
    laps: list[float]
    reasons: list[str] = field(default_factory=list)
    course: CourseProfile | None = None

    @property
    def total(self) -> float:
        return sum(self.laps)

    @property
    def first1000(self) -> float:
        return sum(self.laps[:5])

    @property
    def last1000(self) -> float:
        return sum(self.laps[5:])

    @property
    def last3f(self) -> float:
        return sum(self.laps[-3:])

    @property
    def balance(self) -> float:
        """前後半1000mの差。正=前傾(消耗戦) / 負=後傾(瞬発戦)。"""
        return self.last1000 - self.first1000

    @property
    def shape(self) -> float:
        """均等ペースを基準にしたラップ形状。正=前傾。"""
        return self.last3f - self.total * 0.3

    @property
    def par_balance(self) -> float:
        """このコースで標準的な前後半差。阪神内2000は向正面が下りで元々前傾寄り。"""
        if self.course is None:
            return 0.0
        return sum(self.course.par[5:]) - sum(self.course.par[:5])

    @property
    def pace_label(self) -> str:
        """絶対値ではなくコースの par との差で判定する。

        阪神芝2000内回りは par の時点で前半1000mの方が1.0秒速い。
        絶対値で「前傾ならハイペース」と決めると、このコースは常にハイペースになる。
        """
        b = self.balance - self.par_balance
        if b >= 1.5:
            return "ハイペース(par比 大きく前傾)"
        if b >= 0.5:
            return "やや前傾"
        if b > -0.5:
            return "平均ペース"
        return "スローペース(後傾)"


# 逃げ・先行が増えたときにテンが速くなる度合いを分配する重み
FRONT_WEIGHTS = [0.6, 1.2, 1.0, 0.6, 0.2]
BACK_WEIGHTS = [0.0, 0.1, 0.4, 0.8, 1.3]


def predict_lap(race: RaceConditions, horses: list[Horse]) -> LapPrediction | None:
    """出走馬の脚質構成と馬場から、当日刻まれるラップを予測する。"""
    key = f"{race.course}{race.surface}{race.distance}"
    profile = COURSE_PROFILES.get(key)
    if profile is None:
        return None

    laps = list(profile.par)
    reasons = []

    # --- 1) ハナを主張する馬の数だけテンが速くなり、その反動で終いが掛かる ---
    n_front = sum(1 for h in horses if h.style == "逃げ")
    n_press = sum(1 for h in horses if h.style == "先行")
    extra = max(0, n_front - 1) * 0.10 + max(0, n_press - 2) * 0.04
    if extra > 0:
        for i, w in enumerate(FRONT_WEIGHTS):
            laps[i] -= extra * w
        for i, w in enumerate(BACK_WEIGHTS):
            laps[5 + i] += extra * w
        reasons.append(
            f"逃げ{n_front}頭・先行{n_press}頭。ハナ争いでテンが{extra * 3.6:.1f}秒ぶん"
            f"速くなり、その反動で終い3Fが掛かる"
        )
    elif n_front == 0:
        for i, w in enumerate(FRONT_WEIGHTS):
            laps[i] += 0.12 * w
        for i, w in enumerate(BACK_WEIGHTS):
            laps[5 + i] -= 0.12 * w
        reasons.append("逃げ宣言馬不在。前半が緩んで上がりの速い決着になりやすい")

    # --- 2) 馬場の速さ(開催週・クッション値) ---
    week_adj = 0.0
    if race.meeting_week is not None and race.meeting_week <= 3:
        week_adj = -0.03
        reasons.append(f"開催{race.meeting_week}週目の野芝で馬場の傷みが少なく、全体に時計が出る")
    if race.cushion is not None and race.cushion < 8.5:
        week_adj += 0.015
        reasons.append(f"クッション値{race.cushion}とやや軟らかく、時計の出方は抑えられる")
    laps = [x + week_adj for x in laps]

    # --- 3) 含水率が高いと終いの脚が上がる ---
    if race.moisture is not None and race.moisture >= 14.0:
        extra_wet = min((race.moisture - 14.0) / 4.0, 1.0) * 0.06
        laps[-1] += extra_wet
        laps[-2] += extra_wet * 0.8
        reasons.append(
            f"含水率{race.moisture}%と水分が残り、坂を含む終い2Fでさらに脚が上がる"
        )

    # --- 4) 頭数が多いほど隊列が長くなりテンは緩みにくい ---
    if len(horses) >= 15:
        laps[1] -= 0.05
        laps[2] -= 0.05
        laps[-1] += 0.05
        reasons.append(f"{len(horses)}頭立てで枠を取る争いが激しく、テンは緩まない")

    return LapPrediction(laps=[round(x, 2) for x in laps], reasons=reasons, course=profile)


# ---------------------------------------------------------------- 各馬の適性
def run_shape(run: PastRun) -> float | None:
    """その一戦を「前傾で走ったか後傾で走ったか」。正=前傾(上がりが掛かった)。

    均等ペースなら上がり3Fは 走破タイム×(600/距離) になるはず。
    それより上がりが掛かっていれば前半が速い消耗戦だったことになる。
    """
    if run.time is None or run.last3f is None or not run.is_turf:
        return None
    even3f = run.time * 600.0 / run.distance
    return run.last3f - even3f


def _standardize(xs: list[float]) -> list[float]:
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    sd = math.sqrt(var)
    if sd < 1e-9:
        return [0.0] * n
    return [(x - mean) / sd for x in xs]


LAP_FIT_CAP = 1.2
CORNER_CAP = 0.7
# 阪神芝2000内回りで4角を回るときの理想的な位置(頭数に対する比率)
IDEAL_CORNER4 = 0.42


def lap_aptitude(
    horse: Horse, prediction: LapPrediction, ratings: list[tuple[PastRun, float]]
) -> tuple[float, float, str]:
    """ラップ適性を(形状適性, 4角位置, 説明)で返す。

    形状適性は「その馬が相対的に前傾のレースで走れているか」を、自身の中で
    標準化した共分散で測る。脚質による定常的なズレは標準化で除かれるので、
    差し馬でも追込馬でも同じ尺度で比較できる。
    """
    pairs = [(run_shape(r), p) for r, p in ratings]
    pairs = [(s, p) for s, p in pairs if s is not None]

    shape_pts = 0.0
    note = ""
    if len(pairs) >= 3:
        zs = _standardize([s for s, _ in pairs])
        zp = _standardize([p for _, p in pairs])
        slope = sum(a * b for a, b in zip(zs, zp)) / len(zs)
        # サンプルが少ないので 0 方向に縮小する
        shrink = len(zs) / (len(zs) + 3.0)
        # 当日のラップが par からどれだけ前傾側に振れているか
        par_shape = (sum(prediction.course.par[-3:]) - prediction.course.par_time * 0.3)
        delta = (prediction.shape - par_shape) / 0.7
        shape_pts = max(-LAP_FIT_CAP, min(slope * shrink * delta * 1.3, LAP_FIT_CAP))
        tendency = "前傾の消耗戦向き" if slope > 0.1 else ("上がり勝負向き" if slope < -0.1 else "ラップ形状の好みは薄い")
        note = f"{tendency}(相関{slope:+.2f}, n={len(zs)})"

    # 4角での位置。内回り2000mは4角で後ろだと物理的に届かない。
    rates = [
        r.passes[-1] / r.field_size
        for r, _ in ratings
        if r.passes and len(r.passes) >= 3 and r.field_size > 0
    ]
    corner_pts = 0.0
    if rates:
        avg = sum(rates) / len(rates)
        corner_pts = max(-CORNER_CAP, min((IDEAL_CORNER4 - avg) * 1.6, CORNER_CAP))
        note += f" / 4角平均位置 上位{avg * 100:.0f}%"
    return shape_pts, corner_pts, note
