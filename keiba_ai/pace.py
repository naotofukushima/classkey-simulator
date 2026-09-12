"""展開(ペース)シミュレーションと脚質補正。

逃げたい馬の数から想定ペースを決め、当日のトラックバイアスと合成して
脚質ごとの有利不利をポイント化する。
"""
from __future__ import annotations

from .models import Horse, RaceConditions

STYLES = ["逃げ", "先行", "好位", "差し", "追込"]

# ペース区分ごとの、脚質別の基本補正
# S=スロー / M=平均 / H=ハイ
PACE_TABLE = {
    "S": {"逃げ": 1.8, "先行": 1.0, "好位": 0.2, "差し": -0.8, "追込": -1.6},
    "M": {"逃げ": 0.4, "先行": 0.4, "好位": 0.3, "差し": -0.1, "追込": -0.8},
    "H": {"逃げ": -1.8, "先行": -0.6, "好位": 0.8, "差し": 1.3, "追込": 0.6},
}


def classify_pace(horses: list[Horse]) -> tuple[str, str]:
    """逃げ・先行馬の頭数から想定ペースを判定する。"""
    front = sum(1 for h in horses if h.style == "逃げ")
    pressers = sum(1 for h in horses if h.style == "先行")

    if front >= 3:
        pace = "H"
        why = f"ハナを主張したい馬が{front}頭(先行型も{pressers}頭)。折り合いを欠く隊列になりやすく、前は総崩れのリスク"
    elif front == 2:
        pace = "M" if pressers <= 2 else "H"
        why = f"逃げ候補{front}頭・先行型{pressers}頭。少なくとも楽な単騎逃げにはならない"
    elif front == 1:
        pace = "S" if pressers <= 2 else "M"
        why = f"明確な逃げ馬は{front}頭。先行型{pressers}頭がどこまで絡むか"
    else:
        pace = "S"
        why = "逃げ宣言馬不在。前半は緩みやすい"
    return pace, why


def pace_points(horse: Horse, pace: str, race: RaceConditions) -> float:
    """脚質 × 想定ペース × トラックバイアスの合成。"""
    pts = PACE_TABLE[pace].get(horse.style, 0.0)

    # 当日の差し/先行バイアス(正=差し有利)
    closer_axis = {"逃げ": -1.0, "先行": -0.5, "好位": 0.2, "差し": 0.8, "追込": 0.7}
    pts += race.bias_closer * closer_axis.get(horse.style, 0.0)

    # 内回り2000mは追込が物理的に届きにくい(直線356m)
    if race.turn == "内回り" and horse.style == "追込":
        pts -= 0.5
    return pts


def draw_points(horse: Horse, race: RaceConditions, field_size: int) -> float:
    """枠順補正。阪神芝2000内回りはスタート後すぐ1角で、外枠は距離ロスが出る。"""
    pos = (horse.num - 0.5) / field_size          # 0(最内)〜1(大外)
    pts = (0.5 - pos) * 1.2 * (1.0 + race.bias_inside)

    # 前に行きたい馬にとって内枠の価値はさらに高い
    if horse.style in ("逃げ", "先行") and pos <= 0.4:
        pts += 0.3
    # 差し・追込が最内で包まれるリスク
    if horse.style in ("差し", "追込") and pos <= 0.12:
        pts -= 0.3
    return pts
