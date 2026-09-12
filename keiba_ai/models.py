"""ドメインモデル: レース条件・出走馬・過去走。

外部データ（netkeiba / 競馬ブック / JRA 等）を正規化して保持する。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional


# ---------------------------------------------------------------- クラス格付け
# 各競走クラスの基準指数。1着・同斤量・標準頭数での「勝ち馬の強さ」を表す。
# 実測では「3勝クラス→重賞」の壁は 2000m でおよそ 1.0〜1.3 秒。
# 1秒 = SEC_TO_POINTS(8.0)点 なので、クラス間は 8〜10点前後に収める。
CLASS_BASE = {
    "G1": 106.0,
    "G2": 100.5,
    "G3": 96.0,
    "L": 92.0,     # リステッド
    "OP": 90.0,    # オープン特別
    "3勝": 85.0,
    "2勝": 78.0,
    "1勝": 71.0,
    "未勝利": 62.0,
}

# 3歳限定戦は同クラスでも古馬混合より層が薄い分を割り引く
# (クラシックは層が厚いので割引は控えめ)
THREE_YO_ONLY_DISCOUNT = 3.5

# 芝のレースを評価する際、ダート実績はそのままでは使えない
SURFACE_MISMATCH_DISCOUNT = 7.0


@dataclass
class PastRun:
    """1走分の成績。"""

    date: date
    race: str
    grade: str                 # CLASS_BASE のキー
    surface: str               # "芝" / "ダ"
    distance: int
    going: str                 # 良 / 稍 / 重 / 不
    field_size: int
    finish: int
    margin: float              # 勝ち馬との着差(秒)。勝った場合は負値(=2着につけた差)
    carried: float             # 斤量(kg)
    popularity: Optional[int] = None
    last3f: Optional[float] = None
    passes: list[int] = field(default_factory=list)
    body_weight: Optional[int] = None
    three_yo_only: bool = False
    course: Optional[str] = None       # 例 "阪神芝2000内"
    ref_weight: Optional[float] = None  # そのレースの標準斤量(定量/別定/ハンデ)
    note: str = ""

    @property
    def is_turf(self) -> bool:
        return self.surface == "芝"

    def reference_weight(self, sex: str) -> float:
        """その一戦の「標準斤量」。これより重ければ加点、軽ければ減点する。

        ここを間違えると評価が壊れる。たとえば大阪杯・宝塚記念の58kgは
        古馬牡馬の"定量"であって背負わされたわけではないので加点してはいけない。
        逆にハンデ戦の58kgはハンデキャッパーが課した重量なので加点対象になる。
        """
        if self.ref_weight is not None:
            return self.ref_weight
        is_filly = sex == "牝"
        if self.three_yo_only:                       # クラシック等の3歳定量
            return 55.0 if is_filly else 57.0
        if self.grade in ("G1", "G2"):               # 古馬の定量・別定重賞
            return 56.0 if is_filly else 58.0
        if self.grade in ("3勝", "2勝", "1勝", "未勝利"):  # 条件戦の定量
            return 55.0 if is_filly else 57.0
        return 55.0                                  # ハンデ重賞・OP特別の基準


@dataclass
class Horse:
    """出走馬1頭。"""

    num: int                   # 馬番
    draw: int                  # 枠番
    name: str
    sex: str                   # 牡 / 牝 / セ
    age: int
    carried: float             # 今回の斤量(ハンデ)
    jockey: str
    trainer: str
    stable: str                # 栗東 / 美浦
    sire: str
    broodmare_sire: str
    body_weight: int           # 想定馬体重(前走実測)
    style: str                 # 逃げ / 先行 / 好位 / 差し / 追込
    odds: Optional[float] = None
    popularity: Optional[int] = None
    workout: int = 3           # 追い切り評価 1-5
    course_bonus: float = 0.0  # 当該コース適性(実績ベース、手動入力)
    course_note: str = ""
    freshness: float = 0.0     # 休み明け実績による補正(鉄砲成績・厩舎力)
    runs: list[PastRun] = field(default_factory=list)
    comment: str = ""

    @property
    def burden_ratio(self) -> float:
        """斤量 / 馬体重。小さいほど有利。"""
        return self.carried / self.body_weight


@dataclass
class RaceConditions:
    """レース当日の条件・馬場・バイアス。"""

    name: str
    date: date
    course: str                # 例 "阪神"
    surface: str
    distance: int
    turn: str                  # "内回り" / "外回り"
    grade: str
    handicap: bool
    going: str                 # 発表馬場
    weather: str
    cushion: Optional[float] = None      # クッション値
    moisture: Optional[float] = None     # 含水率(ゴール前)
    meeting_week: Optional[int] = None   # 開催何週目か
    bias_inside: float = 0.0   # 正=内有利 / 負=外有利
    bias_closer: float = 0.0   # 正=差し有利 / 負=先行有利
    note: str = ""
