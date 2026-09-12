"""騎手・厩舎評価。

重賞・関西圏での信頼度を 0-10 で格付け(2026年時点のリーディング水準を反映)。
"""
from __future__ import annotations

JOCKEY_RATING = {
    "川田将雅": 9.6, "C.ルメール": 9.5, "武豊": 8.6, "松山弘平": 8.5,
    "坂井瑠星": 8.5, "岩田望来": 8.0, "西村淳也": 7.9, "横山和生": 7.6,
    "M.デムーロ": 7.5, "佐々木大輔": 7.5, "横山典弘": 7.1, "松若風馬": 7.0,
    "古川吉洋": 6.5, "酒井学": 6.1, "松本大輝": 6.0, "西塚洸二": 5.6,
}

# 仕上げ・重賞での信頼度に差が出る厩舎
TRAINER_BONUS = {
    "友道康夫": 0.4, "池江泰寿": 0.3, "杉山晴紀": 0.4, "須貝尚介": 0.3,
    "斉藤崇史": 0.3, "奥村武": 0.2, "田中博康": 0.2, "辻野泰之": 0.1,
}

NEUTRAL = 7.4


def jockey_points(jockey: str, trainer: str) -> tuple[float, str]:
    r = JOCKEY_RATING.get(jockey, NEUTRAL)
    pts = (r - NEUTRAL) * 0.62 + TRAINER_BONUS.get(trainer, 0.0)
    return pts, f"{jockey}({r:.1f}) / {trainer}"


def workout_points(grade: int) -> float:
    """追い切り評価(1-5)。3を中立とする。"""
    return (grade - 3) * 0.75
