"""JSON のレースデータを読み込む。"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from .models import Horse, PastRun, RaceConditions


def _d(s: str) -> date:
    y, m, dd = (int(x) for x in s.split("-"))
    return date(y, m, dd)


def load_race(path: str | Path) -> tuple[RaceConditions, list[Horse]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    r = data["race"]
    race = RaceConditions(
        name=r["name"], date=_d(r["date"]), course=r["course"], surface=r["surface"],
        distance=r["distance"], turn=r["turn"], grade=r["grade"], handicap=r["handicap"],
        going=r["going"], weather=r["weather"], cushion=r.get("cushion"),
        moisture=r.get("moisture"), meeting_week=r.get("meeting_week"),
        bias_inside=r.get("bias_inside", 0.0), bias_closer=r.get("bias_closer", 0.0),
        note=r.get("note", ""),
    )

    horses = []
    for h in data["horses"]:
        runs = [
            PastRun(
                date=_d(x["date"]), race=x["race"], grade=x["grade"], surface=x["surface"],
                distance=x["distance"], going=x["going"], field_size=x["field_size"],
                finish=x["finish"], margin=x["margin"], carried=x["carried"],
                popularity=x.get("popularity"), last3f=x.get("last3f"),
                passes=x.get("passes", []), body_weight=x.get("body_weight"),
                three_yo_only=x.get("three_yo_only", False), course=x.get("course"),
                ref_weight=x.get("ref_weight"),
                note=x.get("note", ""),
            )
            for x in h["runs"]
        ]
        horses.append(Horse(
            num=h["num"], draw=h["draw"], name=h["name"], sex=h["sex"], age=h["age"],
            carried=h["carried"], jockey=h["jockey"], trainer=h["trainer"], stable=h["stable"],
            sire=h["sire"], broodmare_sire=h["broodmare_sire"], body_weight=h["body_weight"],
            style=h["style"], odds=h.get("odds"), popularity=h.get("popularity"),
            workout=h.get("workout", 3), course_bonus=h.get("course_bonus", 0.0),
            course_note=h.get("course_note", ""), freshness=h.get("freshness", 0.0),
            runs=runs, comment=h.get("comment", ""),
        ))
    return race, horses
