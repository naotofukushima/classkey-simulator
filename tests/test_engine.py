"""モデルの中核ロジックの回帰テスト。"""
from __future__ import annotations

from datetime import date

from keiba_ai.betting import exacta_pair_prob, trio_prob
from keiba_ai.engine import evaluate
from keiba_ai.loader import load_race
from keiba_ai.models import PastRun
from keiba_ai.pace import classify_pace
from keiba_ai.ratings import run_rating

RACE_JSON = "data/races/challenge_cup_2026.json"


def _run(**kw) -> PastRun:
    base = dict(
        date=date(2026, 5, 1), race="テストS", grade="G3", surface="芝", distance=2000,
        going="良", field_size=16, finish=1, margin=-0.2, carried=55.0,
    )
    base.update(kw)
    return PastRun(**base)


def test_定量G1の58kgは加点されない():
    """大阪杯の58kgは古馬牡馬の定量。ハンデ58kgと同一視してはいけない。"""
    teiryo = run_rating(_run(grade="G1", carried=58.0), sex="牡")
    handi = run_rating(_run(grade="G3", carried=58.0), sex="牡")
    g3_base = run_rating(_run(grade="G3", carried=55.0), sex="牡")
    assert handi > g3_base                      # ハンデ58kgは加点される
    assert abs(teiryo - (handi - (58 - 55) * 1.6 + 10.0)) < 1e-6   # G1(106)-G3(96)=10


def test_3歳限定戦の57kgも加点されない():
    a = run_rating(_run(grade="G1", carried=57.0, three_yo_only=True), sex="牡")
    b = run_rating(_run(grade="G1", carried=55.0, three_yo_only=True), sex="牡")
    assert abs((a - b) - (-(55 - 57) * 1.6)) < 1e-6   # 57が基準なので55は減点側


def test_着差が大きいほど指数は下がる():
    assert run_rating(_run(margin=0.0)) > run_rating(_run(margin=1.0))


def test_逃げ馬が3頭ならハイペース想定():
    _, horses = load_race(RACE_JSON)
    pace, _ = classify_pace(horses)
    assert sum(1 for h in horses if h.style == "逃げ") == 3
    assert pace == "H"


def test_勝率の合計は1():
    race, horses = load_race(RACE_JSON)
    assessments, _ = evaluate(horses, race)
    assert abs(sum(a.win_prob for a in assessments) - 1.0) < 1e-9


def test_複勝率は勝率以上かつ1以下():
    race, horses = load_race(RACE_JSON)
    assessments, _ = evaluate(horses, race)
    for a in assessments:
        assert a.win_prob <= a.place3_prob <= 1.0


def test_複勝率の合計はおよそ3():
    """3着以内は3頭ぶんなので、全馬の複勝率の和は3に近くなるはず。"""
    race, horses = load_race(RACE_JSON)
    assessments, _ = evaluate(horses, race)
    assert abs(sum(a.place3_prob for a in assessments) - 3.0) < 0.05


def test_確率分布が現実的なレンジに収まる():
    """16頭立て重賞で1番手が50%を超えるような分布は明らかに過信。"""
    race, horses = load_race(RACE_JSON)
    assessments, _ = evaluate(horses, race)
    assert 0.10 <= assessments[0].win_prob <= 0.40


def test_馬連確率は各馬の勝率以上():
    race, horses = load_race(RACE_JSON)
    a, _ = evaluate(horses, race)
    p = exacta_pair_prob(a[0], a[1])
    assert p > 0 and p < a[0].win_prob + a[1].win_prob
    assert trio_prob(a[0], a[1], a[2]) < p


def test_負担率は斤量を馬体重で割った値():
    _, horses = load_race(RACE_JSON)
    h = next(x for x in horses if x.name == "グランヴィノス")
    assert abs(h.burden_ratio - 56.0 / 532) < 1e-9
    # 532kgに56kgはメンバー最軽量の負担率
    assert h.burden_ratio == min(x.burden_ratio for x in horses)
