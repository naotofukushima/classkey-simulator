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


# ---------------------------------------------------------------- 速度指数・ラップ
from keiba_ai.speed import LAP_CAP, SPEED_CAP, lap_adjustment, speed_rating  # noqa: E402


def _timed(**kw) -> PastRun:
    base = dict(track="阪神芝2000", time=118.0, distance=2000, grade="G3", going="良")
    base.update(kw)
    return _run(**base)


def test_タイムが無ければ速度指数は出ない():
    assert speed_rating(_run())[0] is None                     # time なし
    assert speed_rating(_timed(track="未知の競馬場"))[0] is None  # 基準タイム未登録


def test_速いタイムほど速度指数は高い():
    fast, _ = speed_rating(_timed(time=117.0))
    slow, _ = speed_rating(_timed(time=119.0))
    assert fast > slow
    # 阪神芝2000の基準117.8秒どおりに走れば中心値96.0
    even, _ = speed_rating(_timed(time=117.8))
    assert abs(even - 96.0) < 1e-6


def test_時計のかかる馬場は割り引かれない():
    """重馬場で同じ時計なら、良馬場より高く評価されるべき。"""
    good, _ = speed_rating(_timed(going="良"))
    heavy, _ = speed_rating(_timed(going="重"))
    assert heavy > good


def test_下のクラスで同じ時計を出す方が価値が高い():
    g1, _ = speed_rating(_timed(grade="G1"))
    c3, _ = speed_rating(_timed(grade="3勝"))
    assert c3 > g1


def test_速度指数は飽和して発散しない():
    """コースレコード級でも上限付近に収まり、順序は保たれる。"""
    a, _ = speed_rating(_timed(time=112.0))
    b, _ = speed_rating(_timed(time=110.0))
    assert a < b < 96.0 + SPEED_CAP


def test_前めの位置で速い上がりを使う方が高く評価される():
    front = lap_adjustment(_run(last3f=34.0, passes=[2, 2, 2, 2], field_size=16))
    back = lap_adjustment(_run(last3f=34.0, passes=[14, 14, 13, 12], field_size=16))
    assert front > back > 0
    assert abs(front) <= LAP_CAP


def test_遅い上がりはマイナス補正():
    assert lap_adjustment(_run(last3f=37.0, passes=[3, 3, 3, 3], field_size=16)) < 0


def test_レコード勝ちが能力指数に反映される():
    """マリアイリダータの前走は福島芝2000のコースレコード1:56.7。
    着差だけの評価では3勝クラスの1勝に過ぎず、速度軸で初めて価値が出る。"""
    from keiba_ai.ratings import best_speed

    _, horses = load_race(RACE_JSON)
    maria = next(h for h in horses if h.name == "マリアイリダータ")
    sp, run = best_speed(maria)
    assert run.race == "バーデンバーデンC" and run.time == 116.7
    assert sp > 100.0                       # G3水準(96)を大きく上回る
    assert sp == max(s for s in (best_speed(h)[0] for h in horses) if s)  # 全馬中最速


# ---------------------------------------------------------------- ラップ特性
from keiba_ai.lapprofile import COURSE_PROFILES, predict_lap, run_shape  # noqa: E402


def test_parラップは基準タイムと整合する():
    """阪神芝2000のparは2025年チャレンジC実ラップ由来。合計は基準タイム117.8秒。"""
    from keiba_ai.speed import STANDARD_TIMES

    prof = COURSE_PROFILES["阪神芝2000"]
    assert len(prof.par) == 10
    assert abs(prof.par_time - STANDARD_TIMES["阪神芝2000"]) < 0.05


def test_逃げ馬が多いほど前半が速く終いが掛かる():
    race, horses = load_race(RACE_JSON)
    many = predict_lap(race, horses)

    # 逃げ2頭を控えさせる
    import copy
    few = copy.deepcopy(horses)
    for h in few:
        if h.num in (5, 12):
            h.style = "先行"
    single = predict_lap(race, few)

    assert many.first1000 < single.first1000     # テンが速い
    assert many.last3f > single.last3f           # 終いが掛かる
    assert many.balance > single.balance         # より前傾


def test_ペース判定はコースのparとの差で行う():
    """阪神内2000はparの時点で前傾。絶対値判定だと常にハイペースになってしまう。"""
    race, horses = load_race(RACE_JSON)
    lap = predict_lap(race, horses)
    assert lap.par_balance > 0                   # parからして前傾寄り
    assert lap.balance > lap.par_balance         # 今回はさらに前傾
    assert "前傾" in lap.pace_label


def test_ラップ形状は前傾で正になる():
    """上がりが均等ペースより掛かった=前半が速かった、と読む。"""
    fast_finish = _timed(time=118.0, last3f=34.0)     # 均等なら35.4
    slow_finish = _timed(time=118.0, last3f=36.5)
    assert run_shape(fast_finish) < 0 < run_shape(slow_finish)


def test_1人気は上がり勝負向きと判定される():
    """マテンロウゲイルの最高パフォーマンス(ダービー5着)は最も後傾のレース。
    前傾の消耗戦になる今回は、その適性がマイナスに働く。"""
    race, horses = load_race(RACE_JSON)
    a_list, _ = evaluate(horses, race)
    gale = next(a for a in a_list if a.horse.name == "マテンロウゲイル")
    grand = next(a for a in a_list if a.horse.name == "グランヴィノス")
    assert gale.factors["ラップ適性"] < 0
    assert grand.factors["ラップ適性"] > 0


def test_展開が振れても上位の序列は大きく崩れない():
    """ハナ争いの出方は当日次第なので、結論が展開に過敏だと使えない。"""
    import copy

    race, horses = load_race(RACE_JSON)
    base, _ = evaluate(horses, race)
    base_top = {a.horse.num: a.win_prob for a in base[:5]}

    slow = copy.deepcopy(horses)
    for h in slow:
        if h.num in (5, 12):
            h.style = "先行"
    alt, _ = evaluate(slow, race)
    alt_p = {a.horse.num: a.win_prob for a in alt}
    for num, p in base_top.items():
        assert abs(alt_p[num] - p) < 0.05        # 勝率の振れは5ポイント未満
