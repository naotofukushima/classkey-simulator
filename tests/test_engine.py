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


def test_負担率は当日馬体重で計算する():
    """前走の馬体重ではなく当日発表の馬体重を使う。
    グランヴィノスは532kg想定だったが当日は520kg(-12)で、
    これだけで負担率は10.53%から10.77%に変わり最軽量ではなくなる。"""
    _, horses = load_race(RACE_JSON)
    h = next(x for x in horses if x.name == "グランヴィノス")
    assert h.body_weight == 520 and h.body_weight_change == -12
    assert abs(h.burden_ratio - 56.0 / 520) < 1e-9
    lightest = min(horses, key=lambda x: x.burden_ratio)
    assert lightest.name == "マテンロウゲイル"      # 508kg(+12)に54kg


def test_好走時の体重は指数ではなく着差で測る():
    """グランヴィノスの最高指数は532kgの鳴尾記念だが1番人気6着。
    指数で重み付けすると好走時体重が529kgになり、520kgが「9kg減」と
    誤読される。着差で測れば約523kgで、今回はほぼ適正。"""
    from keiba_ai.handicap import condition_points
    from keiba_ai.ratings import all_run_ratings

    race, horses = load_race(RACE_JSON)
    h = next(x for x in horses if x.name == "グランヴィノス")
    pts, note = condition_points(h, all_run_ratings(h, race))
    assert "約523kg" in note
    assert pts > -0.2                      # ほぼ適正体重なので減点は小さい

    # 532kgで凡走した鳴尾記念より、520kgで勝った関ケ原Sの方が重く扱われる
    best = min(h.runs, key=lambda r: max(r.margin, 0.0))
    assert best.race == "関ケ原S" and best.body_weight == 520


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


# ---------------------------------------------------------------- 買い目の組み立て
from keiba_ai.portfolio import (  # noqa: E402
    AI_WEIGHT, PAYOUT_HAIRCUT, allocate, blended_views, build_tickets, market_probs, simulate,
)


def _assess():
    race, horses = load_race(RACE_JSON)
    return evaluate(horses, race)[0]


def test_市場勝率は正規化されている():
    a = _assess()
    mp = market_probs(a)
    assert abs(sum(mp.values()) - 1.0) < 1e-9
    # 単勝オッズが低いほど市場勝率は高い
    fav = min(a, key=lambda x: x.horse.odds).horse.num
    assert mp[fav] == max(mp.values())


def test_採用勝率はAIと市場の間に入る():
    """AIをそのまま使うと乖離が複数脚に掛かって非現実的な期待値になるため寄せる。"""
    a = _assess()
    mp = market_probs(a)
    bl = {v.horse.num: v.win_prob for v in blended_views(a)}
    assert abs(sum(bl.values()) - 1.0) < 1e-9
    grand = next(x for x in a if x.horse.name == "グランヴィノス")
    n = grand.horse.num
    assert mp[n] < bl[n] < grand.win_prob          # 市場 < 採用 < AI
    assert AI_WEIGHT < 0.5                          # 未検証モデルなので市場を主にする


def test_配当推定には保守的な割引がかかる():
    a = _assess()
    tickets = build_tickets(a, min_ev=0.0, min_prob=0.0)
    wide = next(t for t in tickets if t.kind == "ワイド")
    mkt_implied = (1 - 0.235) / 1.0                 # 割引がなければ (1-控除率)/的中率
    assert PAYOUT_HAIRCUT["ワイド"] < 1.0
    assert PAYOUT_HAIRCUT["単勝"] == 1.0            # 単勝は実オッズなので割引不要
    tan = next(t for t in tickets if t.kind == "単勝")
    assert tan.payout == next(x for x in a if x.horse.num == tan.legs[0]).horse.odds
    assert wide.payout > 0 and mkt_implied > 0


def test_配分は予算と単位を守る():
    a = _assess()
    picks = allocate(build_tickets(a), budget=1500, unit=100, max_lines=8)
    assert sum(t.stake for t in picks) == 1500
    assert all(t.stake % 100 == 0 and t.stake > 0 for t in picks)
    assert len(picks) <= 8


def test_集中度の上限で1頭への依存が下がる():
    a = _assess()
    free = allocate(build_tickets(a), 1500, 100, 8)
    capped = allocate(build_tickets(a), 1500, 100, 8, max_horse_share=0.65)

    def share(picks, num):
        tot = sum(t.stake for t in picks)
        return sum(t.stake for t in picks if num in t.legs) / tot

    grand = next(x for x in a if x.horse.name == "グランヴィノス").horse.num
    assert share(capped, grand) <= share(free, grand)
    assert sum(t.stake for t in capped) == 1500


def test_モンテカルロの的中率は単券の的中率と整合する():
    """単勝1点だけ買えば、的中率は採用勝率に一致するはず。"""
    a = _assess()
    tickets = build_tickets(a, min_ev=0.0, min_prob=0.0)
    tan = next(t for t in tickets if t.kind == "単勝" and t.legs[0] == 13)
    tan.stake = 1500
    sim = simulate([tan], a, n=20000)
    assert abs(sim["hit_rate"] - tan.ai_prob) < 0.02
    assert abs(sim["roi"] - tan.ev) < 0.25


def test_回収率が非現実的な水準にならない():
    """AIをそのまま信じると回収率690%などという数字が出る。寄せた後は常識的な範囲に。"""
    a = _assess()
    picks = allocate(build_tickets(a), 1500, 100, 8, max_horse_share=0.65)
    sim = simulate(picks, a, n=20000)
    assert 0.5 < sim["roi"] < 3.0
    assert 0.0 < sim["hit_rate"] < 1.0


# ---------------------------------------------------------------- フォーメーション
from keiba_ai.portfolio import expand_formation, parse_spec, tickets_from_spec  # noqa: E402


def test_フォーメーションが組合せに展開される():
    field = list(range(1, 17))
    # 3連複 14 - 10,15 - 全 : 14と(10か15)を含む3頭の組合せ
    got = parse_spec("3連複:14-10,15-全", field)
    legs = {t[1] for t in got}
    assert all(k == "3連複" for k, _ in got)
    assert all(14 in l and (10 in l or 15 in l) and len(l) == 3 for l in legs)
    # 10-14-15 は2通りの経路で作れるが1点に畳まれる
    assert (10, 14, 15) in legs
    assert len(legs) == 27          # 14通り + 14通り - 重複1


def test_順不同の券は重複を畳む():
    field = [1, 2, 3, 4]
    got = expand_formation([[1, 2], [1, 2], [3]], "3連複", field)
    assert got == [(1, 2, 3)]


def test_列数が合わなければ弾く():
    import pytest
    with pytest.raises(ValueError):
        parse_spec("3連複:13-15", list(range(1, 17)))


def test_単点指定とフォーメーションが混在できる():
    field = list(range(1, 17))
    got = parse_spec("単勝:13;3連複:13-4-全", field)
    assert ("単勝", (13,)) in got
    assert sum(1 for k, _ in got if k == "3連複") == 14     # 残り14頭


def test_カンマ区切りの複数券指定は従来どおり動く():
    """列内の区切りにも "," を使うので、券の区切りと取り違えないこと。"""
    field = list(range(1, 17))
    got = parse_spec("単勝:13,馬連:9-13", field)
    assert got == [("単勝", (13,)), ("馬連", (9, 13))]


def test_軸の選択で期待値が大きく変わる():
    """3連複フォーメーションの1列目は必ず3着以内に入ることを要求する。
    複勝圏8%のミッキーゴールドを軸に据えると全点が期待値1未満になる。"""
    a = _assess()
    yours = tickets_from_spec("3連複:14-10,15-全", a)
    swapped = tickets_from_spec("3連複:13-10,15-全", a)

    def roi(ts):
        return sum(t.ai_prob * t.payout for t in ts) / len(ts)

    assert len(yours) == len(swapped) == 27
    assert all(t.ev < 1.0 for t in yours)          # 27点すべてマイナス
    assert roi(swapped) > roi(yours) * 2.0         # 軸を替えるだけで2倍以上


def test_実オッズを指定すると推定配当より優先される():
    """複勝系の配当は推定値でしかないので、本物のオッズが分かるなら必ずそちらを使う。"""
    a = _assess()
    est = tickets_from_spec("3連複:9-13-15", a)[0]
    real = tickets_from_spec("3連複:9-13-15@42.9", a)[0]
    assert real.payout == 42.9
    assert real.legs == est.legs and real.ai_prob == est.ai_prob   # 的中率は変わらない
    assert est.payout != 42.9                                      # 推定はズレていた


def test_実オッズ指定はフォーメーションや複数券と併用できる():
    a = _assess()
    ts = tickets_from_spec("単勝:13@20.5;馬連:9-13@33.3;3連複:13-4-全", a)
    tan = next(t for t in ts if t.kind == "単勝")
    uma = next(t for t in ts if t.kind == "馬連")
    assert tan.payout == 20.5 and uma.payout == 33.3
    assert sum(1 for t in ts if t.kind == "3連複") == 14    # 全=残り14頭に展開
