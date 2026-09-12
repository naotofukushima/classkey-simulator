"""コマンドライン: レースJSONを読み、予想を出力する。

    python3 -m keiba_ai.cli data/races/challenge_cup_2026.json
"""
from __future__ import annotations

import argparse

from .betting import quinella_candidates, trio_candidates, value_win_bets
from .engine import evaluate
from .loader import load_race

MARKS = ["◎", "○", "▲", "△", "△", "☆", "×", "×"]


def main() -> None:
    ap = argparse.ArgumentParser(description="競馬予想AI")
    ap.add_argument("race_json")
    ap.add_argument("--detail", action="store_true", help="各馬の要素分解を表示")
    args = ap.parse_args()

    race, horses = load_race(args.race_json)
    assessments, meta = evaluate(horses, race)

    print("=" * 96)
    print(f"  {race.name}   {race.date}  {race.course}{race.surface}{race.distance}m({race.turn})"
          f"  {'ハンデ' if race.handicap else '定量'}  {race.going}/{race.weather}")
    print("=" * 96)
    print(f"馬場: クッション値{race.cushion} 含水率{race.moisture}% 開催{race.meeting_week}週目"
          f" / 内外バイアス{race.bias_inside:+.2f} 差しバイアス{race.bias_closer:+.2f}")
    print(f"想定ペース: {meta['pace']}  — {meta['pace_reason']}")
    print()

    if getattr(race, "odds_asof", ""):
        print(f"オッズ基準時刻: {race.odds_asof}")
        print()
    print(f"{'印':<3}{'馬番':>3} {'馬名':<11}{'スコア':>7}{'速度':>7}{'勝率':>7}{'複勝率':>8}"
          f"{'想定':>8}{'実オッズ':>9}{'期待値':>7}  {'脚質':<4}{'負担率':>7}")
    print("-" * 103)
    for i, a in enumerate(assessments):
        h = a.horse
        mark = MARKS[i] if i < len(MARKS) else "  "
        ev = a.ev
        ev_s = f"{ev:5.2f}" if ev is not None else "  -  "
        odds_s = f"{h.odds:7.1f}" if h.odds else "   -   "
        name = h.name + "　" * max(0, (11 - len(h.name)) // 2)
        sp_s = f"{a.best_speed:7.1f}" if a.best_speed is not None else "      -"
        print(f"{mark:<3}{h.num:>3} {name:<11}{a.total:7.2f}{sp_s}{a.win_prob*100:6.1f}%"
              f"{a.place3_prob*100:7.1f}%{a.fair_odds:8.1f}{odds_s}{ev_s:>7}  "
              f"{h.style:<4}{h.burden_ratio*100:6.2f}%")

    print()
    print("■ 期待値プラスの単勝 (AI勝率 × 実オッズ >= 1.15)")
    vb = value_win_bets(assessments)
    if vb:
        for a, ev in vb:
            print(f"   {a.horse.num:>2} {a.horse.name:<10} 単勝{a.horse.odds:>6.1f}倍 "
                  f"(AI想定{a.fair_odds:.1f}倍) → 期待値 {ev:.2f}")
    else:
        print("   なし")

    axis = assessments[0]
    print()
    print(f"■ 馬連 ({axis.horse.name} 軸) — 「必要配当」を上回るオッズなら買い")
    for a, b, prob, fair in quinella_candidates(assessments, axis, assessments[1:8]):
        print(f"   {a.horse.num:>2}-{b.horse.num:<2} {b.horse.name:<10} 的中率{prob*100:5.1f}%"
              f"  必要配当 {fair:6.1f}倍")

    print()
    print("■ 3連複 期待の組合せ (上位8頭から)")
    for a, b, c, prob, fair in trio_candidates(assessments, assessments[:8], top=10):
        nums = "-".join(str(x.horse.num) for x in sorted([a, b, c], key=lambda x: x.horse.num))
        print(f"   {nums:<10} 的中率{prob*100:5.2f}%  必要配当 {fair:7.1f}倍")

    if args.detail:
        print()
        print("=" * 96)
        print("  要素分解")
        print("=" * 96)
        for a in assessments:
            h = a.horse
            print(f"\n【{h.num}】{h.name} ({h.sex}{h.age}) 斤量{h.carried}kg {h.jockey}"
                  f"  — 総合 {a.total:.2f} / 勝率 {a.win_prob*100:.1f}%")
            print(f"   能力指数(直近上位3走の加重平均): {a.base:.2f}"
                  + (f" / 最高速度指数 {a.best_speed:.1f} — {a.best_speed_race}"
                     if a.best_speed is not None else ""))
            print(f"   採用走: {a.notes['実績']}")
            for k, v in a.factors.items():
                bar = "+" if v >= 0 else "-"
                print(f"     {k:<12} {v:+6.2f} {bar * min(int(abs(v) * 4), 24)}")
            print(f"   負担率 {a.notes['負担率']} / {a.notes['斤量増減']} / {a.notes['間隔']}")
            print(f"   血統 {a.notes['血統']}")
            if h.course_note:
                print(f"   コース {h.course_note}")
            if h.comment:
                print(f"   寸評 {h.comment}")


if __name__ == "__main__":
    main()
