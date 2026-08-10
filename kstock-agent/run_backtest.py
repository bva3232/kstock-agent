#!/usr/bin/env python3
"""신호 검증. 실매매 전에 반드시 한 번 돌려보세요.

    python run_backtest.py --days 400 --horizon 5
    python run_backtest.py --days 400 --horizon 20   # 보유기간을 늘리면 비용 부담이 줄어듭니다

결과에서 t 통계량이 2 미만이면 그 신호는 쓰지 않는 게 맞습니다.
"""
from __future__ import annotations

import argparse
import sys

from kagent import backtest
from kagent.config import Config
from kagent.krx import KrxData, recent_trading_days


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--days", type=int, default=400, help="사용할 과거 영업일 수")
    ap.add_argument("--horizon", type=int, default=5, help="보유기간(영업일)")
    ap.add_argument("--step", type=int, default=5, help="리밸런싱 간격(영업일)")
    ap.add_argument("--buckets", type=int, default=5)
    ap.add_argument("--lag", type=int, default=1,
                    help="신호일 다음날 진입(기본 1). 0 으로 두면 성과가 부풀려집니다")
    ap.add_argument("--null-check", type=int, default=0, metavar="N",
                    help="같은 모양의 무작위 데이터 N회로 노이즈 하한선 측정 (권장 20)")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    krx = KrxData(cfg.cache_dir)

    print(f"· {args.days}영업일 시세 수집 중 (최초 실행은 오래 걸립니다)...")
    days = recent_trading_days(args.days)
    panels = krx.panels(days, market="ALL")
    print(f"· {panels['종가'].shape[0]}일 x {panels['종가'].shape[1]}종목")

    run_kw = dict(
        horizon=args.horizon,
        step=args.step,
        n_buckets=args.buckets,
        cost_pct=cfg["account"]["roundtrip_cost_pct"],
        lag=args.lag,
    )
    res = backtest.run(panels, **run_kw)

    print("\n" + "=" * 62)
    print(f"검증 결과 — 보유기간 {res['horizon_days']}영업일, 구간 {res['periods']}회")
    print("=" * 62)
    print(f"  IC 평균            {res['ic_mean']:+.4f}  (표준편차 {res['ic_std']:.4f})")
    print(f"  IC t 통계량        {res['ic_t_stat']:+.2f}")
    print(f"  IC 양수 비율       {res['ic_hit_rate']:.1f}%")
    print(f"  상위분위 평균수익  {res['top_bucket_mean_pct']:+.2f}%")
    print(f"  비용 차감 후       {res['top_bucket_after_cost_pct']:+.2f}%")
    print(f"  상위분위 승률      {res['top_bucket_win_rate_pct']:.1f}%")
    print(f"  상위-하위 스프레드 {res['long_short_spread_pct']:+.2f}%p")
    print("\n분위별 (0=최하위):")
    print(res["by_bucket"].round(2).to_string())
    if args.null_check:
        print(f"\n· 무작위 데이터 {args.null_check}회로 노이즈 하한선 측정 중...")
        nc = backtest.null_check(panels, n_sims=args.null_check, **run_kw)
        print("\n" + "=" * 62)
        print("귀무분포 점검 — 신호가 없어야 정상인 데이터에서 나온 IC")
        print("=" * 62)
        print(f"  무작위 IC 평균     {nc['null_ic_mean']:+.4f}")
        print(f"  무작위 IC 표준편차 {nc['null_ic_std']:.4f}")
        print(f"  무작위 IC 95분위   {nc['null_ic_p95']:+.4f}")
        print(f"  실제 IC            {nc['real_ic']:+.4f}")
        print(f"  p-value            {nc['p_value']:.3f}")
        if nc["p_value"] > 0.10:
            print("\n  → 실제 IC가 무작위 수준과 구별되지 않습니다. 이 신호는 쓰지 마세요.")
        else:
            print("\n  → 무작위 수준보다는 높습니다. 다만 이것만으로 수익을 보장하지는 않습니다.")

    print("\n" + "-" * 62)
    print(backtest.verdict(res))
    print("-" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
