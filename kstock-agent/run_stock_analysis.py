#!/usr/bin/env python3
"""개별 종목 차트 패턴 분석 + 시나리오.

    python run_stock_analysis.py --ticker 000660
    python run_stock_analysis.py --ticker 000660 --years 6 --event "美 마이크론 실적(현지 8/7)"

출력
  1) 차트 구조 (사실만)
  2) 셋업별 조건부 과거 통계 — '이 패턴 뒤에 실제로 뭐가 일어났는가'
  3) 다음날 변동폭 실측 분포와 분기 시나리오
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

import pandas as pd
from pykrx import stock

from kagent import patterns, scenario
from kagent.config import Config
from kagent.krx import KrxData


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True, help="6자리 종목코드 (예: 000660)")
    ap.add_argument("--years", type=float, default=6.0, help="분석 기간(년)")
    ap.add_argument("--event", action="append", default=[],
                    help="차트에 없는 예정 이벤트. 여러 번 지정 가능")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    tk = args.ticker.zfill(6)
    name = KrxData.name_of(tk)

    end = date.today()
    start = end - timedelta(days=int(365 * args.years))
    print(f"· {name}({tk}) {start}~{end} 일봉 수집 중...")
    df = stock.get_market_ohlcv_by_date(start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), tk)
    if df is None or df.empty:
        print("데이터를 가져오지 못했습니다.")
        return 1
    df = df[df["종가"] > 0]
    print(f"· {len(df)}거래일 확보\n")

    print("=" * 78)
    print(f"{name} ({tk}) — 차트 구조 및 시나리오")
    print("=" * 78)
    sc = scenario.build(df, known_events=args.event)
    print(scenario.format_report(sc))

    print("\n" + "=" * 78)
    print("셋업별 조건부 과거 통계 (● = 현재 해당)")
    print("=" * 78)
    rep = patterns.report(df, horizons=(1, 5))
    print(rep.to_string(index=False))

    print("\n" + "=" * 78)
    print("요약")
    print("=" * 78)
    print(patterns.summarize(df))

    print("\n※ '유의(주의)' 표시는 과거 구간에서 t>2 였다는 뜻일 뿐입니다. "
          "12개 셋업을 한꺼번에 검정하면 그 중 하나쯤은 우연히 t>2 가 나옵니다(다중검정 문제).\n"
          "   실제로 쓰려면 기간을 나눠 재현되는지 먼저 확인하세요.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
