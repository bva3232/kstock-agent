#!/usr/bin/env python3
"""장 마감 후 실행 (15:40 권장) — 프로세스 1의 3~4단계 + 프로세스 2의 3단계.

    python run_evening.py

하는 일
  1. 어제 세운 오늘자 예측을 오늘 종가로 채점
  2. 누적 성적표 출력 (방향 적중률 / 단가 오차 / 구간 커버리지)
  3. [프로세스 1] 전종목 스크리닝 → 내일 상승 후보 TOP 5 예측 생성
  4. [프로세스 2] 핵심 종목(기본 000660, 005930) 예측 생성
  5. 모두 원장(.cache/predictions.db)에 기록
"""
from __future__ import annotations

import argparse
import io
import sys
from datetime import date, timedelta

import pandas as pd
from pykrx import stock

from kagent import disclosure, forecast, screener, signals, telegram
from kagent.config import Config
from kagent.dart import DartError, get_dart
from kagent.krx import KrxData, recent_trading_days
from kagent.ledger import Ledger
from kagent.watchlist import KOSPI_LARGE_CAP

CORE_DEFAULT = ["000660", "005930"]


def daily_ohlcv(ticker: str, days: int = 800) -> pd.DataFrame:
    end = date.today()
    start = end - timedelta(days=int(days * 1.5))
    df = stock.get_market_ohlcv_by_date(start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), ticker)
    return df[df["종가"] > 0] if df is not None and not df.empty else pd.DataFrame()


def _last_price(ticker: str, day: str) -> dict | None:
    """전종목 스냅샷(로그인 필요) 대신 종목 하나씩 조회해서 특정일 종가/시가를 구한다."""
    df = daily_ohlcv(ticker, days=10)
    target = pd.Timestamp(day)
    if df.empty or target not in df.index:
        return None
    row = df.loc[target]
    return {"close": float(row["종가"]), "open": float(row["시가"])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--core", nargs="*", default=None, help="프로세스2 핵심 종목코드")
    ap.add_argument("--no-dart", action="store_true")
    ap.add_argument("--skip-p1", action="store_true", help="스크리닝 건너뛰기 (빠른 실행)")
    ap.add_argument("--no-telegram", action="store_true", help="텔레그램 발송 건너뛰기")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    real_stdout = sys.stdout
    buf = io.StringIO()
    sys.stdout = telegram.TeeStdout(buf, real_stdout)
    try:
        rc = _run(args, cfg)
    finally:
        sys.stdout = real_stdout
    if not args.no_telegram:
        title = f"[마감 리포트] {date.today().strftime('%Y-%m-%d')}"
        telegram.send_message(cfg, buf.getvalue(), title=title)
    return rc


def _run(args, cfg) -> int:
    krx = KrxData(cfg.cache_dir)
    led = Ledger(cfg.cache_dir / "predictions.db")
    core = args.core if args.core is not None else cfg["account"].get("core_tickers", CORE_DEFAULT)

    days = recent_trading_days(cfg["screen"]["lookback_days"])
    today = days[-1]
    print(f"기준일 {today}\n")

    # ---------------------------------------------------------- 1) 채점
    print("=" * 74)
    print("1. 전일 예측 채점")
    print("=" * 74)
    pend = led.pending(today)
    if pend.empty:
        print("채점할 예측이 없습니다. (오늘이 첫 실행이면 정상입니다)\n")
    else:
        prices = {}
        for t in pend["ticker"].astype(str).str.zfill(6).unique():
            p = _last_price(t, today)
            if p:
                prices[t] = p
        res = led.settle(today, prices)
        if res.empty:
            print("가격 매칭 실패.\n")
        else:
            print(res.to_string(index=False), "\n")

    for proc, label in (("P1", "프로세스1 (단타 TOP5)"), ("P2", "프로세스2 (핵심종목)")):
        sc = led.scorecard(process=proc)
        if sc.get("n"):
            print(f"[{label}] {led.verdict(sc)}\n")

    next_day = "다음 영업일"
    all_fc: list[dict] = []
    panels = None

    # ------------------------------------------------- 2) 프로세스1 TOP5
    if not args.skip_p1:
        print("=" * 74)
        print(f"2. [프로세스 1] {next_day} 상승 후보 TOP {args.top} "
              f"(KOSPI 대형주 워치리스트 {len(KOSPI_LARGE_CAP)}종목 기준)")
        print("=" * 74)
        panels = krx.panel_from_tickers(list(KOSPI_LARGE_CAP), days)

        event_scores = None
        if not args.no_dart:
            try:
                dart = get_dart(cfg)
                disc = dart.recent_disclosures(days=2)
                if not disc.empty:
                    event_scores = disclosure.score_events(disc)
            except (DartError, Exception) as e:
                print(f"  [DART 건너뜀] {e}")

        # 전종목 스냅샷 API(시총/PBR/외국인 순매수)는 로그인이 있어야 응답하므로
        # 워치리스트 모드에서는 비워둔다 — 추세/모멘텀/거래대금/공시 이벤트만으로 순위를 매긴다.
        feat = screener.build_features(
            panels, pd.DataFrame(), pd.DataFrame(), event_scores, None)
        scored = screener.composite_score(screener.apply_universe_filter(feat, cfg["universe"]))

        for i, (tk, row) in enumerate(scored.head(args.top).iterrows(), 1):
            df = daily_ohlcv(tk)
            if len(df) < 120:
                continue
            fc = forecast.make(df, tk, row["name"])
            led.record(fc, "P1", "evening", _next_bd(today))
            all_fc.append(fc)
            print(f"\n[{i}] {forecast.format_one(fc)}")
            print(f"   스크리닝 근거: {'; '.join(screener.explain(row)[:3])}")
        print()

    # ------------------------------------------------ 3) 프로세스2 핵심종목
    print("=" * 74)
    print(f"3. [프로세스 2] 핵심 종목 {next_day} 예측")
    print("=" * 74)
    for tk in core:
        tk = str(tk).zfill(6)
        df = daily_ohlcv(tk)
        if df.empty:
            print(f"  {tk}: 데이터 없음")
            continue
        fc = forecast.make(df, tk, KrxData.name_of(tk))
        led.record(fc, "P2", "evening", _next_bd(today))
        all_fc.append(fc)
        print("\n" + forecast.format_one(fc))
    print()

    # ----------------------------------------------------------- 4) 시장
    idx = krx.index_history(days[0], today)
    reg = signals.market_regime(idx, panels["종가"] if panels else None)
    print("=" * 74)
    print(f"4. 시장 국면: {reg['state']} | KOSPI {reg['index_close']:,.2f} "
          f"(20일선 {reg['vs_ma20_pct']:+.1f}%, 20일 {reg['ret_20d_pct']:+.1f}%)")
    print("=" * 74)
    print("\n" + forecast.format_summary(all_fc))
    print("\n내일 07:30 에 run_premarket.py 를 실행해 변동사항을 반영하세요.")
    return 0


def _next_bd(d: str) -> str:
    dt = pd.Timestamp(d) + pd.offsets.BDay(1)
    return dt.strftime("%Y%m%d")


if __name__ == "__main__":
    sys.exit(main())
