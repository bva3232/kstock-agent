#!/usr/bin/env python3
"""개장 전 실행 (07:30) — 프로세스 1의 2단계, 프로세스 2의 4단계.

기본은 **국내 데이터만** 사용합니다. (미국장 연동은 --use-us 로 별도 활성화)

    python run_premarket.py
    python run_premarket.py --gap 000660=-4.2 --gap 005930=-2.9   # 갭 직접 입력
    python run_premarket.py --use-us                              # 미국장 연동 버전

하는 일
  1. 어제 저녁에 세운 오늘자 예측을 불러온다
  2. 밤사이 새로 뜬 DART 공시 확인 (예측 종목 관련 + 시장 전체 이상징후)
  3. 갭이 있으면(수동 입력 또는 NXT 프리마켓) 반영해 예측 재산출
  4. 저녁 예측 대비 무엇이 얼마나 바뀌었는지 출력
  5. 한 줄 요약으로 마무리

08:00~08:50 NXT 프리마켓 호가를 보고 --gap 으로 넣으면 정확도가 올라갑니다.
07:30 실행이면 갭 없이 공시 점검 위주로 돌아갑니다.
"""
from __future__ import annotations

import argparse
import io
import sys
from datetime import date, timedelta

import pandas as pd
from pykrx import stock

from kagent import disclosure, forecast, telegram
from kagent.config import Config
from kagent.dart import DartError, get_dart
from kagent.krx import KrxData, recent_trading_days
from kagent.ledger import Ledger


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--gap", action="append", default=[],
                    help="티커=갭퍼센트 형식. 예: --gap 000660=-4.2")
    ap.add_argument("--no-dart", action="store_true")
    ap.add_argument("--use-us", action="store_true",
                    help="간밤 미국장(SOX/MU/NVDA) 연동. 기본은 국내 데이터만 사용")
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
        title = f"[개장전 리포트] {date.today().strftime('%Y-%m-%d')}"
        telegram.send_message(cfg, buf.getvalue(), title=title)
    return rc


def _run(args, cfg) -> int:
    led = Ledger(cfg.cache_dir / "predictions.db")
    krx = KrxData(cfg.cache_dir)

    manual_gap = {}
    for g in args.gap:
        try:
            t, v = g.split("=")
            manual_gap[t.zfill(6)] = float(v)
        except ValueError:
            print(f"  --gap 형식 오류 무시: {g}")

    target = date.today().strftime("%Y%m%d")
    pend = led.pending(target)
    if pend.empty:
        print(f"{target} 대상 예측이 없습니다. 전날 저녁 run_evening.py 를 먼저 실행하세요.")
        return 1
    tickers = sorted(pend["ticker"].astype(str).str.zfill(6).unique())
    print(f"대상일 {target} · 예측 {len(pend)}건 / {len(tickers)}종목\n")

    # ------------------------------------------------------ 1) 간밤 미국장
    us, models, sox = pd.DataFrame(), {}, None
    if args.use_us:
        from kagent import overnight
        print("=" * 74)
        us = overnight.fetch_us()
        if not us.empty:
            for tk in tickers:
                kr = _ohlcv(tk)
                if not kr.empty:
                    models[tk] = overnight.gap_model(kr, us)
        print(overnight.summarize(us, models))
        print("=" * 74 + "\n")
        sox = overnight.last_night(us).get("returns_pct", {}).get("SOX")
    else:
        print("[국내 데이터만 사용] 미국장 연동은 --use-us 로 활성화합니다.\n")

    # ---------------------------------------------------------- 2) 신규 공시
    new_disc = pd.DataFrame()
    if not args.no_dart:
        try:
            dart = get_dart(cfg)
            new_disc = dart.recent_disclosures(days=1)
        except (DartError, Exception) as e:
            print(f"[DART 건너뜀] {e}\n")

    if not new_disc.empty:
        hit = new_disc[new_disc["stock_code"].astype(str).str.zfill(6).isin(tickers)]
        flags = disclosure.reporter_flags(new_disc)
        if not hit.empty:
            print("[예측 종목 관련 신규 공시]")
            for _, r in disclosure.annotate(hit).iterrows():
                mark = {1: "호재성", -1: "악재성"}.get(r["direction"], "중립")
                print(f"  · {r['corp_name']} [{mark}] {r['report_nm']}")
                print(f"    {r['url']}")
            print()
        if not flags.empty:
            print(f"[시장 전체 이상징후 {len(flags)}건]")
            for _, r in flags.head(5).iterrows():
                print(f"  · [{r['flag']}] {r['corp_name']} — {r['report_nm']}")
            print()

    # ------------------------------------------------------ 3) 예측 재산출
    print("=" * 74)
    print("예측 수정")
    print("=" * 74)
    evening = {}
    for _, p in pend.iterrows():
        evening.setdefault(str(p["ticker"]).zfill(6), []).append(p)

    revised = []
    for tk in tickers:
        df = _ohlcv(tk)
        if df.empty:
            continue

        gap, src = manual_gap.get(tk), "수동입력"
        if gap is None and sox is not None and models.get(tk, {}).get("ok"):
            from kagent import overnight
            eg = overnight.expected_gap(models[tk], sox)
            if eg["usable"]:
                gap, src = eg["expected_gap_pct"], f"SOX {sox:+.2f}% 연동추정"

        fc = forecast.make(df, tk, KrxData.name_of(tk), overnight_gap_pct=gap)
        prev = evening[tk][0]
        led.record(fc, str(prev["process"]), "premarket", target, revised_from=int(prev["id"]))
        revised.append(fc)

        print(f"\n{forecast.format_one(fc)}")
        if gap is not None:
            print(f"   갭 출처: {src}")
        dc = (fc["center"] - prev["center"]) / prev["base_close"] * 100
        dp = (fc["p_up"] - prev["p_up"]) * 100
        print(f"   ── 어제 저녁 대비: 중심값 {dc:+.2f}%p, 상승확률 {dp:+.1f}%p")
        if abs(dc) > 3:
            print("   ⚠ 밤사이 전제가 크게 바뀌었습니다. 저녁 예측은 폐기하고 이 값을 쓰세요.")

    print("\n" + forecast.format_summary(revised))
    print("\n장 마감 후 run_evening.py 를 실행하면 오늘 예측이 채점됩니다.")
    return 0


def _ohlcv(ticker: str, days: int = 800) -> pd.DataFrame:
    end = date.today()
    start = end - timedelta(days=int(days * 1.5))
    df = stock.get_market_ohlcv_by_date(start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), ticker)
    return df[df["종가"] > 0] if df is not None and not df.empty else pd.DataFrame()


if __name__ == "__main__":
    sys.exit(main())
