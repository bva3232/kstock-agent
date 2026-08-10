#!/usr/bin/env python3
"""매일 아침 브리핑 생성.

    python run_morning_brief.py                 # 기본 실행
    python run_morning_brief.py --no-dart       # DART 없이 시세만
    python run_morning_brief.py --top 15

장 시작 전(08:30 전후) 실행을 전제로, 직전 영업일 종가까지의 데이터를 사용한다.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from datetime import date, timedelta

import pandas as pd

from kagent import brief, disclosure, screener, signals
from kagent.config import Config
from kagent.dart import DartError, get_dart
from kagent.krx import KrxData, recent_trading_days


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--top", type=int, default=None)
    ap.add_argument("--no-dart", action="store_true")
    ap.add_argument("--dart-days", type=int, default=3, help="공시 조회 기간(일)")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    acct, uni, scr = cfg["account"], cfg["universe"], cfg["screen"]
    top_n = args.top or scr["top_n"]

    krx = KrxData(cfg.cache_dir)

    # 1) 영업일과 가격 패널 ----------------------------------------------------
    print("· 영업일 확인 중...")
    days = recent_trading_days(scr["lookback_days"])
    if not days:
        print("영업일 데이터를 가져오지 못했습니다. 네트워크를 확인하세요.")
        return 1
    base_date = days[-1]
    print(f"· 기준일 {base_date}, {len(days)}영업일 시세 수집 (최초 실행은 몇 분 걸립니다)")
    panels = krx.panels(days, market="ALL")
    if "종가" not in panels:
        print("시세 수집 실패.")
        return 1

    universe = krx.tickers(base_date, tuple(uni["markets"]))
    if universe:
        cols = [c for c in panels["종가"].columns if c in universe]
        panels = {k: v[cols] for k, v in panels.items()}
    print(f"· 대상 종목 {panels['종가'].shape[1]}개")

    # 2) DART 공시 -----------------------------------------------------------
    event_scores, flags, summary, disc = None, None, None, pd.DataFrame()
    if not args.no_dart:
        try:
            dart = get_dart(cfg)
            print(f"· DART 공시 {args.dart_days}일치 수집 중...")
            disc = dart.recent_disclosures(days=args.dart_days)
            if not disc.empty:
                event_scores = disclosure.score_events(disc)
                flags = disclosure.reporter_flags(disc)
                ann = disclosure.annotate(disc)
                summary = (ann[ann["category"].notna()]
                           .groupby(["category", "direction"]).size()
                           .reset_index(name="count").sort_values("count", ascending=False))
                print(f"  공시 {len(disc)}건, 이벤트 분류 {int(ann['category'].notna().sum())}건")
        except DartError as e:
            print(f"  [DART 건너뜀] {e}")
        except Exception as e:
            print(f"  [DART 오류] {e}")

    # 3) 수급 ----------------------------------------------------------------
    flow_start = days[-6] if len(days) >= 6 else days[0]
    flows = []
    for m in uni["markets"]:
        f = krx.net_purchases(flow_start, base_date, m, "외국인")
        if not f.empty:
            flows.append(f)
    flows = pd.concat(flows) if flows else None

    # 4) 피처 -> 필터 -> 점수 --------------------------------------------------
    print("· 지표 계산 중...")
    feat = screener.build_features(
        panels,
        krx.fundamental(base_date),
        krx.market_cap(base_date),
        event_scores,
        flows,
    )
    filtered = screener.apply_universe_filter(feat, uni)
    log = filtered.attrs.get("filter_log", {})
    print(f"  유니버스 {log.get('before')} → {log.get('after')} ({log.get('reasons')})")
    if filtered.empty:
        print("필터를 통과한 종목이 없습니다. config 의 min_market_cap / min_avg_turnover 를 낮춰보세요.")
        return 1

    scored = screener.composite_score(filtered)

    # 5) 후보 조립 -----------------------------------------------------------
    by_code = {}
    if not disc.empty:
        d = disc[disc["stock_code"].astype(str).str.len() == 6]
        for code, grp in d.groupby("stock_code"):
            by_code[str(code).zfill(6)] = grp.head(3)[["report_nm", "rcept_dt", "url"]].to_dict("records")

    candidates = []
    for ticker, row in scored.head(top_n).iterrows():
        candidates.append({
            "ticker": ticker,
            "name": row["name"],
            "score": float(row["score"]),
            "price": float(row["price"]),
            "ret_1d": float(row["ret_1d"]),
            "ret_20d": float(row["ret_20d"]),
            "atr_pct": float(row["atr_pct"]) if row["atr_pct"] == row["atr_pct"] else 0.0,
            "PBR": float(row["PBR"]) if "PBR" in row and row["PBR"] == row["PBR"] else None,
            "why": screener.explain(row),
            "plan": screener.position_plan(row, acct),
            "disclosures": by_code.get(ticker, []),
        })

    # 6) 시장 국면 -----------------------------------------------------------
    idx = krx.index_history(days[0], base_date)
    regime = signals.market_regime(idx, panels["종가"])

    # 7) 렌더 ----------------------------------------------------------------
    out = cfg.output_dir / f"brief_{base_date}.html"
    brief.render(out, regime, candidates, flags, summary, acct, base_date)
    print(f"\n완료 → {out.resolve()}")
    print(f"시장 국면: {regime['state']} | 후보 {len(candidates)}종목"
          + (f" | 취재 단서 {len(flags)}건" if flags is not None and not flags.empty else ""))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
