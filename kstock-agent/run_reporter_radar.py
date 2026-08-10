#!/usr/bin/env python3
"""기자용 공시 레이더 — 시세 데이터 없이 DART만으로 돌아간다.

    python run_reporter_radar.py                    # 최근 1일
    python run_reporter_radar.py --days 7
    python run_reporter_radar.py --company 삼성전자  # 특정 회사 이력
    python run_reporter_radar.py --days 30 --csv out.csv

취재 관점에서 보는 것
  - 중대사항(횡령·배임, 상장위험, 감사의견) 즉시 노출
  - 정정공시가 반복되는 회사
  - 수주·증자 공시를 냈다가 해지하는 되돌림 패턴
  - 최대주주 변경
  - 금요일 접수된 악재성 공시 (이른바 '올빼미' 1차 필터)
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

import pandas as pd

from kagent import disclosure
from kagent.config import Config
from kagent.dart import get_dart


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--company", default=None, help="회사명 (부분일치)")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    dart = get_dart(cfg)

    if args.company:
        print(f"▶ '{args.company}' 최근 {args.days}일 공시 검색\n")
    else:
        print(f"▶ 최근 {args.days}일 전체 공시 스캔\n")

    df = dart.list_disclosures(date.today() - timedelta(days=args.days), date.today())
    if args.company:
        df = df[df["corp_name"].str.contains(args.company, na=False)].reset_index(drop=True)
        if df.empty:
            print(f"'{args.company}' 관련 공시가 최근 {args.days}일 내에 없습니다.")
            return 1
        hit_names = df["corp_name"].unique()
        if len(hit_names) > 1:
            print(f"여러 회사가 검색됐습니다: {', '.join(hit_names[:20])}\n")

    if df.empty:
        print("해당 기간 공시가 없습니다.")
        return 0

    ann = disclosure.annotate(df)
    print(f"총 {len(df):,}건 수집, 이벤트 분류 {int(ann['category'].notna().sum()):,}건\n")

    # --- 취재 단서 ---------------------------------------------------------
    flags = disclosure.reporter_flags(df)
    if flags.empty:
        print("눈에 띄는 이상징후 없음.\n")
    else:
        print("=" * 74)
        print("취재 단서")
        print("=" * 74)
        for _, r in flags.iterrows():
            print(f"\n[{r['flag']}] {r['corp_name']} ({r['stock_code'] or '-'})")
            print(f"  {r['report_nm']}")
            print(f"  {r['rcept_dt']} · {r['why']}")
            if r.get("note"):
                print(f"  확인포인트: {r['note']}")
            print(f"  {r['url']}")

    # --- 금요일 악재 공시 ----------------------------------------------------
    late = disclosure.late_filings(df)
    if not late.empty:
        print("\n" + "=" * 74)
        print(f"금요일 접수 악재성 공시 {len(late)}건 (접수시각은 원문에서 확인 필요)")
        print("=" * 74)
        for _, r in late.head(20).iterrows():
            print(f"  {r['rcept_dt']} {r['corp_name']:<14} {r['report_nm'][:44]}")

    # --- 유형별 집계 --------------------------------------------------------
    cat = ann[ann["category"].notna()].groupby("category").size().sort_values(ascending=False)
    if not cat.empty:
        print("\n" + "=" * 74)
        print("유형별 집계")
        print("=" * 74)
        for name, n in cat.items():
            print(f"  {name:<14} {n:>5}건")

    if args.csv:
        ann.to_csv(args.csv, index=False, encoding="utf-8-sig")
        print(f"\n저장 → {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
