"""스크리너: 유니버스 구성 -> 피처 계산 -> 합성 점수 -> 포지션 사이징.

합성 점수는 '예측 확률'이 아니다. 여러 관찰 가능한 조건을 한 축으로 줄 세운
순위일 뿐이고, 각 종목마다 어떤 조건이 점수에 기여했는지 근거를 함께 남긴다.
근거 없는 숫자는 리포트에 싣지 않는다는 게 이 모듈의 원칙이다.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import signals as sg

# 합성 점수 가중치. 합계 1.0
WEIGHTS = {
    "trend": 0.25,        # 정배열 (추세)
    "momentum": 0.20,     # 20일 모멘텀
    "surge": 0.20,        # 거래대금 급증
    "event": 0.20,        # DART 공시 이벤트
    "flow": 0.15,         # 외국인/기관 순매수
}
VOL_PENALTY = 0.15        # 변동성 상위 종목에 부과하는 감점


def build_features(
    panels: dict[str, pd.DataFrame],
    fundamental: pd.DataFrame,
    caps: pd.DataFrame,
    event_scores: pd.DataFrame | None = None,
    flows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    close = panels["종가"]
    high = panels.get("고가", close)
    low = panels.get("저가", close)
    value = panels.get("거래대금", close * 0)

    feat = pd.DataFrame(index=close.columns)
    feat["price"] = close.iloc[-1]
    feat["ret_1d"] = (close.iloc[-1] / close.iloc[-2] - 1) * 100 if len(close) > 1 else np.nan
    feat["ret_20d"] = sg.momentum(close, 20)
    feat["ret_60d"] = sg.momentum(close, 60)
    feat["trend"] = sg.trend_alignment(close)
    feat["atr_pct"] = sg.atr_pct(high, low, close)
    feat["vol_20d"] = sg.realized_vol(close)
    feat["dd_60d"] = sg.drawdown_from_high(close, 60)
    feat["surge"] = sg.turnover_surge(value)
    feat["avg_turnover"] = sg.avg_turnover(value)

    if not caps.empty:
        c = caps.set_index("ticker")
        for col in ("시가총액", "상장주식수"):
            if col in c.columns:
                feat[col] = c[col]
    if not fundamental.empty:
        f = fundamental.set_index("ticker")
        for col in ("PER", "PBR", "DIV", "EPS", "BPS"):
            if col in f.columns:
                feat[col] = f[col]

    feat["event_score"] = 0.0
    feat["top_event"] = None
    if event_scores is not None and not event_scores.empty:
        es = event_scores.copy()
        es.index = es.index.astype(str).str.zfill(6)
        feat["event_score"] = es["event_score"].reindex(feat.index).fillna(0.0)
        feat["top_event"] = es["top_event"].reindex(feat.index)

    feat["flow_ratio"] = 0.0
    if flows is not None and not flows.empty and "순매수거래대금" in flows.columns:
        fl = flows.set_index("ticker")["순매수거래대금"]
        feat["flow_ratio"] = (fl.reindex(feat.index) / feat["avg_turnover"].replace(0, np.nan)).fillna(0.0)

    feat["name"] = _names(feat.index)
    return feat


def _names(tickers) -> pd.Series:
    from .krx import KrxData
    return pd.Series({t: KrxData.name_of(t) for t in tickers})


def apply_universe_filter(feat: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """유동성·규모·종목유형 필터. 걸러진 이유를 로그로 남긴다."""
    d = feat.copy()
    before = len(d)
    reasons = {}

    if "시가총액" in d.columns and cfg.get("min_market_cap_krw"):
        keep = d["시가총액"] >= cfg["min_market_cap_krw"]
        reasons["시총 미달"] = int((~keep).sum())
        d = d[keep]

    if cfg.get("min_avg_turnover_krw"):
        keep = d["avg_turnover"] >= cfg["min_avg_turnover_krw"]
        reasons["거래대금 미달"] = int((~keep).sum())
        d = d[keep]

    if cfg.get("exclude_preferred"):
        keep = ~d.index.astype(str).str.endswith(("5", "7", "9", "K", "L", "M"))
        keep |= d.index.astype(str).str.endswith("0")
        reasons["우선주 추정"] = int((~keep).sum())
        d = d[keep]

    kws = cfg.get("exclude_keywords") or []
    if kws:
        pattern = "|".join(kws)
        keep = ~d["name"].astype(str).str.contains(pattern, na=False)
        reasons["제외 키워드"] = int((~keep).sum())
        d = d[keep]

    if cfg.get("max_pbr") and "PBR" in d.columns:
        keep = (d["PBR"] <= cfg["max_pbr"]) & (d["PBR"] > 0)
        reasons["PBR 범위 밖"] = int((~keep).sum())
        d = d[keep]

    d.attrs["filter_log"] = {"before": before, "after": len(d), "reasons": reasons}
    return d


def composite_score(feat: pd.DataFrame) -> pd.DataFrame:
    d = feat.copy()
    r_trend = sg.pct_rank(d["trend"])
    r_mom = sg.pct_rank(d["ret_20d"])
    r_surge = sg.pct_rank(d["surge"].clip(upper=10))
    r_event = sg.pct_rank(d["event_score"]) if d["event_score"].abs().sum() > 0 else pd.Series(0.5, index=d.index)
    r_flow = sg.pct_rank(d["flow_ratio"]) if d["flow_ratio"].abs().sum() > 0 else pd.Series(0.5, index=d.index)
    r_vol = sg.pct_rank(d["vol_20d"])

    d["score"] = (
        WEIGHTS["trend"] * r_trend
        + WEIGHTS["momentum"] * r_mom
        + WEIGHTS["surge"] * r_surge
        + WEIGHTS["event"] * r_event
        + WEIGHTS["flow"] * r_flow
        - VOL_PENALTY * r_vol
    ) * 100

    d["r_trend"], d["r_mom"], d["r_surge"] = r_trend, r_mom, r_surge
    d["r_event"], d["r_flow"], d["r_vol"] = r_event, r_flow, r_vol
    return d.sort_values("score", ascending=False)


def explain(row: pd.Series) -> list[str]:
    """왜 상위에 올라왔는지 사람이 읽을 수 있는 근거 목록."""
    out = []
    if row["trend"] >= 0.75:
        out.append(f"정배열 (종가>5>20>60일선 중 {int(row['trend'] * 4)}/4 충족)")
    if row["ret_20d"] > 0:
        out.append(f"20일 수익률 +{row['ret_20d']:.1f}% (상위 {(1 - row['r_mom']) * 100:.0f}%)")
    if row["surge"] >= 1.8:
        out.append(f"거래대금 20일 평균 대비 {row['surge']:.1f}배")
    if abs(row.get("event_score", 0)) > 0.05:
        sign = "호재성" if row["event_score"] > 0 else "악재성"
        out.append(f"최근 {sign} 공시 감지: {row.get('top_event') or '분류됨'}")
    if row.get("flow_ratio", 0) > 0.05:
        out.append(f"외국인 순매수 (일평균 거래대금의 {row['flow_ratio'] * 100:.0f}%)")
    if row["dd_60d"] > -5:
        out.append(f"60일 고점 대비 {row['dd_60d']:.1f}% (신고가권)")
    if row["vol_20d"] > 60:
        out.append(f"⚠ 연환산 변동성 {row['vol_20d']:.0f}% — 손절폭이 넓어야 함")
    if row.get("PBR", np.nan) == row.get("PBR", np.nan) and row.get("PBR", 0) > 5:
        out.append(f"⚠ PBR {row['PBR']:.1f}배로 밸류 부담")
    return out


def position_plan(row: pd.Series, account: dict) -> dict:
    """리스크 기반 수량 산출.

    손절폭 = ATR x 배수. 1회 거래 감내 손실 = 계좌 x risk_per_trade.
    수량 = 감내손실 / (주당 손절폭). 종목당 상한(계좌/최대종목수)도 함께 적용.
    """
    seed = account["seed_krw"]
    price = float(row["price"])
    atr_pct = float(row["atr_pct"]) if row["atr_pct"] == row["atr_pct"] else 3.0
    stop_pct = max(atr_pct * account["stop_loss_atr_mult"], 4.0)

    risk_budget = seed * account["risk_per_trade"]
    per_share_risk = price * stop_pct / 100
    qty_by_risk = math.floor(risk_budget / per_share_risk) if per_share_risk > 0 else 0
    qty_by_cap = math.floor((seed / account["max_positions"]) / price) if price > 0 else 0
    qty = max(0, min(qty_by_risk, qty_by_cap))

    cost = account.get("roundtrip_cost_pct", 0.3)
    return {
        "price": price,
        "qty": qty,
        "notional": qty * price,
        "stop_pct": stop_pct,
        "stop_price": round(price * (1 - stop_pct / 100)),
        "risk_krw": round(qty * per_share_risk),
        "breakeven_pct": cost,
        "note": "1주도 못 사는 가격대" if qty == 0 else
                f"손절 -{stop_pct:.1f}% 도달 시 손실 약 {qty * per_share_risk:,.0f}원",
    }
