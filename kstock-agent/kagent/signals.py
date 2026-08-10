"""신호 계산. 모든 지표는 전종목 와이드 패널에 대해 벡터 연산으로 계산한다.

지표를 일부러 단순하게 유지했다. 파라미터가 많아질수록 과거 데이터에
맞춰 끼워맞추기(과최적화)가 쉬워지고, 그렇게 나온 숫자는 미래에 재현되지 않는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------------------------------------------- 개별 지표
def moving_average(close: pd.DataFrame, window: int) -> pd.DataFrame:
    return close.rolling(window, min_periods=max(2, window // 2)).mean()


def momentum(close: pd.DataFrame, window: int) -> pd.Series:
    """최근 window 영업일 수익률(%). 마지막 행 기준."""
    if len(close) <= window:
        window = max(1, len(close) - 1)
    return (close.iloc[-1] / close.iloc[-1 - window] - 1.0) * 100


def atr_pct(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int = 14) -> pd.Series:
    """종가 대비 ATR 비율(%). 손절폭과 포지션 사이즈 산정에 쓴다."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).stack(), (high - prev_close).abs().stack(), (low - prev_close).abs().stack()],
        axis=1,
    ).max(axis=1).unstack()
    atr = tr.rolling(window, min_periods=max(3, window // 2)).mean()
    return (atr.iloc[-1] / close.iloc[-1] * 100).replace([np.inf, -np.inf], np.nan)


def turnover_surge(value: pd.DataFrame, window: int = 20) -> pd.Series:
    """당일 거래대금 / 과거 평균 거래대금. 2.0 이상이면 뭔가 벌어지고 있다는 뜻."""
    base = value.iloc[-(window + 1):-1].mean()
    return (value.iloc[-1] / base.replace(0, np.nan))


def avg_turnover(value: pd.DataFrame, window: int = 20) -> pd.Series:
    return value.iloc[-window:].mean()


def trend_alignment(close: pd.DataFrame) -> pd.Series:
    """정배열 점수 0~1. 종가>MA5>MA20>MA60 을 각각 확인."""
    ma5, ma20, ma60 = (moving_average(close, w).iloc[-1] for w in (5, 20, 60))
    px = close.iloc[-1]
    checks = [px > ma5, ma5 > ma20, ma20 > ma60, px > ma20]
    return pd.concat(checks, axis=1).sum(axis=1) / len(checks)


def drawdown_from_high(close: pd.DataFrame, window: int = 60) -> pd.Series:
    """구간 고점 대비 현재가 낙폭(%). 0 에 가까울수록 신고가권."""
    hi = close.iloc[-window:].max()
    return (close.iloc[-1] / hi - 1.0) * 100


def realized_vol(close: pd.DataFrame, window: int = 20) -> pd.Series:
    ret = close.pct_change()
    return ret.iloc[-window:].std() * np.sqrt(252) * 100


# ------------------------------------------------------------ 횡단면 정규화
def pct_rank(s: pd.Series) -> pd.Series:
    """0~1 백분위. 이상치에 강해서 z-score 보다 안전하다."""
    return s.rank(pct=True, na_option="keep")


def zscore(s: pd.Series, clip: float = 3.0) -> pd.Series:
    m, sd = s.mean(), s.std()
    if not sd or np.isnan(sd):
        return pd.Series(0.0, index=s.index)
    return ((s - m) / sd).clip(-clip, clip)


# ---------------------------------------------------------------- 시장 국면
def market_regime(index_df: pd.DataFrame, close_panel: pd.DataFrame | None = None) -> dict:
    """지수 추세 + 시장 폭(breadth)으로 위험선호 국면을 판정한다.

    'risk_off' 면 신규 진입을 줄이는 게 맞다. 개별 종목 신호가 아무리 좋아도
    시장 전체가 무너질 때는 대부분 같이 빠진다.
    """
    close = index_df["종가"] if "종가" in index_df.columns else index_df.iloc[:, -1]
    ma20 = close.rolling(20, min_periods=5).mean()
    ma60 = close.rolling(60, min_periods=15).mean()

    last, m20, m60 = close.iloc[-1], ma20.iloc[-1], ma60.iloc[-1]
    ret5 = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) > 6 else np.nan
    ret20 = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) > 21 else np.nan

    breadth = np.nan
    if close_panel is not None and not close_panel.empty:
        pm20 = moving_average(close_panel, 20).iloc[-1]
        breadth = float((close_panel.iloc[-1] > pm20).mean() * 100)

    score = sum([last > m20, m20 > m60, (ret20 or 0) > 0])
    if score >= 3 and (np.isnan(breadth) or breadth >= 50):
        state = "risk_on"
    elif score <= 1:
        state = "risk_off"
    else:
        state = "neutral"

    return {
        "state": state,
        "index_close": float(last),
        "vs_ma20_pct": float((last / m20 - 1) * 100) if m20 else np.nan,
        "vs_ma60_pct": float((last / m60 - 1) * 100) if m60 else np.nan,
        "ret_5d_pct": float(ret5) if ret5 == ret5 else np.nan,
        "ret_20d_pct": float(ret20) if ret20 == ret20 else np.nan,
        "breadth_above_ma20_pct": breadth,
    }
