"""밤사이 변수 — 07:30 점검의 핵심.

한국 장 마감(15:30) 이후 다음날 개장(09:00)까지 17시간 반 동안 미국장이 열린다.
반도체 대형주의 다음날 시가는 사실상 이 구간에서 결정된다.
차트에는 전혀 없는 정보이고, 그래서 별도 모듈로 뺐다.

두 가지를 한다.
  1) 간밤 미국 지표(SOX, 개별 종목) 수집
  2) 그 종목의 '전일 미국장 수익률 -> 당일 한국 시가 갭' 관계를 과거 회귀로 추정

2번이 중요하다. "SOX가 3% 빠졌으니 우리도 빠지겠지"는 감이지만,
베타 0.82, R² 0.41 은 검증 가능한 숫자다. R² 가 낮게 나오면
그 연결고리 자체가 약하다는 뜻이므로 갭 추정을 쓰지 않으면 된다.

yfinance 가 없으면 이 모듈은 조용히 비활성화되고 나머지는 정상 동작한다.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

DEFAULT_PROXIES = {
    "SOX": "^SOX",        # 필라델피아 반도체 지수
    "MU": "MU",           # 마이크론
    "NVDA": "NVDA",
    "WDC": "WDC",
}


def available() -> bool:
    try:
        import yfinance  # noqa: F401
        return True
    except Exception:
        return False


def fetch_us(symbols: dict[str, str] | None = None, days: int = 400) -> pd.DataFrame:
    """미국 종가 시계열. 실패하면 빈 DataFrame."""
    if not available():
        return pd.DataFrame()
    import yfinance as yf

    symbols = symbols or DEFAULT_PROXIES
    end = date.today() + timedelta(days=1)
    start = end - timedelta(days=days)
    out = {}
    for label, sym in symbols.items():
        try:
            h = yf.Ticker(sym).history(start=start, end=end, auto_adjust=False)
            if h is not None and not h.empty:
                s = h["Close"]
                s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
                out[label] = s
        except Exception:
            continue
    return pd.DataFrame(out).sort_index() if out else pd.DataFrame()


def last_night(us: pd.DataFrame) -> dict:
    """가장 최근 미국 세션의 등락률."""
    if us.empty or len(us) < 2:
        return {}
    ret = (us.iloc[-1] / us.iloc[-2] - 1) * 100
    return {
        "session_date": us.index[-1].date().isoformat(),
        "returns_pct": {k: float(v) for k, v in ret.items() if v == v},
    }


def gap_model(kr: pd.DataFrame, us: pd.DataFrame, proxy: str = "SOX",
              lookback: int = 250) -> dict:
    """'간밤 미국 수익률 -> 당일 한국 시가 갭' 단순회귀.

    kr: pykrx 일봉 (시가/종가 필요), us: fetch_us 결과
    """
    if us.empty or proxy not in us.columns or kr.empty:
        return {"ok": False, "reason": "미국 데이터 없음"}

    gap = (kr["시가"] / kr["종가"].shift(1) - 1) * 100
    gap.index = pd.to_datetime(gap.index).normalize()

    us_ret = (us[proxy] / us[proxy].shift(1) - 1) * 100
    # 한국 D일 시가는 미국 D-1일(현지) 세션의 영향을 받는다.
    us_ret.index = us_ret.index + pd.Timedelta(days=1)
    us_ret = us_ret.reindex(gap.index, method="ffill", limit=3)

    d = pd.concat([gap.rename("gap"), us_ret.rename("us")], axis=1).dropna().iloc[-lookback:]
    if len(d) < 60:
        return {"ok": False, "reason": f"표본 부족 ({len(d)}건)"}

    x, y = d["us"].values, d["gap"].values
    beta, alpha = np.polyfit(x, y, 1)
    pred = alpha + beta * x
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot else np.nan
    resid_sd = float(np.std(y - pred, ddof=2))
    se_beta = resid_sd / (np.std(x) * np.sqrt(len(x))) if np.std(x) else np.nan

    return {
        "ok": True,
        "proxy": proxy,
        "n": len(d),
        "beta": float(beta),
        "alpha": float(alpha),
        "r2": float(r2),
        "t_beta": float(beta / se_beta) if se_beta and se_beta == se_beta else np.nan,
        "resid_sd_pct": resid_sd,
        "usable": bool(r2 >= 0.15 and abs(beta / se_beta) >= 3) if se_beta else False,
    }


def expected_gap(model: dict, us_return_pct: float) -> dict:
    if not model.get("ok"):
        return {"ok": False, "reason": model.get("reason")}
    point = model["alpha"] + model["beta"] * us_return_pct
    sd = model["resid_sd_pct"]
    return {
        "ok": True,
        "expected_gap_pct": float(point),
        "range_68_pct": (float(point - sd), float(point + sd)),
        "usable": model["usable"],
        "note": (f"베타 {model['beta']:.2f}, R²={model['r2']:.2f}, N={model['n']} — "
                 + ("연결고리가 통계적으로 확인됩니다."
                    if model["usable"] else
                    "설명력이 낮아 갭 추정을 신뢰하기 어렵습니다. 참고만 하세요.")),
    }


def summarize(us: pd.DataFrame, models: dict[str, dict] | None = None) -> str:
    ln = last_night(us)
    if not ln:
        L = ["[간밤 미국장] 데이터 없음 (yfinance 미설치 또는 조회 실패)",
             "  → pip install yfinance 로 활성화하거나, --gap 옵션으로 프리마켓 갭을 직접 입력하세요."]
        return "\n".join(L)

    L = [f"[간밤 미국장] {ln['session_date']} 세션"]
    for k, v in ln["returns_pct"].items():
        L.append(f"  {k:<6} {v:+.2f}%")
    if models:
        L.append("\n[갭 연동 모델]")
        for tk, m in models.items():
            if m.get("ok"):
                L.append(f"  {tk}: 베타 {m['beta']:.2f}, R²={m['r2']:.2f}, "
                         f"t={m['t_beta']:.1f}, N={m['n']} "
                         f"{'[사용]' if m['usable'] else '[신뢰도 낮음]'}")
            else:
                L.append(f"  {tk}: {m.get('reason')}")
    return "\n".join(L)
