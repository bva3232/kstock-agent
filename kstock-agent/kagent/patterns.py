"""차트 패턴 분석 — 단, '해석'이 아니라 '조건부 실측'을 한다.

일반적인 기술적 분석은 이렇게 말한다.
    "대량거래 장대양봉에 정배열이므로 상승 지속 가능성이 높다"

이 모듈은 대신 이렇게 말한다.
    "이 종목 과거 6년에서 같은 조건이 41번 나왔다. 다음날 평균 +0.3%, 승률 54%,
     전체 평균 대비 t=0.9 → 무의미. 다음날 표준편차는 4.8%였다."

앞의 문장은 검증이 불가능하고, 뒤의 문장은 반증이 가능하다.
투자 판단에 쓸 수 있는 건 뒤쪽뿐이다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ============================================================ 구조적 사실 추출
def structure(df: pd.DataFrame) -> dict:
    """현재 시점의 차트 구조를 '사실'로만 기술한다. 해석은 넣지 않는다.

    df: index=날짜, columns=[시가,고가,저가,종가,거래량] (pykrx 형식)
    """
    c, o, h, l = (df[k] for k in ("종가", "시가", "고가", "저가"))
    v = df["거래량"] if "거래량" in df else pd.Series(np.nan, index=df.index)
    last = -1

    ma = {w: c.rolling(w, min_periods=max(2, w // 2)).mean().iloc[last] for w in (5, 20, 60, 120)}
    rng = h.iloc[last] - l.iloc[last]
    body = abs(c.iloc[last] - o.iloc[last])

    hi_52w = c.iloc[-250:].max() if len(c) >= 250 else c.max()
    lo_52w = c.iloc[-250:].min() if len(c) >= 250 else c.min()

    ret = c.pct_change()
    streak, i = 0, last
    sign = np.sign(ret.iloc[last])
    while i >= -len(ret) and np.sign(ret.iloc[i]) == sign and sign != 0:
        streak += 1
        i -= 1

    return {
        "date": df.index[last],
        "close": float(c.iloc[last]),
        "ret_1d_pct": float(ret.iloc[last] * 100),
        "gap_pct": float((o.iloc[last] / c.iloc[last - 1] - 1) * 100) if len(c) > 1 else np.nan,
        "range_pct": float(rng / c.iloc[last - 1] * 100) if len(c) > 1 else np.nan,
        "body_ratio": float(body / rng) if rng else np.nan,      # 1에 가까울수록 장대봉
        "close_in_range": float((c.iloc[last] - l.iloc[last]) / rng) if rng else np.nan,  # 1=고가마감
        "vs_ma5_pct": float((c.iloc[last] / ma[5] - 1) * 100),
        "vs_ma20_pct": float((c.iloc[last] / ma[20] - 1) * 100),
        "vs_ma60_pct": float((c.iloc[last] / ma[60] - 1) * 100),
        "ma_aligned": bool(ma[5] > ma[20] > ma[60]),
        "vol_ratio_20d": float(v.iloc[last] / v.iloc[-21:-1].mean()) if v.notna().any() else np.nan,
        "atr_pct": float(_atr(df).iloc[last] / c.iloc[last] * 100),
        "vol_ann_20d": float(ret.iloc[-20:].std() * np.sqrt(252) * 100),
        "from_52w_high_pct": float((c.iloc[last] / hi_52w - 1) * 100),
        "from_52w_low_pct": float((c.iloc[last] / lo_52w - 1) * 100),
        "streak_days": int(streak) * (1 if sign > 0 else -1),
    }


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    pc = df["종가"].shift(1)
    tr = pd.concat([df["고가"] - df["저가"], (df["고가"] - pc).abs(), (df["저가"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=3).mean()


def levels(df: pd.DataFrame, n_levels: int = 4, bins: int = 60, lookback: int = 120,
           max_distance_pct: float = 20.0) -> list[dict]:
    """거래량 가중 가격대(볼륨 프로파일)로 지지/저항 후보를 뽑는다.

    추세선을 그리는 것보다 재현성이 있다. 사람이 자를 대고 긋는 선은
    긋는 사람마다 달라지지만, 거래가 몰린 가격대는 데이터로 확정된다.

    주의 -- 1년 새 몇 배씩 움직인 종목은 과거 거래량이 전부 현재가에서 멀리 떨어진
    가격대에 쌓여 있다. 그런 레벨은 '지지선'으로 아무 의미가 없으므로
    max_distance_pct 밖은 버리고, 남는 게 없으면 최근 스윙 고저로 대체한다.
    """
    d = df.iloc[-lookback:]
    price = (d["고가"] + d["저가"] + d["종가"]) / 3
    vol = d["거래량"] if "거래량" in d else pd.Series(1.0, index=d.index)
    hist, edges = np.histogram(price, bins=bins, weights=vol)
    cur = float(df["종가"].iloc[-1])

    cand = []
    for i in np.argsort(hist)[::-1]:
        if hist[i] <= 0:
            continue
        lo, hi = float(edges[i]), float(edges[i + 1])
        mid = (lo + hi) / 2
        dist = (mid / cur - 1) * 100
        if abs(dist) > max_distance_pct or abs(dist) < 0.3:
            continue
        cand.append({
            "level": round(mid),
            "range": (round(lo), round(hi)),
            "volume_share_pct": float(hist[i] / hist.sum() * 100),
            "side": "저항" if mid > cur else "지지",
            "distance_pct": float(dist),
            "source": "거래밀집",
        })

    out = []
    for side in ("지지", "저항"):
        same = sorted([x for x in cand if x["side"] == side], key=lambda x: abs(x["distance_pct"]))
        if same:
            out.extend(same[: max(1, n_levels // 2)])
        else:
            out.append(_swing_level(df, side, cur))

    out.sort(key=lambda x: abs(x["distance_pct"]))
    return out[:n_levels]


def _swing_level(df: pd.DataFrame, side: str, cur: float, windows=(5, 10, 20, 60)) -> dict:
    """거래밀집대가 없을 때 쓰는 대체 레벨 — 최근 스윙 고점/저점."""
    for w in windows:
        seg = df.iloc[-w:]
        lvl = float(seg["고가"].max()) if side == "저항" else float(seg["저가"].min())
        dist = (lvl / cur - 1) * 100
        if (side == "저항" and dist > 0.3) or (side == "지지" and dist < -0.3):
            return {"level": round(lvl), "range": (round(lvl), round(lvl)),
                    "volume_share_pct": float("nan"), "side": side,
                    "distance_pct": float(dist), "source": f"{w}일 스윙"}
    lvl = cur * (1.03 if side == "저항" else 0.97)
    return {"level": round(lvl), "range": (round(lvl), round(lvl)),
            "volume_share_pct": float("nan"), "side": side,
            "distance_pct": 3.0 if side == "저항" else -3.0, "source": "±3% 대체"}


# ========================================================== 조건부 과거 통계
@dataclass
class Setup:
    name: str
    mask: pd.Series
    description: str = ""


def define_setups(df: pd.DataFrame) -> list[Setup]:
    """관측 가능한 조건들. 각각이 '패턴'에 해당한다."""
    c, o, h, l = (df[k] for k in ("종가", "시가", "고가", "저가"))
    v = df["거래량"] if "거래량" in df else pd.Series(1.0, index=df.index)
    ret = c.pct_change() * 100
    ma20 = c.rolling(20, min_periods=10).mean()
    ma60 = c.rolling(60, min_periods=30).mean()
    ma5 = c.rolling(5, min_periods=3).mean()
    vol_ratio = v / v.rolling(20, min_periods=10).mean().shift(1)
    rng = h - l
    body_ratio = (c - o).abs() / rng.replace(0, np.nan)
    hi250 = c.rolling(250, min_periods=120).max()
    atrp = _atr(df) / c * 100

    return [
        Setup("대량거래 장대양봉", (ret > 3) & (vol_ratio > 2) & (body_ratio > 0.6),
              "거래량 2배 이상, 3% 이상 상승, 몸통이 긴 양봉"),
        Setup("급등 후 (+5% 이상)", ret > 5, "전일 5% 이상 상승"),
        Setup("급락 후 (-5% 이하)", ret < -5, "전일 5% 이상 하락"),
        Setup("상한가 근접 (+20% 이상)", ret > 20, "전일 20% 이상 급등"),
        Setup("정배열 + 20일선 위", (ma5 > ma20) & (ma20 > ma60) & (c > ma20), "이평선 정배열"),
        Setup("20일선 회복", (c > ma20) & (c.shift(1) <= ma20.shift(1)), "20일선 아래에서 위로 돌파"),
        Setup("20일선 이탈", (c < ma20) & (c.shift(1) >= ma20.shift(1)), "20일선 위에서 아래로 이탈"),
        Setup("3일 연속 상승", (ret > 0) & (ret.shift(1) > 0) & (ret.shift(2) > 0), ""),
        Setup("3일 연속 하락", (ret < 0) & (ret.shift(1) < 0) & (ret.shift(2) < 0), ""),
        Setup("52주 신고가권", c >= hi250 * 0.98, "52주 최고가의 98% 이상"),
        Setup("고변동성 국면", atrp > atrp.rolling(250, min_periods=120).quantile(0.8),
              "ATR이 과거 1년 상위 20%"),
        Setup("갭상승 2% 이상", (o / c.shift(1) - 1) * 100 > 2, ""),
    ]


def conditional_stats(df: pd.DataFrame, mask: pd.Series, horizon: int = 1) -> dict:
    """조건 충족일 다음 horizon일 수익률 분포 vs 전체 분포."""
    c = df["종가"]
    fwd = (c.shift(-horizon) / c - 1) * 100
    valid = fwd.notna()
    sel = fwd[mask.fillna(False) & valid]
    base = fwd[valid]

    n = len(sel)
    if n == 0:
        return {"n": 0}

    se = sel.std() / np.sqrt(n) if n > 1 else np.nan
    t = (sel.mean() - base.mean()) / se if se and se > 0 else np.nan
    wins = int((sel > 0).sum())
    p = wins / n
    win_se = np.sqrt(p * (1 - p) / n) * 100

    return {
        "n": n,
        "mean_pct": float(sel.mean()),
        "median_pct": float(sel.median()),
        "std_pct": float(sel.std()),
        "win_rate_pct": float(p * 100),
        "win_rate_se_pct": float(win_se),
        "p10_pct": float(sel.quantile(0.10)),
        "p90_pct": float(sel.quantile(0.90)),
        "base_mean_pct": float(base.mean()),
        "base_win_rate_pct": float((base > 0).mean() * 100),
        "excess_pct": float(sel.mean() - base.mean()),
        "t_stat": float(t) if t == t else np.nan,
    }


def report(df: pd.DataFrame, horizons=(1, 5)) -> pd.DataFrame:
    """전체 셋업 x 보유기간 표. 이게 '패턴 분석'의 실제 결과물이다."""
    rows = []
    for s in define_setups(df):
        active = bool(s.mask.iloc[-1]) if len(s.mask) else False
        for hz in horizons:
            st = conditional_stats(df, s.mask, hz)
            if st["n"] == 0:
                continue
            rows.append({
                "setup": s.name,
                "현재해당": "●" if active else "",
                "horizon": f"{hz}일",
                "N": st["n"],
                "평균%": round(st["mean_pct"], 2),
                "전체평균%": round(st["base_mean_pct"], 2),
                "초과%": round(st["excess_pct"], 2),
                "승률%": round(st["win_rate_pct"], 1),
                "±": round(st["win_rate_se_pct"], 1),
                "표준편차%": round(st["std_pct"], 2),
                "t": round(st["t_stat"], 2) if st["t_stat"] == st["t_stat"] else None,
                "유의": _sig(st),
            })
    return pd.DataFrame(rows)


def _sig(st: dict) -> str:
    if st["n"] < 30:
        return "표본부족"
    t = st["t_stat"]
    if t != t:
        return "-"
    if abs(t) < 2:
        return "무의미"
    return "유의(주의)"


def summarize(df: pd.DataFrame) -> str:
    """현재 활성 셋업 중 통계적으로 뭔가 있는 것만 골라 문장으로."""
    rep = report(df, horizons=(1,))
    act = rep[(rep["현재해당"] == "●")]
    if act.empty:
        return "현재 정의된 셋업 중 해당되는 것이 없습니다."
    strong = act[act["유의"] == "유의(주의)"]
    lines = [f"현재 해당 셋업 {len(act)}개: " + ", ".join(act["setup"].tolist())]
    if strong.empty:
        lines.append("이 중 다음날 수익률에 통계적으로 유의한 영향을 준 셋업은 없습니다.")
        lines.append("→ 차트만으로는 내일 방향에 대해 말할 수 있는 게 없다는 뜻입니다.")
    else:
        for _, r in strong.iterrows():
            lines.append(f"· {r['setup']}: N={r['N']}, 다음날 평균 {r['평균%']:+.2f}% "
                         f"(전체 {r['전체평균%']:+.2f}%), 승률 {r['승률%']:.0f}±{r['±']:.0f}%, t={r['t']}")
        lines.append("→ 과거 구간의 통계이며, 표본 편의와 국면 변화로 재현되지 않을 수 있습니다.")
    return "\n".join(lines)
