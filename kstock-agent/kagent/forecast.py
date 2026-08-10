"""예측 생성 — 방향 확률 + 예상 단가 + 구간.

두 가지 원칙으로 만든다.

1) 방향 확률은 지어내지 않고 그 종목의 과거 조건부 승률에서 뽑되,
   표본 수에 따라 50%로 수축(shrinkage)시킨다.
   표본 12건짜리 셋업의 승률 75%를 그대로 쓰면 그건 예측이 아니라 잡음이다.

2) 구간은 고정 배수(±5% 같은)가 아니라 현재 변동성 국면으로 스케일한다.
   같은 종목도 평온기엔 하루 2%, 지금 같은 국면엔 하루 12%씩 움직인다.
   같은 밴드를 쓰면 한쪽에선 항상 빗나가고 다른 쪽에선 항상 맞는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .patterns import _atr, conditional_stats, define_setups, levels, structure

SHRINK_K = 50.0        # 표본 K건일 때 절반만 반영
MAX_TILT = 0.12        # 방향 확률은 50% ± 12%p 를 넘지 않는다


def _vol_scaled_quantiles(df: pd.DataFrame, vol_window: int = 20) -> dict:
    """표준화 수익률의 과거 분위를 현재 변동성으로 되돌린다."""
    c = df["종가"]
    ret = c.pct_change() * 100
    vol = ret.rolling(vol_window, min_periods=vol_window // 2).std().shift(1)
    z = (ret / vol.replace(0, np.nan)).dropna()
    z = z[np.isfinite(z)]
    cur_vol = float(ret.iloc[-vol_window:].std())
    if not np.isfinite(cur_vol) or cur_vol <= 0 or len(z) < 100:
        cur_vol = float(_atr(df).iloc[-1] / c.iloc[-1] * 100)
        z = pd.Series(np.random.default_rng(0).standard_normal(1000))
    return {
        "cur_vol_pct": cur_vol,
        "q": {p: float(z.quantile(p)) * cur_vol for p in (0.10, 0.20, 0.50, 0.80, 0.90)},
        "n_z": int(len(z)),
    }


def direction_probability(df: pd.DataFrame) -> tuple[float, float, list[str]]:
    """현재 활성 셋업들에서 상승확률과 기대 변동률을 함께 뽑는다.

    확률과 예상 단가를 각각 다른 곳에서 계산하면
    "하락 우위 47%인데 예상 종가는 +0.04%" 같은 모순이 나온다.
    둘 다 같은 조건부 통계(승률 / 평균 초과수익)에서 뽑고 같은 수축을 적용한다.

    반환: (상승확률, 기대 변동률 %, 근거 문자열들)
    """
    fwd = (df["종가"].shift(-1) / df["종가"] - 1) * 100
    base_p = float((fwd.dropna() > 0).mean())
    base_r = float(fwd.dropna().mean())

    p_tilts, r_tilts, weights, notes = [], [], [], []
    for s in define_setups(df):
        if len(s.mask) == 0 or not bool(s.mask.iloc[-1]):
            continue
        st = conditional_stats(df, s.mask, horizon=1)
        if st["n"] < 10:
            notes.append(f"{s.name}: 표본 {st['n']}건 — 반영 안 함")
            continue
        shrink = st["n"] / (st["n"] + SHRINK_K)
        p_tilt = shrink * (st["win_rate_pct"] / 100 - base_p)
        r_tilt = shrink * st["excess_pct"]
        p_tilts.append(p_tilt)
        r_tilts.append(r_tilt)
        weights.append(st["n"])
        notes.append(
            f"{s.name}: N={st['n']}, 승률 {st['win_rate_pct']:.0f}% (기본 {base_p * 100:.0f}%), "
            f"평균 {st['mean_pct']:+.2f}% (기본 {base_r:+.2f}%) → 수축 후 "
            f"확률 {p_tilt * 100:+.1f}%p / 수익률 {r_tilt:+.2f}%p"
        )

    if p_tilts:
        p_tilt = float(np.average(p_tilts, weights=weights))
        r_tilt = float(np.average(r_tilts, weights=weights))
    else:
        p_tilt = r_tilt = 0.0
        notes.append("활성 셋업 없음 — 기본값 사용")

    p = float(np.clip(base_p + p_tilt, 0.5 - MAX_TILT, 0.5 + MAX_TILT))
    exp_ret = base_r + r_tilt
    return p, exp_ret, notes


def _conditional_cases(df: pd.DataFrame, cur_vol: float, vol_window: int = 20) -> dict:
    """상승했을 때의 평균 상승폭 / 하락했을 때의 평균 하락폭, 그리고 큰 움직임 확률.

    꼬리가 두꺼운 분포에서 전체 평균은 거의 쓸모가 없다.
    -0.85% 는 (상승 47% x +6.1%) + (하락 53% x -5.8%) 의 결과인데,
    실제로 -0.85% 로 끝나는 날은 사실상 없다. 상승/하락 각각의 크기를 따로 내야
    손절폭과 목표가를 잡을 수 있다.
    """
    ret = df["종가"].pct_change() * 100
    vol = ret.rolling(vol_window, min_periods=vol_window // 2).std().shift(1)
    z = (ret / vol.replace(0, np.nan)).dropna()
    z = z[np.isfinite(z)]
    if len(z) < 100:
        z = pd.Series(np.random.default_rng(0).standard_normal(1000))

    scaled = z * cur_vol
    up, dn = scaled[scaled > 0], scaled[scaled < 0]
    # '보통'은 중앙값을 쓴다. 평균은 꼬리에 끌려 올라가서 90분위와 거의 붙어버린다.
    return {
        "up_move_pct": float(up.median()) if len(up) else cur_vol,
        "down_move_pct": float(dn.median()) if len(dn) else -cur_vol,
        "up_p75_pct": float(up.quantile(0.90)) if len(up) else cur_vol * 2,
        "down_p75_pct": float(dn.quantile(0.10)) if len(dn) else -cur_vol * 2,
        "p_move_gt_3": float((scaled.abs() > 3).mean()),
        "p_move_gt_5": float((scaled.abs() > 5).mean()),
        "p_move_gt_10": float((scaled.abs() > 10).mean()),
    }


def make(df: pd.DataFrame, ticker: str, name: str | None = None,
         overnight_gap_pct: float | None = None) -> dict:
    """단일 종목 예측 객체 생성.

    overnight_gap_pct 를 주면(개장 전 점검 시) 기준가를 갭 반영가로 옮긴다.
    """
    st = structure(df)
    base = float(st["close"])
    vs = _vol_scaled_quantiles(df)
    p_up, exp_ret, notes = direction_probability(df)

    # 기대 변동률이 하루 변동성의 절반을 넘지 않도록 제한한다.
    # 조건부 평균은 표본 몇 건에 쉽게 끌려다니므로 그대로 쓰면 과한 값이 나온다.
    cap = 0.5 * vs["cur_vol_pct"]
    exp_ret = float(np.clip(exp_ret, -cap, cap))

    anchor = base * (1 + overnight_gap_pct / 100) if overnight_gap_pct is not None else base
    center = anchor * (1 + exp_ret / 100)

    band60 = (anchor * (1 + vs["q"][0.20] / 100), anchor * (1 + vs["q"][0.80] / 100))
    band80 = (anchor * (1 + vs["q"][0.10] / 100), anchor * (1 + vs["q"][0.90] / 100))

    lv = levels(df)
    sup = next((x["level"] for x in lv if x["side"] == "지지"), None)
    res = next((x["level"] for x in lv if x["side"] == "저항"), None)

    cc = _conditional_cases(df, vs["cur_vol_pct"])

    fc = {
        "ticker": ticker,
        "name": name or ticker,
        "base_close": base,
        "anchor": anchor,
        "overnight_gap_pct": overnight_gap_pct,
        "p_up": p_up,
        "p_down": 1 - p_up,
        "exp_ret_pct": exp_ret,
        "center": center,
        "center_pct": (center / base - 1) * 100,
        "band60": band60,
        "band80": band80,
        "band60_pct": ((band60[0] / base - 1) * 100, (band60[1] / base - 1) * 100),
        "band80_pct": ((band80[0] / base - 1) * 100, (band80[1] / base - 1) * 100),
        "support": sup,
        "resistance": res,
        "levels": lv,
        "cur_vol_pct": vs["cur_vol_pct"],
        "atr_pct": st["atr_pct"],
        "structure": st,
        "drivers": notes,
        # 상승/하락 각각의 크기 — 실제로 쓰는 값은 이쪽이다
        "up_pct": cc["up_move_pct"],
        "down_pct": cc["down_move_pct"],
        "up_price": anchor * (1 + cc["up_move_pct"] / 100),
        "down_price": anchor * (1 + cc["down_move_pct"] / 100),
        "up_stretch_price": anchor * (1 + cc["up_p75_pct"] / 100),
        "down_stretch_price": anchor * (1 + cc["down_p75_pct"] / 100),
        "up_stretch_pct": cc["up_p75_pct"],
        "down_stretch_pct": cc["down_p75_pct"],
        "p_move_gt_3": cc["p_move_gt_3"],
        "p_move_gt_5": cc["p_move_gt_5"],
        "p_move_gt_10": cc["p_move_gt_10"],
    }
    fc["divergence"] = _divergence(fc)
    fc["headline"] = headline(fc)
    return fc


def _divergence(fc: dict) -> str | None:
    """확률과 기대값이 서로 다른 방향을 가리키는 경우를 잡아낸다.

    승률 47%인데 기대수익 +0.3% 같은 조합은 오류가 아니라 실제 분포의 성질이다.
    "자주 조금씩 빠지고, 가끔 크게 오른다"는 뜻이고 이건 매매 방식이 달라져야 한다.
    (승률형이면 짧게 자주, 기대값형이면 길게 버티기)
    숨기면 읽는 사람이 둘 중 하나를 잘못 읽으므로 명시한다.
    """
    p_up_side = fc["p_up"] >= 0.5
    ret_up_side = fc["exp_ret_pct"] > 0
    if p_up_side == ret_up_side or abs(fc["exp_ret_pct"]) < 0.05:
        return None
    if ret_up_side:
        return ("확률은 하락 우세인데 기대수익은 플러스 — 하락은 잦고 얕게, "
                "상승은 드물고 크게 나오는 분포입니다. 승률 위주면 관망, 기대값 위주면 소액 진입.")
    return ("확률은 상승 우세인데 기대수익은 마이너스 — 소폭 상승이 잦지만 "
            "가끔 큰 폭으로 빠지는 분포입니다. 손절을 반드시 걸어야 하는 유형입니다.")


def headline(fc: dict) -> str:
    """한 줄 결론.

    전체 평균(-0.85% 같은 값)은 헤드라인에 쓰지 않는다.
    상승했을 때 얼마 / 하락했을 때 얼마 — 이 두 숫자가 실제로 쓰는 값이다.
    """
    side = "상승" if fc["p_up"] >= 0.5 else "하락"
    tail = " ※확률·기대값 불일치" if fc.get("divergence") else ""
    return (f"{fc['name']} — {side} 우세({max(fc['p_up'], fc['p_down']) * 100:.0f}%) · "
            f"오르면 {fc['up_price']:,.0f}원({fc['up_pct']:+.1f}%) / "
            f"빠지면 {fc['down_price']:,.0f}원({fc['down_pct']:+.1f}%) · "
            f"5% 이상 움직일 확률 {fc['p_move_gt_5'] * 100:.0f}%{tail}")


def format_one(fc: dict) -> str:
    L = []
    gap = fc.get("overnight_gap_pct")
    side = "상승" if fc["p_up"] >= 0.5 else "하락"

    L.append(f"■ {fc['name']} ({fc['ticker']})  전일종가 {fc['base_close']:,.0f}원")
    L.append(f"   ▶ {side} 우세 — 상승 {fc['p_up'] * 100:.0f}% / 하락 {fc['p_down'] * 100:.0f}%")
    if gap is not None:
        L.append(f"   갭 {gap:+.2f}% 반영 → 시가 기준선 {fc['anchor']:,.0f}원")
    L.append("")
    L.append(f"   ▲ 오르는 경우  보통 {fc['up_price']:,.0f}원 ({fc['up_pct']:+.1f}%)"
             f"   상위10% {fc['up_stretch_price']:,.0f}원 ({fc['up_stretch_pct']:+.1f}%)")
    L.append(f"   ▼ 빠지는 경우  보통 {fc['down_price']:,.0f}원 ({fc['down_pct']:+.1f}%)"
             f"   하위10% {fc['down_stretch_price']:,.0f}원 ({fc['down_stretch_pct']:+.1f}%)")
    L.append("")
    L.append(f"   큰 변동 확률: ±3% 초과 {fc['p_move_gt_3'] * 100:.0f}% · "
             f"±5% 초과 {fc['p_move_gt_5'] * 100:.0f}% · "
             f"±10% 초과 {fc['p_move_gt_10'] * 100:.0f}%")
    L.append(f"   기대값(참고): {fc['center_pct']:+.2f}%, {fc['center']:,.0f}원 "
             f"— 상승·하락을 확률가중한 값이라 실제로 여기서 끝나는 날은 드뭅니다")
    L.append(f"   60% 구간: {fc['band60'][0]:,.0f} ~ {fc['band60'][1]:,.0f}원 / "
             f"80% 구간: {fc['band80'][0]:,.0f} ~ {fc['band80'][1]:,.0f}원")
    if fc.get("support"):
        line = f"   지지 {fc['support']:,.0f}원"
        if fc.get("resistance"):
            line += f" / 저항 {fc['resistance']:,.0f}원"
        L.append(line)
    L.append(f"   변동성: 일간 {fc['cur_vol_pct']:.1f}% / ATR {fc['atr_pct']:.1f}% "
             f"→ 권장 손절폭 -{fc['atr_pct'] * 2:.1f}%")
    if fc.get("divergence"):
        L.append(f"   ⚠ {fc['divergence']}")
    if fc["drivers"]:
        L.append("   근거:")
        for d in fc["drivers"][:5]:
            L.append(f"     · {d}")
    return "\n".join(L)


def format_summary(fcs: list[dict]) -> str:
    """전체 예측의 한 줄 요약 묶음. 기대 상승폭이 큰 순으로."""
    if not fcs:
        return "예측 없음"
    L = ["┌" + "─" * 76, "│ 한 줄 요약"]
    for fc in sorted(fcs, key=lambda x: -(x["p_up"] * x["up_pct"] + x["p_down"] * x["down_pct"])):
        mark = "▲" if fc["p_up"] >= 0.5 else "▼"
        L.append(f"│ {mark} {fc['headline']}")
    L.append("└" + "─" * 76)
    return "\n".join(L)
