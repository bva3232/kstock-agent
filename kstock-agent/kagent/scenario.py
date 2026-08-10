"""시나리오 생성 — 점 예측 대신 조건부 분기를 만든다.

"내일 170만원까지 오른다"는 검증도 활용도 안 된다. 맞아도 운이고 틀려도 변명이 된다.
쓸 수 있는 형태는 이것이다.

    "다음날 변동폭의 80%는 -4.2% ~ +4.5% 안에 들어왔다(과거 1년 기준).
     아래로 148만원(거래 밀집대)이 1차 지지, 위로 162만원이 1차 저항.
     148만원 이탈 사례 N=23건 중 다음날에도 하락 지속이 61%.
     단, 오늘 밤 미국 반도체 실적 발표가 있어 위 통계는 무력화될 수 있다."

마지막 줄이 가장 중요하다. 차트에 없는 정보가 다음날 방향을 결정하는 경우가
대부분이고, 시스템은 그걸 '모른다'고 말할 수 있어야 한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .patterns import _atr, levels, structure


def next_day_range(df: pd.DataFrame, lookback: int = 250) -> dict:
    """과거 실측 분포로 다음날 변동폭 구간을 잡는다. 정규분포 가정을 쓰지 않는다.

    주가 수익률은 꼬리가 두껍다. 정규분포로 계산한 2시그마 구간은
    실제보다 좁게 나오고, 그 밖으로 나가는 날이 훨씬 자주 생긴다.
    """
    ret = df["종가"].pct_change().iloc[-lookback:].dropna() * 100
    c = float(df["종가"].iloc[-1])
    atrp = float(_atr(df).iloc[-1] / c * 100)

    q = {p: float(ret.quantile(p)) for p in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)}
    return {
        "close": c,
        "atr_pct": atrp,
        "quantiles_pct": q,
        "band_80_pct": (q[0.10], q[0.90]),
        "band_80_price": (round(c * (1 + q[0.10] / 100)), round(c * (1 + q[0.90] / 100))),
        "band_90_price": (round(c * (1 + q[0.05] / 100)), round(c * (1 + q[0.95] / 100))),
        "up_day_rate_pct": float((ret > 0).mean() * 100),
        "n_days": len(ret),
        "extreme_move_rate_pct": float((ret.abs() > 5).mean() * 100),
    }


def swing_breakout(df: pd.DataFrame, window: int = 20, direction: str = "up") -> dict:
    """N일 고점 돌파 / N일 저점 이탈의 다음날 지속률.

    절대 가격 레벨(예: '150만원 돌파')로 과거 통계를 내면 표본이 거의 안 잡힌다.
    몇 배씩 움직인 종목에서 같은 가격이 두 번 나오는 일이 없기 때문이다.
    'N일 고점 돌파'는 척도에 무관하므로 수백 건의 표본이 확보된다.
    """
    c = df["종가"]
    prior_hi = c.rolling(window, min_periods=window // 2).max().shift(1)
    prior_lo = c.rolling(window, min_periods=window // 2).min().shift(1)
    mask = (c > prior_hi) if direction == "up" else (c < prior_lo)

    fwd = (c.shift(-1) / c - 1) * 100
    sel = fwd[mask.fillna(False) & fwd.notna()]
    base = fwd.dropna()
    if len(sel) == 0:
        return {"n": 0, "direction": direction, "window": window}

    cont = (sel > 0).mean() if direction == "up" else (sel < 0).mean()
    base_cont = (base > 0).mean() if direction == "up" else (base < 0).mean()
    se = sel.std() / np.sqrt(len(sel))
    return {
        "n": int(len(sel)),
        "window": window,
        "direction": direction,
        "next_day_mean_pct": float(sel.mean()),
        "followthrough_rate_pct": float(cont * 100),
        "base_rate_pct": float(base_cont * 100),
        "edge_pp": float((cont - base_cont) * 100),
        "t_stat": float((sel.mean() - base.mean()) / se) if se > 0 else float("nan"),
        "reliable": bool(len(sel) >= 30),
    }


def breakout_followthrough(df: pd.DataFrame, level: float, direction: str = "up",
                           tol_pct: float = 0.5) -> dict:
    """특정 가격대를 돌파/이탈한 과거 사례의 다음날 결과.

    '저항선을 뚫으면 간다'는 통념이 이 종목에서 실제로 성립했는지 확인한다.
    """
    c = df["종가"]
    band = level * tol_pct / 100
    if direction == "up":
        mask = (c > level + band) & (c.shift(1) <= level + band)
    else:
        mask = (c < level - band) & (c.shift(1) >= level - band)

    fwd = (c.shift(-1) / c - 1) * 100
    sel = fwd[mask.fillna(False) & fwd.notna()]
    if len(sel) == 0:
        return {"n": 0, "level": level, "direction": direction}

    cont = (sel > 0).mean() if direction == "up" else (sel < 0).mean()
    return {
        "n": int(len(sel)),
        "level": round(level),
        "direction": direction,
        "next_day_mean_pct": float(sel.mean()),
        "followthrough_rate_pct": float(cont * 100),
        "reliable": bool(len(sel) >= 30),
    }


def build(df: pd.DataFrame, known_events: list[str] | None = None) -> dict:
    """구조 + 레벨 + 분포 + 분기 시나리오를 하나로 묶는다."""
    st = structure(df)
    lv = levels(df)
    rng = next_day_range(df)

    support = [x for x in lv if x["side"] == "지지"]
    resist = [x for x in lv if x["side"] == "저항"]
    s1 = support[0]["level"] if support else round(rng["band_80_price"][0])
    r1 = resist[0]["level"] if resist else round(rng["band_80_price"][1])

    return {
        "structure": st,
        "levels": lv,
        "range": rng,
        "scenarios": [
            {
                "name": "상방 시나리오",
                "trigger": f"{r1:,}원 (1차 저항) 돌파",
                "evidence": swing_breakout(df, 20, "up"),
            },
            {
                "name": "하방 시나리오",
                "trigger": f"{s1:,}원 (1차 지지) 이탈",
                "evidence": swing_breakout(df, 20, "down"),
            },
            {
                "name": "횡보 시나리오",
                "trigger": f"{s1:,} ~ {r1:,}원 사이 유지",
                "evidence": {"note": "과거 다음날 종가의 상당수가 이 구간 안에 머물렀습니다."},
            },
        ],
        "known_unknowns": known_events or [],
        "caveat": (
            "위 통계는 모두 차트에서 관측 가능한 정보만으로 계산한 것입니다. "
            "장 마감 후 해외 증시, 실적 발표, 정책 변화 등 차트에 없는 정보가 "
            "다음날 시가를 결정하는 경우가 많고, 그럴 때 위 분포는 무력화됩니다."
        ),
    }


def format_report(sc: dict) -> str:
    st, rng = sc["structure"], sc["range"]
    L = []
    L.append(f"[구조] {st['date']:%Y-%m-%d} 종가 {st['close']:,.0f}원 ({st['ret_1d_pct']:+.2f}%)")
    L.append(f"  20일선 대비 {st['vs_ma20_pct']:+.1f}% / 60일선 대비 {st['vs_ma60_pct']:+.1f}% "
             f"/ 정배열 {'예' if st['ma_aligned'] else '아니오'}")
    L.append(f"  52주 고점 대비 {st['from_52w_high_pct']:+.1f}% / 저점 대비 {st['from_52w_low_pct']:+.1f}%")
    L.append(f"  ATR {st['atr_pct']:.1f}% / 20일 연환산 변동성 {st['vol_ann_20d']:.0f}% "
             f"/ 거래량 20일 평균 대비 {st['vol_ratio_20d']:.1f}배")
    L.append(f"  연속 {abs(st['streak_days'])}일 {'상승' if st['streak_days'] > 0 else '하락'}")

    L.append(f"\n[다음날 변동폭 실측 분포] 과거 {rng['n_days']}일 기준")
    L.append(f"  80% 구간: {rng['band_80_pct'][0]:+.1f}% ~ {rng['band_80_pct'][1]:+.1f}%  "
             f"({rng['band_80_price'][0]:,} ~ {rng['band_80_price'][1]:,}원)")
    L.append(f"  90% 구간: {rng['band_90_price'][0]:,} ~ {rng['band_90_price'][1]:,}원")
    L.append(f"  상승 마감 비율 {rng['up_day_rate_pct']:.1f}% / ±5% 초과일 비율 {rng['extreme_move_rate_pct']:.1f}%")

    L.append("\n[거래 밀집 가격대]")
    for x in sc["levels"]:
        share = ("거래량 비중 %.1f%%" % x["volume_share_pct"]
                 if x["volume_share_pct"] == x["volume_share_pct"] else x.get("source", ""))
        L.append(f"  {x['side']} {x['level']:,}원 ({x['distance_pct']:+.1f}%) — {share}")

    L.append("\n[분기 시나리오]")
    for s in sc["scenarios"]:
        L.append(f"  · {s['name']}: {s['trigger']}")
        e = s["evidence"]
        if e.get("n"):
            tag = "" if e.get("reliable") else "  ※표본부족"
            base = e.get("base_rate_pct")
            line = (f"      과거 {e['window']}일 {'고점 돌파' if e['direction'] == 'up' else '저점 이탈'} "
                    f"{e['n']}회 → 다음날 같은 방향 {e['followthrough_rate_pct']:.0f}%")
            if base is not None:
                line += f" (아무 날이나 {base:.0f}%, 차이 {e['edge_pp']:+.1f}%p, t={e['t_stat']:.1f})"
            L.append(line + tag)
        elif e.get("note"):
            L.append(f"      {e['note']}")
        else:
            L.append("      과거 사례 없음 — 판단 근거 없음")

    if sc["known_unknowns"]:
        L.append("\n[차트에 없는 변수 — 이게 방향을 결정할 가능성이 높음]")
        for e in sc["known_unknowns"]:
            L.append(f"  · {e}")

    L.append(f"\n{sc['caveat']}")
    return "\n".join(L)
