"""신호 검증. 이 모듈이 이 프로젝트에서 가장 중요하다.

리포트가 매일 그럴듯한 종목을 뱉어내는 것과, 그 신호에 실제로 엣지가 있는 것은
전혀 다른 문제다. 여기서는 과거 데이터로 다음을 측정한다.

  - IC (Information Coefficient) : 점수와 미래 수익률의 순위상관.
        0.03 정도면 '약하지만 있음', 0에 붙어 있으면 신호가 무의미하다는 뜻.
  - 분위별 평균 수익률 : 상위 분위가 하위 분위보다 꾸준히 나은가.
  - 승률과 거래비용 차감 후 기대값.

주의: 가격/거래대금 신호만 검증한다. 공시 이벤트 점수는 과거 DART 데이터를
따로 수집해야 하므로 기본 백테스트에서는 제외된다(=실제 성과와 다를 수 있음).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import signals as sg


def _score_at(panels: dict[str, pd.DataFrame], upto: int) -> pd.Series:
    """upto 행까지의 데이터만 사용해 점수를 계산 (look-ahead 방지)."""
    close = panels["종가"].iloc[:upto]
    high = panels.get("고가", close).iloc[:upto]
    low = panels.get("저가", close).iloc[:upto]
    value = panels.get("거래대금", close * 0).iloc[:upto]

    trend = sg.trend_alignment(close)
    mom = sg.momentum(close, 20)
    surge = sg.turnover_surge(value).clip(upper=10)
    vol = sg.realized_vol(close)
    turn = sg.avg_turnover(value)

    liquid = turn >= turn.quantile(0.5)      # 하위 유동성 제거
    score = (
        0.30 * sg.pct_rank(trend)
        + 0.25 * sg.pct_rank(mom)
        + 0.25 * sg.pct_rank(surge)
        - 0.20 * sg.pct_rank(vol)
    )
    return score.where(liquid)


def run(
    panels: dict[str, pd.DataFrame],
    horizon: int = 5,
    step: int = 5,
    min_history: int = 60,
    n_buckets: int = 5,
    cost_pct: float = 0.30,
    lag: int = 1,
) -> dict:
    """lag=1 이 기본. 신호는 t-1 종가까지로 계산하고 진입은 t 종가로 잡는다.

    lag=0 으로 두면 '신호를 만든 그 종가에 매수'가 되는데, 장 마감 후에 계산해서
    마감가로 사는 건 불가능하다. 이 한 칸 차이로 백테스트 성과가 크게 부풀려진다.
    """
    close = panels["종가"]
    n = len(close)
    need = min_history + horizon + step + lag
    if n < need:
        raise ValueError(f"데이터가 부족합니다. 최소 {need}영업일 필요, 현재 {n}일")

    ics, records = [], []
    for t in range(min_history, n - horizon - lag, step):
        score = _score_at(panels, t)                    # t-1 종가까지만 사용
        entry = close.iloc[t - 1 + lag]                 # 진입
        exit_ = close.iloc[t - 1 + lag + horizon]       # 청산
        fwd = (exit_ / entry - 1) * 100
        df = pd.concat([score.rename("score"), fwd.rename("fwd")], axis=1).dropna()
        if len(df) < 50:
            continue
        ics.append(df["score"].corr(df["fwd"], method="spearman"))
        df["bucket"] = pd.qcut(df["score"].rank(method="first"), n_buckets, labels=False)
        df["date"] = close.index[t - 1]
        records.append(df)

    if not records:
        raise ValueError("검증 가능한 구간이 없습니다.")

    all_df = pd.concat(records)
    by_bucket = all_df.groupby("bucket")["fwd"].agg(["mean", "median", "std", "count"])
    by_bucket["win_rate"] = all_df.groupby("bucket")["fwd"].apply(lambda x: (x > 0).mean() * 100)

    top = all_df[all_df["bucket"] == n_buckets - 1]["fwd"]
    bottom = all_df[all_df["bucket"] == 0]["fwd"]
    ic = pd.Series(ics)

    return {
        "periods": len(records),
        "horizon_days": horizon,
        "ic_mean": float(ic.mean()),
        "ic_std": float(ic.std()),
        "ic_hit_rate": float((ic > 0).mean() * 100),
        "ic_t_stat": float(ic.mean() / (ic.std() / np.sqrt(len(ic)))) if ic.std() else np.nan,
        "top_bucket_mean_pct": float(top.mean()),
        "top_bucket_after_cost_pct": float(top.mean() - cost_pct),
        "top_bucket_win_rate_pct": float((top > 0).mean() * 100),
        "long_short_spread_pct": float(top.mean() - bottom.mean()),
        "by_bucket": by_bucket,
    }


def null_check(
    panels: dict[str, pd.DataFrame],
    n_sims: int = 20,
    seed: int = 0,
    **run_kw,
) -> dict:
    """귀무분포 점검 — 검증 코드 자체가 가짜 엣지를 만들어내지는 않는지 본다.

    실제 패널과 같은 모양·같은 변동성의 '순수 무작위' 가격을 만들어 같은 백테스트를
    돌린다. 무작위 데이터의 IC는 0 근처여야 정상이다. 여기서 0이 아니게 나오면
    실제 데이터의 IC도 그만큼은 신호가 아니라 코드 때문에 생긴 것이다.

    반환된 p_value 는 '실제 IC가 무작위 데이터에서 우연히 나올 확률'의 근사치다.
    """
    close = panels["종가"]
    rng = np.random.default_rng(seed)
    ret = close.pct_change()
    mu = float(np.nanmean(ret.values))
    sd = float(np.nanstd(ret.values))
    shape = close.shape

    real = run(panels, **run_kw)
    null_ics = []
    for _ in range(n_sims):
        r = rng.normal(mu, sd, shape)
        sim_close = pd.DataFrame(
            close.iloc[0].values * np.exp(np.cumsum(r, axis=0)),
            index=close.index, columns=close.columns,
        )
        sim = {
            "종가": sim_close,
            "고가": sim_close * (1 + sd),
            "저가": sim_close * (1 - sd),
            "거래대금": pd.DataFrame(
                rng.lognormal(20, 1.0, shape), index=close.index, columns=close.columns
            ),
        }
        try:
            null_ics.append(run(sim, **run_kw)["ic_mean"])
        except Exception:
            continue

    null = pd.Series(null_ics)
    return {
        "real_ic": real["ic_mean"],
        "null_ic_mean": float(null.mean()),
        "null_ic_std": float(null.std()),
        "null_ic_p95": float(null.quantile(0.95)),
        "p_value": float((null >= real["ic_mean"]).mean()),
        "n_sims": len(null),
        "real": real,
    }


def verdict(res: dict) -> str:
    """결과를 한 문장으로 해석. 자기기만 방지용."""
    ic, t, net = res["ic_mean"], res["ic_t_stat"], res["top_bucket_after_cost_pct"]
    if abs(t) < 2:
        return (f"IC {ic:.3f}, t={t:.1f} — 통계적으로 의미 있다고 보기 어렵습니다. "
                "이 신호로 실매매하면 사실상 무작위 매매에 수수료만 더 내는 셈입니다.")
    if net <= 0:
        return (f"IC {ic:.3f} (t={t:.1f}) 로 방향성은 있으나, 거래비용 차감 후 상위분위 "
                f"기대수익이 {net:.2f}%로 남는 게 없습니다. 보유기간을 늘려야 합니다.")
    return (f"IC {ic:.3f} (t={t:.1f}), 거래비용 차감 후 상위분위 평균 {net:.2f}%. "
            "약한 엣지가 관찰됩니다. 단, 과거 구간 한정이며 미래 보장이 아닙니다.")
