"""예측 원장 — 모든 예측을 기록하고 다음날 채점한다.

이 모듈이 프로세스의 심장이다. 예측을 남기지 않으면 시스템이 맞는지 틀리는지
영원히 알 수 없고, 기억에만 의존하면 맞춘 것만 기억에 남는다(생존 편향).

채점 항목 세 가지
  - 방향 적중률   : 상승/하락을 맞췄나
  - 단가 오차     : 예측 중심값과 실제 종가의 차이 (MAE)
  - 구간 커버리지 : 60% 구간이라고 한 곳에 실제로 60%가 들어왔나 (캘리브레이션)

셋 중 가장 중요한 건 커버리지다. 방향 적중률 55%는 운으로도 나오지만,
"80% 구간"에 실제로 80%가 들어오는지는 시스템이 자기 불확실성을
정직하게 표현하고 있는지를 직접 보여준다.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT NOT NULL,
    process       TEXT NOT NULL,          -- 'P1'(단타 top5) | 'P2'(핵심종목)
    stage         TEXT NOT NULL,          -- 'evening'(마감후) | 'premarket'(07:30)
    target_date   TEXT NOT NULL,          -- 예측 대상 영업일 YYYYMMDD
    ticker        TEXT NOT NULL,
    name          TEXT,
    base_close    REAL NOT NULL,          -- 예측 시점 기준가
    p_up          REAL NOT NULL,          -- 상승 확률
    center        REAL NOT NULL,          -- 예상 종가 중심값
    lo60 REAL, hi60 REAL,
    lo80 REAL, hi80 REAL,
    support REAL, resistance REAL,
    drivers       TEXT,                   -- JSON list
    revised_from  INTEGER,                -- 07:30 수정 시 원 예측 id
    -- 채점 결과
    actual_close  REAL,
    actual_open   REAL,
    settled_at    TEXT,
    dir_hit       INTEGER,
    err_pct       REAL,
    in60          INTEGER,
    in80          INTEGER,
    UNIQUE(process, stage, target_date, ticker)
);
CREATE INDEX IF NOT EXISTS idx_target ON predictions(target_date);
"""


class Ledger:
    def __init__(self, path: Path | str = ".cache/predictions.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as con:
            con.executescript(SCHEMA)

    # ------------------------------------------------------------------ 기록
    def record(self, fc: dict, process: str, stage: str, target_date: str,
               revised_from: int | None = None) -> int:
        row = (
            datetime.now().isoformat(timespec="seconds"), process, stage, target_date,
            fc["ticker"], fc.get("name"), float(fc["base_close"]), float(fc["p_up"]),
            float(fc["center"]), float(fc["band60"][0]), float(fc["band60"][1]),
            float(fc["band80"][0]), float(fc["band80"][1]),
            float(fc.get("support") or np.nan), float(fc.get("resistance") or np.nan),
            json.dumps(fc.get("drivers", []), ensure_ascii=False), revised_from,
        )
        with closing(sqlite3.connect(self.path)) as con:
            cur = con.execute(
                """INSERT INTO predictions
                   (created_at,process,stage,target_date,ticker,name,base_close,p_up,center,
                    lo60,hi60,lo80,hi80,support,resistance,drivers,revised_from)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(process,stage,target_date,ticker) DO UPDATE SET
                     created_at=excluded.created_at, base_close=excluded.base_close,
                     p_up=excluded.p_up, center=excluded.center,
                     lo60=excluded.lo60, hi60=excluded.hi60,
                     lo80=excluded.lo80, hi80=excluded.hi80,
                     support=excluded.support, resistance=excluded.resistance,
                     drivers=excluded.drivers, revised_from=excluded.revised_from""",
                row,
            )
            con.commit()
            return int(cur.lastrowid)

    def pending(self, target_date: str | None = None) -> pd.DataFrame:
        q = "SELECT * FROM predictions WHERE actual_close IS NULL"
        params: tuple = ()
        if target_date:
            q += " AND target_date = ?"
            params = (target_date,)
        with closing(sqlite3.connect(self.path)) as con:
            return pd.read_sql_query(q, con, params=params)

    def get(self, process: str, stage: str, target_date: str) -> pd.DataFrame:
        with closing(sqlite3.connect(self.path)) as con:
            return pd.read_sql_query(
                "SELECT * FROM predictions WHERE process=? AND stage=? AND target_date=?",
                con, params=(process, stage, target_date),
            )

    # ------------------------------------------------------------------ 채점
    def settle(self, target_date: str, prices: dict[str, dict]) -> pd.DataFrame:
        """prices: {ticker: {'close': x, 'open': y}}"""
        pend = self.pending(target_date)
        if pend.empty:
            return pend

        rows = []
        with closing(sqlite3.connect(self.path)) as con:
            for _, p in pend.iterrows():
                px = prices.get(str(p["ticker"]).zfill(6))
                if not px:
                    continue
                close, opn = float(px["close"]), float(px.get("open", np.nan))
                actual_up = close > p["base_close"]
                # 예측 방향은 '예상 중심값이 전일 종가보다 위인가'로 판정한다.
                # p_up 은 갭 반영 시 시가 기준이라 그대로 쓰면 갭 하락일에 방향이 뒤집힌다.
                pred_up = p["center"] > p["base_close"]
                dir_hit = int(actual_up == pred_up)
                err = (close - p["center"]) / p["base_close"] * 100
                in60 = int(p["lo60"] <= close <= p["hi60"])
                in80 = int(p["lo80"] <= close <= p["hi80"])
                con.execute(
                    """UPDATE predictions SET actual_close=?, actual_open=?, settled_at=?,
                       dir_hit=?, err_pct=?, in60=?, in80=? WHERE id=?""",
                    (close, opn, datetime.now().isoformat(timespec="seconds"),
                     dir_hit, err, in60, in80, int(p["id"])),
                )
                rows.append({
                    "process": p["process"], "stage": p["stage"], "ticker": p["ticker"],
                    "name": p["name"], "base": p["base_close"], "예측중심": p["center"],
                    "실제종가": close,
                    "예측방향": "상승" if pred_up else "하락",
                    "실제방향": "상승" if actual_up else "하락",
                    "방향적중": "O" if dir_hit else "X",
                    "오차%": round(err, 2),
                    "60%구간": "O" if in60 else "X",
                    "80%구간": "O" if in80 else "X",
                    "실제등락%": round((close / p["base_close"] - 1) * 100, 2),
                })
            con.commit()
        return pd.DataFrame(rows)

    # ------------------------------------------------------------ 성과 집계
    def scorecard(self, process: str | None = None, stage: str | None = None,
                  last_n_days: int | None = None) -> dict:
        q = "SELECT * FROM predictions WHERE actual_close IS NOT NULL"
        params = []
        if process:
            q += " AND process=?"
            params.append(process)
        if stage:
            q += " AND stage=?"
            params.append(stage)
        with closing(sqlite3.connect(self.path)) as con:
            df = pd.read_sql_query(q, con, params=params)
        if df.empty:
            return {"n": 0}

        if last_n_days:
            keep = sorted(df["target_date"].unique())[-last_n_days:]
            df = df[df["target_date"].isin(keep)]

        n = len(df)
        hit = df["dir_hit"].mean() * 100
        se = np.sqrt(0.25 / n) * 100          # 50% 가정 하의 표준오차
        z = (hit - 50) / se if se else np.nan
        return {
            "n": n,
            "days": int(df["target_date"].nunique()),
            "dir_hit_pct": float(hit),
            "dir_hit_se_pct": float(se),
            "z_vs_coinflip": float(z),
            "mae_pct": float(df["err_pct"].abs().mean()),
            "bias_pct": float(df["err_pct"].mean()),
            "cover60_pct": float(df["in60"].mean() * 100),
            "cover80_pct": float(df["in80"].mean() * 100),
            "actual_abs_move_pct": float(
                ((df["actual_close"] / df["base_close"] - 1) * 100).abs().mean()),
        }

    def verdict(self, sc: dict) -> str:
        if sc.get("n", 0) < 20:
            return (f"표본 {sc.get('n', 0)}건 — 아직 판단 불가. "
                    "최소 20건, 가급적 50건 이상 쌓인 뒤에 보세요.")
        L = [f"표본 {sc['n']}건 / {sc['days']}일"]
        if abs(sc["z_vs_coinflip"]) < 2:
            L.append(f"방향 적중률 {sc['dir_hit_pct']:.1f}% (z={sc['z_vs_coinflip']:+.1f}) "
                     "— 동전던지기와 구별되지 않습니다.")
        else:
            L.append(f"방향 적중률 {sc['dir_hit_pct']:.1f}% (z={sc['z_vs_coinflip']:+.1f}) "
                     "— 우연으로 보기 어려운 수준입니다.")
        L.append(f"단가 평균오차 {sc['mae_pct']:.2f}%, 편향 {sc['bias_pct']:+.2f}% "
                 f"(실제 평균 변동폭 {sc['actual_abs_move_pct']:.2f}%)")
        if sc["mae_pct"] >= sc["actual_abs_move_pct"]:
            L.append("  → 오차가 실제 변동폭만큼 큽니다. 단가 예측은 '전일 종가 그대로'와 다를 바 없습니다.")
        c60, c80 = sc["cover60_pct"], sc["cover80_pct"]
        L.append(f"구간 커버리지 60%→{c60:.0f}% / 80%→{c80:.0f}%")
        if c80 < 65:
            L.append("  → 구간이 너무 좁습니다. 변동성을 과소평가하고 있으니 밴드를 넓히세요.")
        elif c80 > 95:
            L.append("  → 구간이 너무 넓어 실용성이 없습니다.")
        else:
            L.append("  → 불확실성 표현은 대체로 정직합니다.")
        return "\n".join(L)
