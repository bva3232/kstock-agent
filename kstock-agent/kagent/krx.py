"""KRX 정보데이터시스템 접근 계층 (pykrx 래퍼).

pykrx 는 버전에 따라 함수명이 바뀌어 왔다(get_market_ohlcv_by_ticker ->
get_market_ohlcv 등). 여기서 이름을 한 번 흡수해두면 상위 코드가 안 깨진다.

전 종목 x 과거 N일 패널을 만들 때, 종목별로 호출하면 2,700번 요청이 나간다.
대신 '날짜별 전종목 스냅샷'을 N번 받아서 피벗한다 (N=90이면 90번).
받은 스냅샷은 날짜별로 캐싱하므로 다음날 실행 시 1건만 새로 받는다.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from pykrx import stock

KOSPI_INDEX = "1001"
KOSDAQ_INDEX = "2001"


def _resolve(*names):
    for n in names:
        fn = getattr(stock, n, None)
        if fn is not None:
            return fn
    raise AttributeError(f"pykrx 에서 {names} 중 어떤 함수도 찾을 수 없습니다. pykrx 를 업데이트하세요.")


_ohlcv_snapshot = _resolve("get_market_ohlcv_by_ticker", "get_market_ohlcv")
_fundamental_snapshot = _resolve("get_market_fundamental_by_ticker", "get_market_fundamental")
_cap_snapshot = _resolve("get_market_cap_by_ticker", "get_market_cap")
_index_ohlcv = _resolve("get_index_ohlcv_by_date", "get_index_ohlcv")
_ticker_name = stock.get_market_ticker_name

# 영업일 캘린더용 기준 종목. KOSPI 지수 API(MDCSTAT00301)는 KRX 로그인이 있어야
# 응답하는데, 개별 종목 시세(MDCSTAT01701)는 로그인 없이도 열려 있어 이걸로 대신한다.
_CALENDAR_TICKER = "005930"


def ymd(d) -> str:
    if isinstance(d, str):
        return d.replace("-", "")
    return d.strftime("%Y%m%d")


def trading_days(start, end) -> list[str]:
    """장이 열린 날 = 영업일 (삼성전자 시세가 존재하는 날로 판정)."""
    df = stock.get_market_ohlcv_by_date(ymd(start), ymd(end), _CALENDAR_TICKER)
    if df is None or df.empty:
        return []
    return [d.strftime("%Y%m%d") for d in df.index]


def _index_history_yf(start, end) -> pd.DataFrame:
    """yfinance 로 KOSPI(^KS11) 시세를 받아 KRX 지수 API와 같은 컬럼 모양으로 맞춘다."""
    try:
        import yfinance as yf
    except ImportError:
        return pd.DataFrame()
    s = pd.Timestamp(ymd(start))
    e = pd.Timestamp(ymd(end)) + timedelta(days=1)
    raw = yf.download("^KS11", start=s.strftime("%Y-%m-%d"), end=e.strftime("%Y-%m-%d"),
                       progress=False, auto_adjust=False)
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.rename(columns={"Open": "시가", "High": "고가", "Low": "저가",
                              "Close": "종가", "Volume": "거래량"})
    keep = [c for c in ("시가", "고가", "저가", "종가", "거래량") if c in df.columns]
    return df[keep]


def recent_trading_days(n: int, end=None) -> list[str]:
    end = end or date.today()
    start = end - timedelta(days=int(n * 2.2) + 15)   # 주말/공휴일 여유
    days = trading_days(start, end)
    return days[-n:]


class KrxData:
    def __init__(self, cache_dir: Path | str = ".cache"):
        self.cache = Path(cache_dir) / "krx"
        self.cache.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ snapshots
    def ohlcv(self, day: str, market: str = "ALL") -> pd.DataFrame:
        """특정 영업일의 전종목 OHLCV. 캐싱됨."""
        f = self.cache / f"ohlcv_{market}_{day}.parquet"
        if f.exists():
            return pd.read_parquet(f)
        df = _ohlcv_snapshot(day, market=market)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index().rename(columns={"티커": "ticker", df.columns[0]: "ticker"})
        df["ticker"] = df["ticker"].astype(str).str.zfill(6)
        df.to_parquet(f, index=False)
        return df

    def fundamental(self, day: str, market: str = "ALL") -> pd.DataFrame:
        """BPS/PER/PBR/EPS/DIV/DPS."""
        df = _fundamental_snapshot(day, market=market)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df.columns = ["ticker"] + list(df.columns[1:])
        df["ticker"] = df["ticker"].astype(str).str.zfill(6)
        return df

    def market_cap(self, day: str, market: str = "ALL") -> pd.DataFrame:
        df = _cap_snapshot(day, market=market)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df.columns = ["ticker"] + list(df.columns[1:])
        df["ticker"] = df["ticker"].astype(str).str.zfill(6)
        return df

    # ---------------------------------------------------------------- panel
    def panel(self, days: list[str], market: str = "ALL", field: str = "종가") -> pd.DataFrame:
        """index=날짜, columns=티커 인 와이드 패널."""
        series = {}
        for d in days:
            snap = self.ohlcv(d, market)
            if snap.empty or field not in snap.columns:
                continue
            series[d] = snap.set_index("ticker")[field]
        if not series:
            return pd.DataFrame()
        out = pd.DataFrame(series).T
        out.index = pd.to_datetime(out.index, format="%Y%m%d")
        return out.sort_index()

    def panels(self, days: list[str], market: str = "ALL",
               fields=("종가", "고가", "저가", "거래대금")) -> dict[str, pd.DataFrame]:
        """여러 필드를 한 번의 스냅샷 순회로 뽑는다."""
        acc: dict[str, dict] = {f: {} for f in fields}
        for d in days:
            snap = self.ohlcv(d, market)
            if snap.empty:
                continue
            s = snap.set_index("ticker")
            for f in fields:
                if f in s.columns:
                    acc[f][d] = s[f]
        out = {}
        for f, series in acc.items():
            if not series:
                continue
            df = pd.DataFrame(series).T
            df.index = pd.to_datetime(df.index, format="%Y%m%d")
            out[f] = df.sort_index()
        return out

    def panel_from_tickers(self, tickers: list[str], days: list[str],
                            fields=("종가", "고가", "저가", "거래대금")) -> dict[str, pd.DataFrame]:
        """전종목 스냅샷 API(로그인 필요)가 막혀 있을 때 쓰는 대안.

        날짜별 스냅샷 대신 종목별 개별 시세를 하나씩 받아 패널로 피벗한다.
        종목별 개별 조회는 로그인 없이 열려 있지만 종목 수만큼 요청이 나가므로
        전종목이 아니라 정해둔 소수 워치리스트에만 쓸 것 (kagent.watchlist 참고).
        """
        start, end = days[0], days[-1]
        acc: dict[str, dict] = {f: {} for f in fields}
        for tk in tickers:
            df = stock.get_market_ohlcv_by_date(start, end, tk)
            if df is None or df.empty:
                continue
            df = df[df["종가"] > 0]
            if "거래대금" not in df.columns and "거래량" in df.columns:
                df = df.assign(거래대금=df["거래량"] * df["종가"])  # 종목별 조회엔 없어 근사
            for f in fields:
                if f in df.columns:
                    acc[f][tk] = df[f]
        out = {}
        for f, series in acc.items():
            if not series:
                continue
            out[f] = pd.DataFrame(series).sort_index()
        return out

    # ------------------------------------------------------------- investors
    def net_purchases(self, start: str, end: str, market: str = "KOSPI",
                      investor: str = "외국인") -> pd.DataFrame:
        """기간 누적 투자자별 순매수. 실패하면 빈 DF (KRX 응답 스펙 변동에 대비)."""
        fn = getattr(stock, "get_market_net_purchases_of_equities", None)
        if fn is None:
            return pd.DataFrame()
        try:
            df = fn(start, end, market, investor)
        except Exception:
            return pd.DataFrame()
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df.columns = ["ticker"] + list(df.columns[1:])
        df["ticker"] = df["ticker"].astype(str).str.zfill(6)
        return df

    # ----------------------------------------------------------------- index
    def index_history(self, start, end, code: str = KOSPI_INDEX) -> pd.DataFrame:
        """KOSPI/KOSDAQ 지수 시세. KRX 지수 API는 로그인이 있어야 응답하므로,
        실패하면 yfinance(^KS11)로 대체한다(KOSPI 한정, 로그인 불필요)."""
        try:
            df = _index_ohlcv(ymd(start), ymd(end), code, name_display=False)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
        if code == KOSPI_INDEX:
            return _index_history_yf(start, end)
        return pd.DataFrame()

    @staticmethod
    def tickers(day: str, markets=("KOSPI", "KOSDAQ")) -> set[str]:
        out: set[str] = set()
        for m in markets:
            try:
                out |= {str(t).zfill(6) for t in stock.get_market_ticker_list(day, market=m)}
            except Exception:
                continue
        return out

    @staticmethod
    def name_of(ticker: str) -> str:
        try:
            return _ticker_name(ticker)
        except Exception:
            return ticker
