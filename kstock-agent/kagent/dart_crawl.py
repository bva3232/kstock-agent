"""DART 공시 웹크롤러 — OpenDART API 키가 없을 때의 대체 백엔드.

dart.fss.or.kr 의 공개 '최근공시(dsac001)' 화면을 크롤링한다. 로그인이나 API 키가
필요없는 페이지지만, 그만큼 화면 자체의 제약을 그대로 물려받는다.

    · 조회 가능 범위: 최근 영업일 약 5일치 (dsac001 UI 자체의 한계, mdayCnt 0~4)
    · 종목코드는 공시 목록에 없어서, DART 기업개황 팝업(dsae001)을 회사별로 1회씩
      추가 조회해 얻는다. 조회 결과는 .cache 에 누적 저장되어 다음 실행부터는 빠르다.

이 범위를 넘는 과거 이력이 필요하면 OpenDART API 키를 발급받아
kagent.dart.DartClient 를 쓰는 편이 정확하고 빠르다.
"""
from __future__ import annotations

import json
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://dart.fss.or.kr"
SEARCH_URL = f"{BASE}/dsac001/search.ax"
MAIN_ALL_URL = f"{BASE}/dsac001/mainAll.do"
POPUP_URL = f"{BASE}/dsae001/selectPopup.ax"

_MARKET_CLASS = {
    "tagCom_kospi": "Y",
    "tagCom_kosdaq": "K",
    "tagCom_konex": "N",
    "tagCom_etc": "E",
}

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

_COLUMNS = ["corp_code", "corp_name", "stock_code", "corp_cls",
            "report_nm", "rcept_no", "flr_nm", "rcept_dt", "rm", "url"]


class DartCrawler:
    """DartClient 와 같은 모양(recent_disclosures/list_disclosures)을 제공하는 크롤러."""

    MAX_NEW_LOOKUPS = 200   # 회당 신규 종목코드(팝업) 조회 상한 — 서버 차단 방지

    def __init__(self, cache_dir: Path | str = ".cache", min_interval: float = 0.35):
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self._min_interval = min_interval
        self._last_call = 0.0
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": _UA})
        retry = Retry(total=3, connect=3, read=3, backoff_factor=1.0,
                       status_forcelist=(500, 502, 503, 504),
                       allowed_methods=frozenset(["GET", "POST"]))
        adapter = HTTPAdapter(max_retries=retry)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)
        self._primed = False

        self._stock_map_path = self.cache / "dart_corp_stock_map.json"
        self._stock_map: dict[str, str] = {}
        if self._stock_map_path.exists():
            try:
                self._stock_map = json.loads(self._stock_map_path.read_text(encoding="utf-8"))
            except Exception:
                self._stock_map = {}

    # ------------------------------------------------------------------ util
    def _throttle(self) -> None:
        gap = time.time() - self._last_call
        if gap < self._min_interval:
            time.sleep(self._min_interval - gap)
        self._last_call = time.time()

    def _prime(self) -> None:
        if self._primed:
            return
        self._throttle()
        self._session.get(MAIN_ALL_URL, timeout=20)
        self._primed = True

    def _post_page(self, mday_cnt: int, page: int, attempts: int = 3) -> str:
        self._prime()
        data = {
            "currentPage": page, "maxResults": 100, "maxLinks": 10,
            "sort": "", "series": "", "pageGrouping": "",
            "mdayCnt": mday_cnt, "selectDate": "", "textCrpCik": "",
        }
        last_err = None
        for attempt in range(1, attempts + 1):
            self._throttle()
            try:
                r = self._session.post(
                    SEARCH_URL, data=data, timeout=20,
                    headers={"X-Requested-With": "XMLHttpRequest", "Referer": MAIN_ALL_URL},
                )
                r.raise_for_status()
                return r.text
            except requests.exceptions.RequestException as e:
                last_err = e
                time.sleep(2.0 * attempt)
        raise last_err

    @staticmethod
    def _parse_page(html: str) -> tuple[list[dict], int]:
        soup = BeautifulSoup(html, "lxml")
        total_el = soup.find(id="totalCnt")
        total = int(total_el["value"]) if total_el and str(total_el.get("value", "")).isdigit() else 0

        rows = []
        for tr in soup.select("table.tbList tbody tr"):
            tds = tr.find_all("td", recursive=False)
            if len(tds) < 6:
                continue

            a_corp = tds[1].find("a")
            corp_name = a_corp.get_text(strip=True) if a_corp else tds[1].get_text(strip=True)
            corp_code = None
            if a_corp:
                m = re.search(r"openCorpInfoNew\('(\d+)'", a_corp.get("href", ""))
                corp_code = m.group(1) if m else None
            span = tds[1].find("span", class_=re.compile(r"^tagCom_"))
            corp_cls = _MARKET_CLASS.get(span.get("class", [None])[0]) if span else None

            a_rpt = tds[2].find("a")
            report_nm = a_rpt.get_text(" ", strip=True) if a_rpt else tds[2].get_text(" ", strip=True)
            rcept_no = None
            if a_rpt:
                m = re.search(r"rcpNo=(\d+)", a_rpt.get("href", ""))
                rcept_no = m.group(1) if m else None
            if not rcept_no or not corp_code:
                continue

            rows.append({
                "corp_code": corp_code,
                "corp_name": corp_name,
                "corp_cls": corp_cls,
                "report_nm": " ".join(report_nm.split()),
                "rcept_no": rcept_no,
                "flr_nm": tds[3].get_text(strip=True),
                "rcept_dt": tds[4].get_text(strip=True).replace(".", ""),
                "rm": tds[5].get_text(strip=True),
            })
        return rows, total

    def _fetch_bucket(self, mday_cnt: int) -> list[dict]:
        html = self._post_page(mday_cnt, 1)
        rows, total = self._parse_page(html)
        page = 2
        while len(rows) < total and page <= 20:
            html = self._post_page(mday_cnt, page)
            more, _ = self._parse_page(html)
            if not more:
                break
            rows.extend(more)
            page += 1
        return rows

    # ------------------------------------------------------------ stock code
    def _stock_code(self, corp_code: str) -> str | None:
        if corp_code in self._stock_map:
            return self._stock_map[corp_code] or None
        code = ""
        try:
            self._throttle()
            r = self._session.get(POPUP_URL, params={"selectKey": corp_code}, timeout=15)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "lxml")
            for th in soup.find_all("th"):
                if "종목코드" in th.get_text():
                    tr = th.find_parent("tr")
                    td = tr.find("td") if tr else None
                    if td:
                        code = td.get_text(strip=True)
                    break
        except Exception:
            code = ""
        self._stock_map[corp_code] = code
        return code or None

    def _save_stock_map(self) -> None:
        try:
            self._stock_map_path.write_text(
                json.dumps(self._stock_map, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass

    # ----------------------------------------------------------------- public
    def listed_map(self) -> pd.DataFrame:
        """지금까지 크롤링하며 확인된 회사명 <-> 종목코드 표 (누적 캐시 기반, 전체 상장사는 아님)."""
        rows = [{"corp_code": k, "stock_code": v} for k, v in self._stock_map.items() if v]
        return pd.DataFrame(rows, columns=["corp_code", "stock_code"])

    def recent_disclosures(self, days: int = 1, **kw) -> pd.DataFrame:
        end = date.today()
        return self.list_disclosures(end - timedelta(days=days), end, **kw)

    def list_disclosures(
        self,
        bgn_de: str | date,
        end_de: str | date | None = None,
        corp_name: str | None = None,
        corp_cls: str | None = None,
        **_kw,
    ) -> pd.DataFrame:
        bgn = _ymd(bgn_de)
        end = _ymd(end_de or bgn_de)
        n_days = (_parse_ymd(end) - _parse_ymd(bgn)).days + 1
        n_buckets = min(max(n_days, 1), 5)
        if n_days > 5:
            print(f"  [DART 크롤러] API 키 없이는 최근 5영업일 안팎만 조회됩니다 "
                  f"({n_days}일 요청됨). 그 이전 데이터는 빠집니다.")

        seen: dict[str, dict] = {}
        failed = 0
        for m in range(n_buckets):
            try:
                for row in self._fetch_bucket(m):
                    seen[row["rcept_no"]] = row
            except requests.exceptions.RequestException as e:
                failed += 1
                print(f"  [DART 크롤러] {m}영업일 전 구간 조회 실패, 건너뜀: {e}")
        if failed == n_buckets and n_buckets > 0:
            raise ConnectionError("DART 웹크롤링 실패: dart.fss.or.kr 에 연결할 수 없습니다.")

        if not seen:
            return pd.DataFrame(columns=_COLUMNS)

        df = pd.DataFrame(seen.values())
        df = df[(df["rcept_dt"] >= bgn) & (df["rcept_dt"] <= end)].reset_index(drop=True)
        if corp_name:
            df = df[df["corp_name"].str.contains(corp_name, na=False)].reset_index(drop=True)
        if corp_cls:
            df = df[df["corp_cls"] == corp_cls].reset_index(drop=True)
        if df.empty:
            return pd.DataFrame(columns=_COLUMNS)

        codes = df["corp_code"].tolist()
        uncached = [c for c in dict.fromkeys(codes) if c not in self._stock_map]
        if len(uncached) > self.MAX_NEW_LOOKUPS:
            print(f"  [DART 크롤러] 신규 종목코드 조회 대상이 {len(uncached)}건이라 "
                  f"서버 보호를 위해 {self.MAX_NEW_LOOKUPS}건까지만 조회합니다 "
                  f"(나머지는 캐시에 없어 stock_code 가 비어있을 수 있음, 다음 실행부터 누적됨).")
        budget = self.MAX_NEW_LOOKUPS
        stock_codes = []
        for c in codes:
            if c in self._stock_map:
                stock_codes.append(self._stock_map[c] or None)
            elif budget > 0:
                stock_codes.append(self._stock_code(c))
                budget -= 1
            else:
                stock_codes.append(None)
        df["stock_code"] = stock_codes
        self._save_stock_map()
        df["url"] = "https://dart.fss.or.kr/dsaf001/main.do?rcpNo=" + df["rcept_no"]
        return df[_COLUMNS]


def _ymd(d) -> str:
    if isinstance(d, str):
        return d.replace("-", "")
    return d.strftime("%Y%m%d")


def _parse_ymd(s: str):
    return datetime.strptime(s, "%Y%m%d").date()
