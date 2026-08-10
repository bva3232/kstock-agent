"""DART OpenAPI 클라이언트.

발급: https://opendart.fss.or.kr  (무료, 분당 호출 제한 있음)
사용하는 엔드포인트
  - corpCode.xml        : 고유번호 <-> 종목코드 매핑 (zip)
  - list.json           : 공시 검색
  - company.json        : 기업 개황
  - fnlttSinglAcnt.json : 단일회사 주요계정 (재무제표 요약)
"""
from __future__ import annotations

import io
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd
import requests

BASE = "https://opendart.fss.or.kr/api"

STATUS_MSG = {
    "000": "정상",
    "010": "등록되지 않은 키",
    "011": "사용할 수 없는 키",
    "012": "접근할 수 없는 IP",
    "013": "조회된 데이터가 없음",
    "014": "파일이 존재하지 않음",
    "020": "요청 제한 초과",
    "021": "조회 가능한 회사 개수 초과",
    "100": "부적절한 필드값",
    "800": "시스템 점검 중",
    "900": "정의되지 않은 오류",
    "901": "사용자 계정의 개인정보 보유기간 만료",
}


class DartError(RuntimeError):
    pass


class DartClient:
    def __init__(self, api_key: str, cache_dir: Path | str = ".cache", min_interval: float = 0.12):
        if not api_key:
            raise DartError("DART API 키가 없습니다. config.yaml 또는 DART_API_KEY 환경변수를 설정하세요.")
        self.key = api_key
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self._min_interval = min_interval
        self._last_call = 0.0
        self._session = requests.Session()

    # ------------------------------------------------------------------ util
    def _throttle(self) -> None:
        gap = time.time() - self._last_call
        if gap < self._min_interval:
            time.sleep(self._min_interval - gap)
        self._last_call = time.time()

    def _get(self, endpoint: str, **params) -> dict:
        self._throttle()
        params["crtfc_key"] = self.key
        r = self._session.get(f"{BASE}/{endpoint}", params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        status = data.get("status")
        if status == "013":          # 데이터 없음은 정상 흐름으로 처리
            return {"status": "013", "list": []}
        if status != "000":
            raise DartError(f"DART {endpoint} 오류 {status}: {STATUS_MSG.get(status, data.get('message'))}")
        return data

    # ------------------------------------------------------------ corp codes
    def corp_codes(self, refresh_after_days: int = 7) -> pd.DataFrame:
        """전체 공시대상회사 고유번호 표. columns: corp_code, corp_name, stock_code, modify_date"""
        path = self.cache / "corp_codes.parquet"
        if path.exists():
            age = (pd.Timestamp.now() - pd.Timestamp(path.stat().st_mtime, unit="s")).days
            if age < refresh_after_days:
                return pd.read_parquet(path)

        self._throttle()
        r = self._session.get(f"{BASE}/corpCode.xml", params={"crtfc_key": self.key}, timeout=60)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            xml_bytes = zf.read(zf.namelist()[0])

        rows = []
        for el in ElementTree.fromstring(xml_bytes).iter("list"):
            rows.append({c.tag: (c.text or "").strip() for c in el})
        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False)
        return df

    def listed_map(self) -> pd.DataFrame:
        """상장사만 (stock_code 가 있는 행)."""
        df = self.corp_codes()
        return df[df["stock_code"].astype(str).str.len() == 6].reset_index(drop=True)

    # ------------------------------------------------------------ disclosures
    def list_disclosures(
        self,
        bgn_de: str | date,
        end_de: str | date | None = None,
        corp_code: str | None = None,
        corp_cls: str | None = None,   # Y=유가, K=코스닥, N=코넥스, E=기타
        pblntf_ty: str | None = None,  # A=정기, B=주요사항, C=발행, D=지분, ...
        max_pages: int = 20,
    ) -> pd.DataFrame:
        """기간 내 공시 목록 전체를 페이징으로 긁어온다."""
        bgn = _ymd(bgn_de)
        end = _ymd(end_de or bgn_de)
        frames, page = [], 1
        while page <= max_pages:
            params = dict(bgn_de=bgn, end_de=end, page_no=page, page_count=100)
            if corp_code:
                params["corp_code"] = corp_code
            if corp_cls:
                params["corp_cls"] = corp_cls
            if pblntf_ty:
                params["pblntf_ty"] = pblntf_ty
            data = self._get("list.json", **params)
            chunk = data.get("list") or []
            if not chunk:
                break
            frames.append(pd.DataFrame(chunk))
            if page >= int(data.get("total_page", 1)):
                break
            page += 1

        if not frames:
            return pd.DataFrame(
                columns=["corp_code", "corp_name", "stock_code", "corp_cls",
                         "report_nm", "rcept_no", "flr_nm", "rcept_dt", "rm"]
            )
        df = pd.concat(frames, ignore_index=True)
        df["url"] = "https://dart.fss.or.kr/dsaf001/main.do?rcpNo=" + df["rcept_no"]
        return df

    def recent_disclosures(self, days: int = 1, **kw) -> pd.DataFrame:
        end = date.today()
        return self.list_disclosures(end - timedelta(days=days), end, **kw)

    # -------------------------------------------------------------- financials
    def key_accounts(self, corp_code: str, year: int, reprt_code: str = "11011") -> pd.DataFrame:
        """단일회사 주요계정. reprt_code: 11011 사업 / 11012 반기 / 11013 1Q / 11014 3Q"""
        data = self._get(
            "fnlttSinglAcnt.json", corp_code=corp_code, bsns_year=str(year), reprt_code=reprt_code
        )
        return pd.DataFrame(data.get("list") or [])

    def company(self, corp_code: str) -> dict:
        data = self._get("company.json", corp_code=corp_code)
        data.pop("status", None)
        data.pop("message", None)
        return data


def get_dart(cfg):
    """DART 접근 백엔드를 고른다. API 키가 있으면 공식 API, 없으면 웹크롤러(DartCrawler)."""
    if cfg.dart_key:
        return DartClient(cfg.dart_key, cfg.cache_dir)
    from .dart_crawl import DartCrawler
    return DartCrawler(cfg.cache_dir)


def _ymd(d) -> str:
    if isinstance(d, str):
        return d.replace("-", "")
    return d.strftime("%Y%m%d")
