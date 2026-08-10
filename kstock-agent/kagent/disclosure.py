"""공시 분류·점수화 엔진.

report_nm(보고서명) 기반 규칙 분류기다. LLM 없이 결정론적으로 동작하므로
같은 입력이면 항상 같은 출력이 나온다 -- 기사에 쓰려면 이 재현성이 중요하다.

두 가지 산출물:
  1) score_events()   : 주가에 영향 줄 수 있는 이벤트의 방향/강도 (스크리너 입력)
  2) reporter_flags() : 기자 관점의 '취재 단서' 플래그 (올빼미 공시, 정정 반복 등)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

# (정규식, 카테고리, 방향 +호재/-악재/0중립, 가중치 0~1, 설명)
RULES: list[tuple[str, str, int, float, str]] = [
    # ---- 자본/주주환원 ---------------------------------------------------
    (r"자기주식.*(취득|신탁계약\s*체결)", "자사주매입", +1, 0.6, "유통물량 축소·주주환원 신호"),
    (r"자기주식.*(처분|신탁계약\s*해지)", "자사주처분", -1, 0.5, "물량 출회 가능성"),
    (r"현금.?현물배당결정", "배당", +1, 0.4, "배당 규모/성향 확인"),
    (r"무상증자", "무상증자", +1, 0.5, "권리락 이벤트, 단기 수급 왜곡 주의"),
    (r"유상증자", "유상증자", -1, 0.7, "지분 희석. 자금 사용목적과 할인율 확인"),
    (r"주식분할|액면분할", "액면분할", 0, 0.3, "유동성 변화"),
    (r"감자", "감자", -1, 0.7, "무상감자면 결손 보전 목적일 가능성"),
    (r"전환사채|신주인수권부사채|교환사채|CB|BW", "메자닌발행", -1, 0.6,
     "전환가액·리픽싱 조항이 향후 오버행. 인수자 명단 확인"),
    # ---- 실적/영업 -------------------------------------------------------
    (r"영업.?실적.*(잠정|공시)|매출액또는손익구조", "실적", 0, 0.8, "전년동기 대비 증감률이 핵심"),
    (r"단일판매.?공급계약", "수주", +1, 0.7, "계약금액/최근 매출액 대비 비중, 계약 상대방 확인"),
    (r"투자판단.*주요경영사항", "주요경영사항", 0, 0.5, "자율공시. 내용 직접 확인 필요"),
    (r"신규시설투자", "설비투자", +1, 0.4, "capex 규모와 자금조달 방식"),
    (r"기술이전|라이선스", "기술이전", +1, 0.6, "계약금(업프론트)과 마일스톤 구분"),
    # ---- 지배구조/지분 ---------------------------------------------------
    (r"최대주주.*변경", "최대주주변경", 0, 0.8, "인수 주체의 자금 출처가 취재 포인트"),
    (r"주식등의?대량보유상황보고", "5%보고", 0, 0.5, "보유목적(단순투자/경영참가) 변경 여부"),
    (r"임원.?주요주주.*소유상황보고", "내부자거래", 0, 0.4, "대표이사 본인 매도/매수 여부"),
    (r"타법인주식.*(취득|처분)", "지분거래", 0, 0.5, "인수가격 산정근거"),
    (r"합병|분할|영업양수|영업양도|주식교환", "구조재편", 0, 0.7, "합병비율과 반대주주 매수청구가"),
    # ---- 위험 신호 -------------------------------------------------------
    (r"횡령|배임", "횡령배임", -1, 1.0, "발생금액/자기자본 대비 비율. 즉시 취재 대상"),
    (r"소송.*(제기|판결)", "소송", -1, 0.4, "청구금액 규모"),
    (r"감사보고서.*(제출|미제출)", "감사보고서", 0, 0.6, "의견거절·한정이면 상장폐지 사유"),
    (r"관리종목|상장폐지|거래정지|불성실공시", "상장위험", -1, 1.0, "즉시 확인"),
    (r"채권자?.?회생|파산|부도|당좌거래정지", "부실", -1, 1.0, "최우선 확인"),
    (r"유상증자.*철회|계약.*(해지|해제)", "계약해지", -1, 0.8, "직전 호재 공시의 되돌림 여부"),
]

_COMPILED = [(re.compile(p), c, d, w, note) for p, c, d, w, note in RULES]
_CORRECTION = re.compile(r"\[?(기재|첨부)?정정\]?")


@dataclass
class Event:
    category: str
    direction: int
    weight: float
    note: str
    is_correction: bool


def classify(report_nm: str) -> Event | None:
    """보고서명 하나를 이벤트로 분류. 매칭 안 되면 None."""
    name = str(report_nm)
    is_corr = bool(_CORRECTION.search(name))
    for rx, cat, direction, weight, note in _COMPILED:
        if rx.search(name):
            # 정정공시는 원 공시의 방향을 약화시키고 신뢰도를 떨어뜨린다
            if is_corr:
                return Event(cat, direction, weight * 0.5, note + " (정정공시)", True)
            return Event(cat, direction, weight, note, False)
    return None


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """공시 DataFrame 에 category/direction/weight/note 컬럼을 붙인다."""
    if df.empty:
        return df.assign(category=None, direction=0, weight=0.0, note=None, is_correction=False)
    ev = df["report_nm"].map(classify)
    out = df.copy()
    out["category"] = [e.category if e else None for e in ev]
    out["direction"] = [e.direction if e else 0 for e in ev]
    out["weight"] = [e.weight if e else 0.0 for e in ev]
    out["note"] = [e.note if e else None for e in ev]
    out["is_correction"] = [e.is_correction if e else False for e in ev]
    return out


def score_events(df: pd.DataFrame, decay_days: int = 5) -> pd.DataFrame:
    """종목별 공시 이벤트 점수. 최근 공시일수록 크게 반영.

    반환: stock_code 인덱스, columns=[event_score, event_count, top_event]
    """
    if df.empty:
        return pd.DataFrame(columns=["event_score", "event_count", "top_event"])

    d = annotate(df)
    d = d[(d["direction"] != 0) & d["stock_code"].astype(str).str.len().eq(6)].copy()
    if d.empty:
        return pd.DataFrame(columns=["event_score", "event_count", "top_event"])

    d["rcept_dt"] = pd.to_datetime(d["rcept_dt"], format="%Y%m%d", errors="coerce")
    latest = d["rcept_dt"].max()
    age = (latest - d["rcept_dt"]).dt.days.clip(lower=0)
    d["decayed"] = d["direction"] * d["weight"] * (0.5 ** (age / max(decay_days, 1)))

    g = d.groupby("stock_code")
    out = pd.DataFrame({
        "event_score": g["decayed"].sum(),
        "event_count": g.size(),
        "top_event": g.apply(
            lambda x: x.loc[x["decayed"].abs().idxmax(), "category"], include_groups=False
        ),
    })
    return out


# --------------------------------------------------------------------------
# 기자용: 취재 단서 탐지
# --------------------------------------------------------------------------
def reporter_flags(df: pd.DataFrame, lookback_corrections: int = 3) -> pd.DataFrame:
    """공시 목록에서 '기사가 될 만한' 패턴을 뽑는다.

    탐지 항목
      - 심각도 높은 악재 공시 (횡령/배임, 상장위험, 부실)
      - 정정공시 반복 (같은 회사가 짧은 기간에 여러 건)
      - 직전 호재의 되돌림 (수주/투자 공시 후 계약해지)
      - 최대주주 변경
    """
    if df.empty:
        return pd.DataFrame()

    d = annotate(df)
    flags = []

    severe = d[d["category"].isin(["횡령배임", "상장위험", "부실", "감사보고서"])]
    for _, r in severe.iterrows():
        flags.append(_flag(r, "중대사항", "즉시 확인 대상 공시"))

    corr = d[d["is_correction"]]
    repeat = corr.groupby("corp_name").size()
    for name in repeat[repeat >= lookback_corrections].index:
        r = corr[corr["corp_name"] == name].iloc[0]
        flags.append(_flag(r, "정정반복", f"{int(repeat[name])}건의 정정공시. 최초 공시 신뢰도 점검"))

    for _, r in d[d["category"] == "계약해지"].iterrows():
        flags.append(_flag(r, "호재되돌림", "직전 수주/증자 공시와 대조 필요"))

    for _, r in d[d["category"] == "최대주주변경"].iterrows():
        flags.append(_flag(r, "지배구조", "인수 주체와 자금 출처 확인"))

    if not flags:
        return pd.DataFrame()
    out = pd.DataFrame(flags)
    order = {"중대사항": 0, "호재되돌림": 1, "정정반복": 2, "지배구조": 3}
    return out.sort_values("flag", key=lambda s: s.map(order)).reset_index(drop=True)


def _flag(row: pd.Series, flag: str, why: str) -> dict:
    return {
        "flag": flag,
        "corp_name": row.get("corp_name"),
        "stock_code": row.get("stock_code"),
        "report_nm": row.get("report_nm"),
        "rcept_dt": row.get("rcept_dt"),
        "why": why,
        "note": row.get("note"),
        "url": row.get("url"),
    }


def late_filings(df: pd.DataFrame) -> pd.DataFrame:
    """'올빼미 공시' 후보: 금요일 접수 건.

    주의 -- DART list.json 은 접수 '일자'만 주고 시각은 주지 않는다.
    분 단위 접수시각이 필요하면 rcept_no 로 원문 페이지를 별도 조회해야 한다.
    여기서는 요일 기준의 1차 필터만 제공한다.
    """
    if df.empty:
        return df
    d = annotate(df)
    dt = pd.to_datetime(d["rcept_dt"], format="%Y%m%d", errors="coerce")
    is_friday = dt.dt.dayofweek == 4
    is_negative = d["direction"] < 0
    return d[is_friday & is_negative].copy()
