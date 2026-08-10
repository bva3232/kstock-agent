"""아침 브리핑 HTML 리포트 생성."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from jinja2 import Template

TEMPLATE = Template("""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ date }} 시장 브리핑</title>
<style>
 :root{--bg:#fbfaf8;--fg:#1c1b19;--mut:#6b6862;--line:#e3e0d9;--up:#c0392b;--dn:#1f6fb2;--warn:#b8860b}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);
   font:15px/1.65 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Malgun Gothic",sans-serif}
 .wrap{max-width:940px;margin:0 auto;padding:32px 20px 64px}
 h1{font-size:26px;margin:0 0 4px;letter-spacing:-.02em}
 h2{font-size:17px;margin:38px 0 12px;padding-bottom:7px;border-bottom:1px solid var(--line)}
 .sub{color:var(--mut);font-size:13px;margin-bottom:22px}
 .regime{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0}
 .kpi{background:#fff;border:1px solid var(--line);border-radius:8px;padding:11px 15px;min-width:132px}
 .kpi b{display:block;font-size:19px;font-variant-numeric:tabular-nums}
 .kpi span{color:var(--mut);font-size:12px}
 .badge{display:inline-block;padding:3px 11px;border-radius:99px;font-size:12px;font-weight:600}
 .risk_on{background:#e8f4ea;color:#1e6b34} .risk_off{background:#fbe9e7;color:#a3291a}
 .neutral{background:#f0eee9;color:#5a574f}
 .card{background:#fff;border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin-bottom:12px}
 .card h3{margin:0 0 3px;font-size:16px} .card .meta{color:var(--mut);font-size:12.5px;margin-bottom:10px}
 ul.why{margin:0 0 12px;padding-left:18px} ul.why li{font-size:13.5px;margin:3px 0}
 .plan{background:#f7f5f1;border-radius:7px;padding:10px 13px;font-size:13px;
   display:grid;grid-template-columns:repeat(auto-fit,minmax(118px,1fr));gap:8px}
 .plan div span{display:block;color:var(--mut);font-size:11.5px}
 table{width:100%;border-collapse:collapse;font-size:13px} th,td{padding:7px 9px;text-align:left;
   border-bottom:1px solid var(--line)} th{color:var(--mut);font-weight:600;font-size:12px}
 td.num{text-align:right;font-variant-numeric:tabular-nums}
 .flag{border-left:3px solid var(--warn);padding-left:12px;margin:11px 0}
 .flag b{font-size:13.5px} .flag p{margin:2px 0;font-size:13px;color:var(--mut)}
 a{color:#2a5db0} .disc{margin-top:40px;padding:15px 17px;background:#f4f2ed;border-radius:8px;
   font-size:12.5px;color:var(--mut);line-height:1.7}
</style></head><body><div class="wrap">

<h1>{{ date }} 아침 시장 브리핑</h1>
<div class="sub">생성 {{ generated_at }} · 기준일 {{ base_date }} · 시드 {{ "{:,}".format(seed) }}원</div>

<h2>시장 국면</h2>
<div><span class="badge {{ regime.state }}">{{ regime_label }}</span></div>
<div class="regime">
  <div class="kpi"><span>KOSPI</span><b>{{ "%.2f"|format(regime.index_close) }}</b></div>
  <div class="kpi"><span>20일선 대비</span><b>{{ "%+.2f"|format(regime.vs_ma20_pct) }}%</b></div>
  <div class="kpi"><span>5일 수익률</span><b>{{ "%+.2f"|format(regime.ret_5d_pct) }}%</b></div>
  <div class="kpi"><span>20일 수익률</span><b>{{ "%+.2f"|format(regime.ret_20d_pct) }}%</b></div>
  {% if regime.breadth_above_ma20_pct == regime.breadth_above_ma20_pct %}
  <div class="kpi"><span>20일선 위 종목</span><b>{{ "%.0f"|format(regime.breadth_above_ma20_pct) }}%</b></div>
  {% endif %}
</div>
<p style="font-size:13.5px;color:var(--mut)">{{ regime_comment }}</p>

<h2>관심 종목 {{ candidates|length }}선 <span style="font-weight:400;color:var(--mut);font-size:13px">— 매수 추천이 아니라 확인 대상 목록입니다</span></h2>
{% for c in candidates %}
<div class="card">
  <h3>{{ c.name }} <span style="color:var(--mut);font-weight:400;font-size:13px">{{ c.ticker }}</span></h3>
  <div class="meta">점수 {{ "%.1f"|format(c.score) }} · 현재가 {{ "{:,}".format(c.price|int) }}원
    · 전일 {{ "%+.2f"|format(c.ret_1d) }}% · 20일 {{ "%+.1f"|format(c.ret_20d) }}%
    {% if c.PBR %}· PBR {{ "%.2f"|format(c.PBR) }}{% endif %}</div>
  <ul class="why">{% for w in c.why %}<li>{{ w }}</li>{% endfor %}</ul>
  <div class="plan">
    <div><span>수량 상한</span>{{ c.plan.qty }}주</div>
    <div><span>투입금액</span>{{ "{:,}".format(c.plan.notional|int) }}원</div>
    <div><span>손절가</span>{{ "{:,}".format(c.plan.stop_price|int) }}원 ({{ "-%.1f"|format(c.plan.stop_pct) }}%)</div>
    <div><span>감내 손실</span>{{ "{:,}".format(c.plan.risk_krw|int) }}원</div>
    <div><span>ATR</span>{{ "%.1f"|format(c.atr_pct) }}%</div>
  </div>
  {% if c.disclosures %}
  <div style="margin-top:10px;font-size:13px">
    <b style="font-size:12.5px;color:var(--mut)">관련 공시</b>
    {% for d in c.disclosures %}<div>· <a href="{{ d.url }}" target="_blank">{{ d.report_nm }}</a>
      <span style="color:var(--mut)">{{ d.rcept_dt }}</span></div>{% endfor %}
  </div>{% endif %}
</div>
{% endfor %}

{% if flags is not none and flags|length %}
<h2>공시 이상징후 (취재 단서)</h2>
{% for f in flags %}
<div class="flag">
  <b>[{{ f.flag }}] {{ f.corp_name }}</b>
  <p>{{ f.report_nm }} · {{ f.rcept_dt }}</p>
  <p>{{ f.why }}{% if f.note %} — {{ f.note }}{% endif %}
    {% if f.url %}· <a href="{{ f.url }}" target="_blank">원문</a>{% endif %}</p>
</div>
{% endfor %}
{% endif %}

{% if disclosure_summary is not none and disclosure_summary|length %}
<h2>전일 공시 유형별 집계</h2>
<table><thead><tr><th>유형</th><th class="num">건수</th><th>방향</th></tr></thead><tbody>
{% for r in disclosure_summary %}
<tr><td>{{ r.category }}</td><td class="num">{{ r.count }}</td>
    <td>{{ "호재성" if r.direction > 0 else ("악재성" if r.direction < 0 else "중립") }}</td></tr>
{% endfor %}</tbody></table>
{% endif %}

<div class="disc">
<b>읽는 방법</b><br>
이 리포트는 DART 공시와 KRX 시세를 규칙 기반으로 집계한 <b>자료 정리물</b>입니다.
점수는 미래 수익률의 예측치가 아니라, 관찰된 조건들을 하나의 축으로 줄 세운 순위입니다.
'관심 종목'은 확인해볼 대상이지 매수 신호가 아닙니다.<br><br>
{{ "표시된 수량·손절가는 계좌 리스크 한도(1회 %.0f%%)에서 역산한 상한선이며, 그만큼 사라는 뜻이 아닙니다."|format(risk_pct) }}
왕복 거래비용 약 {{ "%.2f"|format(cost_pct) }}%를 넘겨야 손익분기입니다.<br><br>
투자 판단과 그 결과는 전적으로 이용자 본인의 책임입니다. 이 도구는 투자자문업 등록 없이
개인이 자기 판단을 돕기 위해 만든 것이며, 타인에게 이 결과를 투자 권유로 제공하면
자본시장법상 문제가 될 수 있습니다.
</div>
</div></body></html>""")

REGIME_COMMENT = {
    "risk_on": "지수가 주요 이동평균 위에 있고 시장 폭도 양호합니다. 다만 추세 후반부일수록 되돌림 폭이 커집니다.",
    "neutral": "방향성이 뚜렷하지 않은 구간입니다. 개별 종목 신호의 신뢰도가 떨어지므로 비중을 줄이는 편이 무난합니다.",
    "risk_off": "지수가 이동평균 아래에 있습니다. 이런 국면에서는 개별 종목이 아무리 좋아 보여도 함께 하락하는 경우가 많습니다. 신규 진입을 미루는 것을 우선 고려하세요.",
}
REGIME_LABEL = {"risk_on": "위험선호", "neutral": "중립", "risk_off": "위험회피"}


def render(
    path: Path | str,
    regime: dict,
    candidates: list[dict],
    flags: pd.DataFrame | None,
    disclosure_summary: pd.DataFrame | None,
    account: dict,
    base_date: str,
) -> Path:
    html = TEMPLATE.render(
        date=datetime.now().strftime("%Y년 %m월 %d일"),
        generated_at=datetime.now().strftime("%H:%M"),
        base_date=base_date,
        seed=account["seed_krw"],
        risk_pct=account["risk_per_trade"] * 100,
        cost_pct=account.get("roundtrip_cost_pct", 0.3),
        regime=regime,
        regime_label=REGIME_LABEL.get(regime["state"], regime["state"]),
        regime_comment=REGIME_COMMENT.get(regime["state"], ""),
        candidates=candidates,
        flags=None if flags is None or flags.empty else flags.to_dict("records"),
        disclosure_summary=None if disclosure_summary is None or disclosure_summary.empty
        else disclosure_summary.to_dict("records"),
    )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")
    return p
