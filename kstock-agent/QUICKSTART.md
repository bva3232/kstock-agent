# 빠른 시작 (5분)

## 1. 준비

```bash
cd kstock-agent
pip install -r requirements.txt
cp config.example.yaml config.yaml
```

## 2. DART API 키 발급

https://opendart.fss.or.kr 접속 → 인증키 신청 → 이메일로 즉시 발급 (무료)

`config.yaml` 을 열어 `api_key` 자리에 붙여넣기.

## 3. 첫 실행 — 공시 레이더부터

시세 데이터가 필요 없어서 가장 빨리 결과를 봅니다.

```bash
python run_reporter_radar.py --days 3
```

## 4. 개별 종목 분석

```bash
python run_stock_analysis.py --ticker 000660   # SK하이닉스
python run_stock_analysis.py --ticker 005930   # 삼성전자
```

6년치 일봉을 받아 셋업별 조건부 통계를 냅니다. 첫 실행은 1~2분 걸립니다.

## 5. 일일 사이클 시작

```bash
python run_evening.py       # 오늘 장 마감 후 (15:40)
python run_premarket.py     # 내일 아침 (07:30)
python run_evening.py       # 내일 장 마감 후 → 첫 채점 결과가 나옵니다
```

`run_evening.py` 첫 실행은 전종목 90영업일치를 받느라 5~10분 걸립니다.
`.cache/` 에 저장되므로 다음날부터는 1일치만 추가로 받습니다.

빠르게 확인하고 싶으면 스크리닝을 건너뛰세요.

```bash
python run_evening.py --skip-p1     # 핵심 2종목만
```

## 6. 자동화

```bash
mkdir -p logs
crontab -e
```

```
40 15 * * 1-5 cd /절대경로/kstock-agent && /usr/bin/python3 run_evening.py   >> logs/evening.log 2>&1
30  7 * * 1-5 cd /절대경로/kstock-agent && /usr/bin/python3 run_premarket.py >> logs/premarket.log 2>&1
```

---

## 실전 투입 전 체크리스트

- [ ] `python run_backtest.py --days 400 --null-check 20` 실행 → p-value 0.10 이하인가
- [ ] 예측 기록 20건 이상 누적 → `run_evening.py` 출력의 성적표 확인
- [ ] 80% 구간 커버리지가 65~95% 안에 들어오는가 (밴드가 정직한가)
- [ ] 방향 적중률 z값이 2 이상인가 (아니면 동전던지기)
- [ ] 손절폭을 리포트의 `권장 손절폭` 이상으로 잡았는가
- [ ] 관심 종목이 시드머니로 1주 이상 매수 가능한 가격인가

여섯 개 중 하나라도 아니면 모의 기록을 더 쌓는 편이 낫습니다.

## 자주 겪는 문제

**`pykrx` 설치 후에도 데이터가 안 옴**
KRX 서버가 간헐적으로 응답하지 않습니다. 잠시 후 재시도하세요.

**`DART API 오류 020`**
분당 호출 한도 초과입니다. `dart.py` 의 `min_interval` 을 0.2 이상으로 올리세요.

**필터 통과 종목이 0개**
`config.yaml` 의 `min_market_cap_krw`, `min_avg_turnover_krw` 를 낮추세요.

**첫 실행이 너무 느림**
`config.yaml` 의 `screen.lookback_days` 를 90 → 60 으로 줄이면 빨라집니다.
단 60일선 계산이 부정확해집니다.
