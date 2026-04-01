# Stock Advisor - LangGraph 기반 주식 분석 시스템

## 프로젝트 개요

LangGraph로 5개 병렬 에이전트(Price, Fundamental, Disclosure, News, Macro)를 실행하여
종합 매매 시그널(STRONG_BUY ~ STRONG_SELL)을 생성하는 주식 분석 시스템.

- **LLM**: gpt-4o-mini (비용 절감)
- **분석 1회당 LLM 호출**: 6회 (5 에이전트 + 1 Synthesizer)
- **언어**: Python
- **프레임워크**: LangGraph

## 아키텍처 규칙

### State 설계 (절대 위반 금지)

- 에이전트 결과는 반드시 `agent_results: Annotated[dict, merge_dicts]` 하나에 병합
- **개별 dict 필드(price_data, fundamental_data 등) 사용 금지** — 병렬 실행 시 마지막 write만 살아남음
- 리스트 필드(error_log, risk_factors)는 반드시 `Annotated[list, append_list]` reducer 사용

### 에이전트 공통 패턴 (3단계)

```
Fetch (API tool 호출, LLM 사용 금지)
  → Analyze (LLM JSON only 응답)
  → Score (Rule-based 또는 LLM-scored, 0.0~1.0)
  → return {"agent_results": {"에이전트키": {data + score}}}
```

- **Fetch 단계에서 LLM 호출 절대 금지**
- **Score는 가능하면 Rule-based** (Disclosure만 예외적으로 LLM 스코어)
- 모든 LLM 응답은 JSON only — free-form text 금지
- JSON 파싱 시 ```json 펜스 제거 포함

### 외부 API 호출 규칙

- **모든 외부 API 호출은 반드시 `cached_api_call()`을 경유**
- 직접 API 호출 금지
- `USE_MOCK=true` 시 fixture 데이터 리턴 (개발/테스트용)
- API 실패 시 만료된 캐시라도 리턴 (Graceful Degradation)
- 캐시도 없으면 error_log 기록 + score 0.0 (크래시 금지)

### 캐시 TTL 정책

| 데이터 | TTL |
|---|---|
| 시세 (OHLCV) | 30분 |
| 뉴스 | 2시간 |
| 공시 | 6시간 |
| 재무제표 | 24시간 |
| 거시지표 (VIX, 지수) | 1시간 |
| 거시지표 (금리, 환율) | 12시간 |

### 스코어 & 시그널

- 모든 스코어는 `validate_score()`로 0.0~1.0 범위 강제
- 기간별 가중치 테이블은 `config/settings.py`의 `WEIGHT_TABLE` 참조
- 모든 기간에 동일 가중치 적용 금지 — short/mid/long 구분 필수

### 시그널 기준

- 0.80+ → STRONG_BUY
- 0.65~0.79 → BUY
- 0.45~0.64 → HOLD
- 0.30~0.44 → SELL
- <0.30 → STRONG_SELL

## 금지 목록

- Supervisor 노드 추가 금지 (불필요한 복잡도)
- State에 개별 agent result dict 필드 만들지 말 것
- Fetch에서 LLM 호출 금지
- cached_api_call 없이 직접 API 호출 금지
- LLM free-form text 응답 금지
- 모든 기간에 동일 가중치 적용 금지
- 에러 시 크래시 금지 — error_log 기록 + score 0.0 리턴

## 알려진 TODO (하드코딩)

- 업종 평균 PER: 현재 18.0 하드코딩, 추후 섹터별 DB 구축
- 한국은행 기준금리: ECOS API 연동 전까지 3.00% 하드코딩
- ticker_mapper: 국내 주요 종목만 하드코딩, 추후 CSV DB 확장

## 구현 순서

Phase 1: 뼈대 + Price Agent → Phase 2: 나머지 4개 에이전트 → Phase 3: 그래프 완성 (병렬, Validator, Retry) → Phase 4: Synthesizer + Report

## 디렉토리 구조

```
stock-advisor/
├── main.py
├── graph/          # state.py, graph.py, edges.py
├── agents/         # base.py, input_parser.py, *_agent.py, validator.py, synthesizer.py, report_generator.py
├── tools/          # cache.py, *_tools.py
├── utils/          # ticker_mapper.py, score_calculator.py
├── data/           # ticker_db.csv
├── config/         # settings.py
├── tests/
├── .cache/         # API 응답 파일 캐시 (.gitignore 대상)
└── requirements.txt
```

## 투자 책임 고지

리포트 하단에 반드시 포함: "본 시스템의 출력은 참고용이며, 실제 매매 결정의 책임은 사용자에게 있음."
