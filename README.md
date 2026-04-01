# Stock Advisor

> LangGraph 기반 7개 병렬 AI 에이전트가 종합 매매 시그널을 생성하는 주식 분석 시스템

```
$ python main.py "삼성전자"

# 삼성전자 (005930.KS) 분석 리포트
# 시그널: [+] BUY | 신뢰도: 89.8%
# 종합 스코어: 0.651
```

## Architecture

![Graph Architecture](docs/graph_architecture.png)

**7개 에이전트가 병렬로 분석 → Validator가 검증 → 실패 시 자동 재시도 → Synthesizer가 종합 판단**

```
Input Parser
    ├── Price Agent        (기술적 분석: RSI, MACD, 볼린저밴드)
    ├── Fundamental Agent  (펀더멘털: PER, PBR, ROE, 부채비율)
    ├── Disclosure Agent   (공시 분석: DART 전자공시)
    ├── News Agent         (뉴스 감성 분석)
    ├── Macro Agent        (거시경제: VIX, 금리, 환율)
    ├── Supply/Demand Agent(수급: 기관/외국인 매매동향)
    └── Consensus Agent    (애널리스트 컨센서스)
         ↓
    Validator (품질 검증 + 자동 재시도)
         ↓
    Synthesizer (기간별 가중치 적용 → 종합 스코어)
         ↓
    Report Generator (마크다운 리포트)
```

## Signal System

| 스코어 | 시그널 |
|---|---|
| 0.80+ | STRONG_BUY |
| 0.65 ~ 0.79 | BUY |
| 0.45 ~ 0.64 | HOLD |
| 0.30 ~ 0.44 | SELL |
| < 0.30 | STRONG_SELL |

기간(단기/중기/장기)에 따라 에이전트별 가중치가 달라집니다:
- **단기**: 기술적 분석(25%) + 수급(20%) 중심
- **중기**: 펀더멘털(20%) + 컨센서스(20%) 중심
- **장기**: 펀더멘털(30%) + 컨센서스(25%) 중심

## Example Results

| 종목 | 시그널 | 스코어 | 신뢰도 | 리포트 |
|---|---|---|---|---|
| 삼성전자 | **BUY** | 0.651 | 89.8% | [samsung_electronics.md](examples/samsung_electronics.md) |
| SK하이닉스 | **HOLD** | 0.641 | 85.8% | [sk_hynix.md](examples/sk_hynix.md) |
| 카카오 | **HOLD** | 0.535 | 91.4% | [kakao.md](examples/kakao.md) |
| 현대차 | **HOLD** | 0.550 | 89.3% | [hyundai_motor.md](examples/hyundai_motor.md) |
| LG에너지솔루션 | **SELL** | 0.431 | 84.3% | [lg_energy_solution.md](examples/lg_energy_solution.md) |

## Tech Stack

- **Framework**: [LangGraph](https://github.com/langchain-ai/langgraph) (StateGraph, 병렬 fan-out/fan-in)
- **LLM**: GPT-4o-mini (분석 1회당 LLM 호출 8회 = 7 에이전트 + 1 Synthesizer)
- **Data Sources**: yfinance, pykrx, DART OpenAPI, NewsAPI, FRED
- **Language**: Python 3.11+

### Key Design Decisions

- **병렬 실행**: 7개 에이전트가 동시에 실행되어 분석 시간 최소화
- **자동 재시도**: Validator가 실패한 에이전트만 선별하여 최대 2회 재시도
- **Graceful Degradation**: API 실패 시 만료 캐시 → 에러 로그 기록 + score 0.0 (크래시 방지)
- **파일 캐시**: 모든 외부 API 응답을 `cached_api_call()`로 캐싱 (TTL 기반)
- **JSON-only LLM 응답**: 모든 LLM 출력을 구조화된 JSON으로 강제하여 파싱 안정성 확보

## Project Structure

```
stock-advisor/
├── main.py                  # 진입점
├── graph/
│   ├── state.py             # StockAnalysisState (Annotated reducer 기반)
│   ├── graph.py             # LangGraph 빌드 (fan-out/fan-in + retry)
│   └── edges.py             # Conditional edge 라우팅
├── agents/
│   ├── base.py              # BaseAgent 추상 클래스
│   ├── input_parser.py      # 종목명 → ticker 변환
│   ├── price_agent.py       # 기술적 분석 (RSI, MACD, BB)
│   ├── fundamental_agent.py # 재무제표 분석
│   ├── disclosure_agent.py  # DART 공시 분석
│   ├── news_agent.py        # 뉴스 감성 분석
│   ├── macro_agent.py       # 거시경제 지표
│   ├── supply_demand_agent.py # 수급 분석
│   ├── consensus_agent.py   # 애널리스트 컨센서스
│   ├── validator.py         # 품질 검증 + 재시도 판단
│   ├── synthesizer.py       # 가중 평균 스코어 → 시그널
│   └── report_generator.py  # 마크다운 리포트 생성
├── tools/
│   └── cache.py             # cached_api_call (TTL 기반 파일 캐시)
├── utils/
│   ├── ticker_mapper.py     # 종목명 ↔ ticker 매핑
│   └── score_calculator.py  # 스코어 유효성 검증
├── config/
│   └── settings.py          # 가중치, 시그널 기준, API 키
├── docs/
│   └── graph_architecture.png
├── examples/                # 분석 결과 샘플
└── tests/
```

## Getting Started

### Prerequisites

- Python 3.11+
- OpenAI API Key

### Installation

```bash
git clone https://github.com/your-username/stock-advisor.git
cd stock-advisor
pip install -r requirements.txt
```

### Configuration

`.env` 파일을 생성합니다:

```env
OPENAI_API_KEY=sk-your-key-here

# Optional (없어도 기본 동작)
DART_API_KEY=your-dart-key
NEWSAPI_KEY=your-newsapi-key
FRED_API_KEY=your-fred-key
```

### Usage

```bash
# 단일 종목 분석
python main.py "삼성전자"
python main.py "SK하이닉스"

# 대화형 모드
python main.py
> 분석할 종목을 입력하세요: 카카오
```

---

*본 시스템의 출력은 참고용이며, 실제 매매 결정의 책임은 사용자에게 있습니다.*
