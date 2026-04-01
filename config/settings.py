import os
from dotenv import load_dotenv

load_dotenv()

# ── 캐시 ──
CACHE_DIR = os.getenv("CACHE_DIR", ".cache/api_responses")

# ── LLM ──
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# ── 외부 API 키 ──
DART_API_KEY = os.getenv("DART_API_KEY", "")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# ── 가중치 테이블 (7개 에이전트) ──
WEIGHT_TABLE = {
    "short": {"price": 0.25, "fundamental": 0.10, "disclosure": 0.10, "news": 0.15, "macro": 0.05, "supply_demand": 0.20, "consensus": 0.15},
    "mid":   {"price": 0.15, "fundamental": 0.20, "disclosure": 0.10, "news": 0.10, "macro": 0.10, "supply_demand": 0.15, "consensus": 0.20},
    "long":  {"price": 0.05, "fundamental": 0.30, "disclosure": 0.10, "news": 0.05, "macro": 0.15, "supply_demand": 0.10, "consensus": 0.25},
}

# ── 시그널 기준 ──
SIGNAL_THRESHOLDS = {
    "STRONG_BUY": 0.80,
    "BUY": 0.65,
    "HOLD": 0.45,
    "SELL": 0.30,
    # 0.30 미만 → STRONG_SELL
}
