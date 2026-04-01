from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def merge_dicts(left: dict, right: dict) -> dict:
    """병렬 에이전트 결과를 안전하게 병합."""
    merged = left.copy()
    merged.update(right)
    return merged


def append_list(left: list, right: list) -> list:
    return left + right


class StockAnalysisState(TypedDict):
    # 입력
    raw_input: str
    ticker: str
    market: Literal["KRX", "NYSE", "NASDAQ"]
    analysis_period: Literal["short", "mid", "long"]

    # 에이전트 결과 — 핵심: Annotated + merge_dicts reducer
    agent_results: Annotated[dict, merge_dicts]

    # 검증
    validation_result: dict
    retry_targets: list[str]
    retry_count: int  # max 2

    # 최종 출력
    signal: Literal["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]
    confidence: float  # 0.0 ~ 1.0
    reasoning: str
    risk_factors: Annotated[list[str], append_list]
    final_report: str

    # 메타
    messages: Annotated[list[AnyMessage], add_messages]
    error_log: Annotated[list[str], append_list]
