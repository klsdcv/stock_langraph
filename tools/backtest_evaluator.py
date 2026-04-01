"""Backtest Evaluator — 과거 시그널 vs 실제 수익률 비교.

history 데이터에서 시그널을 냈던 날의 가격과 N일 후 실제 가격을 비교하여
각 에이전트의 예측 정확도를 평가.
"""

import io
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HISTORY_DIR = Path("data/history")

# 시그널별 기대 수익률 방향
SIGNAL_EXPECTED = {
    "STRONG_BUY": "up",      # 상승 기대
    "BUY": "up",
    "HOLD": "neutral",
    "SELL": "down",           # 하락 기대
    "STRONG_SELL": "down",
}

# 평가 기간 (영업일 기준)
EVAL_DAYS = [5, 10, 20]  # 1주, 2주, 4주 후


def get_actual_return(ticker: str, start_date: str, days: int) -> float | None:
    """시그널 날짜 기준 N일 후 실제 수익률(%) 계산."""
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = start + timedelta(days=days + 10)  # 여유분

        stock = yf.Ticker(ticker)
        df = stock.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))

        if df.empty or len(df) < 2:
            return None

        price_start = float(df["Close"].iloc[0])
        # N 영업일 후 (또는 가용한 마지막 날)
        target_idx = min(days, len(df) - 1)
        price_end = float(df["Close"].iloc[target_idx])

        return round(((price_end - price_start) / price_start) * 100, 2)
    except Exception:
        return None


def evaluate_signal(signal: str, actual_return: float) -> dict:
    """시그널 정확도 평가.

    Returns:
        correct: 시그널 방향이 맞았는지
        score: 정확도 점수 (-1.0 ~ +1.0)
    """
    expected = SIGNAL_EXPECTED.get(signal, "neutral")

    if expected == "up":
        correct = actual_return > 0
        # 상승폭이 클수록 높은 점수
        score = min(actual_return / 10, 1.0) if correct else max(actual_return / 10, -1.0)
    elif expected == "down":
        correct = actual_return < 0
        # 하락폭이 클수록 높은 점수 (SELL이 맞았으니)
        score = min(-actual_return / 10, 1.0) if correct else max(-actual_return / 10, -1.0)
    else:  # neutral (HOLD)
        correct = abs(actual_return) < 5  # ±5% 이내면 정확
        score = 1.0 - min(abs(actual_return) / 10, 1.0)

    return {"correct": correct, "score": round(score, 3)}


def evaluate_agent_accuracy(records: list[dict], eval_days: int = 10) -> dict:
    """에이전트별 예측 정확도 계산.

    각 에이전트 스코어가 높았던 종목이 실제로 올랐는지,
    낮았던 종목이 실제로 떨어졌는지 상관관계를 계산.
    """
    agent_keys = ["price", "fundamental", "disclosure", "news", "macro", "supply_demand", "consensus"]
    agent_stats = {k: {"total": 0, "correct": 0, "score_sum": 0.0, "correlations": []} for k in agent_keys}

    for record in records:
        date = record["date"]
        for stock in record.get("stocks", []):
            ticker = stock.get("ticker", "")
            scores = stock.get("scores", {})

            actual_return = get_actual_return(ticker, date, eval_days)
            if actual_return is None:
                continue

            for agent_key in agent_keys:
                agent_score = scores.get(agent_key, 0)
                if agent_score == 0:
                    continue

                agent_stats[agent_key]["total"] += 1
                # 에이전트 스코어 > 0.5면 "상승 예측", < 0.5면 "하락 예측"
                predicted_up = agent_score > 0.5
                actually_up = actual_return > 0

                if predicted_up == actually_up:
                    agent_stats[agent_key]["correct"] += 1

                # 스코어-수익률 상관관계용 데이터
                agent_stats[agent_key]["correlations"].append({
                    "score": agent_score,
                    "return": actual_return,
                })

    # 정확도 + 상관계수 계산
    results = {}
    for key, stats in agent_stats.items():
        total = stats["total"]
        if total == 0:
            results[key] = {"accuracy": 0.0, "correlation": 0.0, "samples": 0}
            continue

        accuracy = stats["correct"] / total
        correlation = _calc_correlation(stats["correlations"])

        results[key] = {
            "accuracy": round(accuracy, 3),
            "correlation": round(correlation, 3),
            "samples": total,
        }

    return results


def _calc_correlation(data: list[dict]) -> float:
    """스코어와 수익률 간 피어슨 상관계수."""
    if len(data) < 3:
        return 0.0

    scores = [d["score"] for d in data]
    returns = [d["return"] for d in data]

    n = len(scores)
    mean_s = sum(scores) / n
    mean_r = sum(returns) / n

    cov = sum((s - mean_s) * (r - mean_r) for s, r in zip(scores, returns)) / n
    std_s = (sum((s - mean_s) ** 2 for s in scores) / n) ** 0.5
    std_r = (sum((r - mean_r) ** 2 for r in returns) / n) ** 0.5

    if std_s == 0 or std_r == 0:
        return 0.0

    return cov / (std_s * std_r)


def evaluate_all(eval_days: int = 10) -> dict:
    """전체 history 평가."""
    records = []
    for path in sorted(HISTORY_DIR.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            records.append(json.load(f))

    if not records:
        print("history 데이터가 없습니다.")
        return {}

    # 시그널 정확도
    signal_stats = {"total": 0, "correct": 0}
    for record in records:
        for stock in record.get("stocks", []):
            signal = stock.get("signal", "HOLD")
            actual = get_actual_return(stock["ticker"], record["date"], eval_days)
            if actual is None:
                continue

            result = evaluate_signal(signal, actual)
            signal_stats["total"] += 1
            if result["correct"]:
                signal_stats["correct"] += 1

    # 에이전트별 정확도
    agent_accuracy = evaluate_agent_accuracy(records, eval_days)

    overall = {
        "eval_days": eval_days,
        "total_signals": signal_stats["total"],
        "signal_accuracy": round(signal_stats["correct"] / max(signal_stats["total"], 1), 3),
        "agent_accuracy": agent_accuracy,
        "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    return overall


def print_evaluation(eval_result: dict) -> None:
    """평가 결과 출력."""
    if not eval_result:
        return

    print(f"\n=== 시그널 정확도 평가 ({eval_result['eval_days']}일 후 기준) ===")
    print(f"평가 시점: {eval_result['evaluated_at']}")
    print(f"총 시그널: {eval_result['total_signals']}건")
    print(f"전체 정확도: {eval_result['signal_accuracy']:.1%}")
    print()

    print(f"{'에이전트':<18} {'정확도':>8} {'상관계수':>8} {'샘플수':>6}")
    print(f"{'-'*44}")
    for key, stats in eval_result["agent_accuracy"].items():
        print(f"{key:<18} {stats['accuracy']:>7.1%} {stats['correlation']:>+8.3f} {stats['samples']:>6}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backtest Evaluator")
    parser.add_argument("--days", type=int, default=10, help="평가 기간 (영업일)")
    args = parser.parse_args()

    result = evaluate_all(args.days)
    print_evaluation(result)
