"""Historical Backtest — 과거 3개월 시뮬레이션으로 가중치 최적화.

뉴스/공시는 과거 데이터 없으므로 4개 에이전트만 사용:
Price, Fundamental, SupplyDemand, Consensus

과거 매주 월요일 기준으로 분석 → N일 후 실제 수익률과 비교
→ 에이전트별 정확도 → 기본 가중치 + 종목별 보정값 도출
"""

import io
import json
import sys
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from agents.consensus_agent import ConsensusAgent
from agents.fundamental_agent import FundamentalAgent
from agents.price_agent import PriceAgent, compute_indicators
from agents.supply_demand_agent import SupplyDemandAgent
from config.settings import WEIGHT_TABLE

OUTPUT_DIR = Path("data/backtest")

BACKTEST_STOCKS = [
    {"ticker": "005930.KS", "market": "KRX", "name": "삼성전자"},
    {"ticker": "000660.KS", "market": "KRX", "name": "SK하이닉스"},
    {"ticker": "005380.KS", "market": "KRX", "name": "현대차"},
    {"ticker": "035420.KS", "market": "KRX", "name": "NAVER"},
    {"ticker": "035720.KS", "market": "KRX", "name": "카카오"},
    {"ticker": "AAPL", "market": "NASDAQ", "name": "Apple"},
    {"ticker": "NVDA", "market": "NASDAQ", "name": "NVIDIA"},
    {"ticker": "TSLA", "market": "NASDAQ", "name": "Tesla"},
    {"ticker": "MSFT", "market": "NASDAQ", "name": "Microsoft"},
    {"ticker": "GOOGL", "market": "NASDAQ", "name": "Alphabet"},
]

# 백테스트에 사용 가능한 에이전트 (과거 데이터 조회 가능한 것만)
BACKTEST_AGENTS = ["price", "fundamental", "supply_demand", "consensus"]
# 과거 데이터 없는 에이전트 (뉴스, 공시, 거시 → 기본값 0.5로 고정)
FIXED_AGENTS = ["disclosure", "news", "macro"]

EVAL_DAYS = 10  # 10영업일 (2주) 후 수익률로 평가


def fetch_historical_prices(ticker: str, months: int = 6) -> pd.DataFrame:
    """과거 가격 데이터 (백테스트용 넉넉하게)."""
    stock = yf.Ticker(ticker)
    df = stock.history(period=f"{months}mo")
    return df


def simulate_price_score(df: pd.DataFrame, date_idx: int) -> float:
    """특정 시점의 Price Agent 스코어를 Rule-based로 계산 (LLM 없이)."""
    if date_idx < 20:
        return 0.5

    window = df.iloc[max(0, date_idx - 60):date_idx + 1]
    if len(window) < 14:
        return 0.5

    raw_data = {
        "close": window["Close"].tolist(),
        "open": window["Open"].tolist(),
        "high": window["High"].tolist(),
        "low": window["Low"].tolist(),
        "volume": window["Volume"].tolist(),
    }

    try:
        indicators = compute_indicators(raw_data)
        agent = PriceAgent()
        return agent.calculate_score({"indicators": indicators}, {})
    except Exception:
        return 0.5


def simulate_fundamental_score(ticker: str) -> float:
    """Fundamental 스코어 (재무제표는 분기 1회 변경이라 기간 내 동일)."""
    from agents.fundamental_agent import (
        SECTOR_AVG_PER,
        FundamentalAgent,
        _fetch_yfinance_fundamental,
    )

    data = _fetch_yfinance_fundamental(ticker)
    if not data:
        return 0.5

    agent = FundamentalAgent()
    return agent.calculate_score(data, {})


def simulate_consensus_score(ticker: str) -> float:
    """Consensus 스코어 (애널리스트 의견은 천천히 변하므로 현재 데이터 사용)."""
    from agents.consensus_agent import ConsensusAgent, _fetch_consensus

    data = _fetch_consensus(ticker)
    if not data:
        return 0.5

    agent = ConsensusAgent()
    return agent.calculate_score(data, {})


def simulate_supply_demand_score(ticker: str) -> float:
    """SupplyDemand 스코어."""
    from agents.supply_demand_agent import SupplyDemandAgent, _fetch_supply_demand

    data = _fetch_supply_demand(ticker)
    if not data:
        return 0.5

    agent = SupplyDemandAgent()
    return agent.calculate_score(data, {})


def run_backtest(weeks: int = 12, eval_days: int = 10) -> dict:
    """과거 백테스트 실행."""
    print(f"=== Historical Backtest ({weeks}주, {eval_days}일 평가) ===\n")

    all_results = []

    for stock_info in BACKTEST_STOCKS:
        ticker = stock_info["ticker"]
        name = stock_info["name"]
        print(f"[{name}] 데이터 로드 중...", end=" ", flush=True)

        try:
            df = fetch_historical_prices(ticker, months=6)
        except Exception as e:
            print(f"SKIP ({e})")
            continue

        if len(df) < 60:
            print(f"SKIP (데이터 부족: {len(df)}일)")
            continue

        # 재무/컨센서스/수급 스코어 (기간 내 거의 동일)
        fund_score = simulate_fundamental_score(ticker)
        cons_score = simulate_consensus_score(ticker)
        sd_score = simulate_supply_demand_score(ticker)

        # 매주 월요일 기준으로 시뮬레이션
        stock_results = []
        step = 5  # 5영업일 = 1주
        start_idx = 60  # 60일 이후부터 (지표 계산에 충분한 데이터)

        for i in range(start_idx, len(df) - eval_days, step):
            if len(stock_results) >= weeks:
                break

            date = df.index[i].strftime("%Y-%m-%d")
            price_at_signal = float(df["Close"].iloc[i])
            price_after = float(df["Close"].iloc[min(i + eval_days, len(df) - 1)])
            actual_return = ((price_after - price_at_signal) / price_at_signal) * 100

            # Price 스코어는 시점별로 계산
            price_score = simulate_price_score(df, i)

            scores = {
                "price": round(price_score, 4),
                "fundamental": round(fund_score, 4),
                "supply_demand": round(sd_score, 4),
                "consensus": round(cons_score, 4),
                "disclosure": 0.5,  # 기본값
                "news": 0.5,
                "macro": 0.5,
            }

            stock_results.append({
                "date": date,
                "ticker": ticker,
                "name": name,
                "scores": scores,
                "price_at_signal": round(price_at_signal, 2),
                "price_after": round(price_after, 2),
                "actual_return": round(actual_return, 2),
            })

        all_results.extend(stock_results)
        print(f"{len(stock_results)}주 시뮬레이션 완료")

    print(f"\n총 {len(all_results)}건 시뮬레이션")

    # 에이전트별 정확도 분석
    agent_accuracy = _analyze_agent_accuracy(all_results)

    # 종목별 에이전트 정확도
    per_stock_accuracy = _analyze_per_stock(all_results)

    # 최적 가중치 탐색
    optimal_weights = _find_optimal_weights(all_results)

    # 종목별 최적 가중치
    per_stock_weights = _find_per_stock_weights(all_results)

    result = {
        "backtest_date": datetime.now().strftime("%Y-%m-%d"),
        "weeks": weeks,
        "eval_days": eval_days,
        "total_samples": len(all_results),
        "agent_accuracy": agent_accuracy,
        "per_stock_accuracy": per_stock_accuracy,
        "optimal_weights": optimal_weights,
        "per_stock_weights": per_stock_weights,
    }

    # 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"backtest_{datetime.now().strftime('%Y%m%d')}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {output_path}")

    # 결과 출력
    _print_results(result)

    return result


def _analyze_agent_accuracy(results: list) -> dict:
    """에이전트별 스코어-수익률 상관관계."""
    agents = BACKTEST_AGENTS
    accuracy = {}

    for agent in agents:
        pairs = [(r["scores"][agent], r["actual_return"]) for r in results if r["scores"].get(agent, 0) != 0.5]

        if len(pairs) < 5:
            accuracy[agent] = {"correlation": 0.0, "accuracy": 0.5, "samples": len(pairs)}
            continue

        scores, returns = zip(*pairs)
        corr = _pearson(list(scores), list(returns))

        # 정확도: 스코어 > 0.5일 때 실제 상승 비율
        correct = sum(1 for s, r in pairs if (s > 0.5 and r > 0) or (s < 0.5 and r < 0))
        acc = correct / len(pairs)

        accuracy[agent] = {"correlation": round(corr, 3), "accuracy": round(acc, 3), "samples": len(pairs)}

    return accuracy


def _analyze_per_stock(results: list) -> dict:
    """종목별 에이전트 정확도."""
    per_stock = {}
    tickers = set(r["ticker"] for r in results)

    for ticker in tickers:
        stock_results = [r for r in results if r["ticker"] == ticker]
        name = stock_results[0]["name"]

        stock_acc = {}
        for agent in BACKTEST_AGENTS:
            pairs = [(r["scores"][agent], r["actual_return"]) for r in stock_results]
            if len(pairs) < 3:
                continue

            scores, returns = zip(*pairs)
            corr = _pearson(list(scores), list(returns))
            correct = sum(1 for s, r in pairs if (s > 0.5 and r > 0) or (s < 0.5 and r < 0))
            acc = correct / len(pairs)

            stock_acc[agent] = {"correlation": round(corr, 3), "accuracy": round(acc, 3)}

        per_stock[ticker] = {"name": name, "agents": stock_acc}

    return per_stock


def _find_optimal_weights(results: list) -> dict:
    """전체 데이터에서 최적 가중치 탐색 (grid search)."""
    best_score = -999
    best_weights = {}

    # 7개 에이전트 가중치 조합 탐색 (0.05 단위)
    # 4개 에이전트만 가변, 3개(disclosure/news/macro)는 균등 배분
    steps = [i / 20 for i in range(1, 10)]  # 0.05 ~ 0.45

    candidates_tried = 0
    for p, f, s, c in product(steps, steps, steps, steps):
        remaining = 1.0 - p - f - s - c
        if remaining < 0.09 or remaining > 0.45:  # 나머지 3개에 각 0.03~0.15
            continue

        d = n = m = round(remaining / 3, 4)
        weights = {
            "price": p, "fundamental": f, "supply_demand": s, "consensus": c,
            "disclosure": d, "news": n, "macro": m,
        }

        # 이 가중치로 시그널 정확도 계산
        score = _evaluate_weights(results, weights)
        candidates_tried += 1

        if score > best_score:
            best_score = score
            best_weights = weights

    best_weights = {k: round(v, 4) for k, v in best_weights.items()}
    return {"weights": best_weights, "score": round(best_score, 3), "candidates_tried": candidates_tried}


def _find_per_stock_weights(results: list) -> dict:
    """종목별 최적 가중치."""
    per_stock = {}
    tickers = set(r["ticker"] for r in results)

    for ticker in tickers:
        stock_results = [r for r in results if r["ticker"] == ticker]
        if len(stock_results) < 5:
            continue

        name = stock_results[0]["name"]
        optimal = _find_optimal_weights(stock_results)
        per_stock[ticker] = {"name": name, **optimal}

    return per_stock


def _evaluate_weights(results: list, weights: dict) -> float:
    """가중치 조합의 예측 정확도 평가. 높을수록 좋음."""
    correct = 0
    total = 0
    profit_sum = 0.0

    for r in results:
        # 가중치 적용 종합 스코어
        weighted = sum(r["scores"].get(k, 0.5) * w for k, w in weights.items())
        actual = r["actual_return"]

        # 시그널 판단
        if weighted >= 0.65:  # BUY 이상
            predicted_up = True
        elif weighted <= 0.35:  # SELL 이하
            predicted_up = False
        else:
            continue  # HOLD는 평가에서 제외

        total += 1
        actually_up = actual > 0

        if predicted_up == actually_up:
            correct += 1

        # 수익률 시뮬: BUY면 +수익률, SELL이면 -수익률(공매도)
        if predicted_up:
            profit_sum += actual
        else:
            profit_sum -= actual

    if total == 0:
        return 0.0

    accuracy = correct / total
    avg_profit = profit_sum / total

    # 정확도 70% + 평균 수익률 30% (수익률은 ±10% 범위로 정규화)
    return accuracy * 0.7 + min(max(avg_profit / 10, -1), 1) * 0.3


def _pearson(x: list, y: list) -> float:
    """피어슨 상관계수."""
    n = len(x)
    if n < 3:
        return 0.0

    mx, my = sum(x) / n, sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y)) / n
    sx = (sum((a - mx) ** 2 for a in x) / n) ** 0.5
    sy = (sum((b - my) ** 2 for b in y) / n) ** 0.5

    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


def _print_results(result: dict) -> None:
    """결과 출력."""
    print(f"\n{'='*60}")
    print(f"=== 에이전트별 정확도 ===")
    print(f"{'에이전트':<18} {'정확도':>8} {'상관계수':>8} {'샘플':>6}")
    print(f"{'-'*42}")
    for k, v in result["agent_accuracy"].items():
        print(f"{k:<18} {v['accuracy']:>7.1%} {v['correlation']:>+8.3f} {v['samples']:>6}")

    print(f"\n=== 종목별 에이전트 정확도 ===")
    for ticker, data in result["per_stock_accuracy"].items():
        name = data["name"]
        agents = data["agents"]
        best_agent = max(agents.items(), key=lambda x: x[1]["correlation"]) if agents else ("N/A", {"correlation": 0})
        print(f"  {name:<12} 최적 에이전트: {best_agent[0]} (상관계수: {best_agent[1]['correlation']:+.3f})")

    opt = result["optimal_weights"]
    print(f"\n=== 최적 가중치 (전체) ===")
    print(f"정확도 점수: {opt['score']:.3f}")
    for k, v in opt["weights"].items():
        current = WEIGHT_TABLE["short"].get(k, 0)
        diff = v - current
        marker = " ***" if abs(diff) >= 0.05 else ""
        print(f"  {k:<18} {v:>7.2%}  (현재: {current:.2%}, 변화: {diff:+.2%}){marker}")

    print(f"\n=== 종목별 최적 가중치 ===")
    for ticker, data in result.get("per_stock_weights", {}).items():
        name = data["name"]
        w = data["weights"]
        top = sorted(w.items(), key=lambda x: x[1], reverse=True)[:3]
        top_str = ", ".join(f"{k}={v:.0%}" for k, v in top)
        print(f"  {name:<12} [{top_str}]")

    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Historical Backtest")
    parser.add_argument("--weeks", type=int, default=12, help="백테스트 기간 (주)")
    parser.add_argument("--eval-days", type=int, default=10, help="평가 기간 (영업일)")
    args = parser.parse_args()

    run_backtest(args.weeks, args.eval_days)
