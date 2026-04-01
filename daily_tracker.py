"""Daily Tracker — 매일 KOSPI 5 + NASDAQ 5 종목 분석 후 기록 저장."""

import io
import json
import sys
from datetime import datetime
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from graph.graph import app

HISTORY_DIR = Path("data/history")

# 추적 대상 종목
KRX_STOCKS = [
    {"ticker": "005930.KS", "market": "KRX", "name": "삼성전자"},
    {"ticker": "000660.KS", "market": "KRX", "name": "SK하이닉스"},
    {"ticker": "005380.KS", "market": "KRX", "name": "현대차"},
    {"ticker": "035420.KS", "market": "KRX", "name": "NAVER"},
    {"ticker": "035720.KS", "market": "KRX", "name": "카카오"},
]

US_STOCKS = [
    {"ticker": "AAPL", "market": "NASDAQ", "name": "Apple"},
    {"ticker": "NVDA", "market": "NASDAQ", "name": "NVIDIA"},
    {"ticker": "TSLA", "market": "NASDAQ", "name": "Tesla"},
    {"ticker": "MSFT", "market": "NASDAQ", "name": "Microsoft"},
    {"ticker": "GOOGL", "market": "NASDAQ", "name": "Alphabet"},
]

ALL_STOCKS = KRX_STOCKS + US_STOCKS


def analyze_stock(ticker: str, market: str, period: str = "short") -> dict:
    """단일 종목 분석. input_parser를 건너뛰고 직접 ticker 주입."""
    initial_state = {
        "raw_input": "",
        "ticker": ticker,
        "market": market,
        "analysis_period": period,
        "agent_results": {},
        "validation_result": {},
        "retry_targets": [],
        "retry_count": 0,
        "signal": "HOLD",
        "confidence": 0.0,
        "reasoning": "",
        "risk_factors": [],
        "final_report": "",
        "messages": [],
        "error_log": [],
    }
    return app.invoke(initial_state)


def extract_summary(result: dict, stock_info: dict) -> dict:
    """분석 결과에서 기록용 요약 추출."""
    agent_results = result.get("agent_results", {})
    scores = {}
    for key in ["price", "fundamental", "disclosure", "news", "macro", "supply_demand", "consensus"]:
        r = agent_results.get(key, {})
        scores[key] = r.get("score", 0.0)

    # 현재가 추출
    price_data = agent_results.get("price", {}).get("raw_data", {}).get("indicators", {})
    current_price = price_data.get("current_price", 0)

    # 목표가 추출
    consensus_data = agent_results.get("consensus", {}).get("raw_data", {})
    target_mean = consensus_data.get("target_mean", 0)

    return {
        "ticker": stock_info["ticker"],
        "name": stock_info["name"],
        "market": stock_info["market"],
        "signal": result.get("signal", "HOLD"),
        "confidence": result.get("confidence", 0.0),
        "weighted_score": round(sum(
            scores[k] * w
            for k, w in zip(scores.keys(), [0.25, 0.10, 0.10, 0.15, 0.05, 0.20, 0.15])
        ), 4),
        "scores": {k: round(v, 4) for k, v in scores.items()},
        "current_price": current_price,
        "target_mean": target_mean,
        "error_count": len(result.get("error_log", [])),
        "errors": result.get("error_log", []),
    }


def run_daily(period: str = "short", market_filter: str = "all") -> None:
    """추적 종목 분석 + 저장.

    market_filter: "all" | "krx" | "us"
    """
    if market_filter == "krx":
        stocks = KRX_STOCKS
        label = "KRX"
    elif market_filter == "us":
        stocks = US_STOCKS
        label = "US"
    else:
        stocks = ALL_STOCKS
        label = "ALL"

    today = datetime.now().strftime("%Y-%m-%d")
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== Daily Tracker: {today} [{label}] ===")
    print(f"종목 수: {len(stocks)}")
    print()

    results = []
    for i, stock in enumerate(stocks, 1):
        print(f"[{i}/{len(stocks)}] {stock['name']} ({stock['ticker']})...", end=" ", flush=True)
        try:
            result = analyze_stock(stock["ticker"], stock["market"], period)
            summary = extract_summary(result, stock)
            results.append(summary)
            print(f"{summary['signal']} ({summary['weighted_score']:.3f})")
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "ticker": stock["ticker"],
                "name": stock["name"],
                "market": stock["market"],
                "signal": "ERROR",
                "confidence": 0.0,
                "weighted_score": 0.0,
                "scores": {},
                "current_price": 0,
                "target_mean": 0,
                "error_count": 1,
            })

    # 기존 파일이 있으면 다른 시장 결과와 병합
    suffix = f"_{market_filter}" if market_filter != "all" else ""
    output_path = HISTORY_DIR / f"{today}.json"

    existing_stocks = []
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        # 기존 기록에서 현재 시장 종목 제거 후 새 결과 추가
        current_tickers = {s["ticker"] for s in stocks}
        existing_stocks = [s for s in existing.get("stocks", []) if s["ticker"] not in current_tickers]

    record = {
        "date": today,
        "period": period,
        "stocks": existing_stocks + results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료: {output_path}")

    # 요약 테이블
    print(f"\n{'='*70}")
    print(f"{'종목':<12} {'시그널':<12} {'스코어':>8} {'신뢰도':>8} {'현재가':>12} {'목표가':>12}")
    print(f"{'-'*70}")
    for r in sorted(results, key=lambda x: x["weighted_score"], reverse=True):
        print(f"{r['name']:<12} {r['signal']:<12} {r['weighted_score']:>8.3f} {r['confidence']:>7.1%} {r['current_price']:>12,.0f} {r['target_mean']:>12,.0f}")
    print(f"{'='*70}")

    # 이전 기록과 비교
    _print_changes(today, results)


def _print_changes(today: str, results: list) -> None:
    """이전 기록 대비 변화 출력."""
    history_files = sorted(HISTORY_DIR.glob("*.json"))
    # 오늘 파일 제외하고 가장 최근 파일
    prev_files = [f for f in history_files if f.stem != today]
    if not prev_files:
        return

    prev_path = prev_files[-1]
    with open(prev_path, "r", encoding="utf-8") as f:
        prev_record = json.load(f)

    prev_map = {s["ticker"]: s for s in prev_record.get("stocks", [])}

    changes = []
    for r in results:
        prev = prev_map.get(r["ticker"])
        if not prev:
            continue
        if r["signal"] != prev["signal"]:
            changes.append(f"  {r['name']}: {prev['signal']} -> {r['signal']}")
        score_diff = r["weighted_score"] - prev.get("weighted_score", 0)
        if abs(score_diff) >= 0.05:
            direction = "+" if score_diff > 0 else ""
            changes.append(f"  {r['name']}: 스코어 {direction}{score_diff:.3f}")

    if changes:
        print(f"\n[변화 감지] vs {prev_path.stem}")
        for c in changes:
            print(c)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Daily Stock Tracker")
    parser.add_argument("--period", default="short", choices=["short", "mid", "long"])
    parser.add_argument("--market", default="all", choices=["all", "krx", "us"],
                        help="krx=국장만, us=미장만, all=전체")
    args = parser.parse_args()

    run_daily(args.period, args.market)
