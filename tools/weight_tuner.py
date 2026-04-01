"""Weight Tuner — 에이전트 정확도 기반 가중치 자동 조정.

backtest_evaluator 결과를 기반으로 가중치를 조정.
정확한 에이전트의 가중치를 올리고, 부정확한 에이전트의 가중치를 내림.
"""

import io
import json
import sys
from datetime import datetime
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from config.settings import WEIGHT_TABLE
from tools.backtest_evaluator import evaluate_all

TUNING_HISTORY_DIR = Path("data/tuning")
AGENT_KEYS = ["price", "fundamental", "disclosure", "news", "macro", "supply_demand", "consensus"]

# 가중치 조정 강도 (0.0 ~ 1.0)
# 높을수록 정확도 결과에 민감하게 반응
LEARNING_RATE = 0.3

# 최소/최대 가중치 (특정 에이전트가 0이 되거나 지배적이 되는 것 방지)
MIN_WEIGHT = 0.03
MAX_WEIGHT = 0.40


def compute_new_weights(
    current_weights: dict[str, float],
    agent_accuracy: dict[str, dict],
) -> dict[str, float]:
    """에이전트 정확도 기반으로 새 가중치 계산.

    방식: accuracy * correlation 조합으로 "신뢰도 점수"를 매기고,
    기존 가중치와 블렌딩.
    """
    # 1. 각 에이전트의 신뢰도 점수 계산
    trust_scores = {}
    for key in AGENT_KEYS:
        stats = agent_accuracy.get(key, {})
        accuracy = stats.get("accuracy", 0.5)
        correlation = stats.get("correlation", 0.0)
        samples = stats.get("samples", 0)

        if samples < 5:
            # 샘플 부족하면 기존 가중치 유지 (변경 없음)
            trust_scores[key] = current_weights.get(key, 1 / len(AGENT_KEYS))
            continue

        # 신뢰도 = 정확도 * (1 + 상관계수) / 2
        # 정확도 50% + 상관계수 0이면 신뢰도 0.25 (기본)
        # 정확도 80% + 상관계수 0.5이면 신뢰도 0.60 (우수)
        trust = accuracy * (1 + max(correlation, 0)) / 2
        trust_scores[key] = trust

    # 2. 신뢰도 점수를 정규화 (합=1.0)
    total_trust = sum(trust_scores.values())
    if total_trust == 0:
        return current_weights

    ideal_weights = {k: v / total_trust for k, v in trust_scores.items()}

    # 3. 기존 가중치와 블렌딩 (급격한 변화 방지)
    new_weights = {}
    for key in AGENT_KEYS:
        current = current_weights.get(key, 1 / len(AGENT_KEYS))
        ideal = ideal_weights[key]
        blended = current * (1 - LEARNING_RATE) + ideal * LEARNING_RATE

        # 범위 제한
        blended = max(MIN_WEIGHT, min(MAX_WEIGHT, blended))
        new_weights[key] = blended

    # 4. 합이 1.0이 되도록 재정규화
    total = sum(new_weights.values())
    new_weights = {k: round(v / total, 4) for k, v in new_weights.items()}

    # 반올림 오차 보정
    diff = 1.0 - sum(new_weights.values())
    if diff != 0:
        max_key = max(new_weights, key=new_weights.get)
        new_weights[max_key] = round(new_weights[max_key] + diff, 4)

    return new_weights


def tune_weights(eval_days: int = 10) -> dict:
    """가중치 튜닝 실행."""
    print(f"=== 가중치 자동 튜닝 ({eval_days}일 기준) ===\n")

    # 1. 백테스트 평가
    print("1. 백테스트 평가 중...")
    eval_result = evaluate_all(eval_days)

    if not eval_result or eval_result.get("total_signals", 0) == 0:
        print("   평가할 데이터가 부족합니다. (최소 2주 이상의 history 필요)")
        return {}

    agent_accuracy = eval_result.get("agent_accuracy", {})
    print(f"   총 {eval_result['total_signals']}건 평가 완료")
    print(f"   전체 시그널 정확도: {eval_result['signal_accuracy']:.1%}\n")

    # 2. 기간별 가중치 조정
    tuning_result = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "eval_days": eval_days,
        "signal_accuracy": eval_result["signal_accuracy"],
        "agent_accuracy": agent_accuracy,
        "weight_changes": {},
    }

    for period in ["short", "mid", "long"]:
        current = WEIGHT_TABLE[period]
        new_weights = compute_new_weights(current, agent_accuracy)

        changes = {}
        print(f"--- {period} ---")
        print(f"{'에이전트':<18} {'현재':>8} {'새로운':>8} {'변화':>8}")
        print(f"{'-'*44}")

        for key in AGENT_KEYS:
            old_w = current.get(key, 0)
            new_w = new_weights.get(key, 0)
            diff = new_w - old_w
            changes[key] = {"old": old_w, "new": new_w, "diff": round(diff, 4)}

            marker = " ***" if abs(diff) >= 0.03 else ""
            print(f"{key:<18} {old_w:>7.2%} {new_w:>7.2%} {diff:>+7.2%}{marker}")

        tuning_result["weight_changes"][period] = changes
        print()

    # 3. 결과 저장
    TUNING_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TUNING_HISTORY_DIR / f"{tuning_result['date']}_tuning.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tuning_result, f, ensure_ascii=False, indent=2)
    print(f"튜닝 결과 저장: {output_path}")

    return tuning_result


def apply_weights(tuning_path: str) -> None:
    """튜닝 결과를 settings.py에 반영."""
    with open(tuning_path, "r", encoding="utf-8") as f:
        tuning = json.load(f)

    changes = tuning.get("weight_changes", {})
    if not changes:
        print("적용할 변경 사항이 없습니다.")
        return

    # settings.py 업데이트
    settings_path = Path("config/settings.py")
    content = settings_path.read_text(encoding="utf-8")

    for period in ["short", "mid", "long"]:
        period_changes = changes.get(period, {})
        new_weights = {k: v["new"] for k, v in period_changes.items()}

        # 가중치 문자열 생성
        parts = [f'"{k}": {v}' for k, v in new_weights.items()]
        new_line = f'    "{period}": {{{", ".join(parts)}}},'

        # 기존 라인 교체
        import re
        pattern = rf'    "{period}":\s*\{{[^}}]+\}},'
        content = re.sub(pattern, new_line, content)

    settings_path.write_text(content, encoding="utf-8")
    print(f"settings.py 업데이트 완료")
    print(f"새 가중치:")
    for period in ["short", "mid", "long"]:
        period_changes = changes.get(period, {})
        weights = {k: v["new"] for k, v in period_changes.items()}
        print(f"  {period}: {weights}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Weight Tuner")
    parser.add_argument("command", choices=["tune", "apply"], help="tune=분석, apply=적용")
    parser.add_argument("--days", type=int, default=10, help="평가 기간")
    parser.add_argument("--file", help="apply할 튜닝 결과 파일 경로")
    args = parser.parse_args()

    if args.command == "tune":
        tune_weights(args.days)
    elif args.command == "apply":
        if not args.file:
            # 가장 최근 튜닝 파일 자동 선택
            files = sorted(TUNING_HISTORY_DIR.glob("*_tuning.json"))
            if not files:
                print("튜닝 결과 파일이 없습니다. 먼저 tune을 실행하세요.")
            else:
                apply_weights(str(files[-1]))
        else:
            apply_weights(args.file)
