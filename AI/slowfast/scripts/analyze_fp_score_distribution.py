"""
FP clip score distribution 분석.
test_scores.json의 negative(label=0) 클립 스코어 분포 확인.
- FP 이벤트는 negative 클립이 threshold 이상 스코어를 받아 형성되므로
  negative 클립 score 분포가 FP 원인의 직접 지표.
"""
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FOCAL_JSON  = PROJECT_ROOT / "outputs/train/overlap030_focal/test_scores.json"
BASELINE_JSON = PROJECT_ROOT / "outputs/train/overlap051_e20/val_scores.json"

START_SCORE = 0.47
ROUTER_HIGH = 0.60
BINS = [0.0, 0.20, 0.33, 0.40, 0.47, 0.60, 0.70, 0.80, 0.90, 1.01]
BIN_LABELS = ["0.00~0.20","0.20~0.33","0.33~0.40","0.40~0.47",
               "0.47~0.60","0.60~0.70","0.70~0.80","0.80~0.90","0.90~1.00"]


def bin_scores(scores):
    counts = [0] * len(BIN_LABELS)
    for s in scores:
        for i in range(len(BINS) - 1):
            if BINS[i] <= s < BINS[i + 1]:
                counts[i] += 1
                break
    return counts


def print_divider(char="-", width=68):
    print(char * width)


def analyze(label, json_path):
    with open(json_path) as f:
        data = json.load(f)
    rows = data["rows"]

    neg = [r["fighting_prob"] for r in rows if r["label"] == 0]
    pos = [r["fighting_prob"] for r in rows if r["label"] == 1]

    print(f"\n{'='*68}")
    print(f"  {label}")
    print(f"  총 클립: {len(rows)}  |  negative: {len(neg)}  |  positive: {len(pos)}")
    print(f"{'='*68}")

    for name, scores in [("NEGATIVE (label=0)", neg), ("POSITIVE (label=1)", pos)]:
        if not scores:
            continue
        total = len(scores)
        above_start = sum(1 for s in scores if s >= START_SCORE)
        gray_zone   = sum(1 for s in scores if START_SCORE <= s < ROUTER_HIGH)
        above_high  = sum(1 for s in scores if s >= ROUTER_HIGH)

        print(f"\n  [{name}]  n={total}")
        print_divider()
        print(f"  mean={sum(scores)/total:.3f}  "
              f"min={min(scores):.3f}  max={max(scores):.3f}")
        print()

        # 핵심 구간 비율
        print(f"  {'구간':<20} {'count':>7}  {'비율':>7}  {'누적':>7}")
        print_divider()
        cumulative = 0
        counts = bin_scores(scores)
        for bl, cnt in zip(BIN_LABELS, counts):
            cumulative += cnt
            marker = ""
            if bl == "0.47~0.60":
                marker = "  ← gray zone (VLM 라우팅)"
            elif bl == "0.60~0.70":
                marker = "  ← router_high 이상 (VLM 스킵)"
            print(f"  {bl:<20} {cnt:>7}  {cnt/total*100:>6.1f}%  {cumulative/total*100:>6.1f}%{marker}")

        print()
        print(f"  >= start_score(0.47) : {above_start:>5} / {total}  ({above_start/total*100:.1f}%)")
        print(f"  gray zone(0.47~0.60) : {gray_zone:>5} / {total}  ({gray_zone/total*100:.1f}%)")
        print(f"  >= router_high(0.60) : {above_high:>5} / {total}  ({above_high/total*100:.1f}%)  ← VLM 못 봄")


def main():
    analyze("overlap030_focal  (test split)", FOCAL_JSON)

    # baseline test_scores 없으면 val로 대체해서 참고용으로만
    baseline_test = PROJECT_ROOT / "outputs/train/overlap051_e20/test_scores.json"
    if baseline_test.exists():
        analyze("overlap051_e20 baseline  (test split)", baseline_test)
    else:
        print("\n[baseline test_scores.json 없음 — val_scores로 참고 출력]")
        analyze("overlap051_e20 baseline  (val split, 참고용)", BASELINE_JSON)


if __name__ == "__main__":
    main()
