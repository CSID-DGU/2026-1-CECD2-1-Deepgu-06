"""
High-score FP 분석: overlap030_focal에서 0.60 이상 스코어를 받은 negative 클립을
영상별로 분석해 어디서 FP가 집중되는지 파악.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FOCAL_JSON = PROJECT_ROOT / "outputs/train/overlap030_focal/test_scores.json"
FULLTEST_JSON = PROJECT_ROOT / "outputs/eval/fulltest_overlap030_focal_fastonly.json"

THRESHOLDS = {"start_score": 0.47, "router_high": 0.60, "very_high": 0.70}


def print_divider(char="-", width=72):
    print(char * width)


def main():
    with open(FOCAL_JSON) as f:
        score_data = json.load(f)

    with open(FULLTEST_JSON) as f:
        eval_data = json.load(f)

    # event-level FP per video
    event_fp = {v["video_id"]: v["false_positive"] for v in eval_data["per_video"]}

    rows = score_data["rows"]
    neg_rows = [r for r in rows if r["label"] == 0]
    pos_rows = [r for r in rows if r["label"] == 1]

    # 전체 요약
    print("=" * 72)
    print("  High-score FP 분석 (overlap030_focal, test split)")
    print("=" * 72)
    total_neg = len(neg_rows)
    for name, thr in THRESHOLDS.items():
        cnt = sum(1 for r in neg_rows if r["fighting_prob"] >= thr)
        print(f"  negative >= {thr} ({name}): {cnt}/{total_neg} ({cnt/total_neg*100:.1f}%)")

    # 영상별 집계
    vid_stats = defaultdict(lambda: {"neg": [], "pos": [], "high_neg": [], "very_high_neg": []})
    for r in neg_rows:
        vid = r["video_id"]
        vid_stats[vid]["neg"].append(r["fighting_prob"])
        if r["fighting_prob"] >= THRESHOLDS["router_high"]:
            vid_stats[vid]["high_neg"].append(r["fighting_prob"])
        if r["fighting_prob"] >= THRESHOLDS["very_high"]:
            vid_stats[vid]["very_high_neg"].append(r["fighting_prob"])
    for r in pos_rows:
        vid_stats[r["video_id"]]["pos"].append(r["fighting_prob"])

    # 영상별 표: high_neg 많은 순 정렬
    records = []
    for vid, s in vid_stats.items():
        records.append({
            "video_id": vid,
            "neg_total": len(s["neg"]),
            "pos_total": len(s["pos"]),
            "high_neg(>=0.60)": len(s["high_neg"]),
            "very_high_neg(>=0.70)": len(s["very_high_neg"]),
            "neg_mean": round(sum(s["neg"]) / len(s["neg"]), 3) if s["neg"] else 0,
            "event_fp": event_fp.get(vid, 0),
        })

    df = pd.DataFrame(records).sort_values("high_neg(>=0.60)", ascending=False)

    print(f"\n{'영상별 고스코어 negative 클립 (high_neg >= 0.60 내림차순)'}")
    print_divider()
    print(f"  {'video_id':<14} {'neg_tot':>7} {'pos_tot':>7} {'>=0.60':>7} {'>=0.70':>7} "
          f"{'neg_mean':>8} {'event_fp':>8}")
    print_divider()
    for _, row in df.iterrows():
        flag = " ***" if row["high_neg(>=0.60)"] >= 20 else (" **" if row["high_neg(>=0.60)"] >= 10 else "")
        print(f"  {row['video_id']:<14} {row['neg_total']:>7} {row['pos_total']:>7} "
              f"{row['high_neg(>=0.60)']:>7} {row['very_high_neg(>=0.70)']:>7} "
              f"{row['neg_mean']:>8.3f} {row['event_fp']:>8}{flag}")

    # 상위 FP 영상 요약
    top = df[df["high_neg(>=0.60)"] >= 10]
    print(f"\n  *** high_neg >= 20개,  ** >= 10개")
    print(f"\n  [high_neg >= 10인 영상: {len(top)}개]")
    print(f"  이 영상들의 high_neg 합계: {top['high_neg(>=0.60)'].sum()} / {df['high_neg(>=0.60)'].sum()} "
          f"({top['high_neg(>=0.60)'].sum()/df['high_neg(>=0.60)'].sum()*100:.1f}%)")

    # 집중도 분포
    print(f"\n  [high_neg 분포]")
    print_divider()
    buckets = [(0, 0), (1, 4), (5, 9), (10, 19), (20, 999)]
    for lo, hi in buckets:
        label = f"{lo}~{hi}" if hi < 999 else f"{lo}+"
        vids = df[(df["high_neg(>=0.60)"] >= lo) & (df["high_neg(>=0.60)"] <= hi)]
        print(f"  {label:>8}개 영상: {len(vids):>3}개")


if __name__ == "__main__":
    main()
