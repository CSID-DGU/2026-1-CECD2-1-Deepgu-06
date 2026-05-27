"""
weak_fight 3개 영상(fight_0948, fight_0002, fight_0183)에 대해
overlap030_focal vs overlap051_e20(baseline) 클립 스코어 비교.
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.fast_stage import score_clips_fast
from utils.video import load_video_frames
from utils.config import load_config

WEAK_FIGHT_IDS = ["fight_0948", "fight_0002", "fight_0183"]
THRESHOLDS = [0.35, 0.40, 0.45, 0.47]

FOCAL_SCORES_JSON = PROJECT_ROOT / "outputs/train/overlap030_focal/test_scores.json"
BASELINE_CKPT = PROJECT_ROOT / "outputs/train/overlap051_e20/x3d_s_fast_model_best.pt"
TEST_CSV = PROJECT_ROOT / "data/manifests/cctv_x3d_s_overlap030/testing_clips.csv"
BASE_CONFIG = PROJECT_ROOT / "configs/base.yaml"


def load_focal_scores():
    import json
    with open(FOCAL_SCORES_JSON) as f:
        data = json.load(f)
    rows = {r["clip_id"]: r for r in data["rows"]}
    return rows


def run_baseline_inference(df_weak):
    config = load_config(str(BASE_CONFIG))
    config["fast_model"]["checkpoint_path"] = str(BASELINE_CKPT)

    results = {}
    for video_id, group in df_weak.groupby("video_id", sort=False):
        print(f"  [baseline inference] {video_id} ({len(group)} clips)...")
        frames, fps = load_video_frames(group.iloc[0]["video_path"])
        clips = []
        clip_ids = []
        for _, row in group.iterrows():
            clip_frames = frames[int(row["start_frame"]): int(row["end_frame"]) + 1]
            clips.append({
                "clip_id": int(len(clips)),
                "start_frame": int(row["start_frame"]),
                "end_frame": int(row["end_frame"]),
                "start_time": float(row["start_time"]),
                "end_time": float(row["end_time"]),
                "frames": clip_frames,
            })
            clip_ids.append(row["clip_id"])

        scored = score_clips_fast(clips, config["fast_model"], config["clip"])
        for cid, s in zip(clip_ids, scored):
            results[cid] = float(s["fighting_prob"])

    return results


def threshold_recall(probs, thr):
    if not probs:
        return 0, 0
    above = sum(1 for p in probs if p >= thr)
    return above, len(probs)


def print_divider(char="-", width=72):
    print(char * width)


def main():
    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

    print("=" * 72)
    print("  weak_fight 클립 스코어 비교: overlap030_focal  vs  overlap051_e20")
    print("=" * 72)

    # overlap030 test manifest에서 weak_fight 클립만 추출
    df = pd.read_csv(TEST_CSV)
    df_weak = df[df["video_id"].isin(WEAK_FIGHT_IDS)].copy()
    print(f"\n대상 클립: {len(df_weak)}개 ({', '.join(WEAK_FIGHT_IDS)})\n")

    # Step 1: focal scores 로드
    focal_rows = load_focal_scores()

    # Step 2: baseline inference
    print("[1/2] baseline(overlap051_e20) inference 실행 중...")
    baseline_scores = run_baseline_inference(df_weak)
    print()

    # Step 3: 영상별 분석
    print("[2/2] 분석 결과\n")

    for vid in WEAK_FIGHT_IDS:
        group = df_weak[df_weak["video_id"] == vid]
        pos_clips = group[group["label"] == 1]
        neg_clips = group[group["label"] == 0]

        focal_pos = [focal_rows[r]["fighting_prob"] for r in pos_clips["clip_id"] if r in focal_rows]
        base_pos  = [baseline_scores[r] for r in pos_clips["clip_id"] if r in baseline_scores]
        focal_neg = [focal_rows[r]["fighting_prob"] for r in neg_clips["clip_id"] if r in focal_rows]
        base_neg  = [baseline_scores[r] for r in neg_clips["clip_id"] if r in baseline_scores]

        print_divider("=")
        print(f"  {vid}  (positive: {len(pos_clips)}, negative: {len(neg_clips)})")
        print_divider("=")

        # 요약 비교
        def fmt_mean(lst):
            return f"{sum(lst)/len(lst):.3f}" if lst else "N/A"

        print(f"  {'':30s}  {'baseline':>10}  {'focal':>10}")
        print(f"  {'pos mean score':30s}  {fmt_mean(base_pos):>10}  {fmt_mean(focal_pos):>10}")
        print(f"  {'neg mean score':30s}  {fmt_mean(base_neg):>10}  {fmt_mean(focal_neg):>10}")
        print()

        # threshold sensitivity
        print(f"  {'threshold':>10}  {'base above/total':>18}  {'focal above/total':>18}")
        print_divider()
        for thr in THRESHOLDS:
            b_above, b_tot = threshold_recall(base_pos, thr)
            f_above, f_tot = threshold_recall(focal_pos, thr)
            marker = " <-- current" if thr == 0.47 else ""
            print(f"  {thr:>10.2f}  {b_above:>5}/{b_tot:<12}  {f_above:>5}/{f_tot:<12}{marker}")
        print()

        # 클립별 상세 (positive 클립만)
        print(f"  [positive 클립별 스코어]  (start_time / label / baseline / focal)")
        print_divider()
        for _, row in pos_clips.sort_values("start_time").iterrows():
            cid = row["clip_id"]
            bs = baseline_scores.get(cid, float("nan"))
            fs = focal_rows[cid]["fighting_prob"] if cid in focal_rows else float("nan")
            flag_b = " *" if bs >= 0.47 else "  "
            flag_f = " *" if fs >= 0.47 else "  "
            print(f"  t={row['start_time']:6.1f}s  label={int(row['label'])}  "
                  f"base={bs:.3f}{flag_b}  focal={fs:.3f}{flag_f}")
        print()

    print_divider("=")
    print("  * = 0.47(start_score) 이상")
    print_divider("=")


if __name__ == "__main__":
    main()
