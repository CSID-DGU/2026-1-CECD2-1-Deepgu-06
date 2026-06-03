"""
서비스 범위(61개) 기준 event-level threshold sweep.

clip-level 점수(service_scope_clip_scores.json)를 이미 가지고 있으므로
재inference 없이 start_score/end_score 조합별 TP/FP/FN/F1을 계산한다.
"""

import json
import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.event_builder import build_events
from utils.io import write_json

# ──────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────
CLIP_SCORES_JSON = PROJECT_ROOT / "outputs/eval/service_scope_clip_scores.json"
CLIPS_CSV = PROJECT_ROOT / "data/manifests/cctv_x3d_s_overlap030/testing_clips_service_scope.csv"
SERVICE_SCOPE_JSON = PROJECT_ROOT / "data/manifests/test_service_scope.json"
GT_JSON = Path("/home/deepgu/test/cctv/dataset/ground-truth.json")
OUTPUT_JSON = PROJECT_ROOT / "outputs/eval/threshold_sweep_service_scope.json"

IOT_THRESHOLD = 0.1

# sweep 범위
START_SCORES = [0.35, 0.38, 0.40, 0.42, 0.44, 0.47, 0.50, 0.53, 0.55]
END_SCORES   = [0.28, 0.30, 0.33, 0.35, 0.38, 0.40, 0.42, 0.44, 0.47]

# 현재 설정
CURRENT_START = 0.47
CURRENT_END   = 0.42

# ──────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────
def interval_iou(a_start, a_end, b_start, b_end):
    left = max(a_start, b_start)
    right = min(a_end, b_end)
    intersection = max(0, right - left + 1)
    union = (a_end - a_start + 1) + (b_end - b_start + 1) - intersection
    return intersection / union if union > 0 else 0.0


def seconds_to_frame_range(start_sec, end_sec, fps, nb_frames):
    start_frame = max(0, int(math.floor(start_sec * fps)))
    end_frame   = min(int(nb_frames) - 1, int(math.ceil(end_sec * fps)) - 1)
    return start_frame, max(start_frame, end_frame)


def evaluate_events(pred_events, gt_events, iou_thresh=IOT_THRESHOLD):
    # GT-centric: 각 GT가 임의의 pred와 매칭되면 TP
    tp = sum(
        1 for gt in gt_events
        if any(interval_iou(p["start_frame"], p["end_frame"],
                            gt["start_frame"], gt["end_frame"]) >= iou_thresh
               for p in pred_events)
    )
    fn = max(0, len(gt_events) - tp)
    tp_pred = sum(
        1 for p in pred_events
        if any(interval_iou(p["start_frame"], p["end_frame"],
                            gt["start_frame"], gt["end_frame"]) >= iou_thresh
               for gt in gt_events)
    )
    fp = max(0, len(pred_events) - tp_pred)
    precision = tp_pred / max(1, len(pred_events))
    recall    = tp / max(1, tp + fn)
    f1        = 2 * precision * recall / max(precision + recall, 1e-12)
    return tp, fp, fn, precision, recall, f1


# ──────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────
print("Loading data...", flush=True)

service_ids = set(json.loads(SERVICE_SCOPE_JSON.read_text())["video_ids"])

# clip scores
clip_score_rows = json.loads(CLIP_SCORES_JSON.read_text())["rows"]
score_map = {row["clip_id"]: row["fighting_prob"] for row in clip_score_rows}

# CSV (프레임 정보)
df = pd.read_csv(CLIPS_CSV)
df = df[df["video_id"].isin(service_ids)].copy()

# video별 fps/nb_frames
video_meta = (
    df.groupby("video_id")
      .first()[["fps", "nb_frames"]]
      .to_dict("index")
)

# 비디오별 clip 목록 (순서대로 integer clip_id 부여)
video_clips: dict[str, list[dict]] = {}
for video_id, grp in df.sort_values(["video_id", "start_frame"]).groupby("video_id"):
    clips = []
    for idx, (_, row) in enumerate(grp.iterrows()):
        prob = score_map.get(row["clip_id"], 0.0)
        clips.append({
            "clip_id":     idx,
            "start_frame": int(row["start_frame"]),
            "end_frame":   int(row["end_frame"]),
            "final_score": prob,
        })
    video_clips[video_id] = clips

# GT
gt_db = json.loads(GT_JSON.read_text())["database"]
video_gt: dict[str, list[dict]] = {}
for video_id in service_ids:
    meta = gt_db.get(video_id, {})
    fps      = float(video_meta[video_id]["fps"])
    nb_frames = int(video_meta[video_id]["nb_frames"])
    gts = []
    for ann in meta.get("annotations", []):
        s, e = ann["segment"]
        sf, ef = seconds_to_frame_range(s, e, fps, nb_frames)
        gts.append({"start_frame": sf, "end_frame": ef,
                    "start_time": s, "end_time": e})
    video_gt[video_id] = gts

# ──────────────────────────────────────────────
# sweep
# ──────────────────────────────────────────────
base_thresholds = {
    "min_event_duration_sec": 0.5,
    "mean_score_threshold": 0.0,
    "split": {"enabled": True, "score_threshold": 0.33, "min_consecutive_clips": 2},
    "score_smoothing": {"enabled": True, "window_size": 3, "method": "moving_average"},
}

results = []
combos = [
    (s, e) for s in START_SCORES for e in END_SCORES if e <= s
]
print(f"Sweeping {len(combos)} combinations...", flush=True)

for start, end in combos:
    thresholds = {**base_thresholds, "start_score": start, "end_score": end}

    total_tp = total_fp = total_fn = total_npred = 0
    for video_id in service_ids:
        clips = video_clips[video_id]
        fps   = float(video_meta[video_id]["fps"])
        gt    = video_gt[video_id]

        pred_events, _ = build_events(clips, thresholds, fps)
        tp, fp, fn, _, _, _ = evaluate_events(pred_events, gt)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_npred += len(pred_events)

    # precision = tp_pred / n_preds (GT-centric 정의 B).  tp_pred = n_preds - fp.
    total_tp_pred = max(0, total_npred - total_fp)
    precision = total_tp_pred / max(1, total_npred)
    recall    = total_tp / max(1, total_tp + total_fn)
    f1        = 2 * precision * recall / max(precision + recall, 1e-12)
    is_current = (start == CURRENT_START and end == CURRENT_END)
    results.append({
        "start_score": start,
        "end_score":   end,
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "current":   is_current,
    })

results_sorted = sorted(results, key=lambda x: -x["f1"])
best = results_sorted[0]
current = next(r for r in results if r["current"])

# ──────────────────────────────────────────────
# 결과 출력
# ──────────────────────────────────────────────
print(f"\n{'start':>7} {'end':>6} {'TP':>5} {'FP':>5} {'FN':>5} {'P':>6} {'R':>6} {'F1':>6}  mark")
print("-" * 58)
for r in results_sorted[:15]:
    mark = "<< best" if r == best else ("<< current" if r["current"] else "")
    print(f"{r['start_score']:>7.2f} {r['end_score']:>6.2f} "
          f"{r['tp']:>5} {r['fp']:>5} {r['fn']:>5} "
          f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f}  {mark}")

if not current["current"] or current not in results_sorted[:15]:
    print(f"\n[current] start={CURRENT_START} end={CURRENT_END}  "
          f"tp={current['tp']} fp={current['fp']} fn={current['fn']}  "
          f"F1={current['f1']:.4f}")

# ──────────────────────────────────────────────
# 최적 threshold에서 high-score 미탐 분석 (fight_0831, fight_0242)
# ──────────────────────────────────────────────
ANALYZE_IDS = ["fight_0831", "fight_0242"]
best_thresh = {**base_thresholds, "start_score": best["start_score"], "end_score": best["end_score"]}

print(f"\n[IoU miss analysis at best threshold start={best['start_score']} end={best['end_score']}]")
iou_analysis = []
for video_id in ANALYZE_IDS:
    if video_id not in service_ids:
        print(f"  {video_id}: not in service scope")
        continue
    clips  = video_clips[video_id]
    fps    = float(video_meta[video_id]["fps"])
    gt     = video_gt[video_id]
    pred_events, scored = build_events(clips, best_thresh, fps)

    print(f"\n  {video_id}:")
    print(f"    GT events: {len(gt)}, Pred events: {len(pred_events)}")
    for gi, g in enumerate(gt):
        best_iou = 0.0
        best_pred = None
        for pred in pred_events:
            iou = interval_iou(pred["start_frame"], pred["end_frame"],
                               g["start_frame"], g["end_frame"])
            if iou > best_iou:
                best_iou = iou
                best_pred = pred
        matched = "TP" if best_iou >= IOT_THRESHOLD else "FN"
        print(f"    GT[{gi}] {g['start_time']:.1f}s~{g['end_time']:.1f}s  "
              f"best_iou={best_iou:.3f}  {matched}", end="")
        if best_pred:
            print(f"  (pred {best_pred['start_time']:.1f}s~{best_pred['end_time']:.1f}s "
                  f"peak={best_pred['peak_score']:.3f})", end="")
        print()
    max_score = max((c["final_score"] for c in clips), default=0.0)
    print(f"    clip max_score={max_score:.3f}")
    iou_analysis.append({
        "video_id": video_id,
        "gt": gt, "pred_events": pred_events,
        "max_clip_score": max_score,
    })

# ──────────────────────────────────────────────
# 저장
# ──────────────────────────────────────────────
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
write_json(OUTPUT_JSON, {
    "sweep_results": results_sorted,
    "best": best,
    "current": current,
    "iou_miss_analysis": {v["video_id"]: v for v in iou_analysis},
})
print(f"\n[saved] {OUTPUT_JSON}")
