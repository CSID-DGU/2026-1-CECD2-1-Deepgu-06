"""
split threshold (0.33 / 0.40 / 0.45) 별 event generation 결과 비교.
재inference 없이 service_scope_clip_scores.json 사용.
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

CLIP_SCORES_JSON   = PROJECT_ROOT / "outputs/eval/service_scope_clip_scores.json"
CLIPS_CSV          = PROJECT_ROOT / "data/manifests/cctv_x3d_s_overlap030/testing_clips_service_scope.csv"
SERVICE_SCOPE_JSON = PROJECT_ROOT / "data/manifests/test_service_scope.json"
GT_JSON            = Path("/home/deepgu/test/cctv/dataset/ground-truth.json")
OUTPUT_JSON        = PROJECT_ROOT / "outputs/eval/split_threshold_sweep.json"

SPLIT_THRESHOLDS = [0.33, 0.40, 0.45]
START_SCORE  = 0.47
END_SCORE    = 0.42
IOT_THRESH   = 0.1
FOCUS_VIDEOS = ["fight_0920", "fight_0048", "fight_0066"]


def sfr(s, e, fps, nb):
    sf = max(0, int(math.floor(s * fps)))
    ef = min(int(nb) - 1, int(math.ceil(e * fps)) - 1)
    return sf, max(sf, ef)


def iou(a0, a1, b0, b1):
    inter = max(0, min(a1, b1) - max(a0, b0) + 1)
    union = (a1 - a0 + 1) + (b1 - b0 + 1) - inter
    return inter / union if union > 0 else 0.0


def evaluate(pred_events, gt_events):
    matched = set()
    tp = 0
    for pred in pred_events:
        for gi, gt in enumerate(gt_events):
            if gi in matched:
                continue
            if iou(pred["start_frame"], pred["end_frame"],
                   gt["start_frame"],  gt["end_frame"]) >= IOT_THRESH:
                tp += 1
                matched.add(gi)
                break
    fp = max(0, len(pred_events) - tp)
    fn = max(0, len(gt_events) - tp)
    return tp, fp, fn


# ── 데이터 로드 ──────────────────────────────────────────────
service_ids = set(json.loads(SERVICE_SCOPE_JSON.read_text())["video_ids"])
score_map   = {r["clip_id"]: r["fighting_prob"]
               for r in json.loads(CLIP_SCORES_JSON.read_text())["rows"]}

df = pd.read_csv(CLIPS_CSV)
df = df[df["video_id"].isin(service_ids)].copy()
video_meta = df.groupby("video_id").first()[["fps", "nb_frames"]].to_dict("index")

video_clips: dict[str, list[dict]] = {}
for video_id, grp in df.sort_values(["video_id", "start_frame"]).groupby("video_id"):
    clips = []
    for idx, (_, row) in enumerate(grp.iterrows()):
        clips.append({
            "clip_id":     idx,
            "start_frame": int(row["start_frame"]),
            "end_frame":   int(row["end_frame"]),
            "start_time":  float(row["start_time"]),
            "end_time":    float(row["end_time"]),
            "final_score": score_map.get(row["clip_id"], 0.0),
        })
    video_clips[video_id] = clips

gt_db = json.loads(GT_JSON.read_text())["database"]
video_gt: dict[str, list[dict]] = {}
for video_id in service_ids:
    meta = gt_db.get(video_id, {})
    fps, nb = float(video_meta[video_id]["fps"]), int(video_meta[video_id]["nb_frames"])
    gts = []
    for ann in meta.get("annotations", []):
        s, e = ann["segment"]
        sf, ef = sfr(s, e, fps, nb)
        gts.append({"start_frame": sf, "end_frame": ef, "start_time": s, "end_time": e})
    video_gt[video_id] = gts

# ── sweep ────────────────────────────────────────────────────
all_results = {}

for split_thresh in SPLIT_THRESHOLDS:
    thresholds = {
        "start_score": START_SCORE, "end_score": END_SCORE,
        "min_event_duration_sec": 0.5,
        "split": {"enabled": True, "score_threshold": split_thresh,
                  "min_consecutive_clips": 2},
        "score_smoothing": {"enabled": True, "window_size": 3, "method": "moving_average"},
    }

    total_tp = total_fp = total_fn = 0
    total_events = 0
    durations = []
    long_events = []  # >= 30s

    focus_info: dict[str, list[dict]] = {v: [] for v in FOCUS_VIDEOS}

    for video_id in sorted(service_ids):
        fps = float(video_meta[video_id]["fps"])
        clips = video_clips[video_id]
        gt    = video_gt[video_id]

        pred_events, _ = build_events(clips, thresholds, fps)
        tp, fp, fn = evaluate(pred_events, gt)
        total_tp += tp; total_fp += fp; total_fn += fn
        total_events += len(pred_events)

        for pred in pred_events:
            durations.append(pred["duration_sec"])
            if pred["duration_sec"] >= 30:
                long_events.append({"video_id": video_id, **pred})

        if video_id in FOCUS_VIDEOS:
            focus_info[video_id] = [
                {"start": round(p["start_time"], 2),
                 "end":   round(p["end_time"],   2),
                 "dur":   round(p["duration_sec"], 1),
                 "peak":  round(p["peak_score"], 3)}
                for p in pred_events
            ]

    prec = total_tp / max(1, total_tp + total_fp)
    rec  = total_tp / max(1, total_tp + total_fn)
    f1   = 2 * prec * rec / max(prec + rec, 1e-12)
    avg_dur = sum(durations) / max(1, len(durations))

    all_results[split_thresh] = {
        "split_threshold": split_thresh,
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
        "total_events": total_events,
        "avg_duration": round(avg_dur, 2),
        "long_events_count": len(long_events),
        "focus": focus_info,
    }

# ── 출력 ────────────────────────────────────────────────────
print("=" * 70)
print("split threshold sweep  (start=0.47, end=0.42, IoU>=0.10)")
print("=" * 70)

header = f"{'split':>7} {'TP':>5} {'FP':>5} {'FN':>5} {'Prec':>7} {'Rec':>7} {'F1':>7} " \
         f"{'#events':>8} {'avg_dur':>8} {'>=30s':>6}"
print(header)
print("-" * 70)
for st in SPLIT_THRESHOLDS:
    r = all_results[st]
    print(f"{r['split_threshold']:>7.2f} {r['tp']:>5} {r['fp']:>5} {r['fn']:>5} "
          f"{r['precision']:>7.4f} {r['recall']:>7.4f} {r['f1']:>7.4f} "
          f"{r['total_events']:>8} {r['avg_duration']:>8.2f}s {r['long_events_count']:>6}")

# ── focus 비디오 변화 ────────────────────────────────────────
print()
for video_id in FOCUS_VIDEOS:
    gt_list = video_gt[video_id]
    gt_str  = "  ".join(f"{g['start_time']:.1f}~{g['end_time']:.1f}s" for g in gt_list)
    print(f"\n[ {video_id} ]  GT({len(gt_list)}개): {gt_str}")
    print(f"  {'split':>6} | 예측 이벤트")
    print(f"  ------+----------------------------------------------")
    for st in SPLIT_THRESHOLDS:
        events = all_results[st]["focus"][video_id]
        ev_str = "  ".join(f"{e['start']}~{e['end']}s({e['dur']}s,pk={e['peak']})"
                           for e in events)
        print(f"  {st:>6.2f} | {ev_str}")

# ── 저장 ────────────────────────────────────────────────────
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
write_json(OUTPUT_JSON, {k: {kk: vv for kk, vv in v.items() if kk != "focus"}
                          for k, v in all_results.items()})
print(f"\n[saved] {OUTPUT_JSON}")
