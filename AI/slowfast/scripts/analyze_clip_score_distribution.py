import argparse
import csv
import json
import math
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_float(value, default=None):
    if value is None or value == "":
        return default
    return float(value)


def quantile(sorted_values, q):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return float(sorted_values[low])
    frac = pos - low
    return float(sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac)


def summarize(values):
    if not values:
        return {"count": 0}
    values = [float(v) for v in values]
    values_sorted = sorted(values)
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return {
        "count": len(values),
        "mean": mean,
        "std": math.sqrt(var),
        "min": values_sorted[0],
        "p10": quantile(values_sorted, 0.10),
        "p25": quantile(values_sorted, 0.25),
        "p50": quantile(values_sorted, 0.50),
        "p75": quantile(values_sorted, 0.75),
        "p90": quantile(values_sorted, 0.90),
        "p95": quantile(values_sorted, 0.95),
        "max": values_sorted[-1],
    }


def interval_overlap(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def tag_clips_by_intervals(clips, interval_rows, start_key, end_key):
    tagged = set()
    for row in interval_rows:
        video_id = row["video_id"]
        interval_start = safe_float(row[start_key])
        interval_end = safe_float(row[end_key])
        if interval_start is None or interval_end is None:
            continue
        for clip_id, clip in clips.items():
            if clip["video_id"] != video_id:
                continue
            if interval_overlap(clip["start_time"], clip["end_time"], interval_start, interval_end) > 0.0:
                tagged.add(clip_id)
    return tagged


def build_payload(rows, group_ids):
    probs = [rows[clip_id]["fighting_prob"] for clip_id in group_ids]
    entropies = [rows[clip_id]["entropy"] for clip_id in group_ids]
    return {
        "prob_summary": summarize(probs),
        "entropy_summary": summarize(entropies),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-json", required=True)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--missed-gt-csv", required=True)
    parser.add_argument("--matched-gt-csv", required=True)
    parser.add_argument("--false-positive-csv", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    score_payload = load_json(args.score_json)
    manifest_rows = load_csv_rows(args.manifest_csv)
    missed_rows = load_csv_rows(args.missed_gt_csv)
    matched_rows = load_csv_rows(args.matched_gt_csv)
    fp_rows = load_csv_rows(args.false_positive_csv)

    score_rows = {}
    for row in score_payload["rows"]:
        score_rows[row["clip_id"]] = {
            "clip_id": row["clip_id"],
            "video_id": row["video_id"],
            "label": int(row["label"]),
            "fighting_prob": float(row["fighting_prob"]),
            "entropy": float(row["entropy"]),
        }

    clips = {}
    for row in manifest_rows:
        clip_id = row["clip_id"]
        if clip_id not in score_rows:
            continue
        clips[clip_id] = {
            "clip_id": clip_id,
            "video_id": row["video_id"],
            "start_time": float(row["start_time"]),
            "end_time": float(row["end_time"]),
            "label": int(row["label"]),
        }

    positive_ids = {clip_id for clip_id, row in score_rows.items() if row["label"] == 1}
    negative_ids = {clip_id for clip_id, row in score_rows.items() if row["label"] == 0}

    missed_gt_clip_ids = tag_clips_by_intervals(clips, missed_rows, "gt_start", "gt_end")
    matched_gt_clip_ids = tag_clips_by_intervals(clips, matched_rows, "gt_start", "gt_end")
    fp_event_clip_ids = tag_clips_by_intervals(clips, fp_rows, "pred_start", "pred_end")

    groups = {
        "all_positive": positive_ids,
        "all_negative": negative_ids,
        "missed_gt_overlap_clips": missed_gt_clip_ids,
        "matched_gt_overlap_clips": matched_gt_clip_ids,
        "false_positive_event_overlap_clips": fp_event_clip_ids,
        "missed_positive_clips": missed_gt_clip_ids & positive_ids,
        "matched_positive_clips": matched_gt_clip_ids & positive_ids,
        "fp_negative_clips": fp_event_clip_ids & negative_ids,
        "fp_positive_clips": fp_event_clip_ids & positive_ids,
    }

    payload = {
        "score_json": args.score_json,
        "manifest_csv": args.manifest_csv,
        "missed_gt_csv": args.missed_gt_csv,
        "matched_gt_csv": args.matched_gt_csv,
        "false_positive_csv": args.false_positive_csv,
        "groups": {
            group_name: build_payload(score_rows, clip_ids)
            for group_name, clip_ids in groups.items()
        },
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    for group_name, group_payload in payload["groups"].items():
        print(group_name, json.dumps(group_payload["prob_summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
