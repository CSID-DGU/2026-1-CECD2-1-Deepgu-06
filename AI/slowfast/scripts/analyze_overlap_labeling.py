import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.io import ensure_dir, write_json


BUCKETS = [
    (0.0, 0.1),
    (0.1, 0.2),
    (0.2, 0.3),
    (0.3, 0.4),
    (0.4, 0.5),
    (0.5, 0.6),
    (0.6, 0.7),
    (0.7, 0.8),
    (0.8, 0.9),
    (0.9, 1.01),
]

EVENT_LENGTH_BUCKETS = [
    (0.0, 1.0, "lt_1s"),
    (1.0, 2.0, "1_to_2s"),
    (2.0, 4.0, "2_to_4s"),
    (4.0, 8.0, "4_to_8s"),
    (8.0, float("inf"), "ge_8s"),
]


def interval_overlap(a_start, a_end, b_start, b_end):
    left = max(float(a_start), float(b_start))
    right = min(float(a_end), float(b_end))
    return max(0.0, right - left)


def build_video_path(dataset_root, source, subset, video_id):
    if source == "CCTV":
        return Path(dataset_root) / "CCTV_DATA" / subset / f"{video_id}.mpeg"
    return Path(dataset_root) / "NON_CCTV_DATA" / subset / f"{video_id}.mpeg"


def count_candidate_clips(duration, temporal_window_sec, stride_sec):
    if duration < temporal_window_sec:
        return 0
    max_clip_start = max(0.0, float(duration) - float(temporal_window_sec))
    return int(math.floor(max_clip_start / float(stride_sec) + 1e-9)) + 1


def assign_bucket(max_overlap_ratio, positive_threshold, negative_threshold):
    if max_overlap_ratio >= float(positive_threshold):
        return "positive"
    if max_overlap_ratio <= float(negative_threshold):
        return "negative"
    return "discard"


def find_overlap_bin(value):
    for start, end in BUCKETS:
        if value < end or (end > 1.0 and value <= 1.0):
            return f"{start:.1f}_{min(end, 1.0):.1f}"
    return "unknown"


def find_event_length_bucket(length_sec):
    for start, end, label in EVENT_LENGTH_BUCKETS:
        if start <= length_sec < end:
            return label
    return "unknown"


def analyze(
    ground_truth_path,
    dataset_root,
    source_filter,
    temporal_window_sec,
    stride_sec,
    positive_threshold,
    negative_threshold,
):
    payload = json.load(open(ground_truth_path, "r", encoding="utf-8"))
    database = payload["database"]

    clip_rows = []
    split_counts = defaultdict(Counter)
    discard_overlap_hist = defaultdict(Counter)
    event_bucket_stats = defaultdict(lambda: defaultdict(Counter))
    event_coverage = defaultdict(Counter)

    for video_id, meta in sorted(database.items()):
        source = meta.get("source")
        subset = meta.get("subset")
        if source_filter and source != source_filter:
            continue
        if subset not in {"training", "validation", "testing"}:
            continue

        video_path = build_video_path(dataset_root, source, subset, video_id)
        if not video_path.exists():
            continue

        duration = float(meta["duration"])
        if duration < temporal_window_sec:
            continue

        fps = float(meta["frame_rate"])
        nb_frames = int(meta["nb_frames"])
        annotations = meta.get("annotations", [])
        event_hits = [0] * len(annotations)

        split_counts[subset]["videos"] += 1
        split_counts[subset]["candidate_clips"] += count_candidate_clips(duration, temporal_window_sec, stride_sec)

        clip_index = 0
        max_clip_start = max(0.0, duration - float(temporal_window_sec))
        clip_start_sec = 0.0
        while clip_start_sec <= max_clip_start + 1e-9:
            clip_end_sec = clip_start_sec + float(temporal_window_sec)
            max_overlap_ratio = 0.0
            best_event_index = None

            for event_index, annotation in enumerate(annotations):
                pos_start_sec, pos_end_sec = annotation["segment"]
                overlap_sec = interval_overlap(clip_start_sec, clip_end_sec, pos_start_sec, pos_end_sec)
                window_overlap_ratio = overlap_sec / float(temporal_window_sec)
                if window_overlap_ratio > max_overlap_ratio:
                    max_overlap_ratio = window_overlap_ratio
                    best_event_index = event_index

            bucket = assign_bucket(max_overlap_ratio, positive_threshold, negative_threshold)
            split_counts[subset][bucket] += 1
            if bucket == "discard":
                discard_overlap_hist[subset][find_overlap_bin(max_overlap_ratio)] += 1
            if bucket == "positive" and best_event_index is not None:
                event_hits[best_event_index] += 1

            clip_rows.append(
                {
                    "clip_id": f"{video_id}_{clip_index:04d}",
                    "video_id": video_id,
                    "split": subset,
                    "source": source,
                    "fps": fps,
                    "nb_frames": nb_frames,
                    "start_time": float(clip_start_sec),
                    "end_time": float(clip_end_sec),
                    "max_window_overlap_ratio": float(max_overlap_ratio),
                    "assigned_bucket": bucket,
                    "best_event_index": best_event_index,
                }
            )
            clip_index += 1
            clip_start_sec += float(stride_sec)

        for event_index, annotation in enumerate(annotations):
            start_sec, end_sec = annotation["segment"]
            event_length_sec = max(0.0, float(end_sec) - float(start_sec))
            length_bucket = find_event_length_bucket(event_length_sec)
            event_bucket_stats[subset][length_bucket]["events"] += 1
            event_bucket_stats[subset][length_bucket]["positive_clips"] += int(event_hits[event_index])
            covered = int(event_hits[event_index] > 0)
            event_bucket_stats[subset][length_bucket]["covered_events"] += covered
            event_bucket_stats[subset][length_bucket]["uncovered_events"] += int(not covered)
            event_coverage[subset]["events"] += 1
            event_coverage[subset]["covered_events"] += covered
            event_coverage[subset]["uncovered_events"] += int(not covered)

    split_summary_rows = []
    for subset, counter in sorted(split_counts.items()):
        candidate = int(counter.get("candidate_clips", 0))
        positive = int(counter.get("positive", 0))
        negative = int(counter.get("negative", 0))
        discard = int(counter.get("discard", 0))
        usable = positive + negative
        split_summary_rows.append(
            {
                "split": subset,
                "videos": int(counter.get("videos", 0)),
                "candidate_clips": candidate,
                "positive": positive,
                "negative": negative,
                "discard": discard,
                "usable": usable,
                "positive_ratio": (positive / candidate) if candidate else 0.0,
                "negative_ratio": (negative / candidate) if candidate else 0.0,
                "discard_ratio": (discard / candidate) if candidate else 0.0,
            }
        )

    discard_rows = []
    for subset, counter in sorted(discard_overlap_hist.items()):
        total = sum(counter.values())
        for overlap_bin, count in sorted(counter.items()):
            discard_rows.append(
                {
                    "split": subset,
                    "overlap_bin": overlap_bin,
                    "count": int(count),
                    "ratio_within_discard": (count / total) if total else 0.0,
                }
            )

    event_length_rows = []
    for subset, bucket_map in sorted(event_bucket_stats.items()):
        for length_bucket, counter in sorted(bucket_map.items()):
            events = int(counter.get("events", 0))
            event_length_rows.append(
                {
                    "split": subset,
                    "event_length_bucket": length_bucket,
                    "events": events,
                    "covered_events": int(counter.get("covered_events", 0)),
                    "uncovered_events": int(counter.get("uncovered_events", 0)),
                    "positive_clips": int(counter.get("positive_clips", 0)),
                    "coverage_ratio": (counter.get("covered_events", 0) / events) if events else 0.0,
                    "avg_positive_clips_per_event": (counter.get("positive_clips", 0) / events) if events else 0.0,
                }
            )

    coverage_rows = []
    for subset, counter in sorted(event_coverage.items()):
        events = int(counter.get("events", 0))
        coverage_rows.append(
            {
                "split": subset,
                "events": events,
                "covered_events": int(counter.get("covered_events", 0)),
                "uncovered_events": int(counter.get("uncovered_events", 0)),
                "coverage_ratio": (counter.get("covered_events", 0) / events) if events else 0.0,
            }
        )

    return {
        "clip_rows": clip_rows,
        "split_summary_rows": split_summary_rows,
        "discard_rows": discard_rows,
        "event_length_rows": event_length_rows,
        "coverage_rows": coverage_rows,
        "summary": {
            "source_filter": source_filter,
            "temporal_window_sec": float(temporal_window_sec),
            "stride_sec": float(stride_sec),
            "positive_threshold": float(positive_threshold),
            "negative_threshold": float(negative_threshold),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth-json", default="/home/deepgu/test/cctv/dataset/ground-truth.json")
    parser.add_argument("--dataset-root", default="/home/deepgu/test/cctv/dataset")
    parser.add_argument("--output-dir", default="/home/deepgu/slowfast/outputs/analysis/overlap051")
    parser.add_argument("--source-filter", default="CCTV")
    parser.add_argument("--temporal-window-sec", type=float, default=2.0)
    parser.add_argument("--stride-sec", type=float, default=1.0)
    parser.add_argument("--positive-threshold", type=float, default=0.5)
    parser.add_argument("--negative-threshold", type=float, default=0.1)
    args = parser.parse_args()

    if float(args.negative_threshold) > float(args.positive_threshold):
        raise ValueError("negative-threshold must be <= positive-threshold")

    output_dir = ensure_dir(args.output_dir)
    payload = analyze(
        ground_truth_path=args.ground_truth_json,
        dataset_root=args.dataset_root,
        source_filter=args.source_filter,
        temporal_window_sec=args.temporal_window_sec,
        stride_sec=args.stride_sec,
        positive_threshold=args.positive_threshold,
        negative_threshold=args.negative_threshold,
    )

    pd.DataFrame(payload["clip_rows"]).to_csv(output_dir / "clip_overlap_analysis.csv", index=False)
    pd.DataFrame(payload["split_summary_rows"]).to_csv(output_dir / "split_summary.csv", index=False)
    pd.DataFrame(payload["discard_rows"]).to_csv(output_dir / "discard_overlap_hist.csv", index=False)
    pd.DataFrame(payload["event_length_rows"]).to_csv(output_dir / "event_length_coverage.csv", index=False)
    pd.DataFrame(payload["coverage_rows"]).to_csv(output_dir / "event_coverage_summary.csv", index=False)
    write_json(output_dir / "summary.json", payload["summary"])
    print(f"[analysis] output_dir={output_dir}")


if __name__ == "__main__":
    main()
