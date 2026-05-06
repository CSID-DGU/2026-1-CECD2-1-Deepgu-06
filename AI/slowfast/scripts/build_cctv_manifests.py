import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.io import ensure_dir, write_json


def seconds_to_frame_range(start_sec, end_sec, fps, nb_frames):
    start_frame = max(0, int(math.floor(float(start_sec) * float(fps))))
    end_frame = min(int(nb_frames) - 1, int(math.ceil(float(end_sec) * float(fps))) - 1)
    return start_frame, max(start_frame, end_frame)


def interval_overlap(a_start, a_end, b_start, b_end):
    left = max(float(a_start), float(b_start))
    right = min(float(a_end), float(b_end))
    return max(0.0, right - left)


def build_video_path(dataset_root, source, subset, video_id):
    if source == "CCTV":
        return Path(dataset_root) / "CCTV_DATA" / subset / f"{video_id}.mpeg"
    return Path(dataset_root) / "NON_CCTV_DATA" / subset / f"{video_id}.mpeg"


def make_clip_rows(
    ground_truth_path,
    dataset_root,
    source_filter,
    temporal_window_sec,
    stride_sec,
    window_overlap_threshold,
    gt_overlap_threshold,
):
    payload = json.load(open(ground_truth_path, "r", encoding="utf-8"))
    database = payload["database"]

    rows_by_split = {"training": [], "validation": [], "testing": []}
    summary = {
        "source_filter": source_filter,
        "temporal_window_sec": float(temporal_window_sec),
        "stride_sec": float(stride_sec),
        "window_overlap_threshold": float(window_overlap_threshold),
        "gt_overlap_threshold": float(gt_overlap_threshold),
        "videos_by_split": Counter(),
        "clips_by_split": Counter(),
        "labels_by_split": {},
    }

    for video_id, meta in sorted(database.items()):
        source = meta.get("source")
        subset = meta.get("subset")
        if source_filter and source != source_filter:
            continue
        if subset not in rows_by_split:
            continue

        video_path = build_video_path(dataset_root, source, subset, video_id)
        if not video_path.exists():
            continue

        fps = float(meta["frame_rate"])
        nb_frames = int(meta["nb_frames"])
        duration = float(meta["duration"])
        if duration < temporal_window_sec:
            continue

        positive_ranges_sec = []
        for annotation in meta.get("annotations", []):
            start_sec, end_sec = annotation["segment"]
            positive_ranges_sec.append((float(start_sec), float(end_sec)))

        clip_index = 0
        max_clip_start = max(0.0, duration - float(temporal_window_sec))
        clip_start_sec = 0.0
        while clip_start_sec <= max_clip_start + 1e-9:
            clip_end_sec = clip_start_sec + float(temporal_window_sec)
            start_frame, end_frame = seconds_to_frame_range(
                clip_start_sec,
                clip_end_sec,
                fps,
                nb_frames,
            )
            clip_positive = 0
            max_window_overlap_ratio = 0.0
            max_gt_overlap_ratio = 0.0

            for pos_start_sec, pos_end_sec in positive_ranges_sec:
                overlap_sec = interval_overlap(clip_start_sec, clip_end_sec, pos_start_sec, pos_end_sec)
                gt_duration_sec = max(1e-6, float(pos_end_sec) - float(pos_start_sec))
                window_overlap_ratio = overlap_sec / float(temporal_window_sec)
                gt_overlap_ratio = overlap_sec / gt_duration_sec
                if window_overlap_ratio > max_window_overlap_ratio:
                    max_window_overlap_ratio = window_overlap_ratio
                if gt_overlap_ratio > max_gt_overlap_ratio:
                    max_gt_overlap_ratio = gt_overlap_ratio
                if (
                    window_overlap_ratio >= window_overlap_threshold
                    or gt_overlap_ratio >= gt_overlap_threshold
                ):
                    clip_positive = 1

            rows_by_split[subset].append(
                {
                    "clip_id": f"{video_id}_{clip_index:04d}",
                    "video_id": video_id,
                    "video_path": str(video_path),
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                    "start_time": float(clip_start_sec),
                    "end_time": float(clip_end_sec),
                    "label": int(clip_positive),
                    "split": subset,
                    "source": source,
                    "hard_negative": 0,
                    "max_window_overlap_ratio": float(max_window_overlap_ratio),
                    "max_gt_overlap_ratio": float(max_gt_overlap_ratio),
                    "fps": fps,
                    "nb_frames": nb_frames,
                }
            )
            clip_index += 1
            clip_start_sec += float(stride_sec)

        summary["videos_by_split"][subset] += 1

    for subset, rows in rows_by_split.items():
        summary["clips_by_split"][subset] = len(rows)
        label_counter = Counter(int(row["label"]) for row in rows)
        summary["labels_by_split"][subset] = dict(label_counter)

    summary["videos_by_split"] = dict(summary["videos_by_split"])
    summary["clips_by_split"] = dict(summary["clips_by_split"])
    return rows_by_split, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ground-truth-json",
        default="/home/deepgu/test/cctv/dataset/ground-truth.json",
    )
    parser.add_argument(
        "--dataset-root",
        default="/home/deepgu/test/cctv/dataset",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/deepgu/slowfast/data/manifests/cctv_x3d_s",
    )
    parser.add_argument("--source-filter", default="CCTV")
    parser.add_argument("--temporal-window-sec", type=float, default=2.0)
    parser.add_argument("--stride-sec", type=float, default=1.0)
    parser.add_argument("--window-overlap-threshold", type=float, default=0.3)
    parser.add_argument("--gt-overlap-threshold", type=float, default=0.5)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    rows_by_split, summary = make_clip_rows(
        ground_truth_path=args.ground_truth_json,
        dataset_root=args.dataset_root,
        source_filter=args.source_filter,
        temporal_window_sec=args.temporal_window_sec,
        stride_sec=args.stride_sec,
        window_overlap_threshold=args.window_overlap_threshold,
        gt_overlap_threshold=args.gt_overlap_threshold,
    )

    for subset, rows in rows_by_split.items():
        csv_path = output_dir / f"{subset}_clips.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"[manifest] subset={subset} rows={len(rows)} output={csv_path}")

    write_json(output_dir / "summary.json", summary)
    print(f"[summary] {summary}")


if __name__ == "__main__":
    main()
