import argparse
import csv
import json
from pathlib import Path


def interval_intersection(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def interval_union(a_start, a_end, b_start, b_end):
    return max(a_end, b_end) - min(a_start, b_start)


def interval_iou(a_start, a_end, b_start, b_end):
    inter = interval_intersection(a_start, a_end, b_start, b_end)
    union = interval_union(a_start, a_end, b_start, b_end)
    return inter / union if union > 0 else 0.0


def best_match_for_gt(gt_segment, events):
    gt_start, gt_end = gt_segment
    best = None
    for event in events:
        inter = interval_intersection(gt_start, gt_end, event["start_time"], event["end_time"])
        iou = interval_iou(gt_start, gt_end, event["start_time"], event["end_time"])
        coverage = inter / max(gt_end - gt_start, 1e-8)
        candidate = {
            "event_id": int(event["event_id"]),
            "pred_start": float(event["start_time"]),
            "pred_end": float(event["end_time"]),
            "pred_duration": float(event["duration_sec"]),
            "pred_confidence": float(event["confidence"]),
            "intersection": float(inter),
            "iou": float(iou),
            "gt_coverage": float(coverage),
        }
        if best is None or (candidate["intersection"], candidate["iou"], candidate["pred_confidence"]) > (
            best["intersection"],
            best["iou"],
            best["pred_confidence"],
        ):
            best = candidate
    return best


def best_match_for_event(event, gt_segments):
    best = None
    for idx, (gt_start, gt_end) in enumerate(gt_segments):
        inter = interval_intersection(gt_start, gt_end, event["start_time"], event["end_time"])
        iou = interval_iou(gt_start, gt_end, event["start_time"], event["end_time"])
        coverage = inter / max(event["duration_sec"], 1e-8)
        candidate = {
            "gt_index": idx,
            "gt_start": float(gt_start),
            "gt_end": float(gt_end),
            "gt_duration": float(gt_end - gt_start),
            "intersection": float(inter),
            "iou": float(iou),
            "pred_coverage": float(coverage),
        }
        if best is None or (candidate["intersection"], candidate["iou"]) > (
            best["intersection"],
            best["iou"],
        ):
            best = candidate
    return best


def official_matched_gt_indices(events, gt_segments):
    matched = set()
    for event in events:
        overlaps = [
            interval_intersection(event["start_time"], event["end_time"], gt_start, gt_end)
            for gt_start, gt_end in gt_segments
        ]
        if overlaps and max(overlaps) > 0.0:
            best_gt_index = max(range(len(overlaps)), key=lambda index: overlaps[index])
            matched.add(best_gt_index)
    return matched


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tag", default="fast_only")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    missed_rows = []
    fp_rows = []
    matched_rows = []
    overlap_only_rows = []

    official_matched_total = 0

    for video in data["videos"]:
        video_id = video["video_id"]
        video_path = video["video"]
        gt_segments = [(float(seg["start"]), float(seg["end"])) for seg in video.get("ground_truth_segments", [])]
        events = video["fast_only"]["events"]
        official_matched = official_matched_gt_indices(events, gt_segments)
        official_matched_total += len(official_matched)

        for gt_index, gt_segment in enumerate(gt_segments):
            best = best_match_for_gt(gt_segment, events) if events else None
            any_overlap = bool(best and best["intersection"] > 0.0)
            official_matched_flag = gt_index in official_matched
            row = {
                "video_id": video_id,
                "video_path": video_path,
                "gt_index": gt_index,
                "gt_start": gt_segment[0],
                "gt_end": gt_segment[1],
                "gt_duration": gt_segment[1] - gt_segment[0],
                "official_matched": official_matched_flag,
                "any_overlap": any_overlap,
                "best_event_id": "" if best is None else best["event_id"],
                "best_pred_start": "" if best is None else best["pred_start"],
                "best_pred_end": "" if best is None else best["pred_end"],
                "best_pred_duration": "" if best is None else best["pred_duration"],
                "best_pred_confidence": "" if best is None else best["pred_confidence"],
                "best_intersection": 0.0 if best is None else best["intersection"],
                "best_iou": 0.0 if best is None else best["iou"],
                "best_gt_coverage": 0.0 if best is None else best["gt_coverage"],
            }
            if official_matched_flag:
                matched_rows.append(row)
            else:
                missed_rows.append(row)
                if any_overlap:
                    overlap_only_rows.append(row)

        for event in events:
            best = best_match_for_event(event, gt_segments) if gt_segments else None
            row = {
                "video_id": video_id,
                "video_path": video_path,
                "event_id": int(event["event_id"]),
                "pred_start": float(event["start_time"]),
                "pred_end": float(event["end_time"]),
                "pred_duration": float(event["duration_sec"]),
                "pred_confidence": float(event["confidence"]),
                "matched": bool(best and best["intersection"] > 0.0),
                "best_gt_index": "" if best is None else best["gt_index"],
                "best_gt_start": "" if best is None else best["gt_start"],
                "best_gt_end": "" if best is None else best["gt_end"],
                "best_gt_duration": "" if best is None else best["gt_duration"],
                "best_intersection": 0.0 if best is None else best["intersection"],
                "best_iou": 0.0 if best is None else best["iou"],
                "best_pred_coverage": 0.0 if best is None else best["pred_coverage"],
            }
            if not row["matched"]:
                fp_rows.append(row)

    missed_rows.sort(key=lambda row: (row["gt_duration"], row["video_id"], row["gt_start"]), reverse=True)
    fp_rows.sort(key=lambda row: (row["pred_confidence"], row["pred_duration"]), reverse=True)
    matched_rows.sort(key=lambda row: (row["best_iou"], row["best_gt_coverage"]), reverse=True)
    overlap_only_rows.sort(key=lambda row: (row["best_intersection"], row["best_iou"]), reverse=True)

    prefix = output_dir / args.tag
    missed_path = prefix.with_name(f"{args.tag}_missed_gt_events.csv")
    fp_path = prefix.with_name(f"{args.tag}_false_positive_events.csv")
    matched_path = prefix.with_name(f"{args.tag}_matched_gt_events.csv")
    overlap_only_path = prefix.with_name(f"{args.tag}_missed_but_overlap_gt_events.csv")
    summary_path = prefix.with_name(f"{args.tag}_failure_summary.json")

    gt_fieldnames = [
        "video_id", "video_path", "gt_index", "gt_start", "gt_end", "gt_duration", "official_matched",
        "any_overlap", "best_event_id", "best_pred_start", "best_pred_end", "best_pred_duration",
        "best_pred_confidence", "best_intersection", "best_iou", "best_gt_coverage",
    ]
    write_csv(missed_path, missed_rows, gt_fieldnames)
    write_csv(matched_path, matched_rows, gt_fieldnames)
    write_csv(overlap_only_path, overlap_only_rows, gt_fieldnames)
    write_csv(fp_path, fp_rows, [
        "video_id", "video_path", "event_id", "pred_start", "pred_end", "pred_duration", "pred_confidence",
        "matched", "best_gt_index", "best_gt_start", "best_gt_end", "best_gt_duration",
        "best_intersection", "best_iou", "best_pred_coverage",
    ])

    summary = {
        "input": args.input,
        "num_videos": len(data["videos"]),
        "gt_segments_total": sum(len(v.get("ground_truth_segments", [])) for v in data["videos"]),
        "predicted_events_total": sum(len(v["fast_only"]["events"]) for v in data["videos"]),
        "official_matched_gt_events": official_matched_total,
        "official_missed_gt_events": len(missed_rows),
        "overlap_but_officially_missed_gt_events": len(overlap_only_rows),
        "false_positive_events": len(fp_rows),
        "top_missed_by_duration": missed_rows[:20],
        "top_false_positives_by_confidence": fp_rows[:20],
        "top_overlap_but_missed": overlap_only_rows[:20],
        "top_matched_by_iou": matched_rows[:20],
        "artifacts": {
            "missed_gt_csv": str(missed_path),
            "false_positive_csv": str(fp_path),
            "matched_gt_csv": str(matched_path),
            "missed_but_overlap_csv": str(overlap_only_path),
        },
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
