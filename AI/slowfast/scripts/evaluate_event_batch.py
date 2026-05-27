import argparse
import math
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.main_pipeline import run_single_video_pipeline
from utils.config import load_config
from utils.io import ensure_dir, load_json, write_json


def build_video_path(dataset_root, source, subset, video_id):
    if source == "CCTV":
        return Path(dataset_root) / "CCTV_DATA" / subset / f"{video_id}.mpeg"
    return Path(dataset_root) / "NON_CCTV_DATA" / subset / f"{video_id}.mpeg"


def seconds_to_frame_range(start_sec, end_sec, fps, nb_frames):
    start_frame = max(0, int(math.floor(float(start_sec) * float(fps))))
    end_frame = min(int(nb_frames) - 1, int(math.ceil(float(end_sec) * float(fps))) - 1)
    return start_frame, max(start_frame, end_frame)


def interval_iou(a_start, a_end, b_start, b_end):
    left = max(int(a_start), int(b_start))
    right = min(int(a_end), int(b_end))
    intersection = max(0, right - left + 1)
    union = (int(a_end) - int(a_start) + 1) + (int(b_end) - int(b_start) + 1) - intersection
    return intersection / union if union > 0 else 0.0


def build_gt_events(meta):
    fps = float(meta["frame_rate"])
    nb_frames = int(meta["nb_frames"])
    gt_events = []
    for idx, annotation in enumerate(meta.get("annotations", [])):
        start_sec, end_sec = annotation["segment"]
        start_frame, end_frame = seconds_to_frame_range(start_sec, end_sec, fps, nb_frames)
        gt_events.append(
            {
                "event_id": idx,
                "label": str(annotation.get("label", "Fight")),
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "start_time": float(start_sec),
                "end_time": float(end_sec),
                "duration_sec": max(0.0, float(end_sec) - float(start_sec)),
            }
        )
    return gt_events


def evaluate_video(pred_events, gt_events, iou_threshold):
    matched = set()
    true_positive = 0
    matched_predicted_duration_sec = 0.0
    for event in pred_events:
        for index, gt in enumerate(gt_events):
            if index in matched:
                continue
            iou = interval_iou(
                event["start_frame"],
                event["end_frame"],
                gt["start_frame"],
                gt["end_frame"],
            )
            if iou >= iou_threshold:
                true_positive += 1
                matched.add(index)
                matched_predicted_duration_sec += float(event.get("duration_sec", 0.0))
                break

    false_positive = max(0, len(pred_events) - true_positive)
    false_negative = max(0, len(gt_events) - true_positive)
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    predicted_total_duration_sec = sum(float(event.get("duration_sec", 0.0)) for event in pred_events)
    gt_total_duration_sec = sum(float(event.get("duration_sec", 0.0)) for event in gt_events)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
        "predicted_event_count": len(pred_events),
        "gt_event_count": len(gt_events),
        "predicted_total_duration_sec": predicted_total_duration_sec,
        "matched_predicted_duration_sec": matched_predicted_duration_sec,
        "gt_total_duration_sec": gt_total_duration_sec,
        "predicted_to_gt_duration_ratio": predicted_total_duration_sec / max(gt_total_duration_sec, 1e-12),
    }


def summarize(per_video):
    tp = sum(int(item["true_positive"]) for item in per_video)
    fp = sum(int(item["false_positive"]) for item in per_video)
    fn = sum(int(item["false_negative"]) for item in per_video)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    predicted_total_duration_sec = sum(float(item["predicted_total_duration_sec"]) for item in per_video)
    gt_total_duration_sec = sum(float(item["gt_total_duration_sec"]) for item in per_video)
    return {
        "video_count": len(per_video),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
        "fp_event_count": fp,
        "fn_event_count": fn,
        "predicted_total_duration_sec": predicted_total_duration_sec,
        "gt_total_duration_sec": gt_total_duration_sec,
        "predicted_to_gt_duration_ratio": predicted_total_duration_sec / max(gt_total_duration_sec, 1e-12),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ground-truth-json", default="/home/deepgu/test/cctv/dataset/ground-truth.json")
    parser.add_argument("--dataset-root", default="/home/deepgu/test/cctv/dataset")
    parser.add_argument("--subset", default="testing")
    parser.add_argument("--source-filter", default="CCTV")
    parser.add_argument("--iou-threshold", type=float, default=0.1)
    parser.add_argument("--run-prefix", default="event_eval")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--video-ids", nargs="+", default=None, help="평가할 video_id 목록 (미지정 시 전체)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.summary_only:
        outputs_config = config.setdefault("outputs", {})
        outputs_config["save_run_artifacts"] = False
        outputs_config["save_event_media"] = False
        outputs_config["save_clip_manifest"] = False
    database = load_json(args.ground_truth_json)["database"]
    output_path = Path(args.output_json)
    ensure_dir(output_path.parent)

    per_video = []
    processed = 0
    for video_id, meta in sorted(database.items()):
        source = meta.get("source")
        subset = meta.get("subset")
        if args.source_filter and source != args.source_filter:
            continue
        if subset != args.subset:
            continue
        if args.video_ids is not None and video_id not in args.video_ids:
            continue

        video_path = build_video_path(args.dataset_root, source, subset, video_id)
        if not video_path.exists():
            continue

        run_name = f"{args.run_prefix}_{video_id}"
        result = run_single_video_pipeline(str(video_path), deepcopy(config), run_name=run_name, verbose=False)
        gt_events = build_gt_events(meta)
        metrics = evaluate_video(result["events"], gt_events, iou_threshold=float(args.iou_threshold))
        metrics["video_id"] = video_id
        if result.get("output_dir") is not None:
            metrics["output_dir"] = result["output_dir"]
        per_video.append(metrics)
        processed += 1
        print(
            f"[video] id={video_id} tp={metrics['true_positive']} fp={metrics['false_positive']} "
            f"fn={metrics['false_negative']} recall={metrics['event_recall']:.4f} f1={metrics['event_f1']:.4f}",
            flush=True,
        )
        if args.limit is not None and processed >= int(args.limit):
            break

    summary = summarize(per_video)
    payload = {
        "config": args.config,
        "ground_truth_json": args.ground_truth_json,
        "dataset_root": args.dataset_root,
        "subset": args.subset,
        "source_filter": args.source_filter,
        "iou_threshold": float(args.iou_threshold),
        "summary_only": bool(args.summary_only),
        "summary": summary,
        "per_video": per_video,
    }
    write_json(output_path, payload)
    print(f"[summary] {summary}", flush=True)
    print(f"[output] {output_path}", flush=True)


if __name__ == "__main__":
    main()
