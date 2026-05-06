import argparse
import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.vlm.infer import VLMRefiner
from pipeline.clip_generator import build_sliding_clips
from pipeline.event_builder import build_events
from pipeline.fast_stage import score_clips_fast
from pipeline.fusion import fuse_scores
from pipeline.motion_summary import attach_motion_summaries
from pipeline.router import select_vlm_clips
from pipeline.uncertainty import attach_uncertainty
from utils.config import load_config
from utils.video import load_video_frames


DEFAULT_VIDEOS = [
    "/home/deepgu/test/cctv/dataset/CCTV_DATA/testing/fight_0227.mpeg",
    "/home/deepgu/test/cctv/dataset/CCTV_DATA/testing/fight_0017.mpeg",
    "/home/deepgu/test/cctv/dataset/CCTV_DATA/testing/fight_0002.mpeg",
    "/home/deepgu/test/cctv/dataset/CCTV_DATA/testing/fight_0226.mpeg",
    "/home/deepgu/test/cctv/dataset/CCTV_DATA/testing/fight_0999.mpeg",
    "/home/deepgu/test/cctv/dataset/CCTV_DATA/testing/fight_0600.mpeg",
]


def load_ground_truth(gt_path):
    with open(gt_path, "r", encoding="utf-8") as handle:
        return json.load(handle)["database"]


def video_id_from_path(video_path):
    return Path(video_path).stem


def load_videos_from_manifest(manifest_csv):
    seen = []
    seen_ids = set()
    with open(manifest_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            video_path = row.get("video_path")
            video_id = row.get("video_id") or (video_id_from_path(video_path) if video_path else None)
            if not video_path or not video_id or video_id in seen_ids:
                continue
            seen.append(video_path)
            seen_ids.add(video_id)
    return seen


def build_scored_clips(video_path, config):
    frames, fps = load_video_frames(video_path)
    clips = build_sliding_clips(
        frames=frames,
        fps=fps,
        temporal_window_sec=float(config["clip"]["temporal_window_sec"]),
        stride_sec=float(config["clip"]["stride_sec"]),
    )
    scored = score_clips_fast(clips, config["fast_model"], config["clip"])
    scored = attach_uncertainty(
        scored,
        score_key="fighting_prob",
        alpha_entropy=float(config["router"].get("alpha_entropy", 0.7)),
        alpha_variance=float(config["router"].get("alpha_variance", 0.3)),
        variance_window=int(config["router"].get("variance_window", 5)),
    )
    return attach_motion_summaries(scored), fps


def clone_with_fast_scores(scored):
    updated = []
    for item in scored:
        cloned = dict(item)
        cloned["final_score"] = float(item["fighting_prob"])
        cloned["vlm_called"] = False
        cloned["vlm_score"] = None
        updated.append(cloned)
    return updated


def rank_selected_clips(scored, selected_ids):
    selected_set = {int(clip_id) for clip_id in selected_ids}
    ranked = [item for item in scored if int(item["clip_id"]) in selected_set]
    ranked.sort(key=lambda item: (-float(item["uncertainty"]), int(item["clip_id"])))
    return ranked


def interval_intersection(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def interval_union(a_start, a_end, b_start, b_end):
    return max(a_end, b_end) - min(a_start, b_start)


def interval_iou(a_start, a_end, b_start, b_end):
    inter = interval_intersection(a_start, a_end, b_start, b_end)
    union = interval_union(a_start, a_end, b_start, b_end)
    if union <= 0:
        return 0.0
    return inter / union


def summarize_events(events, gt_segments):
    total_duration = sum(float(event["duration_sec"]) for event in events)
    durations = [float(event["duration_sec"]) for event in events]
    false_positive_events = 0
    matched_events = 0
    best_ious = []
    matched_gt_indices = set()

    for event in events:
        overlaps = [
            interval_intersection(event["start_time"], event["end_time"], gt_start, gt_end)
            for gt_start, gt_end in gt_segments
        ]
        ious = [
            interval_iou(event["start_time"], event["end_time"], gt_start, gt_end)
            for gt_start, gt_end in gt_segments
        ]
        best_iou = max(ious) if ious else 0.0
        best_ious.append(best_iou)
        if overlaps and max(overlaps) > 0.0:
            matched_events += 1
            best_gt_index = max(range(len(overlaps)), key=lambda index: overlaps[index])
            matched_gt_indices.add(best_gt_index)
        else:
            false_positive_events += 1

    return {
        "num_events": len(events),
        "total_event_duration_sec": total_duration,
        "avg_event_duration_sec": (total_duration / len(events)) if events else 0.0,
        "event_durations_sec": durations,
        "false_positive_events": false_positive_events,
        "matched_events": matched_events,
        "matched_gt_segments": len(matched_gt_indices),
        "gt_segment_recall": (len(matched_gt_indices) / len(gt_segments)) if gt_segments else 0.0,
        "best_event_ious": best_ious,
        "events": events,
    }


def gt_segments_for_video(ground_truth_db, video_id):
    item = ground_truth_db.get(video_id, {})
    annotations = item.get("annotations", [])
    return [tuple(annotation["segment"]) for annotation in annotations]


def compact_event_view(events):
    return [
        {
            "event_id": int(event["event_id"]),
            "start_time": float(event["start_time"]),
            "end_time": float(event["end_time"]),
            "duration_sec": float(event["duration_sec"]),
            "confidence": float(event["confidence"]),
        }
        for event in events
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/home/deepgu/slowfast/configs/base.yaml")
    parser.add_argument("--fast-checkpoint", default=None)
    parser.add_argument("--videos", nargs="*", default=DEFAULT_VIDEOS)
    parser.add_argument("--manifest-csv", default=None)
    parser.add_argument("--max-calls-per-video", type=int, default=3)
    parser.add_argument("--vlm-sampled-frames", type=int, default=None)
    parser.add_argument("--ground-truth", default="/home/deepgu/test/cctv/dataset/ground-truth.json")
    parser.add_argument("--prob-low", type=float, default=0.35)
    parser.add_argument("--prob-high", type=float, default=0.55)
    parser.add_argument("--disable-topk", action="store_true")
    parser.add_argument("--max-drop", type=float, default=None)
    parser.add_argument("--protect-above-score", type=float, default=None)
    parser.add_argument("--protect-floor-score", type=float, default=None)
    parser.add_argument(
        "--output",
        default="/home/deepgu/slowfast/outputs/validation/real_vlm_subset_compare.json",
    )
    args = parser.parse_args()

    if args.manifest_csv:
        args.videos = load_videos_from_manifest(args.manifest_csv)

    config = load_config(args.config)
    if args.fast_checkpoint:
        config["fast_model"]["checkpoint_path"] = str(args.fast_checkpoint)
    config["vlm"]["provider"] = "internvl"
    if args.vlm_sampled_frames is not None:
        config["vlm"]["sampled_frames"] = int(args.vlm_sampled_frames)
    config["router"]["prob_low"] = float(args.prob_low)
    config["router"]["prob_high"] = float(args.prob_high)
    if args.disable_topk:
        config["router"]["use_topk"] = False
    if args.max_drop is not None:
        config["fusion"]["suppression_bound"]["max_drop"] = float(args.max_drop)
    if args.protect_above_score is not None:
        config["fusion"]["suppression_bound"]["protect_above_score"] = float(args.protect_above_score)
    if args.protect_floor_score is not None:
        config["fusion"]["suppression_bound"]["protect_floor_score"] = float(args.protect_floor_score)
    ground_truth_db = load_ground_truth(args.ground_truth)
    refiner = VLMRefiner(config["vlm"])

    summary = {
        "config_path": args.config,
        "ground_truth_path": args.ground_truth,
        "provider": config["vlm"]["provider"],
        "fast_checkpoint": str(config["fast_model"]["checkpoint_path"]),
        "vlm_sampled_frames": int(config["vlm"]["sampled_frames"]),
        "num_videos": len(args.videos),
        "max_calls_per_video": int(args.max_calls_per_video),
        "router_overrides": {
            "prob_low": float(config["router"]["prob_low"]),
            "prob_high": float(config["router"]["prob_high"]),
            "use_topk": bool(config["router"].get("use_topk", True)),
        },
        "fusion_overrides": {
            "max_drop": float(config["fusion"]["suppression_bound"]["max_drop"]),
            "protect_above_score": float(config["fusion"]["suppression_bound"]["protect_above_score"]),
            "protect_floor_score": float(config["fusion"]["suppression_bound"]["protect_floor_score"]),
        },
        "videos": [],
    }

    overall_start = time.perf_counter()
    total_scored_clips = 0
    total_selected_clips = 0
    total_executed_calls = 0
    total_fast_events = 0
    total_fused_events = 0
    total_fast_fp = 0
    total_fused_fp = 0
    total_fast_duration = 0.0
    total_fused_duration = 0.0
    total_gt_segments = 0
    total_fast_matched_gt = 0
    total_fused_matched_gt = 0

    for video in args.videos:
        video_start = time.perf_counter()
        video_id = video_id_from_path(video)
        gt_segments = gt_segments_for_video(ground_truth_db, video_id)

        scored, fps = build_scored_clips(video, config)
        selected_ids = select_vlm_clips(scored, config["router"])
        ranked_selected = rank_selected_clips(scored, selected_ids)
        executed_targets = ranked_selected[: args.max_calls_per_video]

        vlm_outputs = {}
        executed = []
        for clip in executed_targets:
            call_start = time.perf_counter()
            result = refiner.score_clip(clip)
            call_sec = time.perf_counter() - call_start
            clip_id = int(clip["clip_id"])
            vlm_outputs[clip_id] = result
            executed.append(
                {
                    "clip_id": clip_id,
                    "start_time": float(clip["start_time"]),
                    "end_time": float(clip["end_time"]),
                    "fighting_prob": float(clip["fighting_prob"]),
                    "uncertainty": float(clip["uncertainty"]),
                    "motion_summary": clip["motion_summary"],
                    "call_seconds": call_sec,
                    "vlm_result": result,
                }
            )

        fast_only_inputs = clone_with_fast_scores(scored)
        fast_events, _ = build_events(fast_only_inputs, config["thresholds"], fps=fps)

        fused_inputs = fuse_scores(scored, vlm_outputs, config["fusion"])
        fused_events, _ = build_events(fused_inputs, config["thresholds"], fps=fps)

        fast_summary = summarize_events(fast_events, gt_segments)
        fused_summary = summarize_events(fused_events, gt_segments)

        clip_count = len(scored)
        executed_count = len(executed)
        selected_count = len(selected_ids)
        video_seconds = time.perf_counter() - video_start

        record = {
            "video": video,
            "video_id": video_id,
            "fps": float(fps),
            "ground_truth_segments": [{"start": seg[0], "end": seg[1]} for seg in gt_segments],
            "clip_stats": {
                "total_clips": clip_count,
                "selected_clip_count": selected_count,
                "executed_vlm_call_count": executed_count,
                "selected_clip_ratio": (selected_count / clip_count) if clip_count else 0.0,
                "executed_call_ratio": (executed_count / clip_count) if clip_count else 0.0,
                "selected_clip_ids_preview": [int(item["clip_id"]) for item in ranked_selected[:10]],
            },
            "fast_only": {
                "summary": {key: value for key, value in fast_summary.items() if key != "events"},
                "events": compact_event_view(fast_summary["events"]),
            },
            "fast_plus_vlm": {
                "summary": {key: value for key, value in fused_summary.items() if key != "events"},
                "events": compact_event_view(fused_summary["events"]),
            },
            "deltas": {
                "event_count_delta": fused_summary["num_events"] - fast_summary["num_events"],
                "false_positive_event_delta": (
                    fused_summary["false_positive_events"] - fast_summary["false_positive_events"]
                ),
                "total_event_duration_delta_sec": (
                    fused_summary["total_event_duration_sec"] - fast_summary["total_event_duration_sec"]
                ),
                "avg_event_duration_delta_sec": (
                    fused_summary["avg_event_duration_sec"] - fast_summary["avg_event_duration_sec"]
                ),
            },
            "executed_calls": executed,
            "timing": {
                "video_seconds": video_seconds,
                "vlm_call_seconds_total": sum(item["call_seconds"] for item in executed),
            },
        }
        summary["videos"].append(record)
        print(json.dumps(record, ensure_ascii=False))

        total_scored_clips += clip_count
        total_selected_clips += selected_count
        total_executed_calls += executed_count
        total_fast_events += fast_summary["num_events"]
        total_fused_events += fused_summary["num_events"]
        total_fast_fp += fast_summary["false_positive_events"]
        total_fused_fp += fused_summary["false_positive_events"]
        total_fast_duration += fast_summary["total_event_duration_sec"]
        total_fused_duration += fused_summary["total_event_duration_sec"]
        total_gt_segments += len(gt_segments)
        total_fast_matched_gt += fast_summary["matched_gt_segments"]
        total_fused_matched_gt += fused_summary["matched_gt_segments"]

    summary["overall"] = {
        "total_scored_clips": total_scored_clips,
        "total_selected_clips": total_selected_clips,
        "total_executed_vlm_calls": total_executed_calls,
        "selected_clip_ratio": (total_selected_clips / total_scored_clips) if total_scored_clips else 0.0,
        "executed_call_ratio": (total_executed_calls / total_scored_clips) if total_scored_clips else 0.0,
        "fast_only_total_events": total_fast_events,
        "fast_plus_vlm_total_events": total_fused_events,
        "fast_only_false_positive_events": total_fast_fp,
        "fast_plus_vlm_false_positive_events": total_fused_fp,
        "false_positive_event_delta": total_fused_fp - total_fast_fp,
        "gt_segments_total": total_gt_segments,
        "fast_only_matched_gt_segments": total_fast_matched_gt,
        "fast_plus_vlm_matched_gt_segments": total_fused_matched_gt,
        "fast_only_recall": (total_fast_matched_gt / total_gt_segments) if total_gt_segments else 0.0,
        "fast_plus_vlm_recall": (total_fused_matched_gt / total_gt_segments) if total_gt_segments else 0.0,
        "recall_delta": (
            (total_fused_matched_gt / total_gt_segments) - (total_fast_matched_gt / total_gt_segments)
        ) if total_gt_segments else 0.0,
        "fast_only_total_event_duration_sec": total_fast_duration,
        "fast_plus_vlm_total_event_duration_sec": total_fused_duration,
        "total_event_duration_delta_sec": total_fused_duration - total_fast_duration,
        "overall_seconds": time.perf_counter() - overall_start,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary["overall"], ensure_ascii=False))
    print(json.dumps({"output": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
