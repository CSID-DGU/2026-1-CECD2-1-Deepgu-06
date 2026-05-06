import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.router import select_vlm_clips
from scripts.run_real_vlm_subset import (
    build_scored_clips,
    gt_segments_for_video,
    interval_intersection,
    load_ground_truth,
    load_videos_from_manifest,
    video_id_from_path,
)
from utils.config import load_config


def mean(values):
    return sum(values) / len(values) if values else 0.0


def quantiles(values):
    if not values:
        return {"min": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "max": 0.0}
    ordered = sorted(float(v) for v in values)

    def pick(ratio):
        index = int(round((len(ordered) - 1) * ratio))
        return ordered[index]

    return {
        "min": ordered[0],
        "p25": pick(0.25),
        "p50": pick(0.50),
        "p75": pick(0.75),
        "max": ordered[-1],
    }


def overlaps_gt(clip_item, gt_segments):
    for gt_start, gt_end in gt_segments:
        if interval_intersection(
            float(clip_item["start_time"]),
            float(clip_item["end_time"]),
            float(gt_start),
            float(gt_end),
        ) > 0.0:
            return True
    return False


def classify_router_failure(scored, selected_ids, gt_segments, router_cfg):
    low = float(router_cfg.get("prob_low", 0.2))
    high = float(router_cfg.get("prob_high", 0.8))
    unc_thr = float(router_cfg.get("uncertainty_threshold", 0.2))

    selected_set = {int(x) for x in selected_ids}
    overlap_clips = [item for item in scored if overlaps_gt(item, gt_segments)]

    def build_stats(items):
        probs = [float(item["fighting_prob"]) for item in items]
        uncs = [float(item["uncertainty"]) for item in items]
        gray = [item for item in items if low < float(item["fighting_prob"]) < high]
        unc_ok = [item for item in items if float(item["uncertainty"]) >= unc_thr]
        both = [
            item
            for item in items
            if int(item["clip_id"]) in selected_set
        ]
        below_gray = [item for item in items if float(item["fighting_prob"]) <= low]
        above_gray = [item for item in items if float(item["fighting_prob"]) >= high]
        gray_but_low_unc = [
            item for item in items
            if low < float(item["fighting_prob"]) < high and float(item["uncertainty"]) < unc_thr
        ]
        unc_ok_but_outside_gray = [
            item for item in items
            if not (low < float(item["fighting_prob"]) < high) and float(item["uncertainty"]) >= unc_thr
        ]
        return {
            "count": len(items),
            "prob_mean": mean(probs),
            "prob_quantiles": quantiles(probs),
            "uncertainty_mean": mean(uncs),
            "uncertainty_quantiles": quantiles(uncs),
            "gray_zone_count": len(gray),
            "uncertainty_ok_count": len(unc_ok),
            "selected_count": len(both),
            "below_gray_count": len(below_gray),
            "above_gray_count": len(above_gray),
            "gray_but_low_unc_count": len(gray_but_low_unc),
            "unc_ok_but_outside_gray_count": len(unc_ok_but_outside_gray),
            "top_prob_clips": [
                {
                    "clip_id": int(item["clip_id"]),
                    "start_time": float(item["start_time"]),
                    "end_time": float(item["end_time"]),
                    "fighting_prob": float(item["fighting_prob"]),
                    "uncertainty": float(item["uncertainty"]),
                }
                for item in sorted(items, key=lambda x: float(x["fighting_prob"]), reverse=True)[:5]
            ],
        }

    overall_stats = build_stats(scored)
    overlap_stats = build_stats(overlap_clips)

    failure_reason = "unknown"
    if overlap_stats["selected_count"] > 0:
        failure_reason = "selected_but_not_executed_or_not_counted"
    elif overlap_stats["gray_zone_count"] == 0 and overlap_stats["uncertainty_ok_count"] > 0:
        failure_reason = "gt_clips_outside_gray_zone"
    elif overlap_stats["gray_zone_count"] > 0 and overlap_stats["uncertainty_ok_count"] == 0:
        failure_reason = "gt_clips_low_uncertainty"
    elif overlap_stats["gray_zone_count"] == 0 and overlap_stats["uncertainty_ok_count"] == 0:
        failure_reason = "gt_clips_below_both_conditions"
    elif overlap_stats["gray_zone_count"] > 0 and overlap_stats["selected_count"] == 0:
        failure_reason = "gt_clips_gray_but_not_selected"

    return {
        "failure_reason": failure_reason,
        "overall_clip_stats": overall_stats,
        "gt_overlap_clip_stats": overlap_stats,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/home/deepgu/slowfast/configs/base.yaml")
    parser.add_argument("--manifest-csv", default="/home/deepgu/slowfast/data/manifests/cctv_x3d_s/testing_clips.csv")
    parser.add_argument("--ground-truth", default="/home/deepgu/test/cctv/dataset/ground-truth.json")
    parser.add_argument(
        "--analysis-json",
        default="/home/deepgu/slowfast/outputs/validation/full_testing_compare_floor040_cap3_analysis.json",
    )
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--prob-low", type=float, default=None)
    parser.add_argument("--prob-high", type=float, default=None)
    parser.add_argument("--disable-topk", action="store_true")
    parser.add_argument("--uncertainty-threshold", type=float, default=None)
    parser.add_argument(
        "--output",
        default="/home/deepgu/slowfast/outputs/validation/missed_routing_analysis.json",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.prob_low is not None:
        config["router"]["prob_low"] = float(args.prob_low)
    if args.prob_high is not None:
        config["router"]["prob_high"] = float(args.prob_high)
    if args.disable_topk:
        config["router"]["use_topk"] = False
    if args.uncertainty_threshold is not None:
        config["router"]["uncertainty_threshold"] = float(args.uncertainty_threshold)
    ground_truth_db = load_ground_truth(args.ground_truth)
    manifest_videos = load_videos_from_manifest(args.manifest_csv)
    video_path_by_id = {video_id_from_path(path): path for path in manifest_videos}

    with open(args.analysis_json, "r", encoding="utf-8") as handle:
        analysis = json.load(handle)

    target_ids = [record["video_id"] for record in analysis["top_cases"]["missed_without_vlm"][: args.limit]]
    records = []
    failure_counts = {}

    for video_id in target_ids:
        video_path = video_path_by_id.get(video_id)
        if not video_path:
            continue
        gt_segments = gt_segments_for_video(ground_truth_db, video_id)
        scored, fps = build_scored_clips(video_path, config)
        selected_ids = select_vlm_clips(scored, config["router"])
        detail = classify_router_failure(scored, selected_ids, gt_segments, config["router"])
        failure = detail["failure_reason"]
        failure_counts[failure] = failure_counts.get(failure, 0) + 1
        records.append(
            {
                "video_id": video_id,
                "video": video_path,
                "fps": float(fps),
                "gt_segments": [{"start": float(s), "end": float(e)} for s, e in gt_segments],
                "router_config": {
                    "prob_low": float(config["router"]["prob_low"]),
                    "prob_high": float(config["router"]["prob_high"]),
                    "uncertainty_threshold": float(config["router"]["uncertainty_threshold"]),
                    "use_topk": bool(config["router"].get("use_topk", True)),
                    "topk_ratio": float(config["router"].get("topk_ratio", 0.15)),
                },
                "selected_clip_count": len(selected_ids),
                **detail,
            }
        )

    output = {
        "source_analysis_json": args.analysis_json,
        "config_path": args.config,
        "manifest_csv": args.manifest_csv,
        "num_targets": len(records),
        "failure_reason_counts": failure_counts,
        "videos": records,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)

    print(json.dumps({
        "output": str(output_path),
        "num_targets": len(records),
        "failure_reason_counts": failure_counts,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
