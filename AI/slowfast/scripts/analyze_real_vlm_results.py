import argparse
import json
from pathlib import Path


def safe_div(num, den):
    return num / den if den else 0.0


def summarize_executed_calls(executed_calls):
    label_counts = {"fight": 0, "non_fight": 0, "unknown": 0}
    confidences = []
    fight_probs = []
    vlm_scores = []
    call_seconds = []

    for call in executed_calls:
        result = call.get("vlm_result", {})
        parsed = result.get("parsed", {}) or {}
        label = parsed.get("label", "unknown")
        if label not in label_counts:
            label = "unknown"
        label_counts[label] += 1

        confidence = parsed.get("confidence")
        if isinstance(confidence, (int, float)):
            confidences.append(float(confidence))

        score = result.get("score")
        if isinstance(score, (int, float)):
            vlm_scores.append(float(score))

        fight_prob = call.get("fighting_prob")
        if isinstance(fight_prob, (int, float)):
            fight_probs.append(float(fight_prob))

        sec = call.get("call_seconds")
        if isinstance(sec, (int, float)):
            call_seconds.append(float(sec))

    return {
        "count": len(executed_calls),
        "label_counts": label_counts,
        "mean_confidence": safe_div(sum(confidences), len(confidences)),
        "mean_fast_prob": safe_div(sum(fight_probs), len(fight_probs)),
        "mean_vlm_score": safe_div(sum(vlm_scores), len(vlm_scores)),
        "total_call_seconds": sum(call_seconds),
        "mean_call_seconds": safe_div(sum(call_seconds), len(call_seconds)),
    }


def build_video_record(video):
    fast_summary = video["fast_only"]["summary"]
    fused_summary = video["fast_plus_vlm"]["summary"]
    clip_stats = video["clip_stats"]
    call_summary = summarize_executed_calls(video.get("executed_calls", []))

    recall_delta = float(fused_summary["gt_segment_recall"]) - float(fast_summary["gt_segment_recall"])
    fp_delta = int(fused_summary["false_positive_events"]) - int(fast_summary["false_positive_events"])
    duration_delta = float(fused_summary["total_event_duration_sec"]) - float(fast_summary["total_event_duration_sec"])
    event_delta = int(fused_summary["num_events"]) - int(fast_summary["num_events"])

    over_suppressed = (
        clip_stats["executed_vlm_call_count"] > 0
        and (
            recall_delta < 0
            or (event_delta < 0 and duration_delta <= -2.0)
            or (duration_delta <= -3.0 and fused_summary["matched_gt_segments"] <= fast_summary["matched_gt_segments"])
        )
    )

    helped_without_recall_loss = fp_delta < 0 and recall_delta >= 0
    recall_improved = recall_delta > 0
    recall_worsened = recall_delta < 0
    fp_reduced = fp_delta < 0
    high_vlm_usage = clip_stats["executed_call_ratio"] >= 0.05
    missed_without_vlm = (
        clip_stats["executed_vlm_call_count"] == 0
        and fast_summary["matched_gt_segments"] == 0
        and len(video["ground_truth_segments"]) > 0
    )

    return {
        "video_id": video["video_id"],
        "video": video["video"],
        "gt_segments": len(video["ground_truth_segments"]),
        "fast_only": {
            "events": int(fast_summary["num_events"]),
            "false_positive_events": int(fast_summary["false_positive_events"]),
            "matched_gt_segments": int(fast_summary["matched_gt_segments"]),
            "recall": float(fast_summary["gt_segment_recall"]),
            "total_event_duration_sec": float(fast_summary["total_event_duration_sec"]),
        },
        "fast_plus_vlm": {
            "events": int(fused_summary["num_events"]),
            "false_positive_events": int(fused_summary["false_positive_events"]),
            "matched_gt_segments": int(fused_summary["matched_gt_segments"]),
            "recall": float(fused_summary["gt_segment_recall"]),
            "total_event_duration_sec": float(fused_summary["total_event_duration_sec"]),
        },
        "deltas": {
            "event_count_delta": event_delta,
            "false_positive_event_delta": fp_delta,
            "matched_gt_segment_delta": int(fused_summary["matched_gt_segments"]) - int(fast_summary["matched_gt_segments"]),
            "recall_delta": recall_delta,
            "total_event_duration_delta_sec": duration_delta,
        },
        "clip_stats": {
            "total_clips": int(clip_stats["total_clips"]),
            "selected_clip_count": int(clip_stats["selected_clip_count"]),
            "executed_vlm_call_count": int(clip_stats["executed_vlm_call_count"]),
            "selected_clip_ratio": float(clip_stats["selected_clip_ratio"]),
            "executed_call_ratio": float(clip_stats["executed_call_ratio"]),
        },
        "vlm_calls": call_summary,
        "flags": {
            "fp_reduced": fp_reduced,
            "recall_improved": recall_improved,
            "recall_worsened": recall_worsened,
            "over_suppressed": over_suppressed,
            "helped_without_recall_loss": helped_without_recall_loss,
            "high_vlm_usage": high_vlm_usage,
            "missed_without_vlm": missed_without_vlm,
        },
    }


def top_n(records, predicate, sort_key, limit):
    filtered = [record for record in records if predicate(record)]
    filtered.sort(key=sort_key)
    return filtered[:limit]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/deepgu/slowfast/outputs/validation/full_testing_compare_floor040_cap3.json",
    )
    parser.add_argument(
        "--output",
        default="/home/deepgu/slowfast/outputs/validation/full_testing_compare_floor040_cap3_analysis.json",
    )
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()

    input_path = Path(args.input)
    with open(input_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    per_video = [build_video_record(video) for video in payload["videos"]]
    overall = payload["overall"]

    category_counts = {
        "videos_total": len(per_video),
        "videos_with_vlm_calls": sum(1 for record in per_video if record["clip_stats"]["executed_vlm_call_count"] > 0),
        "fp_reduced_cases": sum(1 for record in per_video if record["flags"]["fp_reduced"]),
        "recall_improved_cases": sum(1 for record in per_video if record["flags"]["recall_improved"]),
        "recall_worsened_cases": sum(1 for record in per_video if record["flags"]["recall_worsened"]),
        "over_suppressed_cases": sum(1 for record in per_video if record["flags"]["over_suppressed"]),
        "helped_without_recall_loss_cases": sum(1 for record in per_video if record["flags"]["helped_without_recall_loss"]),
        "missed_without_vlm_cases": sum(1 for record in per_video if record["flags"]["missed_without_vlm"]),
    }

    top_cases = {
        "fp_reduced": top_n(
            per_video,
            lambda record: record["flags"]["fp_reduced"],
            lambda record: (
                record["deltas"]["false_positive_event_delta"],
                record["deltas"]["recall_delta"],
                record["video_id"],
            ),
            args.top_k,
        ),
        "recall_improved": top_n(
            per_video,
            lambda record: record["flags"]["recall_improved"],
            lambda record: (
                -record["deltas"]["recall_delta"],
                record["deltas"]["false_positive_event_delta"],
                record["video_id"],
            ),
            args.top_k,
        ),
        "recall_worsened": top_n(
            per_video,
            lambda record: record["flags"]["recall_worsened"],
            lambda record: (
                record["deltas"]["recall_delta"],
                record["deltas"]["total_event_duration_delta_sec"],
                record["video_id"],
            ),
            args.top_k,
        ),
        "over_suppressed": top_n(
            per_video,
            lambda record: record["flags"]["over_suppressed"],
            lambda record: (
                record["deltas"]["recall_delta"],
                record["deltas"]["total_event_duration_delta_sec"],
                -record["clip_stats"]["executed_vlm_call_count"],
                record["video_id"],
            ),
            args.top_k,
        ),
        "helped_without_recall_loss": top_n(
            per_video,
            lambda record: record["flags"]["helped_without_recall_loss"],
            lambda record: (
                record["deltas"]["false_positive_event_delta"],
                -record["deltas"]["recall_delta"],
                record["video_id"],
            ),
            args.top_k,
        ),
        "high_vlm_usage": top_n(
            per_video,
            lambda record: record["flags"]["high_vlm_usage"],
            lambda record: (
                -record["clip_stats"]["executed_call_ratio"],
                -record["clip_stats"]["executed_vlm_call_count"],
                record["video_id"],
            ),
            args.top_k,
        ),
        "missed_without_vlm": top_n(
            per_video,
            lambda record: record["flags"]["missed_without_vlm"],
            lambda record: (
                -record["gt_segments"],
                record["video_id"],
            ),
            args.top_k,
        ),
    }

    vlm_label_totals = {"fight": 0, "non_fight": 0, "unknown": 0}
    for record in per_video:
        for label, count in record["vlm_calls"]["label_counts"].items():
            vlm_label_totals[label] += int(count)

    analysis = {
        "source_compare_json": str(input_path),
        "overall": overall,
        "category_counts": category_counts,
        "vlm_label_totals": vlm_label_totals,
        "top_cases": top_cases,
        "per_video": sorted(per_video, key=lambda record: record["video_id"]),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(analysis, handle, ensure_ascii=False, indent=2)

    compact = {
        "output": str(output_path),
        "category_counts": category_counts,
        "vlm_label_totals": vlm_label_totals,
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
