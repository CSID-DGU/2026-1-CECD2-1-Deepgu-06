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


def evaluate_video(pred_events, gt_events, iou_threshold, eval_mode="gt-centric"):
    """
    eval_mode="gt-centric" (기본):
        GT 기준 평가. 각 GT에 대해 IoU >= threshold인 pred가 하나라도 있으면 TP.
        하나의 pred가 여러 GT를 커버해도 모두 TP 인정.
        TP = matched GT 수 (recall 기준)
        FP = 어떤 GT와도 매칭되지 않는 pred 수 (precision 기준)
        FN = 어떤 pred와도 매칭되지 않는 GT 수
    eval_mode="greedy" (레거시):
        pred 순회하며 첫 번째 매칭 GT에 소비. pred 1개 = GT 1개.
    """
    predicted_total_duration_sec = sum(float(e.get("duration_sec", 0.0)) for e in pred_events)
    gt_total_duration_sec = sum(float(e.get("duration_sec", 0.0)) for e in gt_events)

    if eval_mode == "greedy":
        matched = set()
        true_positive = 0
        matched_predicted_duration_sec = 0.0
        for event in pred_events:
            for index, gt in enumerate(gt_events):
                if index in matched:
                    continue
                if interval_iou(event["start_frame"], event["end_frame"],
                                gt["start_frame"], gt["end_frame"]) >= iou_threshold:
                    true_positive += 1
                    matched.add(index)
                    matched_predicted_duration_sec += float(event.get("duration_sec", 0.0))
                    break
        false_positive = max(0, len(pred_events) - true_positive)
        false_negative = max(0, len(gt_events) - true_positive)
    else:
        # GT-centric
        matched_predicted_duration_sec = 0.0
        # GT side: 각 GT가 임의의 pred와 매칭되는지
        true_positive = 0
        for gt in gt_events:
            for pred in pred_events:
                if interval_iou(pred["start_frame"], pred["end_frame"],
                                gt["start_frame"], gt["end_frame"]) >= iou_threshold:
                    true_positive += 1
                    break
        false_negative = max(0, len(gt_events) - true_positive)
        # Pred side: 각 pred가 임의의 GT와 매칭되는지
        tp_pred = 0
        for pred in pred_events:
            for gt in gt_events:
                if interval_iou(pred["start_frame"], pred["end_frame"],
                                gt["start_frame"], gt["end_frame"]) >= iou_threshold:
                    tp_pred += 1
                    matched_predicted_duration_sec += float(pred.get("duration_sec", 0.0))
                    break
        false_positive = max(0, len(pred_events) - tp_pred)

    precision = (len(pred_events) - false_positive) / max(1, len(pred_events))
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
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
    n_preds = sum(int(item["predicted_event_count"]) for item in per_video)
    # precision = 매칭된 pred 수(tp_pred) / 전체 pred 수.  tp_pred = n_preds - fp.
    # (gt-centric: tp(GT측)와 tp_pred(pred측)는 다르므로 precision 분자는 tp_pred를 사용.
    #  greedy 모드에서는 tp_pred == tp 이므로 동일 식이 성립.)
    tp_pred = max(0, n_preds - fp)
    precision = tp_pred / max(1, n_preds)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    predicted_total_duration_sec = sum(float(item["predicted_total_duration_sec"]) for item in per_video)
    gt_total_duration_sec = sum(float(item["gt_total_duration_sec"]) for item in per_video)
    return {
        "video_count": len(per_video),
        "true_positive": tp,
        "matched_pred_count": tp_pred,
        "predicted_event_count": n_preds,
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
    parser.add_argument("--eval-mode", default="gt-centric", choices=["gt-centric", "greedy"],
                        help="gt-centric: 각 GT를 독립 평가 (기본). greedy: pred 순회 greedy 매칭 (레거시).")
    parser.add_argument("--run-prefix", default="event_eval")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--video-ids", nargs="+", default=None,
                        help="평가할 video_id 목록을 직접 지정 (지정 시 service-scope/all 무시)")
    parser.add_argument("--service-scope-json",
                        default=str(PROJECT_ROOT / "data/manifests/test_service_scope.json"),
                        help="61개 service-scope 목록 파일 (기본 평가 범위)")
    parser.add_argument("--all-videos", action="store_true",
                        help="service-scope 제한을 풀고 subset+source 전체(예: testing 70개)를 평가")
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

    # 평가 범위 결정 (우선순위: 명시적 --video-ids > --all-videos > service-scope)
    if args.video_ids is not None:
        allowed_ids = set(args.video_ids)
        video_scope = "explicit"
    elif args.all_videos:
        allowed_ids = None  # subset+source 필터만 적용
        video_scope = "all"
    else:
        allowed_ids = set(load_json(args.service_scope_json)["video_ids"])
        video_scope = "service_scope"
    print(f"[scope] {video_scope} (videos={'all' if allowed_ids is None else len(allowed_ids)})", flush=True)

    per_video = []
    processed = 0
    for video_id, meta in sorted(database.items()):
        source = meta.get("source")
        subset = meta.get("subset")
        if args.source_filter and source != args.source_filter:
            continue
        if subset != args.subset:
            continue
        if allowed_ids is not None and video_id not in allowed_ids:
            continue

        video_path = build_video_path(args.dataset_root, source, subset, video_id)
        if not video_path.exists():
            continue

        run_name = f"{args.run_prefix}_{video_id}"
        result = run_single_video_pipeline(str(video_path), deepcopy(config), run_name=run_name, verbose=False)
        gt_events = build_gt_events(meta)
        metrics = evaluate_video(result["events"], gt_events,
                                 iou_threshold=float(args.iou_threshold),
                                 eval_mode=args.eval_mode)
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
        "eval_mode": args.eval_mode,
        "video_scope": video_scope,
        "summary_only": bool(args.summary_only),
        "summary": summary,
        "per_video": per_video,
    }
    write_json(output_path, payload)
    print(f"[summary] {summary}", flush=True)
    print(f"[output] {output_path}", flush=True)


if __name__ == "__main__":
    main()
