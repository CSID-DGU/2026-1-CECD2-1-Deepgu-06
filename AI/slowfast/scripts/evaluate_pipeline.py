import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.io import load_json


def interval_iou(a_start, a_end, b_start, b_end):
    left = max(a_start, b_start)
    right = min(a_end, b_end)
    intersection = max(0, right - left + 1)
    union = (a_end - a_start + 1) + (b_end - b_start + 1) - intersection
    return intersection / union if union > 0 else 0.0


def evaluate(events, annotations, iou_threshold=0.1):
    gt_events = annotations[0]["events"] if annotations else []
    matched = set()
    true_positive = 0
    for event in events:
        for index, gt in enumerate(gt_events):
            if index in matched:
                continue
            iou = interval_iou(
                int(event["start_frame"]),
                int(event["end_frame"]),
                int(gt["start_frame"]),
                int(gt["end_frame"]),
            )
            if iou >= iou_threshold:
                true_positive += 1
                matched.add(index)
                break
    false_positive = max(0, len(events) - true_positive)
    false_negative = max(0, len(gt_events) - true_positive)
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-json", required=True)
    parser.add_argument("--annotations-json", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.1)
    args = parser.parse_args()

    events = load_json(args.events_json)
    annotations = load_json(args.annotations_json)
    metrics = evaluate(events, annotations, iou_threshold=args.iou_threshold)
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
