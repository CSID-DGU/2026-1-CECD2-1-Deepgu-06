import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect pipeline output JSON.")
    parser.add_argument(
        "--results-path",
        default="/home/deepgu/test/outputs/results.json",
        help="Path to results.json."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of candidate clips to print."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.results_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    print(f"video_path: {payload.get('video_path')}")
    print(f"num_total_clips: {payload.get('num_total_clips')}")
    print(f"num_candidate_clips: {payload.get('num_candidate_clips')}")
    print(f"num_events: {len(payload.get('events', []))}")
    print("=" * 60)

    for event in payload.get("events", []):
        print(
            f"event_id={event['event_id']} "
            f"label={event['label']} "
            f"time=({event['start_time']:.2f}s ~ {event['end_time']:.2f}s) "
            f"confidence={event['confidence']:.2f}"
        )
        print(f"evidence={event['evidence']}")
        print("-" * 60)

    print("Top candidate clips")
    print("=" * 60)

    candidate_clips = sorted(
        payload.get("candidate_clips", []),
        key=lambda item: item.get("candidate_score", 0.0),
        reverse=True
    )

    for candidate in candidate_clips[:args.top_k]:
        print(
            f"clip_id={candidate['clip_id']} "
            f"types={candidate['candidate_types']} "
            f"candidate_score={candidate['candidate_score']:.4f} "
            f"vlm_label={candidate['vlm_output']['label']} "
            f"vlm_confidence={candidate['vlm_output']['confidence']:.2f}"
        )


if __name__ == "__main__":
    main()
