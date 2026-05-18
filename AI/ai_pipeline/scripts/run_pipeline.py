import argparse
import json
import sys
from pathlib import Path


ROOT = Path("/home/deepgu/test")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.main_pipeline import run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run the fight anomaly pipeline.")
    parser.add_argument(
        "--video-path",
        default="/home/deepgu/test/data/raw_videos/Abuse007_x264.mp4",
        help="Input video path."
    )
    parser.add_argument(
        "--clip-dir",
        default="/home/deepgu/test/data/clips",
        help="Legacy all-clips directory. Ignored when outputs.root_dir is set in config."
    )
    parser.add_argument(
        "--config-path",
        default="/home/deepgu/test/configs/fight_pipeline_config.json",
        help="Pipeline config path."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = run_pipeline(
        video_path=args.video_path,
        clip_dir=args.clip_dir,
        config_path=args.config_path
    )

    print(json.dumps({
        "video_path": results["video_path"],
        "num_total_clips": results["num_total_clips"],
        "num_candidate_clips": results["num_candidate_clips"],
        "num_events": len(results["events"]),
        "events": results["events"]
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
