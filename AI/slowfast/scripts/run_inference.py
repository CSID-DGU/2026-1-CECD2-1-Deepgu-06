import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.main_pipeline import run_single_video_pipeline
from utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/home/deepgu/slowfast/configs/base.yaml")
    parser.add_argument("--video", required=True)
    parser.add_argument("--run-name", default="single_video")
    parser.add_argument("--cctv-id", type=int, default=None)
    parser.add_argument("--stream-id", default=None)
    parser.add_argument("--candidate-name", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    config.setdefault("_meta", {})
    config["_meta"]["config_path"] = args.config
    if args.candidate_name:
        config["_meta"]["candidate_name"] = args.candidate_name
    if args.cctv_id is not None or args.stream_id is not None:
        config.setdefault("deployment", {})
        if args.cctv_id is not None:
            config["deployment"]["cctv_id"] = args.cctv_id
        if args.stream_id is not None:
            config["deployment"]["stream_id"] = args.stream_id
    result = run_single_video_pipeline(args.video, config, run_name=args.run_name, verbose=True)
    print(f"[done] output_dir={result['output_dir']}")


if __name__ == "__main__":
    main()
