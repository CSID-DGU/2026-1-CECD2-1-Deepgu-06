import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.event_payload import attach_event_media, build_event_payload_bundle
from utils.config import load_config
from utils.io import load_json, write_json
from utils.video import load_video_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/home/deepgu/slowfast/configs/base.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cctv-id", type=int, default=None)
    parser.add_argument("--stream-id", default=None)
    parser.add_argument("--candidate-name", default=None)
    parser.add_argument("--status", default="new")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    manifest = load_json(output_dir / "manifest.json")
    events = load_json(output_dir / "events.json")
    clip_scores = load_json(output_dir / "clip_scores.json")
    vlm_outputs = load_json(output_dir / "vlm_outputs.json")

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

    payload = build_event_payload_bundle(
        manifest=manifest,
        events=events,
        clip_scores=clip_scores,
        vlm_outputs=vlm_outputs,
        config=config,
        run_name=manifest.get("run_name", output_dir.name),
        cctv_id=config.get("deployment", {}).get("cctv_id"),
        stream_id=config.get("deployment", {}).get("stream_id"),
        status=args.status,
    )
    video_path = manifest.get("video_path")
    if video_path:
        frames, _ = load_video_frames(video_path)
        payload = attach_event_media(
            payload_bundle=payload,
            frames=frames,
            output_root=output_dir,
        )

    output_path = Path(args.output) if args.output else output_dir / "event_payload.json"
    write_json(output_path, payload)
    print(f"[done] event_payload={output_path}")


if __name__ == "__main__":
    main()
