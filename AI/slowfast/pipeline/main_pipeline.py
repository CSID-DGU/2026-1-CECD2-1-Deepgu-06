from pathlib import Path

from models.vlm.infer import VLMRefiner
from pipeline.clip_generator import build_sliding_clips
from pipeline.event_payload import attach_event_media, build_event_payload_bundle
from pipeline.event_builder import build_events
from pipeline.fast_stage import score_clips_fast
from pipeline.fusion import fuse_scores
from pipeline.motion_summary import attach_motion_summaries
from pipeline.router import select_vlm_clips
from pipeline.uncertainty import attach_uncertainty
from utils.io import ensure_dir, write_json
from utils.time_utils import utc_now_iso
from utils.video import load_video_frames


def run_single_video_pipeline(video_path, config, run_name="single_video", verbose=True):
    frames, fps = load_video_frames(video_path)
    if not frames:
        raise ValueError(f"video has no frames: {video_path}")

    output_root = ensure_dir(Path(config["project"]["output_root"]) / run_name)
    if verbose:
        print(f"[video] frames={len(frames)} fps={fps:.3f}")

    clips = build_sliding_clips(
        frames=frames,
        fps=fps,
        temporal_window_sec=float(config["clip"]["temporal_window_sec"]),
        stride_sec=float(config["clip"]["stride_sec"]),
    )
    if verbose:
        print(f"[clips] generated={len(clips)}")

    scored = score_clips_fast(clips, config["fast_model"], config["clip"])
    scored = attach_uncertainty(
        scored,
        score_key="fighting_prob",
        alpha_entropy=float(config["router"].get("alpha_entropy", 0.7)),
        alpha_variance=float(config["router"].get("alpha_variance", 0.3)),
        variance_window=int(config["router"].get("variance_window", 5)),
    )
    scored = attach_motion_summaries(scored)

    vlm_outputs = {}
    selected_clip_ids = []
    if config["vlm"].get("enabled", True):
        selected_clip_ids = select_vlm_clips(scored, config["router"])
        refiner = VLMRefiner(config["vlm"])
        for item in scored:
            if int(item["clip_id"]) not in selected_clip_ids:
                continue
            vlm_outputs[int(item["clip_id"])] = refiner.score_clip(item)
    if verbose:
        print(f"[vlm] selected={len(selected_clip_ids)}")

    fused = fuse_scores(scored, vlm_outputs, config["fusion"])
    events, smoothed_scores = build_events(fused, config["thresholds"], fps=fps)
    if verbose:
        print(f"[events] generated={len(events)}")

    manifest = {
        "created_at": utc_now_iso(),
        "video_path": str(video_path),
        "fps": float(fps),
        "num_frames": len(frames),
        "run_name": run_name,
    }

    clip_manifest = [
        {
            "clip_id": int(item["clip_id"]),
            "start_frame": int(item["start_frame"]),
            "end_frame": int(item["end_frame"]),
        }
        for item in clips
    ]

    serializable_scores = []
    for item in smoothed_scores:
        payload = {key: value for key, value in item.items() if key != "frames"}
        serializable_scores.append(payload)

    write_json(output_root / "manifest.json", manifest)
    write_json(output_root / "events.json", events)
    write_json(output_root / "clip_scores.json", serializable_scores)
    write_json(output_root / "vlm_outputs.json", vlm_outputs)
    event_payload = build_event_payload_bundle(
        manifest=manifest,
        events=events,
        clip_scores=serializable_scores,
        vlm_outputs=vlm_outputs,
        config=config,
        run_name=run_name,
        cctv_id=config.get("deployment", {}).get("cctv_id"),
        stream_id=config.get("deployment", {}).get("stream_id"),
        status="new",
    )
    event_payload = attach_event_media(
        payload_bundle=event_payload,
        frames=frames,
        output_root=output_root,
    )
    write_json(output_root / "event_payload.json", event_payload)
    if config["outputs"].get("save_clip_manifest", True):
        write_json(output_root / "clip_manifest.json", clip_manifest)

    return {
        "manifest": manifest,
        "events": events,
        "clip_scores": serializable_scores,
        "event_payload": event_payload,
        "output_dir": str(output_root),
    }
