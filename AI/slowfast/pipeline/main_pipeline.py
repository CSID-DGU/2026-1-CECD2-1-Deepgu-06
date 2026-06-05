from pathlib import Path

from pipeline.clip_generator import build_sliding_clips
from pipeline.event_payload import attach_event_media, build_event_payload_bundle
from pipeline.event_builder import build_events
from pipeline.event_vlm_filter import filter_events_by_vlm
from pipeline.fast_stage import score_clips_fast
from pipeline.fusion import fuse_scores
from utils.io import ensure_dir, write_json
from utils.time_utils import utc_now_iso
from utils.video import load_video_frames


def run_single_video_pipeline(video_path, config, run_name="single_video", verbose=True):
    frames, fps = load_video_frames(video_path)
    if not frames:
        raise ValueError(f"video has no frames: {video_path}")

    outputs_config = config.get("outputs", {})
    save_run_artifacts = bool(outputs_config.get("save_run_artifacts", True))
    save_event_media = bool(outputs_config.get("save_event_media", True))
    save_clip_manifest = bool(outputs_config.get("save_clip_manifest", True))

    output_root = None
    if save_run_artifacts:
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

    vlm_enabled = bool(config["vlm"].get("enabled", True))
    vlm_level = str(config["vlm"].get("level", "clip")).lower()

    if vlm_enabled and vlm_level == "event":
        # Event-level VLM: fast-only event detection → VLM judges each event
        fused = fuse_scores(scored, {}, config["fusion"])
        events, smoothed_scores = build_events(fused, config["thresholds"], fps=fps)
        if verbose:
            print(f"[events] candidates={len(events)}")
        events, rejected_events, n_vlm_calls = filter_events_by_vlm(events, frames, fps, config["vlm"])
        vlm_outputs = {}
        if verbose:
            decisions = {}
            for e in events + rejected_events:
                d = e.get("vlm_decision", "?")
                decisions[d] = decisions.get(d, 0) + 1
            print(f"[vlm] event-level calls={n_vlm_calls}, kept={len(events)}, "
                  f"rejected={len(rejected_events)}, decisions={decisions}")
    else:
        # Fast-only 경로 (vlm.enabled=false). 운영 파이프라인은 event-level VLM만 사용
        # clip-level VLM 경로(router 라우팅 + clip 단위 score_clip 융합) 제거.
        if vlm_enabled:
            raise NotImplementedError(
                "clip-level VLM 경로는 제거되었습니다. vlm.level='event'(event-level VLM) 또는 "
                "vlm.enabled=false(fast-only)로 설정하세요."
            )
        vlm_outputs = {}
        fused = fuse_scores(scored, vlm_outputs, config["fusion"])
        events, smoothed_scores = build_events(fused, config["thresholds"], fps=fps)
        rejected_events = []

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

    event_payload = None
    if save_run_artifacts:
        write_json(output_root / "manifest.json", manifest)
        write_json(output_root / "events.json", events)
        write_json(output_root / "clip_scores.json", serializable_scores)
        if rejected_events:
            write_json(output_root / "rejected_events.json", rejected_events)
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
        if save_event_media:
            event_payload = attach_event_media(
                payload_bundle=event_payload,
                frames=frames,
                output_root=output_root,
            )
        write_json(output_root / "event_payload.json", event_payload)
        if save_clip_manifest:
            write_json(output_root / "clip_manifest.json", clip_manifest)

    return {
        "manifest": manifest,
        "events": events,
        "rejected_events": rejected_events,
        "clip_scores": serializable_scores,
        "event_payload": event_payload,
        "output_dir": str(output_root) if output_root is not None else None,
    }
