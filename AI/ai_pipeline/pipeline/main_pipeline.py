from pipeline.clip_generator import generate_clips, save_clip
from models.tsm.inference import TSMInference
from models.tsm.scorer import compute_candidate_scores
from models.vlm.inference import InternVL
from pipeline.candidate_selector import select_candidates
from pipeline.frame_sampler import sample_from_candidates
from pipeline.event_merger import merge_candidate_events
from pipeline.result_writer import write_debug, write_results
from utils.config import load_label_map, load_pipeline_config
from utils.paths import ensure_dir
from pathlib import Path


def run_pipeline(video_path, clip_dir, config_path=None):
    config = load_pipeline_config(config_path)
    scorer_type = config.get("scorer", {}).get("type", "binary_fight")
    label_map = load_label_map() if scorer_type == "heuristic_multiclass" else None

    clip_config = config["clip"]
    sampling_config = config["sampling"]
    threshold_config = config["thresholds"]
    output_config = config["outputs"]
    preferred_gpus = config["devices"]["preferred_gpu_indices"]
    tsm_config = config["models"].get("tsm", {})
    vlm_config = config["models"].get("vlm", {})
    vlm_enabled = vlm_config.get("enabled", True)
    video_id = Path(video_path).stem

    output_root = output_config.get("root_dir")
    if output_root:
        video_output_dir = ensure_dir(Path(output_root) / video_id)
        all_clips_dir = ensure_dir(video_output_dir / "all_clips")
        candidate_dir = ensure_dir(video_output_dir / "candidate_clips")
        debug_dir = ensure_dir(video_output_dir / "debug")
        results_path = video_output_dir / "results.json"
    else:
        video_output_dir = None
        all_clips_dir = Path(clip_dir)
        candidate_dir = ensure_dir(output_config["candidate_clip_dir"])
        debug_dir = Path(output_config["debug_dir"])
        results_path = Path(output_config["results_path"])

    clips, video_meta = generate_clips(
        video_path,
        str(all_clips_dir),
        clip_len=clip_config["length"],
        stride=clip_config["stride"],
        save=clip_config["save_all_clips"],
        return_metadata=True,
    )
    source_fps = float(video_meta["fps"])
    candidate_clip_fps = float(video_meta["save_fps"])

    tsm = TSMInference(
        tsm_config["weight_path"],
        preferred_gpu_indices=preferred_gpus,
        num_segments=tsm_config.get("num_segments", 8),
        num_classes=tsm_config.get("num_classes", 1),
        binary_mode=tsm_config.get("binary_mode", True),
    )
    vlm = None
    if vlm_enabled:
        vlm = InternVL(
            model_name=vlm_config["model_name"],
            preferred_gpu_indices=preferred_gpus
        )

    results = []

    for i, clip in enumerate(clips):
        probs = tsm.predict(clip)
        scores = compute_candidate_scores(probs, label_map)
        log_parts = [
            f"[clip {i}]",
            f"fight_candidate={scores['fight_candidate_score']:.4f}",
        ]
        if "fight_prob" in scores:
            log_parts.append(f"fight_prob={scores['fight_prob']:.4f}")
        if "fall_candidate_score" in scores:
            log_parts.append(
                f"fall_candidate={scores['fall_candidate_score']:.4f}"
            )
        print(", ".join(log_parts))

        results.append({
            "clip_id": i,
            "clip": clip,
            "probs": probs.tolist() if hasattr(probs, "tolist") else probs,
            "scores": scores
        })

    candidates = select_candidates(results, threshold_config)

    candidates = sample_from_candidates(
        candidates,
        num_samples=sampling_config["num_samples"],
        strategy=sampling_config["strategy"]
    )

    for c in candidates:
        if vlm_enabled:
            frames = c["sampled_frames"]
            vlm_output = vlm.predict(frames)
        else:
            vlm_output = {
                "label": "fight",
                "confidence": float(c["candidate_score"]),
                "evidence": "binary_tsm_fight_scorer"
            }
        c["vlm_output"] = vlm_output

    for c in candidates:
        clip_id = c["clip_id"]
        clip_frames = c["clip"]

        save_path = candidate_dir / f"clip_{clip_id}.mp4"
        save_clip(clip_frames, str(save_path), fps=candidate_clip_fps)
        c["video_path"] = str(save_path)

    events = merge_candidate_events(
        candidates,
        clip_length=clip_config["length"],
        stride=clip_config["stride"],
        fps=source_fps,
        max_gap=threshold_config["merge_clip_gap"]
    )

    output_payload = {
        "video_path": video_path,
        "video_id": video_id,
        "config_path": config_path or "/home/deepgu/test/configs/pipeline_config.json",
        "output_dir": str(video_output_dir) if video_output_dir else None,
        "source_fps": source_fps,
        "candidate_clip_fps": candidate_clip_fps,
        "num_frames": int(video_meta["num_frames"]),
        "num_total_clips": len(results),
        "num_candidate_clips": len(candidates),
        "events": events,
        "candidate_clips": [
            {
                "clip_id": c["clip_id"],
                "candidate_types": c["candidate_types"],
                "candidate_score": c["candidate_score"],
                "scores": c["scores"],
                "vlm_output": c["vlm_output"],
                "video_path": c["video_path"]
            }
            for c in candidates
        ]
    }

    write_results(results_path, output_payload)
    write_debug(debug_dir, output_payload["candidate_clips"])

    print(f"결과 저장 완료: {results_path}")

    return output_payload
