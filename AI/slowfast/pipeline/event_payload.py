from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import cv2

from utils.video import save_video


DEFAULT_DESCRIPTION = "Possible physical aggression detected in the clip."


def _severity_from_confidence(confidence):
    confidence = float(confidence)
    if confidence >= 0.8:
        return "high"
    if confidence >= 0.6:
        return "medium"
    return "low"


def _extract_evidence(reasoning):
    reasoning = str(reasoning or "").strip()
    if not reasoning:
        return []

    lowered = reasoning.lower()
    cue_map = [
        ("push", "밀침"),
        ("pushed", "밀침"),
        ("grab", "붙잡음"),
        ("holding", "붙잡음"),
        ("shirt", "붙잡음"),
        ("kick", "발로 공격"),
        ("strike", "가격 동작"),
        ("punch", "가격 동작"),
        ("hit", "가격 동작"),
        ("struggle", "몸싸움"),
        ("confrontation", "물리적 대치"),
        ("attack", "공격 행동"),
        ("ground", "바닥에 쓰러짐"),
        ("fall", "넘어짐"),
        ("chasing", "추격"),
    ]

    evidence = []
    for needle, label in cue_map:
        if needle in lowered and label not in evidence:
            evidence.append(label)

    if evidence:
        return evidence[:3]
    return ["신체적 충돌 정황"]


def _thumbnail_times(event):
    start_time = float(event.get("start_time", 0.0))
    end_time = float(event.get("end_time", start_time))
    duration = max(end_time - start_time, 0.0)
    if duration <= 0:
        return [round(start_time, 3)]

    midpoint = start_time + (duration / 2.0)
    return [
        round(start_time, 3),
        round(midpoint, 3),
        round(end_time, 3),
    ]


def _collect_event_vlm_signals(event, score_by_clip, vlm_outputs):
    best_positive = None
    best_fallback = None
    fight_count = 0
    non_fight_count = 0

    for clip_id in event.get("clip_ids", []):
        clip_id = int(clip_id)
        score_item = score_by_clip.get(clip_id, {})
        if not score_item.get("vlm_called"):
            continue

        vlm_item = vlm_outputs.get(clip_id, {})
        parsed = vlm_item.get("parsed", {})
        reasoning = str(parsed.get("reasoning", "")).strip()
        if not reasoning:
            continue

        candidate = {
            "reasoning": reasoning,
            "label": str(parsed.get("label", "non_fight")),
            "confidence": float(parsed.get("confidence", 0.0)),
        }
        if candidate["label"] == "fight":
            fight_count += 1
            if best_positive is None or candidate["confidence"] > best_positive["confidence"]:
                best_positive = candidate
        else:
            non_fight_count += 1
        if best_fallback is None or candidate["confidence"] > best_fallback["confidence"]:
            best_fallback = candidate

    return {
        "best_positive": best_positive,
        "best_fallback": best_fallback,
        "fight_count": fight_count,
        "non_fight_count": non_fight_count,
    }


def _build_event_description(event, score_by_clip, vlm_outputs):
    signals = _collect_event_vlm_signals(event, score_by_clip, vlm_outputs)
    chosen = signals["best_positive"] or signals["best_fallback"]
    reasoning = chosen["reasoning"] if chosen else ""
    evidence = _extract_evidence(reasoning)
    duration_sec = float(event.get("duration_sec", 0.0))

    if signals["best_positive"] is not None:
        if "밀침" in evidence and "붙잡음" in evidence:
            description = "사람들 사이의 밀침과 붙잡음이 포함된 신체적 충돌이 감지되었습니다."
        elif "몸싸움" in evidence or "물리적 대치" in evidence:
            description = "사람들 사이의 몸싸움으로 보이는 신체적 충돌이 감지되었습니다."
        elif "가격 동작" in evidence or "발로 공격" in evidence:
            description = "타격성 공격이 의심되는 신체적 충돌이 감지되었습니다."
        elif "바닥에 쓰러짐" in evidence or "넘어짐" in evidence:
            description = "신체적 충돌 이후 넘어짐 정황이 감지되었습니다."
        else:
            description = "신체적 공격 또는 몸싸움으로 의심되는 장면이 감지되었습니다."
    else:
        if duration_sec >= 10.0:
            description = "지속적인 신체적 충돌로 의심되는 이상행동 구간이 감지되었습니다."
        else:
            description = "짧은 신체적 충돌 정황이 감지되었습니다."

    return description, evidence, reasoning or description


def _build_score_timeline(event, score_by_clip, vlm_outputs):
    timeline = []
    for clip_id in event.get("clip_ids", []):
        clip_id = int(clip_id)
        item = score_by_clip.get(clip_id)
        if item is None:
            continue
        record = {
            "clip_id": clip_id,
            "start_time_sec": float(item.get("start_time", 0.0)),
            "end_time_sec": float(item.get("end_time", 0.0)),
            "fast_score": float(item.get("fighting_prob", 0.0)),
            "fused_score": float(item.get("final_score", item.get("fighting_prob", 0.0))),
            "vlm_used": bool(item.get("vlm_called", False)),
        }
        if clip_id in vlm_outputs:
            parsed = vlm_outputs[clip_id].get("parsed", {})
            record["vlm_label"] = str(parsed.get("label", "non_fight"))
            record["vlm_confidence"] = float(parsed.get("confidence", 0.0))
        timeline.append(record)
    return timeline


def _build_model_info(config):
    fast_checkpoint = Path(config["fast_model"]["checkpoint_path"]).name
    config_path = config.get("_meta", {}).get("config_path")
    return {
        "fast_checkpoint": fast_checkpoint,
        "vlm_provider": str(config["vlm"].get("provider", "mock")),
        "vlm_sampled_frames": int(config["vlm"].get("sampled_frames", 0)),
        "pipeline_config": Path(config_path).name if config_path else None,
        "candidate_name": config.get("_meta", {}).get("candidate_name"),
    }


def _build_routing_info(config, event, score_by_clip):
    clip_ids = [int(clip_id) for clip_id in event.get("clip_ids", [])]
    vlm_call_count = sum(1 for clip_id in clip_ids if score_by_clip.get(clip_id, {}).get("vlm_called"))
    return {
        "selected_for_vlm": vlm_call_count > 0,
        "executed_vlm_calls": vlm_call_count,
        "router_prob_low": float(config["router"].get("prob_low", 0.0)),
        "router_prob_high": float(config["router"].get("prob_high", 1.0)),
        "uncertainty_threshold": float(config["router"].get("uncertainty_threshold", 0.0)),
    }


def build_event_payload_bundle(
    manifest,
    events,
    clip_scores,
    vlm_outputs,
    config,
    run_name,
    cctv_id=None,
    stream_id=None,
    status="new",
):
    video_path = Path(manifest["video_path"])
    video_id = video_path.stem
    created_at = manifest.get("created_at")
    fps = float(manifest.get("fps", 0.0))
    score_by_clip = {int(item["clip_id"]): item for item in clip_scores}
    model_info = _build_model_info(config)

    payload_events = []
    for index, event in enumerate(events):
        description, evidence, reasoning = _build_event_description(event, score_by_clip, vlm_outputs)
        confidence = float(event.get("confidence", event.get("peak_score", 0.0)))
        event_key = f"{video_id}_evt_{index + 1:04d}"
        payload_events.append(
            {
                "event_id": event_key,
                "cctv_id": cctv_id,
                "stream_id": stream_id,
                "video_id": video_id,
                "label": str(event.get("label", "fight")),
                "confidence": confidence,
                "severity": _severity_from_confidence(confidence),
                "start_time_sec": float(event.get("start_time", 0.0)),
                "end_time_sec": float(event.get("end_time", 0.0)),
                "duration_sec": float(event.get("duration_sec", 0.0)),
                "start_frame": int(event.get("start_frame", 0)),
                "end_frame": int(event.get("end_frame", 0)),
                "description": description,
                "evidence": evidence,
                "reasoning_short": reasoning,
                "thumbnail_times_sec": _thumbnail_times(event),
                "thumbnail_urls": [],
                "clip_video_url": None,
                "clip_video_path": None,
                "detected_at": created_at,
                "source_fps": fps,
                "clip_ids": [int(clip_id) for clip_id in event.get("clip_ids", [])],
                "score_timeline": _build_score_timeline(event, score_by_clip, vlm_outputs),
                "model_info": dict(model_info),
                "routing_info": _build_routing_info(config, event, score_by_clip),
                "status": status,
            }
        )

    return {
        "pipeline_run_id": run_name,
        "cctv_id": cctv_id,
        "stream_id": stream_id,
        "video_id": video_id,
        "model_version": model_info.get("candidate_name") or model_info.get("fast_checkpoint"),
        "generated_at": created_at,
        "events": payload_events,
    }


def attach_event_media(payload_bundle, frames, output_root):
    output_root = Path(output_root)
    media_root = output_root / "media"
    media_root.mkdir(parents=True, exist_ok=True)

    enriched = deepcopy(payload_bundle)
    for event in enriched.get("events", []):
        event_id = str(event["event_id"])
        start_frame = max(int(event.get("start_frame", 0)), 0)
        end_frame = min(int(event.get("end_frame", 0)), len(frames) - 1)
        if end_frame < start_frame or not frames:
            continue

        event_frames = frames[start_frame : end_frame + 1]
        clip_filename = f"{event_id}.mp4"
        clip_path = media_root / clip_filename
        save_video(event_frames, clip_path, float(event.get("source_fps", 30.0)))
        event["clip_video_path"] = str(clip_path)
        event["clip_video_url"] = str(Path("media") / clip_filename)

        thumbnail_urls = []
        thumbnail_times = event.get("thumbnail_times_sec", [])
        fps = float(event.get("source_fps", 30.0))
        start_time = float(event.get("start_time_sec", 0.0))
        for index, thumb_time in enumerate(thumbnail_times):
            relative_seconds = max(float(thumb_time) - start_time, 0.0)
            frame_offset = int(round(relative_seconds * fps))
            frame_index = min(max(start_frame + frame_offset, start_frame), end_frame)
            frame = frames[frame_index]
            thumb_filename = f"{event_id}_thumb_{index + 1}.jpg"
            thumb_path = media_root / thumb_filename
            cv2.imwrite(str(thumb_path), frame)
            thumbnail_urls.append(str(Path("media") / thumb_filename))
        event["thumbnail_urls"] = thumbnail_urls

    return enriched
