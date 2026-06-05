"""
Event-level VLM filter.

Fast model이 생성한 이벤트 후보를 VLM이 이벤트 단위로 검증.
- clip-level fusion 없음: fast_peak_score 보존
- 3단계 결정: accept / uncertain / reject
- peak_score 상한(peak_score_max) 초과 이벤트는 VLM 없이 자동 accept
"""
from __future__ import annotations

from typing import List, Dict, Any

import numpy as np


def extract_event_frames(
    all_frames: list,
    event: Dict[str, Any],
    n_frames: int,
    fps: float = 30.0,
    peak_margin_sec: float = 4.0,
) -> list:
    """
    Peak-biased frame sampling: peak_clip 주변 ±peak_margin_sec 구간에서
    ceil(n_frames * 0.75)프레임, 나머지는 이벤트 전체 구간에서 균일 샘플링.
    """
    ev_start = max(0, int(event["start_frame"]))
    ev_end = min(len(all_frames) - 1, int(event["end_frame"]))
    if ev_start > ev_end:
        return [all_frames[ev_start]] if ev_start < len(all_frames) else []

    def _uniform(lo, hi, k):
        lo, hi = max(0, lo), min(len(all_frames) - 1, hi)
        if lo >= hi:
            return [lo] * k
        total = hi - lo + 1
        if total <= k:
            return list(range(lo, hi + 1))
        return [lo + int(round(i * (total - 1) / (k - 1))) for i in range(k)]

    # peak 클립 위치 파악
    peak_clip_id = event.get("peak_clip_id")
    margin_frames = int(peak_margin_sec * fps)

    if peak_clip_id is not None:
        # peak 클립의 프레임 범위를 이벤트 clip_ids로 역추산
        # (clip_id는 0-based 인덱스, stride=fps 가정)
        # clip의 start_frame = clip_id * stride_frames
        # 하지만 정확한 값은 event의 start_frame + clip 위치로 추정
        clip_ids = event.get("clip_ids", [])
        if clip_ids:
            peak_pos = clip_ids.index(peak_clip_id) if peak_clip_id in clip_ids else len(clip_ids) // 2
            # 이벤트 내 clip 위치 비율로 peak frame 추정
            ratio = peak_pos / max(len(clip_ids) - 1, 1)
            peak_frame = ev_start + int(ratio * (ev_end - ev_start))
        else:
            peak_frame = (ev_start + ev_end) // 2

        p_start = max(ev_start, peak_frame - margin_frames)
        p_end = min(ev_end, peak_frame + margin_frames)

        n_peak = max(1, int(np.ceil(n_frames * 0.75)))
        n_full = n_frames - n_peak

        peak_idx = _uniform(p_start, p_end, n_peak)
        full_idx = _uniform(ev_start, ev_end, n_full) if n_full > 0 else []
        indices = sorted(set(peak_idx + full_idx))
        # 부족하면 peak 구간에서 추가
        while len(indices) < n_frames:
            extra = _uniform(p_start, p_end, n_frames - len(indices) + len(peak_idx))
            indices = sorted(set(indices + extra))
            if len(indices) >= n_frames:
                break
        indices = indices[:n_frames]
    else:
        # peak 정보 없으면 전체 균일
        indices = _uniform(ev_start, ev_end, n_frames)

    return [all_frames[i] for i in indices]


def _decide(vlm_score: float, accept_thr: float, reject_thr: float) -> str:
    if vlm_score >= accept_thr:
        return "accept"
    if vlm_score <= reject_thr:
        return "reject"
    return "uncertain"


def filter_events_by_vlm(
    events: List[Dict[str, Any]],
    all_frames: list,
    fps: float,
    vlm_config: Dict[str, Any],
    clip_lookup: Dict[int, Any] = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """
    Returns (kept_events, rejected_events, vlm_call_count).
    각 이벤트에 fast_peak_score / vlm_score / vlm_decision / vlm_called 필드 추가.
    """
    from models.vlm.infer import VLMRefiner

    accept_thr = float(vlm_config.get("accept_threshold", 0.65))
    reject_thr = float(vlm_config.get("reject_threshold", 0.35))
    uncertain_action = str(vlm_config.get("uncertain_action", "keep")).lower()
    peak_score_max = float(vlm_config.get("peak_score_max", 0.85))
    n_frames = int(vlm_config.get("sampled_frames_per_event", 8))
    peak_margin_sec = float(vlm_config.get("peak_margin_sec", 4.0))
    max_calls = int(vlm_config.get("max_calls_per_video", 9999))
    frames_per_clip = int(vlm_config.get("frames_per_clip", 6))
    max_clips_per_event = int(vlm_config.get("max_clips_per_event", 3))
    adaptive_threshold_sec = float(vlm_config.get("adaptive_threshold_sec", 0))
    adaptive_long_frames = int(vlm_config.get("adaptive_long_frames", frames_per_clip))
    adaptive_short_frames = int(vlm_config.get("adaptive_short_frames", frames_per_clip))

    refiner = VLMRefiner(vlm_config)
    kept = []
    rejected = []
    vlm_call_count = 0

    for event in events:
        peak_score = float(event.get("peak_score", 0.0))
        event["fast_peak_score"] = peak_score

        # 매우 확실한 이벤트 → VLM 없이 통과
        if peak_score > peak_score_max:
            event["vlm_called"] = False
            event["vlm_decision"] = "accept"
            event["vlm_score"] = None
            kept.append(event)
            continue

        # VLM 호출 한도 초과 → uncertain 처리
        if vlm_call_count >= max_calls:
            event["vlm_called"] = False
            event["vlm_decision"] = "uncertain"
            event["vlm_score"] = None
            if uncertain_action != "discard":
                kept.append(event)
            else:
                rejected.append(event)
            continue

        # 이벤트 프레임 추출 후 VLM 호출
        if clip_lookup is not None:
            # per-clip 방식: fast_score 상위 clip 선택 → 각 clip에 ResNet+BiGRU → VLM
            clip_ids = [int(cid) for cid in event.get("clip_ids", [])]
            clips = [
                clip_lookup[cid] for cid in clip_ids
                if cid in clip_lookup and clip_lookup[cid].get("frames")
            ]
            top_clips = sorted(clips, key=lambda x: x["fighting_prob"], reverse=True)
            top_clips = top_clips[:max_clips_per_event]

            if not top_clips:
                event["vlm_called"] = False
                event["vlm_decision"] = "uncertain"
                event["vlm_score"] = None
                event["vlm_clip_scores"] = []
                if uncertain_action != "discard":
                    kept.append(event)
                else:
                    rejected.append(event)
                continue

            # adaptive: 이벤트 길이에 따라 프레임 수 / 샘플링 방식 결정
            if adaptive_threshold_sec > 0:
                event_dur = (float(event.get("end_frame", 0)) - float(event.get("start_frame", 0))) / fps
                if event_dur > adaptive_threshold_sec:
                    clip_n_frames = adaptive_long_frames
                    clip_force_uniform = True
                else:
                    clip_n_frames = adaptive_short_frames
                    clip_force_uniform = False
            else:
                clip_n_frames = frames_per_clip
                clip_force_uniform = False

            clip_scores = []
            called_clip_ids = []
            for clip in top_clips:
                if vlm_call_count >= max_calls:
                    break
                result = refiner.score_clip_frames(
                    clip["frames"],
                    n_frames=clip_n_frames,
                    meta={
                        "fighting_prob": clip["fighting_prob"],
                        "duration_sec": 2.0,
                    },
                    force_uniform=clip_force_uniform,
                )
                clip_scores.append(float(result["score"]))
                called_clip_ids.append(clip.get("clip_id"))
                vlm_call_count += 1

            if not clip_scores:
                event["vlm_called"] = False
                event["vlm_decision"] = "uncertain"
                event["vlm_score"] = None
                event["vlm_clip_scores"] = []
                if uncertain_action != "discard":
                    kept.append(event)
                else:
                    rejected.append(event)
                continue

            vlm_score = max(clip_scores)
            event["vlm_clip_scores"] = clip_scores
            event["vlm_clips_called"] = called_clip_ids
        else:
            # fallback: 기존 event 전체 균등 샘플링
            event_frames = extract_event_frames(all_frames, event, n_frames, fps=fps,
                                                peak_margin_sec=peak_margin_sec)
            if not event_frames:
                event["vlm_called"] = False
                event["vlm_decision"] = "uncertain"
                event["vlm_score"] = None
                if uncertain_action != "discard":
                    kept.append(event)
                else:
                    rejected.append(event)
                continue

            result = refiner.score_event(event_frames, event_meta=event)
            vlm_call_count += 1
            vlm_score = float(result["score"])
        decision = _decide(vlm_score, accept_thr, reject_thr)

        event["vlm_called"] = True
        event["vlm_score"] = vlm_score
        event["vlm_decision"] = decision
        event["vlm_raw"] = result.get("raw_response", "")

        if decision == "reject":
            rejected.append(event)
            continue
        if decision == "uncertain" and uncertain_action == "discard":
            rejected.append(event)
            continue
        kept.append(event)

    return kept, rejected, vlm_call_count
