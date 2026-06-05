"""
Event-level VLM filter.

Fast model이 생성한 이벤트 후보를 VLM이 이벤트 단위로 검증.
- clip-level fusion 없음: fast_peak_score 보존
- 3단계 결정: accept / uncertain / reject (uncertain은 기본 keep)
- 모든 이벤트 후보를 VLM에 통과시킨다.
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
    sampling_mode: str = "peak",
    adaptive_threshold_sec: float = 20.0,
    uniform_frames: int = 12,
    peak_frames: int = 8,
) -> list:
    """
    프레임 샘플링.
    - sampling_mode="peak"(기본): peak_clip 주변 ±peak_margin_sec 구간에서
      ceil(n_frames * 0.75)프레임, 나머지는 이벤트 전체 구간에서 균일 샘플링.
    - sampling_mode="uniform": peak 편향 없이 이벤트 전체 구간을 균등 샘플링.
    - sampling_mode="adaptive": duration >= adaptive_threshold_sec 이면 uniform
      (uniform_frames장), 미만이면 peak (peak_frames장). 긴 이벤트의 peak 추정
      오정렬로 인한 TP 손실은 uniform으로 회수, 짧은/집중형 이벤트는 peak 유지.
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

    # 유효 모드/프레임 수 결정 (adaptive는 길이로 분기)
    mode = str(sampling_mode).lower()
    k = int(n_frames)
    if mode == "adaptive":
        dur_sec = (ev_end - ev_start + 1) / max(fps, 1e-6)
        if dur_sec >= adaptive_threshold_sec:
            mode, k = "uniform", int(uniform_frames)
        else:
            mode, k = "peak", int(peak_frames)
    event["_sampling_used"] = mode  # 분석용: 실제 적용된 모드/프레임 수
    event["_sampling_k"] = k

    if mode == "uniform":
        return [all_frames[i] for i in _uniform(ev_start, ev_end, k)]

    # peak 클립 위치 파악
    peak_clip_id = event.get("peak_clip_id")
    margin_frames = int(peak_margin_sec * fps)

    if peak_clip_id is not None:
        # peak 클립의 프레임 범위를 이벤트 clip_ids로 역추산
        clip_ids = event.get("clip_ids", [])
        if clip_ids:
            peak_pos = clip_ids.index(peak_clip_id) if peak_clip_id in clip_ids else len(clip_ids) // 2
            ratio = peak_pos / max(len(clip_ids) - 1, 1)
            peak_frame = ev_start + int(ratio * (ev_end - ev_start))
        else:
            peak_frame = (ev_start + ev_end) // 2

        p_start = max(ev_start, peak_frame - margin_frames)
        p_end = min(ev_end, peak_frame + margin_frames)

        n_peak = max(1, int(np.ceil(k * 0.75)))
        n_full = k - n_peak

        peak_idx = _uniform(p_start, p_end, n_peak)
        full_idx = _uniform(ev_start, ev_end, n_full) if n_full > 0 else []
        indices = sorted(set(peak_idx + full_idx))
        # 부족하면 peak 구간에서 추가
        while len(indices) < k:
            extra = _uniform(p_start, p_end, k - len(indices) + len(peak_idx))
            indices = sorted(set(indices + extra))
            if len(indices) >= k:
                break
        indices = indices[:k]
    else:
        # peak 정보 없으면 전체 균일
        indices = _uniform(ev_start, ev_end, k)

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
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """
    Returns (kept_events, rejected_events, vlm_call_count).
    각 이벤트에 fast_peak_score / vlm_score / vlm_decision / vlm_called 필드 추가.
    """
    from models.vlm.infer import VLMRefiner

    prompt_mode = str(vlm_config.get("prompt_mode", "binary")).lower()
    accept_thr = float(vlm_config.get("accept_threshold", 0.65))
    reject_thr = float(vlm_config.get("reject_threshold", 0.35))
    uncertain_action = str(vlm_config.get("uncertain_action", "keep")).lower()
    n_frames = int(vlm_config.get("sampled_frames_per_event", 8))
    peak_margin_sec = float(vlm_config.get("peak_margin_sec", 4.0))
    sampling_mode = str(vlm_config.get("sampling_mode", "peak")).lower()
    adaptive_threshold_sec = float(vlm_config.get("adaptive_threshold_sec", 20.0))
    uniform_frames = int(vlm_config.get("uniform_frames", 12))
    peak_frames = int(vlm_config.get("peak_frames", 8))

    refiner = VLMRefiner(vlm_config)
    kept = []
    rejected = []
    vlm_call_count = 0

    for event in events:
        peak_score = float(event.get("peak_score", 0.0))
        event["fast_peak_score"] = peak_score

        # 이벤트 프레임 추출
        event_frames = extract_event_frames(all_frames, event, n_frames, fps=fps,
                                            peak_margin_sec=peak_margin_sec,
                                            sampling_mode=sampling_mode,
                                            adaptive_threshold_sec=adaptive_threshold_sec,
                                            uniform_frames=uniform_frames,
                                            peak_frames=peak_frames)
        if not event_frames:
            event["vlm_called"] = False
            event["vlm_decision"] = "uncertain"
            event["vlm_score"] = None
            if uncertain_action != "discard":
                kept.append(event)
            else:
                rejected.append(event)
            continue

        event["vlm_called"] = True

        if prompt_mode == "3label":
            # 3-label 모드: level 0만 reject, level 1/2는 keep
            result = refiner.score_event_3label(event_frames, event_meta=event)
            vlm_call_count += 1
            level = int(result["level"])
            event["vlm_level"] = level
            event["vlm_score"] = float(result["score"])
            event["vlm_decision"] = "reject" if level == 0 else "accept"
            event["vlm_raw"] = result.get("raw_response", "")
            if level == 0:
                rejected.append(event)
            else:
                kept.append(event)
        elif prompt_mode == "v3":
            # v3: violence-related event arc 전체(언쟁→사후)를 keep 기준으로 판단
            result = refiner.score_event_v3(event_frames, event_meta=event)
            vlm_call_count += 1
            vlm_score = float(result["score"])
            decision = _decide(vlm_score, accept_thr, reject_thr)
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
        elif prompt_mode == "v4":
            # v4: guiding sub-question 분해. 점수/임계 로직은 v3와 동일,
            #     응답의 sub-question 불리언만 vlm_subq에 추가 기록(분석용).
            result = refiner.score_event_v4(event_frames, event_meta=event)
            vlm_call_count += 1
            vlm_score = float(result["score"])
            decision = _decide(vlm_score, accept_thr, reject_thr)
            event["vlm_score"] = vlm_score
            event["vlm_decision"] = decision
            event["vlm_raw"] = result.get("raw_response", "")
            event["vlm_subq"] = result.get("subq")
            if decision == "reject":
                rejected.append(event)
                continue
            if decision == "uncertain" and uncertain_action == "discard":
                rejected.append(event)
                continue
            kept.append(event)
        else:
            # binary 모드 (기존)
            result = refiner.score_event(event_frames, event_meta=event)
            vlm_call_count += 1
            vlm_score = float(result["score"])
            decision = _decide(vlm_score, accept_thr, reject_thr)
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
