from utils.video import frames_to_seconds


def smooth_scores(results, score_key, window_size=3, method="moving_average"):
    if window_size <= 1 or not results:
        return results

    radius = max(window_size // 2, 0)
    smoothed = []
    for index, result in enumerate(results):
        left = max(0, index - radius)
        right = min(len(results), index + radius + 1)
        window = [float(item[score_key]) for item in results[left:right]]

        cloned = dict(result)
        if method == "median":
            ordered = sorted(window)
            mid = len(ordered) // 2
            value = ordered[mid] if len(ordered) % 2 == 1 else (ordered[mid - 1] + ordered[mid]) / 2.0
        else:
            value = sum(window) / len(window)
        cloned[score_key] = float(value)
        smoothed.append(cloned)

    return smoothed


def split_events_by_low_score(events, scored_clips, thresholds):
    split_cfg = thresholds.get("split", {})
    if not split_cfg.get("enabled", False):
        return events

    score_threshold = float(split_cfg.get("score_threshold", thresholds.get("end_score", 0.35)))
    min_consecutive = max(1, int(split_cfg.get("min_consecutive_clips", 2)))
    score_by_clip = {int(item["clip_id"]): float(item["final_score"]) for item in scored_clips}

    split_events = []
    for event in events:
        clip_ids = [int(clip_id) for clip_id in event["clip_ids"]]
        if len(clip_ids) < min_consecutive + 1:
            split_events.append(event)
            continue

        segments = []
        segment_start = 0
        low_run = 0
        split_index = None
        for index, clip_id in enumerate(clip_ids):
            score = score_by_clip.get(clip_id, 0.0)
            if score < score_threshold:
                low_run += 1
            else:
                low_run = 0
            if low_run >= min_consecutive:
                split_index = index - min_consecutive + 1
                break

        if split_index is None or split_index <= segment_start:
            split_events.append(event)
            continue

        left_clip_ids = clip_ids[:split_index]
        right_clip_ids = clip_ids[split_index:]
        if not left_clip_ids or not right_clip_ids:
            split_events.append(event)
            continue
        segments.append(left_clip_ids)
        segments.append(right_clip_ids)

        for segment_clip_ids in segments:
            segment_scores = [score_by_clip[clip_id] for clip_id in segment_clip_ids]
            split_events.append(
                {
                    "event_id": len(split_events),
                    "label": event["label"],
                    "start_frame": event["start_frame"] if segment_clip_ids[0] == clip_ids[0] else None,
                    "end_frame": event["end_frame"] if segment_clip_ids[-1] == clip_ids[-1] else None,
                    "clip_ids": segment_clip_ids,
                    "peak_score": max(segment_scores),
                    "confidence": max(segment_scores),
                }
            )

    # Fill frame boundaries from clip ids.
    clip_bounds = {
        int(item["clip_id"]): (int(item["start_frame"]), int(item["end_frame"])) for item in scored_clips
    }
    normalized = []
    for event in split_events:
        start_frame = clip_bounds[event["clip_ids"][0]][0]
        end_frame = clip_bounds[event["clip_ids"][-1]][1]
        event["start_frame"] = start_frame
        event["end_frame"] = end_frame
        event["event_id"] = len(normalized)
        normalized.append(event)
    return normalized


def build_events(scored_clips, thresholds, fps):
    ordered = sorted(scored_clips, key=lambda item: int(item["clip_id"]))
    smoothing = thresholds.get("score_smoothing", {})
    if smoothing.get("enabled", False):
        ordered = smooth_scores(
            ordered,
            score_key="final_score",
            window_size=int(smoothing.get("window_size", 3)),
            method=smoothing.get("method", "moving_average"),
        )

    start_score = float(thresholds["start_score"])
    end_score = float(thresholds["end_score"])
    min_event_duration_sec = float(thresholds.get("min_event_duration_sec", 0.0))
    mean_score_threshold = float(thresholds.get("mean_score_threshold", 0.0))

    # clip_id → frame bounds 매핑 (peak clip 위치 추적용)
    clip_bounds = {int(r["clip_id"]): (int(r["start_frame"]), int(r["end_frame"])) for r in ordered}

    # 이벤트 연속성은 clip_id가 아니라 실제 프레임 위치(stride)로 판정한다.
    # clip_id는 호출자가 0..N으로 재번호할 수 있어(예: 필터링된 매니페스트 기반
    # 오프라인 스크립트) "clip_id == prev+1"가 시간 인접을 보장하지 않는다. 그 경우
    # 가운데가 빠진(비연속) 클립을 연속으로 오인해 갭 너머로 이벤트가 잘못 병합된다.
    # start_frame 간격이 stride를 초과하면 갭으로 보고 이벤트를 끊는다.
    # (라이브 파이프라인은 항상 간격==stride라 동작이 동일하다.)
    start_frames_sorted = [int(r["start_frame"]) for r in ordered]
    positive_gaps = [b - a for a, b in zip(start_frames_sorted, start_frames_sorted[1:]) if b - a > 0]
    stride_frames = min(positive_gaps) if positive_gaps else 1

    # Tripwire: 라이브 파이프라인은 항상 연속(간격==stride)이다. 간격>stride가 보이면
    # 필터링된(비연속) 클립 세트가 들어온 것 — event-level 평가/튜닝엔 부적합한 입력이다.
    # (예: ambiguous 클립을 버린 학습용 매니페스트/캐시. 자세한 경위는 claude_chat/20260603.md)
    n_gap_clips = sum(1 for g in positive_gaps if g > stride_frames)
    if n_gap_clips:
        import warnings
        warnings.warn(
            f"build_events: 비연속 클립 입력 감지 (간격>stride 지점 {n_gap_clips}개). "
            f"필터링된 매니페스트/캐시를 event 생성에 사용하면 안 됨 — 연속 클립 캐시를 쓸 것.",
            RuntimeWarning,
            stacklevel=2,
        )

    events = []
    current = None
    for result in ordered:
        score = float(result["final_score"])
        clip_id = int(result["clip_id"])
        if current is None:
            if score >= start_score:
                current = {
                    "event_id": len(events),
                    "label": "fight",
                    "start_frame": int(result["start_frame"]),
                    "end_frame": int(result["end_frame"]),
                    "clip_ids": [clip_id],
                    "peak_score": score,
                    "peak_clip_id": clip_id,
                    "confidence": score,
                }
            continue

        previous_clip_id = current["clip_ids"][-1]
        previous_start_frame = clip_bounds[previous_clip_id][0]
        is_contiguous = (int(result["start_frame"]) - previous_start_frame) <= stride_frames
        if score >= end_score and is_contiguous:
            current["end_frame"] = int(result["end_frame"])
            current["clip_ids"].append(clip_id)
            if score > current["peak_score"]:
                current["peak_score"] = score
                current["peak_clip_id"] = clip_id
            current["confidence"] = current["peak_score"]
            continue

        events.append(current)
        current = None
        if score >= start_score:
            current = {
                "event_id": len(events),
                "label": "fight",
                "start_frame": int(result["start_frame"]),
                "end_frame": int(result["end_frame"]),
                "clip_ids": [clip_id],
                "peak_score": score,
                "peak_clip_id": clip_id,
                "confidence": score,
            }

    if current is not None:
        events.append(current)

    events = split_events_by_low_score(events, ordered, thresholds)

    score_by_clip = {int(item["clip_id"]): float(item["final_score"]) for item in ordered}

    filtered = []
    for event in events:
        event["start_time"] = frames_to_seconds(event["start_frame"], fps)
        event["end_time"] = frames_to_seconds(event["end_frame"] + 1, fps)
        event["duration_sec"] = event["end_time"] - event["start_time"]
        if event["duration_sec"] < min_event_duration_sec:
            continue
        clip_scores = [score_by_clip.get(cid, 0.0) for cid in event.get("clip_ids", [])]
        event["mean_score"] = sum(clip_scores) / len(clip_scores) if clip_scores else 0.0
        if mean_score_threshold > 0.0 and event["mean_score"] < mean_score_threshold:
            continue
        event["event_id"] = len(filtered)
        filtered.append(event)

    return filtered, ordered
