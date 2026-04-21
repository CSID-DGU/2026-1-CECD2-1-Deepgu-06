from utils.video import clip_frame_span, frames_to_seconds


def merge_candidate_events(candidates, clip_length, stride, fps, max_gap=1):
    if not candidates:
        return []

    candidates = sorted(candidates, key=lambda item: item["clip_id"])
    events = []
    current_event = None

    for candidate in candidates:
        clip_id = candidate["clip_id"]
        start_frame, end_frame = clip_frame_span(clip_id, clip_length, stride)
        label = candidate["vlm_output"]["label"]

        if label in {"normal", "uncertain"}:
            continue

        if (
            current_event is None or
            current_event["label"] != label or
            clip_id - current_event["last_clip_id"] > max_gap
        ):
            if current_event is not None:
                events.append(current_event)

            current_event = {
                "event_id": len(events),
                "label": label,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "confidence": candidate["vlm_output"]["confidence"],
                "evidence": candidate["vlm_output"]["evidence"],
                "clip_ids": [clip_id],
                "last_clip_id": clip_id
            }
            continue

        current_event["end_frame"] = end_frame
        current_event["clip_ids"].append(clip_id)
        current_event["last_clip_id"] = clip_id
        current_event["confidence"] = max(
            current_event["confidence"],
            candidate["vlm_output"]["confidence"]
        )

    if current_event is not None:
        events.append(current_event)

    for event in events:
        event["start_time"] = frames_to_seconds(event["start_frame"], fps)
        event["end_time"] = frames_to_seconds(event["end_frame"] + 1, fps)
        event.pop("last_clip_id", None)

    return events
