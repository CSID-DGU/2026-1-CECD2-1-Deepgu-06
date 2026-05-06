import numpy as np


def summarize_clip_motion(frames):
    if len(frames) < 2:
        return "No strong motion is observed."

    diffs = []
    for prev, cur in zip(frames[:-1], frames[1:]):
        diffs.append(float(np.mean(np.abs(cur.astype(np.float32) - prev.astype(np.float32)))))

    motion = float(np.mean(diffs)) / 255.0
    if motion >= 0.18:
        return "Abrupt motion increase is observed in this clip."
    if motion >= 0.10:
        return "Possible rapid interpersonal interaction is visible."
    return "No strong motion surge is observed."


def attach_motion_summaries(scored_clips):
    updated = []
    for item in scored_clips:
        cloned = dict(item)
        cloned["motion_summary"] = summarize_clip_motion(item["frames"])
        updated.append(cloned)
    return updated
