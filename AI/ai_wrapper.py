from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

import worker_event_payload as stream_worker
from pipeline.main_pipeline import run_single_video_pipeline


_SENT_CACHE: Dict[tuple, datetime] = {}


def analyze_video_file(
    video_path: str,
    camera_id: str,
    cctv_id: Optional[int] = None,
    stream_id: Optional[str] = None,
    run_name: Optional[str] = None,
    verbose: bool = False,
):
    """Run the SlowFast pipeline on a single video file and return the full result."""
    config = stream_worker._load_runtime_config(
        cctv_id=cctv_id,
        stream_id=stream_id or camera_id,
    )
    resolved_run_name = run_name or stream_worker._make_run_name(camera_id)
    return run_single_video_pipeline(
        video_path,
        config,
        run_name=resolved_run_name,
        verbose=verbose,
    )



def analyze_buffered_frames(
    frames: List,
    fps: float,
    camera_id: str,
    cctv_id: Optional[int] = None,
    stream_id: Optional[str] = None,
    clip_started_at: Optional[datetime] = None,
    run_name: Optional[str] = None,
    verbose: bool = False,
    filter_new_events: bool = True,
):
    """Persist a frame buffer to a temp clip, run inference, and return the enriched result."""
    if not frames:
        raise ValueError('frames must not be empty')

    temp_video = stream_worker._save_temp_clip(frames, fps)
    resolved_started_at = clip_started_at or stream_worker._now_utc()
    try:
        result = analyze_video_file(
            video_path=temp_video,
            camera_id=camera_id,
            cctv_id=cctv_id,
            stream_id=stream_id,
            run_name=run_name,
            verbose=verbose,
        )
        payload = stream_worker._absolutize_events(
            payload=result['event_payload'],
            clip_started_at=resolved_started_at,
            output_dir=result['output_dir'],
        )
        if filter_new_events:
            payload = stream_worker._filter_new_events(payload, _SENT_CACHE)
        result['event_payload'] = payload
        return result
    finally:
        try:
            os.remove(temp_video)
        except OSError:
            pass



def post_event_payload(payload: Dict):
    """Forward an event payload to the configured backend endpoint."""
    return stream_worker._post_payload(payload)


__all__ = [
    'analyze_video_file',
    'analyze_buffered_frames',
    'post_event_payload',
]
