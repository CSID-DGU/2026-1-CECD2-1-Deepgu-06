"""
Streaming worker using ai_wrapper.

Usage:
    python worker_main.py --stream rtmp://... --camera-id CAM-001 --cctv-id 1

Environment variables:
    STREAM_BUFFER_SEC           Frame buffer duration (default: 12.0)
    INFER_INTERVAL_SEC          Inference stride interval (default: 2.0)
    MIN_BUFFER_SEC              Minimum buffer before inference starts (default: 6.0)
    EVENT_CONFIDENCE_THRESHOLD  Minimum confidence to report event (default: 0.45)
    RECONNECT_SEC               Reconnect delay on stream failure (default: 5)
    POST_COOLDOWN_SEC           Min seconds between posts (default: 30.0)
    FIGHT_END_SEC               No-detection grace period before fight session ends (default: 5.0)
    MAX_FIGHT_SEC               Max fight session clip length in seconds (default: 120.0)
    DEBUG_SCORE_LOG             Print score summary each cycle (default: 1)
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import cv2

THIS_DIR = Path(__file__).resolve().parent
SLOWFAST_ROOT = THIS_DIR / "slowfast"
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(SLOWFAST_ROOT) not in sys.path:
    sys.path.insert(0, str(SLOWFAST_ROOT))

import ai_wrapper
import yolo_boxer
from utils.video import save_video


BUFFER_SEC           = float(os.getenv("STREAM_BUFFER_SEC", "12.0"))
INFER_INTERVAL_SEC   = float(os.getenv("INFER_INTERVAL_SEC", "2.0"))
MIN_BUFFER_SEC       = float(os.getenv("MIN_BUFFER_SEC", "6.0"))
CONFIDENCE_THRESHOLD = float(os.getenv("EVENT_CONFIDENCE_THRESHOLD", "0.45"))
RECONNECT_SEC        = float(os.getenv("RECONNECT_SEC", "5"))
POST_COOLDOWN_SEC    = float(os.getenv("POST_COOLDOWN_SEC", "30.0"))
# [수정] 마지막 fight 감지 후 이 시간(초) 동안 추가 감지 없으면 fight session 종료
FIGHT_END_SEC        = float(os.getenv("FIGHT_END_SEC", "5.0"))
# [수정] fight session 최대 클립 길이 (메모리 보호)
MAX_FIGHT_SEC        = float(os.getenv("MAX_FIGHT_SEC", "120.0"))
DEBUG_SCORE_LOG      = os.getenv("DEBUG_SCORE_LOG", "1").lower() not in {"0", "false", "no"}


def _now_utc():
    return datetime.now(timezone.utc)


def _buf_frames(fps: float, sec: float) -> int:
    return max(1, int(round(max(fps, 1.0) * sec)))


def _log_result(result: dict):
    if not DEBUG_SCORE_LOG:
        return
    clip_scores = result.get("clip_scores", [])
    if clip_scores:
        max_prob  = max(float(c.get("fighting_prob", 0)) for c in clip_scores)
        max_final = max(float(c.get("final_score", 0)) for c in clip_scores)
        vlm_count = sum(1 for c in clip_scores if c.get("vlm_called"))
    else:
        max_prob = max_final = 0.0
        vlm_count = 0
    events = result.get("event_payload", {}).get("events", [])
    print(
        f"[worker_main] clips={len(clip_scores)} vlm={vlm_count} "
        f"max_prob={max_prob:.4f} max_final={max_final:.4f} "
        f"events={len(events)}"
    )


def _post_fight_session(
    fight_frames: List,
    fps: float,
    fight_started_at: datetime,
    best_payload: dict,
    best_event: dict,
):
    """누적된 fight session 프레임을 하나의 클립으로 저장하고 포스팅."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, prefix="fight_session_")
    tmp.close()
    clip_path = tmp.name

    try:
        save_video(fight_frames, clip_path, fps)

        try:
            yolo_boxer.annotate_clip(clip_path)
        except Exception as exc:
            print(f"[worker_main] YOLO skipped: {exc}")

        # [수정] 누적 fight session 클립으로 clip_video_path 교체
        duration_sec = len(fight_frames) / max(fps, 1.0)
        updated_event = dict(best_event)
        updated_event["clip_video_path"] = clip_path
        # [수정] started_at/ended_at/duration_sec 을 합쳐진 세션 전체 기준으로 맞춤.
        # (기존엔 started_at만 덮어써 ended_at이 best 윈도우 값으로 남아 클립 길이와 불일치)
        updated_event["started_at"] = fight_started_at.isoformat()
        updated_event["ended_at"] = (fight_started_at + timedelta(seconds=duration_sec)).isoformat()
        updated_event["duration_sec"] = round(duration_sec, 3)

        payload = dict(best_payload)
        payload["events"] = [updated_event]
        print(
            f"[worker_main] posting fight session "
            f"frames={len(fight_frames)} duration={duration_sec:.1f}s "
            f"conf={float(best_event.get('confidence', 0)):.2f}"
        )
        response = ai_wrapper.post_event_payload(payload)
        print(f"[worker_main] backend response={response}")
    finally:
        try:
            os.remove(clip_path)
        except OSError:
            pass


def run_stream(stream_url: str, camera_id: str, cctv_id: Optional[int], stream_id: Optional[str]):
    while True:
        print(f"[worker_main] connecting: {stream_url}")
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            print(f"[worker_main] open failed, retry in {RECONNECT_SEC}s")
            time.sleep(RECONNECT_SEC)
            continue

        fps       = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        max_buf   = _buf_frames(fps, BUFFER_SEC)
        min_ready = _buf_frames(fps, MIN_BUFFER_SEC)
        stride    = _buf_frames(fps, INFER_INTERVAL_SEC)
        max_fight = _buf_frames(fps, MAX_FIGHT_SEC)
        print(f"[worker_main] connected fps={fps:.2f} buf={max_buf} stride={stride}")

        frame_buf  = deque(maxlen=max_buf)
        ts_buf     = deque(maxlen=max_buf)
        frame_count = 0
        last_posted_at: Optional[datetime] = None

        # [수정] fight session 추적 상태
        fight_frames: List          = []
        fight_started_at: Optional[datetime] = None
        fight_last_detect_at: Optional[datetime] = None
        best_payload: Optional[dict] = None
        best_event: Optional[dict]   = None
        new_frames_since_stride: List = []

        while True:
            ok, frame = cap.read()
            if not ok:
                print("[worker_main] stream disconnected, reconnecting...")
                # [수정] 연결 끊기기 전 진행 중인 fight session 마무리 후 포스팅
                if fight_last_detect_at is not None and best_payload is not None:
                    print("[worker_main] stream lost — finalizing fight session")
                    _post_fight_session(
                        fight_frames, fps, fight_started_at, best_payload, best_event
                    )
                    last_posted_at = _now_utc()
                break
            if frame is None:
                continue

            frame_buf.append(frame.copy())
            new_frames_since_stride.append(frame.copy())
            ts_buf.append(_now_utc())
            frame_count += 1

            if frame_count % stride != 0:
                continue
            if len(frame_buf) < min_ready:
                continue

            frames_added = new_frames_since_stride[:]
            new_frames_since_stride.clear()
            clip_started_at = ts_buf[0]

            try:
                result = ai_wrapper.analyze_buffered_frames(
                    frames=list(frame_buf),
                    fps=fps,
                    camera_id=camera_id,
                    cctv_id=cctv_id,
                    stream_id=stream_id or camera_id,
                    clip_started_at=clip_started_at,
                    filter_new_events=False,  # [수정] 중복 필터는 여기서 직접 관리
                )
                _log_result(result)

                payload = result.get("event_payload", {})
                events  = [
                    e for e in payload.get("events", [])
                    if float(e.get("confidence", 0)) >= CONFIDENCE_THRESHOLD
                ]
                top_event = (
                    max(events, key=lambda e: float(e.get("confidence", 0)))
                    if events else None
                )

                if top_event:
                    if fight_started_at is None:
                        # 새 fight session 시작: 현재 버퍼 전체로 초기화
                        fight_started_at = clip_started_at
                        fight_frames = list(frame_buf)
                        print(
                            f"[worker_main] fight session started "
                            f"conf={float(top_event.get('confidence', 0)):.2f}"
                        )
                    else:
                        # 기존 session에 새 프레임 추가
                        fight_frames.extend(frames_added)
                        if len(fight_frames) > max_fight:
                            fight_frames = fight_frames[-max_fight:]

                    fight_last_detect_at = _now_utc()
                    # 가장 높은 confidence의 payload/event 유지
                    if best_event is None or float(top_event.get("confidence", 0)) >= float(
                        best_event.get("confidence", 0)
                    ):
                        best_payload = payload
                        best_event   = top_event

                    session_sec = len(fight_frames) / max(fps, 1.0)
                    print(
                        f"[worker_main] fight ongoing "
                        f"session={session_sec:.1f}s "
                        f"conf={float(top_event.get('confidence', 0)):.2f}"
                    )

                else:
                    print("[worker_main] no fight detected")

                    if fight_last_detect_at is not None:
                        elapsed = (_now_utc() - fight_last_detect_at).total_seconds()
                        if elapsed >= FIGHT_END_SEC:
                            # POST_COOLDOWN_SEC 체크
                            if last_posted_at is not None:
                                post_elapsed = (_now_utc() - last_posted_at).total_seconds()
                                if post_elapsed < POST_COOLDOWN_SEC:
                                    print(
                                        f"[worker_main] cooldown ({post_elapsed:.1f}s / "
                                        f"{POST_COOLDOWN_SEC}s), session discarded"
                                    )
                                    fight_frames       = []
                                    fight_started_at   = None
                                    fight_last_detect_at = None
                                    best_payload       = None
                                    best_event         = None
                                    continue

                            _post_fight_session(
                                fight_frames, fps, fight_started_at, best_payload, best_event
                            )
                            last_posted_at       = _now_utc()
                            fight_frames         = []
                            fight_started_at     = None
                            fight_last_detect_at = None
                            best_payload         = None
                            best_event           = None

            except Exception as exc:
                import traceback
                print(f"[worker_main] error: {exc}")
                traceback.print_exc()

        cap.release()
        time.sleep(RECONNECT_SEC)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream",    required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--cctv-id",   type=int, default=None)
    parser.add_argument("--stream-id", default=None)
    args = parser.parse_args()

    run_stream(
        stream_url=args.stream,
        camera_id=args.camera_id,
        cctv_id=args.cctv_id,
        stream_id=args.stream_id,
    )
