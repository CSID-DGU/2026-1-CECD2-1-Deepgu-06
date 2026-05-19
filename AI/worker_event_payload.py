"""
Streaming worker for slowfast event-payload inference.

This worker does NOT modify the legacy `worker.py`.
It uses the slowfast pipeline to:

1. read a live stream
2. collect a recent frame buffer
3. save a short analysis snippet
4. run slowfast inference on the snippet
5. generate `event_payload.json`
6. optionally POST the payload to the backend

Usage:
    python worker_event_payload.py \
      --stream rtmp://... \
      --camera-id CAM-001 \
      --cctv-id 1

Environment variables:
    EVENT_PAYLOAD_API   Full backend endpoint for payload POST
                        default: http://43.201.17.169:8000/internal/event-payloads
    CALLBACK_SECRET     Backend callback secret header value
    PUBLIC_MEDIA_BASE_URL
                        Optional public base URL for media links
                        Example: https://example.com/ai-media
    SLOWFAST_CONFIG     slowfast config path
    SLOWFAST_CANDIDATE  candidate/model version string
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import requests


THIS_DIR = Path(__file__).resolve().parent
SLOWFAST_ROOT = THIS_DIR / "slowfast"
if str(SLOWFAST_ROOT) not in sys.path:
    sys.path.insert(0, str(SLOWFAST_ROOT))

from pipeline.main_pipeline import run_single_video_pipeline
from utils.config import load_config
from utils.video import save_video


EVENT_PAYLOAD_API = os.getenv(
    "EVENT_PAYLOAD_API",
    "http://43.201.17.169:8000/internal/event-payloads",
)
CALLBACK_SECRET = os.getenv("CALLBACK_SECRET", "deepgu")
PUBLIC_MEDIA_BASE_URL = os.getenv("PUBLIC_MEDIA_BASE_URL", "").rstrip("/")
SLOWFAST_CONFIG = os.getenv(
    "SLOWFAST_CONFIG",
    str(SLOWFAST_ROOT / "configs" / "base.yaml"),
)
SLOWFAST_CANDIDATE = os.getenv("SLOWFAST_CANDIDATE", "es_d_t3_mix_cap5")

# [수정] 스트리밍 환경에서 싸움 구간이 버퍼에 온전히 담기도록 버퍼 확대
BUFFER_SEC = float(os.getenv("STREAM_BUFFER_SEC", "12.0"))
INFER_INTERVAL_SEC = float(os.getenv("INFER_INTERVAL_SEC", "2.0"))
MIN_BUFFER_SEC = float(os.getenv("MIN_BUFFER_SEC", "6.0"))
EVENT_COOLDOWN_SEC = float(os.getenv("EVENT_COOLDOWN_SEC", "8.0"))
CONFIDENCE_THRESHOLD = float(os.getenv("EVENT_CONFIDENCE_THRESHOLD", "0.45"))
RECONNECT_SEC = float(os.getenv("RECONNECT_SEC", "5"))
DEBUG_SCORE_LOG = os.getenv("DEBUG_SCORE_LOG", "1").lower() not in {"0", "false", "no"}
# [수정] 버퍼 스니펫을 저장해서 실제 수신 영상 확인용 디버그 옵션
DEBUG_SAVE_SNIPPET_DIR = os.getenv("DEBUG_SAVE_SNIPPET_DIR", "")


def _now_utc():
    return datetime.now(timezone.utc)


def _estimate_stride_frames(fps: float) -> int:
    return max(1, int(round(max(fps, 1.0) * INFER_INTERVAL_SEC)))


def _estimate_buffer_frames(fps: float) -> int:
    return max(1, int(round(max(fps, 1.0) * BUFFER_SEC)))


def _minimum_ready_frames(fps: float) -> int:
    return max(1, int(round(max(fps, 1.0) * MIN_BUFFER_SEC)))


def _load_runtime_config(cctv_id: Optional[int], stream_id: Optional[str]):
    config = load_config(SLOWFAST_CONFIG)
    config.setdefault("_meta", {})
    config["_meta"]["config_path"] = SLOWFAST_CONFIG
    config["_meta"]["candidate_name"] = SLOWFAST_CANDIDATE
    config.setdefault("deployment", {})
    if cctv_id is not None:
        config["deployment"]["cctv_id"] = cctv_id
    if stream_id:
        config["deployment"]["stream_id"] = stream_id
    return config


def _make_run_name(camera_id: str) -> str:
    stamp = _now_utc().strftime("%Y%m%dT%H%M%S")
    safe_camera = camera_id.replace("/", "_").replace(" ", "_")
    return f"stream_{safe_camera}_{stamp}"


def _event_signature(event: Dict) -> Tuple[str, int]:
    started_at = event.get("started_at") or event.get("start_time_abs")
    label = str(event.get("label", "fight"))
    start_key = int(datetime.fromisoformat(started_at).timestamp()) if started_at else 0
    # [수정] 슬라이딩 윈도우마다 시작/종료 시간이 달라서 쿨다운이 안 걸리던 문제 수정
    # start_key를 쿨다운 단위로 버킷화 → 같은 사건의 겹치는 윈도우는 동일 키로 처리
    bucket = (start_key // int(EVENT_COOLDOWN_SEC)) * int(EVENT_COOLDOWN_SEC)
    return (label, bucket)


def _absolutize_events(payload: Dict, clip_started_at: datetime, output_dir: str) -> Dict:
    payload = dict(payload)
    payload["clip_started_at"] = clip_started_at.isoformat()
    payload["output_dir"] = output_dir

    events = []
    for event in payload.get("events", []):
        event = dict(event)
        start_time_sec = float(event.get("start_time_sec", 0.0))
        end_time_sec = float(event.get("end_time_sec", 0.0))
        started_at = clip_started_at + timedelta(seconds=start_time_sec)
        ended_at = clip_started_at + timedelta(seconds=end_time_sec)
        event["started_at"] = started_at.isoformat()
        event["ended_at"] = ended_at.isoformat()

        if PUBLIC_MEDIA_BASE_URL:
            if event.get("clip_video_url"):
                event["clip_video_url"] = (
                    f"{PUBLIC_MEDIA_BASE_URL}/{Path(output_dir).name}/{event['clip_video_url']}"
                )
            event["thumbnail_urls"] = [
                f"{PUBLIC_MEDIA_BASE_URL}/{Path(output_dir).name}/{thumb}"
                for thumb in event.get("thumbnail_urls", [])
            ]
        events.append(event)
    payload["events"] = events
    return payload


def _filter_new_events(payload: Dict, sent_cache: Dict[Tuple[str, int, int], datetime]) -> Dict:
    cutoff = _now_utc() - timedelta(seconds=EVENT_COOLDOWN_SEC)
    expired = [key for key, ts in sent_cache.items() if ts < cutoff]
    for key in expired:
        sent_cache.pop(key, None)

    accepted = []
    for event in payload.get("events", []):
        confidence = float(event.get("confidence", 0.0))
        if confidence < CONFIDENCE_THRESHOLD:
            continue
        signature = _event_signature(event)
        if signature in sent_cache:
            continue
        sent_cache[signature] = _now_utc()
        accepted.append(event)

    payload = dict(payload)
    payload["events"] = accepted
    return payload


def _post_payload(payload: Dict):
    headers = {}
    if CALLBACK_SECRET:
        headers["X-Callback-Secret"] = CALLBACK_SECRET
    data = {
        "payload": json.dumps(payload, ensure_ascii=False),
        "camera_id": str(payload.get("stream_id") or payload.get("cctv_id") or ""),
        "cctv_id": "" if payload.get("cctv_id") is None else str(payload.get("cctv_id")),
        "video_id": str(payload.get("video_id") or ""),
        "model_version": str(payload.get("model_version") or ""),
    }

    files = {}
    opened_files = []
    try:
        for event_index, event in enumerate(payload.get("events", [])):
            clip_path = event.get("clip_video_path")
            if clip_path and os.path.exists(clip_path):
                handle = open(clip_path, "rb")
                opened_files.append(handle)
                files[f"event_{event_index}_video"] = (
                    Path(clip_path).name,
                    handle,
                    "video/mp4",
                )

            for thumb_index, thumb_url in enumerate(event.get("thumbnail_urls", []), start=1):
                if PUBLIC_MEDIA_BASE_URL and thumb_url.startswith(PUBLIC_MEDIA_BASE_URL):
                    continue
                thumb_path = Path(payload.get("output_dir", "")) / thumb_url
                if not thumb_path.exists():
                    continue
                handle = open(thumb_path, "rb")
                opened_files.append(handle)
                files[f"event_{event_index}_thumb_{thumb_index}"] = (
                    thumb_path.name,
                    handle,
                    "image/jpeg",
                )

        response = requests.post(
            EVENT_PAYLOAD_API,
            headers=headers,
            data=data,
            files=files if files else None,
            timeout=60,
        )
        response.raise_for_status()
        try:
            return response.json()
        except Exception:
            return {"status_code": response.status_code, "text": response.text[:300]}
    finally:
        for handle in opened_files:
            try:
                handle.close()
            except Exception:
                pass


def _save_temp_clip(frames: List, fps: float) -> str:
    temp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp.close()
    save_video(frames, temp.name, fps)
    return temp.name


def _log_score_summary(result: Dict):
    if not DEBUG_SCORE_LOG:
        return

    clip_scores = result.get("clip_scores", [])
    events = result.get("events", [])
    payload = result.get("event_payload", {})
    payload_events = payload.get("events", [])

    if clip_scores:
        max_fast = max(float(item.get("fighting_prob", 0.0)) for item in clip_scores)
        max_final = max(float(item.get("final_score", 0.0)) for item in clip_scores)
        max_vlm = max(
            (
                float(item.get("vlm_score", 0.0))
                for item in clip_scores
                if item.get("vlm_score") is not None
            ),
            default=None,
        )
        selected_vlm = sum(1 for item in clip_scores if bool(item.get("vlm_called", False)))
    else:
        max_fast = 0.0
        max_final = 0.0
        max_vlm = None
        selected_vlm = 0

    event_confidences = [round(float(event.get("confidence", 0.0)), 4) for event in events]
    payload_confidences = [
        round(float(event.get("confidence", 0.0)), 4) for event in payload_events
    ]
    top_fast = sorted(
        (
            {
                "clip_id": int(item.get("clip_id", -1)),
                "prob": round(float(item.get("fighting_prob", 0.0)), 4),
                "final": round(float(item.get("final_score", 0.0)), 4),
                "raw_logit": (
                    None
                    if item.get("raw_logit") is None
                    else round(float(item.get("raw_logit", 0.0)), 4)
                ),
                "cal_logit": (
                    None
                    if item.get("calibrated_logit") is None
                    else round(float(item.get("calibrated_logit", 0.0)), 4)
                ),
                "temp": (
                    None
                    if item.get("temperature") is None
                    else round(float(item.get("temperature", 0.0)), 4)
                ),
                "time": (
                    round(float(item.get("start_time", 0.0)), 2),
                    round(float(item.get("end_time", 0.0)), 2),
                ),
                "vlm": bool(item.get("vlm_called", False)),
            }
            for item in clip_scores
        ),
        key=lambda row: row["prob"],
        reverse=True,
    )[:3]
    vlm_clip_ids = [int(item.get("clip_id", -1)) for item in clip_scores if bool(item.get("vlm_called", False))]
    start_threshold = float(result.get("thresholds", {}).get("start_score", 0.0))
    end_threshold = float(result.get("thresholds", {}).get("end_score", 0.0))
    router_info = result.get("router", {})
    router_low = router_info.get("prob_low")
    router_high = router_info.get("prob_high")
    router_unc = router_info.get("uncertainty_threshold")

    max_vlm_text = "n/a" if max_vlm is None else f"{max_vlm:.4f}"
    print(
        "[worker_event_payload] scores "
        f"clips={len(clip_scores)} "
        f"vlm_selected={selected_vlm} "
        f"max_fighting_prob={max_fast:.4f} "
        f"max_final_score={max_final:.4f} "
        f"max_vlm_score={max_vlm_text} "
        f"start_thr={start_threshold:.2f} "
        f"end_thr={end_threshold:.2f} "
        f"router=({router_low},{router_high},unc={router_unc}) "
        f"vlm_clip_ids={vlm_clip_ids[:10]} "
        f"top_fast={top_fast} "
        f"event_confidences={event_confidences} "
        f"payload_confidences={payload_confidences}"
    )


# [수정] VLM 첫 호출 시 수분 블로킹 방지 → 워커 시작 시 사전 로드
def _warmup_vlm(config):
    vlm_cfg = config.get("vlm", {})
    if not vlm_cfg.get("enabled", False):
        return
    if str(vlm_cfg.get("provider", "mock")).lower() != "internvl":
        return
    print("[worker_event_payload] VLM 모델 사전 로드 중 (첫 실행 시 수분 소요)...")
    try:
        import numpy as np
        from models.vlm.infer import InternVLVLMProvider
        from models.vlm.prompts import build_binary_fight_prompt
        from models.vlm.parser import parse_vlm_response

        provider = InternVLVLMProvider(vlm_cfg)
        provider._load_runtime()
        print("[worker_event_payload] VLM 모델 로드 완료 — 동작 테스트 중...")

        # [수정] 워밍업 후 더미 프레임으로 추론 1회 실행해 정상 작동 확인
        dummy_frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * 6
        dummy_record = {"frames": dummy_frames, "fighting_prob": 0.5, "uncertainty": 0.3,
                        "motion_summary": {"mean_magnitude": 0.0, "active_ratio": 0.0}}
        dummy_prompt = build_binary_fight_prompt(dummy_record["motion_summary"], num_frames=6)
        raw = provider.invoke(dummy_record, dummy_prompt)
        parsed = parse_vlm_response(raw)
        print(f"[worker_event_payload] VLM 테스트 완료 → label={parsed.get('label')} confidence={parsed.get('confidence'):.2f}")
    except Exception as e:
        import traceback
        print(f"[worker_event_payload] VLM 사전 로드 실패 (추론 시 재시도): {e}")
        traceback.print_exc()


def run_stream(stream_url: str, camera_id: str, cctv_id: Optional[int], stream_id: Optional[str]):
    config = _load_runtime_config(cctv_id=cctv_id, stream_id=stream_id or camera_id)
    _warmup_vlm(config)
    sent_cache: Dict[Tuple[str, int], datetime] = {}

    while True:
        print(f"[worker_event_payload] connecting: {stream_url}")
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            print(f"[worker_event_payload] open failed, retry in {RECONNECT_SEC}s")
            time.sleep(RECONNECT_SEC)
            continue

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        stride_frames = _estimate_stride_frames(fps)
        max_buffer = _estimate_buffer_frames(fps)
        min_ready = _minimum_ready_frames(fps)
        print(
            f"[worker_event_payload] connected fps={fps:.2f} "
            f"buffer_frames={max_buffer} stride_frames={stride_frames}"
        )

        frame_buf = deque(maxlen=max_buffer)
        ts_buf = deque(maxlen=max_buffer)
        frame_count = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                print("[worker_event_payload] stream disconnected, reconnecting...")
                break
            # [수정] RTMP 오디오/비디오 패킷 혼재 시 None 프레임 유입 방지
            if frame is None:
                continue

            frame_buf.append(frame.copy())
            ts_buf.append(_now_utc())
            frame_count += 1

            if frame_count % stride_frames != 0:
                continue
            if len(frame_buf) < min_ready:
                continue

            snippet_frames = list(frame_buf)
            snippet_started_at = ts_buf[0]
            temp_video = _save_temp_clip(snippet_frames, fps)
            # [수정] DEBUG_SAVE_SNIPPET_DIR 설정 시 버퍼 스니펫을 지정 폴더에 복사 저장
            if DEBUG_SAVE_SNIPPET_DIR:
                import shutil
                Path(DEBUG_SAVE_SNIPPET_DIR).mkdir(parents=True, exist_ok=True)
                stamp = snippet_started_at.strftime("%Y%m%dT%H%M%S")
                shutil.copy(temp_video, str(Path(DEBUG_SAVE_SNIPPET_DIR) / f"snippet_{stamp}.mp4"))
            run_name = _make_run_name(camera_id)

            try:
                result = run_single_video_pipeline(
                    temp_video,
                    config,
                    run_name=run_name,
                    verbose=False,
                )
                result["thresholds"] = dict(config.get("thresholds", {}))
                result["router"] = dict(config.get("router", {}))
                _log_score_summary(result)
                payload = result["event_payload"]
                payload = _absolutize_events(
                    payload=payload,
                    clip_started_at=snippet_started_at,
                    output_dir=result["output_dir"],
                )
                payload = _filter_new_events(payload, sent_cache)

                if not payload.get("events"):
                    print("[worker_event_payload] no new events")
                    continue

                print(
                    f"[worker_event_payload] events={len(payload['events'])} "
                    f"run={run_name}"
                )
                response = _post_payload(payload)
                print(f"[worker_event_payload] backend response={response}")
            except Exception as exc:
                print(f"[worker_event_payload] inference/post error: {exc}")
            finally:
                try:
                    os.remove(temp_video)
                except OSError:
                    pass

        cap.release()
        time.sleep(RECONNECT_SEC)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--cctv-id", type=int, default=None)
    parser.add_argument("--stream-id", default=None)
    args = parser.parse_args()

    run_stream(
        stream_url=args.stream,
        camera_id=args.camera_id,
        cctv_id=args.cctv_id,
        stream_id=args.stream_id,
    )
