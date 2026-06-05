"""
기존 train.json의 label을 InternVL2-4B VLM 판정으로 교체.

기존 clip feature(.npy)는 재사용하고, anomaly clip만 프레임 디코딩 → VLM 추론.
normal clip은 VLM 없이 label=0 유지.

사용법:
  cd AI/keyframe
  python scripts/relabel_with_vlm.py \
      --input_json  /home/hyrn2/github/2026-1-CECD2-1-Deepgu-06/AI/ai_pipeline/outputs/training_data/train.json \
      --output_json outputs/training_data_vlm/train_vlm.json \
      --video_dir   /home/hyrn2/github/2026-1-CECD2-1-Deepgu-06/AI/ai_pipeline/videos \
      --archive_dir /home/hyrn2/github/archive_extracted/dataset \
      --model_path  /home/hyrn2/github/2026-1-CECD2-1-Deepgu-06/AI/models/internvl2_4b \
      --n_frames 6 \
      --device cuda:1

출력:
  train_vlm.json : label=vlm 판정 (anomaly clip) / 0 (normal clip)
                   features_path는 기존 그대로 유지
"""

import os
import sys
import json
import argparse
import datetime
import time
import random
import re

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..")))

from pipeline.sampler import KeyframeSampler


def _ts():
    return datetime.datetime.now().strftime("%H:%M:%S")


def _fmt_sec(s):
    s = int(s)
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ------------------------------------------------------------------
# 비디오 경로 인덱스 빌드
# ------------------------------------------------------------------

_VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mpeg", ".mpg", ".mov")


def build_video_index(*dirs):
    """dirs 아래 모든 비디오를 재귀 탐색해 stem→path 딕셔너리 반환."""
    index = {}
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for fname in files:
                if fname.lower().endswith(_VIDEO_EXTS):
                    stem = os.path.splitext(fname)[0]
                    # 같은 stem이 여러 경로에 있으면 첫 번째 우선
                    if stem not in index:
                        index[stem] = os.path.join(root, fname)
    return index


# ------------------------------------------------------------------
# 클립 프레임 디코딩
# ------------------------------------------------------------------

def decode_clip_frames(video_path, start_frame, clip_len):
    """video_path에서 start_frame ~ start_frame+clip_len-1 프레임 디코딩.

    반환: List[np.ndarray(BGR)] (길이 < clip_len이면 끝까지만)
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(clip_len):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


# ------------------------------------------------------------------
# clip_id 파싱
# ------------------------------------------------------------------

_CLIP_RE = re.compile(r"^(.+)_clip(\d+)$")


def parse_clip_id(clip_id):
    """'Fighting041_x264_clip0007' → ('Fighting041_x264', 7)"""
    m = _CLIP_RE.match(clip_id)
    if not m:
        raise ValueError(f"clip_id 파싱 실패: {clip_id!r}")
    return m.group(1), int(m.group(2))


# ------------------------------------------------------------------
# VLM 응답 파싱
# ------------------------------------------------------------------

def parse_vlm_label(response) -> int:
    """VLM predict() 결과(dict 또는 str)에서 label(1/0) 추출."""
    if isinstance(response, dict):
        label_str = str(response.get("label", "")).lower()
    else:
        label_str = str(response).lower()

    if any(w in label_str for w in ["anomaly", "abnormal", "fight", "violence", "assault", "abuse"]):
        return 1
    return 0


# ------------------------------------------------------------------
# 저장 유틸
# ------------------------------------------------------------------

def save_json(records, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def load_done_ids(path):
    if not os.path.isfile(path):
        return set(), []
    with open(path) as f:
        records = json.load(f)
    return {r["clip_id"] for r in records}, records


# ------------------------------------------------------------------
# 메인
# ------------------------------------------------------------------

def main(args):
    random.seed(42)
    np.random.seed(42)

    print(f"[{_ts()}] 비디오 인덱스 빌드 중...", flush=True)
    video_index = build_video_index(args.video_dir, args.archive_dir)
    print(f"[{_ts()}] 비디오 {len(video_index)}개 인덱싱 완료", flush=True)

    print(f"[{_ts()}] VLM 로드: {args.model_path}", flush=True)
    from models.vlm.inference import load_vlm
    vlm = load_vlm(args.model_path, device=args.device, max_new_tokens=128)

    sampler = KeyframeSampler(n_frames=args.n_frames)

    print(f"[{_ts()}] 입력 JSON 로드: {args.input_json}", flush=True)
    with open(args.input_json) as f:
        all_records = json.load(f)

    # features_path가 상대경로면 input_json의 부모 디렉토리를 올라가며 resolve
    json_base = os.path.dirname(os.path.abspath(args.input_json))
    def _resolve_feat_path(fp, base):
        if os.path.isabs(fp):
            return fp
        b = base
        for _ in range(6):
            candidate = os.path.normpath(os.path.join(b, fp))
            if os.path.isfile(candidate):
                return candidate
            b = os.path.dirname(b)
        return os.path.normpath(os.path.join(base, fp))  # fallback

    # 첫 레코드로 base 결정 후 고정 (모두 같은 base 사용)
    _sample_fp = next(r["features_path"] for r in all_records if not os.path.isabs(r["features_path"]))
    _resolved_base = json_base
    _b = json_base
    for _ in range(6):
        if os.path.isfile(os.path.normpath(os.path.join(_b, _sample_fp))):
            _resolved_base = _b
            break
        _b = os.path.dirname(_b)

    for r in all_records:
        fp = r["features_path"]
        if not os.path.isabs(fp):
            r["features_path"] = os.path.normpath(os.path.join(_resolved_base, fp))

    print(f"[{_ts()}] 전체 clip: {len(all_records)} "
          f"(anomaly={sum(r['gt_label'] for r in all_records)}, "
          f"normal={sum(1-r['gt_label'] for r in all_records)})", flush=True)

    done_ids, output_records = load_done_ids(args.output_json)
    print(f"[{_ts()}] 기완료 clip: {len(done_ids)}개", flush=True)

    # 미처리 clip 분리
    to_process = [r for r in all_records if r["clip_id"] not in done_ids]
    anom_to_process = [r for r in to_process if r["gt_label"] == 1]
    norm_to_process = [r for r in to_process if r["gt_label"] == 0]

    print(f"[{_ts()}] 처리 예정 (서브샘플 전): anomaly={len(anom_to_process)}, normal={len(norm_to_process)}", flush=True)

    # 1) anomaly 먼저 서브샘플 → 실제 목표 anomaly 수 결정
    if args.max_clips_per_video and args.max_clips_per_video > 0:
        from collections import defaultdict
        per_video = defaultdict(list)
        for r in anom_to_process:
            stem, _ = parse_clip_id(r["clip_id"])
            per_video[stem].append(r)
        sampled = []
        for stem, clips in per_video.items():
            if len(clips) <= args.max_clips_per_video:
                sampled.extend(clips)
            else:
                indices = np.linspace(0, len(clips) - 1, args.max_clips_per_video).astype(int)
                sampled.extend(clips[i] for i in indices)
        print(f"[{_ts()}] anomaly clip 서브샘플: {len(anom_to_process)} → {len(sampled)} "
              f"(max {args.max_clips_per_video}/video)", flush=True)
        anom_to_process = sampled

    # 2) normal 서브샘플: ratio * anomaly 목표 수 기준, 이미 저장된 normal 수 차감
    target_anom_count = len(anom_to_process)
    max_normal_total = int(target_anom_count * args.normal_ratio) if args.normal_ratio > 0 else None
    already_done_normal = sum(1 for r in output_records if r.get("gt_label", 0) == 0)
    need_more_normal = (max_normal_total - already_done_normal) if max_normal_total is not None else len(norm_to_process)
    need_more_normal = max(0, need_more_normal)

    if len(norm_to_process) > need_more_normal:
        random.shuffle(norm_to_process)
        norm_to_process = norm_to_process[:need_more_normal]
        print(f"[{_ts()}] normal clip 서브샘플: → {need_more_normal}개 추가 "
              f"(already={already_done_normal}, target_total={max_normal_total})", flush=True)

    for r in norm_to_process:
        output_records.append({
            "clip_id"       : r["clip_id"],
            "label"         : 0,
            "gt_label"      : r["gt_label"],
            "vlm_label"     : 0,
            "features_path" : r["features_path"],
            "selected_indices": r.get("selected_indices", []),
        })
    done_ids.update(r["clip_id"] for r in norm_to_process)

    save_json(output_records, args.output_json)
    print(f"[{_ts()}] normal {len(norm_to_process)}개 추가 (총 {already_done_normal + len(norm_to_process)}개)", flush=True)

    # anomaly clip 처리
    miss_video, vlm_fail = 0, 0
    t_start = time.time()
    times = []

    for i, r in enumerate(anom_to_process):
        clip_id = r["clip_id"]

        try:
            stem, clip_idx = parse_clip_id(clip_id)
        except ValueError as e:
            print(f"[{_ts()}] [skip] {e}", flush=True)
            continue

        video_path = video_index.get(stem)
        if video_path is None:
            if miss_video < 5:
                print(f"[{_ts()}] [miss] 비디오 없음: {stem}", flush=True)
            miss_video += 1
            continue

        t_clip = time.time()
        start_frame = clip_idx * args.clip_stride
        frames = decode_clip_frames(video_path, start_frame, args.clip_len)

        if len(frames) < 4:
            continue

        # Phase 1 sampler로 대표 프레임 선택 (기존 features 재사용)
        features = np.load(r["features_path"]).astype(np.float32)
        candidate = {"clip": frames, "clip_id": clip_id, "features": features}
        sampler.sample(candidate)
        selected_frames = candidate["sampled_frames"]
        selected_indices = candidate["selected_indices"]

        # VLM 판정
        try:
            response = vlm.predict(selected_frames)
            vlm_label = parse_vlm_label(response)
        except Exception as e:
            print(f"[{_ts()}] [vlm_fail] {clip_id}: {e}", flush=True)
            vlm_fail += 1
            vlm_label = r["gt_label"]  # fallback: gt_label

        output_records.append({
            "clip_id"         : clip_id,
            "label"           : vlm_label,
            "gt_label"        : r["gt_label"],
            "vlm_label"       : vlm_label,
            "features_path"   : r["features_path"],
            "selected_indices": selected_indices,
        })
        done_ids.add(clip_id)

        elapsed = time.time() - t_clip
        times.append(elapsed)

        if (i + 1) % 100 == 0:
            avg = sum(times) / len(times)
            remaining = len(anom_to_process) - (i + 1)
            eta = _fmt_sec(avg * remaining)
            n_anom = sum(1 for rec in output_records if rec["vlm_label"] == 1)
            n_done = len([rec for rec in output_records if rec["gt_label"] == 1])
            print(f"[{_ts()}] [{i+1}/{len(anom_to_process)}] "
                  f"vlm_anom={n_anom}/{n_done} miss={miss_video} fail={vlm_fail} "
                  f"{avg:.1f}s/clip ETA {eta}", flush=True)
            save_json(output_records, args.output_json)

    save_json(output_records, args.output_json)

    n_total = len(output_records)
    n_vlm_anom = sum(1 for r in output_records if r["vlm_label"] == 1)
    total_t = _fmt_sec(time.time() - t_start)

    print(f"\n[{_ts()}] 완료: {n_total}개 clip", flush=True)
    print(f"  vlm_label=1 (anomaly): {n_vlm_anom}", flush=True)
    print(f"  vlm_label=0 (normal) : {n_total - n_vlm_anom}", flush=True)
    print(f"  비디오 못 찾음: {miss_video}개", flush=True)
    print(f"  VLM 실패 (gt fallback): {vlm_fail}개", flush=True)
    print(f"  소요: {total_t}", flush=True)
    print(f"  → {args.output_json}", flush=True)
    print(f"\n다음 단계:", flush=True)
    print(f"  python scripts/train_frame_selector.py \\", flush=True)
    print(f"      --label_path {args.output_json} \\", flush=True)
    print(f"      --save_path  outputs/frame_selector_vlm.pth \\", flush=True)
    print(f"      --input_dim  2048 \\", flush=True)
    print(f"      --n_frames   {args.n_frames}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train.json의 label을 VLM 판정으로 교체")
    parser.add_argument("--input_json",  required=True,
                        help="기존 train.json 경로 (features_path 포함)")
    parser.add_argument("--output_json", default="outputs/training_data_vlm/train_vlm.json")
    parser.add_argument("--video_dir",   required=True,
                        help="UCF-Crime 비디오 루트 (Abuse/, Assault/, Fighting/, Normal_*/ 포함)")
    parser.add_argument("--archive_dir", default=None,
                        help="archive fight 비디오 루트 (CCTV_DATA/, NON_CCTV_DATA/ 포함)")
    parser.add_argument("--model_path",  required=True,
                        help="InternVL2-4B 모델 경로")
    parser.add_argument("--n_frames",    type=int, default=6,
                        help="Phase 1 sampler로 고를 프레임 수 (VLM 입력)")
    parser.add_argument("--clip_len",    type=int, default=16,
                        help="clip 당 프레임 수 (prepare_data.py와 동일하게)")
    parser.add_argument("--clip_stride", type=int, default=8,
                        help="clip 슬라이딩 간격 (prepare_data.py와 동일하게)")
    parser.add_argument("--max_clips_per_video", type=int, default=20,
                        help="anomaly 비디오당 최대 clip 수. 0=무제한. 균등 샘플링.")
    parser.add_argument("--normal_ratio", type=float, default=1.0,
                        help="normal clip 수 = anomaly clip 수 × 이 값. 0=전부 사용.")
    parser.add_argument("--device",      default="cuda:1")
    args = parser.parse_args()
    main(args)
