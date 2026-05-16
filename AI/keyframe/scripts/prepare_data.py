"""
학습 데이터 준비 스크립트.

동작:
  1. 비디오를 clip으로 분할 (clip_generator 재사용)
  2. ResNet-50으로 clip별 frame features 추출
  3. Phase 1 sampler(Temporal Clustering + Motion)로 대표 frame 선택
  4. InternVL2로 선택된 frame을 보고 anomaly/normal 판별 → pseudo-label
     (--skip_vlm 시 gt_label 직접 사용)
  5. 비디오 단위로 train/test 분할 후 저장

입력 디렉토리 구조 (두 가지 모두 자동 지원):
  [형식 A] anomaly/normal 분리
    VIDEO_DIR/anomaly/{video}.mp4
    VIDEO_DIR/normal/{video}.mp4

  [형식 B] UCF-Crime 클래스별 (videos/{ClassName}/)
    VIDEO_DIR/Abuse/{video}.mp4
    VIDEO_DIR/Fighting/{video}.mp4
    VIDEO_DIR/Normal_Videos_event/{video}.mp4
    ...

출력:
  OUTPUT_DIR/
  ├── features/
  │   └── {clip_id}.npy          # ResNet-50 features (T, 2048)
  ├── train.json                  # 학습용 pseudo-labels
  └── test.json                   # 평가용 pseudo-labels

사용법:
  # 형식 B (UCF-Crime 클래스 구조), VLM 없이 gt_label 사용
  python scripts/prepare_data.py \
      --video_dir videos \
      --output_dir outputs/training_data \
      --skip_vlm \
      --device cuda

  # VLM pseudo-label 포함
  python scripts/prepare_data.py \
      --video_dir data/videos \
      --output_dir outputs/training_data \
      --device cuda
"""

import os
import sys
import json
import argparse
import random
import time
import datetime
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.clip_generator import generate_clips
from pipeline.sampler import KeyframeSampler
from models.feature_extractor import ResNet50Extractor


def _ts():
    return datetime.datetime.now().strftime("%H:%M:%S")


def _fmt_sec(s):
    s = int(s)
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

# PGL-SUM 학습에 사용한 3개 클래스만 anomaly로 사용
_TARGET_ANOMALY_CLASSES = {"Abuse", "Assault", "Fighting"}

# UCF-Crime normal 클래스 디렉토리 이름 패턴
_NORMAL_PREFIXES = (
    "Normal_Videos_event",
    "Training-Normal-Videos-Part-1",
    "Training-Normal-Videos-Part-2",
    "Testing_Normal_Videos_Anomaly",
)


def get_frame_count(video_path):
    """OpenCV로 frame 수만 빠르게 읽기 (디코딩 없음)."""
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


# ------------------------------------------------------------------
# VLM 응답 파싱
# ------------------------------------------------------------------

def parse_vlm_response(response: str) -> int:
    """
    InternVL2 JSON 응답 → label (1: anomaly, 0: normal)
    응답 형식: {"label": "anomaly" or "normal", "description": "..."}
    """
    try:
        start = response.find("{")
        end   = response.rfind("}") + 1
        if start != -1 and end > start:
            import json as _json
            data = _json.loads(response[start:end])
            label_str = data.get("label", "").lower()
            if "anomaly" in label_str:
                return 1
            if "normal" in label_str:
                return 0
    except Exception:
        pass

    # fallback: 키워드 기반
    r = response.lower()
    if any(w in r for w in ["anomaly", "abnormal", "fight", "violence", "assault", "abuse", "theft"]):
        return 1
    return 0


# ------------------------------------------------------------------
# 비디오 목록 수집
# ------------------------------------------------------------------

def collect_videos(video_dir, archive_dir=None, max_normal_frames=None):
    """
    anomaly  : UCF videos/{Abuse,Assault,Fighting}/ + archive CCTV/NON_CCTV (fight)
    normal   : UCF videos/Normal_*/ — max_normal_frames 초과 시 skip
    그 외 UCF 클래스(Robbery 등)는 제외.

    반환: [(video_path, gt_label), ...]  gt_label: 1=anomaly, 0=normal
    """
    _ext = (".mp4", ".avi", ".mkv", ".mpeg", ".mpg")
    anomaly, normal = [], []

    # UCF-Crime: Abuse/Assault/Fighting → anomaly, Normal_* → normal
    subdirs = [d for d in sorted(os.listdir(video_dir))
               if os.path.isdir(os.path.join(video_dir, d))]
    for cls_name in subdirs:
        cls_dir = os.path.join(video_dir, cls_name)
        if cls_name in _TARGET_ANOMALY_CLASSES:
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(_ext):
                    anomaly.append(os.path.join(cls_dir, fname))
        elif cls_name.startswith(_NORMAL_PREFIXES):
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(_ext):
                    normal.append(os.path.join(cls_dir, fname))
        # 나머지 UCF 클래스는 무시

    # Archive fight 영상 → anomaly
    if archive_dir and os.path.isdir(archive_dir):
        for root, _, files in os.walk(archive_dir):
            for fname in sorted(files):
                if fname.lower().endswith(_ext):
                    anomaly.append(os.path.join(root, fname))

    # normal 필터링: max_normal_frames 초과 skip
    if max_normal_frames:
        filtered, skipped = [], []
        print(f"[{_ts()}] normal 영상 frame 수 확인 중 ({len(normal)}개)...", flush=True)
        for p in normal:
            n = get_frame_count(p)
            if n <= max_normal_frames:
                filtered.append(p)
            else:
                skipped.append((os.path.basename(p), n))
        if skipped:
            print(f"[{_ts()}]   skip {len(skipped)}개 (>{max_normal_frames} frames):", flush=True)
            for name, n in skipped[:10]:
                print(f"[{_ts()}]     {name}: {n:,}f", flush=True)
            if len(skipped) > 10:
                print(f"[{_ts()}]     ... 외 {len(skipped)-10}개", flush=True)
        normal = filtered

    videos = [(p, 1) for p in anomaly] + [(p, 0) for p in normal]
    print(f"[{_ts()}] 비디오 수집: {len(videos)}개 "
          f"(anomaly={len(anomaly)}, normal={len(normal)})", flush=True)
    return videos


# ------------------------------------------------------------------
# 단일 비디오 처리
# ------------------------------------------------------------------

def process_video(video_path, gt_label, extractor, sampler, vlm,
                  clip_len, clip_stride, feat_dir, n_frames,
                  skip_vlm_for_normal):
    """
    비디오 하나를 clip으로 분할하고 pseudo-label을 생성합니다.
    반환: list of { clip_id, label, features_path, selected_indices }
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # clip 생성 (frame 리스트만, 저장 안 함)
    try:
        clips = generate_clips(video_path, output_dir="", clip_len=clip_len,
                               stride=clip_stride, save=False)
    except Exception as e:
        print(f"[{_ts()}]   [오류] clip 생성 실패 ({video_name}): {e}", flush=True)
        return []

    records = []

    for clip_idx, frames in enumerate(clips):
        clip_id = f"{video_name}_clip{clip_idx:04d}"
        feat_path = os.path.join(feat_dir, f"{clip_id}.npy")

        # 이미 처리된 clip은 건너뜀 (재시작 대응)
        if os.path.isfile(feat_path):
            # features만 있고 label이 없는 경우도 있으므로 features만 skip
            pass
        else:
            features = extractor.extract_from_frames(frames)  # (T, 2048)
            np.save(feat_path, features)

        features = np.load(feat_path)

        # Phase 1: frame 선택
        candidate = {"clip": frames, "features": features, "clip_id": clip_id}
        sampler.sample(candidate)
        selected_frames   = candidate["sampled_frames"]
        selected_indices  = candidate["selected_indices"]

        # VLM으로 pseudo-label 생성
        if vlm is None:
            label = gt_label   # --skip_vlm: gt_label 직접 사용
        elif skip_vlm_for_normal and gt_label == 0:
            label = 0          # normal 비디오는 VLM 없이 0 부여
        else:
            try:
                response = vlm.predict(selected_frames)
                label    = parse_vlm_response(response)
            except Exception as e:
                print(f"  [경고] VLM 실패 ({clip_id}): {e}, gt_label={gt_label} 사용")
                label = gt_label

        records.append({
            "clip_id"         : clip_id,
            "label"           : label,
            "gt_label"        : gt_label,       # 비디오 카테고리 기반 실제 레이블
            "features_path"   : feat_path,
            "selected_indices": selected_indices,
        })

    return records


# ------------------------------------------------------------------
# 메인
# ------------------------------------------------------------------

def _save_json(records, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _load_done_clips(json_path):
    """이미 저장된 JSON에서 완료된 clip_id 집합 반환."""
    if not os.path.isfile(json_path):
        return set(), []
    with open(json_path) as f:
        records = json.load(f)
    return {r["clip_id"] for r in records}, records


def main(args):
    random.seed(42)
    np.random.seed(42)

    os.makedirs(args.output_dir, exist_ok=True)
    feat_dir = os.path.join(args.output_dir, "features")
    os.makedirs(feat_dir, exist_ok=True)

    print(f"[{_ts()}] 모델 초기화 중...")
    extractor = ResNet50Extractor(device=args.device)
    sampler   = KeyframeSampler(n_frames=args.n_frames)

    if args.skip_vlm:
        print(f"[{_ts()}] --skip_vlm: VLM 없이 gt_label을 pseudo-label로 사용합니다.")
        vlm = None
    else:
        from models.vlm.inference import InternVL
        vlm = InternVL()

    videos = collect_videos(args.video_dir, args.archive_dir, args.max_normal_frames)
    if not videos:
        print("비디오가 없습니다.")
        return

    random.shuffle(videos)
    n_test  = max(1, int(len(videos) * args.test_split))
    test_videos  = videos[:n_test]
    train_videos = videos[n_test:]

    # anomaly 먼저 처리 (크래시 대비)
    train_videos.sort(key=lambda x: x[1], reverse=True)
    test_videos.sort(key=lambda x: x[1], reverse=True)

    print(f"[{_ts()}] train={len(train_videos)}, test={len(test_videos)} 비디오 "
          f"(anomaly 먼저 처리)")

    for split_name, split_videos in [("train", train_videos), ("test", test_videos)]:
        out_path = os.path.join(args.output_dir, f"{split_name}.json")
        done_clips, all_records = _load_done_clips(out_path)

        n_skip = len({r["clip_id"].rsplit("_clip", 1)[0] for r in all_records})
        print(f"\n[{_ts()}] --- {split_name} 처리 중 ({len(split_videos)}개 비디오, "
              f"기완료 비디오 ~{n_skip}개 / clip {len(done_clips)}개) ---", flush=True)

        split_start = time.time()
        video_times = []

        for v_idx, (vpath, gt_label) in enumerate(split_videos):
            vname = os.path.basename(vpath)
            v_start = time.time()

            # 이 비디오의 첫 clip이 이미 있으면 skip
            stem = os.path.splitext(vname)[0]
            first_clip_id = f"{stem}_clip0000"
            if first_clip_id in done_clips:
                print(f"[{_ts()}] [{v_idx+1}/{len(split_videos)}] {vname} — SKIP (already done)",
                      flush=True)
                continue

            label_str = "anomaly" if gt_label == 1 else "normal"
            print(f"[{_ts()}] [{v_idx+1}/{len(split_videos)}] {vname} (gt={gt_label}/{label_str})",
                  flush=True)

            records = process_video(
                vpath, gt_label, extractor, sampler, vlm,
                args.clip_len, args.clip_stride, feat_dir,
                args.n_frames, args.skip_vlm_for_normal,
            )
            all_records.extend(records)
            done_clips.update(r["clip_id"] for r in records)

            elapsed_v = time.time() - v_start
            video_times.append(elapsed_v)

            n_anom  = sum(r["label"] for r in all_records)
            n_total = len(all_records)
            avg_t   = sum(video_times) / len(video_times)
            remaining = len(split_videos) - (v_idx + 1)
            eta = _fmt_sec(avg_t * remaining)

            print(f"[{_ts()}]   clips={len(records)}  누적 clip={n_total} "
                  f"(anomaly={n_anom})  {elapsed_v:.1f}s/video  ETA {eta}",
                  flush=True)

            # 50개 비디오마다 중간 저장
            if (v_idx + 1) % 50 == 0:
                _save_json(all_records, out_path)
                print(f"[{_ts()}]   [중간저장] {out_path}  ({len(all_records)}개 clip)",
                      flush=True)

        _save_json(all_records, out_path)
        n_anom = sum(r["label"] for r in all_records)
        total_t = _fmt_sec(time.time() - split_start)
        print(f"\n[{_ts()}] {split_name} 완료: {len(all_records)}개 clip "
              f"(anomaly={n_anom}, normal={len(all_records)-n_anom})  소요={total_t}")
        print(f"[{_ts()}] → {out_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="학습 데이터 준비")
    parser.add_argument("--video_dir",  required=True,
                        help="anomaly/, normal/ 하위 디렉토리를 포함한 비디오 루트")
    parser.add_argument("--output_dir", default="outputs/training_data")
    parser.add_argument("--n_frames",   type=int,   default=8,
                        help="Phase 1에서 선택할 frame 수")
    parser.add_argument("--clip_len",   type=int,   default=16,
                        help="clip 당 frame 수")
    parser.add_argument("--clip_stride",type=int,   default=8,
                        help="clip 슬라이딩 간격")
    parser.add_argument("--test_split", type=float, default=0.1,
                        help="test 비율 (비디오 단위 분할)")
    parser.add_argument("--skip_vlm",    action="store_true",
                        help="VLM 없이 gt_label을 pseudo-label로 직접 사용")
    parser.add_argument("--skip_vlm_for_normal", action="store_true",
                        help="normal 비디오는 VLM 없이 label=0 부여 (속도 향상)")
    parser.add_argument("--archive_dir", default=None,
                        help="archive fight 영상 루트 (CCTV_DATA/NON_CCTV_DATA 포함)")
    parser.add_argument("--max_normal_frames", type=int, default=None,
                        help="normal 영상 최대 frame 수. 초과 시 skip")
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()
    main(args)
