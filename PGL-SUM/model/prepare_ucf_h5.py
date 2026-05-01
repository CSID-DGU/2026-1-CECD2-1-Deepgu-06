"""
UCF-Crime 영상(mp4)에서 ResNet-50 feature를 추출하고 PGL-SUM 학습용 h5 파일을 생성합니다.

데이터 경로: AI/ai_pipeline/videos/{클래스명}/{비디오}.mp4
  - 이상행동: Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting,
              RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism
  - 정상    : Training-Normal-Videos-Part-1, Training-Normal-Videos-Part-2,
              Normal_Videos_event, Testing_Normal_Videos_Anomaly

출력:
  ../data/UCF/ucf_crime_resnet50.h5   : 비디오별 features / gtscore 등
  ../data/splits/ucf_splits.json      : 5-split train/test 비디오 목록

h5 구조 (SumMe/TVSum 포맷 동일):
  /video_name
    /features         (T, 2048)   ResNet-50 avgpool features
    /gtscore          (T,)        motion proxy score [0, 1]
    /user_summary     (1, T)      상위 motion frame 기반 binary summary
    /change_points    (n_seg, 2)  균등 분할 segment 경계
    /n_frame_per_seg  (n_seg,)    segment별 frame 수
    /n_frames         scalar      원본 총 frame 수
    /picks            (T,)        원본 영상에서 subsample한 frame index
    /n_steps          scalar      subsample된 frame 수 (= T)

실행:
  cd PGL-SUM
  python model/prepare_ucf_h5.py [--max_frames 300] [--n_per_anomaly 10]
"""

import os
import sys
import argparse
import random
import json
from pathlib import Path

import cv2
import numpy as np
import h5py
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm

# ── 경로 설정 ──────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent          # PGL-SUM/model/
PGLSUM_DIR  = SCRIPT_DIR.parent             # PGL-SUM/
VIDEOS_DIR  = PGLSUM_DIR.parent / "AI" / "ai_pipeline" / "videos"
OUTPUT_H5   = PGLSUM_DIR / "data" / "UCF" / "ucf_crime_resnet50.h5"
OUTPUT_SPLITS = PGLSUM_DIR / "data" / "splits" / "ucf_splits.json"

# 사용할 이상행동 클래스 (폭행 관련 3개)
ANOMALY_DIRS = [
    "Abuse",
    "Assault",
    "Fighting",
]

# 정상 영상 디렉토리
NORMAL_DIRS = [
    "Training-Normal-Videos-Part-1",
    "Training-Normal-Videos-Part-2",
    "Normal_Videos_event",
    "Testing_Normal_Videos_Anomaly",
]


# -----------------------------------------------------------------------
# ResNet-50 Feature Extractor (AI pipeline 과 동일 전처리)
# -----------------------------------------------------------------------

class ResNet50Extractor:
    def __init__(self, device="cuda", batch_size=32):
        self.device = device
        self.batch_size = batch_size

        model = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(model.children())[:-1])  # fc 제거
        self.backbone.eval().to(device)

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract_from_bgr_frames(self, bgr_frames):
        """
        Args:
            bgr_frames: List[np.ndarray]  BGR, HWC, uint8
        Returns:
            np.ndarray (T, 2048)
        """
        if not bgr_frames:
            return np.zeros((0, 2048), dtype=np.float32)

        all_features = []
        for i in range(0, len(bgr_frames), self.batch_size):
            batch = bgr_frames[i:i + self.batch_size]
            tensors = []
            for bgr in batch:
                rgb = bgr[..., ::-1].copy()        # BGR → RGB
                img = Image.fromarray(rgb.astype(np.uint8))
                tensors.append(self.transform(img))
            x = torch.stack(tensors).to(self.device)
            feat = self.backbone(x).squeeze(-1).squeeze(-1)  # (B, 2048)
            all_features.append(feat.cpu().numpy())
        return np.concatenate(all_features, axis=0).astype(np.float32)


# -----------------------------------------------------------------------
# MP4에서 균등 subsample 프레임 추출
# -----------------------------------------------------------------------

def load_frames_from_video(video_path, max_frames):
    """
    OpenCV로 mp4를 읽고 max_frames개로 균등 subsample한 BGR 프레임을 반환.

    Returns:
        bgr_frames: List[np.ndarray]
        picks     : np.ndarray  (T,)  원본 프레임 인덱스
        n_orig    : int               원본 총 프레임 수
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], np.array([]), 0

    n_orig = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_orig <= 0:
        cap.release()
        return [], np.array([]), 0

    T = min(max_frames, n_orig)
    picks = np.linspace(0, n_orig - 1, T, dtype=int)

    bgr_frames = []
    prev_idx = -1
    for idx in picks:
        if idx != prev_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            prev_idx = idx
        ret, frame = cap.read()
        if not ret or frame is None:
            # 읽기 실패 → 검은 프레임으로 대체
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 240
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 320
            frame = np.zeros((h, w, 3), dtype=np.uint8)
        bgr_frames.append(frame)

    cap.release()
    return bgr_frames, picks, n_orig


# -----------------------------------------------------------------------
# Motion Proxy Score (Phase 1과 동일 수식)
# -----------------------------------------------------------------------

def motion_proxy(features, alpha=0.5):
    """velocity + acceleration → [0,1] 정규화된 gtscore."""
    vel = np.linalg.norm(features[1:] - features[:-1], axis=1)
    vel = np.concatenate([[0.0], vel])

    accel = np.abs(vel[1:] - vel[:-1])
    accel = np.concatenate([[0.0], accel])

    proxy = alpha * vel + (1 - alpha) * accel
    lo, hi = proxy.min(), proxy.max()
    if hi - lo < 1e-8:
        return np.zeros_like(proxy, dtype=np.float32)
    return ((proxy - lo) / (hi - lo)).astype(np.float32)


# -----------------------------------------------------------------------
# Change Points (균등 분할)
# -----------------------------------------------------------------------

def make_change_points(T, n_segments=4):
    seg_len   = T // n_segments
    remainder = T % n_segments
    cp, nfps  = [], []
    start = 0
    for s in range(n_segments):
        extra = 1 if s < remainder else 0
        end   = start + seg_len + extra
        cp.append([start, end - 1])
        nfps.append(end - start)
        start = end
    return np.array(cp, dtype=np.int32), np.array(nfps, dtype=np.int32)


# -----------------------------------------------------------------------
# 비디오 목록 수집
# -----------------------------------------------------------------------

def collect_videos(videos_dir, n_per_anomaly, n_normal, seed):
    """
    Returns:
        List[(video_path, unique_key, label)]
        unique_key : "ClassName__videoname"  (h5 key, / 없음)
        label      : "anomaly" | "normal"
    """
    rng = random.Random(seed)
    selected = []

    # 이상행동
    for cls in ANOMALY_DIRS:
        cls_dir = videos_dir / cls
        if not cls_dir.exists():
            continue
        mp4s = sorted(cls_dir.glob("*.mp4"))
        chosen = rng.sample(mp4s, min(n_per_anomaly, len(mp4s)))
        for p in chosen:
            key = f"{cls}__{p.stem}"   # e.g. "Abuse__Abuse001_x264"
            selected.append((p, key, "anomaly"))

    # 정상: 여러 디렉토리에서 모아서 n_normal개 샘플링
    all_normal = []
    for d in NORMAL_DIRS:
        nd = videos_dir / d
        if nd.exists():
            all_normal.extend(nd.glob("*.mp4"))
    all_normal = sorted(all_normal)
    chosen_normal = rng.sample(all_normal, min(n_normal, len(all_normal)))
    for p in chosen_normal:
        # 부모 디렉토리명도 키에 포함 (중복 방지)
        key = f"Normal__{p.parent.name}__{p.stem}"
        selected.append((p, key, "normal"))

    return selected


# -----------------------------------------------------------------------
# 단일 비디오 처리
# -----------------------------------------------------------------------

def process_video(video_path, extractor, max_frames, n_segments=4):
    bgr_frames, picks, n_orig = load_frames_from_video(video_path, max_frames)
    if not bgr_frames:
        return None

    features = extractor.extract_from_bgr_frames(bgr_frames)  # (T, 2048)
    T = len(features)

    gtscore  = motion_proxy(features)                          # (T,)

    k = max(1, int(0.15 * T))
    top_idx  = np.argsort(gtscore)[::-1][:k]
    user_sum = np.zeros((1, T), dtype=np.float32)
    user_sum[0, top_idx] = 1.0

    cp, nfps = make_change_points(T, n_segments)

    return {
        "features":        features,
        "gtscore":         gtscore,
        "user_summary":    user_sum,
        "change_points":   cp,
        "n_frame_per_seg": nfps,
        "n_frames":        n_orig,
        "picks":           picks.astype(np.int32),
        "n_steps":         T,
    }


# -----------------------------------------------------------------------
# Train/Test Splits
# -----------------------------------------------------------------------

def make_splits(keys, n_splits=5, train_ratio=0.8, seed=42):
    rng = random.Random(seed)
    splits = []
    for _ in range(n_splits):
        shuffled = keys[:]
        rng.shuffle(shuffled)
        n_train = int(len(shuffled) * train_ratio)
        splits.append({
            "train_keys": shuffled[:n_train],
            "test_keys":  shuffled[n_train:],
        })
    return splits


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device     : {device}")
    print(f"Videos dir : {VIDEOS_DIR}")
    print(f"Output h5  : {OUTPUT_H5}")

    if not VIDEOS_DIR.exists():
        print(f"[ERROR] 비디오 디렉토리가 없습니다: {VIDEOS_DIR}")
        sys.exit(1)

    OUTPUT_H5.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_SPLITS.parent.mkdir(parents=True, exist_ok=True)

    # 비디오 목록
    print(f"\n비디오 수집 (클래스당 {args.n_per_anomaly}개, 정상 {args.n_normal}개)...")
    video_list = collect_videos(VIDEOS_DIR, args.n_per_anomaly, args.n_normal, args.seed)
    n_anom   = sum(1 for _, _, l in video_list if l == "anomaly")
    n_normal = sum(1 for _, _, l in video_list if l == "normal")
    print(f"  이상행동: {n_anom}개  |  정상: {n_normal}개  |  합계: {len(video_list)}개")

    # Feature extractor
    print("\nResNet-50 로드 중...")
    extractor = ResNet50Extractor(device=device, batch_size=args.batch_size)

    # H5 생성
    print(f"\nFeature 추출 및 H5 작성 중...")
    valid_keys = []
    failed     = []

    with h5py.File(OUTPUT_H5, "w") as hf:
        for video_path, key, label in tqdm(video_list, desc="처리 중"):
            result = process_video(video_path, extractor,
                                   max_frames=args.max_frames)
            if result is None:
                failed.append(str(video_path))
                continue

            grp = hf.create_group(key)
            grp.create_dataset("features",        data=result["features"])
            grp.create_dataset("gtscore",         data=result["gtscore"])
            grp.create_dataset("user_summary",    data=result["user_summary"])
            grp.create_dataset("change_points",   data=result["change_points"])
            grp.create_dataset("n_frame_per_seg", data=result["n_frame_per_seg"])
            grp.create_dataset("n_frames",        data=np.int32(result["n_frames"]))
            grp.create_dataset("picks",           data=result["picks"])
            grp.create_dataset("n_steps",         data=np.int32(result["n_steps"]))
            grp.attrs["label"] = label
            valid_keys.append(key)

    print(f"\nH5 저장 완료: {len(valid_keys)}개 / {len(video_list)}개")
    if failed:
        print(f"  실패: {len(failed)}개 — {failed[:3]} ...")
    print(f"H5 크기: {OUTPUT_H5.stat().st_size / 1e6:.1f} MB")

    # Splits
    print(f"\nSplit 생성 ({args.n_splits}개, train {int(args.train_ratio*100)}%)...")
    splits = make_splits(valid_keys, args.n_splits, args.train_ratio, args.seed)
    with open(OUTPUT_SPLITS, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Splits 저장: {OUTPUT_SPLITS}")
    print(f"  split 0: train={len(splits[0]['train_keys'])}개, "
          f"test={len(splits[0]['test_keys'])}개")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UCF-Crime h5 데이터셋 준비 (mp4 → ResNet-50 feature)")
    parser.add_argument("--max_frames",    type=int,   default=300,
                        help="비디오당 subsample frame 수 (default: 300)")
    parser.add_argument("--n_per_anomaly", type=int,   default=10,
                        help="이상행동 클래스당 비디오 수 (default: 10)")
    parser.add_argument("--n_normal",      type=int,   default=50,
                        help="정상 비디오 수 (default: 50)")
    parser.add_argument("--n_splits",      type=int,   default=5,
                        help="split 수 (default: 5)")
    parser.add_argument("--train_ratio",   type=float, default=0.8,
                        help="train 비율 (default: 0.8)")
    parser.add_argument("--batch_size",    type=int,   default=32,
                        help="ResNet-50 batch size (default: 32)")
    parser.add_argument("--seed",          type=int,   default=42,
                        help="random seed (default: 42)")
    args = parser.parse_args()
    main(args)
