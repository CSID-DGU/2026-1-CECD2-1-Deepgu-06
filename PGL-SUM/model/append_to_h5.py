"""
기존 h5 파일에 추가 비디오(archive + UCF normal 전체)를 append합니다.

배경:
  Step 1에서 생성한 h5에는 UCF-Crime 3클래스(150개) + 정상(50개) = 200개만 들어있음.
  여기에 아래 데이터를 추가합니다:
    - archive_extracted/dataset/{CCTV_DATA,NON_CCTV_DATA}/.../*.mpeg  (fight 1000개)
    - AI/ai_pipeline/videos/Training-Normal-Videos-Part-{1,2}/        (정상 800개)
    - AI/ai_pipeline/videos/Normal_Videos_event/                       (정상 50개)
    - AI/ai_pipeline/videos/Testing_Normal_Videos_Anomaly/             (정상 150개)

이미 h5에 있는 key는 skip합니다 (재시작 안전).

실행:
  python PGL-SUM/model/append_to_h5.py
"""

import sys
import json
import random
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

# ── 경로 ──────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
PGLSUM_DIR   = SCRIPT_DIR.parent
REPO_DIR     = PGLSUM_DIR.parent

ARCHIVE_DIR  = Path("/home/hyrn2/github/archive_extracted/dataset")
VIDEOS_DIR   = REPO_DIR / "AI" / "ai_pipeline" / "videos"
H5_PATH      = PGLSUM_DIR / "data" / "UCF" / "ucf_crime_resnet50.h5"
SPLITS_PATH  = PGLSUM_DIR / "data" / "splits" / "ucf_splits.json"

# archive 내 fight 영상 디렉토리
ARCHIVE_SUBDIRS = [
    "CCTV_DATA/training",
    "CCTV_DATA/validation",
    "CCTV_DATA/testing",
    "NON_CCTV_DATA/training",
    "NON_CCTV_DATA/validation",
    "NON_CCTV_DATA/testing",
]

# 추가할 UCF 정상 영상 디렉토리 (기존 50개 외 나머지 전부)
EXTRA_NORMAL_DIRS = [
    "Training-Normal-Videos-Part-1",
    "Training-Normal-Videos-Part-2",
    "Normal_Videos_event",
    "Testing_Normal_Videos_Anomaly",
]

MAX_FRAMES   = 300
BATCH_SIZE   = 32
N_SPLITS     = 5
TRAIN_RATIO  = 0.8
SEED         = 42


# ── Feature Extractor (prepare_ucf_h5.py 와 동일) ─────────────────────

class ResNet50Extractor:
    def __init__(self, device="cuda", batch_size=32):
        self.device = device
        self.batch_size = batch_size
        model = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(model.children())[:-1])
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
        if not bgr_frames:
            return np.zeros((0, 2048), dtype=np.float32)
        all_features = []
        for i in range(0, len(bgr_frames), self.batch_size):
            batch = bgr_frames[i:i + self.batch_size]
            tensors = [self.transform(Image.fromarray(f[..., ::-1].copy().astype(np.uint8)))
                       for f in batch]
            x = torch.stack(tensors).to(self.device)
            feat = self.backbone(x).squeeze(-1).squeeze(-1)
            all_features.append(feat.cpu().numpy())
        return np.concatenate(all_features, axis=0).astype(np.float32)


# ── 공통 유틸 ─────────────────────────────────────────────────────────

def load_frames(video_path, max_frames):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], np.array([]), 0
    n_orig = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_orig <= 0:
        cap.release()
        return [], np.array([]), 0
    T = min(max_frames, n_orig)
    picks = np.linspace(0, n_orig - 1, T, dtype=int)
    frames = []
    for idx in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 240
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 320
            frame = np.zeros((h, w, 3), dtype=np.uint8)
        frames.append(frame)
    cap.release()
    return frames, picks, n_orig


def motion_proxy(features, alpha=0.5):
    vel   = np.linalg.norm(features[1:] - features[:-1], axis=1)
    vel   = np.concatenate([[0.0], vel])
    accel = np.abs(vel[1:] - vel[:-1])
    accel = np.concatenate([[0.0], accel])
    proxy = alpha * vel + (1 - alpha) * accel
    lo, hi = proxy.min(), proxy.max()
    if hi - lo < 1e-8:
        return np.zeros_like(proxy, dtype=np.float32)
    return ((proxy - lo) / (hi - lo)).astype(np.float32)


def make_change_points(T, n_segments=4):
    seg_len = T // n_segments
    rem = T % n_segments
    cp, nfps = [], []
    start = 0
    for s in range(n_segments):
        end = start + seg_len + (1 if s < rem else 0)
        cp.append([start, end - 1])
        nfps.append(end - start)
        start = end
    return np.array(cp, dtype=np.int32), np.array(nfps, dtype=np.int32)


def process_and_write(hf, key, video_path, label, extractor):
    """비디오 하나를 처리해서 h5에 씁니다. 이미 있는 key는 skip."""
    if key in hf:
        return False  # already exists

    frames, picks, n_orig = load_frames(video_path, MAX_FRAMES)
    if not frames:
        return False

    features = extractor.extract_from_bgr_frames(frames)
    T = len(features)
    gtscore  = motion_proxy(features)
    k        = max(1, int(0.15 * T))
    top_idx  = np.argsort(gtscore)[::-1][:k]
    user_sum = np.zeros((1, T), dtype=np.float32)
    user_sum[0, top_idx] = 1.0
    cp, nfps = make_change_points(T)

    grp = hf.create_group(key)
    grp.create_dataset("features",        data=features)
    grp.create_dataset("gtscore",         data=gtscore)
    grp.create_dataset("user_summary",    data=user_sum)
    grp.create_dataset("change_points",   data=cp)
    grp.create_dataset("n_frame_per_seg", data=nfps)
    grp.create_dataset("n_frames",        data=np.int32(n_orig))
    grp.create_dataset("picks",           data=picks.astype(np.int32))
    grp.create_dataset("n_steps",         data=np.int32(T))
    grp.attrs["label"] = label
    return True


def rebuild_splits(all_keys, n_splits=5, train_ratio=0.8, seed=42):
    rng = random.Random(seed)
    splits = []
    for _ in range(n_splits):
        shuffled = all_keys[:]
        rng.shuffle(shuffled)
        n_train = int(len(shuffled) * train_ratio)
        splits.append({"train_keys": shuffled[:n_train],
                        "test_keys":  shuffled[n_train:]})
    return splits


# ── 비디오 목록 수집 ──────────────────────────────────────────────────

def collect_archive_videos():
    """archive_extracted 에서 fight 영상 전체 수집."""
    videos = []
    for sub in ARCHIVE_SUBDIRS:
        d = ARCHIVE_DIR / sub
        if not d.exists():
            continue
        for ext in ("*.mpeg", "*.mp4", "*.avi"):
            for p in sorted(d.glob(ext)):
                # key: "Archive__CCTV_DATA__training__fight_0002"
                key = f"Archive__{sub.replace('/', '__')}__{p.stem}"
                videos.append((p, key, "anomaly"))
    return videos


def collect_extra_normal_videos(existing_keys):
    """UCF 정상 영상 중 h5에 없는 것 전체 수집."""
    videos = []
    for d_name in EXTRA_NORMAL_DIRS:
        d = VIDEOS_DIR / d_name
        if not d.exists():
            continue
        for ext in ("*.mp4", "*.mpeg", "*.avi"):
            for p in sorted(d.glob(ext)):
                key = f"Normal__{d_name}__{p.stem}"
                if key not in existing_keys:
                    videos.append((p, key, "normal"))
    return videos


# ── Main ──────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if not H5_PATH.exists():
        print(f"[ERROR] 기존 h5 없음. prepare_ucf_h5.py 먼저 실행: {H5_PATH}")
        sys.exit(1)
    if not ARCHIVE_DIR.exists():
        print(f"[ERROR] archive 미추출: {ARCHIVE_DIR}")
        print("       unzip /home/hyrn2/github/archive.zip -d /home/hyrn2/github/archive_extracted/")
        sys.exit(1)

    # 기존 h5 key 목록
    with h5py.File(H5_PATH, "r") as hf:
        existing_keys = set(hf.keys())
    print(f"기존 h5: {len(existing_keys)}개 비디오")

    # 추가할 비디오 수집
    archive_videos = collect_archive_videos()
    extra_normal   = collect_extra_normal_videos(existing_keys)

    new_fight  = [(p, k, l) for p, k, l in archive_videos if k not in existing_keys]
    new_normal = [(p, k, l) for p, k, l in extra_normal   if k not in existing_keys]
    all_new    = new_fight + new_normal

    print(f"\n추가할 비디오:")
    print(f"  Fight (archive) : {len(new_fight)}개")
    print(f"  Normal (UCF)    : {len(new_normal)}개")
    print(f"  합계            : {len(all_new)}개")

    if not all_new:
        print("추가할 비디오 없음. 이미 완료된 상태.")
        return

    # Feature extractor 초기화
    print("\nResNet-50 로드 중...")
    extractor = ResNet50Extractor(device=device, batch_size=BATCH_SIZE)

    # h5 append
    print(f"\nH5 append 시작: {H5_PATH}")
    added, failed = 0, 0

    with h5py.File(H5_PATH, "a") as hf:
        for video_path, key, label in tqdm(all_new, desc="Append 중"):
            try:
                ok = process_and_write(hf, key, video_path, label, extractor)
                if ok:
                    added += 1
                # else: already exists (skip)
            except Exception as e:
                failed += 1
                tqdm.write(f"  [FAIL] {key}: {e}")

    total_in_h5 = len(existing_keys) + added
    print(f"\nAppend 완료: +{added}개 추가  (실패: {failed}개)")
    print(f"총 h5 비디오: {total_in_h5}개")
    print(f"H5 크기: {H5_PATH.stat().st_size / 1e9:.2f} GB")

    # Splits 재생성 (전체 key 기준)
    print("\nSplits 재생성 중...")
    with h5py.File(H5_PATH, "r") as hf:
        all_keys = list(hf.keys())

    splits = rebuild_splits(all_keys, N_SPLITS, TRAIN_RATIO, SEED)
    with open(SPLITS_PATH, "w") as f:
        json.dump(splits, f, indent=2)

    n_train = len(splits[0]["train_keys"])
    n_test  = len(splits[0]["test_keys"])
    print(f"Splits 저장: {SPLITS_PATH}")
    print(f"  split 0: train={n_train}개, test={n_test}개")
    print(f"\n완료! 이제 train_ucf.py를 실행하세요.")


if __name__ == "__main__":
    main()
