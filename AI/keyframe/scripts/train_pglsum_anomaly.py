"""
PGL-SUM을 이상행동 감지 목적으로 재학습.

기존 PGL-SUM은 SumMe/TVSum 영상 요약 데이터로 학습됨 → "대표 장면 선택".
이 스크립트는 UCF-Crime 데이터의 anomaly/normal 레이블로 재학습하여
"이상행동이 담긴 프레임 선택"에 맞게 목적함수를 바꿉니다.

학습 데이터 구성:
  train.json 의 clip-level 특징(ResNet-50, 16×2048)을 영상 단위로 묶어서 사용.
  - temporal annotation 있는 영상: 프레임 레벨 binary 레이블
  - 그 외 anomaly 영상: 전체 프레임 → 1.0
  - normal 영상: 전체 프레임 → 0.0

손실 함수: BCELoss (sigmoid 출력 → 이상행동 확률)

사용법:
  cd AI/ai_pipeline
  python scripts/train_pglsum_anomaly.py \\
      --train_json        outputs/training_data/train.json \\
      --annotation_path   Data/annotations/temporal_anomaly.txt \\
      --pglsum_init_path  ../../PGL-SUM/Summaries/UCF/models/split0/best_model.pth \\
      --save_path         outputs/pglsum_anomaly.pth \\
      --epochs            30 \\
      --device            cuda
"""

import os
import sys
import json
import random
import argparse
import datetime
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# PGL-SUM import path 설정
_pglsum_model  = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../PGL-SUM/model")
)
_pglsum_layers = os.path.join(_pglsum_model, "layers")
for _p in [_pglsum_layers, _pglsum_model]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from summarizer import PGL_SUM


def _ts():
    return datetime.datetime.now().strftime("%H:%M:%S")


# -----------------------------------------------------------------------
# Temporal annotation 로드
# -----------------------------------------------------------------------

def load_annotations(path):
    """
    반환: { video_stem: [(start_frame, end_frame), ...] }
    format: VideoName.mp4  Class  start  end  [start2  end2]
    """
    ann = {}
    if not path or not os.path.isfile(path):
        return ann
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            stem = parts[0].replace(".mp4", "")
            s1, e1 = int(parts[2]), int(parts[3])
            segs = [(s1, e1)]
            if len(parts) >= 6 and parts[4] != "-1":
                segs.append((int(parts[4]), int(parts[5])))
            ann[stem] = segs
    return ann


# -----------------------------------------------------------------------
# 비디오별 클립 그룹화 및 특징/타겟 구성
# -----------------------------------------------------------------------

_STRIDE = 8
_CLIP_LEN = 16
_TARGET_ANOMALY_CLASSES = {"Abuse", "Assault", "Fighting"}


def build_video_index(train_json, temporal_ann):
    """
    메모리 효율적 인덱스만 구성. 특징은 학습 중 영상별로 로드.

    반환: list of (stem, gt_label, non_overlap_clips, has_temporal)
      non_overlap_clips: [(clip_idx, features_path), ...]
    """
    with open(train_json) as f:
        all_clips = json.load(f)

    video_clips = defaultdict(list)
    for rec in all_clips:
        stem = rec["clip_id"].rsplit("_clip", 1)[0]
        clip_idx = int(rec["clip_id"].rsplit("_clip", 1)[1])
        video_clips[stem].append((clip_idx, rec["features_path"], int(rec["gt_label"])))

    index = []
    n_temporal = 0
    for stem, clips in video_clips.items():
        clips.sort(key=lambda x: x[0])
        non_overlap = [(ci, fp, gt) for ci, fp, gt in clips if ci % 2 == 0]
        if not non_overlap:
            non_overlap = clips[:1]
        gt_label = non_overlap[0][2]
        has_temporal = stem in temporal_ann
        if has_temporal:
            n_temporal += 1
        index.append((stem, gt_label, non_overlap, has_temporal))

    print(f"  총 영상: {len(index)} (temporal annotation: {n_temporal}개)")
    return index


def load_video_data(stem, gt_label, non_overlap_clips, temporal_ann, max_frames):
    """영상 한 개의 features와 targets를 로드. 학습 루프에서 호출."""
    feat_list = []
    for ci, fp, _ in non_overlap_clips:
        if not os.path.isfile(fp):
            continue
        feat_list.append((ci, np.load(fp).astype(np.float32)))
    if not feat_list:
        return None, None

    features = np.concatenate([f for _, f in feat_list], axis=0)  # (T, 2048)
    T = len(features)

    targets = np.zeros(T, dtype=np.float32)
    if stem in temporal_ann:
        segs = temporal_ann[stem]
        anomaly_set = set()
        for s, e in segs:
            anomaly_set.update(range(s, e + 1))
        for local_t, (ci, _) in enumerate(feat_list):
            start_frame = ci * _STRIDE
            for f in range(_CLIP_LEN):
                frame_t = local_t * _CLIP_LEN + f
                if frame_t < T and (start_frame + f) in anomaly_set:
                    targets[frame_t] = 1.0
    elif gt_label == 1:
        targets[:] = 1.0

    # 긴 영상 서브샘플
    if T > max_frames:
        idx = sorted(random.sample(range(T), max_frames))
        features = features[idx]
        targets  = targets[idx]

    return features, targets


# -----------------------------------------------------------------------
# 학습
# -----------------------------------------------------------------------

def train(args):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Temporal annotation 로드
    temporal_ann = load_annotations(args.annotation_path)
    target_classes = {"Abuse", "Assault", "Fighting"}
    temporal_ann = {k: v for k, v in temporal_ann.items()
                    if any(k.startswith(c) for c in target_classes)}
    print(f"[{_ts()}] Temporal annotation: {len(temporal_ann)}개 영상")

    # 데이터셋 인덱스 구성 (특징은 학습 중 로드)
    print(f"[{_ts()}] 데이터셋 인덱스 구성 중... (train.json 로딩)")
    dataset = build_video_index(args.train_json, temporal_ann)
    random.shuffle(dataset)

    anom_vids = [d for d in dataset if d[1] == 1]
    norm_vids  = [d for d in dataset if d[1] == 0]
    print(f"[{_ts()}] anomaly={len(anom_vids)}, normal={len(norm_vids)}")

    # PGL-SUM 모델 초기화 (pretrained 가중치 로드)
    model = PGL_SUM(
        input_size=2048,
        output_size=2048,
        num_segments=4,
        heads=8,
        fusion="add",
        pos_enc="absolute",
    ).to(args.device)

    if args.pglsum_init_path and os.path.isfile(args.pglsum_init_path):
        state = torch.load(args.pglsum_init_path, map_location=args.device)
        model.load_state_dict(state)
        print(f"[{_ts()}] 초기 가중치 로드: {args.pglsum_init_path}")
    else:
        print(f"[{_ts()}] 초기 가중치 없음 — 랜덤 초기화")

    # 옵티마이저: 사전학습 가중치를 천천히 변경하기 위해 낮은 lr
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.BCELoss()

    best_loss = float("inf")
    best_state = None

    print(f"\n[{_ts()}] 학습 시작 (epochs={args.epochs}, lr={args.lr})\n")
    print(f" {'Epoch':>5} | {'TrainLoss':>10} | {'AnomalyAcc':>10} | {'NormalAcc':>10}")
    print("-" * 50)

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(dataset)

        loss_hist = []
        anom_correct = anom_total = 0
        norm_correct = norm_total = 0

        # 영상 단위 업데이트 (batch_size=1, gradient accumulation)
        optimizer.zero_grad()
        accum_steps = 0

        for stem, gt_label, non_overlap_clips, has_temporal in dataset:
            features, targets = load_video_data(
                stem, gt_label, non_overlap_clips, temporal_ann, args.max_frames
            )
            if features is None or len(features) < 8:
                continue
            T = len(features)

            x = torch.FloatTensor(features).to(args.device)          # (T, 2048)
            y = torch.FloatTensor(targets).to(args.device)            # (T,)

            scores, _ = model(x)          # (1, T)
            scores = scores.squeeze(0)    # (T,)

            loss = criterion(scores, y)
            (loss / args.grad_accum).backward()
            loss_hist.append(loss.item())

            accum_steps += 1
            if accum_steps % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad()

            # 정확도 (0.5 threshold)
            pred = (scores.detach().cpu().numpy() > 0.5).astype(float)
            true = targets
            if gt_label == 1:
                anom_correct += int((pred == true).sum())
                anom_total   += len(pred)
            else:
                norm_correct += int((pred == true).sum())
                norm_total   += len(pred)

        # 마지막 남은 gradient
        if accum_steps % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        avg_loss = np.mean(loss_hist)
        anom_acc = anom_correct / anom_total * 100 if anom_total else 0
        norm_acc = norm_correct / norm_total * 100 if norm_total else 0

        marker = " ★" if avg_loss < best_loss else ""
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(f" {epoch:>5} | {avg_loss:>10.4f} | {anom_acc:>9.1f}% | {norm_acc:>9.1f}%{marker}")

    # 최적 모델 저장
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    torch.save(best_state, args.save_path)
    print(f"\n[{_ts()}] 저장 완료: {args.save_path}  (best_loss={best_loss:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json",       default="outputs/training_data/train.json")
    parser.add_argument("--annotation_path",  default="Data/annotations/temporal_anomaly.txt")
    parser.add_argument("--pglsum_init_path", default=None,
                        help="초기화할 PGL-SUM 체크포인트. None이면 랜덤 초기화.")
    parser.add_argument("--save_path",        default="outputs/pglsum_anomaly.pth")
    parser.add_argument("--epochs",           type=int,   default=30)
    parser.add_argument("--lr",               type=float, default=1e-5)
    parser.add_argument("--max_frames",       type=int,   default=512,
                        help="영상당 최대 학습 프레임 수 (메모리 조절)")
    parser.add_argument("--grad_accum",       type=int,   default=8,
                        help="Gradient accumulation steps (effective batch size)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    train(args)
