"""
DifferentiableFrameSelector 학습 스크립트.

pseudo-label(train.json)을 이용해 FrameScorer를 학습합니다.
학습 목표: "선택한 frame으로 anomaly/normal을 classifier가 맞히도록"
         → FrameScorer는 classifier가 맞히게 해주는 frame을 고르는 법을 배움

출력:
  SAVE_PATH: 최고 val accuracy의 FrameScorer weight (.pth)
  SAVE_PATH.log.json: epoch별 loss/accuracy 기록

사용법:
  python scripts/train_frame_selector.py \
      --label_path outputs/training_data/train.json \
      --save_path  outputs/frame_selector.pth \
      --n_frames 8 \
      --epochs 50 \
      --device cuda
"""

import os
import sys
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.frame_selector import DifferentiableFrameSelector


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class ClipDataset(Dataset):
    def __init__(self, records, max_len=300):
        self.records = records
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        features = np.load(item["features_path"]).astype(np.float32)  # (T, 2048)

        T = len(features)
        if T > self.max_len:
            indices  = np.linspace(0, T - 1, self.max_len).astype(int)
            features = features[indices]

        return torch.FloatTensor(features), int(item["label"])


def collate_fn(batch):
    """길이가 다른 feature를 zero-padding으로 맞춥니다."""
    feats, labels = zip(*batch)
    max_t = max(f.shape[0] for f in feats)
    dim   = feats[0].shape[1]

    padded  = torch.zeros(len(feats), max_t, dim)
    lengths = []
    for i, f in enumerate(feats):
        t = f.shape[0]
        padded[i, :t] = f
        lengths.append(t)

    return padded, torch.LongTensor(labels), torch.LongTensor(lengths)


# ------------------------------------------------------------------
# 학습/검증 step
# ------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, device, temperature, train=True):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0

    for feats, labels, lengths in loader:
        feats  = feats.to(device)
        labels = labels.to(device)

        batch_logits = []
        for i in range(len(feats)):
            t      = lengths[i].item()
            feat_i = feats[i, :t]                        # (T, 2048)
            logits, _ = model(feat_i, temperature)       # (n_classes,)
            batch_logits.append(logits)

        logits_batch = torch.stack(batch_logits)         # (B, n_classes)
        loss = criterion(logits_batch, labels)

        if train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        correct    += (logits_batch.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(loader), correct / total * 100


# ------------------------------------------------------------------
# 메인
# ------------------------------------------------------------------

def main(args):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 데이터 로드
    with open(args.label_path) as f:
        records = json.load(f)

    print(f"전체 clip: {len(records)}개 "
          f"(anomaly={sum(r['label'] for r in records)}, "
          f"normal={sum(1-r['label'] for r in records)})")

    # stratified train/val split (8:2)
    anom_recs   = [r for r in records if r["label"] == 1]
    normal_recs = [r for r in records if r["label"] == 0]
    random.shuffle(anom_recs)
    random.shuffle(normal_recs)

    def split(lst, ratio=0.8):
        n = int(len(lst) * ratio)
        return lst[:n], lst[n:]

    train_anom,   val_anom   = split(anom_recs)
    train_normal, val_normal = split(normal_recs)
    train_recs = train_anom + train_normal
    val_recs   = val_anom   + val_normal
    random.shuffle(train_recs)
    random.shuffle(val_recs)

    print(f"train: {len(train_recs)}, val: {len(val_recs)}")

    train_loader = DataLoader(ClipDataset(train_recs), batch_size=args.batch_size,
                              shuffle=True,  collate_fn=collate_fn, num_workers=2)
    val_loader   = DataLoader(ClipDataset(val_recs),   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn, num_workers=2)

    # 모델
    model = DifferentiableFrameSelector(
        input_dim=2048, hidden_dim=256,
        n_frames=args.n_frames, n_classes=2,
    ).to(args.device)

    optimizer  = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion  = nn.CrossEntropyLoss()

    # temperature 스케줄: 학습 초기 1.0 → 후기 0.1 (점점 hard selection)
    temp_start, temp_end = 1.0, 0.1

    best_val_acc = 0.0
    log = []

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    print(f"\n학습 시작 (epochs={args.epochs}, device={args.device})\n")
    print(f"{'Epoch':>6} | {'TrainLoss':>9} | {'TrainAcc':>8} | {'ValLoss':>7} | {'ValAcc':>6} | {'Temp':>5}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        temp = temp_start + (temp_end - temp_start) * (epoch / args.epochs)

        tr_loss, tr_acc = run_epoch(model, train_loader, criterion,
                                    optimizer, args.device, temp, train=True)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion,
                                    None,      args.device, temp, train=False)
        scheduler.step()

        mark = " ★" if va_acc > best_val_acc else ""
        print(f"{epoch:>6} | {tr_loss:>9.4f} | {tr_acc:>7.1f}% | "
              f"{va_loss:>7.4f} | {va_acc:>5.1f}%{mark} | {temp:.3f}")

        log.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                    "val_loss": va_loss, "val_acc": va_acc, "temperature": temp})

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            # FrameScorer weight만 저장 (KeyframeSampler Phase 2에서 로드)
            torch.save(model.scorer.state_dict(), args.save_path)

    # 로그 저장
    log_path = args.save_path.replace(".pth", ".log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n학습 완료. best val acc = {best_val_acc:.1f}%")
    print(f"FrameScorer 저장: {args.save_path}")
    print(f"학습 로그 저장  : {log_path}")
    print(f"\nPhase 2 전환 방법:")
    print(f"  from pipeline.sampler import KeyframeSampler")
    print(f"  sampler = KeyframeSampler(n_frames={args.n_frames}, "
          f"model_path='{args.save_path}')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FrameScorer 학습")
    parser.add_argument("--label_path",  required=True,
                        help="prepare_data.py가 생성한 train.json 경로")
    parser.add_argument("--save_path",   default="outputs/frame_selector.pth")
    parser.add_argument("--n_frames",    type=int,   default=8)
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args)
