"""
UCF-Crime 데이터(ResNet-50, 2048-dim)로 PGL-SUM을 학습합니다.

prepare_ucf_h5.py 실행 후 사용하세요.

출력:
  ../Summaries/UCF/models/split{N}/best_model.pth  — best val loss 기준 checkpoint
  ../Summaries/UCF/logs/split{N}/train_log.json     — epoch별 loss 기록

실행:
  cd PGL-SUM
  python model/train_ucf.py --split_index 0 --n_epochs 100

AI pipeline evaluate_selector.py에서 사용:
  pglsum_model = PGLSumSampler(
      model_path="PGL-SUM/Summaries/UCF/models/split0/best_model.pth",
      input_size=2048,
  )
"""

import json
import random
import sys
import argparse
import time
import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# ── PGL-SUM layers import ──────────────────────────────────────────────
# model/ 을 추가해야 'from layers.attention import ...' 가 동작함
# layers/ 도 추가해야 'from summarizer import PGL_SUM' 이 동작함
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))           # PGL-SUM/model/
sys.path.insert(0, str(SCRIPT_DIR / "layers")) # PGL-SUM/model/layers/
from summarizer import PGL_SUM  # noqa: E402

PGLSUM_DIR   = SCRIPT_DIR.parent
UCF_H5_PATH  = PGLSUM_DIR / "data" / "UCF" / "ucf_crime_resnet50.h5"
UCF_SPLITS   = PGLSUM_DIR / "data" / "splits" / "ucf_splits.json"
SUMMARY_BASE = PGLSUM_DIR / "Summaries" / "UCF"


# -----------------------------------------------------------------------
# Logger: stdout + 파일에 동시 기록
# -----------------------------------------------------------------------

class Logger:
    """타임스탬프 붙여 stdout에 출력. nohup 리다이렉션으로 파일에 저장."""
    def log(self, msg=""):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)

    def close(self):
        pass


# -----------------------------------------------------------------------
# Dataset: UCF-Crime h5 파일 전체를 메모리에 로드
# -----------------------------------------------------------------------

class UCFDataset:
    """
    h5에서 비디오별 (features, gtscore)를 메모리에 올려 두는 단순 Dataset.

    self.items : List[dict]  — {"name", "features"(Tensor), "gtscore"(Tensor)}
    """

    def __init__(self, h5_path, splits_json, split_index, mode):
        with open(splits_json) as f:
            splits = json.load(f)
        keys = splits[split_index][f"{mode}_keys"]

        self.items = []
        with h5py.File(h5_path, "r") as hf:
            for key in keys:
                if key not in hf:
                    continue
                features = torch.FloatTensor(np.array(hf[key]["features"]))  # (T, 2048)
                gtscore  = torch.FloatTensor(np.array(hf[key]["gtscore"]))   # (T,)
                self.items.append({"name": key, "features": features, "gtscore": gtscore})

        print(f"  [{mode}] {len(self.items)}개 비디오 로드 완료 (split {split_index})")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# -----------------------------------------------------------------------
# Weight initialization
# -----------------------------------------------------------------------

def init_weights(model, init_type="xavier"):
    for name, param in model.named_parameters():
        if "weight" in name and "norm" not in name:
            if init_type == "xavier":
                nn.init.xavier_uniform_(param, gain=np.sqrt(2.0))
            elif init_type == "kaiming":
                nn.init.kaiming_uniform_(param, mode="fan_in", nonlinearity="relu")
            elif init_type == "normal":
                nn.init.normal_(param, 0.0, 0.02)
            elif init_type == "orthogonal":
                nn.init.orthogonal_(param, gain=np.sqrt(2.0))
        elif "bias" in name:
            nn.init.constant_(param, 0.1)


# -----------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------

def run_epoch_train(model, dataset, optimizer, criterion, device, batch_size, clip):
    """
    전체 train 데이터를 한 epoch 학습.
    batch_size개 비디오마다 optimizer step (gradient accumulation).
    """
    model.train()
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    losses = []
    optimizer.zero_grad()

    for step, i in enumerate(indices):
        item = dataset[i]
        features = item["features"].to(device)   # (T, 2048)
        gtscore  = item["gtscore"].to(device)    # (T,)

        output, _ = model(features)              # (1, T)
        loss = criterion(output.squeeze(0), gtscore)
        loss.backward()
        losses.append(loss.item())

        # batch_size개마다 step
        if (step + 1) % batch_size == 0 or (step + 1) == len(indices):
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            optimizer.zero_grad()

    return float(np.mean(losses))


def run_epoch_val(model, dataset, criterion, device):
    """Validation: gtscore와 model output의 MSE."""
    model.eval()
    losses = []
    with torch.no_grad():
        for item in dataset.items:
            features = item["features"].to(device)
            gtscore  = item["gtscore"].to(device)
            output, _ = model(features)
            loss = criterion(output.squeeze(0), gtscore)
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("inf")


# -----------------------------------------------------------------------
# Main training function
# -----------------------------------------------------------------------

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 출력 디렉토리
    split_tag = f"split{args.split_index}"
    model_dir = SUMMARY_BASE / "models" / split_tag
    log_dir   = SUMMARY_BASE / "logs"   / split_tag
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Logger 초기화 (stdout → nohup이 파일로 redirect)
    logger = Logger()

    logger.log("=" * 60)
    logger.log(f"PGL-SUM 학습 시작")
    logger.log(f"  Device      : {device}")
    logger.log(f"  H5 경로     : {UCF_H5_PATH}")
    logger.log(f"  Splits 경로 : {UCF_SPLITS}")
    logger.log(f"  Split index : {args.split_index}")
    logger.log(f"  Epochs      : {args.n_epochs}")
    logger.log(f"  Batch size  : {args.batch_size} (gradient accumulation)")
    logger.log(f"  LR          : {args.lr}  L2: {args.l2_req}  Clip: {args.clip}")
    logger.log("=" * 60)

    if not UCF_H5_PATH.exists():
        logger.log(f"[ERROR] h5 파일 없음: {UCF_H5_PATH}")
        sys.exit(1)
    if not UCF_SPLITS.exists():
        logger.log(f"[ERROR] splits JSON 없음: {UCF_SPLITS}")
        sys.exit(1)

    # 데이터 로드
    logger.log(f"\n데이터 로드 중...")
    t0 = time.time()
    train_set = UCFDataset(UCF_H5_PATH, UCF_SPLITS, args.split_index, "train")
    val_set   = UCFDataset(UCF_H5_PATH, UCF_SPLITS, args.split_index, "test")
    logger.log(f"  train: {len(train_set)}개  val: {len(val_set)}개  "
               f"(로드 소요: {time.time()-t0:.1f}s)")

    if len(train_set) == 0:
        logger.log("[ERROR] train 데이터가 없습니다.")
        sys.exit(1)

    # 모델 생성
    logger.log(f"\nPGL-SUM 모델 생성:")
    logger.log(f"  input_size={args.input_size}, n_segments={args.n_segments}, heads={args.heads}")
    logger.log(f"  fusion={args.fusion}, pos_enc={args.pos_enc}, init={args.init_type}")
    model = PGL_SUM(
        input_size=args.input_size,
        output_size=args.input_size,
        num_segments=args.n_segments,
        heads=args.heads,
        fusion=args.fusion,
        pos_enc=args.pos_enc,
    ).to(device)
    init_weights(model, args.init_type)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"  학습 가능 파라미터: {n_params:,}개")

    # GPU 메모리 초기 상태
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1e6
        reserved  = torch.cuda.memory_reserved(device)  / 1e6
        logger.log(f"  GPU 메모리: allocated={allocated:.0f}MB  reserved={reserved:.0f}MB")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_req)
    criterion = nn.MSELoss()

    # 학습
    logger.log(f"\n{'─'*60}")
    logger.log(f"학습 시작: {args.n_epochs} epochs")
    logger.log(f"{'─'*60}")
    logger.log(f"{'Epoch':>6}  {'Train':>8}  {'Val':>8}  {'Best':>8}  {'BestEp':>7}  {'ETA':>8}  GPU(MB)")
    logger.log(f"{'─'*60}")

    best_val_loss = float("inf")
    best_epoch    = -1
    log           = []
    train_start   = time.time()

    for epoch in range(args.n_epochs):
        ep_start = time.time()

        train_loss = run_epoch_train(
            model, train_set, optimizer, criterion, device,
            args.batch_size, args.clip
        )
        val_loss = run_epoch_val(model, val_set, criterion, device)

        ep_sec = time.time() - ep_start

        # Best checkpoint 저장
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch    = epoch
            torch.save(model.state_dict(), model_dir / "best_model.pth")

        # ETA 계산
        elapsed   = time.time() - train_start
        done_frac = (epoch + 1) / args.n_epochs
        eta_sec   = elapsed / done_frac * (1 - done_frac) if done_frac > 0 else 0
        eta_str   = str(datetime.timedelta(seconds=int(eta_sec)))

        # GPU 메모리
        gpu_mb = ""
        if torch.cuda.is_available():
            gpu_mb = f"{torch.cuda.memory_allocated(device)/1e6:.0f}"

        log_entry = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss, 6),
            "ep_sec":     round(ep_sec, 1),
            "is_best":    is_best,
        }
        log.append(log_entry)

        # 매 epoch 로그 (파일에는 항상, stdout은 매 epoch)
        best_marker = " *" if is_best else ""
        logger.log(
            f"{epoch+1:>6}/{args.n_epochs}  "
            f"{train_loss:>8.5f}  {val_loss:>8.5f}  "
            f"{best_val_loss:>8.5f}  ep{best_epoch+1:>4}  "
            f"{eta_str:>8}  {gpu_mb}{best_marker}"
        )

        # JSON 로그 매 epoch 덮어쓰기 (실시간 확인 가능)
        log_path = log_dir / "train_log.json"
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

    # 완료
    total_time = str(datetime.timedelta(seconds=int(time.time() - train_start)))
    logger.log(f"\n{'='*60}")
    logger.log(f"학습 완료!")
    logger.log(f"  총 소요 시간  : {total_time}")
    logger.log(f"  Best val loss : {best_val_loss:.6f}  (epoch {best_epoch+1})")
    logger.log(f"  Model 저장    : {model_dir / 'best_model.pth'}")
    logger.log(f"  JSON 로그     : {log_path}")
    logger.log(f"  텍스트 로그   : {log_dir / 'train_progress.log'}")
    logger.log(f"{'='*60}")
    logger.close()

    return str(model_dir / "best_model.pth")


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UCF-Crime에서 PGL-SUM 학습")

    # 데이터
    parser.add_argument("--split_index",  type=int,   default=0,
                        help="사용할 split index (0-4, default: 0)")

    # 모델
    parser.add_argument("--input_size",   type=int,   default=2048,
                        help="입력 feature 차원 (ResNet-50: 2048, default: 2048)")
    parser.add_argument("--n_segments",   type=int,   default=4,
                        help="local attention segment 수 (default: 4)")
    parser.add_argument("--heads",        type=int,   default=8,
                        help="global attention head 수 (default: 8)")
    parser.add_argument("--fusion",       type=str,   default="add",
                        help="local/global fusion (add/mult/avg/max, default: add)")
    parser.add_argument("--pos_enc",      type=str,   default="absolute",
                        help="positional encoding (absolute/relative/None, default: absolute)")
    parser.add_argument("--init_type",    type=str,   default="xavier",
                        help="weight 초기화 (xavier/kaiming/normal/orthogonal, default: xavier)")

    # 학습
    parser.add_argument("--n_epochs",     type=int,   default=100,
                        help="학습 epoch 수 (default: 100)")
    parser.add_argument("--batch_size",   type=int,   default=20,
                        help="gradient 누적 비디오 수 (default: 20)")
    parser.add_argument("--lr",           type=float, default=5e-5,
                        help="learning rate (default: 5e-5)")
    parser.add_argument("--l2_req",       type=float, default=1e-5,
                        help="L2 regularization (default: 1e-5)")
    parser.add_argument("--clip",         type=float, default=5.0,
                        help="gradient norm clip (default: 5.0)")
    parser.add_argument("--seed",         type=int,   default=42,
                        help="random seed (default: 42)")
    parser.add_argument("--device",       type=str,   default="cuda:0",
                        help="학습 device (default: cuda:0)")

    args = parser.parse_args()

    # 재현성
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train(args)
