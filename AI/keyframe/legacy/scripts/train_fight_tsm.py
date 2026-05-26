import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader


ROOT = Path("/home/deepgu/test")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.tsm.inference import TSMModel
from training.fight_dataset import (
    FightClipDataset,
    compute_pos_weight,
    make_balanced_sampler,
    seed_worker,
)
from utils.device import choose_torch_device
from utils.paths import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata-path",
        default="/home/deepgu/test/data/metadata/fight_clip_metadata.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/deepgu/test/outputs/fight_tsm",
    )
    parser.add_argument(
        "--pretrained-path",
        default="/home/deepgu/test/models/tsm/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment16_e50.pth",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-len", type=int, default=16)
    parser.add_argument("--num-segments", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument(
        "--preferred-gpu-indices",
        type=int,
        nargs="*",
        default=[1, 2],
    )
    parser.add_argument(
        "--include-datasets",
        nargs="*",
        default=["cctv_fights", "ucf_crime_eval"],
    )
    parser.add_argument(
        "--train-subsets",
        nargs="*",
        default=["training", "eval"],
    )
    parser.add_argument(
        "--val-subsets",
        nargs="*",
        default=["validation"],
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pretrained_backbone(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        if key.startswith("base_model.fc."):
            continue
        cleaned[key] = value

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"loaded backbone from {checkpoint_path}")
    print(f"missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")


def build_loaders(args):
    train_dataset = FightClipDataset(
        metadata_path=args.metadata_path,
        split="train",
        include_datasets=args.include_datasets,
        train_subsets=args.train_subsets,
        val_subsets=args.val_subsets,
        num_segments=args.num_segments,
        max_samples=args.max_train_samples,
        seed=args.seed,
    )
    val_dataset = FightClipDataset(
        metadata_path=args.metadata_path,
        split="val",
        include_datasets=args.include_datasets,
        train_subsets=args.train_subsets,
        val_subsets=args.val_subsets,
        num_segments=args.num_segments,
        max_samples=args.max_val_samples,
        seed=args.seed,
    )

    train_sampler = make_balanced_sampler(train_dataset)

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    return train_dataset, val_dataset, train_loader, val_loader


def compute_metrics(logits, labels):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()

    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def run_epoch(model, loader, criterion, optimizer, device, train=True, log_interval=100):
    model.train(train)

    total_loss = 0.0
    all_logits = []
    all_labels = []
    total_batches = len(loader)

    for step, batch in enumerate(loader, start=1):
        inputs = batch["frames"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).view(-1, 1)

        with torch.set_grad_enabled(train):
            logits = model(inputs)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

        if step == 1 or step % log_interval == 0 or step == total_batches:
            mode = "train" if train else "val"
            avg_loss = total_loss / max(step * inputs.size(0), 1)
            print(
                f"[{mode}] step {step}/{total_batches} "
                f"loss={loss.item():.4f} avg_loss={avg_loss:.4f}"
            )

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(logits, labels)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics


def save_checkpoint(output_dir, epoch, model, optimizer, metrics, is_best=False):
    ensure_dir(output_dir)
    payload = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
    }

    latest_path = Path(output_dir) / "last.pth"
    torch.save(payload, latest_path)

    if is_best:
        best_path = Path(output_dir) / "best.pth"
        torch.save(payload, best_path)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = choose_torch_device(
        preferred_gpu_indices=args.preferred_gpu_indices,
        allow_cpu_fallback=True,
    )
    print(f"device: {device}")

    train_dataset, val_dataset, train_loader, val_loader = build_loaders(args)
    print(f"train clips: {len(train_dataset)}")
    print(f"val clips: {len(val_dataset)}")

    pos_weight = compute_pos_weight(train_dataset).to(device)
    print(f"pos_weight: {float(pos_weight.item()):.4f}")

    model = TSMModel(
        num_segments=args.num_segments,
        num_classes=1,
    )
    load_pretrained_backbone(model, args.pretrained_path)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_recall = -1.0
    output_dir = ensure_dir(args.output_dir)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train=True,
            log_interval=args.log_interval,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            train=False,
            log_interval=args.log_interval,
        )

        print(
            f"[epoch {epoch}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_recall={train_metrics['recall']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_precision={val_metrics['precision']:.4f} "
            f"val_recall={val_metrics['recall']:.4f}"
        )

        is_best = val_metrics["recall"] > best_recall
        if is_best:
            best_recall = val_metrics["recall"]

        save_checkpoint(
            output_dir=output_dir,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            metrics={
                "train": train_metrics,
                "val": val_metrics,
            },
            is_best=is_best,
        )


if __name__ == "__main__":
    main()
