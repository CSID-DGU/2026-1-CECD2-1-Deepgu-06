import argparse

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score

from models.fast.dataset import FightClipDataset
from models.fast.losses import BinaryFocalLoss
from models.fast.x3d_model import build_fast_model
from utils.config import load_config
from utils.io import ensure_dir


def compute_pos_weight(csv_path, device):
    table = pd.read_csv(csv_path)
    label_counts = table["label"].value_counts().to_dict()
    pos_count = int(label_counts.get(1, 0))
    neg_count = int(label_counts.get(0, 0))
    if pos_count <= 0 or neg_count <= 0:
        return None, {"pos_count": pos_count, "neg_count": neg_count}
    value = float(neg_count) / float(pos_count)
    tensor = torch.tensor([value], dtype=torch.float32, device=device)
    return tensor, {"pos_count": pos_count, "neg_count": neg_count, "pos_weight": value}


def train(args):
    config = load_config(args.config)
    loss_config = config.get("loss", {})
    sampler_config = config.get("sampler", {})
    train_csv = args.csv or config.get("train", {}).get("train_csv")
    val_csv = args.val_csv or config.get("train", {}).get("val_csv")
    if not train_csv:
        raise ValueError("training csv is required via --csv or train.train_csv")

    dataset = FightClipDataset(
        train_csv,
        resize_width=int(config["clip"]["resize"]["width"]),
        resize_height=int(config["clip"]["resize"]["height"]),
        num_samples=int(config["clip"]["sampled_frames"]),
        sampling=str(config["clip"].get("sampling", "uniform")),
    )
    loader_config = config.get("data_loader", {})
    num_workers = int(loader_config.get("num_workers", 0))
    pin_memory = bool(loader_config.get("pin_memory", False))
    persistent_workers = bool(loader_config.get("persistent_workers", False)) and num_workers > 0
    sampler = build_weighted_sampler(dataset.table, sampler_config) if bool(sampler_config.get("enabled", False)) else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = None
    if val_csv:
        val_dataset = FightClipDataset(
            val_csv,
            resize_width=int(config["clip"]["resize"]["width"]),
            resize_height=int(config["clip"]["resize"]["height"]),
            num_samples=int(config["clip"]["sampled_frames"]),
            sampling=str(config["clip"].get("sampling", "uniform")),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    model = build_fast_model(
        architecture=config["fast_model"].get("architecture", "x3d_s"),
        num_classes=1,
        pretrained=bool(config["fast_model"].get("use_pretrained_backbone", False)),
        input_clip_length=int(config["fast_model"].get("input_clip_length", config["clip"]["sampled_frames"])),
        input_crop_size=int(config["fast_model"].get("input_crop_size", config["clip"]["resize"]["width"])),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    pos_weight = None
    pos_weight_info = None
    if bool(config["fast_model"].get("use_pos_weight", True)):
        pos_weight, pos_weight_info = compute_pos_weight(train_csv, device=device)
    loss_type = str(loss_config.get("type", "bce")).lower()
    if loss_type == "focal":
        criterion = BinaryFocalLoss(
            gamma=float(loss_config.get("gamma", 2.0)),
            alpha=loss_config.get("alpha"),
            pos_weight=pos_weight,
            reduction="mean",
        )
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    use_amp = bool(config["fast_model"].get("amp", True)) and device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if pos_weight_info:
        print(f"[train] class_balance={pos_weight_info}")
    print(f"[train] loss={{'type': '{loss_type}', 'gamma': {loss_config.get('gamma', 2.0)}, 'alpha': {loss_config.get('alpha', None)}}}")
    print(f"[train] clip_sampling={config['clip'].get('sampling', 'uniform')}")
    if sampler is not None:
        print(f"[train] sampler={sampler_config}")
    early_stopping = config.get("early_stopping", {})
    monitor = str(early_stopping.get("monitor", "pr_auc")).lower()
    enabled = bool(early_stopping.get("enabled", bool(val_loader)))
    patience = int(early_stopping.get("patience", 2))
    min_delta = float(early_stopping.get("min_delta", 1e-4))
    mode = "max" if monitor in {"pr_auc", "roc_auc"} else "min"
    best_metric = float("-inf") if mode == "max" else float("inf")
    best_epoch = 0
    stale_epochs = 0

    output_dir = ensure_dir(args.output_dir)
    architecture = config["fast_model"].get("architecture", "x3d_s")
    best_checkpoint_path = output_dir / f"{architecture}_fast_model_best.pt"
    last_checkpoint_path = output_dir / f"{architecture}_fast_model.pt"
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}"):
            inputs = batch["inputs"].to(device, non_blocking=pin_memory)
            labels = batch["label"].to(device, non_blocking=pin_memory)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(inputs).flatten()
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())
        train_loss = running_loss / max(len(loader), 1)
        log_message = f"[train] epoch={epoch + 1} train_loss={train_loss:.4f}"

        val_metrics = None
        if val_loader is not None:
            val_metrics = evaluate_model(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                use_amp=use_amp,
                pin_memory=pin_memory,
            )
            log_message += (
                f" val_loss={val_metrics['loss']:.4f}"
                f" val_pr_auc={val_metrics['pr_auc']:.4f}"
                f" val_roc_auc={val_metrics['roc_auc']:.4f}"
            )
        print(log_message)

        if val_metrics is not None:
            current_metric = float(val_metrics[monitor])
            improved = (
                current_metric > best_metric + min_delta
                if mode == "max"
                else current_metric < best_metric - min_delta
            )
            if improved:
                best_metric = current_metric
                best_epoch = epoch + 1
                stale_epochs = 0
                save_checkpoint(
                    path=best_checkpoint_path,
                    model=model,
                    architecture=architecture,
                    config=config,
                    pos_weight=pos_weight,
                    loss_type=loss_type,
                    loss_config=loss_config,
                    extra_metadata={
                        "best_epoch": best_epoch,
                        "best_monitor": monitor,
                        "best_metric": best_metric,
                        "val_metrics": val_metrics,
                    },
                )
                print(
                    f"[train] new_best epoch={best_epoch} monitor={monitor} "
                    f"value={best_metric:.4f} path={best_checkpoint_path}"
                )
            else:
                stale_epochs += 1
                print(
                    f"[train] no_improve monitor={monitor} stale_epochs={stale_epochs}/{patience}"
                )
                if enabled and stale_epochs >= patience:
                    print(
                        f"[train] early_stopping triggered at epoch={epoch + 1} "
                        f"best_epoch={best_epoch} best_{monitor}={best_metric:.4f}"
                    )
                    break

    save_checkpoint(
        path=last_checkpoint_path,
        model=model,
        architecture=architecture,
        config=config,
        pos_weight=pos_weight,
        loss_type=loss_type,
        loss_config=loss_config,
        extra_metadata={
            "best_epoch": best_epoch,
            "best_monitor": monitor,
            "best_metric": best_metric if best_epoch else None,
        },
    )
    print(f"[train] saved last checkpoint: {last_checkpoint_path}")
    if best_epoch:
        print(
            f"[train] best checkpoint summary: epoch={best_epoch} "
            f"{monitor}={best_metric:.4f} path={best_checkpoint_path}"
        )


def build_weighted_sampler(table, sampler_config):
    weights_path = sampler_config.get("weights_csv")
    if not weights_path:
        raise ValueError("sampler.weights_csv must be set when sampler.enabled=true")

    weights_table = pd.read_csv(weights_path)
    if "clip_id" not in weights_table.columns or "sample_weight" not in weights_table.columns:
        raise ValueError("weights_csv must contain clip_id and sample_weight columns")
    weight_by_clip = {
        str(row["clip_id"]): float(row["sample_weight"])
        for _, row in weights_table.iterrows()
    }
    weights = []
    missing = 0
    default_weight = float(sampler_config.get("default_weight", 1.0))
    for _, row in table.iterrows():
        clip_id = str(row["clip_id"])
        if clip_id not in weight_by_clip:
            missing += 1
        weights.append(weight_by_clip.get(clip_id, default_weight))
    if missing:
        print(f"[train] sampler missing_weights={missing}, default_weight={default_weight}")
    weight_tensor = torch.tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(weight_tensor, num_samples=len(weight_tensor), replacement=True)


@torch.inference_mode()
def evaluate_model(model, loader, criterion, device, use_amp, pin_memory):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_probs = []
    for batch in loader:
        inputs = batch["inputs"].to(device, non_blocking=pin_memory)
        labels = batch["label"].to(device, non_blocking=pin_memory)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(inputs).flatten()
            loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)
        running_loss += float(loss.item())
        all_labels.extend(labels.detach().cpu().tolist())
        all_probs.extend(probs.detach().cpu().tolist())

    avg_loss = running_loss / max(len(loader), 1)
    pr_auc = safe_metric(average_precision_score, all_labels, all_probs)
    roc_auc = safe_metric(roc_auc_score, all_labels, all_probs)
    return {"loss": avg_loss, "pr_auc": pr_auc, "roc_auc": roc_auc}


def safe_metric(metric_fn, labels, probs):
    if len(set(labels)) < 2:
        return 0.0
    try:
        return float(metric_fn(labels, probs))
    except ValueError:
        return 0.0


def save_checkpoint(path, model, architecture, config, pos_weight, loss_type, loss_config, extra_metadata=None):
    payload = {
        "state_dict": model.state_dict(),
        "architecture": architecture,
        "input_clip_length": int(config["clip"]["sampled_frames"]),
        "input_crop_size": int(config["clip"]["resize"]["width"]),
        "clip_sampling": str(config["clip"].get("sampling", "uniform")),
        "pos_weight": None if pos_weight is None else float(pos_weight.detach().cpu().item()),
        "loss": {
            "type": loss_type,
            "gamma": float(loss_config.get("gamma", 2.0)),
            "alpha": loss_config.get("alpha"),
        },
    }
    if extra_metadata:
        payload.update(extra_metadata)
    torch.save(payload, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/home/deepgu/slowfast/configs/base.yaml")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--val-csv", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", default="/home/deepgu/slowfast/models/fast/checkpoints")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
