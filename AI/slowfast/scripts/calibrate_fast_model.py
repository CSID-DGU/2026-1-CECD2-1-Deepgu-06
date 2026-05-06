import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.fast.calibration import evaluate_logits, fit_temperature
from models.fast.dataset import FightClipDataset
from models.fast.x3d_model import build_fast_model
from utils.config import load_config
from utils.io import write_json


def load_model(config, checkpoint_path, device):
    model = build_fast_model(
        architecture=config["fast_model"].get("architecture", "x3d_s"),
        num_classes=1,
        pretrained=bool(config["fast_model"].get("use_pretrained_backbone", False)),
        input_clip_length=int(config["fast_model"].get("input_clip_length", config["clip"]["sampled_frames"])),
        input_crop_size=int(config["fast_model"].get("input_crop_size", config["clip"]["resize"]["width"])),
    ).to(device)
    state = torch.load(checkpoint_path, map_location="cpu")
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, state


def collect_logits(model, loader, device, use_amp):
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="collect_logits"):
            inputs = batch["inputs"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(inputs).flatten()
            for clip_id, video_id, label, logit in zip(
                batch["clip_id"],
                batch["video_id"],
                labels.detach().cpu().tolist(),
                logits.detach().cpu().tolist(),
            ):
                rows.append(
                    {
                        "clip_id": str(clip_id),
                        "video_id": str(video_id),
                        "label": float(label),
                        "logit": float(logit),
                    }
                )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/home/deepgu/slowfast/configs/base.yaml")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--output-json",
        default="/home/deepgu/slowfast/outputs/validation/fast_model_calibration.json",
    )
    parser.add_argument(
        "--output-checkpoint",
        default=None,
        help="Optional path to save a checkpoint copy with calibration metadata embedded.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint_path = args.checkpoint or config["fast_model"]["checkpoint_path"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = bool(config["fast_model"].get("amp", True)) and device == "cuda"

    dataset = FightClipDataset(
        args.csv,
        resize_width=int(config["clip"]["resize"]["width"]),
        resize_height=int(config["clip"]["resize"]["height"]),
        num_samples=int(config["clip"]["sampled_frames"]),
    )
    loader_config = config.get("data_loader", {})
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(loader_config.get("num_workers", 0)),
        pin_memory=bool(loader_config.get("pin_memory", False)),
        persistent_workers=bool(loader_config.get("persistent_workers", False)) and int(loader_config.get("num_workers", 0)) > 0,
    )

    model, checkpoint_state = load_model(config, checkpoint_path, device)
    rows = collect_logits(model, loader, device, use_amp)
    logits = [row["logit"] for row in rows]
    labels = [row["label"] for row in rows]

    before = evaluate_logits(logits, labels, temperature=1.0)
    fitted_temperature = fit_temperature(logits, labels)
    after = evaluate_logits(logits, labels, temperature=fitted_temperature)

    payload = {
        "config_path": args.config,
        "checkpoint_path": str(checkpoint_path),
        "csv": args.csv,
        "num_rows": len(rows),
        "temperature": float(fitted_temperature),
        "before": before,
        "after": after,
        "improvements": {
            "bce_delta": float(after["bce"] - before["bce"]),
            "brier_delta": float(after["brier"] - before["brier"]),
            "ece_delta": float(after["ece"] - before["ece"]),
            "accuracy_delta": float(after["accuracy"] - before["accuracy"]),
        },
        "rows": rows,
    }
    write_json(args.output_json, payload)

    if args.output_checkpoint:
        updated_state = checkpoint_state if isinstance(checkpoint_state, dict) else {"state_dict": checkpoint_state}
        updated_state["calibration"] = {
            "method": "temperature_scaling",
            "temperature": float(fitted_temperature),
            "source_csv": args.csv,
        }
        torch.save(updated_state, args.output_checkpoint)

    print(f"[calibration] output={args.output_json}")
    print(f"[calibration] temperature={fitted_temperature:.6f}")
    print(f"[calibration] before={before}")
    print(f"[calibration] after={after}")


if __name__ == "__main__":
    main()
