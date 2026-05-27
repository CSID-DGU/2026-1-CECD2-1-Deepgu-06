import argparse
import math
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.fast_stage import score_clips_fast
from utils.video import load_video_frames
from utils.config import load_config
from utils.io import ensure_dir, write_json


def binary_entropy(probability, eps=1e-6):
    p = min(max(float(probability), eps), 1.0 - eps)
    return float(-(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)))


def summarize(values):
    if not values:
        return {"count": 0}
    series = pd.Series(values)
    return {
        "count": int(series.shape[0]),
        "mean": float(series.mean()),
        "std": float(series.std(ddof=0)),
        "min": float(series.min()),
        "p10": float(series.quantile(0.10)),
        "p25": float(series.quantile(0.25)),
        "p50": float(series.quantile(0.50)),
        "p75": float(series.quantile(0.75)),
        "p90": float(series.quantile(0.90)),
        "p95": float(series.quantile(0.95)),
        "max": float(series.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/home/deepgu/slowfast/configs/base.yaml")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint_path = args.checkpoint or config["fast_model"]["checkpoint_path"]
    config["fast_model"]["checkpoint_path"] = checkpoint_path

    table = pd.read_csv(args.csv)
    rows = []
    for video_path, video_rows in table.groupby("video_path", sort=False):
        frames, fps = load_video_frames(video_path)
        clips = []
        metadata = []
        for _, row in video_rows.iterrows():
            start_frame = int(row["start_frame"])
            end_frame = int(row["end_frame"])
            clip_frames = frames[start_frame : end_frame + 1]
            clips.append(
                {
                    "clip_id": int(len(clips)),
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_time": float(row.get("start_time", start_frame / max(fps, 1e-6))),
                    "end_time": float(row.get("end_time", (end_frame + 1) / max(fps, 1e-6))),
                    "frames": clip_frames,
                }
            )
            metadata.append(
                {
                    "clip_id": row["clip_id"],
                    "video_id": row["video_id"],
                    "label": int(row["label"]),
                }
            )

        scored = score_clips_fast(clips, config["fast_model"], config["clip"])
        for meta, scored_item in zip(metadata, scored):
            prob = float(scored_item["fighting_prob"])
            rows.append(
                {
                    "clip_id": meta["clip_id"],
                    "video_id": meta["video_id"],
                    "label": meta["label"],
                    "fighting_prob": prob,
                    "entropy": binary_entropy(prob),
                }
            )

    positives = [row["fighting_prob"] for row in rows if row["label"] == 1]
    negatives = [row["fighting_prob"] for row in rows if row["label"] == 0]

    all_labels = [row["label"] for row in rows]
    all_probs = [row["fighting_prob"] for row in rows]
    pr_auc = float(average_precision_score(all_labels, all_probs))
    roc_auc = float(roc_auc_score(all_labels, all_probs))

    payload = {
        "checkpoint": str(checkpoint_path),
        "csv": args.csv,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "positive_prob_summary": summarize(positives),
        "negative_prob_summary": summarize(negatives),
        "rows": rows,
    }

    output_path = Path(args.output_json)
    ensure_dir(output_path.parent)
    write_json(output_path, payload)
    print(f"[validation] output={output_path}")
    print(f"[validation] pr_auc={pr_auc:.4f}  roc_auc={roc_auc:.4f}")
    print(f"[validation] positive={payload['positive_prob_summary']}")
    print(f"[validation] negative={payload['negative_prob_summary']}")


if __name__ == "__main__":
    main()
