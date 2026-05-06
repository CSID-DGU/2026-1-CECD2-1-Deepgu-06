import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--score-json", required=True)
    parser.add_argument(
        "--output-csv",
        default="/home/deepgu/slowfast/data/manifests/cctv_x3d_s/training_hard_mining_weights.csv",
    )
    parser.add_argument("--hard-positive-threshold", type=float, default=0.35)
    parser.add_argument("--hard-negative-threshold", type=float, default=0.40)
    parser.add_argument("--negative-weight", type=float, default=1.0)
    parser.add_argument("--positive-weight", type=float, default=1.5)
    parser.add_argument("--hard-positive-weight", type=float, default=3.0)
    parser.add_argument("--hard-negative-weight", type=float, default=1.5)
    args = parser.parse_args()

    table = pd.read_csv(args.csv)
    with open(args.score_json, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    score_table = pd.DataFrame(payload["rows"])
    score_table["clip_id"] = score_table["clip_id"].astype(str)
    table["clip_id"] = table["clip_id"].astype(str)

    merged = table.merge(
        score_table[["clip_id", "fighting_prob", "entropy"]],
        on="clip_id",
        how="left",
        validate="one_to_one",
    )
    if merged["fighting_prob"].isna().any():
        missing = int(merged["fighting_prob"].isna().sum())
        raise ValueError(f"missing fighting_prob for {missing} clips")

    merged["hard_positive"] = (
        (merged["label"].astype(int) == 1)
        & (merged["fighting_prob"].astype(float) < float(args.hard_positive_threshold))
    ).astype(int)
    merged["hard_negative_dynamic"] = (
        (merged["label"].astype(int) == 0)
        & (merged["fighting_prob"].astype(float) > float(args.hard_negative_threshold))
    ).astype(int)

    def compute_weight(row):
        if int(row["hard_positive"]) == 1:
            return float(args.hard_positive_weight)
        if int(row["hard_negative_dynamic"]) == 1:
            return float(args.hard_negative_weight)
        if int(row["label"]) == 1:
            return float(args.positive_weight)
        return float(args.negative_weight)

    merged["sample_weight"] = merged.apply(compute_weight, axis=1)

    summary = {
        "total_rows": int(merged.shape[0]),
        "positive_rows": int((merged["label"] == 1).sum()),
        "negative_rows": int((merged["label"] == 0).sum()),
        "hard_positive_rows": int(merged["hard_positive"].sum()),
        "hard_negative_rows": int(merged["hard_negative_dynamic"].sum()),
        "sample_weight_mean": float(merged["sample_weight"].mean()),
        "sample_weight_max": float(merged["sample_weight"].max()),
    }

    output_columns = [
        "clip_id",
        "video_id",
        "label",
        "fighting_prob",
        "entropy",
        "hard_positive",
        "hard_negative_dynamic",
        "sample_weight",
    ]
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, columns=output_columns)

    print(f"[hard_mining] output={output_path}")
    print(f"[hard_mining] summary={summary}")


if __name__ == "__main__":
    main()
