import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.io import ensure_dir


POSITIVE_HEAVY_DEFAULT = [
    "fight_0023",
    "fight_0832",
    "fight_0382",
    "fight_0947",
    "fight_0020",
    "fight_0045",
]

MOSTLY_NEGATIVE_DEFAULT = [
    "fight_0085",
    "fight_0061",
    "fight_0959",
    "fight_0205",
    "fight_0001",
    "fight_0046",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        default="/home/deepgu/slowfast/data/manifests/cctv_x3d_s/validation_clips.csv",
    )
    parser.add_argument(
        "--output-csv",
        default="/home/deepgu/slowfast/data/manifests/cctv_x3d_s/validation_subset_clips.csv",
    )
    parser.add_argument("--positive-heavy", nargs="*", default=POSITIVE_HEAVY_DEFAULT)
    parser.add_argument("--mostly-negative", nargs="*", default=MOSTLY_NEGATIVE_DEFAULT)
    args = parser.parse_args()

    selected_videos = list(dict.fromkeys(args.positive_heavy + args.mostly_negative))
    table = pd.read_csv(args.input_csv)
    subset = table[table["video_id"].isin(selected_videos)].copy()
    output_path = Path(args.output_csv)
    ensure_dir(output_path.parent)
    subset.to_csv(output_path, index=False)

    counts = subset.groupby("video_id")["label"].agg(["size", "sum"]).reset_index()
    counts["pos_ratio"] = counts["sum"] / counts["size"]
    print(f"[subset] videos={len(selected_videos)} clips={len(subset)} output={output_path}")
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
