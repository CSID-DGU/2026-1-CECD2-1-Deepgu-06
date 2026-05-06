import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.io import ensure_dir, load_json


def build_rows(video_entries, clip_length, stride, split):
    rows = []
    for entry in video_entries:
        video_id = entry["video_id"]
        video_path = entry["video_path"]
        label = int(entry["label"])
        source = entry.get("source", "unknown")
        hard_negative = int(entry.get("hard_negative", 0))
        intervals = entry["intervals"]
        for interval_index, interval in enumerate(intervals):
            start_frame = int(interval["start_frame"])
            end_frame = int(interval["end_frame"])
            clip_index = 0
            for clip_start in range(start_frame, max(start_frame, end_frame - clip_length + 2), stride):
                clip_end = clip_start + clip_length - 1
                if clip_end > end_frame:
                    break
                rows.append(
                    {
                        "clip_id": f"{video_id}_{interval_index:02d}_{clip_index:04d}",
                        "video_id": video_id,
                        "video_path": video_path,
                        "start_frame": clip_start,
                        "end_frame": clip_end,
                        "label": label,
                        "split": split,
                        "source": source,
                        "hard_negative": hard_negative,
                    }
                )
                clip_index += 1
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True, help="Video interval manifest JSON.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    payload = load_json(args.input_json)
    rows = build_rows(payload, args.clip_length, args.stride, args.split)
    table = pd.DataFrame(rows)
    output_path = Path(args.output_csv)
    ensure_dir(output_path.parent)
    table.to_csv(output_path, index=False)
    print(f"[manifest] rows={len(table)} output={output_path}")


if __name__ == "__main__":
    main()
