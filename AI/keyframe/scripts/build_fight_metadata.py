import argparse
import json
import sys
from pathlib import Path


ROOT = Path("/home/deepgu/test")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.paths import ensure_dir


TEST_ROOT = ROOT
CCTV_GT_PATH = TEST_ROOT / "cctv/dataset/ground-truth.json"
CCTV_VIDEO_ROOT = TEST_ROOT / "cctv/dataset/CCTV_DATA"
UCF_ROOT = Path("/home/deepgu/VERA/Data")
UCF_EVAL_PATH = UCF_ROOT / "UCF_Eval.json"
UCF_VIDEO_ROOT = UCF_ROOT / "videos"
VIOLENCE_CLASSES = {"Abuse", "Assault", "Fighting"}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path, records):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _event_record(label, start_sec, end_sec):
    return {
        "label": label,
        "start_sec": float(start_sec),
        "end_sec": float(end_sec),
    }


def build_cctv_master(include_sources=None):
    raw = load_json(CCTV_GT_PATH)["database"]
    include_sources = set(include_sources or ["CCTV"])
    records = []

    for video_id, meta in raw.items():
        if meta["source"] not in include_sources:
            continue

        video_path = CCTV_VIDEO_ROOT / meta["subset"] / f"{video_id}.mpeg"
        if not video_path.exists():
            continue

        records.append({
            "video_id": video_id,
            "video_path": str(video_path),
            "dataset": "cctv_fights",
            "source": meta["source"],
            "subset": meta["subset"],
            "fps": float(meta["frame_rate"]),
            "num_frames": int(meta["nb_frames"]),
            "duration": float(meta["duration"]),
            "events": [
                _event_record("fight", ann["segment"][0], ann["segment"][1])
                for ann in meta.get("annotations", [])
                if str(ann.get("label", "")).lower() == "fight"
            ],
        })

    return records


def _resolve_ucf_video_path(video_name, category):
    direct = UCF_VIDEO_ROOT / category / video_name
    if direct.exists():
        return direct

    normal_candidates = [
        UCF_VIDEO_ROOT / "Testing_Normal_Videos_Anomaly" / video_name,
        UCF_VIDEO_ROOT / "Normal_Videos_event" / video_name,
        UCF_VIDEO_ROOT / "Training-Normal-Videos-Part-1" / video_name,
        UCF_VIDEO_ROOT / "Training-Normal-Videos-Part-2" / video_name,
    ]
    for candidate in normal_candidates:
        if candidate.exists():
            return candidate

    return None


def _temporal_labels_to_events(temporal_label, fps):
    events = []
    for start_frame, end_frame in zip(temporal_label[::2], temporal_label[1::2]):
        if start_frame < 0 or end_frame < 0:
            continue
        events.append(_event_record("fight", start_frame / fps, end_frame / fps))
    return events


def build_ucf_master(include_normal=True):
    raw = load_json(UCF_EVAL_PATH)
    records = []

    for item in raw:
        video_name = Path(item["video"]).name
        category = video_name.split("_")[0]

        is_violence = category in VIOLENCE_CLASSES
        is_normal = video_name.startswith("Normal_Videos_")
        if not is_violence and not (include_normal and is_normal):
            continue

        video_path = _resolve_ucf_video_path(video_name, category)
        if video_path is None:
            continue

        fps = 30.0
        label = "fight" if is_violence else "normal"
        events = _temporal_labels_to_events(item["temporal_label"], fps) if is_violence else []

        records.append({
            "video_id": video_name.rsplit(".", 1)[0],
            "video_path": str(video_path),
            "dataset": "ucf_crime_eval",
            "source": "UCF-Crime",
            "subset": "eval",
            "category": category,
            "fps": fps,
            "num_frames": int(item["length"]),
            "duration": float(item["length"]) / fps,
            "events": events,
            "label": label,
        })

    return records


def build_master_metadata(include_ucf_normal=True):
    records = []
    records.extend(build_cctv_master(include_sources=["CCTV"]))
    records.extend(build_ucf_master(include_normal=include_ucf_normal))
    return records


def _segment_overlap(start_a, end_a, start_b, end_b):
    left = max(start_a, start_b)
    right = min(end_a, end_b)
    return max(0.0, right - left)


def build_clip_metadata(master_records, clip_len=16, stride=8, overlap_ratio=0.3):
    clips = []

    for record in master_records:
        num_frames = int(record["num_frames"])
        fps = float(record["fps"])
        if num_frames < clip_len:
            continue

        for clip_start in range(0, num_frames - clip_len + 1, stride):
            clip_end = clip_start + clip_len
            clip_start_sec = clip_start / fps
            clip_end_sec = clip_end / fps
            clip_duration_sec = clip_end_sec - clip_start_sec

            max_overlap_ratio = 0.0
            for event in record.get("events", []):
                overlap_sec = _segment_overlap(
                    clip_start_sec,
                    clip_end_sec,
                    float(event["start_sec"]),
                    float(event["end_sec"]),
                )
                max_overlap_ratio = max(
                    max_overlap_ratio,
                    overlap_sec / clip_duration_sec if clip_duration_sec > 0 else 0.0
                )

            label = 1 if max_overlap_ratio >= overlap_ratio else 0
            clips.append({
                "video_id": record["video_id"],
                "video_path": record["video_path"],
                "dataset": record["dataset"],
                "source": record["source"],
                "subset": record["subset"],
                "fps": fps,
                "clip_start_frame": clip_start,
                "clip_end_frame": clip_end - 1,
                "clip_start_sec": clip_start_sec,
                "clip_end_sec": clip_end_sec,
                "label": label,
                "overlap_ratio": max_overlap_ratio,
            })

    return clips


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--master-out",
        default="/home/deepgu/test/data/metadata/fight_master_metadata.jsonl",
    )
    parser.add_argument(
        "--clips-out",
        default="/home/deepgu/test/data/metadata/fight_clip_metadata.jsonl",
    )
    parser.add_argument("--clip-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--overlap-ratio", type=float, default=0.3)
    parser.add_argument(
        "--exclude-ucf-normal",
        action="store_true",
        help="Generate positives only from CCTV/UCF violence without UCF normal negatives.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    master_records = build_master_metadata(
        include_ucf_normal=not args.exclude_ucf_normal
    )
    clip_records = build_clip_metadata(
        master_records,
        clip_len=args.clip_len,
        stride=args.stride,
        overlap_ratio=args.overlap_ratio,
    )

    write_jsonl(args.master_out, master_records)
    write_jsonl(args.clips_out, clip_records)

    print(f"master metadata: {len(master_records)} videos -> {args.master_out}")
    print(f"clip metadata: {len(clip_records)} clips -> {args.clips_out}")


if __name__ == "__main__":
    main()
