"""
Phase 1 vs Phase 2 처리 속도 벤치마크.

클립당 keyframe 선택 latency와 throughput을 비교합니다.

사용법:
  python scripts/bench_speed.py \
      --label_path outputs/training_data_x3d/test.json \
      --model_path outputs/frame_selector_x3d.pth \
      --n_clips 1000
"""

import os
import sys
import json
import time
import random
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))        # keyframe/

from pipeline.sampler import KeyframeSampler


def bench(sampler, records, label):
    times = []
    for rec in records:
        features = np.load(rec["features_path"]).astype(np.float32)
        T = len(features)
        candidate = {"clip": [None] * T, "features": features}

        t0 = time.perf_counter()
        sampler.sample(candidate)
        times.append(time.perf_counter() - t0)

    times = np.array(times) * 1000  # ms
    print(f"\n[{label}]")
    print(f"  클립 수        : {len(times)}")
    print(f"  평균 latency   : {times.mean():.2f} ms/clip")
    print(f"  중앙값 latency : {np.median(times):.2f} ms/clip")
    print(f"  P95 latency    : {np.percentile(times, 95):.2f} ms/clip")
    print(f"  Throughput     : {1000/times.mean():.1f} clips/sec")
    return times


def main(args):
    with open(args.label_path) as f:
        all_records = json.load(f)

    random.seed(42)
    records = random.sample(all_records, min(args.n_clips, len(all_records)))
    print(f"벤치마크 클립: {len(records)}개  (device={args.device})")

    p1 = KeyframeSampler(n_frames=args.n_frames, device=args.device)
    p2 = KeyframeSampler(n_frames=args.n_frames, model_path=args.model_path, device=args.device)

    # warmup
    for rec in records[:10]:
        feat = np.load(rec["features_path"]).astype(np.float32)
        for s in [p1, p2]:
            s.sample({"clip": [None]*len(feat), "features": feat})

    t1 = bench(p1, records, "Phase 1 (K-means + Motion)")
    t2 = bench(p2, records, "Phase 2 (BiGRU FrameScorer)")

    overhead_ms = t2.mean() - t1.mean()
    print(f"\n[오버헤드] Phase 2 - Phase 1 = {overhead_ms:+.2f} ms/clip")

    os.makedirs("outputs", exist_ok=True)
    result = {
        "n_clips": len(records),
        "n_frames": args.n_frames,
        "device": args.device,
        "phase1": {
            "mean_ms": round(float(t1.mean()), 3),
            "median_ms": round(float(np.median(t1)), 3),
            "p95_ms": round(float(np.percentile(t1, 95)), 3),
            "throughput_clips_per_sec": round(1000/t1.mean(), 1),
        },
        "phase2": {
            "mean_ms": round(float(t2.mean()), 3),
            "median_ms": round(float(np.median(t2)), 3),
            "p95_ms": round(float(np.percentile(t2, 95)), 3),
            "throughput_clips_per_sec": round(1000/t2.mean(), 1),
        },
        "overhead_ms": round(overhead_ms, 3),
    }
    out = "outputs/bench_speed_result.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n결과 저장: {out}")


if __name__ == "__main__":
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--n_clips",  type=int, default=1000)
    parser.add_argument("--n_frames", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(parser.parse_args())
