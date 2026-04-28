"""
Temporal Annotation 기반 Anomaly Region Coverage 평가.

UCF-Crime Temporal Annotations를 이용해서, 선택기가 실제 이상행동 구간의
프레임을 얼마나 잡아내는지 측정합니다.

지표:
  Anomaly Region Coverage = (선택된 프레임 중 실제 이상행동 구간에 속하는 수) / 8
  → 높을수록 선택기가 이상행동 장면에 집중함

사용법:
  python scripts/eval_temporal.py \
      --annotation_path Data/annotations/temporal_anomaly.txt \
      --video_dir       videos \
      --pglsum_model_path ../../PGL-SUM/Summaries/UCF/models/split0/best_model.pth \
      --data_dirs       outputs/training_data/train.json outputs/training_data/test.json
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.sampler import KeyframeSampler


# ------------------------------------------------------------------
# Annotation 파싱
# ------------------------------------------------------------------

def load_annotations(path):
    """
    반환: { video_stem: [(start1,end1), (start2,end2)] }
    형식: VideoName.mp4  Class  start1  end1  start2  end2
    -1은 두 번째 구간 없음을 의미
    """
    ann = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            stem = parts[0].replace(".mp4", "")
            s1, e1 = int(parts[2]), int(parts[3])
            segments = [(s1, e1)]
            if len(parts) >= 6 and parts[4] != "-1":
                s2, e2 = int(parts[4]), int(parts[5])
                segments.append((s2, e2))
            ann[stem] = segments
    return ann


# ------------------------------------------------------------------
# 클립 → 프레임 범위
# ------------------------------------------------------------------

CLIP_LEN = 16
STRIDE   = 8

def clip_frame_range(clip_idx):
    start = clip_idx * STRIDE
    return start, start + CLIP_LEN


def anomaly_overlap(clip_start, clip_end, segments):
    """클립이 anomaly 구간과 겹치는 프레임 집합 반환."""
    anomaly_frames = set()
    for s, e in segments:
        for f in range(max(clip_start, s), min(clip_end, e) + 1):
            anomaly_frames.add(f)
    return anomaly_frames


# ------------------------------------------------------------------
# 비디오 인덱스 빌드
# ------------------------------------------------------------------

def build_video_index(video_dir):
    """stem → 절대 경로 딕셔너리."""
    index = {}
    for root, _, files in os.walk(video_dir):
        for fname in files:
            if fname.lower().endswith((".mp4", ".avi", ".mpeg", ".mkv")):
                stem = os.path.splitext(fname)[0]
                index[stem] = os.path.join(root, fname)
    return index


# ------------------------------------------------------------------
# 메인
# ------------------------------------------------------------------

def main(args):
    ann = load_annotations(args.annotation_path)
    # 우리 3개 클래스만
    target_classes = {"Abuse", "Assault", "Fighting"}
    ann = {k: v for k, v in ann.items()
           if any(k.startswith(c) for c in target_classes)}

    print(f"Temporal annotation 영상: {len(ann)}개")
    for stem, segs in ann.items():
        print(f"  {stem}: {segs}")

    # 클립 레코드 로드 (train + test 모두)
    all_records = []
    for path in args.data_dirs:
        if os.path.isfile(path):
            with open(path) as f:
                all_records.extend(json.load(f))

    # annotation 영상에 속하는 클립만 필터링
    target_stems = set(ann.keys())
    target_records = []
    for rec in all_records:
        stem = rec["clip_id"].rsplit("_clip", 1)[0]
        if stem in target_stems:
            target_records.append((stem, rec))

    print(f"\n대상 클립: {len(target_records)}개 "
          f"({len(target_stems)}개 영상에서)\n")

    # 선택기 초기화
    p1_sampler = KeyframeSampler(n_frames=8)

    pgl_sampler = None
    if args.pglsum_model_path and os.path.isfile(args.pglsum_model_path):
        from pipeline.sampler import PGLSumSampler
        pgl_sampler = PGLSumSampler(
            model_path=args.pglsum_model_path,
            n_frames=8,
            input_size=args.pglsum_input_size,
            device=args.device,
        )

    p2_sampler = None
    if args.model_path and os.path.isfile(args.model_path):
        p2_sampler = KeyframeSampler(
            n_frames=8,
            model_path=args.model_path,
            device=args.device,
        )

    # 결과 누적
    # { method: { video_stem: [coverage_per_overlapping_clip] } }
    methods = ["p1"]
    if p2_sampler:  methods.append("p2")
    if pgl_sampler: methods.append("pgl")

    results = {m: {} for m in methods}
    for stem in target_stems:
        for m in methods:
            results[m][stem] = []

    # 클립별 평가
    for stem, rec in target_records:
        clip_idx  = int(rec["clip_id"].rsplit("_clip", 1)[1])
        clip_s, clip_e = clip_frame_range(clip_idx)
        segments  = ann[stem]

        anomaly_frames = anomaly_overlap(clip_s, clip_e, segments)
        if not anomaly_frames:
            continue   # 이 클립은 이상행동 구간과 겹치지 않음 → skip

        features  = np.load(rec["features_path"]).astype(np.float32)
        dummy_clip = [None] * len(features)

        def coverage(selected_indices):
            # 선택 인덱스 → 절대 프레임 번호
            abs_frames = {clip_s + i for i in selected_indices}
            hit = abs_frames & anomaly_frames
            return len(hit) / len(selected_indices)

        c1 = {"clip": dummy_clip, "features": features}
        p1_sampler.sample(c1)
        results["p1"][stem].append(coverage(c1["selected_indices"]))

        if p2_sampler:
            c2 = {"clip": dummy_clip, "features": features}
            p2_sampler.sample(c2)
            results["p2"][stem].append(coverage(c2["selected_indices"]))

        if pgl_sampler:
            c3 = {"clip": dummy_clip, "features": features}
            pgl_sampler.sample(c3)
            results["pgl"][stem].append(coverage(c3["selected_indices"]))

    # ------------------------------------------------------------------
    # 결과 출력
    # ------------------------------------------------------------------
    print("=" * 70)
    print(f"{'영상':<30} {'클립수':>6}", end="")
    for m in methods:
        label = {"p1":"Phase1","p2":"Phase2","pgl":"PGL-SUM"}[m]
        print(f"  {label:>10}", end="")
    print()
    print("-" * 70)

    agg = {m: [] for m in methods}
    for stem in sorted(target_stems):
        n = len(results[methods[0]][stem])
        print(f"{stem:<30} {n:>6}", end="")
        for m in methods:
            vals = results[m][stem]
            avg  = np.mean(vals) * 100 if vals else float("nan")
            agg[m].extend(vals)
            print(f"  {avg:>9.1f}%", end="")
        print()

    print("-" * 70)
    print(f"{'전체 평균':<30} {sum(len(results[methods[0]][s]) for s in target_stems):>6}", end="")
    for m in methods:
        avg = np.mean(agg[m]) * 100 if agg[m] else float("nan")
        print(f"  {avg:>9.1f}%", end="")
    print()
    print("=" * 70)
    print("\n* Coverage = (선택된 8프레임 중 실제 이상행동 구간에 속하는 비율)")
    print("  이상행동 구간과 겹치는 클립에서만 측정 (비겹침 클립 제외)")

    # 저장
    out = {
        "per_video": {
            stem: {m: float(np.mean(results[m][stem])) if results[m][stem] else None
                   for m in methods}
            for stem in target_stems
        },
        "aggregate": {m: float(np.mean(agg[m])) if agg[m] else None for m in methods},
    }
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/eval_temporal_result.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n결과 저장: outputs/eval_temporal_result.json")


if __name__ == "__main__":
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_path", default="Data/annotations/temporal_anomaly.txt")
    parser.add_argument("--video_dir",       default="videos")
    parser.add_argument("--data_dirs",       nargs="+",
                        default=["outputs/training_data/train.json",
                                 "outputs/training_data/test.json"])
    parser.add_argument("--pglsum_model_path", default=None)
    parser.add_argument("--pglsum_input_size", type=int, default=2048)
    parser.add_argument("--model_path",        default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args)
