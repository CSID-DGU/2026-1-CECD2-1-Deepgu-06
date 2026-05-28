"""
Val split high-score negative 클립 시각화.
Type 분류(A/B/C/D) 판단을 위해 영상별 고득점 negative 클립 대표 프레임 저장.
"""
import json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VAL_JSON     = PROJECT_ROOT / "outputs/train/overlap030_focal/val_scores.json"
VAL_CSV      = PROJECT_ROOT / "data/manifests/cctv_x3d_s_overlap030/validation_clips.csv"
OUTPUT_DIR   = PROJECT_ROOT / "outputs/viz/val_high_score_neg"

HIGH_THR        = 0.60
TOP_N_VIDEOS    = 12
CLIPS_PER_VIDEO = 10
THUMB_W, THUMB_H = 224, 224


def extract_middle_frame(video_path, start_frame, end_frame):
    cap = cv2.VideoCapture(str(video_path))
    mid = (start_frame + end_frame) // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
    return cv2.resize(frame, (THUMB_W, THUMB_H))


def make_grid(frames, scores, times, cols=5):
    rows = (len(frames) + cols - 1) // cols
    cell_h = THUMB_H + 30
    cell_w = THUMB_W + 4
    canvas = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8) + 30

    for idx, (frame, score, t) in enumerate(zip(frames, scores, times)):
        r, c = divmod(idx, cols)
        y, x = r * cell_h, c * cell_w
        canvas[y:y + THUMB_H, x:x + THUMB_W] = frame
        color = (0, 0, 220) if score >= 0.80 else (0, 140, 255)
        cv2.rectangle(canvas, (x, y), (x + THUMB_W - 1, y + THUMB_H - 1), color, 2)
        label = f"t={t:.0f}s  {score:.3f}"
        cv2.putText(canvas, label, (x + 3, y + THUMB_H + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return canvas


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(VAL_JSON) as f:
        score_data = json.load(f)

    df_manifest = pd.read_csv(VAL_CSV)
    manifest_map = {row["clip_id"]: row for _, row in df_manifest.iterrows()}

    vid_clips = defaultdict(list)
    for r in score_data["rows"]:
        if r["label"] == 0 and r["fighting_prob"] >= HIGH_THR:
            cid = r["clip_id"]
            if cid in manifest_map:
                meta = manifest_map[cid]
                vid_clips[r["video_id"]].append({
                    "clip_id": cid,
                    "score": r["fighting_prob"],
                    "start_frame": int(meta["start_frame"]),
                    "end_frame": int(meta["end_frame"]),
                    "start_time": float(meta["start_time"]),
                    "video_path": meta["video_path"],
                })

    sorted_vids = sorted(vid_clips.keys(), key=lambda v: len(vid_clips[v]), reverse=True)
    top_vids = sorted_vids[:TOP_N_VIDEOS]

    print(f"시각화 대상: {len(top_vids)}개 영상 (high_neg >= {HIGH_THR})")
    print(f"{'영상':<20} {'high_neg':>10} {'neg_total':>10}")
    for vid in top_vids:
        total_neg = sum(1 for r in score_data["rows"] if r["video_id"] == vid and r["label"] == 0)
        print(f"  {vid:<18} {len(vid_clips[vid]):>10} {total_neg:>10}")

    print()
    for vid in top_vids:
        clips = sorted(vid_clips[vid], key=lambda c: c["score"], reverse=True)
        selected = clips[:CLIPS_PER_VIDEO]

        print(f"  [{vid}] {len(selected)}개 클립 추출 중...")

        frames, scores, times = [], [], []
        for clip in selected:
            frame = extract_middle_frame(clip["video_path"], clip["start_frame"], clip["end_frame"])
            frames.append(frame)
            scores.append(clip["score"])
            times.append(clip["start_time"])

        grid = make_grid(frames, scores, times, cols=5)

        header = np.zeros((40, grid.shape[1], 3), dtype=np.uint8) + 20
        title = f"{vid}  |  high_neg(>={HIGH_THR}): {len(vid_clips[vid])}  |  top-{len(selected)} by score"
        cv2.putText(header, title, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 100), 1)
        full = np.vstack([header, grid])

        out_path = OUTPUT_DIR / f"{vid}_val_high_neg.jpg"
        cv2.imwrite(str(out_path), full)
        print(f"    saved → {out_path}")

    print(f"\n완료: {OUTPUT_DIR}")
    print("범례: 빨간 테두리=score>=0.80, 주황=0.60~0.80")


if __name__ == "__main__":
    main()
