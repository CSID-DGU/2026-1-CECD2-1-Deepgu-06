"""
평가 신뢰성 검증 (3 항목)
  1. precision 정의 불일치 재현 (tp/(tp+fp) vs tp_pred/n_preds)
  2. fps 정합성 점검 (GT json vs CSV vs cv2 실제 디코딩)
  3. 길이 편향 진단 (멀티-GT pred, 평균 pred 길이, tp vs tp_pred)

재추론 없이 캐시된 clip score(service_scope_clip_scores.json) 사용.
"""
import json
import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.event_builder import build_events

CLIP_SCORES_JSON = PROJECT_ROOT / "outputs/eval/service_scope_clip_scores.json"
CLIPS_CSV = PROJECT_ROOT / "data/manifests/cctv_x3d_s_overlap030/testing_clips_service_scope.csv"
SERVICE_SCOPE_JSON = PROJECT_ROOT / "data/manifests/test_service_scope.json"
GT_JSON = Path("/home/deepgu/test/cctv/dataset/ground-truth.json")

IOU_THRESHOLD = 0.10
START, END = 0.47, 0.42

BASE_THRESHOLDS = {
    "min_event_duration_sec": 0.5,
    "mean_score_threshold": 0.0,
    "split": {"enabled": True, "score_threshold": 0.33, "min_consecutive_clips": 2},
    "score_smoothing": {"enabled": True, "window_size": 3, "method": "moving_average"},
    "start_score": START, "end_score": END,
}


def interval_iou(a_start, a_end, b_start, b_end):
    left = max(a_start, b_start)
    right = min(a_end, b_end)
    intersection = max(0, right - left + 1)
    union = (a_end - a_start + 1) + (b_end - b_start + 1) - intersection
    return intersection / union if union > 0 else 0.0


def seconds_to_frame_range(start_sec, end_sec, fps, nb_frames):
    start_frame = max(0, int(math.floor(start_sec * fps)))
    end_frame = min(int(nb_frames) - 1, int(math.ceil(end_sec * fps)) - 1)
    return start_frame, max(start_frame, end_frame)


# ── 데이터 로드 ──
service_ids = sorted(json.loads(SERVICE_SCOPE_JSON.read_text())["video_ids"])
score_map = {r["clip_id"]: r["fighting_prob"]
             for r in json.loads(CLIP_SCORES_JSON.read_text())["rows"]}
df = pd.read_csv(CLIPS_CSV)
df = df[df["video_id"].isin(service_ids)].copy()
video_meta_csv = df.groupby("video_id").first()[["fps", "nb_frames"]].to_dict("index")

video_clips = {}
for vid, grp in df.sort_values(["video_id", "start_frame"]).groupby("video_id"):
    clips = []
    for idx, (_, row) in enumerate(grp.iterrows()):
        clips.append({
            "clip_id": idx,
            "start_frame": int(row["start_frame"]),
            "end_frame": int(row["end_frame"]),
            "final_score": score_map.get(row["clip_id"], 0.0),
        })
    video_clips[vid] = clips

gt_db = json.loads(GT_JSON.read_text())["database"]


def build_gt(vid, fps, nb_frames):
    gts = []
    for ann in gt_db.get(vid, {}).get("annotations", []):
        s, e = ann["segment"]
        sf, ef = seconds_to_frame_range(s, e, fps, nb_frames)
        gts.append({"start_frame": sf, "end_frame": ef, "start_time": s, "end_time": e,
                    "duration_sec": e - s})
    return gts


# ════════════════════════════════════════════════════════
# POINT 2: fps 정합성
# ════════════════════════════════════════════════════════
print("=" * 70)
print("POINT 2: fps 정합성 (GT json vs CSV vs cv2 실제 디코딩)")
print("=" * 70)

# (a) GT json vs CSV: 전체 61개 비교
mismatch_fps, mismatch_nbf = [], []
for vid in service_ids:
    gt_fps = float(gt_db[vid]["frame_rate"])
    gt_nbf = int(gt_db[vid]["nb_frames"])
    csv_fps = float(video_meta_csv[vid]["fps"])
    csv_nbf = int(video_meta_csv[vid]["nb_frames"])
    if abs(gt_fps - csv_fps) > 0.01:
        mismatch_fps.append((vid, gt_fps, csv_fps))
    if gt_nbf != csv_nbf:
        mismatch_nbf.append((vid, gt_nbf, csv_nbf))

print(f"\n(a) GT json vs CSV  [전체 {len(service_ids)}개]")
print(f"    fps 불일치(>0.01): {len(mismatch_fps)}개")
for vid, g, c in mismatch_fps[:10]:
    print(f"      {vid}: GT={g:.4f}  CSV={c:.4f}")
print(f"    nb_frames 불일치: {len(mismatch_nbf)}개")
for vid, g, c in mismatch_nbf[:10]:
    print(f"      {vid}: GT={g}  CSV={c}  (diff={g-c})")

# (b) cv2 실제 디코딩 비교 (샘플 8개)
print(f"\n(b) cv2 실제 디코딩 vs GT json  [샘플]")
try:
    import cv2
    sample = service_ids[:8]
    print(f"    {'video':>12} {'GTfps':>8} {'cv2fps':>8} {'GTnbf':>7} {'cv2cnt':>7} {'GTprop':>7} {'shift_s':>8}")
    cv2_rows = []
    for vid in sample:
        path = gt_db[vid]
        src = "CCTV_DATA" if gt_db[vid].get("source") == "CCTV" else "NON_CCTV_DATA"
        vpath = f"/home/deepgu/test/cctv/dataset/{src}/testing/{vid}.mpeg"
        cap = cv2.VideoCapture(vpath)
        cv2_fps = cap.get(cv2.CAP_PROP_FPS)
        cv2_propcount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cnt = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            cnt += 1
        cap.release()
        gt_fps = float(gt_db[vid]["frame_rate"])
        gt_nbf = int(gt_db[vid]["nb_frames"])
        # 동일 GT 시각이 두 fps에서 몇 초 어긋나는지: t초 지점 frame을 cv2fps로 다시 초환산
        # frame = floor(t*gt_fps); 그 frame을 cv2_fps로 환산한 시각 차이
        t = 60.0
        shift = t * (cv2_fps / gt_fps) - t if gt_fps > 0 else 0.0
        print(f"    {vid:>12} {gt_fps:>8.3f} {cv2_fps:>8.3f} {gt_nbf:>7} {cnt:>7} {cv2_propcount:>7} {shift:>8.3f}")
        cv2_rows.append({"vid": vid, "gt_fps": gt_fps, "cv2_fps": cv2_fps,
                         "gt_nbf": gt_nbf, "cv2_count": cnt, "shift_60s": shift})
except Exception as exc:
    print(f"    cv2 디코딩 스킵: {exc}")
    cv2_rows = []


# ════════════════════════════════════════════════════════
# POINT 1 & 3: 이벤트 재구성 후 매칭 통계
# ════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"POINT 1 & 3: 이벤트 매칭 통계 (fast-only, start={START} end={END}, IoU={IOU_THRESHOLD})")
print("=" * 70)

total_tp = total_tp_pred = total_fn = total_n_preds = 0
pred_durations = []
multi_gt_preds = []      # 한 pred가 2개 이상 GT를 매칭한 사례
gt_total = 0

for vid in service_ids:
    fps = float(video_meta_csv[vid]["fps"])
    nbf = int(video_meta_csv[vid]["nb_frames"])
    gts = build_gt(vid, fps, nbf)
    gt_total += len(gts)
    preds, _ = build_events(video_clips[vid], BASE_THRESHOLDS, fps)
    total_n_preds += len(preds)

    # GT-side
    for gt in gts:
        if any(interval_iou(p["start_frame"], p["end_frame"],
                            gt["start_frame"], gt["end_frame"]) >= IOU_THRESHOLD for p in preds):
            total_tp += 1
        else:
            total_fn += 1
    # Pred-side
    for p in preds:
        dur = (p["end_frame"] - p["start_frame"] + 1) / fps
        pred_durations.append(dur)
        matched_gts = [gi for gi, gt in enumerate(gts)
                       if interval_iou(p["start_frame"], p["end_frame"],
                                       gt["start_frame"], gt["end_frame"]) >= IOU_THRESHOLD]
        if matched_gts:
            total_tp_pred += 1
        if len(matched_gts) >= 2:
            multi_gt_preds.append((vid, round(dur, 1), len(matched_gts)))

total_fp = total_n_preds - total_tp_pred

# ── POINT 1: 두 가지 precision 정의 ──
recall = total_tp / max(1, total_tp + total_fn)
prec_A = total_tp / max(1, total_tp + total_fp)          # summarize 방식 (GT측 분자)
prec_B = total_tp_pred / max(1, total_n_preds)            # pred측 분자 (엄밀)
f1_A = 2 * prec_A * recall / max(prec_A + recall, 1e-12)
f1_B = 2 * prec_B * recall / max(prec_B + recall, 1e-12)

print(f"\n[원자료]")
print(f"  GT 총수            = {gt_total}")
print(f"  pred 총수 (n_preds) = {total_n_preds}")
print(f"  TP (GT측 매칭 GT수) = {total_tp}")
print(f"  tp_pred (매칭된 pred수) = {total_tp_pred}")
print(f"  FN (미매칭 GT)      = {total_fn}")
print(f"  FP (미매칭 pred)    = {total_fp}")
print(f"\n[POINT 1: precision 정의별 F1]")
print(f"  recall (공통)              = {recall:.4f}  (= {total_tp}/{total_tp+total_fn})")
print(f"  (A) summarize: P=tp/(tp+fp) = {prec_A:.4f}  (= {total_tp}/{total_tp+total_fp})  → F1={f1_A:.4f}")
print(f"  (B) 엄밀:  P=tp_pred/n_preds = {prec_B:.4f}  (= {total_tp_pred}/{total_n_preds})  → F1={f1_B:.4f}")
print(f"  >>> 동일 예측인데 F1 차이 = {abs(f1_A - f1_B):.4f}")

# ── POINT 3: 길이 편향 ──
import statistics
pred_durations.sort()
n = len(pred_durations)


def pct(p):
    if n == 0:
        return 0.0
    return pred_durations[min(n - 1, int(p * n))]


print(f"\n[POINT 3: 길이 편향]")
print(f"  pred 길이(초)  p10={pct(.1):.1f}  p50={pct(.5):.1f}  p90={pct(.9):.1f}  "
      f"mean={statistics.mean(pred_durations):.1f}  max={max(pred_durations):.1f}")
print(f"  멀티-GT pred (한 pred가 2+ GT 매칭): {len(multi_gt_preds)}개")
extra = sum(c - 1 for _, _, c in multi_gt_preds)
print(f"  → 이로 인해 부풀려진 TP(분자) = +{extra}  "
      f"(tp_pred {total_tp_pred} → TP {total_tp}, 차이 {total_tp - total_tp_pred})")
print(f"  멀티-GT pred 사례 (video, 길이s, 매칭GT수):")
for vid, dur, c in sorted(multi_gt_preds, key=lambda x: -x[2])[:12]:
    print(f"      {vid}: {dur}s → {c}개 GT")
