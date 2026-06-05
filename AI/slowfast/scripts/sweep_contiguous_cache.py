"""
연속 클립 캐시 기반 operating point sweep.
- 실제 pipeline.event_builder.build_events 사용 (패치된 frame-gap-aware 버전).
- 평가: scripts.evaluate_event_batch.evaluate_video / summarize 그대로.
- 먼저 0.48/0.36/split0.40에서 라이브 harness 재현 검증.
캐시가 라이브의 충실한 대리자임이 확인되면 sweep 결과 신뢰 가능.
"""
import glob
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.event_builder import build_events
from scripts.evaluate_event_batch import build_gt_events, evaluate_video, summarize

CACHE_DIR = PROJECT_ROOT / "outputs/cache/test_clip_scores_service61"
GT_JSON = Path("/home/deepgu/test/cctv/dataset/ground-truth.json")
IOU = 0.10

BASE_TH = {
    "min_event_duration_sec": 0.5,
    "mean_score_threshold": 0.0,
    "score_smoothing": {"enabled": True, "window_size": 3, "method": "moving_average"},
    "split": {"enabled": True, "score_threshold": 0.40, "min_consecutive_clips": 2},
}

# 캐시 로드
caches = {}
for f in glob.glob(str(CACHE_DIR / "*.json")):
    d = json.load(open(f))
    caches[d["video_id"]] = d
gtdb = json.load(open(GT_JSON))["database"]
vids = sorted(caches)


def clips_of(vid):
    return [
        {"clip_id": int(c["clip_id"]), "start_frame": int(c["start_frame"]),
         "end_frame": int(c["end_frame"]), "final_score": float(c["score"])}
        for c in caches[vid]["clips"]
    ]


def run(start, end, split_thr):
    th = dict(BASE_TH)
    th = {**BASE_TH, "start_score": start, "end_score": end,
          "split": {**BASE_TH["split"], "score_threshold": split_thr}}
    per_video = []
    for vid in vids:
        fps = float(caches[vid]["fps"])
        events, _ = build_events(clips_of(vid), th, fps)
        gt_events = build_gt_events(gtdb[vid])
        m = evaluate_video(events, gt_events, iou_threshold=IOU, eval_mode="gt-centric")
        m["video_id"] = vid
        per_video.append(m)
    return summarize(per_video)


# ── 1. 라이브 재현 검증 ──
chk = run(0.48, 0.36, 0.40)
print("=" * 70)
print("재현 검증 @ 0.48/0.36/split0.40 (연속 캐시 sweep vs 라이브 harness)")
print("=" * 70)
print(f"  CACHE sweep : TP={chk['true_positive']} FP={chk['false_positive']} FN={chk['false_negative']} "
      f"n_pred={chk['predicted_event_count']} F1={chk['event_f1']:.4f}")
print(f"  LIVE harness: TP=109 FP=132 FN=73 n_pred=245 F1=0.5211")
match = (chk['true_positive'] == 109 and chk['false_positive'] == 132 and chk['predicted_event_count'] == 245)
print(f"  => {' 일치 — 캐시 신뢰 가능' if match else ' 불일치 — sweep 신뢰 불가, 원인 조사 필요'}")
print()

# ── 2. sweep ──
STARTS = [round(0.40 + 0.02 * i, 2) for i in range(11)]   # 0.40..0.60
ENDS = [round(0.28 + 0.02 * i, 2) for i in range(13)]      # 0.28..0.52
SPLITS = [0.33, 0.36, 0.40, 0.44, 0.48]

results = []
for s in STARTS:
    for e in ENDS:
        if e > s:
            continue
        for sp in SPLITS:
            r = run(s, e, sp)
            results.append({"start": s, "end": e, "split": sp,
                            "tp": r["true_positive"], "fp": r["false_positive"],
                            "fn": r["false_negative"], "n_pred": r["predicted_event_count"],
                            "P": r["event_precision"], "R": r["event_recall"], "F1": r["event_f1"]})
results.sort(key=lambda x: -x["F1"])

print("=" * 70)
print(f"연속 캐시 sweep 결과 (61 service_scope, GT-centric, IoU=0.10) — 상위 15")
print("=" * 70)
print(f"{'rank':>4} {'start':>5} {'end':>5} {'split':>5} {'TP':>4} {'FP':>4} {'FN':>4} {'nPred':>5} {'P':>6} {'R':>6} {'F1':>6}")
for i, r in enumerate(results[:15]):
    print(f"{i+1:>4} {r['start']:>5.2f} {r['end']:>5.2f} {r['split']:>5.2f} {r['tp']:>4} {r['fp']:>4} "
          f"{r['fn']:>4} {r['n_pred']:>5} {r['P']:>6.3f} {r['R']:>6.3f} {r['F1']:>6.4f}")

b = results[0]
print(f"\nBEST: start={b['start']} end={b['end']} split={b['split']}  "
      f"F1={b['F1']:.4f} (TP={b['tp']} FP={b['fp']} FN={b['fn']} P={b['P']:.3f} R={b['R']:.3f})")

OUT = PROJECT_ROOT / "outputs/eval/sweep_contiguous_cache_61.json"
OUT.write_text(json.dumps({"validation_match": match, "check_0p48_0p36_0p40": chk,
                           "results": results}, ensure_ascii=False, indent=2))
print(f"[saved] {OUT}")
