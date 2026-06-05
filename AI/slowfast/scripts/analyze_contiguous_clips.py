"""
연속 클립 캐시(test_clip_scores_service61) 상세 분석.
1) overlap 구간별(neg≤0.1 / ambiguous 0.1~0.3 / pos≥0.3) 점수 분포
2) clip-level 판별력 (PR-AUC / ROC-AUC, clean: pos≥0.3 vs neg≤0.1) → 옛 캐시와 비교
3) GT recall ceiling (각 GT의 최대 overlapping 클립 점수) → start_score 상한 정보
4) start_score 후보별 event 진입 가능 클립/구간
"""
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "outputs/cache/test_clip_scores_service61"
GT_JSON = Path("/home/deepgu/test/cctv/dataset/ground-truth.json")
OLD_CACHE = PROJECT_ROOT / "outputs/eval/service_scope_clip_scores.json"

caches = {}
for f in glob.glob(str(CACHE_DIR / "*.json")):
    d = json.load(open(f))
    caches[d["video_id"]] = d
gtdb = json.load(open(GT_JSON))["database"]
vids = sorted(caches)


def pct(a, ps=(10, 25, 50, 75, 90, 95)):
    a = np.asarray(a, dtype=float)
    return {f"p{p}": round(float(np.percentile(a, p)), 3) for p in ps}


def summ(scores):
    a = np.asarray(scores, dtype=float)
    if a.size == 0:
        return {"count": 0}
    return {"count": int(a.size), "mean": round(float(a.mean()), 3),
            "std": round(float(a.std()), 3), "min": round(float(a.min()), 3),
            "max": round(float(a.max()), 3), **pct(a)}


# ── 1) overlap 구간별 분포 ──
neg, amb, pos = [], [], []
all_clips = 0
for vid in vids:
    for c in caches[vid]["clips"]:
        all_clips += 1
        ov, s = float(c["gt_overlap"]), float(c["score"])
        if ov <= 0.1:
            neg.append(s)
        elif ov < 0.3:
            amb.append(s)
        else:
            pos.append(s)

print("=" * 78)
print(f"1) 연속 클립 점수 분포 by GT-overlap 구간  (총 {all_clips} clips, 61 videos)")
print("=" * 78)
print(f"{'bin':<22}{'count':>7}{'mean':>7}{'std':>7}{'p10':>7}{'p50':>7}{'p90':>7}{'p95':>7}{'max':>7}")
for name, arr in [("neg (ov<=0.1)", neg), ("ambiguous (0.1~0.3)", amb), ("pos (ov>=0.3)", pos)]:
    st = summ(arr)
    print(f"{name:<22}{st['count']:>7}{st['mean']:>7.3f}{st['std']:>7.3f}"
          f"{st['p10']:>7.3f}{st['p50']:>7.3f}{st['p90']:>7.3f}{st['p95']:>7.3f}{st['max']:>7.3f}")

print(f"\n  >> ambiguous 클립이 학습에서 빠졌던 것 — 이제 event 생성에 참여. "
      f"평균 {summ(amb)['mean']:.3f} (neg {summ(neg)['mean']:.3f} ~ pos {summ(pos)['mean']:.3f} 사이)")

# 옛 캐시 비교 (label 기반 pos/neg)
if OLD_CACHE.exists():
    old = json.load(open(OLD_CACHE))
    print("\n  [옛 필터링 캐시 비교] pos_summary mean=%.3f (n=%d) / neg_summary mean=%.3f (n=%d)"
          % (old["positive_prob_summary"]["mean"], old["positive_prob_summary"]["count"],
             old["negative_prob_summary"]["mean"], old["negative_prob_summary"]["count"]))
    print("   (옛 pos/neg는 label 기반=overlap≥0.3/≤0.1, ambiguous 제외 → 새 clean pos/neg와 거의 동일해야 정상)")

# ── 2) clip-level 판별력 (clean: pos vs neg, ambiguous 제외) ──
try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    y = [1] * len(pos) + [0] * len(neg)
    s = pos + neg
    prauc = average_precision_score(y, s)
    rocauc = roc_auc_score(y, s)
    print("\n" + "=" * 78)
    print("2) clip-level 판별력 (clean: pos≥0.3 vs neg≤0.1, ambiguous 제외)")
    print("=" * 78)
    print(f"  PR-AUC={prauc:.4f}  ROC-AUC={rocauc:.4f}   (옛 캐시 focal: PR-AUC=0.4209 ROC-AUC=0.6314)")
    print(f"  pos n={len(pos)} neg n={len(neg)}  (옛: pos 1847 / neg 4471)")
except ImportError:
    print("\n[2 skip] sklearn 없음")

# ── 3) GT recall ceiling ──
def s2f(s, e, fps, nbf):
    sf = max(0, int(math.floor(s * fps)))
    ef = min(int(nbf) - 1, int(math.ceil(e * fps)) - 1)
    return sf, max(sf, ef)

gt_max_scores = []
n_gt = 0
for vid in vids:
    fps = float(caches[vid]["fps"]); nbf = int(caches[vid]["nb_frames"])
    clips = caches[vid]["clips"]
    for ann in gtdb[vid].get("annotations", []):
        n_gt += 1
        gs, ge = s2f(ann["segment"][0], ann["segment"][1], fps, nbf)
        overl = [float(c["score"]) for c in clips
                 if max(c["start_frame"], gs) <= min(c["end_frame"], ge)]
        gt_max_scores.append(max(overl) if overl else 0.0)

gt_max_scores = np.array(gt_max_scores)
print("\n" + "=" * 78)
print(f"3) GT recall ceiling — 각 GT 구간의 최대 overlapping 클립 점수 (n_gt={n_gt})")
print("=" * 78)
print(f"  GT max-score 분포: {summ(gt_max_scores.tolist())}")
print(f"  {'start_score':>11}{'GT 도달가능':>12}{'recall ceiling':>15}")
for thr in [0.40, 0.44, 0.48, 0.52, 0.56, 0.60]:
    reach = int((gt_max_scores >= thr).sum())
    print(f"  {thr:>11.2f}{reach:>12}{reach / n_gt:>15.3f}")

print("\n  >> recall ceiling = 그 start_score에서 이론상 잡을 수 있는 GT 상한 "
      "(실제 event recall은 smoothing·IoU로 더 낮음).")

OUT = PROJECT_ROOT / "outputs/eval/analyze_contiguous_clips_61.json"
OUT.write_text(json.dumps({
    "n_clips": all_clips, "n_gt": n_gt,
    "dist": {"neg": summ(neg), "ambiguous": summ(amb), "pos": summ(pos)},
    "gt_max_score": summ(gt_max_scores.tolist()),
    "recall_ceiling": {str(t): float((gt_max_scores >= t).mean()) for t in [0.40,0.44,0.48,0.52,0.56,0.60]},
}, ensure_ascii=False, indent=2))
print(f"\n[saved] {OUT}")
