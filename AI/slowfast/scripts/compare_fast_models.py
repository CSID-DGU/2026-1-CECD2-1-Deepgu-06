"""
두 fast model clip-level 비교 (Step 2~4).
재추론 없이 캐시된 clip score 2개 사용.
  Step 2: GT max-score 분포 비교 (각 GT 구간 내 최고 클립 점수)
  Step 3: Recall ceiling (각 GT 안에 >=tau 클립이 하나라도 있는 GT 수 = event recall 상한)
  Step 4: PR-AUC / ROC-AUC + pos/neg 분포
"""
import json
import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FOCAL = PROJECT_ROOT / "outputs/eval/service_scope_clip_scores.json"
EARLY = PROJECT_ROOT / "outputs/eval/service_scope_clip_scores_earlystop.json"
CSV = PROJECT_ROOT / "data/manifests/cctv_x3d_s_overlap030/testing_clips_service_scope.csv"
SERVICE = PROJECT_ROOT / "data/manifests/test_service_scope.json"
GT_JSON = Path("/home/deepgu/test/cctv/dataset/ground-truth.json")
OUT = PROJECT_ROOT / "outputs/eval/fast_model_compare.json"


def s2f(s, e, fps, nbf):
    sf = max(0, int(math.floor(s * fps)))
    ef = min(int(nbf) - 1, int(math.ceil(e * fps)) - 1)
    return sf, max(sf, ef)


def pct(vals, p):
    if not vals:
        return 0.0
    v = sorted(vals)
    return v[min(len(v) - 1, int(p * len(v)))]


# ── load ──
sids = set(json.loads(SERVICE.read_text())["video_ids"])
focal = json.loads(FOCAL.read_text())
early = json.loads(EARLY.read_text())
focal_map = {r["clip_id"]: r["fighting_prob"] for r in focal["rows"]}
early_map = {r["clip_id"]: r["fighting_prob"] for r in early["rows"]}

df = pd.read_csv(CSV)
df = df[df.video_id.isin(sids)].copy()
vm = df.groupby("video_id").first()[["fps", "nb_frames"]].to_dict("index")

# 비디오별 클립 frame 범위
vclips = {}
for vid, grp in df.groupby("video_id"):
    vclips[vid] = [(r.clip_id, int(r.start_frame), int(r.end_frame)) for _, r in grp.iterrows()]

gtdb = json.loads(GT_JSON.read_text())["database"]

# 각 GT의 모델별 max overlapping clip score
gt_records = []  # {vid, gi, focal_max, early_max, gt_dur}
for vid in sorted(sids):
    fps = float(vm[vid]["fps"]); nbf = int(vm[vid]["nb_frames"])
    clips = vclips[vid]
    for gi, ann in enumerate(gtdb[vid].get("annotations", [])):
        s, e = ann["segment"]
        gs, ge = s2f(s, e, fps, nbf)
        f_scores = [focal_map.get(cid, 0.0) for cid, cs, ce in clips if max(cs, gs) <= min(ce, ge)]
        e_scores = [early_map.get(cid, 0.0) for cid, cs, ce in clips if max(cs, gs) <= min(ce, ge)]
        gt_records.append({
            "vid": vid, "gi": gi,
            "focal_max": max(f_scores) if f_scores else 0.0,
            "early_max": max(e_scores) if e_scores else 0.0,
            "gt_dur": e - s,
        })

n_gt = len(gt_records)

# ════ Step 4: PR-AUC / ROC-AUC + 분포 ════
print("=" * 64)
print("Step 4 — Clip-level 판별력 (scale-무관 주지표)")
print("=" * 64)
print(f"{'metric':<14}{'focal':>12}{'early_stop':>14}")
print(f"{'PR-AUC':<14}{focal['pr_auc']:>12.4f}{early['pr_auc']:>14.4f}")
print(f"{'ROC-AUC':<14}{focal['roc_auc']:>12.4f}{early['roc_auc']:>14.4f}")
fp_, fn_ = focal["positive_prob_summary"], focal["negative_prob_summary"]
ep_, en_ = early["positive_prob_summary"], early["negative_prob_summary"]
print(f"\n{'분포':<14}{'focal':>12}{'early_stop':>14}")
print(f"{'pos_mean':<14}{fp_['mean']:>12.3f}{ep_['mean']:>14.3f}")
print(f"{'neg_mean':<14}{fn_['mean']:>12.3f}{en_['mean']:>14.3f}")
print(f"{'gap(pos-neg)':<14}{fp_['mean']-fn_['mean']:>12.3f}{ep_['mean']-en_['mean']:>14.3f}")
print(f"{'pos_p50':<14}{fp_['p50']:>12.3f}{ep_['p50']:>14.3f}")
print(f"{'neg_p90':<14}{fn_['p90']:>12.3f}{en_['p90']:>14.3f}  (낮을수록 FP 적음)")

# ════ Step 2: GT max-score 분포 ════
print("\n" + "=" * 64)
print(f"Step 2 — GT별 max 클립점수 분포 (GT {n_gt}개)")
print("=" * 64)
f_maxes = [r["focal_max"] for r in gt_records]
e_maxes = [r["early_max"] for r in gt_records]
print(f"{'percentile':<12}{'focal':>12}{'early_stop':>14}")
for p in [0.10, 0.25, 0.50, 0.75, 0.90]:
    print(f"p{int(p*100):<11}{pct(f_maxes,p):>12.3f}{pct(e_maxes,p):>14.3f}")
print(f"{'mean':<12}{sum(f_maxes)/n_gt:>12.3f}{sum(e_maxes)/n_gt:>14.3f}")
for lo, hi in [(0.0,0.10),(0.10,0.35),(0.35,0.47),(0.47,1.01)]:
    fc = sum(1 for v in f_maxes if lo<=v<hi); ec = sum(1 for v in e_maxes if lo<=v<hi)
    print(f"  [{lo:.2f},{hi:.2f}) GT수    focal={fc:>3}   early={ec:>3}")

# ════ Step 3: Recall ceiling ════
print("\n" + "=" * 64)
print(f"Step 3 — Recall ceiling: GT 안에 >=tau 클립 하나라도 있는 GT 수 / {n_gt}")
print("=" * 64)
print(f"{'tau':>6}{'focal':>10}{'focal_R':>9}{'early':>9}{'early_R':>9}")
ceiling = {}
for tau in [0.40, 0.42, 0.44, 0.47, 0.50, 0.55]:
    fc = sum(1 for r in gt_records if r["focal_max"] >= tau)
    ec = sum(1 for r in gt_records if r["early_max"] >= tau)
    ceiling[tau] = {"focal": fc, "early": ec}
    print(f"{tau:>6.2f}{fc:>10}{fc/n_gt:>9.3f}{ec:>9}{ec/n_gt:>9.3f}")

# 모델별 단독 우위 GT (한쪽만 >=0.47)
only_f = [r for r in gt_records if r["focal_max"]>=0.47 and r["early_max"]<0.47]
only_e = [r for r in gt_records if r["early_max"]>=0.47 and r["focal_max"]<0.47]
print(f"\ntau=0.47에서 한쪽만 잡는 GT:  focal만={len(only_f)}   early만={len(only_e)}")

OUT.write_text(json.dumps({
    "pr_auc": {"focal": focal["pr_auc"], "early": early["pr_auc"]},
    "roc_auc": {"focal": focal["roc_auc"], "early": early["roc_auc"]},
    "gt_max_records": gt_records,
    "recall_ceiling": ceiling,
    "only_focal_0.47": [(r["vid"], r["gi"]) for r in only_f],
    "only_early_0.47": [(r["vid"], r["gi"]) for r in only_e],
}, ensure_ascii=False, indent=2))
print(f"\n[saved] {OUT}")
