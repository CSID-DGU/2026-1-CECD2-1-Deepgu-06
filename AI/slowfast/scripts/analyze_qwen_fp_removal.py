"""
Qwen3-VL FP 제거 / TP 손실 분석

단일 Qwen 파이프라인 실행에서:
  - 후보 이벤트(kept + rejected) = Fast-only 이벤트 집합
  - kept 이벤트                  = Fast + Qwen 이벤트 집합
을 함께 수집해, 동일 후보집합 위에서 Fast-only vs Fast+Qwen을 비교한다.

출력: [1] 성능표  [2] VLM 자체 성능(유지/제거)  [3] Precision/Recall/F1 변화
      [4] 제거 이벤트 reasoning 분류  + 전체 per-event JSON 저장.
"""
import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.main_pipeline import run_single_video_pipeline
from scripts.evaluate_event_batch import build_gt_events, build_video_path, interval_iou
from utils.config import load_config
from utils.io import load_json, write_json, ensure_dir

IOU = 0.10

# reasoning 키워드 분류 (우선순위 순서대로 첫 매칭 채택)
CATEGORIES = [
    ("Sports/Play", ["sport", "playing", "play ", "game", "dancing", "dance", "exercise"]),
    ("Aftermath", ["aftermath", "restrain", "lying", "injured", "fallen", "assist", "helping",
                   "help ", "police", "handcuff", "medical", "treat", "care"]),
    ("Bystanders", ["bystander", "onlooker", "watching", "observer", "spectator"]),
    ("Crowd Scene", ["crowd", "gathering", "group of", "many people", "people gathered"]),
    ("Argument", ["argument", "arguing", "argue", "shout", "verbal", "confront", "conversation",
                  "talking", "discussion", "dispute"]),
    ("Walking/Standing", ["walking", "walk", "standing", "stand", "waiting", "passing", "pass by",
                          "normal", "casual", "ordinary", "no aggression", "no visible", "no fight"]),
]


def classify_reasoning(text):
    t = (text or "").lower()
    for name, kws in CATEGORIES:
        if any(k in t for k in kws):
            return name
    return "Other"


def gt_centric_counts(events, gt_events):
    """events 집합에 대한 (tp_gt, fp_pred, fn, n_pred)."""
    tp_gt = 0
    for g in gt_events:
        if any(interval_iou(e["start_frame"], e["end_frame"], g["start_frame"], g["end_frame"]) >= IOU
               for e in events):
            tp_gt += 1
    fp_pred = sum(1 for e in events if not e["_gt_match"])
    fn = max(0, len(gt_events) - tp_gt)
    return tp_gt, fp_pred, fn, len(events)


def prf(tp, fp, fn, n_pred):
    tp_pred = max(0, n_pred - fp)
    precision = tp_pred / max(1, n_pred)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval_event_v3_start040_bedrock_qwen.yaml")
    ap.add_argument("--ground-truth-json", default="/home/deepgu/test/cctv/dataset/ground-truth.json")
    ap.add_argument("--dataset-root", default="/home/deepgu/test/cctv/dataset")
    ap.add_argument("--service-scope-json", default=str(PROJECT_ROOT / "data/manifests/test_service_scope.json"))
    ap.add_argument("--output-json", default="outputs/eval/analyze_qwen_fp_removal_61.json")
    args = ap.parse_args()

    config = load_config(args.config)
    config.setdefault("outputs", {}).update(
        save_run_artifacts=False, save_event_media=False, save_clip_manifest=False)
    database = load_json(args.ground_truth_json)["database"]
    allowed = set(load_json(args.service_scope_json)["video_ids"])

    # 집계용
    fast_tp = fast_fp = fast_fn = fast_np = 0
    qwen_tp = qwen_fp = qwen_fn = qwen_np = 0
    cell = {"TP유지": 0, "TP제거": 0, "FP제거": 0, "FP유지": 0}
    removed_records = []   # 제거(reject)된 이벤트
    per_event_all = []

    vids = sorted(v for v in allowed if v in database)
    print(f"[start] {len(vids)} videos", flush=True)
    for i, vid in enumerate(vids):
        meta = database[vid]
        vpath = build_video_path(args.dataset_root, meta.get("source"), meta.get("subset"), vid)
        if not vpath.exists():
            print(f"  [skip] {vid} (no file)", flush=True)
            continue
        res = run_single_video_pipeline(str(vpath), deepcopy(config), run_name=f"qa_{vid}", verbose=False)
        gt = build_gt_events(meta)
        kept = res["events"]
        rejected = res["rejected_events"]
        candidates = kept + rejected
        for e in candidates:
            e["_gt_match"] = any(
                interval_iou(e["start_frame"], e["end_frame"], g["start_frame"], g["end_frame"]) >= IOU
                for g in gt)
            e["_kept"] = e.get("vlm_decision") != "reject"

        # gt-centric 집계
        ftp, ffp, ffn, fnp = gt_centric_counts(candidates, gt)
        qtp, qfp, qfn, qnp = gt_centric_counts([e for e in candidates if e["_kept"]], gt)
        fast_tp += ftp; fast_fp += ffp; fast_fn += ffn; fast_np += fnp
        qwen_tp += qtp; qwen_fp += qfp; qwen_fn += qfn; qwen_np += qnp

        # pred-level 유지/제거 분류
        for e in candidates:
            gtm, kp = e["_gt_match"], e["_kept"]
            if gtm and kp:        cell["TP유지"] += 1
            elif gtm and not kp:  cell["TP제거"] += 1
            elif not gtm and not kp: cell["FP제거"] += 1
            else:                 cell["FP유지"] += 1
            rec = {
                "video_id": vid, "start_frame": e["start_frame"], "end_frame": e["end_frame"],
                "duration_sec": round(e.get("duration_sec", 0.0), 1),
                "gt_match": gtm, "kept": kp,
                "vlm_decision": e.get("vlm_decision"), "vlm_score": e.get("vlm_score"),
                "fast_peak_score": round(e.get("fast_peak_score", 0.0), 3),
                "reasoning": (e.get("vlm_raw", "") or "")[:600],
                "vlm_subq": e.get("vlm_subq"),
                "sampling_used": e.get("_sampling_used"),
                "sampling_k": e.get("_sampling_k"),
            }
            per_event_all.append(rec)
            if not kp:  # 제거됨
                rec2 = dict(rec); rec2["category"] = classify_reasoning(rec["reasoning"])
                rec2["loss_type"] = "TP_lost" if gtm else "FP_removed"
                removed_records.append(rec2)
        print(f"  [{i+1}/{len(vids)}] {vid} cand={len(candidates)} kept={sum(e['_kept'] for e in candidates)} "
              f"rej={sum(not e['_kept'] for e in candidates)}", flush=True)

    # ---- 지표 계산 ----
    fp_p, fp_r, fp_f1 = prf(fast_tp, fast_fp, fast_fn, fast_np)
    qp, qr, qf1 = prf(qwen_tp, qwen_fp, qwen_fn, qwen_np)

    tp_total = cell["TP유지"] + cell["TP제거"]
    fp_total = cell["FP제거"] + cell["FP유지"]
    tp_ret = cell["TP유지"] / max(1, tp_total)
    fp_rem = cell["FP제거"] / max(1, fp_total)

    # reasoning 분류 집계
    def tally(loss_type):
        out = {}
        for r in removed_records:
            if r["loss_type"] == loss_type:
                out[r["category"]] = out.get(r["category"], 0) + 1
        return dict(sorted(out.items(), key=lambda x: -x[1]))
    fp_cat = tally("FP_removed")
    tp_cat = tally("TP_lost")

    # ---- 출력 ----
    print("\n" + "=" * 60)
    print("[1] Fast Only vs Fast + Qwen (61, GT-centric, IoU>=0.10)")
    print("| Setting     | TP | FP | FN | Precision | Recall | F1 |")
    print("|-------------|----|----|----|-----------|--------|-----|")
    print(f"| Fast Only   | {fast_tp} | {fast_fp} | {fast_fn} | {fp_p:.4f} | {fp_r:.4f} | {fp_f1:.4f} |")
    print(f"| Fast + Qwen | {qwen_tp} | {qwen_fp} | {qwen_fn} | {qp:.4f} | {qr:.4f} | {qf1:.4f} |")
    print(f"  (n_pred: fast={fast_np}, qwen={qwen_np})")

    print("\n[2] VLM 자체 성능 (Fast-only 후보 이벤트, pred-level)")
    print(f"  TP 유지 : {cell['TP유지']}")
    print(f"  TP 제거 : {cell['TP제거']}")
    print(f"  FP 제거 : {cell['FP제거']}")
    print(f"  FP 유지 : {cell['FP유지']}")
    print(f"  TP Retention Rate = {cell['TP유지']}/{tp_total} = {tp_ret:.4f}")
    print(f"  FP Removal Rate   = {cell['FP제거']}/{fp_total} = {fp_rem:.4f}")

    print("\n[3] Precision Gain vs Recall Loss (Fast-only 대비)")
    print(f"  Precision: {qp - fp_p:+.4f}")
    print(f"  Recall   : {qr - fp_r:+.4f}")
    print(f"  F1       : {qf1 - fp_f1:+.4f}")

    print("\n[4] 제거된 이벤트 reasoning 분류")
    print(f"  FP 제거 이벤트 유형 ({sum(fp_cat.values())}개): {fp_cat}")
    print(f"  TP 손실 이벤트 유형 ({sum(tp_cat.values())}개): {tp_cat}")

    payload = {
        "iou": IOU, "n_videos": len(vids),
        "metrics": {
            "fast_only": {"tp": fast_tp, "fp": fast_fp, "fn": fast_fn, "n_pred": fast_np,
                          "precision": fp_p, "recall": fp_r, "f1": fp_f1},
            "fast_qwen": {"tp": qwen_tp, "fp": qwen_fp, "fn": qwen_fn, "n_pred": qwen_np,
                          "precision": qp, "recall": qr, "f1": qf1},
            "delta": {"precision": qp - fp_p, "recall": qr - fp_r, "f1": qf1 - fp_f1},
        },
        "vlm_cells": cell, "tp_retention_rate": tp_ret, "fp_removal_rate": fp_rem,
        "removed_fp_categories": fp_cat, "removed_tp_categories": tp_cat,
        "removed_records": removed_records, "per_event_all": per_event_all,
    }
    ensure_dir(Path(args.output_json).parent)
    write_json(args.output_json, payload)
    print(f"\n[output] {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
