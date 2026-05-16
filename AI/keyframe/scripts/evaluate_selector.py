"""
Phase 1(Temporal Clustering) vs Phase 2(학습된 FrameScorer) vs PGL-SUM+Anomaly 3-way 성능 평가.

평가 지표:
  1. Classifier Accuracy  : Phase 2 학습 classifier로 test label 예측 정확도
                            (Phase 2 classifier를 proxy로 사용하므로 참고 지표)
  2. Frame Diversity      : 선택된 frame feature 간 평균 cosine 거리
                            (높을수록 다양한 장면 커버)
  3. Temporal Spread      : 선택 index의 표준편차 / clip 길이
                            (높을수록 clip 전반에 고르게 분포)
  4. VLM Accuracy (선택)  : 실제 InternVL2로 선택 frame을 보고 label 예측 정확도
                            --use_vlm 플래그 필요, 시간 오래 걸림

비교 방식:
  - Phase 1  : K-means + Motion (학습 불필요, 항상 실행)
  - Phase 2  : 학습된 BiGRU FrameScorer  (--model_path 필요)
  - PGL-SUM  : PGL-SUM + Anomaly Proxy   (--pglsum_model_path 필요, 없으면 skip)

사용법:
  # Phase 1 vs Phase 2 (PGL-SUM 제외)
  python scripts/evaluate_selector.py \\
      --label_path outputs/training_data/test.json \\
      --model_path outputs/frame_selector.pth \\
      --n_frames 8

  # 3-way 비교 (PGL-SUM 포함)
  python scripts/evaluate_selector.py \\
      --label_path         outputs/training_data/test.json \\
      --model_path         outputs/frame_selector.pth \\
      --pglsum_model_path  outputs/pglsum.pth \\
      --pglsum_input_size  2048 \\
      --n_frames 8

  # VLM 포함 전체 평가
  python scripts/evaluate_selector.py \\
      --label_path outputs/training_data/test.json \\
      --model_path outputs/frame_selector.pth \\
      --n_frames 8 \\
      --use_vlm
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.sampler import KeyframeSampler
from models.frame_selector import DifferentiableFrameSelector


# ------------------------------------------------------------------
# 지표 계산
# ------------------------------------------------------------------

def frame_diversity(features, indices):
    """
    선택된 frame feature 간 평균 cosine 거리.
    값이 클수록(최대 2.0) 선택된 frame들이 서로 다른 장면을 담고 있음.
    """
    if len(indices) < 2:
        return 0.0

    sel = features[indices]                        # (k, D)
    sel_norm = sel / (np.linalg.norm(sel, axis=1, keepdims=True) + 1e-8)
    sim_matrix = sel_norm @ sel_norm.T             # (k, k)
    n = len(indices)
    upper = [(1 - sim_matrix[i, j])
             for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(upper)) if upper else 0.0


def temporal_spread(indices, T):
    """
    선택된 frame index의 표준편차 / clip 길이.
    높을수록 clip 전반에 고르게 분포.
    """
    if len(indices) < 2 or T < 2:
        return 0.0
    return float(np.std(indices) / T)


# ------------------------------------------------------------------
# classifier 기반 예측 (VLM 없이)
# ------------------------------------------------------------------

def classifier_predict(model, features, temperature=0.1):
    """
    학습된 DifferentiableFrameSelector로 선택된 frame들의 label 예측.

    주의: 이 classifier는 Phase 2 frame 선택으로 학습되었으므로,
    Phase 1/PGL-SUM 선택 frame에 적용하면 proxy 지표로만 해석해야 함.
    """
    feat_t = torch.FloatTensor(features)
    model.eval()
    with torch.no_grad():
        logits, _ = model(feat_t, temperature)
    return int(logits.argmax().item())


# ------------------------------------------------------------------
# VLM 기반 예측
# ------------------------------------------------------------------

def vlm_predict(vlm, frames, selected_indices):
    """InternVL2로 선택된 frame을 보고 label 예측."""
    from scripts.prepare_data import parse_vlm_response
    selected_frames = [frames[i] for i in selected_indices]
    try:
        response = vlm.predict(selected_frames)
        return parse_vlm_response(response)
    except Exception as e:
        print(f"  VLM 오류: {e}")
        return -1


# ------------------------------------------------------------------
# 결과 초기화 헬퍼
# ------------------------------------------------------------------

def _empty_result():
    return {"correct": 0, "diversity": [], "spread": [], "vlm_correct": 0, "vlm_total": 0}


# ------------------------------------------------------------------
# 메인
# ------------------------------------------------------------------

def main(args):
    # test.json 로드
    with open(args.label_path) as f:
        records = json.load(f)

    n_anom   = sum(r["label"] for r in records)
    n_normal = len(records) - n_anom
    print(f"평가 clip: {len(records)}개  (anomaly={n_anom}, normal={n_normal})")

    # ── Phase 1 sampler (항상)
    p1_sampler = KeyframeSampler(n_frames=args.n_frames)

    # ── Phase 2 sampler (model_path 없으면 skip)
    p2_sampler = None
    if args.model_path and os.path.isfile(args.model_path):
        p2_sampler = KeyframeSampler(
            n_frames=args.n_frames,
            model_path=args.model_path,
            device=args.device,
        )

    # ── PGL-SUM sampler (선택)
    pgl_sampler = None
    if args.pglsum_model_path:
        from pipeline.sampler import PGLSumSampler
        pgl_sampler = PGLSumSampler(
            model_path=args.pglsum_model_path,
            n_frames=args.n_frames,
            min_gap=args.min_gap,
            input_size=args.pglsum_input_size,
            device=args.device,
        )

    # ── classifier (Phase 2 proxy, model_path 없으면 skip)
    clf_model = None
    if args.model_path and os.path.isfile(args.model_path):
        clf_model = DifferentiableFrameSelector(
            input_dim=2048, hidden_dim=256, n_frames=args.n_frames, n_classes=2
        ).to(args.device)
        clf_model.scorer.load_state_dict(
            torch.load(args.model_path, map_location=args.device)
        )
        clf_model.eval()

    # ── VLM (선택)
    vlm = None
    if args.use_vlm:
        from models.vlm.inference import InternVL
        vlm = InternVL()

    # ── 결과 수집 초기화
    methods = ["p1"]
    if p2_sampler is not None:
        methods.append("p2")
    if pgl_sampler is not None:
        methods.append("pgl")
    results = {m: _empty_result() for m in methods}

    import time
    t_start = time.time()
    log_interval = max(1, len(records) // 20)   # ~5% 마다 출력

    print(f"\n[평가 시작] 총 {len(records)}개 clip  방법: {methods}")
    print(f"{'':>6}  {'완료':>8}  {'경과':>8}  {'ETA':>8}  {'P1 div':>8}  {'PGL div':>8}")
    print("-" * 65)

    for i, rec in enumerate(records):
        features = np.load(rec["features_path"]).astype(np.float32)
        gt_label = int(rec["label"])
        T = len(features)

        dummy_clip = [None] * T   # 지표 계산에는 clip 불필요

        # Phase 1
        c1 = {"clip": dummy_clip, "features": features}
        p1_sampler.sample(c1)
        p1_idx = c1["selected_indices"]

        # Phase 2 (있으면)
        p2_idx = None
        if p2_sampler is not None:
            c2 = {"clip": dummy_clip, "features": features}
            p2_sampler.sample(c2)
            p2_idx = c2["selected_indices"]

        # PGL-SUM (있으면)
        pgl_idx = None
        if pgl_sampler is not None:
            c3 = {"clip": dummy_clip, "features": features}
            pgl_sampler.sample(c3)
            pgl_idx = c3["selected_indices"]

        # ── Classifier Accuracy (clf_model 있을 때만)
        if clf_model is not None:
            results["p1"]["correct"] += int(classifier_predict(clf_model, features[p1_idx]) == gt_label)
            if p2_idx is not None:
                results["p2"]["correct"] += int(classifier_predict(clf_model, features[p2_idx]) == gt_label)
            if pgl_idx is not None:
                results["pgl"]["correct"] += int(classifier_predict(clf_model, features[pgl_idx]) == gt_label)

        # ── Frame Diversity
        results["p1"]["diversity"].append(frame_diversity(features, p1_idx))
        if p2_idx is not None:
            results["p2"]["diversity"].append(frame_diversity(features, p2_idx))
        if pgl_idx is not None:
            results["pgl"]["diversity"].append(frame_diversity(features, pgl_idx))

        # ── Temporal Spread
        results["p1"]["spread"].append(temporal_spread(p1_idx, T))
        if p2_idx is not None:
            results["p2"]["spread"].append(temporal_spread(p2_idx, T))
        if pgl_idx is not None:
            results["pgl"]["spread"].append(temporal_spread(pgl_idx, T))

        # ── VLM Accuracy
        if vlm and rec.get("clip_frames_path"):
            pass  # frame 저장이 없으면 skip

        # ── 진행 로그
        if (i + 1) % log_interval == 0 or (i + 1) == len(records):
            elapsed = time.time() - t_start
            pct = (i + 1) / len(records)
            eta = elapsed / pct * (1 - pct) if pct > 0 else 0
            p1_div_now = float(np.mean(results["p1"]["diversity"])) if results["p1"]["diversity"] else 0
            pgl_div_now = float(np.mean(results["pgl"]["diversity"])) if (pgl_sampler and results["pgl"]["diversity"]) else float("nan")
            print(
                f"[{pct:5.1%}]"
                f"  {i+1:>8}/{len(records)}"
                f"  {elapsed:>6.0f}s"
                f"  ETA {eta:>5.0f}s"
                f"  p1_div={p1_div_now:.4f}"
                + (f"  pgl_div={pgl_div_now:.4f}" if pgl_sampler else "")
            )

    n = len(records)
    total_elapsed = time.time() - t_start

    # ── 결과 집계
    has_p2  = p2_sampler  is not None
    has_pgl = pgl_sampler is not None
    has_clf = clf_model   is not None

    p1_div = float(np.mean(results["p1"]["diversity"]))
    p1_spr = float(np.mean(results["p1"]["spread"]))
    p2_div = float(np.mean(results["p2"]["diversity"])) if has_p2  else None
    p2_spr = float(np.mean(results["p2"]["spread"]))    if has_p2  else None
    pgl_div= float(np.mean(results["pgl"]["diversity"]))if has_pgl else None
    pgl_spr= float(np.mean(results["pgl"]["spread"]))   if has_pgl else None

    p1_acc  = results["p1"]["correct"]  / n * 100 if has_clf else None
    p2_acc  = results["p2"]["correct"]  / n * 100 if (has_clf and has_p2)  else None
    pgl_acc = results["pgl"]["correct"] / n * 100 if (has_clf and has_pgl) else None

    # ── 결과 출력
    n_cols = 1 + int(has_p2) + int(has_pgl)
    col_w  = 12
    total_w = 20 + col_w * n_cols + 3

    print(f"\n[평가 완료] {n}개 clip  소요={total_elapsed:.0f}s")
    print("=" * total_w)
    header = f"{'지표':<20} {'Phase 1':>{col_w}}"
    if has_p2:  header += f" {'Phase 2':>{col_w}}"
    if has_pgl: header += f" {'PGL-SUM':>{col_w}}"
    print(header)
    print("-" * total_w)

    def _fmt_row(label, vals, fmt):
        line = f"{label:<20}"
        for v in vals:
            cell = fmt.format(v) if v is not None else "  N/A"
            line += f" {cell:>{col_w}}"
        return line

    if has_clf:
        print(_fmt_row("Classifier Acc(*)", [p1_acc, p2_acc, pgl_acc] if has_p2 else [p1_acc, pgl_acc], "{:.1f}%"))
    print(_fmt_row("Frame Diversity",
                   [p1_div] + ([p2_div] if has_p2 else []) + ([pgl_div] if has_pgl else []),
                   "{:.4f}"))
    print(_fmt_row("Temporal Spread",
                   [p1_spr] + ([p2_spr] if has_p2 else []) + ([pgl_spr] if has_pgl else []),
                   "{:.4f}"))
    print("=" * total_w)
    if has_clf:
        print("(*) Classifier Acc: Phase 2 기준 proxy 지표 — 참고용으로만 해석.")

    # ── 결과 저장
    eval_result = {
        "n_clips": n,
        "elapsed_sec": round(total_elapsed, 1),
        "phase1": {
            "classifier_acc": p1_acc,
            "frame_diversity": p1_div,
            "temporal_spread": p1_spr,
        },
    }
    if has_p2:
        eval_result["phase2"] = {
            "classifier_acc": p2_acc,
            "frame_diversity": p2_div,
            "temporal_spread": p2_spr,
        }
    if has_pgl:
        eval_result["pglsum"] = {
            "classifier_acc": pgl_acc,
            "frame_diversity": pgl_div,
            "temporal_spread": pgl_spr,
        }

    os.makedirs("outputs", exist_ok=True)
    out_path = (
        args.pglsum_model_path.replace(".pth", ".eval.json")
        if args.pglsum_model_path
        else "outputs/eval_result.json"
    )
    with open(out_path, "w") as f:
        json.dump(eval_result, f, indent=2, ensure_ascii=False)
    print(f"\n평가 결과 저장: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3-way keyframe selector 성능 평가")

    parser.add_argument("--label_path",        required=True,
                        help="prepare_data.py 가 생성한 test.json 경로")
    parser.add_argument("--model_path",         default=None,
                        help="train_frame_selector.py 가 저장한 Phase 2 .pth 경로. 없으면 Phase 2 비교 생략.")

    # PGL-SUM 옵션 (없으면 PGL-SUM 비교 skip)
    parser.add_argument("--pglsum_model_path",  default=None,
                        help="PGL-SUM checkpoint (.pth) 경로. 없으면 PGL-SUM 비교 생략.")
    parser.add_argument("--pglsum_input_size",  type=int, default=2048,
                        help="PGL-SUM 입력 feature 차원. ResNet-50=2048, GoogleNet=1024.")
    parser.add_argument("--min_gap",            type=int, default=4,
                        help="PGLSumSampler 의 최소 frame 간격.")

    parser.add_argument("--n_frames",  type=int,  default=8)
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_vlm",   action="store_true",
                        help="InternVL2 로 실제 VLM accuracy 도 측정 (느림)")

    args = parser.parse_args()
    main(args)
