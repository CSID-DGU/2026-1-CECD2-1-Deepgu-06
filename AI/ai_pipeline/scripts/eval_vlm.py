"""
클립 단위 VLM Accuracy 평가 — 최종 평가 v2 (4-way)

평가 철학 (Option B 채택):
  - 1차 지표: Anomaly Accuracy — TSM이 normal을 pre-filter하므로
    VLM+keyframe selector의 핵심 역할은 anomaly 클립 판별.
  - 2차 지표: Normal Accuracy.

clip_len 설계 원칙:
  - 고정값이어야 함 (클립 간 비교 일관성).
  - 16프레임이 아니어도 됨. --clip_len으로 조정.
  - 짧은 클립(16f): 선택기 간 차이 거의 없음 (세션 분석 결과).
  - 긴 클립(32f, 64f): 선택 여지가 커져 방법 간 차이 부각.
  권장: --clip_len 32 이상.

features는 on-the-fly로 추출 (ResNet-50) → clip_len 무관하게 동작.

비교 선택기 (4-way):
  Uniform      — 등간격 (baseline)
  Phase 1      — K-means + Motion
  Phase 2      — BiGRU FrameScorer (--model_path)
  PGL-SUM      — anomaly 재학습 + CLIP proxy (--pglsum_anomaly_path + --pglsum_clip)

사용법:
  cd AI/ai_pipeline
  nohup python -u scripts/eval_vlm.py \\
      --label_path          outputs/training_data/test.json \\
      --video_dir           videos \\
      --pglsum_anomaly_path outputs/pglsum_anomaly.pth \\
      --pglsum_clip \\
      --model_path          outputs/frame_selector.pth \\
      --clip_len            32 \\
      --n_samples           200 \\
      --output_path         outputs/eval_vlm_clip_result.json \\
      --device              cuda \\
      > outputs/logs/eval_vlm_clip.log 2>&1 &

중단/재시작 안전:
  --output_path에 중간 결과 즉시 저장. 재실행 시 완료된 clip_id는 skip.
"""

import os
import sys
import json
import random
import argparse
import datetime
import time
import glob
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.sampler import KeyframeSampler
from models.feature_extractor import ResNet50Extractor
from scripts.prepare_data import parse_vlm_response


def _ts():
    return datetime.datetime.now().strftime("%H:%M:%S")


def _bar(val, total, width=20):
    filled = int(width * val / total) if total > 0 else 0
    return "[" + "█" * filled + "░" * (width - filled) + f"] {val}/{total}"


# -----------------------------------------------------------------------
# 영상 인덱스
# -----------------------------------------------------------------------

def build_video_index(video_dir):
    index = {}
    for root, _, files in os.walk(video_dir):
        for fname in files:
            if fname.lower().endswith((".mp4", ".avi", ".mpeg", ".mkv")):
                stem = os.path.splitext(fname)[0]
                index[stem] = os.path.join(root, fname)
    return index


def get_class_from_stem(stem):
    for cls in ("Abuse", "Assault", "Fighting"):
        if stem.startswith(cls):
            return cls
    return "Normal"


# -----------------------------------------------------------------------
# 프레임 디코딩 (고정 clip_len)
# -----------------------------------------------------------------------

def decode_clip_frames(video_path, clip_idx, clip_len, stride):
    start = clip_idx * stride
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(clip_len):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


# -----------------------------------------------------------------------
# Uniform selector
# -----------------------------------------------------------------------

def uniform_select(n_total, n_frames):
    if n_total <= n_frames:
        return list(range(n_total))
    return np.linspace(0, n_total - 1, n_frames).astype(int).tolist()


# -----------------------------------------------------------------------
# 결과 저장/로드
# -----------------------------------------------------------------------

def load_done(output_path):
    if not os.path.isfile(output_path):
        return set(), []
    with open(output_path) as f:
        data = json.load(f)
    done = {r["clip_id"] for r in data["results"]}
    return done, data["results"]


def aggregate(results, methods):
    agg = {}
    for m in methods:
        pred_key = f"{m}_pred"
        rows = [(r[pred_key], r["gt_label"], r.get("cls_name", "?"))
                for r in results if r.get(pred_key) is not None]
        if not rows:
            continue
        total   = len(rows)
        correct = sum(p == g for p, g, _ in rows)
        anom    = [(p, g) for p, g, _ in rows if g == 1]
        norm    = [(p, g) for p, g, _ in rows if g == 0]

        cls_breakdown = {}
        for cls in ("Abuse", "Assault", "Fighting", "Normal"):
            cls_rows = [(p, g) for p, g, c in rows if c == cls]
            if cls_rows:
                cls_breakdown[cls] = {
                    "correct":  sum(p == g for p, g in cls_rows),
                    "total":    len(cls_rows),
                    "accuracy": sum(p == g for p, g in cls_rows) / len(cls_rows) * 100,
                }

        agg[m] = {
            "accuracy":         correct / total * 100,
            "anomaly_accuracy": sum(p == g for p, g in anom) / len(anom) * 100 if anom else None,
            "normal_accuracy":  sum(p == g for p, g in norm) / len(norm) * 100 if norm else None,
            "n_total":          total,
            "n_anomaly":        len(anom),
            "n_normal":         len(norm),
            "cls_breakdown":    cls_breakdown,
        }
    return agg


def save_results(output_path, results, methods):
    tmp = output_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"results": results, "aggregate": aggregate(results, methods)},
                  f, indent=2, ensure_ascii=False)
    os.replace(tmp, output_path)


def save_report_txt(output_path, results, methods, args):
    agg = aggregate(results, methods)
    txt_path = os.path.splitext(output_path)[0] + "_report.txt"
    label_map = {
        "uniform": "Uniform     (baseline)",
        "p1":      "Phase 1     (K-means+Motion)",
        "p2":      "Phase 2     (BiGRU)",
        "pgl":     "PGL-SUM     (anomaly-retrained+CLIP)",
    }
    lines = []
    lines.append("=" * 72)
    lines.append("  VLM Keyframe Selector 최종 평가 리포트 — 클립 단위")
    lines.append(f"  생성: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  평가 클립: {len(results)}개 | clip_len={args.clip_len} | "
                 f"n_frames={args.n_frames} | stride={args.stride} | device={args.device}")
    lines.append("=" * 72)
    lines.append("")
    lines.append("  ★ 평가 철학 (Option B)")
    lines.append("  1차 지표: Anomaly Accuracy — VLM+selector의 핵심 역할은 anomaly 판별.")
    lines.append("  2차 지표: Normal Accuracy  — TSM false positive 처리 능력.")
    lines.append("")
    lines.append("─" * 72)
    lines.append(f"  {'선택기':<36} {'AnomAcc(★)':>10} {'NormAcc':>9} {'Overall':>9}")
    lines.append("─" * 72)
    for m in methods:
        if m not in agg:
            continue
        a = agg[m]
        anom = f"{a['anomaly_accuracy']:.1f}%" if a['anomaly_accuracy'] is not None else "N/A"
        norm = f"{a['normal_accuracy']:.1f}%"  if a['normal_accuracy']  is not None else "N/A"
        ov   = f"{a['accuracy']:.1f}%"
        lines.append(f"  {label_map.get(m, m):<36} {anom:>10} {norm:>9} {ov:>9}")
    lines.append("─" * 72)
    lines.append("")
    lines.append("  클래스별 Anomaly Accuracy")
    lines.append("─" * 72)
    header = f"  {'선택기':<20}"
    for cls in ("Abuse", "Assault", "Fighting", "Normal"):
        header += f"  {cls:>10}"
    lines.append(header)
    lines.append("─" * 72)
    for m in methods:
        if m not in agg:
            continue
        bd = agg[m].get("cls_breakdown", {})
        row = f"  {label_map.get(m, m)[:20]:<20}"
        for cls in ("Abuse", "Assault", "Fighting", "Normal"):
            v = bd.get(cls, {}).get("accuracy")
            row += f"  {(f'{v:.1f}%' if v is not None else 'N/A'):>10}"
        lines.append(row)
    lines.append("─" * 72)
    lines.append(f"\n  결과 JSON: {output_path}")
    lines.append("=" * 72)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[{_ts()}] 리포트 저장: {txt_path}")


def print_running_summary(agg, methods, n_done, n_total, elapsed):
    label_map = {"uniform": "Uniform", "p1": "Phase 1", "p2": "Phase 2", "pgl": "PGL-SUM"}
    pct = n_done / n_total if n_total > 0 else 0
    eta = elapsed / pct * (1 - pct) if pct > 0 else 0
    print(f"\n  ┌─── 진행 현황 {_bar(n_done, n_total)}  ETA {int(eta//60)}:{int(eta%60):02d} ───")
    print(f"  │  {'선택기':<12} {'AnomAcc(★)':>12} {'NormAcc':>10} {'Overall':>10}")
    print(f"  │  {'─'*12}  {'─'*12}  {'─'*8}  {'─'*8}")
    for m in methods:
        if m not in agg:
            continue
        a = agg[m]
        anom = f"{a['anomaly_accuracy']:.1f}%" if a['anomaly_accuracy'] is not None else "  N/A"
        norm = f"{a['normal_accuracy']:.1f}%"  if a['normal_accuracy']  is not None else "  N/A"
        ov   = f"{a['accuracy']:.1f}%"
        print(f"  │  {label_map.get(m,m):<12} {anom:>12} {norm:>10} {ov:>10}")
    print(f"  └{'─'*55}")


# -----------------------------------------------------------------------
# 메인
# -----------------------------------------------------------------------

def main(args):
    random.seed(42)
    np.random.seed(42)

    print(f"\n{'='*60}")
    print(f"  VLM Keyframe Selector 클립 단위 평가 시작")
    print(f"  clip_len={args.clip_len}  n_frames={args.n_frames}  stride={args.stride}")
    print(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # ---------- 선택기 초기화 ----------
    methods = ["uniform", "p1"]

    p1_sampler = KeyframeSampler(n_frames=args.n_frames)
    print(f"[{_ts()}] Phase 1 (K-means+Motion) 준비 완료")

    p2_sampler = None
    if args.model_path and os.path.isfile(args.model_path):
        p2_sampler = KeyframeSampler(n_frames=args.n_frames,
                                     model_path=args.model_path,
                                     device=args.device)
        methods.append("p2")
        print(f"[{_ts()}] Phase 2 (BiGRU) 로드: {args.model_path}")
    else:
        print(f"[{_ts()}] Phase 2 skip")

    pgl_sampler = None
    _pgl_path = (args.pglsum_anomaly_path
                 if args.pglsum_anomaly_path and os.path.isfile(args.pglsum_anomaly_path)
                 else args.pglsum_model_path)
    if _pgl_path and os.path.isfile(_pgl_path):
        from pipeline.sampler import PGLSumSampler
        pgl_sampler = PGLSumSampler(
            model_path=_pgl_path,
            n_frames=args.n_frames,
            input_size=2048,
            device=args.device,
            use_clip=args.pglsum_clip,
            clip_weight=args.pglsum_clip_weight,
        )
        tag      = "(anomaly-retrained)" if _pgl_path == args.pglsum_anomaly_path else ""
        clip_tag = "+CLIP" if args.pglsum_clip else ""
        print(f"[{_ts()}] PGL-SUM {tag}{clip_tag} 로드: {_pgl_path}")
        methods.append("pgl")
    else:
        print(f"[{_ts()}] PGL-SUM skip")

    print(f"\n[{_ts()}] 비교 방법: {methods}\n")

    # ---------- Feature extractor (on-the-fly) ----------
    print(f"[{_ts()}] ResNet-50 로딩 중...")
    extractor = ResNet50Extractor(device=args.device, batch_size=64)
    print(f"[{_ts()}] ResNet-50 로딩 완료")

    # ---------- VLM ----------
    print(f"[{_ts()}] InternVL2 로딩 중...")
    from models.vlm.inference import InternVL
    vlm = InternVL(device=args.device)
    print(f"[{_ts()}] InternVL2 로딩 완료\n")

    # ---------- 클립 목록 ----------
    with open(args.label_path) as f:
        all_records = json.load(f)

    anom_recs = [r for r in all_records if r["gt_label"] == 1]
    norm_recs = [r for r in all_records if r["gt_label"] == 0]
    random.shuffle(anom_recs)
    random.shuffle(norm_recs)
    n_each  = args.n_samples // 2
    samples = anom_recs[:n_each] + norm_recs[:n_each]
    random.shuffle(samples)
    print(f"[{_ts()}] 평가 클립: {len(samples)}개 "
          f"(anomaly={min(n_each,len(anom_recs))}, normal={min(n_each,len(norm_recs))})\n")

    # ---------- 필요한 영상만 인덱싱 (최적화) ----------
    required_stems = {rec["clip_id"].rsplit("_clip", 1)[0] for rec in samples}
    print(f"[{_ts()}] 영상 인덱스 빌드 중 (필요한 {len(required_stems)}개만)...")

    # 디렉토리에서 영상 파일 찾기
    video_index = {}
    for video_path in sorted(glob.glob(os.path.join(args.video_dir, "**/*.mp4"), recursive=True)):
        stem = os.path.splitext(os.path.basename(video_path))[0]
        if stem in required_stems:
            video_index[stem] = video_path

    print(f"[{_ts()}] {len(video_index)}개 영상 인덱싱 완료 (필요한 {len(required_stems)}개 중)\n")

    # ---------- 중간 결과 로드 ----------
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    done_ids, all_results = load_done(args.output_path)
    if done_ids:
        print(f"[{_ts()}] 이미 완료: {len(done_ids)}개 → 이어서 실행\n")

    t_start = time.time()

    for i, rec in enumerate(samples):
        clip_id = rec["clip_id"]
        if clip_id in done_ids:
            continue

        stem, idx_str = clip_id.rsplit("_clip", 1)
        clip_idx  = int(idx_str)
        gt_label  = int(rec["gt_label"])
        cls_name  = get_class_from_stem(stem)
        label_str = "ANOMALY" if gt_label == 1 else "NORMAL "

        if stem not in video_index:
            print(f"[{_ts()}] SKIP (영상 없음): {stem}")
            continue

        # 프레임 디코딩 (고정 clip_len)
        frames = decode_clip_frames(video_index[stem], clip_idx,
                                    clip_len=args.clip_len, stride=args.stride)
        if len(frames) < args.n_frames:
            print(f"[{_ts()}] SKIP (프레임 부족 {len(frames)}): {clip_id}")
            continue

        # on-the-fly feature 추출
        features = extractor.extract_from_frames(frames)  # (T, 2048)

        print(f"[{_ts()}] ── [{len(done_ids)+1}/{len(samples)}] "
              f"{label_str}  [{cls_name}]  {clip_id}  ({len(frames)}f)")

        row = {"clip_id": clip_id, "gt_label": gt_label, "cls_name": cls_name}

        # ── Uniform ──────────────────────────────────────────────────
        u_idx    = uniform_select(len(frames), args.n_frames)
        u_frames = [frames[j] for j in u_idx]
        try:
            u_raw  = vlm.predict(u_frames)
            u_pred = parse_vlm_response(u_raw)
            row["uniform_pred"]             = u_pred
            row["uniform_selected_indices"] = u_idx
            row["uniform_vlm_raw"]          = (u_raw[:200] if isinstance(u_raw, str) else str(u_raw)[:200])
        except Exception as e:
            u_pred = None
            row["uniform_pred"] = None
            print(f"  [경고] uniform VLM 오류: {e}")
        mark = "✓" if u_pred == gt_label else "✗"
        print(f"  Uniform  idx={u_idx}  pred={'ANOM' if u_pred==1 else 'NORM' if u_pred==0 else 'ERR'} {mark}")

        # ── Phase 1 ──────────────────────────────────────────────────
        c1 = {"clip": [None]*len(frames), "features": features}
        p1_sampler.sample(c1)
        p1_idx    = c1["selected_indices"]
        p1_frames = [frames[j] for j in p1_idx if j < len(frames)]
        try:
            p1_raw  = vlm.predict(p1_frames)
            p1_pred = parse_vlm_response(p1_raw)
            row["p1_pred"]             = p1_pred
            row["p1_selected_indices"] = p1_idx
            row["p1_vlm_raw"]          = (p1_raw[:200] if isinstance(p1_raw, str) else str(p1_raw)[:200])
        except Exception as e:
            p1_pred = None
            row["p1_pred"] = None
            print(f"  [경고] p1 VLM 오류: {e}")
        mark = "✓" if p1_pred == gt_label else "✗"
        print(f"  Phase 1  idx={p1_idx}  pred={'ANOM' if p1_pred==1 else 'NORM' if p1_pred==0 else 'ERR'} {mark}")

        # ── Phase 2 (BiGRU) ──────────────────────────────────────────
        if p2_sampler:
            c2 = {"clip": [None]*len(frames), "features": features}
            p2_sampler.sample(c2)
            p2_idx    = c2["selected_indices"]
            p2_frames = [frames[j] for j in p2_idx if j < len(frames)]
            try:
                p2_raw  = vlm.predict(p2_frames)
                p2_pred = parse_vlm_response(p2_raw)
                row["p2_pred"]             = p2_pred
                row["p2_selected_indices"] = p2_idx
                row["p2_vlm_raw"]          = (p2_raw[:200] if isinstance(p2_raw, str) else str(p2_raw)[:200])
            except Exception as e:
                p2_pred = None
                row["p2_pred"] = None
                print(f"  [경고] p2 VLM 오류: {e}")
            mark = "✓" if p2_pred == gt_label else "✗"
            print(f"  Phase 2  idx={p2_idx}  pred={'ANOM' if p2_pred==1 else 'NORM' if p2_pred==0 else 'ERR'} {mark}")

        # ── PGL-SUM ──────────────────────────────────────────────────
        if pgl_sampler:
            c3 = {"clip": frames if args.pglsum_clip else [None]*len(frames), "features": features}
            pgl_sampler.sample(c3)
            pgl_idx    = c3["selected_indices"]
            pgl_frames = [frames[j] for j in pgl_idx if j < len(frames)]
            try:
                pgl_raw  = vlm.predict(pgl_frames)
                pgl_pred = parse_vlm_response(pgl_raw)
                row["pgl_pred"]             = pgl_pred
                row["pgl_selected_indices"] = pgl_idx
                row["pgl_vlm_raw"]          = (pgl_raw[:200] if isinstance(pgl_raw, str) else str(pgl_raw)[:200])
            except Exception as e:
                pgl_pred = None
                row["pgl_pred"] = None
                print(f"  [경고] pgl VLM 오류: {e}")
            mark = "✓" if pgl_pred == gt_label else "✗"
            print(f"  PGL-SUM  idx={pgl_idx}  pred={'ANOM' if pgl_pred==1 else 'NORM' if pgl_pred==0 else 'ERR'} {mark}")

        all_results.append(row)
        done_ids.add(clip_id)
        save_results(args.output_path, all_results, methods)

        elapsed = time.time() - t_start
        if len(done_ids) % 10 == 0 or len(done_ids) == len(samples):
            agg = aggregate(all_results, methods)
            print_running_summary(agg, methods, len(done_ids), len(samples), elapsed)

        print()

    # ---------- 최종 결과 ----------
    agg = aggregate(all_results, methods)
    label_map = {
        "uniform": "Uniform     ",
        "p1":      "Phase 1     ",
        "p2":      "Phase 2(BiGRU)",
        "pgl":     "PGL-SUM     ",
    }

    print(f"\n{'='*80}")
    print(f"  최종 평가 결과  ({len(all_results)}개 클립 | clip_len={args.clip_len}f | n_frames={args.n_frames} | stride={args.stride})")
    print(f"  ★ = 1차 지표 (Anomaly Accuracy)  |  Option B 채택 기준")
    print(f"{'='*80}")
    print(f"  {'선택기':<16} {'AnomAcc(★)':>12} {'NormAcc':>10} {'Overall':>10}")
    print(f"  {'─'*16}  {'─'*12}  {'─'*8}  {'─'*8}")
    for m in methods:
        if m not in agg:
            continue
        a = agg[m]
        anom = f"{a['anomaly_accuracy']:.1f}%" if a['anomaly_accuracy'] is not None else "N/A"
        norm = f"{a['normal_accuracy']:.1f}%"  if a['normal_accuracy']  is not None else "N/A"
        ov   = f"{a['accuracy']:.1f}%"
        print(f"  {label_map.get(m,m):<16} {anom:>12} {norm:>10} {ov:>10}")
    print(f"{'='*80}")

    print(f"\n  클래스별 Anomaly Accuracy")
    print(f"  {'선택기':<16}" + "".join(f"  {c:>10}" for c in ("Abuse","Assault","Fighting","Normal")))
    print(f"  {'─'*16}  {'─────────'*4}")
    for m in methods:
        if m not in agg:
            continue
        bd = agg[m].get("cls_breakdown", {})
        row_str = f"  {label_map.get(m,m):<16}"
        for cls in ("Abuse", "Assault", "Fighting", "Normal"):
            v = bd.get(cls, {}).get("accuracy")
            row_str += f"  {(f'{v:.1f}%' if v is not None else 'N/A'):>10}"
        print(row_str)

    print(f"\n  결과 JSON: {args.output_path}")
    save_report_txt(args.output_path, all_results, methods, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="클립 단위 VLM Accuracy 최종 평가 (4-way)")
    parser.add_argument("--label_path",           required=True,
                        help="prepare_data.py가 생성한 test.json")
    parser.add_argument("--video_dir",            default="videos")
    parser.add_argument("--n_samples",            type=int, default=40,
                        help="평가 총 클립 수 (anomaly N/2 + normal N/2). video eval과 동일하게 40 추천.")
    parser.add_argument("--n_frames",             type=int, default=8,
                        help="선택기가 고를 프레임 수")
    parser.add_argument("--clip_len",             type=int, default=32,
                        help="클립당 디코딩할 총 프레임 수. 16보다 크게 권장.")
    parser.add_argument("--stride",               type=int, default=8,
                        help="클립 시작 프레임 = clip_idx × stride")
    parser.add_argument("--model_path",           default=None,
                        help="Phase 2 BiGRU checkpoint (.pth)")
    parser.add_argument("--pglsum_model_path",    default=None)
    parser.add_argument("--pglsum_anomaly_path",  default=None)
    parser.add_argument("--pglsum_clip",          action="store_true")
    parser.add_argument("--pglsum_clip_weight",   type=float, default=0.5)
    parser.add_argument("--output_path",          default="outputs/eval_vlm_clip_result.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args)
