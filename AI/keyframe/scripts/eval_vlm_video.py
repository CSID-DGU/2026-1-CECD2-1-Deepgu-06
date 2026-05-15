"""
영상 단위 VLM Accuracy 평가 — 최종 평가 v3 (4-way)

평가 철학 (Option B 채택):
  - 1차 지표: Anomaly Accuracy (recall) — 파이프라인 전단 TSM이 normal을 필터링하므로
    VLM+keyframe selector의 핵심 역할은 anomaly 클립을 정확히 판별하는 것.
  - 2차 지표: Normal Accuracy (specificity) — TSM false positive 처리 능력.
  - 3차 지표: Temporal Coverage — anomaly 구간 커버율.

비교 선택기 (4-way):
  Uniform      — 등간격 8프레임 (baseline)
  Phase 1      — K-means + Motion
  Phase 2      — BiGRU FrameScorer (--model_path)
  PGL-SUM      — anomaly 재학습 + CLIP proxy (--pglsum_anomaly_path + --pglsum_clip)

로깅:
  - 영상별: 각 방법 선택 인덱스, VLM raw 응답, ✓/✗, 진행 중 anomaly acc 실시간 표시
  - JSON:  selected_indices, vlm_raw_response, coverage 전 방법 포함
  - TXT:   최종 인간 가독 리포트 (JSON과 동일 경로, .txt 확장자)
  - 클래스별 breakdown (Abuse / Assault / Fighting / Normal)

사용법:
  cd AI/ai_pipeline
  nohup python -u scripts/eval_vlm_video.py \\
      --video_dir              videos \\
      --annotation_path        Data/annotations/temporal_anomaly.txt \\
      --pglsum_anomaly_path    outputs/pglsum_anomaly.pth \\
      --pglsum_clip \\
      --model_path             outputs/frame_selector.pth \\
      --n_videos               40 \\
      --output_path            outputs/eval_vlm_video_result_v2.json \\
      --device                 cuda \\
      > outputs/logs/eval_vlm_video_v2.log 2>&1 &

중단/재시작 안전:
  --output_path에 영상별 중간 결과를 즉시 저장. 재실행 시 완료된 stem은 skip.
"""

import os
import sys
import json
import random
import argparse
import datetime
import time
import textwrap
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.sampler import KeyframeSampler
from models.feature_extractor import ResNet50Extractor
from scripts.prepare_data import parse_vlm_response


# -----------------------------------------------------------------------
# 유틸
# -----------------------------------------------------------------------

def _ts():
    return datetime.datetime.now().strftime("%H:%M:%S")


def _bar(val, total, width=20):
    filled = int(width * val / total) if total > 0 else 0
    return "[" + "█" * filled + "░" * (width - filled) + f"] {val}/{total}"


# -----------------------------------------------------------------------
# 영상 목록 수집
# -----------------------------------------------------------------------

_TARGET_ANOMALY_CLASSES = {"Abuse", "Assault", "Fighting"}
_NORMAL_PREFIXES = (
    "Normal_Videos_event",
    "Training-Normal-Videos-Part-1",
    "Training-Normal-Videos-Part-2",
    "Testing_Normal_Videos_Anomaly",
)


def collect_videos(video_dir):
    records = []
    for cls_dir in os.listdir(video_dir):
        cls_path = os.path.join(video_dir, cls_dir)
        if not os.path.isdir(cls_path):
            continue
        if cls_dir in _TARGET_ANOMALY_CLASSES:
            gt, cls_name = 1, cls_dir
        elif cls_dir.startswith(_NORMAL_PREFIXES):
            gt, cls_name = 0, "Normal"
        else:
            continue
        for fname in os.listdir(cls_path):
            if not fname.lower().endswith((".mp4", ".avi", ".mpeg", ".mkv")):
                continue
            stem = os.path.splitext(fname)[0]
            records.append((os.path.join(cls_path, fname), gt, stem, cls_name))
    return records


# -----------------------------------------------------------------------
# Temporal annotation
# -----------------------------------------------------------------------

def load_annotations(path):
    ann = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            stem = parts[0].replace(".mp4", "")
            segs = [(int(parts[2]), int(parts[3]))]
            if len(parts) >= 6 and parts[4] != "-1":
                segs.append((int(parts[4]), int(parts[5])))
            ann[stem] = segs
    return ann


# -----------------------------------------------------------------------
# 영상 디코딩
# -----------------------------------------------------------------------

def decode_video_frames(video_path, max_frames=500):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return [], []
    sample_indices = (
        list(range(total)) if total <= max_frames
        else np.linspace(0, total - 1, max_frames).astype(int).tolist()
    )
    frames, idx_list = [], []
    for fi in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        frames.append(frame)
        idx_list.append(fi)
    cap.release()
    return frames, idx_list


# -----------------------------------------------------------------------
# Uniform selector
# -----------------------------------------------------------------------

def uniform_select(T, n_frames=8):
    if T <= n_frames:
        return list(range(T))
    return np.linspace(0, T - 1, n_frames).astype(int).tolist()


# -----------------------------------------------------------------------
# Temporal Coverage
# -----------------------------------------------------------------------

def temporal_coverage(selected_idx, frame_indices, anomaly_segments):
    anomaly_set = set()
    for s, e in anomaly_segments:
        anomaly_set.update(range(s, e + 1))
    hits = sum(1 for li in selected_idx
               if li < len(frame_indices) and frame_indices[li] in anomaly_set)
    return hits / max(len(selected_idx), 1)


# -----------------------------------------------------------------------
# 결과 저장/로드
# -----------------------------------------------------------------------

def load_done(output_path):
    if not os.path.isfile(output_path):
        return set(), []
    with open(output_path) as f:
        data = json.load(f)
    done = {r["stem"] for r in data["results"]}
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
        cov_key = f"{m}_coverage"
        covs    = [r[cov_key] for r in results if r.get(cov_key) is not None]

        # 클래스별 breakdown
        cls_breakdown = {}
        for cls in _TARGET_ANOMALY_CLASSES | {"Normal"}:
            cls_rows = [(p, g) for p, g, c in rows if c == cls]
            if cls_rows:
                cls_breakdown[cls] = {
                    "correct": sum(p == g for p, g in cls_rows),
                    "total":   len(cls_rows),
                    "accuracy": sum(p == g for p, g in cls_rows) / len(cls_rows) * 100,
                }

        agg[m] = {
            "accuracy":          correct / total * 100,
            "anomaly_accuracy":  sum(p == g for p, g in anom) / len(anom) * 100 if anom else None,
            "normal_accuracy":   sum(p == g for p, g in norm) / len(norm) * 100 if norm else None,
            "temporal_coverage": float(np.mean(covs)) * 100 if covs else None,
            "n_total":           total,
            "n_anomaly":         len(anom),
            "n_normal":          len(norm),
            "cls_breakdown":     cls_breakdown,
        }
    return agg


def save_results(output_path, results, methods):
    tmp = output_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"results": results, "aggregate": aggregate(results, methods)},
                  f, indent=2, ensure_ascii=False)
    os.replace(tmp, output_path)


def save_report_txt(output_path, results, methods, args):
    """최종 인간 가독 리포트를 .txt 파일로 저장."""
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
    lines.append("  VLM Keyframe Selector 최종 평가 리포트")
    lines.append(f"  생성: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  평가 영상: {len(results)}개 | n_frames={args.n_frames} | device={args.device}")
    lines.append("=" * 72)
    lines.append("")
    lines.append("  ★ 평가 철학 (Option B)")
    lines.append("  1차 지표: Anomaly Accuracy — TSM이 normal을 pre-filter하므로")
    lines.append("            VLM+selector의 핵심 역할은 anomaly 판별.")
    lines.append("  2차 지표: Normal Accuracy  — TSM false positive 처리.")
    lines.append("  3차 지표: Temporal Coverage.")
    lines.append("")
    lines.append("─" * 72)
    lines.append(f"  {'선택기':<36} {'AnomAcc':>9} {'NormAcc':>9} {'Overall':>9} {'Coverage':>9}")
    lines.append("─" * 72)
    for m in methods:
        if m not in agg:
            continue
        a = agg[m]
        name = label_map.get(m, m)
        anom = f"{a['anomaly_accuracy']:.1f}%" if a['anomaly_accuracy'] is not None else "N/A"
        norm = f"{a['normal_accuracy']:.1f}%"  if a['normal_accuracy']  is not None else "N/A"
        ov   = f"{a['accuracy']:.1f}%"
        cov  = f"{a['temporal_coverage']:.1f}%" if a['temporal_coverage'] is not None else "N/A"
        lines.append(f"  {name:<36} {anom:>9} {norm:>9} {ov:>9} {cov:>9}")
    lines.append("─" * 72)
    lines.append("")

    # 클래스별 breakdown
    lines.append("  클래스별 Anomaly Accuracy")
    lines.append("─" * 72)
    header = f"  {'선택기':<20}"
    for cls in sorted(_TARGET_ANOMALY_CLASSES):
        header += f"  {cls:>10}"
    header += f"  {'Normal':>10}"
    lines.append(header)
    lines.append("─" * 72)
    for m in methods:
        if m not in agg:
            continue
        bd = agg[m].get("cls_breakdown", {})
        row = f"  {label_map.get(m, m)[:20]:<20}"
        for cls in sorted(_TARGET_ANOMALY_CLASSES):
            v = bd.get(cls, {}).get("accuracy")
            row += f"  {(f'{v:.1f}%' if v is not None else 'N/A'):>10}"
        v = bd.get("Normal", {}).get("accuracy")
        row += f"  {(f'{v:.1f}%' if v is not None else 'N/A'):>10}"
        lines.append(row)
    lines.append("─" * 72)
    lines.append("")

    # 영상별 상세
    lines.append("  영상별 예측 상세")
    lines.append("─" * 72)
    pred_header = f"  {'영상':40} {'GT':6}"
    for m in methods:
        pred_header += f" {m.upper():>8}"
    lines.append(pred_header)
    lines.append("─" * 72)
    for r in results:
        gt_str = "ANOM" if r["gt_label"] == 1 else "NORM"
        row = f"  {r['stem'][:40]:40} {gt_str:6}"
        for m in methods:
            pred = r.get(f"{m}_pred")
            if pred is None:
                row += f" {'ERR':>8}"
            else:
                pred_str = "ANOM" if pred == 1 else "NORM"
                mark = "✓" if pred == r["gt_label"] else "✗"
                row += f" {pred_str+mark:>8}"
        lines.append(row)
    lines.append("─" * 72)
    lines.append("")
    lines.append(f"  결과 JSON: {output_path}")
    lines.append("=" * 72)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[{_ts()}] 리포트 저장: {txt_path}")


# -----------------------------------------------------------------------
# 진행 중 로그 출력
# -----------------------------------------------------------------------

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
    print(f"  VLM Keyframe Selector 최종 평가 시작")
    print(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # ---------- 선택기 초기화 ----------
    methods = ["uniform", "p1"]

    p1_sampler = KeyframeSampler(n_frames=args.n_frames)
    print(f"[{_ts()}] Phase 1 (K-means+Motion) 준비 완료")

    # Phase 2 (BiGRU)
    p2_sampler = None
    if args.model_path and os.path.isfile(args.model_path):
        p2_sampler = KeyframeSampler(n_frames=args.n_frames,
                                     model_path=args.model_path,
                                     device=args.device)
        methods.append("p2")
        print(f"[{_ts()}] Phase 2 (BiGRU) 로드: {args.model_path}")
    else:
        print(f"[{_ts()}] Phase 2 skip (--model_path 없음 또는 파일 없음)")

    # PGL-SUM
    pgl_sampler = None
    _pgl_path = (args.pglsum_anomaly_path
                 if args.pglsum_anomaly_path and os.path.isfile(args.pglsum_anomaly_path)
                 else args.pglsum_model_path)
    if _pgl_path and os.path.isfile(_pgl_path):
        from pipeline.sampler import PGLSumSampler
        pgl_sampler = PGLSumSampler(
            model_path=_pgl_path,
            n_frames=args.n_frames,
            input_size=args.pglsum_input_size,
            device=args.device,
            use_clip=args.pglsum_clip,
            clip_weight=args.pglsum_clip_weight,
        )
        tag      = "(anomaly-retrained)" if _pgl_path == args.pglsum_anomaly_path else ""
        clip_tag = "+CLIP" if args.pglsum_clip else ""
        print(f"[{_ts()}] PGL-SUM {tag}{clip_tag} 로드: {_pgl_path}")
        methods.append("pgl")
    else:
        print(f"[{_ts()}] PGL-SUM skip (checkpoint 없음)")

    print(f"\n[{_ts()}] 비교 방법: {methods}\n")

    # ---------- Feature extractor ----------
    print(f"[{_ts()}] ResNet-50 로딩 중...")
    extractor = ResNet50Extractor(device=args.device, batch_size=64)
    print(f"[{_ts()}] ResNet-50 로딩 완료")

    # ---------- VLM ----------
    print(f"[{_ts()}] InternVL2 로딩 중...")
    from models.vlm.inference import InternVL
    vlm = InternVL(device=args.device)
    print(f"[{_ts()}] InternVL2 로딩 완료\n")

    # ---------- Temporal annotation ----------
    ann = {}
    if args.annotation_path and os.path.isfile(args.annotation_path):
        raw = load_annotations(args.annotation_path)
        ann = {k: v for k, v in raw.items()
               if any(k.startswith(c) for c in _TARGET_ANOMALY_CLASSES)}
        print(f"[{_ts()}] Temporal annotation: {len(ann)}개 영상\n")

    # ---------- 영상 목록 ----------
    all_vids = collect_videos(args.video_dir)
    anom_vids = [(p, g, s, c) for p, g, s, c in all_vids if g == 1]
    norm_vids = [(p, g, s, c) for p, g, s, c in all_vids if g == 0]
    random.shuffle(anom_vids)
    random.shuffle(norm_vids)
    n_each   = args.n_videos // 2
    selected = anom_vids[:n_each] + norm_vids[:n_each]
    random.shuffle(selected)
    print(f"[{_ts()}] 평가 영상: {len(selected)}개 "
          f"(anomaly={min(n_each,len(anom_vids))}, normal={min(n_each,len(norm_vids))})\n")

    # ---------- 중간 결과 로드 ----------
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    done_stems, all_results = load_done(args.output_path)
    if done_stems:
        print(f"[{_ts()}] 이미 완료: {len(done_stems)}개 → 이어서 실행\n")

    t_start = time.time()

    for vi, (vpath, gt, stem, cls_name) in enumerate(selected):
        if stem in done_stems:
            continue

        label_str = "ANOMALY" if gt == 1 else "NORMAL "
        print(f"[{_ts()}] ── [{len(done_stems)+1}/{len(selected)}] {label_str}  [{cls_name}]  {stem}")

        # 프레임 디코딩
        frames, frame_indices = decode_video_frames(vpath, max_frames=args.max_frames)
        if len(frames) < args.n_frames:
            print(f"  SKIP: 프레임 부족 ({len(frames)}개)")
            continue
        print(f"  디코딩: {len(frames)}프레임 (원본 {frame_indices[-1]+1}f 중 샘플)")

        # Feature 추출
        features = extractor.extract_from_frames(frames)  # (T, 2048)

        row = {"stem": stem, "gt_label": gt, "cls_name": cls_name}

        # ── Uniform ──────────────────────────────────────────────────
        u_idx = uniform_select(len(frames), args.n_frames)
        u_frames = [frames[i] for i in u_idx]
        try:
            u_raw = vlm.predict(u_frames)
            u_pred = parse_vlm_response(u_raw)
            row["uniform_pred"] = u_pred
            row["uniform_selected_indices"] = u_idx
            row["uniform_vlm_raw"] = u_raw[:200] if isinstance(u_raw, str) else str(u_raw)[:200]
        except Exception as e:
            u_pred = None
            row["uniform_pred"] = None
            print(f"  [경고] uniform VLM 오류: {e}")
        mark = "✓" if u_pred == gt else "✗"
        print(f"  Uniform  idx={u_idx}  pred={'ANOM' if u_pred==1 else 'NORM' if u_pred==0 else 'ERR'} {mark}")

        # ── Phase 1 ──────────────────────────────────────────────────
        c1 = {"clip": [None]*len(frames), "features": features}
        p1_sampler.sample(c1)
        p1_idx    = c1["selected_indices"]
        p1_frames = [frames[i] for i in p1_idx if i < len(frames)]
        try:
            p1_raw = vlm.predict(p1_frames)
            p1_pred = parse_vlm_response(p1_raw)
            row["p1_pred"] = p1_pred
            row["p1_selected_indices"] = p1_idx
            row["p1_vlm_raw"] = p1_raw[:200] if isinstance(p1_raw, str) else str(p1_raw)[:200]
        except Exception as e:
            p1_pred = None
            row["p1_pred"] = None
            print(f"  [경고] p1 VLM 오류: {e}")
        mark = "✓" if p1_pred == gt else "✗"
        print(f"  Phase 1  idx={p1_idx}  pred={'ANOM' if p1_pred==1 else 'NORM' if p1_pred==0 else 'ERR'} {mark}")

        # ── Phase 2 (BiGRU) ──────────────────────────────────────────
        if p2_sampler:
            c2 = {"clip": [None]*len(frames), "features": features}
            p2_sampler.sample(c2)
            p2_idx    = c2["selected_indices"]
            p2_frames = [frames[i] for i in p2_idx if i < len(frames)]
            try:
                p2_raw = vlm.predict(p2_frames)
                p2_pred = parse_vlm_response(p2_raw)
                row["p2_pred"] = p2_pred
                row["p2_selected_indices"] = p2_idx
                row["p2_vlm_raw"] = p2_raw[:200] if isinstance(p2_raw, str) else str(p2_raw)[:200]
            except Exception as e:
                p2_pred = None
                row["p2_pred"] = None
                print(f"  [경고] p2 VLM 오류: {e}")
            mark = "✓" if p2_pred == gt else "✗"
            print(f"  Phase 2  idx={p2_idx}  pred={'ANOM' if p2_pred==1 else 'NORM' if p2_pred==0 else 'ERR'} {mark}")

        # ── PGL-SUM ──────────────────────────────────────────────────
        if pgl_sampler:
            c3 = {"clip": frames, "features": features}
            pgl_sampler.sample(c3)
            pgl_idx    = c3["selected_indices"]
            pgl_frames = [frames[i] for i in pgl_idx if i < len(frames)]
            try:
                pgl_raw = vlm.predict(pgl_frames)
                pgl_pred = parse_vlm_response(pgl_raw)
                row["pgl_pred"] = pgl_pred
                row["pgl_selected_indices"] = pgl_idx
                row["pgl_vlm_raw"] = pgl_raw[:200] if isinstance(pgl_raw, str) else str(pgl_raw)[:200]
            except Exception as e:
                pgl_pred = None
                row["pgl_pred"] = None
                print(f"  [경고] pgl VLM 오류: {e}")
            mark = "✓" if pgl_pred == gt else "✗"
            print(f"  PGL-SUM  idx={pgl_idx}  pred={'ANOM' if pgl_pred==1 else 'NORM' if pgl_pred==0 else 'ERR'} {mark}")

        # ── Temporal Coverage (annotation 있는 영상, 전 방법) ────────
        if stem in ann:
            for m, idx in [("uniform", u_idx), ("p1", p1_idx),
                           *([("p2", p2_idx)] if p2_sampler else []),
                           *([("pgl", pgl_idx)] if pgl_sampler else [])]:
                cov = temporal_coverage(idx, frame_indices, ann[stem])
                row[f"{m}_coverage"] = cov
            cov_str = "  ".join(
                f"{m}={row[f'{m}_coverage']:.0%}"
                for m in methods if f"{m}_coverage" in row
            )
            print(f"  Coverage: {cov_str}")

        all_results.append(row)
        done_stems.add(stem)
        save_results(args.output_path, all_results, methods)

        # 진행 중 running summary (5영상마다)
        elapsed = time.time() - t_start
        if len(done_stems) % 5 == 0 or len(done_stems) == len(selected):
            agg = aggregate(all_results, methods)
            print_running_summary(agg, methods, len(done_stems), len(selected), elapsed)

        print()

    # ---------- 최종 결과 출력 ----------
    agg = aggregate(all_results, methods)
    label_map = {
        "uniform": "Uniform     ",
        "p1":      "Phase 1     ",
        "p2":      "Phase 2(BiGRU)",
        "pgl":     "PGL-SUM     ",
    }

    print(f"\n{'='*80}")
    print(f"  최종 평가 결과  ({len(all_results)}개 영상 | n_frames={args.n_frames} | max_frames={args.max_frames})")
    print(f"  ★ = 1차 지표 (Anomaly Accuracy)  |  Option B 채택 기준")
    print(f"{'='*80}")
    print(f"  {'선택기':<16} {'AnomAcc(★)':>12} {'NormAcc':>10} {'Overall':>10} {'Coverage':>10}")
    print(f"  {'─'*16}  {'─'*12}  {'─'*8}  {'─'*8}  {'─'*8}")
    for m in methods:
        if m not in agg:
            continue
        a = agg[m]
        anom = f"{a['anomaly_accuracy']:.1f}%" if a['anomaly_accuracy'] is not None else "N/A"
        norm = f"{a['normal_accuracy']:.1f}%"  if a['normal_accuracy']  is not None else "N/A"
        ov   = f"{a['accuracy']:.1f}%"
        cov  = f"{a['temporal_coverage']:.1f}%" if a['temporal_coverage'] is not None else "N/A"
        print(f"  {label_map.get(m,m):<16} {anom:>12} {norm:>10} {ov:>10} {cov:>10}")
    print(f"{'='*80}")

    print(f"\n  클래스별 Anomaly Accuracy")
    print(f"  {'선택기':<16}" + "".join(f"  {c:>10}" for c in sorted(_TARGET_ANOMALY_CLASSES)) + f"  {'Normal':>10}")
    print(f"  {'─'*16}" + "  ─────────" * (len(_TARGET_ANOMALY_CLASSES) + 1))
    for m in methods:
        if m not in agg:
            continue
        bd = agg[m].get("cls_breakdown", {})
        row_str = f"  {label_map.get(m,m):<16}"
        for cls in sorted(_TARGET_ANOMALY_CLASSES):
            v = bd.get(cls, {}).get("accuracy")
            row_str += f"  {(f'{v:.1f}%' if v is not None else 'N/A'):>10}"
        v = bd.get("Normal", {}).get("accuracy")
        row_str += f"  {(f'{v:.1f}%' if v is not None else 'N/A'):>10}"
        print(row_str)

    print(f"\n  결과 JSON: {args.output_path}")
    save_report_txt(args.output_path, all_results, methods, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="영상 단위 VLM Accuracy 최종 평가 (4-way)")
    parser.add_argument("--video_dir",            default="videos")
    parser.add_argument("--annotation_path",      default="Data/annotations/temporal_anomaly.txt")
    parser.add_argument("--model_path",           default=None,
                        help="Phase 2 BiGRU checkpoint (.pth)")
    parser.add_argument("--pglsum_model_path",    default=None)
    parser.add_argument("--pglsum_anomaly_path",  default=None)
    parser.add_argument("--pglsum_clip",          action="store_true")
    parser.add_argument("--pglsum_clip_weight",   type=float, default=0.5)
    parser.add_argument("--pglsum_input_size",    type=int,   default=2048)
    parser.add_argument("--n_videos",             type=int,   default=40)
    parser.add_argument("--max_frames",           type=int,   default=500)
    parser.add_argument("--n_frames",             type=int,   default=8)
    parser.add_argument("--output_path",          default="outputs/eval_vlm_video_result_v2.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args)
