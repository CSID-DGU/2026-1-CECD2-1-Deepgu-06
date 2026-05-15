"""
CLIP Score 기반 keyframe selector 평가.

각 선택기가 고른 8프레임을 CLIP으로 인코딩하고,
이상행동 관련 텍스트("a person fighting" 등)와의 유사도를 측정합니다.

목표: anomaly 클립에서 선택된 프레임이 얼마나 "이상행동처럼 생겼는가"
      (VLM 없이 측정하는 가장 의미 있는 대리 지표)

지표:
  Anomaly CLIP Score  = anomaly 클립에서 선택 프레임들의 이상행동 텍스트 유사도 평균
  Normal  CLIP Score  = normal  클립에서 선택 프레임들의 이상행동 텍스트 유사도 평균
  Discrimination Gap  = Anomaly Score - Normal Score
                        → 높을수록 선택기가 anomaly/normal을 잘 구분하는 프레임을 선택함

사용법:
  python scripts/eval_clip_score.py \\
      --label_path        outputs/training_data/test.json \\
      --video_dir         videos \\
      --pglsum_model_path ../../PGL-SUM/Summaries/UCF/models/split0/best_model.pth \\
      --n_samples         2000 \\
      --device            cuda
"""

import os
import sys
import json
import random
import argparse
import time
import datetime
import numpy as np
import cv2
from PIL import Image

import torch
import clip

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.sampler import KeyframeSampler


def _ts():
    return datetime.datetime.now().strftime("%H:%M:%S")


# ------------------------------------------------------------------
# 이상행동 관련 텍스트 프롬프트
# ------------------------------------------------------------------

ANOMALY_PROMPTS = [
    "a person fighting",
    "a person assaulting another person",
    "a violent physical altercation",
    "people involved in a fight",
    "an act of abuse or violence",
]

NORMAL_PROMPTS = [
    "people walking normally",
    "a normal street scene",
    "people going about their day",
]


# ------------------------------------------------------------------
# 비디오 인덱스
# ------------------------------------------------------------------

def build_video_index(video_dir):
    index = {}
    for root, _, files in os.walk(video_dir):
        for fname in files:
            if fname.lower().endswith((".mp4", ".avi", ".mpeg", ".mkv")):
                stem = os.path.splitext(fname)[0]
                index[stem] = os.path.join(root, fname)
    return index


def decode_clip_frames(video_path, clip_idx, clip_len=16, stride=8):
    start_frame = clip_idx * stride
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(clip_len):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


# ------------------------------------------------------------------
# Uniform selector
# ------------------------------------------------------------------

def uniform_select(n_total, n_frames=8):
    if n_total <= n_frames:
        return list(range(n_total))
    return np.linspace(0, n_total - 1, n_frames).astype(int).tolist()


# ------------------------------------------------------------------
# CLIP 점수 계산
# ------------------------------------------------------------------

def compute_clip_scores(clip_model, preprocess, frames_bgr, selected_indices,
                        anomaly_feats, normal_feats, device):
    """
    선택된 프레임들의 CLIP 이미지 feature를 추출하고
    anomaly/normal 텍스트 feature와의 cosine 유사도를 계산합니다.

    반환:
        anomaly_score: 이상행동 텍스트들과의 평균 유사도
        normal_score:  정상 텍스트들과의 평균 유사도
    """
    if not selected_indices or not frames_bgr:
        return 0.0, 0.0

    images = []
    for idx in selected_indices:
        if idx >= len(frames_bgr):
            continue
        bgr = frames_bgr[idx]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        images.append(preprocess(pil))

    if not images:
        return 0.0, 0.0

    img_tensor = torch.stack(images).to(device)  # (k, 3, 224, 224)

    with torch.no_grad():
        img_feats = clip_model.encode_image(img_tensor)           # (k, D)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

    # anomaly 텍스트들과 유사도 평균
    anom_sim = (img_feats @ anomaly_feats.T).mean().item()  # 스칼라
    norm_sim  = (img_feats @ normal_feats.T).mean().item()

    return float(anom_sim), float(norm_sim)


# ------------------------------------------------------------------
# 메인
# ------------------------------------------------------------------

def main(args):
    random.seed(42)
    np.random.seed(42)

    # CLIP 로드
    print(f"[{_ts()}] CLIP 모델 로딩 중... (ViT-B/32)")
    clip_model, preprocess = clip.load("ViT-B/32", device=args.device)
    clip_model.eval()

    # 텍스트 feature 사전 계산
    with torch.no_grad():
        anom_tokens  = clip.tokenize(ANOMALY_PROMPTS).to(args.device)
        normal_tokens= clip.tokenize(NORMAL_PROMPTS).to(args.device)
        anomaly_feats = clip_model.encode_text(anom_tokens)
        normal_feats  = clip_model.encode_text(normal_tokens)
        anomaly_feats = anomaly_feats / anomaly_feats.norm(dim=-1, keepdim=True)  # (5, D)
        normal_feats  = normal_feats  / normal_feats.norm(dim=-1, keepdim=True)   # (3, D)
    print(f"[{_ts()}] CLIP 로딩 완료")

    # test.json 로드 + stratified 샘플링
    with open(args.label_path) as f:
        all_records = json.load(f)

    anom_recs   = [r for r in all_records if r["gt_label"] == 1]
    normal_recs = [r for r in all_records if r["gt_label"] == 0]
    random.shuffle(anom_recs)
    random.shuffle(normal_recs)
    n_each  = args.n_samples // 2
    samples = anom_recs[:n_each] + normal_recs[:n_each]
    random.shuffle(samples)
    print(f"[{_ts()}] 평가 클립: {len(samples)}개 (anomaly={n_each}, normal={n_each})")

    # 비디오 인덱스
    print(f"[{_ts()}] 비디오 인덱스 빌드 중...")
    video_index = build_video_index(args.video_dir)
    print(f"[{_ts()}] {len(video_index)}개 영상 인덱싱 완료")

    # 선택기
    methods = ["uniform", "p1"]
    p1_sampler = KeyframeSampler(n_frames=args.n_frames)

    p2_sampler = None
    if args.model_path and os.path.isfile(args.model_path):
        p2_sampler = KeyframeSampler(n_frames=args.n_frames,
                                     model_path=args.model_path,
                                     device=args.device)
        methods.append("p2")

    pgl_sampler = None
    if args.pglsum_model_path and os.path.isfile(args.pglsum_model_path):
        from pipeline.sampler import PGLSumSampler
        pgl_sampler = PGLSumSampler(model_path=args.pglsum_model_path,
                                     n_frames=args.n_frames,
                                     input_size=args.pglsum_input_size,
                                     device=args.device)
        methods.append("pgl")

    print(f"[{_ts()}] 비교 방법: {methods}\n")

    # 결과 누적: method → {anom: [], normal: []} (각각 anomaly_score 리스트)
    scores = {m: {"anom": [], "normal": []} for m in methods}
    gaps   = {m: [] for m in methods}  # anomaly_score - normal_score per clip

    t_start = time.time()
    log_interval = max(1, len(samples) // 20)

    for i, rec in enumerate(samples):
        clip_id  = rec["clip_id"]
        gt_label = int(rec["gt_label"])
        stem, idx_str = clip_id.rsplit("_clip", 1)
        clip_idx = int(idx_str)

        if stem not in video_index:
            continue

        frames = decode_clip_frames(video_index[stem], clip_idx,
                                    clip_len=args.clip_len, stride=args.stride)
        if len(frames) < 4:
            continue

        features   = np.load(rec["features_path"]).astype(np.float32)
        dummy_clip = [None] * len(features)

        def run_method(selector, idx_list=None):
            if idx_list is None:
                c = {"clip": dummy_clip, "features": features}
                selector.sample(c)
                idx_list = c["selected_indices"]
            a_score, n_score = compute_clip_scores(
                clip_model, preprocess, frames, idx_list,
                anomaly_feats, normal_feats, args.device
            )
            return a_score, n_score, idx_list

        # Uniform
        u_idx = uniform_select(len(frames), args.n_frames)
        a, n, _ = run_method(None, idx_list=u_idx)
        bucket = "anom" if gt_label == 1 else "normal"
        scores["uniform"][bucket].append(a)
        gaps["uniform"].append(a - n)

        # Phase 1
        a, n, _ = run_method(p1_sampler)
        scores["p1"][bucket].append(a)
        gaps["p1"].append(a - n)

        # Phase 2
        if p2_sampler:
            a, n, _ = run_method(p2_sampler)
            scores["p2"][bucket].append(a)
            gaps["p2"].append(a - n)

        # PGL-SUM
        if pgl_sampler:
            a, n, _ = run_method(pgl_sampler)
            scores["pgl"][bucket].append(a)
            gaps["pgl"].append(a - n)

        # 진행 로그
        if (i + 1) % log_interval == 0 or (i + 1) == len(samples):
            elapsed = time.time() - t_start
            pct = (i + 1) / len(samples)
            eta = elapsed / pct * (1 - pct) if pct > 0 else 0
            gap_str = "  ".join(
                f"{m}={np.mean(gaps[m]):.4f}" for m in methods if gaps[m]
            )
            print(f"[{_ts()}] [{i+1}/{len(samples)}]  gap: {gap_str}"
                  f"  ETA {int(eta//60)}:{int(eta%60):02d}")

    # ------------------------------------------------------------------
    # 결과 출력
    # ------------------------------------------------------------------
    total_w = 26 + 14 * len(methods)
    label_map = {"uniform":"Uniform","p1":"Phase 1","p2":"Phase 2","pgl":"PGL-SUM"}

    print(f"\n{'='*total_w}")
    header = f"{'지표':<26}" + "".join(f"{label_map.get(m,m):>14}" for m in methods)
    print(header)
    print(f"{'-'*total_w}")

    def row(label, vals):
        line = f"{label:<26}"
        for m in methods:
            v = vals.get(m)
            line += f"{(f'{v:.4f}' if v is not None else 'N/A'):>14}"
        return line

    anom_scores = {m: float(np.mean(scores[m]["anom"]))   if scores[m]["anom"]   else None for m in methods}
    norm_scores = {m: float(np.mean(scores[m]["normal"])) if scores[m]["normal"] else None for m in methods}
    disc_gaps   = {m: float(np.mean(gaps[m]))             if gaps[m]             else None for m in methods}

    print(row("Anomaly CLIP Score (↑)",  anom_scores))
    print(row("Normal  CLIP Score (↓)",  norm_scores))
    print(row("Discrimination Gap (↑)",  disc_gaps))
    print(f"{'='*total_w}")
    print("\n* Anomaly CLIP Score: anomaly 클립에서 선택된 프레임의 이상행동 텍스트 유사도")
    print("  높을수록 선택된 프레임이 이상행동처럼 생겼음")
    print("* Discrimination Gap: Anomaly Score - Normal Score")
    print("  높을수록 anomaly/normal 클립을 구분하는 프레임을 선택함 ← 핵심 지표")

    # 저장
    result = {
        "n_samples": len(samples),
        "anomaly_prompts": ANOMALY_PROMPTS,
        "methods": {
            m: {
                "anomaly_clip_score": anom_scores[m],
                "normal_clip_score":  norm_scores[m],
                "discrimination_gap": disc_gaps[m],
            }
            for m in methods
        }
    }
    os.makedirs("outputs", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP Score 기반 keyframe selector 평가")
    parser.add_argument("--label_path",         required=True)
    parser.add_argument("--video_dir",           default="videos")
    parser.add_argument("--n_samples",           type=int, default=2000)
    parser.add_argument("--n_frames",            type=int, default=8)
    parser.add_argument("--clip_len",            type=int, default=16)
    parser.add_argument("--stride",              type=int, default=8)
    parser.add_argument("--output_path",         default="outputs/eval_clip_score_result.json")
    parser.add_argument("--pglsum_model_path",   default=None)
    parser.add_argument("--pglsum_input_size",   type=int, default=2048)
    parser.add_argument("--model_path",          default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args)
