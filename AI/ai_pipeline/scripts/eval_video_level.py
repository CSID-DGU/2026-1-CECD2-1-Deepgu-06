"""
영상 단위 keyframe selector 평가 (성능 비교 전용).

clip(16프레임) 단위 평가는 선택기 간 차이가 거의 없어서 무의미하므로,
영상 전체에서 8프레임을 고르는 방식으로 비교합니다.

평가 지표:
  1. CLIP Score Discrimination Gap
       anomaly/normal 영상에서 고른 8프레임의 이상행동 텍스트 유사도 차이.
       높을수록 "anomaly 영상에서 이상행동 프레임을 잘 골랐음".

  2. Temporal Coverage (temporal annotation 있는 10개 영상만)
       실제 이상행동 구간 프레임을 얼마나 커버하는가.
       clip 평가와 달리 전체 영상 중 8프레임만 고르므로 진짜 의미가 있음.

비교 선택기:
  Uniform   — 등간격 8프레임
  Phase 1   — K-means(k=8) + Motion
  PGL-SUM   — pglsum_model_path 제공 시
  Phase 2   — model_path 제공 시 (BiGRU)

사용법:
  cd AI/ai_pipeline
  python scripts/eval_video_level.py \\
      --video_dir         videos \\
      --annotation_path   Data/annotations/temporal_anomaly.txt \\
      --pglsum_model_path ../../PGL-SUM/Summaries/UCF/models/split0/best_model.pth \\
      --model_path        outputs/frame_selector.pth \\
      --n_videos          40 \\
      --device            cuda
"""

import os
import sys
import json
import random
import argparse
import datetime
import time
import numpy as np
import cv2
from PIL import Image

import torch
import clip

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.sampler import KeyframeSampler
from models.feature_extractor import ResNet50Extractor


def _ts():
    return datetime.datetime.now().strftime("%H:%M:%S")


# -----------------------------------------------------------------------
# 이상행동 텍스트 프롬프트
# -----------------------------------------------------------------------

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

# 평가 대상 3개 클래스만 anomaly로 사용 (prepare_data.py 와 동일)
_TARGET_ANOMALY_CLASSES = {"Abuse", "Assault", "Fighting"}
_NORMAL_PREFIXES = (
    "Normal_Videos_event",
    "Training-Normal-Videos-Part-1",
    "Training-Normal-Videos-Part-2",
    "Testing_Normal_Videos_Anomaly",
)


# -----------------------------------------------------------------------
# 영상 목록 수집 (anomaly / normal 라벨 포함)
# -----------------------------------------------------------------------

def collect_videos(video_dir):
    """
    반환: list of (video_path, gt_label, stem)
      gt_label 1 = anomaly (Abuse/Assault/Fighting)
              0 = normal
    """
    records = []
    for cls_dir in os.listdir(video_dir):
        cls_path = os.path.join(video_dir, cls_dir)
        if not os.path.isdir(cls_path):
            continue

        if cls_dir in _TARGET_ANOMALY_CLASSES:
            gt = 1
        elif cls_dir.startswith(_NORMAL_PREFIXES):
            gt = 0
        else:
            continue  # Arrest, Arson 등 비대상 클래스 제외

        for fname in os.listdir(cls_path):
            if not fname.lower().endswith((".mp4", ".avi", ".mpeg", ".mkv")):
                continue
            stem = os.path.splitext(fname)[0]
            records.append((os.path.join(cls_path, fname), gt, stem))

    return records


# -----------------------------------------------------------------------
# Temporal annotation 로드
# -----------------------------------------------------------------------

def load_annotations(path):
    """
    반환: { video_stem: [(start_frame, end_frame), ...] }
    형식: VideoName.mp4  Class  start1  end1  [start2  end2]
    """
    ann = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            stem = parts[0].replace(".mp4", "")
            s1, e1 = int(parts[2]), int(parts[3])
            segs = [(s1, e1)]
            if len(parts) >= 6 and parts[4] != "-1":
                segs.append((int(parts[4]), int(parts[5])))
            ann[stem] = segs
    return ann


# -----------------------------------------------------------------------
# 영상 디코딩: 최대 max_frames 프레임 균등 추출
# -----------------------------------------------------------------------

def decode_video_frames(video_path, max_frames=500):
    """
    영상 전체에서 최대 max_frames 프레임을 균등 간격으로 읽습니다.
    반환: (frames_bgr, frame_indices)
      frames_bgr    : list of np.ndarray (BGR)
      frame_indices : 각 프레임의 원본 영상 내 절대 프레임 번호
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return [], []

    if total <= max_frames:
        sample_indices = list(range(total))
    else:
        sample_indices = np.linspace(0, total - 1, max_frames).astype(int).tolist()

    frames = []
    idx_list = []
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
# Uniform selector (전체 T 프레임 중 n_frames 등간격)
# -----------------------------------------------------------------------

def uniform_select(T, n_frames=8):
    if T <= n_frames:
        return list(range(T))
    return np.linspace(0, T - 1, n_frames).astype(int).tolist()


# -----------------------------------------------------------------------
# CLIP 점수 계산
# -----------------------------------------------------------------------

def compute_clip_scores(clip_model, preprocess, frames_bgr, selected_local_idx,
                        anomaly_feats, normal_feats, device):
    """
    selected_local_idx: frames_bgr 내 인덱스 (0-based)
    반환: (anomaly_score, normal_score)
    """
    images = []
    for i in selected_local_idx:
        if i >= len(frames_bgr):
            continue
        bgr = frames_bgr[i]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        images.append(preprocess(Image.fromarray(rgb)))

    if not images:
        return 0.0, 0.0

    img_tensor = torch.stack(images).to(device)
    with torch.no_grad():
        img_feats = clip_model.encode_image(img_tensor)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

    anom_sim = (img_feats @ anomaly_feats.T).mean().item()
    norm_sim  = (img_feats @ normal_feats.T).mean().item()
    return float(anom_sim), float(norm_sim)


# -----------------------------------------------------------------------
# Temporal Coverage 계산
# -----------------------------------------------------------------------

def temporal_coverage(selected_local_idx, frame_indices, anomaly_segments):
    """
    selected_local_idx : frames_bgr 내 인덱스
    frame_indices       : 각 로컬 인덱스의 원본 절대 프레임 번호
    anomaly_segments    : [(start, end), ...]
    반환: fraction of selected frames that fall in anomaly region
    """
    anomaly_set = set()
    for s, e in anomaly_segments:
        anomaly_set.update(range(s, e + 1))

    hits = sum(1 for li in selected_local_idx
               if li < len(frame_indices) and frame_indices[li] in anomaly_set)
    return hits / max(len(selected_local_idx), 1)


# -----------------------------------------------------------------------
# 메인
# -----------------------------------------------------------------------

def main(args):
    random.seed(42)
    np.random.seed(42)

    # ---------- 모델 로딩 ----------
    print(f"[{_ts()}] CLIP 모델 로딩 중... (ViT-B/32)")
    clip_model, preprocess = clip.load("ViT-B/32", device=args.device)
    clip_model.eval()
    with torch.no_grad():
        anom_tokens  = clip.tokenize(ANOMALY_PROMPTS).to(args.device)
        norm_tokens  = clip.tokenize(NORMAL_PROMPTS).to(args.device)
        anomaly_feats = clip_model.encode_text(anom_tokens)
        normal_feats  = clip_model.encode_text(norm_tokens)
        anomaly_feats /= anomaly_feats.norm(dim=-1, keepdim=True)
        normal_feats  /= normal_feats.norm(dim=-1, keepdim=True)
    print(f"[{_ts()}] CLIP 로딩 완료")

    print(f"[{_ts()}] ResNet-50 feature extractor 로딩 중...")
    extractor = ResNet50Extractor(device=args.device, batch_size=64)
    print(f"[{_ts()}] ResNet-50 로딩 완료")

    # ---------- 선택기 ----------
    methods = ["uniform", "p1"]
    p1_sampler = KeyframeSampler(n_frames=args.n_frames)

    p2_sampler = None
    if args.model_path and os.path.isfile(args.model_path):
        p2_sampler = KeyframeSampler(n_frames=args.n_frames,
                                     model_path=args.model_path,
                                     device=args.device)
        methods.append("p2")

    pgl_sampler = None
    # anomaly-retrained 체크포인트 우선, 없으면 원본 사용
    _pgl_path = (args.pglsum_anomaly_path
                 if args.pglsum_anomaly_path and os.path.isfile(args.pglsum_anomaly_path)
                 else args.pglsum_model_path)
    if _pgl_path and os.path.isfile(_pgl_path):
        from pipeline.sampler import PGLSumSampler
        pgl_sampler = PGLSumSampler(model_path=_pgl_path,
                                     n_frames=args.n_frames,
                                     input_size=args.pglsum_input_size,
                                     device=args.device,
                                     use_clip=args.pglsum_clip,
                                     clip_weight=args.pglsum_clip_weight)
        tag = "(anomaly-retrained)" if _pgl_path == args.pglsum_anomaly_path else ""
        clip_tag = "+CLIP" if args.pglsum_clip else ""
        print(f"[{_ts()}] PGL-SUM {tag}{clip_tag}: {_pgl_path}")
        methods.append("pgl")

    print(f"[{_ts()}] 비교 방법: {methods}")

    # ---------- Temporal Annotation ----------
    ann = {}
    if args.annotation_path and os.path.isfile(args.annotation_path):
        raw = load_annotations(args.annotation_path)
        ann = {k: v for k, v in raw.items()
               if any(k.startswith(c) for c in _TARGET_ANOMALY_CLASSES)}
        print(f"[{_ts()}] Temporal annotation: {len(ann)}개 영상")

    # ---------- 영상 목록 & 샘플링 ----------
    all_videos = collect_videos(args.video_dir)
    anom_vids  = [(p, g, s) for p, g, s in all_videos if g == 1]
    norm_vids  = [(p, g, s) for p, g, s in all_videos if g == 0]
    random.shuffle(anom_vids)
    random.shuffle(norm_vids)

    n_each   = args.n_videos // 2
    selected = anom_vids[:n_each] + norm_vids[:n_each]
    random.shuffle(selected)
    print(f"[{_ts()}] 평가 영상: {len(selected)}개 "
          f"(anomaly={min(n_each, len(anom_vids))}, "
          f"normal={min(n_each, len(norm_vids))})\n")

    # ---------- 결과 누적 ----------
    scores = {m: {"anom": [], "normal": []} for m in methods}
    gaps   = {m: [] for m in methods}
    coverage = {m: [] for m in methods}  # annotated 영상만

    t_start = time.time()

    for vi, (vpath, gt, stem) in enumerate(selected):
        label_str = "anomaly" if gt == 1 else "normal "

        # 프레임 디코딩
        frames, frame_indices = decode_video_frames(vpath, max_frames=args.max_frames)
        if len(frames) < args.n_frames:
            print(f"[{_ts()}] [{vi+1}/{len(selected)}] SKIP (프레임 부족 {len(frames)}): {stem}")
            continue

        # ResNet-50 feature 추출
        features = extractor.extract_from_frames(frames)  # (T, 2048)

        dummy_clip = [None] * len(frames)

        def run_selector(sampler_obj, local_idx_override=None, pass_frames=False):
            if local_idx_override is not None:
                return local_idx_override
            clip_arg = frames if pass_frames else dummy_clip
            c = {"clip": clip_arg, "features": features}
            sampler_obj.sample(c)
            return c["selected_indices"]

        # 각 선택기 실행
        sel = {}
        sel["uniform"] = uniform_select(len(frames), args.n_frames)
        sel["p1"]      = run_selector(p1_sampler)
        if p2_sampler:
            sel["p2"]  = run_selector(p2_sampler)
        if pgl_sampler:
            # PGL-SUM에는 실제 프레임 전달 (CLIP use_clip=True일 때 사용)
            sel["pgl"] = run_selector(pgl_sampler, pass_frames=True)

        bucket = "anom" if gt == 1 else "normal"

        for m in methods:
            if m not in sel:
                continue
            a_score, n_score = compute_clip_scores(
                clip_model, preprocess, frames, sel[m],
                anomaly_feats, normal_feats, args.device
            )
            scores[m][bucket].append(a_score)
            gaps[m].append(a_score - n_score)

            # Temporal Coverage (annotation 있는 영상만)
            if stem in ann:
                cov = temporal_coverage(sel[m], frame_indices, ann[stem])
                coverage[m].append(cov)

        # 진행 로그
        elapsed = time.time() - t_start
        pct = (vi + 1) / len(selected)
        eta = elapsed / pct * (1 - pct) if pct > 0 else 0
        gap_str = "  ".join(
            f"{m}={np.mean(gaps[m]):.4f}" for m in methods if gaps[m]
        )
        cov_tag = ""
        if stem in ann and coverage["p1"]:
            cov_vals = "  ".join(
                f"{m}={np.mean(coverage[m])*100:.1f}%" for m in methods if coverage[m]
            )
            cov_tag = f"  cov:[{cov_vals}]"
        print(f"[{_ts()}] [{vi+1}/{len(selected)}] {label_str}  {stem}")
        print(f"          gap:[{gap_str}]{cov_tag}"
              f"  ETA {int(eta//60)}:{int(eta%60):02d}")

    # ---------- 최종 결과 ----------
    label_map = {"uniform": "Uniform", "p1": "Phase 1", "p2": "Phase 2", "pgl": "PGL-SUM"}
    col_w = 12
    total_w = 28 + col_w * len(methods)

    print(f"\n{'='*total_w}")
    header = f"{'지표':<28}" + "".join(f"{label_map.get(m,m):>{col_w}}" for m in methods)
    print(header)
    print(f"{'-'*total_w}")

    def row(label, vals_dict):
        line = f"{label:<28}"
        for m in methods:
            v = vals_dict.get(m)
            line += f"{(f'{v:.4f}' if v is not None else 'N/A'):>{col_w}}"
        return line

    anom_sc = {m: float(np.mean(scores[m]["anom"]))   if scores[m]["anom"]   else None for m in methods}
    norm_sc = {m: float(np.mean(scores[m]["normal"])) if scores[m]["normal"] else None for m in methods}
    disc_gap= {m: float(np.mean(gaps[m]))             if gaps[m]             else None for m in methods}
    cov_avg = {m: float(np.mean(coverage[m]))         if coverage[m]         else None for m in methods}

    print(row("Anomaly CLIP Score (↑)",  anom_sc))
    print(row("Normal  CLIP Score (↓)",  norm_sc))
    print(row("Discrimination Gap (↑)",  disc_gap))
    if any(v is not None for v in cov_avg.values()):
        print(row("Temporal Coverage (↑)",  cov_avg))
    print(f"{'='*total_w}")

    if any(v is not None for v in cov_avg.values()):
        print(f"\n* Temporal Coverage는 annotation 있는 {len(coverage['p1'])}개 영상 기준")
    print("* Discrimination Gap: anomaly 영상 - normal 영상 CLIP 유사도 차이")
    print("  clip 단위와 달리 영상 전체에서 8프레임을 고르므로 실질적인 차이가 나타남")

    # 저장
    result = {
        "n_videos": len(selected),
        "max_frames_per_video": args.max_frames,
        "n_frames_selected": args.n_frames,
        "methods": {
            m: {
                "anomaly_clip_score":  anom_sc.get(m),
                "normal_clip_score":   norm_sc.get(m),
                "discrimination_gap":  disc_gap.get(m),
                "temporal_coverage":   cov_avg.get(m),
            }
            for m in methods
        }
    }
    os.makedirs("outputs", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="영상 단위 keyframe selector 평가")
    parser.add_argument("--video_dir",           default="videos")
    parser.add_argument("--annotation_path",     default="Data/annotations/temporal_anomaly.txt")
    parser.add_argument("--pglsum_model_path",   default=None)
    parser.add_argument("--pglsum_anomaly_path", default=None,
                        help="anomaly 재학습 PGL-SUM .pth. 제공 시 원본 대신 사용.")
    parser.add_argument("--pglsum_clip",         action="store_true",
                        help="PGL-SUM에 CLIP anomaly score 추가 (use_clip=True)")
    parser.add_argument("--pglsum_clip_weight",  type=float, default=0.5)
    parser.add_argument("--pglsum_input_size",   type=int, default=2048)
    parser.add_argument("--model_path",          default=None,
                        help="Phase 2 BiGRU .pth 경로. 없으면 skip.")
    parser.add_argument("--n_videos",            type=int, default=40,
                        help="평가할 총 영상 수 (anomaly N/2 + normal N/2)")
    parser.add_argument("--max_frames",          type=int, default=500,
                        help="영상당 최대 디코딩 프레임 수 (메모리/속도 조절)")
    parser.add_argument("--n_frames",            type=int, default=8,
                        help="선택기가 고를 프레임 수")
    parser.add_argument("--output_path",         default="outputs/eval_video_level_result.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args)
