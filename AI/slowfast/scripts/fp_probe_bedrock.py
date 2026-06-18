"""
FP Top-N probe (Bedrock VLM). 목적: 모델의 FP 억제 능력 + reasoning 시각검증.
- operating point: start=0.40/end=0.36/split=0.40, smoothing on (연속 캐시 기반, GPU 불필요)
- FP = GT-centric IoU<0.10인 예측 이벤트
- peak band 다양성으로 선정 (>0.85:10, 0.70~0.85:5, 0.40~0.70:5)
- peak_score_max 자동 accept 무시, 전원 강제로 VLM 통과
- 각 샘플: 6프레임(8→6 재샘플, InternVL 동일) 몽타주 저장 + VLM label/score/decision/reasoning
"""
import argparse
import glob
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.event_builder import build_events
from pipeline.event_vlm_filter import extract_event_frames
from models.vlm.infer import VLMRefiner
from models.vlm.parser import parse_vlm_response
from scripts.evaluate_event_batch import build_gt_events, interval_iou
from utils.video import load_video_frames

CACHE_DIR = PROJECT_ROOT / "outputs/cache/test_clip_scores_service61"
GT_JSON = Path("/home/deepgu/test/cctv/dataset/ground-truth.json")
DATASET = Path("/home/deepgu/test/cctv/dataset/CCTV_DATA/testing")

TH = {"start_score": 0.40, "end_score": 0.36, "min_event_duration_sec": 0.5, "mean_score_threshold": 0.0,
      "split": {"enabled": True, "score_threshold": 0.40, "min_consecutive_clips": 2},
      "score_smoothing": {"enabled": True, "window_size": 3, "method": "moving_average"}}
IOU = 0.10
ACCEPT, REJECT = 0.65, 0.35
BANDS = [(">0.85", 0.85, 1.01, 10), ("0.70-0.85", 0.70, 0.85, 5), ("0.40-0.70", 0.40, 0.70, 5)]


def detect_model_id(region):
    import boto3
    rt = boto3.client("bedrock-runtime", region_name=region)
    img = (np.random.rand(64, 64, 3) * 255).astype("uint8")
    _, buf = cv2.imencode(".jpg", img)
    content = [{"image": {"format": "jpeg", "source": {"bytes": buf.tobytes()}}}, {"text": "reply ok"}]
    for mid in ["us.anthropic.claude-sonnet-4-6", "global.anthropic.claude-sonnet-4-6",
                "anthropic.claude-sonnet-4-6", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"]:
        try:
            rt.converse(modelId=mid, messages=[{"role": "user", "content": content}],
                        inferenceConfig={"maxTokens": 20, "temperature": 0.0})
            return mid
        except Exception as e:
            print(f"  [id試] {mid} 실패: {type(e).__name__}: {str(e)[:70]}")
    raise RuntimeError("사용 가능한 Sonnet 모델 ID를 못 찾음")


def gt_relation(ev, gts, fps):
    """참고용: FP와 가장 가까운 GT의 관계."""
    best_iou, best_gap = 0.0, 1e9
    for g in gts:
        i = interval_iou(ev["start_frame"], ev["end_frame"], g["start_frame"], g["end_frame"])
        best_iou = max(best_iou, i)
        gap = max(0, max(ev["start_frame"] - g["end_frame"], g["start_frame"] - ev["end_frame"])) / fps
        best_gap = min(best_gap, gap)
    if best_iou > 0:
        rel = "near-GT(overlap<0.10)"
    elif best_gap <= 2.0:
        rel = "near-GT(adjacent<=2s)"
    elif not gts:
        rel = "isolated(no-GT-in-video)"
    else:
        rel = "isolated"
    return rel, round(best_iou, 3), round(best_gap, 1)


def montage(frames6, path):
    tiles = [cv2.resize(f, (240, 240)) for f in frames6]
    while len(tiles) < 6:
        tiles.append(np.zeros((240, 240, 3), np.uint8))
    row = cv2.hconcat(tiles[:6])
    cv2.imwrite(str(path), row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--model-id", default=None, help="미지정 시 Sonnet 자동탐지")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--peak-lo", type=float, default=None, help="단일 밴드 하한 (지정 시 기본 3밴드 대신 사용)")
    ap.add_argument("--peak-hi", type=float, default=None)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--sort", choices=["asc", "desc"], default="desc", help="밴드 내 peak 정렬(asc=약한 신호 우선)")
    ap.add_argument("--tag", default="sonnet", help="출력 파일 태그")
    ap.add_argument("--mode", choices=["fp", "tp"], default="fp",
                    help="fp=GT 미매칭 이벤트(오탐), tp=GT 매칭 이벤트(정탐, recall 손실 측정)")
    args = ap.parse_args()

    global BANDS
    if args.peak_lo is not None:
        BANDS = [(f"{args.peak_lo}-{args.peak_hi}", args.peak_lo, args.peak_hi, args.n)]
    elif args.mode == "tp":
        BANDS = [("tp-all", 0.40, 1.01, args.n)]  # 저득점 TP 우선(--sort asc)으로 recall 취약점 측정
    FRAMES_DIR = PROJECT_ROOT / f"outputs/eval/fp_probe_frames_{args.tag}"
    OUT_JSON = PROJECT_ROOT / f"outputs/eval/fp_probe_{args.tag}.json"
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    caches = {json.load(open(f))["video_id"]: json.load(open(f)) for f in glob.glob(str(CACHE_DIR / "*.json"))}
    gtdb = json.load(open(GT_JSON))["database"]

    # 1) 전 영상 이벤트 수집 (FP/TP 플래그)
    pool_all = []
    for vid in sorted(caches):
        fps = float(caches[vid]["fps"])
        clips = [{"clip_id": int(c["clip_id"]), "start_frame": int(c["start_frame"]),
                  "end_frame": int(c["end_frame"]), "final_score": float(c["score"])} for c in caches[vid]["clips"]]
        events, _ = build_events(clips, TH, fps)
        gts = build_gt_events(gtdb[vid])
        for ev in events:
            is_tp = any(interval_iou(ev["start_frame"], ev["end_frame"], g["start_frame"], g["end_frame"]) >= IOU for g in gts)
            rel, iou0, gap = gt_relation(ev, gts, fps)
            pool_all.append({"video_id": vid, "fps": fps, "event": ev, "peak": float(ev["peak_score"]),
                             "is_tp": is_tp, "gt_relation": rel, "nearest_iou": iou0, "nearest_gap_sec": gap})
    want_tp = (args.mode == "tp")
    fps_all = [x for x in pool_all if x["is_tp"] == want_tp]
    print(f"[probe] mode={args.mode} 대상 이벤트 {len(fps_all)}개 (전체 {len(pool_all)}, operating point 0.40/0.36/split0.40)")

    # 2) band별 선정 (video 다양성: 같은 영상 ≤3, peak desc)
    rng = np.random.default_rng(args.seed)
    selected = []
    for name, lo, hi, k in BANDS:
        pool = [x for x in fps_all if lo <= x["peak"] < hi]
        pool.sort(key=lambda x: x["peak"] if args.sort == "asc" else -x["peak"])
        picked, per_vid = [], {}
        for x in pool:
            if per_vid.get(x["video_id"], 0) >= 3:
                continue
            picked.append(x); per_vid[x["video_id"]] = per_vid.get(x["video_id"], 0) + 1
            if len(picked) >= k:
                break
        if len(picked) < k:  # 다양성 제약으로 부족하면 채움
            for x in pool:
                if x not in picked:
                    picked.append(x)
                if len(picked) >= k:
                    break
        for x in picked:
            x["peak_band"] = name
        selected.extend(picked)
        print(f"  band {name}: pool {len(pool)} → 선정 {len(picked)}")

    # 3) VLM
    model_id = args.model_id or detect_model_id(args.region)
    print(f"[probe] 모델: {model_id}")
    refiner = VLMRefiner({"provider": "bedrock", "model_id": model_id, "region": args.region,
                          "max_tokens": 256, "sampled_frames": 8, "prompt_mode": "v3"})

    # 영상별로 프레임 1회 디코딩
    by_vid = {}
    for x in selected:
        by_vid.setdefault(x["video_id"], []).append(x)

    results = []
    for vid, items in by_vid.items():
        frames, fps = load_video_frames(str(DATASET / f"{vid}.mpeg"))
        for x in items:
            ev = x["event"]
            f8 = extract_event_frames(frames, ev, 8, fps=fps, peak_margin_sec=4.0)
            # 재샘플 없이 8장 그대로 모델에 입력 (provider sampled_frames=8와 일치)
            eid = f"{vid}_f{ev['start_frame']}-{ev['end_frame']}"
            mpath = FRAMES_DIR / f"{eid}.jpg"
            montage(f8, mpath)
            r = refiner.score_event_v3(f8, event_meta=ev)
            score = float(r["score"])  # fight 확률
            decision = "reject" if score <= REJECT else ("accept" if score >= ACCEPT else "uncertain")
            parsed = r.get("parsed", {})
            results.append({
                "event_id": eid, "video_id": vid, "peak_score": round(x["peak"], 3), "peak_band": x["peak_band"],
                "gt_relation": x["gt_relation"], "nearest_iou": x["nearest_iou"], "nearest_gap_sec": x["nearest_gap_sec"],
                "vlm_label": parsed.get("label"), "vlm_confidence": parsed.get("confidence"),
                "vlm_score": round(score, 3), "vlm_decision": decision,
                "vlm_reasoning": parsed.get("reasoning", "")[:200],
                "frames_path": str(mpath),
            })
            print(f"  {eid} peak={x['peak']:.2f}[{x['peak_band']}] → {decision} (score={score:.2f}) {parsed.get('reasoning','')[:60]}")

    # 4) 리포트
    OUT_JSON.write_text(json.dumps({"model_id": model_id, "operating_point": "0.40/0.36/split0.40",
                                    "n_fp_total": len(fps_all), "results": results}, ensure_ascii=False, indent=2))

    def rate(rows):
        n = len(rows); rej = sum(1 for r in rows if r["vlm_decision"] == "reject")
        return rej, n

    interp = "recall 손실(낮을수록 좋음)" if want_tp else "FP 제거(높을수록 좋음)"
    print("\n" + "=" * 70)
    print(f"{args.mode.upper()} probe 결과 (model={model_id}, 샘플 {len(results)}개) — reject = {interp}")
    print("=" * 70)
    rej, n = rate(results)
    print(f"전체 reject: {rej}/{n}")
    print("peak 밴드별 reject:")
    for name, *_ in BANDS:
        rows = [r for r in results if r["peak_band"] == name]
        rj, nn = rate(rows)
        print(f"  {name:>10}: {rj}/{nn} reject")
    print("GT-relation별(참고) reject:")
    for rel in sorted(set(r["gt_relation"] for r in results)):
        rows = [r for r in results if r["gt_relation"] == rel]
        rj, nn = rate(rows)
        print(f"  {rel:>24}: {rj}/{nn}")
    if not want_tp:
        hi = [r for r in results if r["peak_band"] == ">0.85"]
        hr, hn = rate(hi)
        if hn:
            print(f"\n[핵심] high-confidence FP(peak>0.85) reject {hr}/{hn} "
                  f"→ {'자동 accept(0.85)가 성능 제한 중' if hr >= hn*0.5 else 'VLM도 reject 약함 → 자동 accept 유지 무방'}")
    else:
        print(f"\n[핵심] TP false-reject {rej}/{n} = recall 손실. accept/uncertain {n-rej}/{n} 유지.")
    print(f"\n[몽타주] {FRAMES_DIR}  /  [결과] {OUT_JSON}")


if __name__ == "__main__":
    main()
