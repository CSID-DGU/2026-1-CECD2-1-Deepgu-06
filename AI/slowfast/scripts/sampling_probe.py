"""샘플링 병목 검증: TP 손실 4개 이벤트에 대해 여러 프레임 샘플링 전략으로
Qwen3-VL(v4 sub-question)에 재질의하고, 5지표가 False→True로 바뀌는지 확인.

절차:
  1) v4 config로 파이프라인을 1회 패스(Bedrock invoke는 더미)하여 대상 이벤트의
     event 객체 + all_frames + fps를 비용 0으로 캡처.
  2) 각 이벤트에 대해 4개 전략으로 프레임 인덱스를 재선정:
       - peak8    : 현행(extract_event_frames n=8, peak 편향)  [baseline]
       - peak16   : extract_event_frames n=16
       - uniform16: 이벤트 구간 전체 균등 16장 (peak 편향 제거)
  3) 각 전략 프레임셋을 실제 Qwen v4로 질의 → sub-question 5지표 + label 기록.
  4) 전략별 몽타주 PNG 저장 + 요약 JSON.

사용: python3 scripts/sampling_probe.py
"""
import json
import sys
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pipeline.event_vlm_filter as evf
import models.vlm.infer as infer
from models.vlm.infer import BedrockVLMProvider, _V4_SUBQ_KEYS
from models.vlm.parser import parse_vlm_response
from models.vlm.prompts import build_event_fight_prompt_v4
from pipeline.main_pipeline import run_single_video_pipeline
from utils.config import load_config

CONFIG = "configs/eval_event_v4_start040_bedrock_qwen.yaml"
DATASET = Path("/home/deepgu/test/cctv/dataset/CCTV_DATA/testing")
OUT = PROJECT_ROOT / "outputs/eval/sampling_probe"
TARGETS = ["fight_0048", "fight_0134", "fight_0350", "fight_0990"]
WANT = {
    "fight_0048": [(6510, 8219)],
    "fight_0134": [(1980, 3059)],
    "fight_0350": [(2970, 3809)],
    "fight_0990": [(75, 1074), (2975, 3224)],
}

_ORIG_INVOKE = BedrockVLMProvider.invoke      # 실제 Qwen 호출용 보관
_orig_extract = evf.extract_event_frames
_cur = {"id": None}
_captured = []  # {vid, event, frames(all), fps, peak_margin}


def _dummy_invoke(self, clip_record, prompt):
    return '{"label":"non_fight","confidence":0.5,"reasoning":"dummy"}'


def _patched_extract(all_frames, event, n_frames, fps=30.0, peak_margin_sec=4.0):
    frames = _orig_extract(all_frames, event, n_frames, fps=fps, peak_margin_sec=peak_margin_sec)
    vid = _cur["id"]
    s, e = int(event["start_frame"]), int(event["end_frame"])
    for ws, we in WANT.get(vid, []):
        if abs(s - ws) <= 30 and abs(e - we) <= 30:
            _captured.append({
                "vid": vid, "event": deepcopy(event),
                "all_frames": all_frames, "fps": fps, "peak_margin": peak_margin_sec,
            })
    return frames


def _uniform_indices(lo, hi, k, n_total):
    lo, hi = max(0, lo), min(n_total - 1, hi)
    if lo >= hi:
        return [lo] * k
    total = hi - lo + 1
    if total <= k:
        return list(range(lo, hi + 1))
    return [lo + int(round(i * (total - 1) / (k - 1))) for i in range(k)]


def _peak_window(ev, fps, peak_margin, n_total):
    """extract_event_frames와 동일한 방식으로 peak 윈도 [p_start, p_end] 추정."""
    ev_start = max(0, int(ev["start_frame"]))
    ev_end = min(n_total - 1, int(ev["end_frame"]))
    margin = int(peak_margin * fps)
    peak_clip_id = ev.get("peak_clip_id")
    clip_ids = ev.get("clip_ids", [])
    if peak_clip_id is not None and clip_ids:
        peak_pos = clip_ids.index(peak_clip_id) if peak_clip_id in clip_ids else len(clip_ids) // 2
        ratio = peak_pos / max(len(clip_ids) - 1, 1)
        peak_frame = ev_start + int(ratio * (ev_end - ev_start))
    else:
        peak_frame = (ev_start + ev_end) // 2
    return max(ev_start, peak_frame - margin), min(ev_end, peak_frame + margin), ev_start, ev_end


def _hybrid_indices(ev, fps, peak_margin, n_total, k):
    """k장을 peak 윈도 50% + 이벤트 전체 균등 50%로 분배(중복 제거 후 k로 보정)."""
    p_start, p_end, ev_start, ev_end = _peak_window(ev, fps, peak_margin, n_total)
    n_peak = k // 2
    n_full = k - n_peak
    idx = set(_uniform_indices(p_start, p_end, n_peak, n_total))
    idx |= set(_uniform_indices(ev_start, ev_end, n_full, n_total))
    idx = sorted(idx)
    # 부족하면 전체 구간 균등으로 채움
    if len(idx) < k:
        idx = sorted(set(idx) | set(_uniform_indices(ev_start, ev_end, k, n_total)))
    return idx[:k]


def _strategy_frames(cap, strategy):
    af, ev, fps, pm = cap["all_frames"], cap["event"], cap["fps"], cap["peak_margin"]
    s, e = int(ev["start_frame"]), int(ev["end_frame"])
    if strategy.startswith("peak"):
        return _orig_extract(af, ev, int(strategy[4:]), fps=fps, peak_margin_sec=pm)
    if strategy.startswith("uniform"):
        idx = _uniform_indices(s, e, int(strategy[7:]), len(af))
    elif strategy.startswith("hybrid"):
        idx = _hybrid_indices(ev, fps, pm, len(af), int(strategy[6:]))
    else:
        raise ValueError(strategy)
    return [af[i] for i in idx]


def _save_montage(frames, path, cols=8):
    tiles = []
    for i, f in enumerate(frames):
        bgr = f if f.ndim == 3 else cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        t = cv2.resize(bgr, (200, 200))
        cv2.putText(t, f"#{i}", (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        tiles.append(t)
    rows = []
    for r in range(0, len(tiles), cols):
        chunk = tiles[r:r + cols]
        while len(chunk) < cols:
            chunk.append(np.zeros((200, 200, 3), np.uint8))
        rows.append(cv2.hconcat(chunk))
    cv2.imwrite(str(path), cv2.vconcat(rows))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config(CONFIG)
    config.setdefault("outputs", {}).update(
        save_run_artifacts=False, save_event_media=False, save_clip_manifest=False)

    # --- 1) 캡처 (더미 invoke로 비용 0) ---
    evf.extract_event_frames = _patched_extract
    infer.BedrockVLMProvider.invoke = _dummy_invoke
    for vid in TARGETS:
        vpath = DATASET / f"{vid}.mpeg"
        if not vpath.exists():
            print(f"[skip] {vid} (no file)")
            continue
        _cur["id"] = vid
        print(f"[capture] {vid}")
        run_single_video_pipeline(str(vpath), deepcopy(config), run_name=f"sp_{vid}", verbose=False)
    print(f"captured {len(_captured)} events")

    # --- 2~3) 실제 Qwen 질의 ---
    infer.BedrockVLMProvider.invoke = _ORIG_INVOKE   # 복원
    provider = BedrockVLMProvider(config["vlm"])
    # uniform32 등 다수 이미지 추론은 느림 → read timeout 상향, botocore 자체 재시도 끔
    import boto3
    from botocore.config import Config as _BotoConfig
    provider._client = boto3.client(
        "bedrock-runtime", region_name=provider.region,
        config=_BotoConfig(read_timeout=600, connect_timeout=30, retries={"max_attempts": 0}))
    strategies = ["peak8", "uniform12", "hybrid12"]
    out_json = OUT / "sampling_probe_12.json"
    results = []
    for cap in _captured:
        ev = cap["event"]
        tag = f"{cap['vid']}_{int(ev['start_frame'])}-{int(ev['end_frame'])}"
        dur = ev.get("duration_sec", (ev["end_frame"] - ev["start_frame"] + 1) / cap["fps"])
        peak = ev.get("fast_peak_score", ev.get("peak_score", 0.0))
        print(f"\n=== {tag}  ({dur:.1f}s, peak{peak:.2f}) ===")
        rec = {"event": tag, "duration_sec": round(float(dur), 1),
               "peak": round(float(peak), 3), "strategies": {}}
        for st in strategies:
            frames = _strategy_frames(cap, st)
            provider.sampled_frames = len(frames)  # 재샘플 없이 그대로 입력
            prompt = build_event_fight_prompt_v4(num_frames=len(frames), duration_sec=float(dur))
            _save_montage(frames, OUT / f"{tag}__{st}.png")
            try:
                raw = _ORIG_INVOKE(provider, {"frames": frames}, prompt)
                p = parse_vlm_response(raw)
            except Exception as exc:  # noqa: BLE001
                print(f"  {st:10} n={len(frames):2}  ERROR: {type(exc).__name__}: {str(exc)[:80]}")
                rec["strategies"][st] = {"n_frames": len(frames), "error": f"{type(exc).__name__}: {str(exc)[:120]}"}
                continue
            subq = {}
            for k in _V4_SUBQ_KEYS:
                v = p.get(k)
                subq[k] = (v is True) if isinstance(v, bool) else \
                    (str(v).strip().lower() in {"true", "yes", "1"} if v is not None else None)
            rec["strategies"][st] = {
                "n_frames": len(frames), "label": p.get("label"),
                "confidence": p.get("confidence"), "subq": subq,
                "any_true": any(subq.get(k) for k in _V4_SUBQ_KEYS),
                "reasoning": p.get("reasoning", "")[:160],
            }
            flags = "".join("T" if subq[k] else ("F" if subq[k] is False else "?") for k in _V4_SUBQ_KEYS)
            print(f"  {st:10} n={len(frames):2}  AGG/THR/CRWD/AFT/PHYS={flags}  -> {p.get('label')} ({p.get('confidence')})")
        results.append(rec)

    json.dump(results, open(out_json, "w"), ensure_ascii=False, indent=1)
    print(f"\n[output] {out_json}")


if __name__ == "__main__":
    main()
