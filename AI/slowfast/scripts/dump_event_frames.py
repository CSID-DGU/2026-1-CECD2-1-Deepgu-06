"""TP 손실 이벤트의 '실제 Qwen 입력 8프레임'을 Bedrock 호출 없이 캡처/저장.

- 이벤트 구성은 fast model(GPU) 결정이라 VLM 응답과 무관 → provider.invoke를 더미로
  패치해 비용 0으로 동일 이벤트/동일 extract_event_frames 호출을 재현.
- extract_event_frames를 래핑해 호출 시점의 8프레임을 (video, start, end)별로 저장.
- 각 이벤트: 개별 프레임 + 가로 몽타주(2행 x 4열) PNG.

사용: python3 scripts/dump_event_frames.py
"""
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
from pipeline.main_pipeline import run_single_video_pipeline
from utils.config import load_config

CONFIG = "configs/eval_event_v4_start040_bedrock_qwen.yaml"
DATASET = Path("/home/deepgu/test/cctv/dataset/CCTV_DATA/testing")
OUT = PROJECT_ROOT / "outputs/eval/tp_lost_frames"
TARGETS = ["fight_0048", "fight_0134", "fight_0350", "fight_0990"]

# 검수 대상 이벤트(frame range, v3에서 잘린 TP). (video, start_frame, end_frame)
WANT = {
    "fight_0048": [(6510, 8219)],              # 57.1s, peak0.954
    "fight_0134": [(1980, 3059)],              # 36.0s, peak0.893
    "fight_0350": [(2970, 3809)],              # 28.0s, peak0.914
    "fight_0990": [(75, 1074), (2975, 3224)],  # 40.0s + 10.0s
}

_cur_video = {"id": None}
_orig_extract = evf.extract_event_frames


def _patched_extract(all_frames, event, n_frames, fps=30.0, peak_margin_sec=4.0):
    frames = _orig_extract(all_frames, event, n_frames, fps=fps, peak_margin_sec=peak_margin_sec)
    vid = _cur_video["id"]
    s, e = int(event["start_frame"]), int(event["end_frame"])
    # 대상 이벤트만 저장 (frame range 근접 매칭: ±30프레임)
    for ws, we in WANT.get(vid, []):
        if abs(s - ws) <= 30 and abs(e - we) <= 30:
            _save(vid, s, e, fps, frames, event)
    return frames


def _save(vid, s, e, fps, frames, event):
    tag = f"{vid}_{s}-{e}"
    d = OUT / tag
    d.mkdir(parents=True, exist_ok=True)
    tiles = []
    for i, f in enumerate(frames):
        bgr = f if f.ndim == 3 else cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(d / f"frame_{i}.png"), bgr)
        t = cv2.resize(bgr, (320, 320))
        cv2.putText(t, f"#{i}", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        tiles.append(t)
    while len(tiles) < 8:
        tiles.append(np.zeros((320, 320, 3), np.uint8))
    row1 = cv2.hconcat(tiles[:4])
    row2 = cv2.hconcat(tiles[4:8])
    cv2.imwrite(str(OUT / f"{tag}_montage.png"), cv2.vconcat([row1, row2]))
    dur = (e - s + 1) / fps
    print(f"  saved {tag}  ({s/fps:.1f}-{(e+1)/fps:.1f}s, {dur:.1f}s, {len(frames)} frames)")


def _dummy_invoke(self, clip_record, prompt):
    return '{"label":"non_fight","confidence":0.5,"reasoning":"dummy"}'


def main():
    evf.extract_event_frames = _patched_extract
    infer.BedrockVLMProvider.invoke = _dummy_invoke
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config(CONFIG)
    config.setdefault("outputs", {}).update(
        save_run_artifacts=False, save_event_media=False, save_clip_manifest=False)
    for vid in TARGETS:
        vpath = DATASET / f"{vid}.mpeg"
        if not vpath.exists():
            print(f"[skip] {vid} (no file)")
            continue
        _cur_video["id"] = vid
        print(f"[run] {vid}")
        run_single_video_pipeline(str(vpath), deepcopy(config), run_name=f"dump_{vid}", verbose=False)
    print(f"\n[output] {OUT}")


if __name__ == "__main__":
    main()
