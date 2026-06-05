"""
캐시 score(service_scope_clip_scores.json) vs 라이브 파이프라인 score 직접 비교.
- clip 개수 일치 여부
- start_frame 정렬 후 per-clip score 차이
원인 규명: (a) 점수 자체 skew  vs  (b) clip 구성/개수 차이.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.clip_generator import build_sliding_clips
from pipeline.fast_stage import score_clips_fast
from utils.config import load_config
from utils.video import load_video_frames

CACHE = PROJECT_ROOT / "outputs/eval/service_scope_clip_scores.json"
CSV = PROJECT_ROOT / "data/manifests/cctv_x3d_s_overlap030/testing_clips_service_scope.csv"
CONFIG = PROJECT_ROOT / "configs/eval_event_overlap030_focal_FASTONLY.yaml"
DATASET = Path("/home/deepgu/test/cctv/dataset/CCTV_DATA/testing")

VIDS = sys.argv[1:] or ["fight_0002", "fight_0005", "fight_0060", "fight_0400", "fight_0920"]

cache = json.load(open(CACHE))
cache_score = {r["clip_id"]: r["fighting_prob"] for r in cache["rows"]}
df = pd.read_csv(CSV)
config = load_config(str(CONFIG))

print(f"{'video':<12} {'cacheN':>6} {'liveN':>6} {'matched':>7} {'maxΔ':>7} {'meanΔ':>7} {'>0.05':>6} {'flip@.48':>8}")
for vid in VIDS:
    sub = df[df.video_id == vid].sort_values("start_frame")
    if sub.empty:
        print(f"{vid:<12}  not in CSV"); continue
    cache_by_sf = {int(r.start_frame): cache_score.get(r.clip_id) for r in sub.itertuples()}

    frames, fps = load_video_frames(str(DATASET / f"{vid}.mpeg"))
    clips = build_sliding_clips(frames=frames, fps=fps,
                                temporal_window_sec=float(config["clip"]["temporal_window_sec"]),
                                stride_sec=float(config["clip"]["stride_sec"]))
    scored = score_clips_fast(clips, config["fast_model"], config["clip"])
    live_by_sf = {int(s["start_frame"]): float(s["fighting_prob"]) for s in scored}

    common = sorted(set(cache_by_sf) & set(live_by_sf))
    diffs = [abs(cache_by_sf[sf] - live_by_sf[sf]) for sf in common if cache_by_sf[sf] is not None]
    flips = sum(1 for sf in common if cache_by_sf[sf] is not None
                and (cache_by_sf[sf] >= 0.48) != (live_by_sf[sf] >= 0.48))
    big = sum(1 for d in diffs if d > 0.05)
    maxd = max(diffs) if diffs else 0.0
    meand = float(np.mean(diffs)) if diffs else 0.0
    print(f"{vid:<12} {len(sub):>6} {len(scored):>6} {len(common):>7} {maxd:>7.3f} {meand:>7.3f} {big:>6} {flips:>8}")
