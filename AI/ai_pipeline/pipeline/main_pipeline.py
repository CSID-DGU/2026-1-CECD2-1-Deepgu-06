from pipeline.clip_generator import generate_clips, save_clip
from models.tsm.inference import DummyTSM
from pipeline.scorer import compute_scores
from pipeline.filter import filter_clips
from pipeline.sampler import sample_from_candidates
from models.vlm.inference import InternVL

import os
import json

def run_pipeline(video_path, clip_dir):

    clips = generate_clips(video_path, clip_dir, save=False)

    tsm = DummyTSM()
    vlm = InternVL()

    results = []

    for i, clip in enumerate(clips):
        prob = tsm.predict(clip)
        scores = compute_scores(prob)

        results.append({
            "clip_id": i,
            "clip": clip,  # 중요 (sampling용)
            "prob": prob,
            "scores": scores
        })

    # filtering
    candidates = filter_clips(results, threshold=0.4)

    # frame sampling
    candidates = sample_from_candidates(candidates, num_samples=4)
    
    # VLM inference
    for c in candidates:
        frames = c["sampled_frames"]
        response = vlm.predict(frames)
        c["vlm_output"] = response

    # clip 저장
    save_dir = "/home/deepgu/test/data/candidate_clips"
    os.makedirs(save_dir, exist_ok=True)

    for c in candidates:
        clip_id = c["clip_id"]
        clip_frames = c["clip"]

        save_path = os.path.join(save_dir, f"clip_{clip_id}.mp4")
        save_clip(clip_frames, save_path)

        # 경로도 같이 저장
        c["video_path"] = save_path

    # JSON 저장
    output_dir = "/home/deepgu/test/outputs"
    os.makedirs(output_dir, exist_ok=True)

    output_data = []

    for c in candidates:
        output_data.append({
            "clip_id": c["clip_id"],
            "scores": c["scores"],
            "vlm_output": c["vlm_output"],
            "video_path": c["video_path"]
        })

    json_path = os.path.join(output_dir, "results.json")

    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f" 결과 저장 완료: {json_path}")

    return candidates
