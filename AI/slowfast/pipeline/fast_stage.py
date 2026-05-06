from models.fast.infer import FastViolenceScorer


def score_clips_fast(clips, fast_model_config, clip_config):
    scorer = FastViolenceScorer(fast_model_config)
    scores = scorer.score_clips(clips, clip_config)
    results = []
    for clip, score in zip(clips, scores):
        results.append(
            {
                "clip_id": clip["clip_id"],
                "start_frame": clip["start_frame"],
                "end_frame": clip["end_frame"],
                "start_time": float(clip["start_time"]),
                "end_time": float(clip["end_time"]),
                "frames": clip["frames"],
                "fighting_prob": float(score),
            }
        )
    return results
