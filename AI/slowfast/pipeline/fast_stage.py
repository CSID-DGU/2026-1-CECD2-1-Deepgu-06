from models.fast.infer import FastViolenceScorer


def score_clips_fast(clips, fast_model_config, clip_config, return_features=False):
    scorer = FastViolenceScorer(fast_model_config)

    if return_features:
        scores, features = scorer.score_clips(clips, clip_config, return_features=True)
    else:
        scores = scorer.score_clips(clips, clip_config)
        features = [None] * len(scores)

    results = []
    for clip, score, feat in zip(clips, scores, features):
        item = {
            "clip_id": clip["clip_id"],
            "start_frame": clip["start_frame"],
            "end_frame": clip["end_frame"],
            "start_time": float(clip["start_time"]),
            "end_time": float(clip["end_time"]),
            "frames": clip["frames"],
            "fighting_prob": float(score),
        }
        if feat is not None:
            item["x3d_features"] = feat  # np.ndarray (T'=13, 192)
        results.append(item)
    return results
