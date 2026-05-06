def select_vlm_clips(scored_clips, router_config):
    low = float(router_config.get("prob_low", 0.2))
    high = float(router_config.get("prob_high", 0.8))
    uncertainty_threshold = float(router_config.get("uncertainty_threshold", 0.2))
    topk_ratio = float(router_config.get("topk_ratio", 0.15))
    use_topk = bool(router_config.get("use_topk", True))

    eligible = []
    for item in scored_clips:
        in_gray_zone = low < float(item["fighting_prob"]) < high
        if in_gray_zone and float(item["uncertainty"]) >= uncertainty_threshold:
            eligible.append(item["clip_id"])

    if use_topk and scored_clips:
        ranked = sorted(scored_clips, key=lambda item: float(item["uncertainty"]), reverse=True)
        count = max(1, int(len(ranked) * topk_ratio))
        eligible.extend(item["clip_id"] for item in ranked[:count])

    return sorted(set(eligible))
