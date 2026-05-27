def fuse_scores(scored_clips, vlm_outputs, fusion_config):
    fast_weight = float(fusion_config.get("fast_weight", 0.5))
    vlm_weight = float(fusion_config.get("vlm_weight", 0.5))
    only_when_called = bool(fusion_config.get("only_when_vlm_called", True))
    suppress_only = bool(fusion_config.get("suppress_only", False))
    suppression_cfg = fusion_config.get("suppression_bound", {})
    suppression_enabled = bool(suppression_cfg.get("enabled", False))
    max_drop = float(suppression_cfg.get("max_drop", 0.08))
    protect_above = suppression_cfg.get("protect_above_score")
    protect_floor = suppression_cfg.get("protect_floor_score")
    protect_above = None if protect_above is None else float(protect_above)
    protect_floor = None if protect_floor is None else float(protect_floor)
    vlm_outputs = vlm_outputs or {}

    fused = []
    for item in scored_clips:
        cloned = dict(item)
        clip_id = int(item["clip_id"])
        fast_score = float(item["fighting_prob"])
        if clip_id in vlm_outputs:
            vlm_score = float(vlm_outputs[clip_id]["score"])
            weighted_score = (fast_weight * fast_score) + (vlm_weight * vlm_score)
            if suppress_only:
                # Never raise score above fast_score; only allow suppression
                final_score = min(fast_score, weighted_score)
            else:
                final_score = weighted_score
            if suppression_enabled and final_score < fast_score:
                bounded_floor = fast_score - max_drop
                if protect_above is not None and protect_floor is not None and fast_score >= protect_above:
                    bounded_floor = max(bounded_floor, protect_floor)
                final_score = max(final_score, bounded_floor)
            cloned["vlm_called"] = True
            cloned["vlm_score"] = vlm_score
            cloned["weighted_score"] = float(weighted_score)
        else:
            final_score = fast_score
            cloned["vlm_called"] = False
            cloned["vlm_score"] = None
            cloned["weighted_score"] = None
            if not only_when_called:
                final_score = (fast_weight * fast_score) + (vlm_weight * fast_score)
        cloned["final_score"] = float(max(0.0, min(1.0, final_score)))
        fused.append(cloned)
    return fused
