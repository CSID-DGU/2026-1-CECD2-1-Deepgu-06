import torch


def _sum_probs(probs, indices):
    return sum(probs[i] for i in indices if i < len(probs))


def _topk(probs, k=5):
    ranked = sorted(enumerate(probs), key=lambda item: item[1], reverse=True)
    return [
        {"class_index": idx, "score": float(score)}
        for idx, score in ranked[:k]
    ]


def _to_float_score(probs):
    if isinstance(probs, torch.Tensor):
        if probs.numel() == 1:
            return float(probs.item())
        probs = probs.tolist()

    if isinstance(probs, (float, int)):
        return float(probs)

    if isinstance(probs, list) and len(probs) == 1:
        return float(probs[0])

    return None


def compute_candidate_scores(probs, label_map=None):
    binary_score = _to_float_score(probs)
    if binary_score is not None:
        return {
            "fight_candidate_score": binary_score,
            "fight_prob": binary_score,
            "uncertainty": float(1.0 - abs(binary_score - 0.5) * 2.0),
        }

    if isinstance(probs, torch.Tensor):
        probs = probs.tolist()

    if not label_map:
        raise ValueError("label_map is required for multi-class heuristic scoring")

    fight_score = _sum_probs(probs, label_map["fight_indices"])
    attack_score = _sum_probs(probs, label_map["attack_indices"])
    fall_score = _sum_probs(probs, label_map["fall_indices"])
    abnormal_score = _sum_probs(probs, label_map["abnormal_indices"])
    uncertainty_score = 1 - max(probs)

    fight_candidate_score = (
        0.45 * fight_score +
        0.30 * attack_score +
        0.15 * abnormal_score +
        0.10 * uncertainty_score
    )
    fall_candidate_score = (
        0.65 * fall_score +
        0.20 * abnormal_score +
        0.15 * uncertainty_score
    )

    return {
        "fight_candidate_score": float(fight_candidate_score),
        "fall_candidate_score": float(fall_candidate_score),
        "fight_score": float(fight_score),
        "attack_score": float(attack_score),
        "fall_score": float(fall_score),
        "abnormal_score": float(abnormal_score),
        "uncertainty": float(uncertainty_score),
        "top5": _topk(probs, k=5)
    }
