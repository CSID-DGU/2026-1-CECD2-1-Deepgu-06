import torch

FIGHT_IDX = [259, 314, 345, 395]
ATTACK_IDX = [150, 152, 302]
FALL_IDX = [122]
ABNORMAL_IDX = [79, 149]


def compute_scores(probs):

    if isinstance(probs, torch.Tensor):
        probs = probs.tolist()

    # score 계산
    fight_score = sum(probs[i] for i in FIGHT_IDX if i < len(probs))
    attack_score = sum(probs[i] for i in ATTACK_IDX if i < len(probs))
    fall_score = sum(probs[i] for i in FALL_IDX if i < len(probs))
    abnormal_score = sum(probs[i] for i in ABNORMAL_IDX if i < len(probs))

    # uncertainty
    max_prob = max(probs)
    uncertainty_score = 1 - max_prob

    # final score
    final_score = (
        0.5 * uncertainty_score +
        0.3 * fight_score +
        0.1 * attack_score +
        0.1 * fall_score
    )

    return {
        "final_score": final_score,
        "uncertainty": uncertainty_score,
        "fight": fight_score,
        "attack": attack_score,
        "fall": fall_score,
        "abnormal": abnormal_score
    }