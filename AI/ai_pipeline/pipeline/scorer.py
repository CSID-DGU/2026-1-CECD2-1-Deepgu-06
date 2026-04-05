import numpy as np

ANOMALY_CLASSES = ["fighting"]


def compute_scores(prob_dict):
    """
    prob_dict: {"walking": 0.6, ...}
    """

    anomaly_score = sum(prob_dict.get(c, 0) for c in ANOMALY_CLASSES)
    uncertainty_score = 1 - max(prob_dict.values())

    final_score = 0.7 * anomaly_score + 0.3 * uncertainty_score

    return {
        "anomaly_score": anomaly_score,
        "uncertainty_score": uncertainty_score,
        "final_score": final_score
    }