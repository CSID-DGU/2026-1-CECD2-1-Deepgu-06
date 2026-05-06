import math


def binary_entropy(probability, eps=1e-6):
    p = min(max(float(probability), eps), 1.0 - eps)
    return float(-(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)))


def local_variance(values, index, window_size):
    if not values:
        return 0.0
    radius = max(int(window_size) // 2, 0)
    left = max(0, index - radius)
    right = min(len(values), index + radius + 1)
    window = values[left:right]
    if len(window) <= 1:
        return 0.0
    mean = sum(window) / len(window)
    return float(sum((item - mean) ** 2 for item in window) / len(window))


def attach_uncertainty(
    scored_clips,
    score_key="fighting_prob",
    alpha_entropy=0.7,
    alpha_variance=0.3,
    variance_window=5,
):
    probabilities = [float(item[score_key]) for item in scored_clips]
    updated = []
    for index, item in enumerate(scored_clips):
        entropy = binary_entropy(item[score_key])
        variance = local_variance(probabilities, index, variance_window)
        uncertainty = (alpha_entropy * entropy) + (alpha_variance * variance)
        cloned = dict(item)
        cloned["entropy"] = float(entropy)
        cloned["local_variance"] = float(variance)
        cloned["uncertainty"] = float(uncertainty)
        updated.append(cloned)
    return updated
