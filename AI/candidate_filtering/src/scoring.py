from config import W_MOTION, W_INTERACTION, W_TRAJECTORY


def normalize(values):

    if len(values) == 0:
        return values

    min_v = min(values)
    max_v = max(values)

    if max_v - min_v == 0:
        return [0] * len(values)

    return [(v - min_v) / (max_v - min_v) for v in values]


def smooth_signal(signal, window=3):

    smoothed = []

    for i in range(len(signal)):

        start = max(0, i - window)
        end = min(len(signal), i + window)

        smoothed.append(sum(signal[start:end]) / (end - start))

    return smoothed


def persistence_filter(scores, threshold=0.2, min_frames=3):

    filtered = [0] * len(scores)

    count = 0

    for i, s in enumerate(scores):

        if s > threshold:
            count += 1
        else:
            count = 0

        if count >= min_frames:
            filtered[i] = s

    return filtered


def combine_scores(motion, interaction, trajectory):

    return (
        W_MOTION * motion +
        W_INTERACTION * interaction +
        W_TRAJECTORY * trajectory
    )

def temporal_aggregation(scores, window=3):

    aggregated = []

    for i in range(len(scores)):

        start = max(0, i-window)
        end = i+1

        aggregated.append(sum(scores[start:end]))

    return aggregated