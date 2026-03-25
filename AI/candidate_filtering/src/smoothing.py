import numpy as np
from config import SMOOTH_WINDOW

def temporal_smoothing(scores):

    smoothed = []

    for i in range(len(scores)):

        start = max(0,i-SMOOTH_WINDOW)
        end = min(len(scores),i+SMOOTH_WINDOW)

        smoothed.append(np.mean(scores[start:end]))

    return smoothed