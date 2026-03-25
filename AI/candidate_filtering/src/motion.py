import cv2
import numpy as np

def motion_signal(prev_frame, frame):

    diff = cv2.absdiff(prev_frame, frame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    return float(np.mean(gray))