"""SIFT, Lowe ratio filtering and mutual consistency filtering."""
import cv2
import numpy as np


def detect_and_match(left, right, ratio=.75):
    sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=.02)
    k1, d1 = sift.detectAndCompute(left, None)
    k2, d2 = sift.detectAndCompute(right, None)
    if d1 is None or d2 is None or min(len(d1), len(d2)) < 2:
        raise RuntimeError("Insufficient SIFT descriptors")
    bf = cv2.BFMatcher(cv2.NORM_L2)
    forward = [pair[0] for pair in bf.knnMatch(d1, d2, k=2)
               if len(pair) == 2 and pair[0].distance < ratio*pair[1].distance]
    backward = [pair[0] for pair in bf.knnMatch(d2, d1, k=2)
                if len(pair) == 2 and pair[0].distance < ratio*pair[1].distance]
    reverse = {(m.trainIdx, m.queryIdx) for m in backward}
    mutual = sorted([m for m in forward if (m.queryIdx, m.trainIdx) in reverse],
                    key=lambda m: (m.queryIdx, m.trainIdx))
    p1 = np.array([k1[m.queryIdx].pt for m in mutual], dtype=float)
    p2 = np.array([k2[m.trainIdx].pt for m in mutual], dtype=float)
    return k1, k2, forward, mutual, p1, p2
