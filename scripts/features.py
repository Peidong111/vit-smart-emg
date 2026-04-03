import numpy as np


def extract_features(segment):
    rms = np.sqrt(np.mean(segment ** 2))
    mav = np.mean(np.abs(segment))
    std = np.std(segment)
    max_abs = np.max(np.abs(segment))
    return np.array([rms, mav, std, max_abs])


def extract_feature_matrix(windows):
    return np.array([extract_features(w) for w in windows])