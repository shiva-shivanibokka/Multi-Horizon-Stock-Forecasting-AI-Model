import numpy as np
from scipy.stats import spearmanr

from mhf.constants import QUANTILES


def pinball_loss(y, q_pred, quantiles=QUANTILES) -> float:
    y = np.asarray(y, dtype=float)
    q_pred = np.asarray(q_pred, dtype=float)
    total = 0.0
    for j, q in enumerate(quantiles):
        diff = y - q_pred[:, j]
        total += np.mean(np.maximum(q * diff, (q - 1) * diff))
    return float(total / len(quantiles))


def coverage(y, q_low, q_high) -> float:
    y = np.asarray(y, dtype=float)
    inside = (y >= np.asarray(q_low, dtype=float)) & (y <= np.asarray(q_high, dtype=float))
    return float(np.mean(inside))


def directional_hit_rate(y, p50) -> float:
    y = np.asarray(y, dtype=float)
    p50 = np.asarray(p50, dtype=float)
    return float(np.mean((p50 >= 0) == (y >= 0)))


def information_coefficient(y, p50) -> float:
    rho, _ = spearmanr(np.asarray(y, dtype=float), np.asarray(p50, dtype=float))
    return float(rho)
