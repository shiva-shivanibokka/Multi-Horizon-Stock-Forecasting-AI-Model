import numpy as np

from mhf.constants import QUANTILES
from mhf.eval.metrics import pinball_loss


def blend(preds_a: np.ndarray, preds_b: np.ndarray, w) -> np.ndarray:
    # w is a scalar OR a per-horizon vector of shape (n_horizons,). A per-horizon
    # weight lets each horizon pick its own mix — GBM can own 6m while Chronos owns
    # 1m/3m — instead of one global winner-take-all weight.
    w = np.asarray(w, dtype=float)
    if w.ndim == 1:
        w = w[None, :, None]  # broadcast over (rows, horizons, quantiles)
    out = w * preds_a + (1.0 - w) * preds_b
    return np.sort(out, axis=2)  # keep quantiles monotone after mixing


def fit_blend_weight(preds_a, preds_b, y, quantiles=QUANTILES) -> np.ndarray:
    # One weight PER HORIZON, each minimising that horizon's pinball independently.
    n_h = y.shape[1]
    grid = np.linspace(0.0, 1.0, 21)
    weights = np.empty(n_h)
    for h in range(n_h):
        best_w, best_loss = 1.0, np.inf
        for w in grid:
            blended = np.sort(w * preds_a[:, h, :] + (1.0 - w) * preds_b[:, h, :], axis=1)
            loss = pinball_loss(y[:, h], blended, quantiles)
            if loss < best_loss:
                best_loss, best_w = loss, float(w)
        weights[h] = best_w
    return weights


def fit_conformal(q_cal: np.ndarray, y_cal: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """Conformalized quantile regression (CQR) correction, one delta per horizon.

    On the calibration fold, score how far each truth fell OUTSIDE the predicted
    [q_low, q_high] band (negative when comfortably inside), then take the finite-
    sample (1-alpha) quantile of those scores. Widening the band by delta makes its
    out-of-sample coverage ~ (1-alpha) on exchangeable data. alpha=0.2 targets the
    80% interval (our q10/q90). The median is left untouched.
    """
    lo, hi = q_cal[:, :, 0], q_cal[:, :, 2]      # (n, n_horizons)
    scores = np.maximum(lo - y_cal, y_cal - hi)  # (n, n_horizons)
    n = scores.shape[0]
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)  # finite-sample adj.
    return np.quantile(scores, level, axis=0)    # (n_horizons,)


def apply_conformal(q: np.ndarray, delta: np.ndarray) -> np.ndarray:
    q = q.copy()
    q[:, :, 0] -= delta[None, :]   # widen lower bound
    q[:, :, 2] += delta[None, :]   # widen upper bound
    return np.sort(q, axis=2)
