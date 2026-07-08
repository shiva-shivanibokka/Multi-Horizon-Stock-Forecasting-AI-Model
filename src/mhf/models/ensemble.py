import numpy as np

from mhf.constants import QUANTILES
from mhf.eval.metrics import pinball_loss


def blend(preds_a: np.ndarray, preds_b: np.ndarray, w: float) -> np.ndarray:
    out = w * preds_a + (1.0 - w) * preds_b
    out = np.sort(out, axis=2)
    return out


def fit_blend_weight(preds_a, preds_b, y, quantiles=QUANTILES) -> float:
    best_w, best_loss = 1.0, np.inf
    for w in np.linspace(0.0, 1.0, 21):
        blended = blend(preds_a, preds_b, w)
        loss = sum(
            pinball_loss(y[:, h], blended[:, h, :], quantiles)
            for h in range(y.shape[1])
        )
        if loss < best_loss:
            best_loss, best_w = loss, float(w)
    return best_w
