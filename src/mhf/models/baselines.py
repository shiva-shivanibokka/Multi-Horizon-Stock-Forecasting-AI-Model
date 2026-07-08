import logging

import numpy as np
import pandas as pd

from mhf.constants import QUANTILES
from mhf.data.assemble import Y_COLS

logger = logging.getLogger(__name__)


class RandomWalk:
    """Efficient-market null: E[forward return] = 0, band = zero-mean Gaussian on train vol."""

    def __init__(self, quantiles=QUANTILES):
        self.quantiles = quantiles
        self.sigma_: np.ndarray | None = None  # (n_horizons,)

    def fit(self, panel_train: pd.DataFrame) -> "RandomWalk":
        self.sigma_ = np.array([panel_train[c].std() for c in Y_COLS])
        return self

    def predict_quantiles(self, panel_rows: pd.DataFrame) -> np.ndarray:
        from scipy.stats import norm

        assert self.sigma_ is not None, "fit first"
        z = norm.ppf(self.quantiles)  # zero-mean quantile z-scores
        band = self.sigma_[:, None] * z[None, :]  # (n_horizons, n_quantiles)
        n = len(panel_rows)
        return np.broadcast_to(band[None], (n, *band.shape)).copy()


class HistoricalQuantile:
    def __init__(self, quantiles=QUANTILES):
        self.quantiles = quantiles
        self.table_: np.ndarray | None = None  # (n_horizons, n_quantiles)

    def fit(self, panel_train: pd.DataFrame) -> "HistoricalQuantile":
        self.table_ = np.stack(
            [np.quantile(panel_train[c].to_numpy(), self.quantiles) for c in Y_COLS]
        )
        return self

    def predict_quantiles(self, panel_rows: pd.DataFrame) -> np.ndarray:
        assert self.table_ is not None, "fit first"
        n = len(panel_rows)
        return np.broadcast_to(self.table_[None], (n, *self.table_.shape)).copy()


def garch_volatility(returns: pd.Series, horizon: int) -> float:
    from arch import arch_model

    r = returns.dropna().to_numpy() * 100.0
    try:
        res = arch_model(r, vol="GARCH", p=1, q=1, mean="Zero").fit(disp="off")
        fc = res.forecast(horizon=horizon, reindex=False)
        var_path = fc.variance.to_numpy().ravel()[:horizon]
        return float(np.sqrt(var_path.sum()) / 100.0)
    except Exception as e:  # noqa: BLE001 - GARCH convergence is genuinely flaky
        logger.warning("GARCH failed (%s); falling back to realized std", e)
        return float(returns.dropna().std() * np.sqrt(horizon))
