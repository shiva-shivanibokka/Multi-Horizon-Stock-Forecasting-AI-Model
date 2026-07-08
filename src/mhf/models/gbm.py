import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from mhf.constants import QUANTILES
from mhf.data.assemble import Y_COLS
from mhf.data.features import FEATURES


class GBMQuantile:
    def __init__(self, n_estimators=300, learning_rate=0.05, num_leaves=31,
                 min_child_samples=50, random_state=0, quantiles=QUANTILES):
        self.params = dict(
            n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
            min_child_samples=min_child_samples, random_state=random_state, verbose=-1,
        )
        self.quantiles = quantiles
        self.models_: dict[tuple[int, int], LGBMRegressor] = {}

    def fit(self, panel_train: pd.DataFrame) -> "GBMQuantile":
        # Pass the named FEATURES frame (not .to_numpy()) so LightGBM records the
        # real column names; predicting with the same-named frame then validates
        # column identity instead of relying on positional order.
        X = panel_train[FEATURES]
        for h, ycol in enumerate(Y_COLS):
            y = panel_train[ycol].to_numpy()
            for qi, q in enumerate(self.quantiles):
                model = LGBMRegressor(objective="quantile", alpha=q, **self.params)
                model.fit(X, y)
                self.models_[(h, qi)] = model
        return self

    def predict_quantiles(self, panel_rows: pd.DataFrame) -> np.ndarray:
        X = panel_rows[FEATURES]
        n = len(panel_rows)
        out = np.empty((n, len(Y_COLS), len(self.quantiles)))
        for (h, qi), model in self.models_.items():
            out[:, h, qi] = model.predict(X)
        out.sort(axis=2)  # enforce p10 <= p50 <= p90
        return out
