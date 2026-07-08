import logging
from pathlib import Path

import numpy as np
import pandas as pd

from mhf.constants import QUANTILES
from mhf.config import settings

logger = logging.getLogger(__name__)
_QCOLS = [str(q) for q in QUANTILES]


def to_tsdf(series_by_ticker: dict[str, pd.Series]):
    from autogluon.timeseries import TimeSeriesDataFrame

    frames = []
    for ticker, s in series_by_ticker.items():
        s = s.dropna()
        frames.append(pd.DataFrame({
            "item_id": ticker,
            "timestamp": pd.to_datetime(s.index),
            "target": s.to_numpy(dtype=float),
        }))
    long_df = pd.concat(frames, ignore_index=True)
    return TimeSeriesDataFrame.from_data_frame(long_df)


def forecast_to_return_quantiles(pred_df: pd.DataFrame, anchor_log_price: float,
                                 horizons=None) -> np.ndarray:
    if horizons is None:
        horizons = settings.horizons
    # pred_df is one item's 126-step forecast; rows already time-ordered.
    steps = list(horizons.values())
    q = pred_df[_QCOLS].to_numpy()  # (126, n_quantiles) of forecast log-price
    out = np.empty((len(steps), len(_QCOLS)))
    for i, h in enumerate(steps):
        out[i] = np.exp(q[h - 1] - anchor_log_price) - 1.0
    out.sort(axis=1)
    return out


class ChronosForecaster:
    def __init__(self, prediction_length: int = 126, quantiles=QUANTILES):
        self.prediction_length = prediction_length
        self.quantiles = quantiles
        self.predictor_ = None
        self._series: dict[str, pd.Series] = {}

    def set_series(self, series_by_ticker: dict[str, pd.Series]) -> "ChronosForecaster":
        # log-price series per ticker, indexed by date (causal, full history)
        self._series = {k: np.log(v.dropna()) for k, v in series_by_ticker.items()}
        return self

    def fit(self, train_series: dict[str, pd.Series], fine_tune_steps: int = 1000):
        from autogluon.timeseries import TimeSeriesPredictor

        self.set_series(train_series)
        train_data = to_tsdf(self._series)
        self.predictor_ = TimeSeriesPredictor(
            prediction_length=self.prediction_length,
            quantile_levels=list(self.quantiles),
            target="target",
            freq="B",
        ).fit(
            train_data,
            hyperparameters={
                "Chronos": {
                    "model_path": "bolt_small",
                    "fine_tune": True,
                    "fine_tune_steps": fine_tune_steps,
                    "ag_args": {"name_suffix": "FineTuned"},
                }
            },
            enable_ensemble=False,
            verbosity=1,
        )
        return self

    def predict_quantiles(self, panel_rows: pd.DataFrame) -> np.ndarray:
        assert self.predictor_ is not None, "fit first"
        rows = panel_rows.reset_index(drop=True)
        n = len(rows)
        out = np.empty((n, len(settings.horizons), len(self.quantiles)))
        # Batch by anchor date: every ticker sharing a window-end date is forecast
        # in ONE predict() call (each series truncated causally to <= that date).
        # This turns O(rows) AutoGluon predict calls into O(distinct dates) with
        # many items each — the difference between a feasible and an infeasible
        # full-universe evaluation (tens of thousands of calls -> ~one per month).
        for date, grp in rows.groupby("end_date", sort=False):
            ts = pd.Timestamp(date)
            ctx_series: dict[str, pd.Series] = {}
            anchors: dict[str, float] = {}
            for pos, row in grp.iterrows():
                hist = self._series[row["ticker"]]
                hist = hist[hist.index <= ts]
                ctx_series[row["ticker"]] = hist
                anchors[row["ticker"]] = float(hist.iloc[-1])
            pred = self.predictor_.predict(to_tsdf(ctx_series))
            for pos, row in grp.iterrows():
                item_pred = pred.loc[row["ticker"]]
                out[pos] = forecast_to_return_quantiles(
                    item_pred, anchor_log_price=anchors[row["ticker"]]
                )
        return out

    def save(self, path: str | Path) -> None:
        # AutoGluon persists the fitted predictor to self.predictor_.path during
        # fit(); TimeSeriesPredictor.save() takes no destination arg, so copy that
        # directory to the requested location (overwriting any prior copy).
        import shutil

        dst = Path(path)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(self.predictor_.path, dst)

    @classmethod
    def load(cls, path: str | Path, series_by_ticker: dict[str, pd.Series]):
        from autogluon.timeseries import TimeSeriesPredictor

        obj = cls()
        obj.predictor_ = TimeSeriesPredictor.load(str(path))
        obj.set_series(series_by_ticker)
        return obj
