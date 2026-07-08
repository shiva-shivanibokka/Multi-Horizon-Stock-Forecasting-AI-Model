from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from mhf.data.features import FEATURES


@dataclass
class WindowSet:
    X: np.ndarray
    y_ret: np.ndarray
    end_dates: np.ndarray
    base_close: np.ndarray


def build_windows(feats: pd.DataFrame, window: int, horizons: dict[str, int]) -> WindowSet:
    arr = feats[FEATURES].to_numpy(dtype=np.float32)
    close = feats["Close"].to_numpy(dtype=np.float64)
    dates = feats.index.to_numpy()
    n_rows = len(arr)
    max_h = max(horizons.values())
    n = n_rows - window - max_h + 1
    if n <= 0:
        raise ValueError(f"not enough rows ({n_rows}) for window={window} + max_h={max_h}")

    X = sliding_window_view(arr, window_shape=window, axis=0)[:n].transpose(0, 2, 1).copy()
    end_idx = np.arange(n) + window - 1
    base_close = close[end_idx]
    h_vals = list(horizons.values())
    fut = np.array([[close[e + h] for h in h_vals] for e in end_idx], dtype=np.float64)
    y_ret = (fut / base_close[:, None] - 1).astype(np.float32)
    end_dates = dates[end_idx]
    return WindowSet(
        X=X, y_ret=y_ret, end_dates=end_dates, base_close=base_close.astype(np.float32)
    )
