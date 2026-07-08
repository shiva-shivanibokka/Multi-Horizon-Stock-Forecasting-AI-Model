from dataclasses import dataclass

import numpy as np
import pandas as pd

from mhf.config import settings


@dataclass
class Fold:
    train: np.ndarray  # boolean mask over rows
    test: np.ndarray


def walk_forward_folds(end_dates, n_folds: int = 4, embargo: int | None = None,
                       min_train: int = 252) -> list[Fold]:
    if embargo is None:
        embargo = settings.max_horizon
    ed = pd.to_datetime(pd.Series(end_dates)).reset_index(drop=True)
    uniq = np.sort(ed.unique())
    if len(uniq) <= min_train + n_folds:
        raise ValueError(f"not enough distinct dates ({len(uniq)}) for {n_folds} folds")

    bday = pd.tseries.offsets.BDay(embargo)
    test_pool = uniq[min_train:]
    blocks = np.array_split(test_pool, n_folds)

    folds: list[Fold] = []
    for block in blocks:
        block_start = pd.Timestamp(block[0])
        block_end = pd.Timestamp(block[-1])
        # embargo the head of the test block
        test_lo = block_start + bday
        test_mask = ((ed >= test_lo) & (ed <= block_end)).to_numpy()
        # train = everything whose label window ends before the (embargoed) test span
        train_cut = test_lo  # first scored test date
        label_end = ed + bday
        train_mask = (label_end < train_cut).to_numpy()
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        folds.append(Fold(train=train_mask, test=test_mask))
    return folds
