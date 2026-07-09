import numpy as np
import pandas as pd

from mhf.config import settings
from mhf.eval.cv import walk_forward_folds


def _dates(n=1500):
    # repeat each business day across 5 tickers to mimic a panel
    base = pd.bdate_range("2015-01-01", periods=n)
    return np.repeat(base.to_numpy(), 5)


def test_no_train_test_date_overlap():
    ed = _dates()
    for fold in walk_forward_folds(ed, n_folds=4):
        train_dates = set(pd.to_datetime(ed[fold.train]))
        test_dates = set(pd.to_datetime(ed[fold.test]))
        assert train_dates.isdisjoint(test_dates)


def test_embargo_gap_at_least_max_horizon():
    ed = _dates()
    embargo = settings.max_horizon
    for fold in walk_forward_folds(ed, n_folds=4, embargo=embargo):
        last_train = pd.to_datetime(ed[fold.train]).max()
        first_test = pd.to_datetime(ed[fold.test]).min()
        gap_bdays = np.busday_count(last_train.date(), first_test.date())
        assert gap_bdays >= embargo, f"gap {gap_bdays} < embargo {embargo}"


def test_purge_removes_overlapping_labels():
    # A train sample whose 126-day label window reaches into the test span must be purged.
    ed = _dates()
    embargo = settings.max_horizon
    for fold in walk_forward_folds(ed, n_folds=4, embargo=embargo):
        first_test = pd.to_datetime(ed[fold.test]).min()
        train_ed = pd.to_datetime(ed[fold.train])
        # every train label window must END strictly before the test span begins
        label_end = train_ed + pd.tseries.offsets.BDay(embargo)
        assert (label_end < first_test).all()


def test_folds_are_nonempty_and_expanding():
    ed = _dates()
    folds = walk_forward_folds(ed, n_folds=4)
    assert len(folds) == 4
    train_sizes = [int(f.train.sum()) for f in folds]
    for f in folds:
        assert f.train.sum() > 0 and f.test.sum() > 0
    assert train_sizes == sorted(train_sizes)  # expanding window


def test_masks_are_disjoint_boolean():
    ed = _dates()
    for f in walk_forward_folds(ed, n_folds=3):
        assert f.train.dtype == bool and f.test.dtype == bool
        assert not (f.train & f.test).any()
        assert len(f.train) == len(ed)
