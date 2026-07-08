import numpy as np

from mhf.constants import QUANTILES
from mhf.eval.metrics import (
    coverage,
    directional_hit_rate,
    information_coefficient,
    pinball_loss,
)


def test_pinball_zero_for_perfect_median_at_p50():
    y = np.array([0.0, 1.0, -1.0])
    # perfect prediction at every quantile -> zero loss
    q = np.tile(y[:, None], (1, len(QUANTILES)))
    assert pinball_loss(y, q) == 0.0


def test_pinball_positive_for_wrong_prediction():
    y = np.array([1.0, 1.0])
    q = np.zeros((2, len(QUANTILES)))
    assert pinball_loss(y, q) > 0


def test_coverage_counts_band_membership():
    y = np.array([0.0, 5.0, -5.0, 0.5])
    lo = np.full(4, -1.0)
    hi = np.full(4, 1.0)
    assert coverage(y, lo, hi) == 0.5  # 0.0 and 0.5 are inside


def test_directional_hit_rate():
    y = np.array([1.0, -1.0, 2.0, -3.0])
    p50 = np.array([0.5, -0.2, -0.1, -1.0])  # 3 of 4 correct sign
    assert directional_hit_rate(y, p50) == 0.75


def test_information_coefficient_perfect_rank():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p50 = np.array([10.0, 20.0, 30.0, 40.0])
    assert abs(information_coefficient(y, p50) - 1.0) < 1e-9
