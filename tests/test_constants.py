from mhf.constants import QUANTILES


def test_quantiles_are_p10_p50_p90():
    assert QUANTILES == [0.1, 0.5, 0.9]
    assert QUANTILES == sorted(QUANTILES)
