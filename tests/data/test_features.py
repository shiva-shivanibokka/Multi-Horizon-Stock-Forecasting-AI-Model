import numpy as np

from mhf.data.features import FEATURES, compute_features


def test_features_shape_and_no_nan(ohlcv, market):
    feats = compute_features(ohlcv, market)
    assert list(feats.columns) == FEATURES
    assert not feats.isna().any().any()
    assert len(feats) > 0


def test_features_are_causal(ohlcv, market):
    """Perturbing FUTURE rows must not change any earlier feature value."""
    base = compute_features(ohlcv, market)
    cut = base.index[len(base) // 2]

    tampered = ohlcv.copy()
    future_mask = tampered.index > cut
    tampered.loc[future_mask, ["Open", "High", "Low", "Close"]] *= 5.0  # wreck the future
    after = compute_features(tampered, market)

    common = base.index[base.index <= cut]
    np.testing.assert_allclose(
        base.loc[common].values, after.loc[common].values, rtol=1e-9, atol=1e-9
    )
