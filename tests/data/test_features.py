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


def test_market_features_are_causal(ohlcv, market):
    """Rows at/before `cut` must never pick up market data from after `cut`.
    Guards specifically against a .bfill() on the market join (the v1 leak).

    The `market` fixture has no interior gaps over the compared range, so a
    plain future-value perturbation never produces a NaN for ffill/bfill to
    act on and can't distinguish the two. Instead we delete all market
    history at/before `cut` and wreck what's left (the future). A causal
    ffill-only join then has no data to draw on for those early dates, so
    they're NaN and dropped. A leaky bfill would pull the wrecked future
    values backward and keep them.
    """
    base = compute_features(ohlcv, market)
    cut = base.index[len(base) // 2]

    tampered_market = market.copy()
    tampered_market = tampered_market.loc[tampered_market.index > cut] * 5.0
    after = compute_features(ohlcv, tampered_market)

    assert not (after.index <= cut).any(), "future market data leaked into rows <= cut"
