import numpy as np

from mhf.models.ensemble import blend, fit_blend_weight


def test_blend_prefers_the_better_model():
    rng = np.random.default_rng(0)
    n, H, Q = 200, 3, 3
    y = rng.normal(0, 0.05, size=(n, H))
    # model A is near-perfect, model B is noise
    good = np.repeat(y[:, :, None], Q, axis=2) + rng.normal(0, 0.001, size=(n, H, Q))
    bad = rng.normal(0, 0.5, size=(n, H, Q))
    w = fit_blend_weight(good, bad, y)
    assert w > 0.8  # weight should land mostly on the good model


def test_blend_is_convex_and_sorted():
    a = np.array([[[0.0, 0.1, 0.3]]])
    b = np.array([[[0.2, 0.1, 0.0]]])
    out = blend(a, b, 0.5)
    assert out.shape == a.shape
    assert (np.diff(out, axis=2) >= 0).all()
