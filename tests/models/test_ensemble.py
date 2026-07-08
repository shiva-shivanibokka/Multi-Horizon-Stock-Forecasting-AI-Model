import numpy as np

from mhf.models.ensemble import apply_conformal, blend, fit_blend_weight, fit_conformal


def test_blend_prefers_the_better_model():
    rng = np.random.default_rng(0)
    n, H, Q = 200, 3, 3
    y = rng.normal(0, 0.05, size=(n, H))
    good = np.repeat(y[:, :, None], Q, axis=2) + rng.normal(0, 0.001, size=(n, H, Q))
    bad = rng.normal(0, 0.5, size=(n, H, Q))
    w = fit_blend_weight(good, bad, y)
    assert w.shape == (H,)
    assert (w > 0.8).all()  # every horizon should lean on the good model


def test_blend_weight_is_per_horizon():
    # model A nails horizons 0 and 2; model B nails horizon 1 -> weights must split
    rng = np.random.default_rng(1)
    n, H, Q = 300, 3, 3
    y = rng.normal(0, 0.05, size=(n, H))
    yb = np.repeat(y[:, :, None], Q, axis=2)
    a = yb + rng.normal(0, 0.001, size=(n, H, Q))
    b = yb + rng.normal(0, 0.001, size=(n, H, Q))
    a[:, 1, :] = rng.normal(0, 0.5, size=(n, Q))  # A is noise on horizon 1
    b[:, 0, :] = rng.normal(0, 0.5, size=(n, Q))  # B is noise on horizon 0
    b[:, 2, :] = rng.normal(0, 0.5, size=(n, Q))  # B is noise on horizon 2
    w = fit_blend_weight(a, b, y)
    assert w[0] > 0.8 and w[2] > 0.8  # horizons 0,2 -> model A
    assert w[1] < 0.2               # horizon 1 -> model B


def test_blend_is_convex_and_sorted():
    a = np.array([[[0.0, 0.1, 0.3]]])
    b = np.array([[[0.2, 0.1, 0.0]]])
    out = blend(a, b, 0.5)
    assert out.shape == a.shape
    assert (np.diff(out, axis=2) >= 0).all()


def test_conformal_restores_target_coverage():
    # Predicted 80% band is far too narrow ([-0.5,0.5] on N(0,1) ~ 38% coverage).
    # Calibrating on one draw and applying to a fresh draw should restore ~80%.
    rng = np.random.default_rng(2)
    n, H = 4000, 2
    y_cal = rng.normal(0, 1, size=(n, H))
    q_cal = np.stack([np.full((n, H), -0.5), np.zeros((n, H)), np.full((n, H), 0.5)], axis=2)
    delta = fit_conformal(q_cal, y_cal, alpha=0.2)
    assert (delta > 0).all()  # band was too narrow, so it must widen

    y_te = rng.normal(0, 1, size=(n, H))
    q_te = q_cal.copy()
    adj = apply_conformal(q_te, delta)
    cov = np.mean((y_te >= adj[:, :, 0]) & (y_te <= adj[:, :, 2]))
    assert 0.76 <= cov <= 0.84  # ~80% target
