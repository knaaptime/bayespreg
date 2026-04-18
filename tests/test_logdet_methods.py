"""Tests for MATLAB-style log-determinant helper implementations."""

from __future__ import annotations

import numpy as np

from bayespecon.logdet import lndetfull, lndetichol, lndetint, lndetmc, make_logdet_fn


def _toy_w() -> np.ndarray:
    # Small row-stochastic matrix with spectral radius <= 1.
    return np.array(
        [
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )


def test_lndetfull_returns_expected_keys_and_lengths() -> None:
    W = _toy_w()
    out = lndetfull(W, -0.5, 0.5, grid=0.25)

    assert set(out.keys()) == {"rho", "lndet"}
    assert out["rho"].shape == out["lndet"].shape
    assert np.all(np.isfinite(out["lndet"]))


def test_lndetint_returns_expected_keys_and_lengths() -> None:
    W = _toy_w()
    out = lndetint(W, -0.5, 0.5, n_grid=30)

    assert set(out.keys()) == {"rho", "lndet"}
    assert out["rho"].shape == out["lndet"].shape
    assert np.all(np.isfinite(out["lndet"]))


def test_lndetmc_returns_confidence_bounds() -> None:
    W = _toy_w()
    out = lndetmc(order=10, iter=8, W=W, rmin=-0.5, rmax=0.5, grid=0.25, random_state=7)

    assert set(out.keys()) == {"rho", "lndet", "up95", "lo95"}
    assert out["rho"].shape == out["lndet"].shape == out["up95"].shape == out["lo95"].shape
    assert np.all(out["up95"] >= out["lo95"])


def test_lndetichol_returns_expected_keys_and_lengths() -> None:
    W = _toy_w()
    out = lndetichol(W, -0.5, 0.5, grid=0.25)

    assert set(out.keys()) == {"rho", "lndet"}
    assert out["rho"].shape == out["lndet"].shape
    assert np.all(np.isfinite(out["lndet"]))


def test_make_logdet_fn_accepts_new_method_names() -> None:
    W = _toy_w()

    for method in ("full", "int", "mc", "ichol"):
        fn = make_logdet_fn(W, method=method, rho_min=-0.5, rho_max=0.5)
        assert callable(fn)
