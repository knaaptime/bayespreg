"""Cross-sectional linear spatial DGP functions."""

from __future__ import annotations

import numpy as np

from .utils import ensure_rng, make_design_matrix, resolve_weights


def simulate_sar(
    n: int | None = None,
    W=None,
    gdf=None,
    rho: float = 0.5,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
) -> dict:
    """Simulate data from SAR DGP ``y = (I-rho W)^(-1)(X beta + eps)``.

    Parameters
    ----------
    n : int, optional
        Number of observations. Ignored when ``W`` or ``gdf`` defines size.
    W : Graph or sparse/dense matrix, optional
        Spatial weights. If supplied, takes precedence over ``gdf``.
    gdf : geopandas.GeoDataFrame, optional
        Spatial units source used when ``W`` is not provided.
    rho : float, default=0.5
        Spatial autoregressive coefficient.
    beta : np.ndarray, optional
        Regression coefficients including intercept.
    sigma : float, default=1.0
        Innovation standard deviation.
    rng : np.random.Generator, optional
        Random generator.
    seed : int, optional
        Seed used when ``rng`` is not supplied.
    contiguity : str, default="queen"
        Neighbor rule for ``gdf`` mode.

    Returns
    -------
    dict
        Keys: ``y``, ``X``, ``W_dense``, ``W_graph``, ``params_true``.
    """
    rng = ensure_rng(rng, seed)
    Wd, Wg = resolve_weights(W=W, gdf=gdf, contiguity=contiguity)
    nobs = Wd.shape[0] if n is None else int(n)
    if nobs != Wd.shape[0]:
        raise ValueError("n must match the size implied by W/gdf.")

    if beta is None:
        beta = np.array([1.0, 2.0], dtype=float)
    beta = np.asarray(beta, dtype=float)

    X = make_design_matrix(rng, nobs, k=max(len(beta) - 1, 0), add_intercept=True)
    eps = sigma * rng.standard_normal(nobs)
    y = np.linalg.solve(np.eye(nobs) - rho * Wd, X @ beta + eps)
    return {
        "y": y,
        "X": X,
        "W_dense": Wd,
        "W_graph": Wg,
        "params_true": {"rho": rho, "beta": beta, "sigma": sigma},
    }


def simulate_sem(
    n: int | None = None,
    W=None,
    gdf=None,
    lam: float = 0.5,
    beta: np.ndarray | None = None,
    sigma: float = 1.0,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
) -> dict:
    """Simulate data from SEM DGP ``y = X beta + (I-lam W)^(-1) eps``.

    Parameters are analogous to :func:`simulate_sar` with ``lam`` replacing
    ``rho``.

    Returns
    -------
    dict
        Keys: ``y``, ``X``, ``W_dense``, ``W_graph``, ``params_true``.
    """
    rng = ensure_rng(rng, seed)
    Wd, Wg = resolve_weights(W=W, gdf=gdf, contiguity=contiguity)
    nobs = Wd.shape[0] if n is None else int(n)
    if nobs != Wd.shape[0]:
        raise ValueError("n must match the size implied by W/gdf.")

    if beta is None:
        beta = np.array([1.0, 2.0], dtype=float)
    beta = np.asarray(beta, dtype=float)

    X = make_design_matrix(rng, nobs, k=max(len(beta) - 1, 0), add_intercept=True)
    eps = sigma * rng.standard_normal(nobs)
    u = np.linalg.solve(np.eye(nobs) - lam * Wd, eps)
    y = X @ beta + u
    return {
        "y": y,
        "X": X,
        "W_dense": Wd,
        "W_graph": Wg,
        "params_true": {"lam": lam, "beta": beta, "sigma": sigma},
    }


def simulate_slx(
    n: int | None = None,
    W=None,
    gdf=None,
    beta1: np.ndarray | None = None,
    beta2: np.ndarray | None = None,
    sigma: float = 1.0,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
) -> dict:
    """Simulate data from SLX DGP ``y = X beta1 + W X_no_intercept beta2 + eps``.

    Returns
    -------
    dict
        Keys: ``y``, ``X``, ``W_dense``, ``W_graph``, ``params_true``.
    """
    rng = ensure_rng(rng, seed)
    Wd, Wg = resolve_weights(W=W, gdf=gdf, contiguity=contiguity)
    nobs = Wd.shape[0] if n is None else int(n)
    if nobs != Wd.shape[0]:
        raise ValueError("n must match the size implied by W/gdf.")

    if beta1 is None:
        beta1 = np.array([1.0, 2.0], dtype=float)
    beta1 = np.asarray(beta1, dtype=float)
    if beta2 is None:
        beta2 = np.array([0.8], dtype=float)
    beta2 = np.asarray(beta2, dtype=float)

    X = make_design_matrix(rng, nobs, k=max(len(beta1) - 1, 0), add_intercept=True)
    Wx = Wd @ X[:, 1:]
    if Wx.shape[1] != len(beta2):
        raise ValueError("len(beta2) must match number of non-intercept regressors.")

    y = X @ beta1 + Wx @ beta2 + sigma * rng.standard_normal(nobs)
    return {
        "y": y,
        "X": X,
        "W_dense": Wd,
        "W_graph": Wg,
        "params_true": {"beta1": beta1, "beta2": beta2, "sigma": sigma},
    }


def simulate_sdm(
    n: int | None = None,
    W=None,
    gdf=None,
    rho: float = 0.4,
    beta1: np.ndarray | None = None,
    beta2: np.ndarray | None = None,
    sigma: float = 1.0,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
) -> dict:
    """Simulate data from SDM DGP ``y = (I-rho W)^(-1)(Xb1 + WXb2 + eps)``.

    Returns
    -------
    dict
        Keys: ``y``, ``X``, ``W_dense``, ``W_graph``, ``params_true``.
    """
    rng = ensure_rng(rng, seed)
    Wd, Wg = resolve_weights(W=W, gdf=gdf, contiguity=contiguity)
    nobs = Wd.shape[0] if n is None else int(n)
    if nobs != Wd.shape[0]:
        raise ValueError("n must match the size implied by W/gdf.")

    if beta1 is None:
        beta1 = np.array([1.0, 2.0], dtype=float)
    beta1 = np.asarray(beta1, dtype=float)
    if beta2 is None:
        beta2 = np.array([0.8], dtype=float)
    beta2 = np.asarray(beta2, dtype=float)

    X = make_design_matrix(rng, nobs, k=max(len(beta1) - 1, 0), add_intercept=True)
    Wx = Wd @ X[:, 1:]
    if Wx.shape[1] != len(beta2):
        raise ValueError("len(beta2) must match number of non-intercept regressors.")

    eps = sigma * rng.standard_normal(nobs)
    y = np.linalg.solve(np.eye(nobs) - rho * Wd, X @ beta1 + Wx @ beta2 + eps)
    return {
        "y": y,
        "X": X,
        "W_dense": Wd,
        "W_graph": Wg,
        "params_true": {"rho": rho, "beta1": beta1, "beta2": beta2, "sigma": sigma},
    }


def simulate_sdem(
    n: int | None = None,
    W=None,
    gdf=None,
    lam: float = 0.4,
    beta1: np.ndarray | None = None,
    beta2: np.ndarray | None = None,
    sigma: float = 1.0,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    contiguity: str = "queen",
) -> dict:
    """Simulate data from SDEM DGP ``y = Xb1 + WXb2 + (I-lam W)^(-1)eps``.

    Returns
    -------
    dict
        Keys: ``y``, ``X``, ``W_dense``, ``W_graph``, ``params_true``.
    """
    rng = ensure_rng(rng, seed)
    Wd, Wg = resolve_weights(W=W, gdf=gdf, contiguity=contiguity)
    nobs = Wd.shape[0] if n is None else int(n)
    if nobs != Wd.shape[0]:
        raise ValueError("n must match the size implied by W/gdf.")

    if beta1 is None:
        beta1 = np.array([1.0, 2.0], dtype=float)
    beta1 = np.asarray(beta1, dtype=float)
    if beta2 is None:
        beta2 = np.array([0.8], dtype=float)
    beta2 = np.asarray(beta2, dtype=float)

    X = make_design_matrix(rng, nobs, k=max(len(beta1) - 1, 0), add_intercept=True)
    Wx = Wd @ X[:, 1:]
    if Wx.shape[1] != len(beta2):
        raise ValueError("len(beta2) must match number of non-intercept regressors.")

    u = np.linalg.solve(np.eye(nobs) - lam * Wd, sigma * rng.standard_normal(nobs))
    y = X @ beta1 + Wx @ beta2 + u
    return {
        "y": y,
        "X": X,
        "W_dense": Wd,
        "W_graph": Wg,
        "params_true": {"lam": lam, "beta1": beta1, "beta2": beta2, "sigma": sigma},
    }
