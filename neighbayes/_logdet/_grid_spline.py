"""Grid-and-spline log-determinant: the incumbent precompute-then-interpolate baseline.

The Spatial Econometrics Toolbox (LeSage 2021; Pace and Barry 1997; LeSage and
Pace 2009) evaluates ``log|I - rho W|`` exactly by sparse LU on an equispaced
grid of ``n_grid`` values of rho, then interpolates with a cubic spline during
estimation.  It is carried here so the literature baseline can be run through
the same model API as the methods that replace it, rather than only in a
benchmark harness.

The grid count is fixed by convention rather than derived from the function --
that is the property the Bernstein-ellipse order rule and AAA are meant to
remove -- so ``n_grid`` defaults to the toolbox's 100 and is not adaptive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from ._aaa import _make_reusable_lu_logdet

__all__ = [
    "GridSplinePrecompute",
    "grid_spline_logdet_precompute",
    "grid_spline_logdet_eval",
    "grid_spline_breaks_coeffs",
    "grid_spline_eval_jax",
]


@dataclass(frozen=True)
class GridSplinePrecompute:
    """Cubic spline through exact log-determinants on an equispaced rho grid."""

    spline: object
    rho_min: float
    rho_max: float
    n_grid: int


def grid_spline_logdet_precompute(
    W,
    rho_min: float = -1.0,
    rho_max: float = 1.0,
    n_grid: int = 100,
) -> GridSplinePrecompute:
    """Exact LU log-determinants on ``n_grid`` equispaced nodes, then a cubic spline.

    Uses the same reusable KLU evaluator as :func:`aaa_logdet_precompute`, so
    the symbolic analysis is computed once and each node costs one numeric
    refactorization.  A comparison against AAA at equal ``n_grid`` therefore
    isolates the interpolant, not the LU backend.

    Parameters
    ----------
    W : array-like or scipy.sparse matrix
        Spatial weights matrix (dense or sparse, non-symmetric OK).
    rho_min, rho_max : float
        Endpoints of the interpolation interval.  A cubic spline does not
        extrapolate meaningfully, so evaluation outside is clipped.
    n_grid : int, default 100
        Number of exact factorizations, fixed by convention.
    """
    from scipy.interpolate import CubicSpline

    if n_grid < 4:
        raise ValueError(f"n_grid must be at least 4 for a cubic spline, got {n_grid}")
    if not rho_max > rho_min:
        raise ValueError(f"rho_max must exceed rho_min, got [{rho_min}, {rho_max}]")

    Wc = sp.csc_matrix(W)
    n = Wc.shape[0]
    eye = sp.eye(n, format="csc")
    lu_logdet = _make_reusable_lu_logdet()

    rhos = np.linspace(float(rho_min), float(rho_max), int(n_grid))
    vals = np.array([lu_logdet(eye - float(r) * Wc) for r in rhos], dtype=np.float64)
    return GridSplinePrecompute(
        spline=CubicSpline(rhos, vals),
        rho_min=float(rho_min),
        rho_max=float(rho_max),
        n_grid=int(n_grid),
    )


def grid_spline_logdet_eval(pre: GridSplinePrecompute, rho: float) -> float:
    """Evaluate the fitted spline at ``rho``, clipped to the fitted interval."""
    r = min(max(float(rho), pre.rho_min), pre.rho_max)
    return float(pre.spline(r))


def grid_spline_breaks_coeffs(pre: GridSplinePrecompute):
    """``(breaks, coeffs)`` of the fitted spline, for array-library backends.

    ``coeffs`` has shape ``(4, n_grid - 1)`` in SciPy's descending-power layout:
    on ``[breaks[i], breaks[i + 1])`` the spline is
    ``sum_k coeffs[k, i] * (x - breaks[i]) ** (3 - k)``.
    """
    return (
        np.asarray(pre.spline.x, dtype=np.float64),
        np.asarray(pre.spline.c, dtype=np.float64),
    )


def grid_spline_eval_jax(rho, breaks, coeffs, T: int = 1):
    """JAX-native piecewise-cubic evaluation, differentiable and JIT-compatible.

    Mirrors :func:`grid_spline_logdet_eval`: ``rho`` is clipped to the fitted
    interval, since a cubic spline does not extrapolate meaningfully.
    """
    import jax.numpy as jnp

    r = jnp.clip(rho, breaks[0], breaks[-1])
    i = jnp.clip(jnp.searchsorted(breaks, r, side="right") - 1, 0, breaks.shape[0] - 2)
    dx = r - breaks[i]
    val = (((coeffs[0, i] * dx + coeffs[1, i]) * dx + coeffs[2, i]) * dx) + coeffs[3, i]
    return val if T == 1 else T * val
