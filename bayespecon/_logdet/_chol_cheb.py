"""Cholesky-Chebyshev log-determinant for SPD ``I - ρW``.

For row-standardised ``W`` with spectrum in ``[-1, 1]``, the D-symmetrised
matrix ``W_sym = D^{-1/2} A D^{-1/2}`` is symmetric with the same eigenvalues
as ``W``.  This makes ``I - ρW_sym`` **symmetric positive definite** (SPD)
for all ``|ρ| < 1``, enabling **sparse Cholesky** factorisation:

* **Exact**: ``log|det(I - ρW)| = 2 Σ log(diag(L))`` — no stochastic noise.
* **Fast**: CHOLMOD sparse Cholesky is ~2× faster than sparse LU for SPD.
* **Scalable**: ``O(nnz^{1.5})`` per factorisation, no ``O(n³)`` eigendecomposition.
* **Full range**: works for ``ρ ∈ (-1, 1)`` — the entire stable region.
  The order is selected from the interval's Bernstein-ellipse convergence
  rate (:func:`~._chebyshev.cheb_order_for_tolerance`) — its distance to the
  ``ρ = ±1`` singularities, not its width — so it rises for wide intervals
  (17 nodes for ``[0.1, 0.8]``, 52 for ``[-0.95, 0.95]``) and falls sharply
  for narrow ones (6 for ``[0.55, 0.65]``).

The method evaluates ``log|det(I - ρW)|`` exactly at ``order`` Chebyshev nodes
in ``[ρ_min, ρ_max]``, then fits a Chebyshev polynomial in ``ρ`` for ``O(order)``
per-``ρ`` evaluation via Clenshaw recurrence.

**Symbolic reuse**: all ``I - ρW`` matrices share the same sparsity pattern,
so CHOLMOD's symbolic analysis (AMD ordering + elimination tree) is performed
only once and reused for all subsequent numeric factorisations via
``factor.factorize()``.  This saves ~64% of per-node cost.

**When to use**: ``n ∈ (500, 60000]``, any ``ρ ∈ (-1, 1)``.  For ``n ≤ 500``
use ``eigenvalue`` (exact eigendecomposition).  For ``n > 60000`` use
``cheb_stochastic`` (avoids ``O(nnz^{1.5})`` Cholesky fill-in).  For
non-symmetric ``W`` (directed graphs: KNN, travel time) use ``aaa`` (rational
approximation via sparse LU).

**Benchmark** (2D rook grid, adaptive order, ρ ∈ [0.1, 0.8], 2026-07):

========== ============= ============= =========== ==================
n          chol setup    chol eval     chol error  stoch(200)
========== ============= ============= =========== ==================
484        3.8ms         1.7μs         5e-9        2.7ms, 0.42 err
4,900      30ms          1.7μs         2e-7        26ms, 0.69 err
10,000     96ms          1.7μs         5e-7        62ms, 0.75 err
19,881     248ms         1.7μs         9e-7        110ms, 1.7 err
40,000     583ms         1.7μs         1.8e-6      236ms, 1.8 err
59,536     1.18s         1.7μs         2.6e-6      328ms, 3.5 err
========== ============= ============= =========== ==================

Cholesky-Chebyshev is the accuracy leader across this range: exact (5e-9 to
2.6e-6 vs 0.4-3.5 for stochastic) and ~30× faster eval (1.7μs vs ~57μs).  Its
setup grows with Cholesky fill-in, reaching ~3.6× the stochastic cost by
n≈60k — small in absolute terms against any chain that runs for seconds, which
is why the auto-selection cutoff sits at 60,000 rather than where the setup
curves first cross.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from ._chebyshev import (
    cheb_order_for_tolerance,
    cheb_tail_error,
    chebyshev_coeffs_dct1,
    chebyshev_gauss_nodes,
)
from ._clenshaw import clenshaw_scalar, clenshaw_vec


@dataclass(frozen=True)
class CholChebPrecompute:
    """Precomputed Chebyshev coefficients from Cholesky log-determinant.

    Attributes
    ----------
    coeffs : np.ndarray, shape (order,)
        Chebyshev coefficients of ``log|I - ρW|`` in ``ρ``.
    rho_min : float
        Lower bound of the ρ approximation interval.
    rho_max : float
        Upper bound of the ρ approximation interval.
    order : int
        Chebyshev polynomial degree.
    n : int
        Matrix dimension.
    err_est : float
        A-posteriori truncation-error estimate from the coefficient tail (see
        :func:`~._chebyshev.cheb_tail_error`).  ``nan`` when not computed.
    """

    coeffs: np.ndarray
    rho_min: float
    rho_max: float
    order: int
    n: int
    err_est: float = float("nan")


def _d_symmetrize(W: sp.csr_matrix) -> sp.csc_matrix:
    """D-symmetrise row-standardised ``W``.

    For ``W = D⁻¹A`` (row-standardised, ``A`` symmetric adjacency),
    ``W_sym = D^{1/2} W D^{-1/2} = D^{-1/2} A D^{-1/2}`` is symmetric
    with the **same eigenvalues** as ``W``.

    The symmetrizing degrees are recovered from the *values* of ``W``
    via the edge ratios ``D[i]/D[j] = W[j,i]/W[i,j]`` (BFS propagation).
    The neighbor count (``getnnz``) equals the standardizing degree only
    for binary adjacency — using it for weighted graphs breaks the
    symmetry that CHOLMOD relies on (it reads a single triangle) and
    silently corrupts the log-determinant.

    This makes ``I - ρW_sym`` SPD for ``|ρ| < 1``, enabling sparse Cholesky.

    Raises
    ------
    ValueError
        If no symmetrizing diagonal exists (directed graph, or weights
        inconsistent with ``W = D⁻¹A`` for symmetric ``A``).  Pass
        ``logdet_method="aaa"`` for such matrices.
    """
    n = W.shape[0]
    W = sp.csr_matrix(W)

    # Fast path: W already symmetric — no scaling needed.
    diff = (W - W.T).tocoo()
    if diff.nnz == 0 or np.all(np.abs(diff.data) <= 1e-12):
        return sp.csc_matrix(W)

    from ._slq import _recover_symmetrizing_diagonal

    D = _recover_symmetrizing_diagonal(W)
    if D is None or not np.all(np.isfinite(D)) or np.any(D <= 0):
        raise ValueError(
            "cheb_cholesky requires a D-symmetrizable W (row-standardised "
            "undirected graph); no valid symmetrizing diagonal was found. "
            'Use logdet_method="aaa" for directed or non-symmetrizable W.'
        )

    D_sqrt = np.sqrt(D)
    D_inv_sqrt = 1.0 / D_sqrt
    # W_sym = D^{1/2} W D^{-1/2}  — sparse scaling, no densification
    # W_sym[i,j] = sqrt(d_i) * W[i,j] / sqrt(d_j)
    W_coo = W.tocoo()
    scaled_data = D_sqrt[W_coo.row] * W_coo.data * D_inv_sqrt[W_coo.col]
    W_sym = sp.csc_matrix((scaled_data, (W_coo.row, W_coo.col)), shape=(n, n))

    # Hard guard: CHOLMOD reads one triangle of its input, so a
    # non-symmetric W_sym would produce a silently wrong logdet.
    sym_diff = (W_sym - W_sym.T).tocoo()
    sym_err = float(np.abs(sym_diff.data).max()) if sym_diff.nnz else 0.0
    if sym_err > 1e-10:
        raise ValueError(
            f"D-symmetrization failed (max asymmetry {sym_err:.2e}); W is "
            "not of the form D^-1 A with symmetric A. Use "
            'logdet_method="aaa" for this weights matrix.'
        )
    return W_sym


def _clamp_interval(rho_min: float, rho_max: float) -> tuple[float, float]:
    """Clamp a ρ interval away from the logarithmic singularities at ``±1``."""
    return max(float(rho_min), -0.99), min(float(rho_max), 0.99)


class CholChebContext:
    """Reusable D-symmetrisation and CHOLMOD symbolic analysis for one ``W``.

    Everything a Chebyshev interpolant of ``log|I - ρW|`` needs that does *not*
    depend on the interval — the symmetrised matrix ``W_sym``, the identity, and
    CHOLMOD's symbolic analysis (AMD ordering + elimination tree) — is built
    once here and reused by every call to :meth:`coeffs_on`.  Fitting a *second*
    interpolant on a different interval therefore costs only its numeric
    factorisations.

    That is what makes a warmup-adaptive refit affordable.  A sampler past
    warmup knows ρ to within a fraction of the prior's support, and a narrow
    interval both needs fewer nodes (see
    :func:`~._chebyshev.cheb_order_for_tolerance`) and resolves the Jacobian far
    more accurately over the region the posterior actually occupies — but only
    if the refit does not re-pay the setup.  Measured on the rook lattice
    (n = 10,000): a fresh build at 15 nodes costs ~77 ms, a refit at 12 through
    this context ~60 ms, at 6 nodes ~30 ms.

    Parameters
    ----------
    W : array-like or scipy.sparse matrix
        Spatial weights matrix (dense or sparse, row-standardised).

    Raises
    ------
    ValueError
        If ``W`` admits no symmetrizing diagonal (directed graph); use
        ``logdet_method="aaa"`` for those.
    """

    __slots__ = ("W_sym", "_eye", "_factor", "n")

    def __init__(self, W):
        if sp.issparse(W) or hasattr(W, "format"):
            W_sp = sp.csr_matrix(W, dtype=np.float64)
        else:
            W_sp = sp.csr_matrix(np.asarray(W, dtype=np.float64))
        self.n = int(W_sp.shape[0])
        self.W_sym = _d_symmetrize(W_sp)
        self._eye = sp.eye(self.n, format="csc")
        self._factor = None

    def logdet_at(self, rho_nodes: np.ndarray) -> np.ndarray:
        """Exact ``log|I - ρW|`` at each node, by sparse Cholesky.

        The first call performs CHOLMOD's symbolic analysis and caches it on the
        instance; every later node — and every later interval — reuses it, since
        the sparsity pattern of ``I - ρW_sym`` does not depend on ρ.
        """
        from sksparse.cholmod import cho_factor as cholmod_cho_factor

        out = np.empty(len(rho_nodes), dtype=np.float64)
        for i, rho in enumerate(rho_nodes):
            A = sp.csc_matrix(self._eye - float(rho) * self.W_sym)
            if self._factor is None:
                # First node ever: symbolic analysis + numeric factorisation.
                self._factor = cholmod_cho_factor(A)
            else:
                # Numeric factorisation only — symbolic analysis reused.
                self._factor.factorize(A)
            out[i] = self._factor.logdet()
        return out

    def coeffs_on(
        self,
        rho_min: float = 0.1,
        rho_max: float = 0.8,
        order: int | None = None,
        tol: float | None = None,
    ) -> CholChebPrecompute:
        """Fit the Chebyshev interpolant on ``[rho_min, rho_max]``.

        Parameters
        ----------
        rho_min, rho_max : float
            The ρ interval, clamped to ``[-0.99, 0.99]``.
        order : int or None, default None
            Number of nodes.  ``None`` selects it from the interval, ``n`` and
            ``tol`` via :func:`~._chebyshev.cheb_order_for_tolerance`.
        tol : float, optional
            Target absolute error when ``order`` is ``None``.  Defaults to the
            relative target ``DEFAULT_CHEB_RTOL · n``.
        """
        rho_min, rho_max = _clamp_interval(rho_min, rho_max)
        if rho_max <= rho_min:
            raise ValueError(
                f"Invalid rho interval: rho_min={rho_min}, rho_max={rho_max}."
            )
        if order is None:
            order = cheb_order_for_tolerance(rho_min, rho_max, self.n, tol=tol)

        rho_nodes, _ = chebyshev_gauss_nodes(order, rho_min, rho_max)
        coeffs = chebyshev_coeffs_dct1(self.logdet_at(rho_nodes))

        return CholChebPrecompute(
            coeffs=coeffs,
            rho_min=rho_min,
            rho_max=rho_max,
            order=order,
            n=self.n,
            err_est=cheb_tail_error(coeffs),
        )


def chol_cheb_logdet_precompute(
    W,
    order: int | None = None,
    rho_min: float = 0.1,
    rho_max: float = 0.8,
    tol: float | None = None,
) -> CholChebPrecompute:
    """Precompute Chebyshev coefficients via sparse Cholesky log-determinant.

    Evaluates ``log|det(I - ρW)|`` exactly at ``order`` Chebyshev nodes
    in ``[ρ_min, ρ_max]`` via sparse Cholesky factorisation, then fits
    a Chebyshev polynomial in ``ρ``.

    **Symbolic reuse**: all ``I - ρW`` matrices share the same sparsity
    pattern, so CHOLMOD's symbolic analysis (AMD ordering + elimination
    tree) is performed only once and reused for all subsequent numeric
    factorisations.

    This is a one-shot convenience wrapper over :class:`CholChebContext`.  To
    fit more than one interval for the same ``W`` — a warmup-adaptive refit —
    hold the context and call :meth:`CholChebContext.coeffs_on` repeatedly, so
    the symmetrisation and symbolic analysis are paid once rather than per fit.

    Parameters
    ----------
    W : array-like or scipy.sparse matrix
        Spatial weights matrix (dense or sparse, row-standardised).
    order : int or None, default None
        Chebyshev polynomial degree.  If ``None``, selected from the interval's
        Bernstein-ellipse convergence rate, ``n``, and ``tol``.
    rho_min : float, default 0.1
        Lower bound of the ρ approximation interval.  Clamped to -0.99
        to avoid the singularity at ρ = -1.
    rho_max : float, default 0.8
        Upper bound of the ρ approximation interval.  Clamped to 0.99
        to avoid the singularity at ρ = 1.
    tol : float, optional
        Target absolute error used to pick ``order`` when it is ``None``.
        Defaults to the relative target ``DEFAULT_CHEB_RTOL · n``.

    Returns
    -------
    CholChebPrecompute
        Precomputed Chebyshev coefficients.
    """
    return CholChebContext(W).coeffs_on(
        rho_min=rho_min, rho_max=rho_max, order=order, tol=tol
    )


def chol_cheb_logdet_eval(pre: CholChebPrecompute, rho: float) -> float:
    """Evaluate ``log|I - ρW|`` from precomputed Chebyshev coefficients.

    Uses Clenshaw recurrence: ``O(order)`` per evaluation.

    Parameters
    ----------
    pre : CholChebPrecompute
        Precomputed coefficients from :func:`chol_cheb_logdet_precompute`.
    rho : float
        Spatial autoregressive parameter.
    """
    return clenshaw_scalar(pre.coeffs, rho, pre.rho_min, pre.rho_max)


def chol_cheb_logdet_eval_vec(
    pre: CholChebPrecompute, rho_arr: np.ndarray
) -> np.ndarray:
    """Vectorized evaluation over an array of ρ values."""
    return clenshaw_vec(pre.coeffs, rho_arr, pre.rho_min, pre.rho_max)
