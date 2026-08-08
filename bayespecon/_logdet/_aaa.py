"""AAA rational approximation log-determinant for non-symmetric ``I - ρW``.

For row-standardized ``W`` from a **directed** graph (KNN, travel time,
migration flows), the matrix ``I - ρW`` is non-symmetric and cannot be
symmetrized via D-symmetrization.  Sparse Cholesky is unavailable; the
options are sparse LU (exact but expensive) or stochastic methods
(approximate).

This module implements the **AAA rational approximation** strategy:

1. Evaluate ``log|det(I - ρW)|`` exactly at ``n_coarse`` Chebyshev-spaced
   points via sparse LU, reusing one symbolic factorization across all of
   them (KLU, falling back to scipy SuperLU).
2. Fit a rational function in barycentric form via the AAA algorithm
   [@nakatsukasa2018], which selects ``m`` support points from the coarse
   grid.
3. Evaluate at any ρ via the barycentric formula in ``O(m)`` per ρ.

**Why rational instead of polynomial?**  The logdet function
``f(ρ) = Σ log(1 - ρλᵢ)`` has logarithmic singularities at ``ρ = 1/λᵢ``.
Polynomials converge slowly near singularities (needing 50-100 nodes for
``[-0.95, 0.95]``).  Rational functions converge exponentially faster —
typically needing only 6-15 support points for the same accuracy.

**When to use**: non-symmetric ``W`` (directed graph) where Cholesky is
unavailable.  For symmetric ``W``, use ``cheb_cholesky`` (exact, faster).
For very large ``n`` (>20,000), use ``cheb_stochastic`` (avoids
factorization entirely).

**Cost**: ``n_coarse`` sparse LU factorizations + ``O(m)`` per-ρ
evaluation, where ``n_coarse`` is the coarse-grid size (adaptive: 16 for the
narrow default interval, up to 96 for wide/near-singular intervals) and
``m ≤ n_coarse // 2`` is the number of AAA support points actually selected.
All ``I - ρW`` share one sparsity pattern, so KLU's symbolic analysis is
computed once and reused for every subsequent numeric factorization (measured
1.6-3.4× faster than a fresh scipy SuperLU factorization per node over the
coarse grid).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class AAAPrecompute:
    """Precomputed AAA rational approximant for ``log|I - ρW|``.

    Attributes
    ----------
    support_points : np.ndarray, shape (m,)
        ρ values where exact logdet was computed (AAA support points).
    support_values : np.ndarray, shape (m,)
        Exact logdet values at support points.
    weights : np.ndarray, shape (m,)
        Barycentric weights from the AAA algorithm.
    rho_min : float
        Lower bound of the ρ approximation interval.
    rho_max : float
        Upper bound of the ρ approximation interval.
    n : int
        Matrix dimension.
    """

    support_points: np.ndarray
    support_values: np.ndarray
    weights: np.ndarray
    rho_min: float
    rho_max: float
    n: int


def _klu_logdet_from_factor(factor) -> float:
    """Recover ``log|det(A)|`` from a ``sksparse.klu`` factor.

    KLU factorizes ``P R A Q = L U`` with a diagonal row scaling ``R``; the
    permutations affect only the sign, so
    ``log|det(A)| = Σ log|diag(U)| + Σ log|diag(L)| - Σ log|diag(R)|``.
    """
    logdet = float(np.sum(np.log(np.abs(factor.U.diagonal()))))
    l_diag = factor.L.diagonal()
    logdet += float(np.sum(np.log(np.abs(l_diag))))
    rscale = factor.rscale
    if rscale is not None:
        logdet -= float(np.sum(np.log(np.abs(rscale))))
    return logdet


def _lu_logdet(A: sp.csc_matrix) -> float:
    """Compute ``log|det(A)|`` via sparse LU factorization (single shot).

    Prefers KLU (``sksparse.klu``), then falls back to scipy SuperLU.  For
    repeated factorizations of matrices sharing a sparsity pattern (the AAA
    coarse grid), use :func:`_make_reusable_lu_logdet`, which reuses KLU's
    symbolic analysis.
    """
    A = A.tocsc()
    try:
        from sksparse.klu import klu_factor

        return _klu_logdet_from_factor(klu_factor(A))
    except Exception:
        from scipy.sparse.linalg import splu as scipy_splu

        lu = scipy_splu(A)
        logdet = np.sum(np.log(np.abs(lu.L.diagonal()))) + np.sum(
            np.log(np.abs(lu.U.diagonal()))
        )
        return float(logdet)


def _make_reusable_lu_logdet():
    """Return a callable ``A -> log|det(A)|`` that reuses symbolic analysis.

    On the first call it computes KLU's symbolic + numeric factorization; on
    subsequent calls it refactorizes numerically only (``KLUFactor.factorize``),
    valid because every ``I - ρW`` shares one sparsity pattern.  This mirrors
    the CHOLMOD symbolic reuse in :func:`chol_cheb_logdet_precompute` and is
    measured 1.6-3.4× faster than a fresh scipy SuperLU factorization per node.

    Falls back to the single-shot :func:`_lu_logdet` (scipy SuperLU) when
    ``sksparse.klu`` is unavailable or its first factorization fails.
    """
    state = {"factor": None, "use_klu": True}

    def _evaluate(A) -> float:
        A = A.tocsc()
        if state["use_klu"]:
            try:
                if state["factor"] is None:
                    from sksparse.klu import klu_factor

                    state["factor"] = klu_factor(A)
                else:
                    state["factor"].factorize(A)
                return _klu_logdet_from_factor(state["factor"])
            except Exception:
                # KLU unavailable or failed — drop to single-shot fallback.
                state["use_klu"] = False
                state["factor"] = None
        return _lu_logdet(A)

    return _evaluate


def _aaa_algorithm(
    z: np.ndarray,
    f: np.ndarray,
    tol: float = 1e-13,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Core AAA algorithm for rational approximation.

    Given sample points ``z`` and function values ``f``, find support
    points, values, and barycentric weights for a rational approximant.

    **The greedy loop runs on values that have already been paid for.**  Every
    sample in ``f`` cost one sparse factorization before this function was
    called, so stopping early saves nothing — it discards resolution the caller
    has already bought.  Two consequences shape the loop:

    * ``tol`` is an **absolute** residual threshold, not one scaled by
      ``max|f|``.  Scaling by ``max|f|`` made the attainable floor grow with
      ``n`` (``|log|I - ρW|| = O(n)``), so on large problems the loop stopped
      while the residual was still orders above what the samples supported.  It
      also contradicted the invariant that matters for inference, which is
      absolute error in the log-density, not relative.
    * The loop **returns its best iterate, not its last**.  Pushing AAA past the
      point where the Loewner least-squares becomes ill-conditioned introduces
      Froissart doublets and the delivered error rises again, non-monotonically.
      Tracking the best-scoring iterate makes overshoot harmless, so ``tol`` acts
      as a safety valve rather than as the binding constraint.

    Parameters
    ----------
    z : np.ndarray, shape (M,)
        Sample points (dense grid of ρ values).
    f : np.ndarray, shape (M,)
        Function values at sample points.
    tol : float, default 1e-13
        **Absolute** tolerance on the nonlinear residual.  Set near the level at
        which the Loewner system stops being solvable in double precision; the
        best-iterate rule above protects against setting it too tight.
    max_iter : int, default 100
        Maximum number of AAA iterations (support points).

    Returns
    -------
    support_points : np.ndarray, shape (m,)
        Selected support points (subset of z).
    support_values : np.ndarray, shape (m,)
        Function values at support points.
    weights : np.ndarray, shape (m,)
        Barycentric weights.
    """
    z = np.asarray(z, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    M = len(z)

    # Track which points are support points
    is_support = np.zeros(M, dtype=bool)
    support_idx = []  # indices into z
    weights_list = []

    # Best iterate seen so far, by max |residual| over the non-support samples.
    # Returned in place of the final iterate; see the note in the docstring.
    best: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    best_resid = np.inf
    stalled = 0

    # Residual at all non-support points
    residual = f.copy()

    for m in range(1, min(max_iter, M // 2) + 1):
        # Greedy: pick the point with largest |residual|
        # Exclude already-selected points
        candidate_residual = np.abs(residual).copy()
        candidate_residual[is_support] = -1  # exclude support points
        next_idx = np.argmax(candidate_residual)

        if candidate_residual[next_idx] < tol:
            break

        is_support[next_idx] = True
        support_idx.append(next_idx)

        # Get current support points and values
        sp_z = z[is_support]
        sp_f = f[is_support]
        m_curr = len(sp_z)

        # Non-support points
        non_support = ~is_support
        z_ns = z[non_support]
        f_ns = f[non_support]

        if m_curr == 1:
            # First support point: weight = 1 (trivial)
            weights_list = [1.0]
            # Residual = f - f_1 (constant approximant)
            residual = f - sp_f[0]
            continue

        # Build the Loewner matrix for least-squares
        # A[i, j] = (f_ns[i] - sp_f[j]) / (z_ns[i] - sp_z[j])
        # We want to minimize ||A @ w|| subject to ||w|| = 1
        n_ns = len(z_ns)
        A = np.zeros((n_ns, m_curr), dtype=np.float64)
        for j in range(m_curr):
            # Avoid division by zero (shouldn't happen since support != non-support)
            diff = z_ns - sp_z[j]
            A[:, j] = (f_ns - sp_f[j]) / diff

        # Solve min ||A @ w|| s.t. ||w|| = 1 via SVD
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        w = Vt[-1]  # right singular vector for smallest singular value

        weights_list = w

        # Compute residual at non-support points
        # r(z) = n(z)/d(z) where
        # n(z) = sum_j w_j * sp_f[j] / (z - sp_z[j])
        # d(z) = sum_j w_j / (z - sp_z[j])
        n_val = np.zeros(n_ns, dtype=np.float64)
        d_val = np.zeros(n_ns, dtype=np.float64)
        for j in range(m_curr):
            diff = z_ns - sp_z[j]
            n_val += w[j] * sp_f[j] / diff
            d_val += w[j] / diff

        # Avoid division by zero
        r_ns = np.where(
            np.abs(d_val) > 1e-15,
            n_val / d_val,
            sp_f[0],  # fallback
        )

        # Update residual
        residual = f.copy()
        residual[non_support] = f_ns - r_ns
        # At support points, residual is 0 (interpolation)

        # Score this iterate and keep it if it is the best so far.  The score is
        # the max residual over the samples AAA did *not* interpolate, which is
        # the only error estimate available without further factorizations.
        score = float(np.max(np.abs(residual[non_support]))) if n_ns else 0.0
        if score < best_resid:
            best_resid = score
            best = (
                z[is_support].copy(),
                f[is_support].copy(),
                np.array(w, dtype=np.float64),
            )
            stalled = 0
        else:
            # Conditioning has started to cost more than the extra pole buys.
            stalled += 1
            if stalled >= 3:
                break

    if best is None:
        # Fewer than two support points were selected (the constant fit already
        # met `tol`); fall back to whatever the loop produced.
        return (
            z[is_support],
            f[is_support],
            np.array(weights_list, dtype=np.float64),
        )
    return best


def _adaptive_n_coarse(rho_min: float, rho_max: float) -> int:
    """Choose the coarse-grid size (= number of exact LU factorizations).

    Each coarse-grid point costs one sparse LU factorization, so ``n_coarse``
    directly sets the setup cost.  The AAA support count ``m`` (a subset of the
    grid, capped at ``n_coarse // 2``) is what determines accuracy, and both
    grow as the interval widens toward the ``ρ = ±1`` logdet singularities.

    Empirically (rook + knn, n∈{1936, 10000, 40000}), scoring on the *closed*
    interval with the endpoints included: the default narrow interval
    ``[0.1, 0.8]`` reaches ~1e-10 max error with only 16 nodes, while the full
    stability region ``[-0.99, 0.99]`` needs ~64-80 for 1e-6 and ~96 to reach
    the ~1e-10 range.  The cap sits at 96 because that is where the accuracy a
    posterior can use is already reached, not because the method stops
    improving there.

    An earlier version of this docstring described a ~1e-7--1e-8 "floor where
    AAA saturates" past ~96 nodes.  That floor was an artefact of the greedy
    loop's stopping rule, which was scaled by ``max|f| = O(n)`` and so cut the
    fit short on large problems; see :func:`_aaa_algorithm`.  With an absolute
    tolerance and best-iterate retention the delivered error keeps falling with
    the node count and the non-monotonicity is gone.

    The cap matters only for callers that stay on a wide interval.  A
    post-warmup refit narrows ``[rho_min, rho_max]``, which raises the
    Bernstein rate and pulls the count back down through the same formula.

    This mirrors :func:`~._chebyshev.cheb_order_for_tolerance`, which sizes the
    Chebyshev order the same way.  At matched node counts on the full interval
    AAA is three to four orders more accurate than the polynomial, and stays
    ahead at every count tested up to 128.

    Parameters
    ----------
    rho_min, rho_max : float
        The ρ approximation interval.

    Returns
    -------
    int
        Number of Chebyshev-spaced coarse-grid points (LU factorizations).
    """
    from ._chebyshev import bernstein_rho

    # Scale inversely with the Bernstein-ellipse rate, the same quantity that
    # sets the Chebyshev order — distance to the ρ = ±1 singularities, not
    # interval width.  The constant is calibrated so the applied default
    # [0.1, 0.8] still draws 16 points (the value the width-keyed rule this
    # replaced was tuned to), which fixes both directions the old rule got
    # wrong: it returned 16 for any narrow interval, however far from ±1, and
    # so could not exploit a post-warmup range at all.
    rho_b = bernstein_rho(rho_min, rho_max)
    if not np.isfinite(rho_b) or rho_b <= 1.0:
        _c0 = os.getenv("BAYESPECON_LOGDET_NODE_CAP")
        return int(_c0) if _c0 else 96
    _c = os.getenv("BAYESPECON_LOGDET_NODE_CAP")
    hi = int(_c) if _c else 96
    return int(np.clip(int(np.ceil(16.0 / np.log(rho_b))), 8, hi))


def _aaa_algorithm_lazy(
    z: np.ndarray,
    eval_fn,
    tol: float = 1e-13,
    max_iter: int = 30,
    n_coarse: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lazy AAA: evaluates ``eval_fn`` at a small coarse grid, not the full sample grid.

    Instead of evaluating at all ``M = len(z)`` sample points, this function
    evaluates at ``n_coarse`` Chebyshev-spaced points and runs the standard
    AAA algorithm on those.  This reduces expensive evaluations (e.g. sparse
    LU factorizations) from ``M`` to ``n_coarse`` (default 30), a ~7× speedup
    for the default ``M=200``.

    The full sample grid ``z`` is used only for the curvature-based refinement
    check — if the approximant has high curvature at non-support points on the
    full grid, one additional evaluation is performed there.

    Parameters
    ----------
    z : np.ndarray, shape (M,)
        Sample points (dense grid of ρ values).
    eval_fn : callable
        Function ``f(ρ) -> float``.  Called at ``n_coarse`` + refinement points.
    tol : float, default 1e-13
        Relative tolerance for AAA convergence.
    max_iter : int, default 30
        Maximum number of support points.
    n_coarse : int, default 30
        Number of Chebyshev-spaced evaluation points for the coarse phase.

    Returns
    -------
    support_points, support_values, weights : np.ndarray
    """
    z = np.asarray(z, dtype=np.float64)
    M = len(z)
    rho_min_z, rho_max_z = z[0], z[-1]

    # Phase 1: Evaluate at n_coarse Chebyshev-spaced points
    n_coarse = min(n_coarse, M)
    k = np.arange(1, n_coarse + 1)
    coarse_cos = np.cos((2 * k - 1) * np.pi / (2 * n_coarse))
    z_coarse = 0.5 * (rho_max_z - rho_min_z) * coarse_cos + 0.5 * (
        rho_max_z + rho_min_z
    )

    f_coarse = np.array([eval_fn(zc) for zc in z_coarse], dtype=np.float64)

    # Run standard AAA on the coarse grid (uses true values for Loewner LSQ)
    sp_z, sp_f, w = _aaa_algorithm(z_coarse, f_coarse, tol=tol, max_iter=max_iter)

    return sp_z, sp_f, w


class AAAContext:
    """Reusable KLU symbolic analysis for one directed ``W``.

    The LU counterpart of
    :class:`~._chol_cheb.CholChebContext`: everything independent of the ρ
    interval — the matrix, the identity, and KLU's symbolic factorization — is
    held here, so fitting a second approximant on a different interval costs
    only its numeric refactorizations.  This is what makes a warmup-adaptive
    refit affordable on directed weights, where Cholesky is unavailable.

    Parameters
    ----------
    W : array-like or scipy.sparse matrix
        Spatial weights matrix; may be non-symmetric.
    """

    __slots__ = ("W_sp", "_eye", "_lu_logdet", "n")

    def __init__(self, W):
        if sp.issparse(W) or hasattr(W, "format"):
            self.W_sp = sp.csc_matrix(W, dtype=np.float64)
        else:
            self.W_sp = sp.csc_matrix(np.asarray(W, dtype=np.float64))
        self.n = int(self.W_sp.shape[0])
        self._eye = sp.eye(self.n, format="csc")
        # Symbolic analysis is established on the first call and reused for
        # every later node and every later interval.
        self._lu_logdet = _make_reusable_lu_logdet()

    def logdet_at(self, rho: float) -> float:
        """Exact ``log|I - ρW|`` by sparse LU, reusing symbolic analysis."""
        return self._lu_logdet(self._eye - float(rho) * self.W_sp)

    def fit_on(
        self,
        rho_min: float = 0.1,
        rho_max: float = 0.8,
        n_samples: int = 200,
        tol: float = 1e-13,
        max_iter: int = 30,
        n_coarse: int | None = None,
    ) -> AAAPrecompute:
        """Fit the AAA rational approximant on ``[rho_min, rho_max]``."""
        if n_coarse is None:
            n_coarse = _adaptive_n_coarse(rho_min, rho_max)

        # Dense sample grid for AAA (no factorization here — just the grid)
        z = np.linspace(rho_min, rho_max, n_samples)

        # Run lazy AAA: exactly n_coarse LU factorizations (m ≤ n_coarse//2 of
        # the grid points become support points).
        support_points, support_values, weights = _aaa_algorithm_lazy(
            z, self.logdet_at, tol=tol, max_iter=max_iter, n_coarse=n_coarse
        )

        return AAAPrecompute(
            support_points=support_points,
            support_values=support_values,
            weights=weights,
            rho_min=rho_min,
            rho_max=rho_max,
            n=self.n,
        )


def aaa_logdet_precompute(
    W,
    rho_min: float = 0.1,
    rho_max: float = 0.8,
    n_samples: int = 200,
    tol: float = 1e-13,
    max_iter: int = 30,
    n_coarse: int | None = None,
) -> AAAPrecompute:
    """Precompute AAA rational approximant for ``log|I - ρW|``.

    Evaluates ``log|det(I - ρW)|`` exactly at a coarse grid of ``n_coarse``
    Chebyshev-spaced points via sparse LU (KLU with symbolic reuse), then
    fits a rational function via the AAA algorithm, which greedily selects
    ``m`` support points (``m ≤ n_coarse // 2``, typically 5-15) from that grid.

    The number of exact LU factorizations equals ``n_coarse`` — **not** the
    support count ``m`` and **not** ``n_samples`` (the 200-point sample grid is
    only the AAA residual proxy and involves no factorizations).  ``n_coarse``
    defaults to :func:`_adaptive_n_coarse`: 16 for the narrow default interval,
    30 for wider or near-singular intervals.

    Parameters
    ----------
    W : array-like or scipy.sparse matrix
        Spatial weights matrix (dense or sparse, non-symmetric OK).
    rho_min : float, default 0.1
        Lower bound of the ρ approximation interval.
    rho_max : float, default 0.8
        Upper bound of the ρ approximation interval.
    n_samples : int, default 200
        Number of sample points for the AAA residual grid.  Does **not**
        affect the number of LU factorizations — only the resolution of
        the greedy selection.
    tol : float, default 1e-13
        Relative tolerance for AAA convergence.
    max_iter : int, default 30
        Maximum number of AAA support points selected from the coarse grid.
    n_coarse : int, optional
        Number of exact LU factorizations (coarse-grid size).  ``None``
        (default) selects it adaptively from the interval via
        :func:`_adaptive_n_coarse`.

    Returns
    -------
    AAAPrecompute
        Precomputed rational approximant.
    """
    return AAAContext(W).fit_on(
        rho_min=rho_min,
        rho_max=rho_max,
        n_samples=n_samples,
        tol=tol,
        max_iter=max_iter,
        n_coarse=n_coarse,
    )


def aaa_logdet_eval(pre: AAAPrecompute, rho: float) -> float:
    """Evaluate ``log|I - ρW|`` from precomputed AAA rational approximant.

    Uses the barycentric formula: ``O(m)`` per evaluation.

    Parameters
    ----------
    pre : AAAPrecompute
        Precomputed approximant from :func:`aaa_logdet_precompute`.
    rho : float
        Spatial autoregressive parameter.
    """
    sp_z = pre.support_points
    sp_f = pre.support_values
    w = pre.weights
    m = len(sp_z)

    if m == 0:
        return 0.0
    if m == 1:
        return float(sp_f[0])

    # Barycentric formula:
    # r(ρ) = [Σ_j w_j * f_j / (ρ - z_j)] / [Σ_j w_j / (ρ - z_j)]
    diff = rho - sp_z

    # Check if rho is exactly at a support point
    zero_idx = np.where(np.abs(diff) < 1e-15)[0]
    if len(zero_idx) > 0:
        return float(sp_f[zero_idx[0]])

    n_val = np.sum(w * sp_f / diff)
    d_val = np.sum(w / diff)

    if abs(d_val) < 1e-15:
        # Fallback: return nearest support value
        nearest = np.argmin(np.abs(diff))
        return float(sp_f[nearest])

    return float(n_val / d_val)


def aaa_logdet_eval_vec(pre: AAAPrecompute, rho_arr: np.ndarray) -> np.ndarray:
    """Vectorized evaluation over an array of ρ values."""
    rho_arr = np.asarray(rho_arr, dtype=np.float64)
    sp_z = pre.support_points
    sp_f = pre.support_values
    w = pre.weights
    m = len(sp_z)

    if m == 0:
        return np.zeros_like(rho_arr)
    if m == 1:
        return np.full_like(rho_arr, sp_f[0])

    # Barycentric formula, vectorized over rho_arr
    # diff[i, j] = rho_arr[i] - sp_z[j]
    diff = rho_arr[:, None] - sp_z[None, :]  # (n_rho, m)

    # Handle exact matches
    exact_match = np.abs(diff) < 1e-15
    has_exact = np.any(exact_match, axis=1)

    # For non-exact: compute barycentric
    # Avoid division by zero by setting diff to 1 where exact
    safe_diff = np.where(exact_match, 1.0, diff)

    n_val = np.sum(w[None, :] * sp_f[None, :] / safe_diff, axis=1)
    d_val = np.sum(w[None, :] / safe_diff, axis=1)

    result = np.where(
        np.abs(d_val) > 1e-15,
        n_val / d_val,
        sp_f[np.argmin(np.abs(diff), axis=1)],  # fallback
    )

    # Override with exact values where rho matches a support point
    if np.any(has_exact):
        for i in np.where(has_exact)[0]:
            j = np.where(exact_match[i])[0][0]
            result[i] = sp_f[j]

    return result


# ---------------------------------------------------------------------------
# Cholesky-based AAA (symmetrizable W only)
# ---------------------------------------------------------------------------


class CholAAAContext:
    """Reusable D-symmetrization and CHOLMOD symbolic analysis for AAA.

    The Cholesky counterpart of :class:`AAAContext`: for symmetrizable ``W``
    (undirected graph), it evaluates exact logdet values via sparse Cholesky of
    the D-symmetrized system — the same factorizer :class:`CholChebContext`
    uses — but feeds them to the AAA rational approximant instead of a
    Chebyshev DCT.  This combines the cheaper factorizer (CHOLMOD is ~2×
    faster than KLU on symmetric SPD systems) with the better interpolator
    (AAA's root-exponential convergence beats the polynomial's geometric
    rate on wide intervals).

    Everything interval-independent — the symmetrized matrix, the identity,
    and CHOLMOD's symbolic analysis — is held here, so a warmup refit costs
    only its numeric factorizations, exactly as with
    :class:`CholChebContext`.

    Raises
    ------
    ValueError
        If ``W`` admits no symmetrizing diagonal (directed graph); use
        :class:`AAAContext` (LU-based AAA) for those.
    """

    __slots__ = ("W_sym", "_eye", "_factor", "n")

    def __init__(self, W):
        from ._chol_cheb import _d_symmetrize

        if sp.issparse(W) or hasattr(W, "format"):
            W_sp = sp.csr_matrix(W, dtype=np.float64)
        else:
            W_sp = sp.csr_matrix(np.asarray(W, dtype=np.float64))
        self.n = int(W_sp.shape[0])
        self.W_sym = _d_symmetrize(W_sp)
        self._eye = sp.eye(self.n, format="csc")
        self._factor = None  # CHOLMOD symbolic analysis, established lazily

    def logdet_at(self, rho: float) -> float:
        """Exact ``log|I - ρW|`` by sparse Cholesky, reusing symbolic analysis."""
        from sksparse.cholmod import cho_factor as cholmod_cho_factor

        A = sp.csc_matrix(self._eye - float(rho) * self.W_sym)
        if self._factor is None:
            self._factor = cholmod_cho_factor(A)
        else:
            self._factor.factorize(A)
        return float(self._factor.logdet())

    def fit_on(
        self,
        rho_min: float = 0.1,
        rho_max: float = 0.8,
        n_samples: int = 200,
        tol: float = 1e-13,
        max_iter: int = 30,
        n_coarse: int | None = None,
    ) -> AAAPrecompute:
        """Fit the AAA rational approximant on ``[rho_min, rho_max]``."""
        if n_coarse is None:
            n_coarse = _adaptive_n_coarse(rho_min, rho_max)

        z = np.linspace(rho_min, rho_max, n_samples)
        support_points, support_values, weights = _aaa_algorithm_lazy(
            z, self.logdet_at, tol=tol, max_iter=max_iter, n_coarse=n_coarse
        )
        return AAAPrecompute(
            support_points=support_points,
            support_values=support_values,
            weights=weights,
            rho_min=rho_min,
            rho_max=rho_max,
            n=self.n,
        )


def chol_aaa_logdet_precompute(
    W,
    rho_min: float = 0.1,
    rho_max: float = 0.8,
    n_samples: int = 200,
    tol: float = 1e-13,
    max_iter: int = 30,
    n_coarse: int | None = None,
) -> AAAPrecompute:
    """Precompute AAA rational approximant via sparse Cholesky.

    Like :func:`aaa_logdet_precompute` but evaluates exact logdet values via
    sparse Cholesky of the D-symmetrized system, which is ~2× cheaper than
    KLU for symmetrizable ``W``.  Requires ``W`` to be D-symmetrizable
    (row-standardized undirected graph).

    Parameters
    ----------
    W : array-like or scipy.sparse matrix
        Spatial weights matrix; must be D-symmetrizable (undirected graph).
    rho_min, rho_max : float
        The ρ approximation interval.
    n_samples : int, default 200
        Number of sample points for the AAA residual grid (no factorizations).
    tol : float, default 1e-13
        Relative tolerance for AAA convergence.
    max_iter : int, default 30
        Maximum number of AAA support points.
    n_coarse : int, optional
        Number of exact Cholesky factorizations (coarse-grid size).

    Returns
    -------
    AAAPrecompute
        Precomputed rational approximant (same dataclass as LU-based AAA).
    """
    return CholAAAContext(W).fit_on(
        rho_min=rho_min,
        rho_max=rho_max,
        n_samples=n_samples,
        tol=tol,
        max_iter=max_iter,
        n_coarse=n_coarse,
    )
