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

import importlib
import os
import time
import warnings
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


def _lu_logdet_from_factor(factor) -> float:
    """Recover ``log|det(A)|`` from a ``scikit-sparse`` LU factor.

    Both bindings factorize ``P R A Q = L U`` with a diagonal row scaling ``R``;
    the permutations affect only the sign, so
    ``log|det(A)| = Σ log|diag(U)| + Σ log|diag(L)| - Σ log|diag(R)|``.
    Verified identical between ``sksparse.klu`` and ``sksparse.umfpack``.

    **This is not free.**  ``factor.U`` and ``factor.L`` build whole scipy
    matrices in order to read ``n`` numbers off their diagonals, at a cost that
    scales with ``nnz(L) + nnz(U)`` — i.e. with fill-in.  Measured at
    ``n = 3,000`` on a directed graph, it is **52% of KLU's per-node cost at
    mean degree 2**, 20% at degree 8, and 3% at degree 65.  UMFPACK escapes it
    via :func:`_umf_logdet_from_factor`; KLU exposes no equivalent, so this is
    the only route there.
    """
    logdet = float(np.sum(np.log(np.abs(factor.U.diagonal()))))
    l_diag = factor.L.diagonal()
    logdet += float(np.sum(np.log(np.abs(l_diag))))
    rscale = factor.rscale
    if rscale is not None:
        logdet -= float(np.sum(np.log(np.abs(rscale))))
    return logdet


# Retained under its original name: the diagonal formula is shared by both
# backends, and this alias keeps existing importers working.
_klu_logdet_from_factor = _lu_logdet_from_factor


def _umf_logdet_from_factor(factor) -> float:
    """Recover ``log|det(A)|`` from a ``sksparse.umfpack`` factor via UMFPACK.

    ``UMFFactor.slogdet`` calls ``umfpack_di_get_determinant``, which returns
    the determinant split into mantissa and base-10 exponent and costs ``O(n)``
    inside the library.  It therefore avoids the scipy ``L``/``U``
    materialization that :func:`_lu_logdet_from_factor` pays for — measured
    10-20% of UMFPACK's per-node cost at ``n = 3,000``, and constant at ~0.04ms
    against an extraction that grows to 7ms at mean degree 65.

    The split form is what keeps it usable where the determinant itself is far
    outside double range: verified against the diagonal formula to 2e-11 at
    ``n = 300,000`` with ``log|det| = -8155`` (``det ≈ 1e-3542``), with no
    overflow or underflow warning raised.
    """
    return float(factor.slogdet()[1])


# name -> (module, factory attribute, logdet extractor).  Order is the
# preference order used when routing is disabled or only one backend imports.
_LU_BACKENDS: dict[str, tuple[str, str, object]] = {
    "klu": ("sksparse.klu", "klu_factor", _lu_logdet_from_factor),
    "umfpack": ("sksparse.umfpack", "umf_factor", _umf_logdet_from_factor),
}


def _lu_backend_preference() -> str:
    """Resolve the LU backend policy for the AAA/spline coarse grids.

    Environment
    -----------
    BAYESPECON_LOGDET_LU_BACKEND : {"auto", "klu", "umfpack", "scipy"}
        Default ``auto``, which times both backends on the grid itself and
        keeps the faster (see :class:`_ReusableLULogdet`).  Naming a backend
        pins it and skips the probe; ``scipy`` forces SuperLU.
    """
    requested = os.environ.get("BAYESPECON_LOGDET_LU_BACKEND", "auto").strip().lower()
    if requested in {"", "auto"}:
        return "auto"
    if requested in {"scipy", "superlu"}:
        return "scipy"
    if requested in _LU_BACKENDS:
        return requested
    warnings.warn(
        f"Unknown BAYESPECON_LOGDET_LU_BACKEND={requested!r}. Valid values are: "
        "auto, klu, umfpack, scipy. Falling back to auto.",
        RuntimeWarning,
        stacklevel=2,
    )
    return "auto"


def _load_lu_backend(name: str):
    """Import one backend, returning ``(factory, logdet_fn)``."""
    module_name, attr, logdet_fn = _LU_BACKENDS[name]
    module = importlib.import_module(module_name)
    return getattr(module, attr), logdet_fn


def _superlu_logdet(A: sp.csc_matrix) -> float:
    """``log|det(A)|`` via scipy SuperLU — the backend-of-last-resort."""
    from scipy.sparse.linalg import splu as scipy_splu

    lu = scipy_splu(A)
    return float(
        np.sum(np.log(np.abs(lu.L.diagonal())))
        + np.sum(np.log(np.abs(lu.U.diagonal())))
    )


def _lu_logdet(A: sp.csc_matrix) -> float:
    """Compute ``log|det(A)|`` via sparse LU factorization (single shot).

    Walks the backend ladder — KLU, then UMFPACK, then scipy SuperLU — taking
    the first that imports and factorizes.  For repeated factorizations of
    matrices sharing a sparsity pattern (the AAA coarse grid), use
    :func:`_make_reusable_lu_logdet` instead: it reuses the symbolic analysis
    *and* routes between the two SuiteSparse backends by measurement.
    """
    A = A.tocsc()
    for name in _LU_BACKENDS:
        try:
            factory, logdet_fn = _load_lu_backend(name)
            return logdet_fn(factory(A))
        except Exception:
            continue
    return _superlu_logdet(A)


class _ReusableLULogdet:
    """``A -> log|det(A)|`` reusing symbolic analysis, routed by measurement.

    Two independent things happen here.

    **Symbolic reuse.**  The first call for a given backend pays symbolic
    analysis plus a numeric factorization; every later call refactorizes
    numerically only (``factorize``), which is valid because every ``I - ρW``
    on the grid shares one sparsity pattern.  This mirrors the CHOLMOD symbolic
    reuse in :func:`chol_cheb_logdet_precompute`.

    **Backend routing.**  KLU and UMFPACK trade places on this workload, and
    the crossover is a property of the graph, not a constant.  Measured at
    ``n = 3,000`` over a 16-node Chebyshev grid on a directed graph, total grid
    time in ms:

    ==========  =========  =========  ===========
    mean deg          KLU    UMFPACK  winner
    ==========  =========  =========  ===========
    2                 6.2       17.4  KLU 2.8×
    5                23.4       64.6  KLU 2.8×
    8                92.5       98.3  KLU 1.1×
    12              251.3      152.5  UMF 1.7×
    21             1025.1      224.0  UMF 4.6×
    40             4414.5      333.6  UMF 13.2×
    65            11763.5      628.8  UMF 18.7×
    ==========  =========  =========  ===========

    So a pinned backend is wrong by up to 18.7× at one end or 2.8× at the
    other, and **a mean-degree cutoff is wrong too** — the crossover moves with
    ``n``, so a constant calibrated at one size misroutes at every other size.
    Instead the grid times itself: the **first** call factorizes with every
    surviving backend, and the fastest keeps the work.  Later calls refactorize
    on the winner alone.

    The timings cover extraction as well as factorization, because the two
    backends do not extract alike: KLU must materialize ``L`` and ``U`` to read
    their diagonals, which is 52% of its per-node cost at mean degree 2, while
    UMFPACK reads the determinant out of the library in ``O(n)``
    (:func:`_umf_logdet_from_factor`).  Timing factorization alone would route
    on the wrong quantity.

    **Why one probe call and not two.**  A first call also pays symbolic
    analysis, so it overstates steady-state cost — by 1.1-3.8× for KLU and
    1.2-4.7× for UMFPACK — and the tempting fix is to route on a second,
    numeric-only call instead.  It is not worth it.  Measured across nine
    densities, the first call already picks the same backend as the
    steady-state cost everywhere except mean degree 8, where the true gap is
    1.04× and misrouting therefore costs 4%.  The inflation largely cancels
    because it falls hardest on whichever backend has the cheaper numeric
    phase, which is the winner.

    A second call would double the one cost that actually bites.  The probe's
    overhead is one factorization on each losing backend, and at mean degree 65
    a single KLU factorization (812ms) already exceeds the entire 16-node
    UMFPACK grid (634ms) — so the probe is bounded by roughly one node, but a
    node can be a large fraction of the grid precisely where the backends
    diverge most.  Measured end to end on the same grids, against the KLU-only
    path this replaces:

    ==========  =========  =========  ===========
    mean deg      routed    KLU-only  change
    ==========  =========  =========  ===========
    2                 7.2        4.6  1.6× slower
    5                28.9       22.9  1.3× slower
    8               107.7       89.9  1.2× slower
    12              181.0      250.3  1.4× faster
    21              340.1     1010.4  3.0× faster
    40              645.8     4431.5  6.9× faster
    65             1439.1    12084.7  8.4× faster
    ==========  =========  =========  ===========

    The regression is confined to the regime where the grid is a few
    milliseconds outright, so it is bounded in absolute terms (+2.6ms at degree
    2, +18ms at degree 8) against savings of seconds at the dense end.  That
    asymmetry is the whole case for probing: the measurement costs the most
    exactly where it matters least.

    One numeric factor per candidate is held during the first call, so peak
    memory there is that of the candidates together; the losers are freed as
    soon as the decision is taken.

    Parameters
    ----------
    backend : {"auto", "klu", "umfpack", "scipy"} or None, optional
        Overrides ``BAYESPECON_LOGDET_LU_BACKEND``.  Naming a backend pins it
        and skips the probe entirely.

    Attributes
    ----------
    backend : str
        The backend in use: ``"klu"``, ``"umfpack"``, ``"scipy"``, or
        ``"probing"`` before the routing decision is taken.
    """

    __slots__ = ("_candidates", "_factors", "_timings", "_probing")

    def __init__(self, backend: str | None = None) -> None:
        preference = backend if backend is not None else _lu_backend_preference()
        if preference == "scipy":
            candidates: list[str] = []
        elif preference == "auto":
            candidates = list(_LU_BACKENDS)
        else:
            candidates = [preference]
        self._candidates = candidates
        self._factors: dict[str, object] = {}
        self._timings: dict[str, float] = {}
        # Only worth probing when there is an actual choice to make.
        self._probing = preference == "auto" and len(candidates) > 1

    @property
    def backend(self) -> str:
        if self._probing:
            return "probing"
        return self._candidates[0] if self._candidates else "scipy"

    def _drop(self, name: str, exc: Exception) -> None:
        """Retire a backend that failed, saying so rather than failing silently.

        The distinction this preserves is the one a bare ``except Exception``
        loses: a backend that is merely absent, and a backend that broke on
        *this* matrix, both retire — but only to the next SuiteSparse backend,
        never straight past it to SuperLU.
        """
        if name in self._candidates:
            self._candidates.remove(name)
        self._factors.pop(name, None)
        self._timings.pop(name, None)
        remaining = self._candidates[0] if self._candidates else "scipy SuperLU"
        if not isinstance(exc, ImportError):
            warnings.warn(
                f"Sparse LU backend {name!r} failed during log-determinant "
                f"evaluation ({type(exc).__name__}: {exc}); falling back to "
                f"{remaining}.",
                RuntimeWarning,
                stacklevel=3,
            )

    def _evaluate_one(self, name: str, A) -> tuple[float, float]:
        """Factorize with ``name`` and return ``(logdet, elapsed_seconds)``."""
        factory, logdet_fn = _load_lu_backend(name)
        factor = self._factors.get(name)
        start = time.perf_counter()
        if factor is None:
            factor = factory(A)
        else:
            factor.factorize(A)
        value = logdet_fn(factor)
        elapsed = time.perf_counter() - start
        self._factors[name] = factor
        return value, elapsed

    def __call__(self, A) -> float:
        A = A.tocsc()
        value: float | None = None

        # Probe phase (first call only): every candidate factorizes, fastest wins.
        if self._probing:
            for name in list(self._candidates):
                try:
                    result, elapsed = self._evaluate_one(name, A)
                except Exception as exc:  # noqa: BLE001 - retire, don't abort
                    self._drop(name, exc)
                    continue
                if value is None:
                    value = result
                self._timings[name] = elapsed
            self._settle()
            if value is not None:
                return value
            return _superlu_logdet(A)

        # Settled: the chosen backend, with the ladder still beneath it.
        while self._candidates:
            name = self._candidates[0]
            try:
                return self._evaluate_one(name, A)[0]
            except Exception as exc:  # noqa: BLE001 - retire, don't abort
                self._drop(name, exc)
        return _superlu_logdet(A)

    def _settle(self) -> None:
        """Keep the backend that was fastest on the probe call."""
        self._probing = False
        if not self._timings:
            return
        winner = min(self._timings, key=self._timings.__getitem__)
        self._candidates = [winner] + [n for n in self._candidates if n != winner]
        for name in list(self._factors):
            if name != winner:
                # Free the losers' factors; fill-in makes these large.
                del self._factors[name]


def _make_reusable_lu_logdet(backend: str | None = None):
    """Return a callable ``A -> log|det(A)|`` that reuses symbolic analysis.

    Thin constructor for :class:`_ReusableLULogdet`; see that class for the
    symbolic-reuse and backend-routing behaviour.
    """
    return _ReusableLULogdet(backend=backend)


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
