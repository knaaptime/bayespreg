"""Log-determinant configuration: method enum, resolution, and bounds.

Five methods are supported:

* ``"eigenvalue"`` — exact O(n) per-call after one-time O(n³) eigendecomposition.
* ``"slq"`` — stochastic Lanczos quadrature; D-symmetrised batched Lanczos
  with Gauss quadrature trace estimation → Chebyshev coefficients.
* ``"chebyshev"`` — Barry-Pace Monte Carlo traces → Chebyshev approximation; O(m) per call.
* ``"cheb_stochastic"`` — stochastic Chebyshev expansion (Han et al. 2015);
  operator-valued Chebyshev polynomials with geometric convergence via
  Bernstein ellipse.  Same matvec cost as ``chebyshev`` but better accuracy at high |ρ|.
* ``"traces"`` — multinomial trace expansion for unrestricted 3-parameter
  flow models (the only option when the system matrix doesn't factor).

When ``logdet_method`` is ``None`` the method is auto-selected:
``"eigenvalue"`` for n ≤ ``BAYESPECON_LOGDET_EIGEN_MAX_N`` (default 500),
``"chol_aaa"`` for n ≤ ``BAYESPECON_LOGDET_CHEB_MAX_N`` (default 60000)
when ``W`` is symmetric (undirected graph), ``"aaa"`` when ``W`` is
non-symmetric (directed graph), otherwise ``"cheb_stochastic"``
(geometric convergence, same cost as Barry-Pace).
``"cheb_cholesky"`` (Chebyshev interpolation via Cholesky) and
``"slq"`` and ``"chebyshev"`` are available as explicit opt-ins.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Mapping

# ---------------------------------------------------------------------------
# Cache constants
# ---------------------------------------------------------------------------

_LOGDET_FN_CACHE_MAXSIZE = 64
_LOGDET_FN_CACHE: OrderedDict[tuple, Any] = OrderedDict()

# ---------------------------------------------------------------------------
# Enum and type alias
# ---------------------------------------------------------------------------


class LogDetMethod(str, Enum):
    """Canonical log-determinant computation methods."""

    EIGENVALUE = "eigenvalue"
    SLQ = "slq"
    CHEBYSHEV = "chebyshev"
    CHEB_STOCHASTIC = "cheb_stochastic"
    CHEB_CHOLESKY = "cheb_cholesky"
    LU_CHEB = "lu_cheb"
    AAA = "aaa"
    CHOL_AAA = "chol_aaa"
    TRACES = "traces"
    CHOLMOD = "cholmod"  # JAX-native sparse CHOLMOD logdet (requires sparsax)


VALID_LOGDET_METHODS: frozenset[str] = frozenset(m.value for m in LogDetMethod)

LogDetMethodName = Literal[
    "eigenvalue",
    "slq",
    "chebyshev",
    "cheb_stochastic",
    "cheb_cholesky",
    "lu_cheb",
    "aaa",
    "chol_aaa",
    "traces",
    "cholmod",
]


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogdetBounds:
    """Resolved logdet method and rho interval."""

    method: str
    rho_min: float
    rho_max: float
    source: str


# ---------------------------------------------------------------------------
# Resolution functions
# ---------------------------------------------------------------------------


def resolve_logdet_method(
    method: str | None,
    *,
    n: int,
    W=None,
) -> str:
    """Validate ``method`` and auto-select when ``None``.

    Parameters
    ----------
    method
        One of the valid method names, or ``None`` for auto-selection.
    n
        Spatial dimension; used for auto-selection.
    W
        Optional spatial weights matrix.  When ``method`` is ``None``
        and ``n`` is in the medium range, the auto-selector checks
        whether ``W`` is symmetric (undirected graph) to choose between
        ``"cheb_cholesky"`` (symmetric) and ``"aaa"`` (non-symmetric).
        If ``W`` is not provided, defaults to ``"cheb_cholesky"``.

    Returns
    -------
    str
        Canonical method name.
    """
    if method is None:
        return _auto_logdet_method(int(n), W=W)
    if method not in VALID_LOGDET_METHODS:
        valid = ", ".join(sorted(VALID_LOGDET_METHODS))
        raise ValueError(f"Unknown logdet method: {method!r}. Valid options: {valid}.")
    return method


def _is_symmetric_W(W) -> bool:
    """Check whether ``W`` describes an undirected graph → the Cholesky logdet.

    The discriminator is **D-symmetrizability**, not literal matrix symmetry:
    a *row-standardised* undirected graph ``W = D⁻¹A`` (``A`` symmetric) is not
    literally symmetric, yet ``cheb_cholesky`` handles it via D-symmetrisation.
    Only genuinely directed graphs (asymmetric adjacency, e.g. KNN / travel time
    / migration) fall through to the LU-based ``aaa`` path.

    Uses ``libpysal.graph.Graph.asymmetry(intrinsic=False)`` (a topology check)
    when ``W`` is a Graph.  For a sparse/dense matrix — which is what the models
    actually pass (``self._W_sparse``) — a literally symmetric matrix returns
    ``True`` immediately; otherwise D-symmetrizability is tested with the same
    ``_d_symmetrize`` routine ``cheb_cholesky`` relies on, so this predicate
    agrees exactly with whether the Cholesky path is applicable.
    """
    import numpy as np
    import scipy.sparse as sp

    if W is None:
        return True  # default: assume symmetric

    # libpysal Graph: use built-in topology asymmetry check (intrinsic=False
    # ignores weight values, so row-standardisation does not read as directed).
    if hasattr(W, "asymmetry"):
        try:
            asym = W.asymmetry(intrinsic=False)
            return asym.empty
        except Exception:
            pass

    if sp.issparse(W):
        # Sparse difference stays sparse — never densify (n=20k dense is ~3.2GB).
        Wc = W.tocsr()
        diff = (Wc - Wc.T).tocoo()
        if diff.nnz == 0 or bool(np.all(np.abs(diff.data) <= 1e-10)):
            return True
        # Not literally symmetric: may still be a D-symmetrizable (row-
        # standardised undirected) graph, which cheb_cholesky handles.  Test
        # with the actual symmetrisation so routing == applicability.
        try:
            from ._chol_cheb import _d_symmetrize

            _d_symmetrize(Wc)
            return True
        except Exception:
            return False
    else:
        W_arr = np.asarray(W)
        if W_arr.ndim != 2:
            return True  # 1-D eigenvalue array — not applicable
        if np.allclose(W_arr, W_arr.T, atol=1e-10):
            return True
        try:
            from ._chol_cheb import _d_symmetrize

            _d_symmetrize(sp.csr_matrix(W_arr))
            return True
        except Exception:
            return False


def _auto_logdet_method(n: int, W=None) -> str:
    """Auto-select based on matrix dimension n and W symmetry.

    - ``eigenvalue`` for n ≤ eigen_cutoff (default 500): exact O(n³) eigendecomposition.
    - ``cheb_cholesky`` for n ≤ cheb_cutoff (default 60000) when W is symmetric:
      exact logdet via sparse Cholesky at Chebyshev nodes with symbolic reuse.
      Measured setup (2D rook, ρ ∈ [0.1, 0.8], 2026-07): ~96ms at n=10k, ~583ms
      at n=40k, ~1.18s at n=60k.  Accuracy: 4.6e-7 (n=10k) to 2.6e-6 (n=60k).
      Eval: ~1.7μs/ρ via Clenshaw.
    - ``aaa`` for n ≤ cheb_cutoff when W is non-symmetric (directed graph):
      exact logdet via sparse LU (KLU with symbolic reuse) at adaptively-selected
      AAA support points.  Rational approximation converges exponentially near
      singularities.  Uses an adaptive coarse grid of 8–30 LU factorisations,
      sized from the interval's Bernstein-ellipse rate, selecting ~7 support
      points.  Measured setup ~157ms at n=10k; eval ~5μs/ρ; error 1e-8 to 5e-8.
    - ``cheb_stochastic`` for n > cheb_cutoff: stochastic Chebyshev expansion.
      Lower setup cost (~62ms at n=10k, ~328ms at n=60k) but carries stochastic
      error 0.7 to 3.5 with 200 probes.  Eval: ~57μs/ρ.  Use when factorisation
      fill-in makes exact setup too expensive.

    The ``cheb_cutoff`` default of 60,000 is where the benchmark ends, not where
    the exact path stops paying.  It was raised from 20,000 after vectorising the
    symmetrizing-diagonal recovery roughly halved Cholesky setup: at n = 60,000
    the exact path now costs ~850ms more than the stochastic one for six orders
    of magnitude less error, which is negligible against any chain that runs for
    seconds.  Raise it further via ``BAYESPECON_LOGDET_CHEB_MAX_N`` if Cholesky
    fill-in on your graph stays affordable past that.
    """
    eigen_cutoff_raw = os.getenv("BAYESPECON_LOGDET_EIGEN_MAX_N", "500")
    cheb_cutoff_raw = os.getenv("BAYESPECON_LOGDET_CHEB_MAX_N", "60000")
    try:
        eigen_cutoff = max(1, int(eigen_cutoff_raw))
    except ValueError:
        eigen_cutoff = 500
    try:
        cheb_cutoff = max(eigen_cutoff + 1, int(cheb_cutoff_raw))
    except ValueError:
        cheb_cutoff = 60000
    if n <= eigen_cutoff:
        return "eigenvalue"
    if n <= cheb_cutoff:
        # Check W symmetry: chol_aaa for symmetric (undirected graph),
        # aaa for non-symmetric (directed graph: KNN, travel time, migration).
        # chol_aaa combines CHOLMOD's cheaper factorization with AAA's
        # root-exponential convergence — the best of both.
        if _is_symmetric_W(W):
            return "chol_aaa"
        else:
            return "aaa"
    # Stochastic Chebyshev (Han et al. 2015): geometric convergence via
    # Bernstein ellipse, avoids O(n³) eigendecomposition.
    return "cheb_stochastic"


def resolve_logdet_bounds(
    method: str | None,
    *,
    n: int,
    priors: Mapping[str, Any] | None = None,
    rho_min: float | None = None,
    rho_max: float | None = None,
    W=None,
) -> LogdetBounds:
    """Resolve rho bounds from explicit overrides, priors, or defaults.

    For row-standardised W the stability interval is approximately (-1, 1).

    ``W`` (when supplied) participates in auto-selection so that the
    method recorded here agrees with every other resolution site —
    without it a directed graph would be auto-routed to the
    symmetric-only ``cheb_cholesky``.
    """
    resolved_method = resolve_logdet_method(method, n=int(n), W=W)
    source = "default"

    if rho_min is not None or rho_max is not None:
        if rho_min is None or rho_max is None:
            raise ValueError("Both rho_min and rho_max must be provided together.")
        lo = float(rho_min)
        hi = float(rho_max)
        source = "override"
    else:
        p = priors or {}
        lo_prior = None
        hi_prior = None
        for lk, hk in (("rho_lower", "rho_upper"), ("lam_lower", "lam_upper")):
            if lk in p and hk in p:
                lo_prior = float(p[lk])
                hi_prior = float(p[hk])
                break

        if lo_prior is not None and hi_prior is not None:
            lo = lo_prior
            hi = hi_prior
            source = "prior"
        else:
            lo = -1.0
            hi = 1.0

    if hi <= lo:
        raise ValueError(f"Invalid rho interval: rho_min={lo}, rho_max={hi}.")

    return LogdetBounds(
        method=resolved_method,
        rho_min=float(lo),
        rho_max=float(hi),
        source=source,
    )
