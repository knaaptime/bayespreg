"""Stochastic Lanczos Quadrature (SLQ) for log|I - ρW|.

For row-standardised W from an **undirected graph** (rook, queen, distance-band),
W = D⁻¹A where A is symmetric and D = diag(degrees).  W is diagonally similar
to the symmetric matrix W_sym = D^{1/2} W D^{-1/2}, which has the **same
eigenvalues** but real — enabling valid Gauss quadrature via Lanczos.

For directed W (non-symmetric sparsity pattern), falls back to Arnoldi with
complex Ritz values.

Algorithm (D-symmetrised Lanczos)
---------------------------------
1. Recover D from W's sparsity pattern (O(nnz) BFS).
2. Form W_sym = D^{1/2} W D^{-1/2} as a LinearOperator (two O(n) scalings
   + one O(nnz) sparse matvec — never materialised).
3. For each probe z: run k steps of Lanczos on W_sym from the unit start
   q₁ = z/‖z‖, build tridiagonal T_k, eigendecompose → (θ_i, v_i).  Canonical
   SLQ (Ubaru–Chen–Saad) weights: w_i = n · v_{1,i}².
4. Evaluate: log|I - ρW| ≈ (n/n_probes) Σ_j Σ_i v_{1,j,i}² · log(1 - ρθ_{j,i})

Gauss quadrature from k Lanczos steps is exact for polynomials of degree ≤ 2k-1,
giving 3× more spectral information per Krylov step than the Barry-Pace Taylor
series (degree k from k trace moments).

The ``n``-scaling (rather than the sample ``‖z‖²``) removes the χ² radial
fluctuation and makes a constant integrand exact per probe.

**Exact low-order moments** (``n_exact``, default 4).  The Gauss rule implies
its own estimates of the power traces, ``m̂_j = Σᵢ wᵢ θᵢʲ``, and
``log(1-ρx) = -Σⱼ ρʲxʲ/j`` means those estimates *are* the low-order terms of
the SLQ answer.  Replacing them with the exact ``tr(Wʲ)`` removes their sampling
error at no extra matrix-vector products — the same free axis ``cheb_stochastic``
exploits.  Measured on rook/knn at n = 2,500, K = 50: RMSE falls ~700× at
ρ = 0.5, ~12× at ρ = 0.9 and ~4× at ρ = 0.99.

An earlier version of this docstring said SLQ "carries the full
``‖log(I-ρW_sym)‖_F`` Hutchinson variance" and was therefore inherently less
accurate than ``cheb_stochastic`` on flat spatial spectra.  That was a statement
about the *uncorrected* estimator: with both given the same control-variate
depth and the same matvec budget the two are indistinguishable on this problem
class, trading places across budgets with overlapping seed spreads.  SLQ remains
opt-in rather than the auto-selected default, but on cost grounds rather than
accuracy ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ._chebyshev import chebyshev_coeffs_dct1


@dataclass(frozen=True)
class SLQPrecompute:
    """Precomputed SLQ quadrature rules for log|I - ρW|.

    Attributes
    ----------
    nodes : np.ndarray, shape (n_probes, k)
        Gauss quadrature nodes (real for D-symmetrised, complex for Arnoldi).
    weights : np.ndarray, shape (n_probes, k)
        Quadrature weights, ``n``-scaled (``Σᵢ weights = n`` per probe): real
        ``n · v₁ᵢ²`` for Lanczos, complex ``n · γᵢ`` (bilinear) for Arnoldi.
    n : int
        Matrix dimension.
    method : str
        "lanczos" (D-symmetrised) or "arnoldi" (non-symmetric fallback).
    cv_coeffs : np.ndarray or None
        Control-variate corrections ``m̂_j - tr(Wʲ)`` for ``j = 1 .. n_exact``,
        or ``None`` when the correction is disabled.  See
        :func:`slq_logdet_precompute` for what these are and why they are free.
    """

    nodes: np.ndarray
    weights: np.ndarray
    n: int
    method: str = "lanczos"
    cv_coeffs: np.ndarray | None = None

    @property
    def n_exact(self) -> int:
        """Depth of the exact-moment correction (0 when disabled)."""
        return 0 if self.cv_coeffs is None else len(self.cv_coeffs)

    def _cv_correction(self, rho):
        """``Σ_j (ρʲ/j)(m̂_j − m_j)`` — the control-variate term at ``rho``.

        Broadcasts over an array of ``rho``.  Zero when the correction is off.
        """
        if self.cv_coeffs is None:
            return 0.0
        rho = np.asarray(rho, dtype=np.float64)
        j = np.arange(1, len(self.cv_coeffs) + 1, dtype=np.float64)
        powers = rho[..., None] ** j / j
        return np.sum(powers * self.cv_coeffs, axis=-1)

    @property
    def n_probes(self) -> int:
        return self.nodes.shape[0]

    @property
    def lanczos_deg(self) -> int:
        return self.nodes.shape[1]


# ---------------------------------------------------------------------------
# D-recovery: find diagonal D such that D^{1/2} W D^{-1/2} is symmetric
# ---------------------------------------------------------------------------


def _recover_symmetrizing_diagonal(W: sp.csr_matrix) -> np.ndarray | None:
    """Recover D such that D^{1/2} W D^{-1/2} is symmetric.

    For W = D⁻¹A (row-standardised, A symmetric), D[i]/D[j] = W[j,i]/W[i,j]
    for each edge (i,j), so ``log D`` is a potential on the graph and is
    recovered by accumulating edge log-ratios along a spanning forest.

    The traversal is a BFS spanning forest from ``scipy.sparse.csgraph``,
    seeded at the lowest-index node of each connected component, and the
    accumulation is done by pointer doubling — ``O(log depth)`` vectorised
    passes rather than a Python loop over edges.

    Accumulating in log space, then centring each component before
    exponentiating, also extends the range of graphs that can be symmetrised at
    all: the multiplicative propagation it replaces overflowed once the edge
    ratios compounded past ``~1e308`` along a path, whereas the same graph is
    representable here because ``D`` is free up to a per-component scalar.  The
    sign of each ratio is carried separately as a parity bit, so a
    sign-inconsistent ``W`` still yields a negative ``D`` and is rejected by the
    caller rather than silently losing its sign to ``log|·|``.

    Edges whose value is below ``1e-300`` in either direction are excluded from
    the traversal; a node they leave isolated becomes its own component and so
    comes back as ``D = 1``.

    Returns
    -------
    np.ndarray or None
        D (up to scalar multiple), or None if W has asymmetric sparsity
        (directed graph — D-symmetrisation not applicable).
    """
    from scipy.sparse.csgraph import breadth_first_order, connected_components

    n = W.shape[0]

    # Check symmetric sparsity pattern without densifying: the boolean
    # patterns differ iff their sparse XOR has any stored entries.
    pattern = (W != 0).tocsr()
    if (pattern != pattern.T.tocsr()).nnz > 0:
        return None

    if n == 0:
        return np.ones(0, dtype=np.float64)

    # Drop stored zeros so the surviving pattern is exactly the one the check
    # above proved symmetric.  W and Wᵀ then share an index structure, and
    # entry k of ``fwd`` is W[i,j] while entry k of ``rev`` is W[j,i].
    Wc = sp.csr_matrix(W, dtype=np.float64)
    Wc.sum_duplicates()
    Wc.eliminate_zeros()
    Wc.sort_indices()
    A = Wc
    B = Wc.T.tocsr()
    B.sort_indices()

    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(A.indptr))
    cols = A.indices.astype(np.int64)
    fwd = A.data  # W[i, j]
    rev = B.data  # W[j, i]

    # Traversal graph: off-diagonal edges carrying usable values both ways.
    # The predicate is symmetric in (i, j), so the graph is undirected.
    usable = (rows != cols) & (np.abs(fwd) >= 1e-300) & (np.abs(rev) >= 1e-300)
    graph = sp.csr_matrix(
        (np.ones(int(usable.sum())), (rows[usable], cols[usable])),
        shape=(n, n),
    )

    # Per-edge log-ratio: D[j] = D[i] · W[i,j] / W[j,i].
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(usable, fwd / np.where(usable, rev, 1.0), 1.0)
    edge_logabs = np.log(np.abs(ratio))
    edge_neg = (ratio < 0).astype(np.int8)

    # Flat row-major keys are already sorted (CSR with sorted indices), so a
    # single searchsorted resolves every (parent, child) lookup at once.
    keys = rows * n + cols

    n_comp, labels = connected_components(graph, directed=False)

    # BFS spanning forest, seeded at the lowest-index node of each component
    # (matching the seeding order of the loop this replaces).
    order = np.argsort(labels, kind="stable")
    seeds = order[np.searchsorted(labels[order], np.arange(n_comp))]

    parent = np.arange(n, dtype=np.int64)  # roots point at themselves
    for seed in seeds:
        nodes, pred = breadth_first_order(
            graph, int(seed), directed=False, return_predecessors=True
        )
        child = nodes[nodes != seed]
        parent[child] = pred[child]

    # Edge into each node from its BFS parent; roots contribute nothing.
    is_root = parent == np.arange(n, dtype=np.int64)
    logabs = np.zeros(n, dtype=np.float64)
    neg = np.zeros(n, dtype=np.int64)
    if keys.size:
        pos = np.searchsorted(keys, parent * n + np.arange(n, dtype=np.int64))
        pos = np.clip(pos, 0, keys.size - 1)
        logabs = np.where(is_root, 0.0, edge_logabs[pos])
        neg = np.where(is_root, 0, edge_neg[pos]).astype(np.int64)

    # Pointer doubling: accumulate the path sum from each node to its root in
    # O(log depth) passes.  Roots are self-parents, so they are fixed points.
    anc = parent
    while True:
        nxt = anc[anc]
        if np.array_equal(nxt, anc):
            break
        logabs = logabs + logabs[anc]
        neg = neg + neg[anc]
        anc = nxt

    # ``D`` is defined only up to a scalar multiple *per connected component*
    # (every edge of W_sym rescales by the same factor within a component), so
    # centring each component's log before exponentiating is free and keeps the
    # representable range centred on 1.  Without it, a graph whose edge ratios
    # compound in one direction — a long chain, a steep density gradient —
    # overflows to ``inf`` at one end and underflows to ``0`` at the other, and
    # both are rejected downstream as a failed symmetrisation.
    #
    # Via bincount rather than a mask per component: contiguity weights routinely
    # have many islands, and ``labels == c`` in a loop is O(n · n_comp).
    logabs -= (np.bincount(labels, weights=logabs) / np.bincount(labels))[labels]

    # Every node is in some component and every component is traversed from its
    # seed, so an isolated node is its own component: centring leaves its log at
    # zero and it comes back as D = 1 without a special case.
    return np.where(neg % 2 == 0, 1.0, -1.0) * np.exp(logabs)


# ---------------------------------------------------------------------------
# Batched Lanczos (all probes simultaneously — one block matvec per step)
# ---------------------------------------------------------------------------


def _batched_lanczos(
    matvec_fn,  # callable: (n, n_probes) -> (n, n_probes)
    n: int,
    k: int,
    Z: np.ndarray,
    n_probes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run k steps of **batched** Lanczos on W_sym for all probes at once.

    Fully vectorized: stores Lanczos vectors as (n, k, n_probes) and uses
    einsum for batched reorthogonalization.  One batched sparse matvec per
    Lanczos step; all probe operations are vectorized.

    Parameters
    ----------
    matvec_fn : callable
        Function that computes W_sym @ Q for a (n, n_probes) block.
    n : int
        Matrix dimension.
    k : int
        Lanczos steps per probe.
    Z : np.ndarray, shape (n, n_probes)
        Probe vectors (columns).
    n_probes : int

    Returns
    -------
    (nodes, weights, z_norms_sq) : tuple
    """
    # Normalize each probe column (unit starts; canonical SLQ scales by n).
    z_norms = np.linalg.norm(Z, axis=0)  # (n_probes,)
    Q0 = Z / np.where(z_norms < 1e-15, 1.0, z_norms)  # (n, n_probes)

    # 3D storage: Q_all[:, step, probe] — (n, k, n_probes)
    Q_all = np.zeros((n, k, n_probes), dtype=np.float64)
    Q_all[:, 0, :] = Q0

    alphas = np.zeros((n_probes, k), dtype=np.float64)
    betas = np.zeros((n_probes, k - 1), dtype=np.float64)
    active = np.ones(n_probes, dtype=bool)  # which probes haven't broken down
    deg = np.full(n_probes, k, dtype=int)  # effective Lanczos degree per probe

    # First matvec (batched)
    R = matvec_fn(Q0)  # (n, n_probes) — ONE batched matvec

    # alpha_0 = Q0' R (per-probe dot products)
    alphas[:, 0] = np.sum(Q0 * R, axis=0)  # (n_probes,)
    R = R - alphas[:, 0] * Q0  # broadcast: (n, n_probes)

    # Lanczos steps 1..k-1
    for i in range(1, k):
        # Compute beta and new q (vectorized)
        beta = np.linalg.norm(R, axis=0)  # (n_probes,)
        betas[:, i - 1] = beta

        # A probe that breaks down here (β≈0) has an i-dimensional Krylov
        # subspace; record its effective degree before deactivating it.
        newly_dead = active & (beta < 1e-15)
        deg[newly_dead] = i
        active &= beta >= 1e-15
        if not active.any():
            break

        # Normalize R → q_new (vectorized, with safe division)
        safe_beta = np.where(beta < 1e-15, 1.0, beta)
        q_new = R / safe_beta  # (n, n_probes) — zeros for inactive
        Q_all[:, i, :] = q_new

        # Batched matvec
        R_new = matvec_fn(q_new)  # (n, n_probes) — ONE batched matvec

        # alpha_i = q_new' R_new (vectorized)
        alphas[:, i] = np.sum(q_new * R_new, axis=0)

        # Three-term recurrence: R = R_new - alpha * q_new - beta_prev * Q_prev
        R = R_new - alphas[:, i] * q_new - betas[:, i - 1] * Q_all[:, i - 1, :]

        # Full reorthogonalization (vectorized via einsum)
        # For each probe j, project R[:, j] against Q_all[:, :i+1, j]
        # Q_slice: (n, i+1, n_probes), R: (n, n_probes)
        # coeffs = einsum('nsj,nj->sj', Q_slice, R)  → (i+1, n_probes)
        Q_slice = Q_all[:, : i + 1, :]  # (n, i+1, n_probes)
        proj_coeffs = np.einsum("nsj,nj->sj", Q_slice, R)  # (i+1, n_probes)
        # R -= Q_slice @ proj_coeffs (per-probe)
        R = R - np.einsum("nsj,sj->nj", Q_slice, proj_coeffs)

    # Eigendecompose each probe's tridiagonal (vectorized loop — k×k is tiny).
    # Canonical SLQ weight is n·v₁ᵢ² (unit start scaled by n), not ‖z‖²·v₁ᵢ²:
    # the χ² fluctuation of ‖z‖² is removed and Σᵢ n·v₁ᵢ² = n is exact per
    # probe (constant integrand recovered exactly).  This matches the
    # normalization already used by ``slq_to_chebyshev_coeffs``.
    nodes = np.zeros((n_probes, k), dtype=np.float64)
    weights = np.zeros((n_probes, k), dtype=np.float64)
    z_norms_sq = z_norms**2

    for j in range(n_probes):
        m = deg[j]
        # Build tridiagonal
        T = np.diag(alphas[j, :m])
        if m > 1:
            T += np.diag(betas[j, : m - 1], 1) + np.diag(betas[j, : m - 1], -1)
        theta, eigvecs = np.linalg.eigh(T)
        nodes[j, :m] = theta
        weights[j, :m] = n * eigvecs[0, :] ** 2

    return nodes, weights, z_norms_sq


# ---------------------------------------------------------------------------
# Arnoldi iteration (non-symmetric fallback, complex Ritz values)
# ---------------------------------------------------------------------------


def _arnoldi_iteration(
    W_op: spla.LinearOperator | sp.csr_matrix,
    n: int,
    k: int,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run k steps of Arnoldi on a non-symmetric operator starting from z.

    Returns ``(theta, gamma)`` where ``theta`` are the complex Ritz values of
    the Hessenberg ``H`` and ``gamma`` are the **bilinear** quadrature weights
    for the unit start ``q₁ = e₁``::

        e₁ᵀ f(H) e₁ = Σᵢ γᵢ f(θᵢ),   γᵢ = (e₁ᵀ V)ᵢ (V⁻¹ e₁)ᵢ

    where ``H = V diag(θ) V⁻¹``.  Because ``H`` is non-normal its eigenvectors
    are not orthogonal, so the symmetric-case rule ``|V[0, i]|²`` is wrong —
    the left/right (biorthogonal) product ``V[0, i]·(V⁻¹e₁)ᵢ`` is required and
    is generally complex.
    """
    z_norm = np.linalg.norm(z)
    if z_norm == 0:
        return np.empty(0, dtype=np.complex128), np.empty(0, dtype=np.complex128)

    q = z / z_norm

    Q = np.zeros((n, k), dtype=np.float64)
    H = np.zeros((k, k), dtype=np.float64)
    Q[:, 0] = q

    m = k
    for i in range(k - 1):
        w = W_op @ Q[:, i]
        for j in range(i + 1):
            H[j, i] = float(Q[:, j] @ w)
            w = w - H[j, i] * Q[:, j]
        H[i + 1, i] = np.linalg.norm(w)
        if H[i + 1, i] < 1e-15:
            H = H[: i + 1, : i + 1]
            m = i + 1
            break
        Q[:, i + 1] = w / H[i + 1, i]

    theta, V = np.linalg.eig(H)
    e1 = np.zeros(m, dtype=np.complex128)
    e1[0] = 1.0
    # γ = V[0, :] ∘ (V⁻¹ e₁): the biorthogonal bilinear-form weights.
    gamma = V[0, :] * np.linalg.solve(V, e1)

    return theta, gamma


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


#: Default depth of the exact-moment control variate, matching
#: ``cheb_stochastic``'s ``DEFAULT_N_EXACT`` so the two estimators are
#: comparable out of the box.  Measured gains on rook/knn at n = 2,500,
#: K = 50: RMSE at ρ = 0.5 falls ~700×, at ρ = 0.9 ~12×, at ρ = 0.99 ~4×.
DEFAULT_SLQ_N_EXACT = 4


#: Default Lanczos depth.  Lowered from 30 after measuring that the Gauss rule
#: converges well before it: at a fixed matvec budget, spending the saved steps
#: on probes instead is 1.3-1.6x more accurate on rook and knn at
#: n ∈ {2,500, 10,000}, because what remains after convergence is Monte Carlo
#: variance rather than quadrature error.  15 rather than 12 because the
#: convergence point is problem-dependent: 12 wins at n = 2,500 but buys nothing
#: at n = 10,000, and below 10 the rule is clearly unconverged.
DEFAULT_LANCZOS_DEG = 15


def slq_logdet_precompute(
    W,
    n_probes: int = 50,
    lanczos_deg: int = DEFAULT_LANCZOS_DEG,
    rng: np.random.Generator | None = None,
    n_exact: int | None = None,
) -> SLQPrecompute:
    """Precompute SLQ quadrature rules for log|I - ρW|.

    For undirected-graph W (symmetric sparsity), uses D-symmetrised Lanczos
    with real eigenvalues and valid Gauss quadrature.  For directed W,
    falls back to Arnoldi with complex Ritz values.

    ``n_exact`` sets the depth of the exact-moment control variate described in
    :func:`_slq_cv_coeffs`, which costs no matrix-vector products.  Pass ``0``
    to disable it and recover the uncorrected estimator.

    Parameters
    ----------
    W : array-like or scipy.sparse matrix
        Spatial weights matrix (dense or sparse).
    n_probes : int, default 50
    lanczos_deg : int, default 30
    rng : np.random.Generator, optional
        Probe-vector RNG.  Defaults to a *seeded* generator so the
        precomputed quadrature (and thus the logdet approximation) is
        reproducible run-to-run; pass your own Generator to randomize.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # Reuse ``cheb_stochastic``'s mean-degree guard: depths above 2 need
    # sparse-sparse products whose fill-in grows with degree, and on dense
    # graphs those can dominate a matvec-only precompute.
    if n_exact == 0:
        depth = 0
    else:
        from ._cheb_stochastic import _resolve_exact_depth

        _W_guard = sp.csr_matrix(W) if not sp.issparse(W) else sp.csr_matrix(W)
        depth = _resolve_exact_depth(
            _W_guard,
            DEFAULT_SLQ_N_EXACT if n_exact is None else int(n_exact),
            16.0,
        )

    if sp.issparse(W) or hasattr(W, "format"):
        W_sp = sp.csr_matrix(W)
        n = W_sp.shape[0]
    else:
        W_arr = np.asarray(W, dtype=np.float64)
        n = W_arr.shape[0]
        W_sp = sp.csr_matrix(W_arr)

    # Try D-symmetrisation
    D = _recover_symmetrizing_diagonal(W_sp)

    if D is not None:
        # D-symmetrised batched Lanczos (real eigenvalues, valid Gauss quadrature)
        # Use the sparse matrix directly for batched matvec (LinearOperator doesn't
        # support matmat efficiently).
        sqrt_D = np.sqrt(D)
        inv_sqrt_D = 1.0 / sqrt_D
        W_sp_for_lanczos = W_sp  # original sparse W

        def batch_matvec(Q_block):
            """W_sym @ Q = sqrt_D * (W @ (inv_sqrt_D * Q)) — handles 2D."""
            return sqrt_D[:, None] * (
                W_sp_for_lanczos @ (inv_sqrt_D[:, None] * Q_block)
            )

        method = "lanczos"
        Z = rng.standard_normal((n, n_probes))
        all_nodes, all_weights, _ = _batched_lanczos(
            batch_matvec, n, lanczos_deg, Z, n_probes
        )
        return SLQPrecompute(
            nodes=all_nodes,
            weights=all_weights,
            n=n,
            method=method,
            cv_coeffs=_slq_cv_coeffs(W_sp, all_nodes, all_weights, depth),
        )
    else:
        # Arnoldi fallback (complex Ritz values) — per-probe loop.  Unit
        # starts scaled by n (matching the Lanczos convention); the bilinear
        # weights γ are complex, so ``weights`` is complex here.
        W_op = W_sp
        method = "arnoldi"
        all_nodes = np.zeros((n_probes, lanczos_deg), dtype=np.complex128)
        all_weights = np.zeros((n_probes, lanczos_deg), dtype=np.complex128)

        for j in range(n_probes):
            z = rng.standard_normal(n)
            theta, gamma = _arnoldi_iteration(W_op, n, lanczos_deg, z)
            m = len(theta)
            all_nodes[j, :m] = theta
            all_weights[j, :m] = n * gamma

        return SLQPrecompute(
            nodes=all_nodes,
            weights=all_weights,
            n=n,
            method=method,
            cv_coeffs=_slq_cv_coeffs(W_sp, all_nodes, all_weights, depth),
        )


def _slq_cv_coeffs(
    W_sp: sp.csr_matrix,
    nodes: np.ndarray,
    weights: np.ndarray,
    n_exact: int,
) -> np.ndarray | None:
    """Control-variate corrections ``m̂_j - tr(Wʲ)`` for ``j = 1 .. n_exact``.

    The quadrature rule this precompute already holds implies its own estimates
    of the power traces,

        m̂_j = (1/K) Σ_probes Σ_i w_i θ_iʲ,

    and ``log(1 - ρx) = -Σ_j ρʲ xʲ / j`` means the low-order terms of the SLQ
    estimate are exactly ``-Σ_j (ρʲ/j) m̂_j``.  Since ``tr(Wʲ)`` is available
    exactly and probe-free for small ``j`` (:func:`~._cheb_stochastic._power_traces`,
    at most two sparse-sparse products), replacing the estimated moments with the
    exact ones removes their sampling error at **no additional matrix-vector
    products** — the same axis on which ``cheb_stochastic``'s ``n_exact`` is free.

    This is a control variate on the existing Gauss rule, not eigenpair
    deflation: it needs no eigenvectors and costs no extra Krylov work.

    Returns ``None`` when ``n_exact`` is zero or the moments cannot be formed.
    """
    if n_exact <= 0:
        return None
    from ._cheb_stochastic import _power_traces

    exact = _power_traces(W_sp, n_exact)
    est = np.array(
        [
            float(np.real(np.mean(np.sum(weights * nodes**j, axis=1))))
            for j in range(n_exact + 1)
        ]
    )
    return est[1:] - exact[1 : n_exact + 1]


def _slq_log_vals(vals: np.ndarray, method: str) -> np.ndarray:
    """``log(1 - ρθ)`` for the quadrature nodes.

    Lanczos nodes are real, so ``log|1 - ρθ|`` (the real logdet integrand) is
    used directly.  Arnoldi nodes and bilinear weights are complex; the
    complex logarithm is required because ``Re(Σ γᵢ log(1-ρθᵢ))`` keeps the
    cross term ``Im(γ)·Im(log)`` that a magnitude-only log would drop.
    """
    if method == "arnoldi":
        vals = np.where(np.abs(vals) < 1e-300, 1e-300, vals)
        return np.log(vals.astype(np.complex128))
    return np.log(np.maximum(np.abs(vals), 1e-300))


def slq_logdet_eval(pre: SLQPrecompute, rho: float) -> float:
    """Evaluate log|I - ρW| from precomputed SLQ quadrature rules."""
    log_vals = _slq_log_vals(1.0 - rho * pre.nodes, pre.method)
    base = float(np.real(np.sum(pre.weights * log_vals)) / pre.n_probes)
    return base + float(pre._cv_correction(rho))


def slq_logdet_eval_vec(pre: SLQPrecompute, rho_arr: np.ndarray) -> np.ndarray:
    """Vectorized SLQ logdet evaluation over an array of ρ values."""
    rho_arr = np.asarray(rho_arr, dtype=np.float64)
    vals = 1.0 - rho_arr[:, None, None] * pre.nodes[None, :, :]
    log_vals = _slq_log_vals(vals, pre.method)
    base = (
        np.real(np.sum(pre.weights[None, :, :] * log_vals, axis=(1, 2))) / pre.n_probes
    )
    return base + pre._cv_correction(rho_arr)


# ---------------------------------------------------------------------------
# SLQ → Chebyshev coefficient conversion (fast O(m) evaluation per ρ)
# ---------------------------------------------------------------------------


def slq_to_chebyshev_coeffs(
    pre: SLQPrecompute,
    W: sp.csr_matrix | None = None,
    order: int = 20,
    rho_min: float = -1.0,
    rho_max: float = 1.0,
) -> dict:
    """Convert SLQ quadrature rules into Chebyshev polynomial coefficients.

    Uses SLQ's Gauss quadrature to estimate ``tr(W^k)`` for ``k=1..order``,
    then feeds into the same Taylor series + DCT-I pipeline as Barry-Pace.
    Gauss quadrature from *m* Lanczos steps is exact for polynomials of degree
    ``≤ 2m-1``, so ``m=30`` Lanczos steps give exact traces up to ``tr(W^59)``
    — far beyond the 20 traces needed.

    The first two traces (``tr(W)``, ``tr(W²)``) are overridden with exact
    values when ``W`` is provided, matching Barry-Pace's variance reduction.

    Parameters
    ----------
    pre : SLQPrecompute
        Quadrature rules from :func:`slq_logdet_precompute`.
    W : scipy.sparse.csr_matrix, optional
        Spatial weights matrix for exact trace overrides.
    order : int, default 20
        Chebyshev polynomial degree.
    rho_min, rho_max : float
        Interval bounds for the Chebyshev approximation.

    Returns
    -------
    dict
        ``{"coeffs", "rmin", "rmax", "order", "method"}`` — same format as
        :func:`chebyshev`, compatible with :func:`logdet_chebyshev`.
    """
    n = pre.n

    # Recover the per-probe spectral weights (Lanczos: v₁ᵢ²; Arnoldi: the
    # complex bilinear γᵢ) from the stored quadrature weights, which are the
    # n-scaled canonical form (Σᵢ weights = n per probe).  Dividing by that
    # row sum yields the unit-mass weights independent of the n scaling, so
    # ``tr(Wᵏ) ≈ (n / n_probes) Σ_j Σ_i eᵢ θᵢᵏ``.
    weight_sum = np.sum(pre.weights, axis=1)  # = n per probe
    e1_sq = pre.weights / weight_sum[:, None]  # (n_probes, k)
    nodes = pre.nodes  # real (lanczos) or complex (arnoldi)

    # Estimate traces via Gauss quadrature: tr(W^k) = (n/n_probes) Σ_j Σ_i e1² θ^k
    # (real part: Arnoldi's complex quadrature estimates a real trace).
    traces = np.zeros(order, dtype=np.float64)
    for p in range(1, order + 1):
        traces[p - 1] = n * np.real(np.mean(np.sum(e1_sq * nodes**p, axis=1)))

    # Override first two traces with exact values (major variance reduction)
    if W is not None:
        W_csr = sp.csr_matrix(W) if not sp.issparse(W) else W
        traces[0] = float(W_csr.diagonal().sum())
        if order >= 2:
            traces[1] = float(W_csr.multiply(W_csr.T).sum())

    # Taylor series: log|I - ρW| = -Σ_k tr(W^k) ρ^k / k
    # Evaluate at Chebyshev nodes, then DCT-I → Chebyshev coefficients
    k_arr = np.arange(1, order + 1)
    nodes_cos = np.cos((2 * k_arr - 1) * np.pi / (2 * order))
    rho_nodes = 0.5 * (rho_max - rho_min) * nodes_cos + 0.5 * (rho_max + rho_min)

    td = traces / np.arange(1, order + 1, dtype=np.float64)
    logdet_at_nodes = np.zeros(order, dtype=np.float64)
    for idx, r in enumerate(rho_nodes):
        powers = np.power(r, np.arange(1, order + 1, dtype=np.float64))
        logdet_at_nodes[idx] = -powers @ td

    # DCT-I → Chebyshev coefficients
    coeffs = chebyshev_coeffs_dct1(logdet_at_nodes)

    return {
        "coeffs": coeffs,
        "rmin": rho_min,
        "rmax": rho_max,
        "order": order,
        "method": "slq",
    }
