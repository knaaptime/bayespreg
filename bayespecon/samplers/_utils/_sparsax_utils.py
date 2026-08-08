"""Shared helpers for sparsax integration in JAX Gibbs samplers.

Provides the COO sparsity-pattern precomputation and value-assembly
utilities needed to use :mod:`sparsax` (JAX-native sparse CHOLMOD)
inside JIT-compiled Gibbs steps.

The precision matrix

.. math::

    P = I + \\mathrm{diag}(\\omega) - \\rho (W + W^T) + \\rho^2 W^T W

is symmetric positive definite for any valid ``ρ`` and ``ω ≥ 0``.
Its sparsity pattern is **fixed** (independent of ``ρ`` and ``ω``),
so we precompute the COO indices ``(Ai, Aj)`` once on the host and
assemble only the values ``Ax(ρ, ω)`` inside the JIT boundary.

This mirrors the NumPy-CHOLMOD pattern in
:func:`bayespecon.samplers.negbin_reduced._core._make_cholmod_pattern`
but returns int32 COO arrays suitable for ``sparsax``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def precompute_sparsax_pattern(
    W_csc: sp.csc_matrix,
    n: int,
) -> dict:
    """Precompute the fixed COO sparsity pattern for the precision matrix.

    The pattern covers all fill-in positions of
    ``P = I + diag(ω) − ρ(W+Wᵀ) + ρ²WᵀW`` for any valid ``ρ`` and ``ω ≥ 0``.
    Built as ``I + 0.5*(W+Wᵀ) + 0.25*WᵀW`` so every possible nonzero is present.

    Parameters
    ----------
    W_csc : scipy.sparse.csc_matrix
        The **raw** (row-standardized) spatial weights matrix ``W`` in CSC
        format — *not* ``W+Wᵀ``.  This function derives ``W+Wᵀ`` and ``WᵀW``
        from it internally; passing an already-symmetrized matrix would double
        the symmetric part and corrupt ``WᵀW``.
    n : int
        Matrix dimension.

    Returns
    -------
    dict with keys:
        ``Ai`` : np.ndarray, shape (nnz,), dtype int32 — COO row indices.
        ``Aj`` : np.ndarray, shape (nnz,), dtype int32 — COO column indices.
        ``W_sym_vals`` : np.ndarray, shape (nnz,), dtype float64 —
            Values of ``W + Wᵀ`` at the pattern positions (0 where pattern
            has entries but W+Wᵀ does not).
        ``WtW_vals`` : np.ndarray, shape (nnz,), dtype float64 —
            Values of ``WᵀW`` at the pattern positions.
        ``is_diag`` : np.ndarray, shape (nnz,), dtype bool —
            Boolean mask for diagonal entries (``Ai == Aj``).
        ``diag_idx`` : np.ndarray, shape (n,), dtype int32 —
            Indices into the pattern arrays where the diagonal entries live.
            Used to scatter ``1 + ω`` into ``Ax``.
        ``n`` : int — Matrix dimension.
    """
    W_sym = (W_csc + W_csc.T).tocsc()
    WtW = (W_csc.T @ W_csc).tocsc()
    pattern = (sp.eye(n, format="csc") + 0.5 * W_sym + 0.25 * WtW).tocoo()

    Ai = pattern.row.astype(np.int32)
    Aj = pattern.col.astype(np.int32)
    nnz = len(Ai)

    # Extract W_sym and WtW values at the pattern positions.
    # We do this by converting to COO and aligning with the pattern.
    W_sym_coo = W_sym.tocoo()
    WtW_coo = WtW.tocoo()

    # Build a lookup from (row, col) → index in the pattern array.
    # The pattern is symmetric (upper triangle), so we only need to
    # match entries with Ai <= Aj (lower triangle entries are ignored
    # by sparsax).
    pattern_lookup: dict[tuple[int, int], int] = {}
    for k in range(nnz):
        pattern_lookup[(int(Ai[k]), int(Aj[k]))] = k

    W_sym_vals = np.zeros(nnz, dtype=np.float64)
    WtW_vals = np.zeros(nnz, dtype=np.float64)

    for k in range(len(W_sym_coo.row)):
        i, j = int(W_sym_coo.row[k]), int(W_sym_coo.col[k])
        # sparsax reads only upper triangle (Ai <= Aj)
        if i <= j:
            idx = pattern_lookup.get((i, j))
            if idx is not None:
                W_sym_vals[idx] = W_sym_coo.data[k]

    for k in range(len(WtW_coo.row)):
        i, j = int(WtW_coo.row[k]), int(WtW_coo.col[k])
        if i <= j:
            idx = pattern_lookup.get((i, j))
            if idx is not None:
                WtW_vals[idx] = WtW_coo.data[k]

    is_diag = Ai == Aj
    # For each diagonal position i, find its index in the pattern.
    diag_idx = np.full(n, -1, dtype=np.int32)
    for k in range(nnz):
        if is_diag[k]:
            diag_idx[Ai[k]] = k

    return {
        "Ai": Ai,
        "Aj": Aj,
        "W_sym_vals": W_sym_vals,
        "WtW_vals": WtW_vals,
        "is_diag": is_diag,
        "diag_idx": diag_idx,
        "n": n,
    }


def make_sparsax_ops(Ai, Aj, n: int):
    """Return ``(eta_sample, solve_logdet)`` factor-once closures over a fixed pattern.

    Both do **one** numeric factorization per call (matching numpy's
    ``CholmodFactor`` reuse), using sparsax 0.4's factor-once primitives when
    available and falling back to the 0.3 idiom otherwise:

    - ``eta_sample(Ax, mean_term, z) -> N(P⁻¹ mean_term, P⁻¹)`` draw — 0.4:
      :func:`sparsax.sample_gaussian` (one factorization); 0.3: mean solve +
      ``MODE_LT`` + ``MODE_PT`` (three solves ≈ three factorizations under vmap).
    - ``solve_logdet(Ax, b) -> (P⁻¹ b, log|P|)`` — 0.4:
      :func:`sparsax.factor_solve` with ``want_logdet=True`` (one factorization,
      no working-copy); 0.3: :func:`sparsax.update_solve` with a zero update
      column and ``return_logdet=True``.

    ``Ai``/``Aj`` are the fixed COO indices (int32); ``b`` may be ``(n,)`` or
    ``(n, n_rhs)``.
    """
    import jax.numpy as jnp
    import sparsax as _chj

    Ai = jnp.asarray(Ai, dtype=jnp.int32)
    Aj = jnp.asarray(Aj, dtype=jnp.int32)

    if hasattr(_chj, "sample_gaussian"):  # sparsax >= 0.4
        _MODE_A = getattr(_chj, "MODE_A", 0)

        def eta_sample(Ax, mean_term, z):
            eta, _mean = _chj.sample_gaussian(Ai, Aj, Ax, mean_term, z)
            return eta

        def solve_logdet(Ax, b):
            sols, ld = _chj.factor_solve(Ai, Aj, Ax, [(b, _MODE_A)], want_logdet=True)
            return sols[0], ld

    else:  # sparsax 0.3 fallback
        _Czero = jnp.zeros((n, 1), dtype=jnp.float64)
        _MODE_LT, _MODE_PT = _chj.MODE_LT, _chj.MODE_PT

        def eta_sample(Ax, mean_term, z):
            m = _chj.solve(Ai, Aj, Ax, mean_term)
            w = _chj.solve(Ai, Aj, Ax, z, mode=_MODE_LT)
            w = _chj.solve(Ai, Aj, Ax, w, mode=_MODE_PT)
            return m + w

        def solve_logdet(Ax, b):
            x, ld = _chj.update_solve(Ai, Aj, Ax, _Czero, b, return_logdet=True)
            return x, ld

    return eta_sample, solve_logdet


def resolve_pg_jax_backend(backend, *, W_sparse, W_sym, WtW, n, logdet_bounds):
    """Resolve the PG-Gibbs backend method and its JAX precomputes.

    Shared by the SAR-logit / SEM-logit / structural SAR-NB Gibbs fits, which
    previously each carried this ~40-line block verbatim.

    Parameters
    ----------
    backend : {"jax", "numpy"}
        Resolved execution backend.
    W_sparse, W_sym, WtW : scipy.sparse matrices
        Raw row-standardized ``W``, ``W + Wᵀ`` and ``WᵀW``.
    n : int
        Number of observations.
    logdet_bounds : LogdetBounds
        The model's resolved logdet bounds (method, rho_min, rho_max).

    Returns
    -------
    method : str
        One of ``"cholmod"`` (numpy), ``"jax_dense"``, ``"cholmod_jax"`` —
        used for all three of the cache's solve/logdet_P/sample methods.
    jax_parts : dict
        ``W_sym_dense``, ``WtW_dense``, ``logdet_jax``, ``sparsax_pattern``
        (all ``None`` on the numpy path).
    """
    jax_parts = {
        "W_sym_dense": None,
        "WtW_dense": None,
        "logdet_jax": None,
        "sparsax_pattern": None,
    }
    if backend != "jax":
        return "cholmod", jax_parts

    from bayespecon._jax_dispatch import (
        _sparsax_available,
        _sparsax_jax_enabled,
        ensure_x64,
    )

    method = (
        "cholmod_jax"
        if _sparsax_jax_enabled() and _sparsax_available()
        else "jax_dense"
    )

    import jax.numpy as jnp

    ensure_x64()

    # Only the dense-Cholesky fallback needs the dense (W+Wᵀ) and WᵀW; the
    # cholmod_jax path assembles P from the sparse COO pattern and does its
    # matvecs via BCOO, so we never densify W there.
    if method == "jax_dense":
        jax_parts["W_sym_dense"] = jnp.asarray(W_sym.toarray(), dtype=jnp.float64)
        jax_parts["WtW_dense"] = jnp.asarray(WtW.toarray(), dtype=jnp.float64)

    from bayespecon._logdet import make_logdet_jax_fn

    jax_parts["logdet_jax"] = make_logdet_jax_fn(
        W_sparse,
        method=logdet_bounds.method,
        rho_min=logdet_bounds.rho_min,
        rho_max=logdet_bounds.rho_max,
    )

    if method == "cholmod_jax":
        # Pass the raw (row-standardized) W; the helper derives W+Wᵀ and WᵀW
        # internally.  Passing W_sym here would double the symmetric part and
        # corrupt WᵀW.
        jax_parts["sparsax_pattern"] = precompute_sparsax_pattern(W_sparse.tocsc(), n)

    return method, jax_parts


# ---------------------------------------------------------------------------
# NumPy-side cached-pattern sparse solve (host loops, no JIT)
# ---------------------------------------------------------------------------


class CachedSparseSolver:
    r"""Sparse direct solver that reuses one symbolic analysis across calls.

    Many posterior-loop hot paths solve

    .. math::

        A(\theta)\, x = b, \qquad A(\theta) = I - \sum_k \theta_k\, W_k,

    repeatedly for many values of ``θ`` (posterior draws, ρ-grid search,
    posterior-predictive replications) with a **fixed** sparsity pattern —
    only the numeric values rescale.  sparsax's ``lu_solve`` caches the
    fill-reducing symbolic analysis keyed on the ``(Ai, Aj)`` COO indices,
    so calls sharing a pattern pay the symbolic cost once and each later
    call is just a numeric refactor + triangular solves.

    This helper precomputes the merged COO pattern
    ``(Ai, Aj, const_vals, w_vals_list)`` once and assembles
    ``Ax = const_vals + Σ_k θ_k · w_vals_list[k]`` per call, dispatching to
    sparsax when available and falling back to a per-call scipy ``splu``
    when it is not.  It is the host-side analogue of
    :func:`precompute_sparsax_pattern` / :func:`make_sparsax_ops` for the
    JAX Gibbs path.

    Parameters
    ----------
    weight_matrices : list of scipy.sparse matrices
        The :math:`W_k` (any sparse format).  All must share the same
        shape.  The identity :math:`I` is added internally as a constant
        coefficient (its pattern is merged with that of the :math:`W_k`).
    n : int
        Matrix dimension (``weight_matrices[0].shape[0]``).

    Attributes
    ----------
    Ai, Aj : np.ndarray of int32
        Merged COO row/column indices.
    const_vals : np.ndarray of float64
        Values of the identity at the pattern positions.
    w_vals_list : list of np.ndarray of float64
        Values of each :math:`W_k` at the pattern positions.

    Notes
    -----
    sparsax availability is resolved once at construction time via
    :func:`bayespecon._jax_dispatch._sparsax_available`; no JAX import is
    required on the fallback path.  The merged pattern is built so every
    nonzero of any :math:`W_k` plus the diagonal is present.

    Examples
    --------
    Single-ρ SAR system, many posterior draws::

        solver = CachedSparseSolver([W_sparse], n)
        for rho in rho_draws:
            x = solver.solve([rho], rhs)   # one cached symbolic analysis

    Three-ρ flow system::

        solver = CachedSparseSolver([Wd, Wo, Ww], N)
        for rd, ro, rw in zip(rd_draws, ro_draws, rw_draws):
            x = solver.solve([rd, ro, rw], rhs)
    """

    def __init__(self, weight_matrices, n):
        self.n = int(n)
        mats = [sp.csc_matrix(m) for m in weight_matrices]
        shapes = {m.shape[0] for m in mats}
        if len(shapes) != 1 or shapes.pop() != self.n:
            raise ValueError(
                "All weight matrices must be square with shape (n, n) matching n."
            )
        # Merge the I and every W_k patterns into one COO layout so a single
        # (Ai, Aj) tuple drives every solve; duplicate (i, j) entries are
        # summed, matching how Ax = const + Σ θ_k W_k is evaluated.
        I_coo = sp.eye(self.n, format="coo")
        rows = [I_coo.row]
        cols = [I_coo.col]
        data = [np.ones(I_coo.nnz, dtype=np.float64)]
        for Wk in mats:
            Wk_coo = Wk.tocoo()
            rows.append(Wk_coo.row)
            cols.append(Wk_coo.col)
            data.append(Wk_coo.data.astype(np.float64, copy=False))
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.concatenate(data)
        merged = sp.coo_matrix((data, (rows, cols)), shape=(self.n, self.n))
        merged.eliminate_zeros()
        merged.sum_duplicates()
        # Split merged values back into const (I) and per-W contributions.
        self.Ai = merged.row.astype(np.int32)
        self.Aj = merged.col.astype(np.int32)
        # const_vals: identity at each pattern position
        I_at = sp.coo_matrix(
            (np.ones(I_coo.nnz, dtype=np.float64), (I_coo.row, I_coo.col)),
            shape=(self.n, self.n),
        ).tocsr()
        self.const_vals = np.asarray(
            I_at[self.Ai, self.Aj].A1
            if hasattr(I_at, "A1")
            else np.asarray(I_at.todense()[self.Ai, self.Aj]).ravel(),
            dtype=np.float64,
        )
        # Robust alignment: build per-W value vectors at the pattern positions
        self.w_vals_list = []
        for Wk in mats:
            Wk_csr = Wk.tocsr()
            vals = np.asarray(
                Wk_csr[self.Ai, self.Aj].A1
                if hasattr(Wk_csr, "A1")
                else np.asarray(Wk_csr.todense()[self.Ai, self.Aj]).ravel(),
                dtype=np.float64,
            )
            self.w_vals_list.append(vals)

        from bayespecon._jax_dispatch import _sparsax_available

        self._use_sparsax = _sparsax_available()
        self._Ai_jax = None
        self._Aj_jax = None
        self._const_jax = None
        self._w_jax_list = None
        if self._use_sparsax:
            import jax.numpy as jnp

            from bayespecon._jax_dispatch import ensure_x64

            ensure_x64()
            self._Ai_jax = jnp.asarray(self.Ai, dtype=jnp.int32)
            self._Aj_jax = jnp.asarray(self.Aj, dtype=jnp.int32)
            self._const_jax = jnp.asarray(self.const_vals, dtype=jnp.float64)
            self._w_jax_list = [
                jnp.asarray(v, dtype=jnp.float64) for v in self.w_vals_list
            ]

    def _assemble_Ax(self, coeffs):
        ax = self._const_jax.copy()
        for c, wv in zip(coeffs, self._w_jax_list):
            ax = ax + float(c) * wv
        return ax

    def solve(self, coeffs, rhs):
        """Solve :math:`A(\\theta) x = b` for vector RHS.

        Parameters
        ----------
        coeffs : sequence of float
            Coefficients :math:`\\theta_k` for each weight matrix, in the
            order passed to the constructor.  ``A = I + Σ θ_k W_k``; for the
            usual ``I - ρ W`` form pass ``[-ρ]`` (or equivalently ``solve``
            assembles ``const + Σ θ_k W_k``, so ``[-ρ]`` gives ``I - ρ W``).
        rhs : ndarray, shape (n,) or (n, k)
            Right-hand side(s).

        Returns
        -------
        x : ndarray, shape matching ``rhs``
        """
        rhs_np = np.asarray(rhs, dtype=np.float64)
        single = rhs_np.ndim == 1
        if single:
            rhs_np = rhs_np[:, None]
        n_rhs = rhs_np.shape[1]
        out = np.empty_like(rhs_np)
        if self._use_sparsax:
            import jax.numpy as jnp
            import sparsax

            Ax = self._assemble_Ax(coeffs)
            for c in range(n_rhs):
                out[:, c] = np.asarray(
                    sparsax.lu_solve(
                        self._Ai_jax, self._Aj_jax, Ax, jnp.asarray(rhs_np[:, c])
                    ),
                    dtype=np.float64,
                )
        else:
            # Fallback: assemble scipy sparse A and factorize per call.
            A_csc = sp.csc_matrix(
                (self._numpy_Ax(coeffs), (self.Ai, self.Aj)),
                shape=(self.n, self.n),
            )
            lu = sp.linalg.splu(A_csc)
            for c in range(n_rhs):
                out[:, c] = np.asarray(lu.solve(rhs_np[:, c]), dtype=np.float64)
        return out[:, 0] if single else out

    def _numpy_Ax(self, coeffs):
        ax = self.const_vals.copy()
        for c, wv in zip(coeffs, self.w_vals_list):
            ax = ax + float(c) * wv
        return ax


def profile_loglik_rho_grid(
    y,
    X,
    W_sparse,
    *,
    rho_min: float = 0.05,
    rho_max: float = 0.95,
    rho_step: float = 0.05,
):
    r"""Profile-log-likelihood ρ-grid search with cached sparse solves.

    For each candidate ρ on ``[rho_min, rho_max]`` step ``rho_step``, solves
    :math:`\tilde X = (I - \rho W)^{-1} X` and computes the Gaussian
    profile log-likelihood

    .. math::

        \ell_p(\rho) = -\tfrac{n}{2}\log\hat\sigma^2(\rho) - \tfrac{n}{2},
        \quad \hat\beta(\rho) = (\tilde X^\top \tilde X)^{-1} \tilde X^\top y,
        \quad \hat\sigma^2(\rho) = \tfrac{1}{n}\|y - \tilde X \hat\beta\|^2.

    The sparsity pattern of :math:`I - \rho W` is independent of ρ, so a
    single :class:`CachedSparseSolver` is built once and reused across the
    whole grid (sparsax caches the fill-reducing symbolic analysis; scipy
    fallback still benefits from the precomputed pattern assembly).

    Parameters
    ----------
    y, X : ndarray, shapes (n,) and (n, k)
        Response and design matrix.
    W_sparse : scipy.sparse matrix, shape (n, n)
        Row-standardized spatial weights.
    rho_min, rho_max, rho_step : float
        Grid definition.

    Returns
    -------
    best_rho : float
        Grid argmax of the profile log-likelihood.
    best_beta : ndarray, shape (k,)
        Least-squares β at ``best_rho``.
    best_ll : float
        Profile log-likelihood at ``best_rho``.  ``-np.inf`` when every solve
        failed (e.g. ρ outside the valid range for ``W``).
    """
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    n, k = X.shape
    solver = CachedSparseSolver([W_sparse], n)
    grid = np.arange(rho_min, rho_max, rho_step)
    best_rho, best_beta, best_ll = 0.0, np.zeros(k), -np.inf
    for rho_g in grid:
        try:
            Xtilde = solver.solve([-float(rho_g)], X)
            beta_g = np.linalg.lstsq(Xtilde, y, rcond=None)[0]
            eta_g = Xtilde @ beta_g
            sig2_g = float(np.mean((y - eta_g) ** 2))
            if sig2_g > 1e-10:
                ll_g = -0.5 * n * np.log(sig2_g) - 0.5 * n
                if ll_g > best_ll:
                    best_ll, best_rho, best_beta = ll_g, float(rho_g), beta_g.copy()
        except Exception:
            pass
    return best_rho, best_beta, best_ll


def cached_sar_solve(W_sparse, n, rho, rhs):
    """Solve :math:`(I - \\rho W) x = b` with one cached symbolic analysis.

    Thin convenience wrapper over :class:`CachedSparseSolver` for the common
    one-off case: build the solver, solve once, return the result.  Callers
    doing many solves should construct a :class:`CachedSparseSolver` once
    and call ``solve([-ρ], rhs)`` per draw to amortise the pattern build.
    """
    return CachedSparseSolver([W_sparse], n).solve([-float(rho)], rhs)
