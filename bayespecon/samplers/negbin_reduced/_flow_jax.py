r"""JAX/klujax sparse solve primitives for the unrestricted flow NB Gibbs sampler.

The unrestricted origin–destination flow model has system matrix

.. math::

    A(\rho_d, \rho_o, \rho_w) = I - \rho_d W_d - \rho_o W_o - \rho_w W_w

on the ``N = n^2`` flow lattice.  ``A`` is **directed** (non-symmetric,
non-D-symmetrizable), so no Cholesky applies; and it is far too large to
densify (``N \times N`` with ``N = n^2``).  The numpy chain factorises the
sparse ``A`` on the host every time a ``\rho`` moves (see
``_flow._solve_A_unrestricted``).  This module provides the JAX-native
equivalent: a single ``klujax`` symbolic analysis reused across the whole
run, with per-``\rho`` numeric refactor-and-solve that is JIT-compatible and
autodiff-capable — the enabling piece for a GPU-friendly flow backend.

The crucial invariant is that **the sparsity pattern of ``A`` is constant**
across ``\rho`` (it is the structural union of ``I, W_d, W_o, W_w``).  We
build that shared pattern once and carry four value vectors aligned to it, so
each solve only rescales values and calls ``klujax.solve_with_symbol`` — the
symbolic factorisation (AMD ordering + elimination tree) is never redone.

Keeping this alongside the numpy host path is intentional: klujax shines on
GPU, while host KLU/UMFPACK remains competitive on CPU.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def build_flow_pattern(
    Wd: sp.spmatrix,
    Wo: sp.spmatrix,
    Ww: sp.spmatrix,
    N: int,
) -> dict:
    """Build the shared COO pattern of ``I - ρ_d W_d - ρ_o W_o - ρ_w W_w``.

    Returns ``Ai, Aj`` (int32 COO coordinates of the structural union) plus
    four float64 value vectors aligned to that pattern — ``eye_vals`` (1 on
    the diagonal), ``wd_vals``, ``wo_vals``, ``ww_vals`` — such that

    ``Ax(ρ) = eye_vals - ρ_d·wd_vals - ρ_o·wo_vals - ρ_w·ww_vals``

    is exactly ``A(ρ)`` on that pattern.  All four vectors share identical
    ``(Ai, Aj)`` ordering because they are assembled over the same
    concatenated coordinate list and ``sum_duplicates`` sorts deterministically.
    """
    eye = sp.eye(N, format="coo")
    Wd, Wo, Ww = Wd.tocoo(), Wo.tocoo(), Ww.tocoo()
    rows = np.concatenate([eye.row, Wd.row, Wo.row, Ww.row])
    cols = np.concatenate([eye.col, Wd.col, Wo.col, Ww.col])
    nnz = (eye.nnz, Wd.nnz, Wo.nnz, Ww.nnz)

    def _slot(parts: list[np.ndarray]) -> sp.coo_matrix:
        c = sp.coo_matrix((np.concatenate(parts), (rows, cols)), shape=(N, N))
        c.sum_duplicates()
        return c

    z = [np.zeros(m) for m in nnz]
    eye_c = _slot([np.ones(nnz[0]), z[1], z[2], z[3]])
    wd_c = _slot([z[0], Wd.data, z[2], z[3]])
    wo_c = _slot([z[0], z[1], Wo.data, z[3]])
    ww_c = _slot([z[0], z[1], z[2], Ww.data])

    # All four coo's share identical (row, col) after sum_duplicates.
    return {
        "Ai": np.asarray(eye_c.row, dtype=np.int32),
        "Aj": np.asarray(eye_c.col, dtype=np.int32),
        "eye_vals": np.asarray(eye_c.data, dtype=np.float64),
        "wd_vals": np.asarray(wd_c.data, dtype=np.float64),
        "wo_vals": np.asarray(wo_c.data, dtype=np.float64),
        "ww_vals": np.asarray(ww_c.data, dtype=np.float64),
        "N": int(N),
    }


def build_sar_pattern(W: sp.spmatrix, n: int) -> dict:
    """Build the shared COO pattern of ``I - ρW`` (single-ρ reduced-form SAR).

    The single-ρ analogue of :func:`build_flow_pattern`: returns ``Ai, Aj``
    (int32 COO of the structural union of ``I`` and ``W``) plus aligned value
    vectors ``eye_vals`` (1 on the diagonal) and ``w_vals``, so that
    ``Ax(ρ) = eye_vals - ρ·w_vals`` is exactly ``I - ρW`` on that pattern.
    Never densifies ``W``.
    """
    eye = sp.eye(n, format="coo")
    Wc = W.tocoo()
    rows = np.concatenate([eye.row, Wc.row])
    cols = np.concatenate([eye.col, Wc.col])

    def _slot(parts: list[np.ndarray]) -> sp.coo_matrix:
        c = sp.coo_matrix((np.concatenate(parts), (rows, cols)), shape=(n, n))
        c.sum_duplicates()
        return c

    eye_c = _slot([np.ones(eye.nnz), np.zeros(Wc.nnz)])
    w_c = _slot([np.zeros(eye.nnz), Wc.data])
    return {
        "Ai": np.asarray(eye_c.row, dtype=np.int32),
        "Aj": np.asarray(eye_c.col, dtype=np.int32),
        "eye_vals": np.asarray(eye_c.data, dtype=np.float64),
        "w_vals": np.asarray(w_c.data, dtype=np.float64),
        "N": int(n),
    }


def make_flow_solve(pattern: dict):
    """Build a JIT-compiled ``solve(ρ_d, ρ_o, ρ_w, rhs) -> A(ρ)⁻¹ rhs``.

    Uses one cached ``klujax`` symbolic analysis over the shared pattern; each
    call only rebuilds the value vector ``Ax(ρ)`` and calls
    ``solve_with_symbol``.  ``rhs`` may be a vector ``(N,)`` or matrix
    ``(N, k)`` (batched solve — used for ``X̃ = A⁻¹X``).
    """
    import jax
    import jax.numpy as jnp
    import klujax

    from bayespecon._jax_dispatch import ensure_x64

    ensure_x64()

    Ai = jnp.asarray(pattern["Ai"])
    Aj = jnp.asarray(pattern["Aj"])
    eye_vals = jnp.asarray(pattern["eye_vals"])
    wd_vals = jnp.asarray(pattern["wd_vals"])
    wo_vals = jnp.asarray(pattern["wo_vals"])
    ww_vals = jnp.asarray(pattern["ww_vals"])
    N = pattern["N"]
    symbolic = klujax.analyze(pattern["Ai"], pattern["Aj"], N)

    @jax.jit
    def solve(rho_d, rho_o, rho_w, rhs):
        Ax = eye_vals - rho_d * wd_vals - rho_o * wo_vals - rho_w * ww_vals
        return klujax.solve_with_symbol(Ai, Aj, Ax, rhs, symbolic)

    return solve


def build_flow_ctx(Wd, Wo, Ww, N) -> dict:
    """Sparse klujax context for the unrestricted flow (W never densified).

    Bundles the shared COO pattern (:func:`build_flow_pattern`), a cached
    klujax symbolic factorisation, and BCOO copies of the three lag matrices
    for sparse matvecs.
    """
    import klujax
    from jax.experimental import sparse as jsparse

    ctx = build_flow_pattern(Wd.tocsr(), Wo.tocsr(), Ww.tocsr(), N)
    ctx["symbolic"] = klujax.analyze(ctx["Ai"], ctx["Aj"], N)
    ctx["Wd_bcoo"] = jsparse.BCOO.from_scipy_sparse(Wd.tocsr())
    ctx["Wo_bcoo"] = jsparse.BCOO.from_scipy_sparse(Wo.tocsr())
    ctx["Ww_bcoo"] = jsparse.BCOO.from_scipy_sparse(Ww.tocsr())
    return ctx


def _make_flow_solvers(ctx):
    """Build klujax solve closures for ``A(ρ_d,ρ_o,ρ_w) = I−ρ_dWd−ρ_oWo−ρ_wWw``.

    Returns ``(factor_at, solve_num, matvec)`` where ``factor_at(ρ_d,ρ_o,ρ_w)``
    is a reusable numeric factorisation, ``solve_num(numeric, rhs)`` reuses it,
    and ``matvec`` is a dict ``{"d","o","w"}`` of sparse (BCOO) lag matvecs.
    The symbolic factorisation is cached and reused across the whole run.
    """
    import jax.numpy as jnp
    import klujax

    Ai = jnp.asarray(ctx["Ai"])
    Aj = jnp.asarray(ctx["Aj"])
    eye_vals = jnp.asarray(ctx["eye_vals"])
    wd_vals = jnp.asarray(ctx["wd_vals"])
    wo_vals = jnp.asarray(ctx["wo_vals"])
    ww_vals = jnp.asarray(ctx["ww_vals"])
    symbolic = ctx["symbolic"]
    Wd_bcoo, Wo_bcoo, Ww_bcoo = ctx["Wd_bcoo"], ctx["Wo_bcoo"], ctx["Ww_bcoo"]

    def factor_at(rho_d, rho_o, rho_w):
        Ax = eye_vals - rho_d * wd_vals - rho_o * wo_vals - rho_w * ww_vals
        return klujax.factor(Ai, Aj, Ax, symbolic)

    def solve_num(numeric, rhs):
        return klujax.solve_with_numeric(numeric, rhs, symbolic)

    matvec = {
        "d": lambda v: Wd_bcoo @ v,
        "o": lambda v: Wo_bcoo @ v,
        "w": lambda v: Ww_bcoo @ v,
    }
    return factor_at, solve_num, matvec


def _make_flow_gibbs_step(
    y_jax,
    X_jax,
    ctx,
    n,
    k,
    priors,
    *,
    krylov_degree,
    krylov_dmax,
    positive,
    n_cycles,
):
    """Build a JIT-compiled unrestricted-flow NB Gibbs step (ω → 3×ρ → β → α).

    Reuses the reduced-form cross-section blocks: sampling one ρ_k (holding the
    other two fixed) is ``(A_0 − Δρ_k W_k)⁻¹X``, structurally identical to the
    single-ρ SAR slice, so the same shift-invert Krylov basis + slice sampler
    apply with ``W_k`` as the direction and the current ``A_0`` as the base.
    The joint stability wall ``|ρ_d|+|ρ_o|+|ρ_w| < ρ_upper`` is enforced through
    the per-ρ_k slice bounds.  W is never densified.
    """
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import cho_solve, solve_triangular

    from bayespecon._jax_dispatch import ensure_x64

    from .._utils._jax_polyagamma import jax_polyagamma
    from ._jax import (
        _build_krylov_basis_jax,
        _sample_alpha_jax_reduced,
        _slice_sample_rho_jax,
    )

    ensure_x64()
    factor_at, solve_num, matvec = _make_flow_solvers(ctx)

    beta_sigma = priors.beta_sigma
    if np.isscalar(beta_sigma):
        V0_inv_diag = jnp.full(k, 1.0 / (float(beta_sigma) ** 2))
    else:
        V0_inv_diag = 1.0 / jnp.asarray(beta_sigma, dtype=jnp.float64) ** 2
    beta_mu = priors.beta_mu
    mu0 = (
        jnp.full(k, float(beta_mu))
        if np.isscalar(beta_mu)
        else jnp.asarray(beta_mu, dtype=jnp.float64)
    )
    rho_lo = jnp.float64(priors.rho_lower)
    rho_hi = jnp.float64(priors.rho_upper)
    B = jnp.float64(priors.rho_upper)  # joint stability wall bound
    alpha_sigma = jnp.float64(priors.alpha_sigma)
    alpha_nu = jnp.float64(priors.alpha_nu)
    dmax = jnp.float64(krylov_dmax)
    _pos = bool(positive)

    def _wall_bounds(other_abs_sum):
        room = B - other_abs_sum
        lo = jnp.maximum(rho_lo, -room)
        hi = jnp.minimum(rho_hi, room)
        if _pos:
            lo = jnp.maximum(lo, 0.0)
        return lo, hi

    def _draw_omega(y, alpha, eta, key):
        h = jnp.maximum(y + alpha, 1e-3)
        z = jnp.clip(eta - jnp.log(alpha), -20.0, 20.0)
        return jax_polyagamma(h, z, key=key, method="callback")

    def _draw_beta(Xtilde, omega, alpha, key):
        kappa = 0.5 * (y_jax - alpha)
        log_alpha = jnp.log(alpha)
        Xt_omega = Xtilde * omega[:, None]
        Sig_inv = Xt_omega.T @ Xtilde + jnp.diag(V0_inv_diag) + 1e-10 * jnp.eye(k)
        rhs = Xtilde.T @ (kappa + omega * log_alpha) + V0_inv_diag * mu0
        L = jnp.linalg.cholesky(Sig_inv)
        m = cho_solve((L, True), rhs)
        z = jax.random.normal(key, shape=(k,), dtype=jnp.float64)
        return m + solve_triangular(L.T, z, lower=False)

    def _slice_one(rho_k, rd, ro, rw, wkey, other_abs, omega, alpha, slice_width, key):
        """One ρ_k slice with a W_k-direction basis at the current A_0."""
        num0 = factor_at(rd, ro, rw)
        V_stack = _build_krylov_basis_jax(
            num0, X_jax, solve_num, matvec[wkey], n, k, krylov_degree
        )
        lo, hi = _wall_bounds(other_abs)

        # Solve at a candidate ρ_k holding the other two fixed.
        if wkey == "d":
            solve_at = lambda rc, rhs: solve_num(factor_at(rc, ro, rw), rhs)  # noqa: E731
        elif wkey == "o":
            solve_at = lambda rc, rhs: solve_num(factor_at(rd, rc, rw), rhs)  # noqa: E731
        else:
            solve_at = lambda rc, rhs: solve_num(factor_at(rd, ro, rc), rhs)  # noqa: E731

        return _slice_sample_rho_jax(
            rho_current=rho_k,
            V_stack=V_stack,
            rho_basis=rho_k,
            omega=omega,
            y_jax=y_jax,
            alpha=alpha,
            V0_inv_diag=V0_inv_diag,
            mu0=mu0,
            intercept_col=-1,
            rho_lower=lo,
            rho_upper=hi,
            krylov_dmax=dmax,
            slice_width=slice_width,
            key=key,
            X_jax=X_jax,
            solve_at=solve_at,
        )

    @jax.jit
    def gibbs_step(state, key, slice_width):
        beta = state["beta"]
        rd, ro, rw = state["rho_d"], state["rho_o"], state["rho_w"]
        alpha = state["alpha"]

        num0 = factor_at(rd, ro, rw)
        eta = solve_num(num0, X_jax @ beta)
        key, kpg = jax.random.split(key)
        omega = _draw_omega(y_jax, alpha, eta, kpg)

        for cyc in range(n_cycles):
            key, kd, ko, kw, kb = jax.random.split(key, 5)
            rd = _slice_one(
                rd,
                rd,
                ro,
                rw,
                "d",
                jnp.abs(ro) + jnp.abs(rw),
                omega,
                alpha,
                slice_width,
                kd,
            )
            ro = _slice_one(
                ro,
                rd,
                ro,
                rw,
                "o",
                jnp.abs(rd) + jnp.abs(rw),
                omega,
                alpha,
                slice_width,
                ko,
            )
            rw = _slice_one(
                rw,
                rd,
                ro,
                rw,
                "w",
                jnp.abs(rd) + jnp.abs(ro),
                omega,
                alpha,
                slice_width,
                kw,
            )

            num_f = factor_at(rd, ro, rw)
            Xtilde = solve_num(num_f, X_jax)
            beta = _draw_beta(Xtilde, omega, alpha, kb)
            eta = Xtilde @ beta
            if cyc < n_cycles - 1:
                key, kpg2 = jax.random.split(key)
                omega = _draw_omega(y_jax, alpha, eta, kpg2)

        key, ka = jax.random.split(key)
        alpha = _sample_alpha_jax_reduced(eta, y_jax, alpha, alpha_sigma, alpha_nu, ka)

        return {
            "beta": beta,
            "rho_d": rd,
            "rho_o": ro,
            "rho_w": rw,
            "alpha": alpha,
            "omega": omega,
        }, jnp.float64(1.0)

    return gibbs_step


def run_chains_jax_flow(
    y,
    X,
    Wd,
    Wo,
    Ww,
    priors,
    inits,
    draws,
    tune,
    *,
    thin=1,
    krylov_degree=8,
    krylov_dmax=0.15,
    positive=False,
    n_cycles=1,
    jax_seeds=None,
    progressbar=False,
    slice_width=0.2,
):
    """Run the unrestricted flow NB Gibbs sampler on the JAX/klujax backend.

    Drop-in analogue of the numpy ``run_chain_unrestricted`` (one chain per
    ``inits`` entry) that never densifies the ``N×N`` (``N = n²``) flow matrix:
    all solves go through klujax sparse LU with a cached symbolic factorisation.

    Returns one dict per chain with keys ``rho_d``, ``rho_o``, ``rho_w``,
    ``beta``, ``alpha``, ``log_lik``.
    """
    import jax
    import jax.numpy as jnp
    from scipy.special import gammaln

    from bayespecon._jax_dispatch import ensure_x64

    ensure_x64()
    N, k = X.shape
    y_jax = jnp.asarray(y, dtype=jnp.float64)
    X_jax = jnp.asarray(X, dtype=jnp.float64)
    ctx = build_flow_ctx(Wd, Wo, Ww, N)
    flow_solve = make_flow_solve(ctx)  # host-side post-hoc eta for log-lik
    slice_width_jax = jnp.float64(slice_width)

    gibbs_step = _make_flow_gibbs_step(
        y_jax,
        X_jax,
        ctx,
        N,
        k,
        priors,
        krylov_degree=krylov_degree,
        krylov_dmax=krylov_dmax,
        positive=positive,
        n_cycles=n_cycles,
    )

    if jax_seeds is None:
        jax_seeds = list(range(len(inits)))
    n_keep = draws // thin if thin > 0 else draws
    y_np = np.asarray(y, dtype=np.float64)

    results = []
    for c, init in enumerate(inits):
        key = jax.random.PRNGKey(jax_seeds[c])
        state = {
            "beta": jnp.asarray(init.beta, dtype=jnp.float64),
            "rho_d": jnp.float64(init.rho_d),
            "rho_o": jnp.float64(init.rho_o),
            "rho_w": jnp.float64(init.rho_w if init.rho_w is not None else 0.0),
            "alpha": jnp.float64(init.alpha),
            "omega": jnp.asarray(init.omega, dtype=jnp.float64),
        }
        # Warmup
        for _ in range(tune):
            key, sk = jax.random.split(key)
            state, _ = gibbs_step(state, sk, slice_width_jax)
        # Sampling
        rd_s = np.empty(n_keep)
        ro_s = np.empty(n_keep)
        rw_s = np.empty(n_keep)
        beta_s = np.empty((n_keep, k))
        alpha_s = np.empty(n_keep)
        kept = 0
        for i in range(draws):
            key, sk = jax.random.split(key)
            state, _ = gibbs_step(state, sk, slice_width_jax)
            if i % thin == 0 and kept < n_keep:
                rd_s[kept] = float(state["rho_d"])
                ro_s[kept] = float(state["rho_o"])
                rw_s[kept] = float(state["rho_w"])
                alpha_s[kept] = float(state["alpha"])
                beta_s[kept] = np.asarray(state["beta"])
                kept += 1

        # Post-hoc pointwise log-likelihood via sparse flow solves (no densify).
        log_lik = np.empty((n_keep, N))
        for j in range(n_keep):
            eta = np.asarray(
                flow_solve(
                    float(rd_s[j]),
                    float(ro_s[j]),
                    float(rw_s[j]),
                    X_jax @ jnp.asarray(beta_s[j]),
                )
            )
            mu = np.exp(np.clip(eta, -30.0, 30.0))
            a = alpha_s[j]
            log_lik[j] = (
                gammaln(y_np + a)
                - gammaln(a)
                + y_np * np.log(np.maximum(mu / (mu + a), 1e-300))
                + a * np.log(np.maximum(a / (mu + a), 1e-300))
            )
        results.append(
            {
                "rho_d": rd_s,
                "rho_o": ro_s,
                "rho_w": rw_s,
                "beta": beta_s,
                "alpha": alpha_s,
                "log_lik": log_lik,
            }
        )
    return results
