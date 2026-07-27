r"""JAX/cholgraph sparse solve primitives for the unrestricted flow NB Gibbs sampler.

The unrestricted origin–destination flow model has system matrix

.. math::

    A(\rho_d, \rho_o, \rho_w) = I - \rho_d W_d - \rho_o W_o - \rho_w W_w

on the ``N = n^2`` flow lattice.  ``A`` is **directed** (non-symmetric,
non-D-symmetrizable), so no Cholesky applies; and it is far too large to
densify (``N \times N`` with ``N = n^2``).  The numpy chain factorises the
sparse ``A`` on the host every time a ``\rho`` moves (see
``_flow._solve_A_unrestricted``).  This module provides the JAX-native
equivalent: a single ``cholgraph`` symbolic analysis reused across the whole
run, with per-``\rho`` numeric refactor-and-solve that is JIT-compatible and
autodiff-capable — the enabling piece for a GPU-friendly flow backend.

The crucial invariant is that **the sparsity pattern of ``A`` is constant**
across ``\rho`` (it is the structural union of ``I, W_d, W_o, W_w``).  We
build that shared pattern once and carry four value vectors aligned to it, so
each solve only rescales values and calls ``cholgraph.lu_solve`` — the
symbolic factorisation (AMD ordering + elimination tree) is never redone.

Keeping this alongside the numpy host path is intentional: cholgraph shines on
GPU, while host KLU/UMFPACK remains competitive on CPU.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ._core import _KRYLOV_DEGREE_DEFAULT, _KRYLOV_DMAX_DEFAULT


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

    Uses ``cholgraph.lu_solve`` (SuiteSparse KLU): the fill-reducing analysis is
    cached by the shared pattern, so each call only rebuilds the value vector
    ``Ax(ρ)``.  ``rhs`` may be a vector ``(N,)`` or matrix ``(N, k)`` (batched
    solve — used for ``X̃ = A⁻¹X``).
    """
    import cholgraph
    import jax
    import jax.numpy as jnp

    from bayespecon._jax_dispatch import ensure_x64

    ensure_x64()

    Ai = jnp.asarray(pattern["Ai"], jnp.int32)
    Aj = jnp.asarray(pattern["Aj"], jnp.int32)
    eye_vals = jnp.asarray(pattern["eye_vals"])
    wd_vals = jnp.asarray(pattern["wd_vals"])
    wo_vals = jnp.asarray(pattern["wo_vals"])
    ww_vals = jnp.asarray(pattern["ww_vals"])

    @jax.jit
    def solve(rho_d, rho_o, rho_w, rhs):
        Ax = eye_vals - rho_d * wd_vals - rho_o * wo_vals - rho_w * ww_vals
        return cholgraph.lu_solve(Ai, Aj, Ax, rhs)

    return solve


def build_flow_ctx(Wd, Wo, Ww, N) -> dict:
    """Sparse solve context for the unrestricted flow (W never densified).

    Bundles the shared COO pattern (:func:`build_flow_pattern`) and BCOO copies
    of the three lag matrices for sparse matvecs.  The fill-reducing symbolic
    analysis is cached inside cholgraph, keyed on the (constant) pattern.
    """
    from jax.experimental import sparse as jsparse

    ctx = build_flow_pattern(Wd.tocsr(), Wo.tocsr(), Ww.tocsr(), N)
    ctx["Wd_bcoo"] = jsparse.BCOO.from_scipy_sparse(Wd.tocsr())
    ctx["Wo_bcoo"] = jsparse.BCOO.from_scipy_sparse(Wo.tocsr())
    ctx["Ww_bcoo"] = jsparse.BCOO.from_scipy_sparse(Ww.tocsr())
    return ctx


def _make_flow_solvers(ctx):
    """Build sparse-LU solve closures for ``A(ρ_d,ρ_o,ρ_w) = I−ρ_dWd−ρ_oWo−ρ_wWw``.

    Returns ``(solve, matvec)`` where ``solve(ρ_d,ρ_o,ρ_w,rhs)`` →
    ``A(ρ)⁻¹ rhs`` via ``cholgraph.lu_solve`` (SuiteSparse KLU) and ``matvec``
    is a dict ``{"d","o","w"}`` of sparse (BCOO) lag matvecs.

    ``cholgraph.lu_solve`` is vmap-safe and reuses its numeric factorisation via
    a content-addressed cache: the m+1 solves of a Krylov basis at a fixed
    (ρ_d,ρ_o,ρ_w) pay one ``klu_factor`` and m cheap solves — per chain — even
    under ``jax.vmap`` over chains, which stays vmap-safe under ``jit(vmap(...))``;
    see ``set_lu_cache_size``.
    """
    import cholgraph
    import jax.numpy as jnp

    Ai = jnp.asarray(ctx["Ai"], jnp.int32)
    Aj = jnp.asarray(ctx["Aj"], jnp.int32)
    eye_vals = jnp.asarray(ctx["eye_vals"])
    wd_vals = jnp.asarray(ctx["wd_vals"])
    wo_vals = jnp.asarray(ctx["wo_vals"])
    ww_vals = jnp.asarray(ctx["ww_vals"])
    Wd_bcoo, Wo_bcoo, Ww_bcoo = ctx["Wd_bcoo"], ctx["Wo_bcoo"], ctx["Ww_bcoo"]

    def solve(rho_d, rho_o, rho_w, rhs):
        Ax = eye_vals - rho_d * wd_vals - rho_o * wo_vals - rho_w * ww_vals
        return cholgraph.lu_solve(Ai, Aj, Ax, rhs)

    matvec = {
        "d": lambda v: Wd_bcoo @ v,
        "o": lambda v: Wo_bcoo @ v,
        "w": lambda v: Ww_bcoo @ v,
    }
    return solve, matvec


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
    solve, matvec = _make_flow_solvers(ctx)

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
        """One ρ_k slice with a W_k-direction basis at the current A_0.

        Krylov-only (``solve_at=None``): candidates outside the Krylov radius are
        rejected rather than evaluated with a per-candidate direct solve, which
        under ``jax.vmap`` would be computed for *every* candidate (the dominant
        cost).  The bounded ρ_k step this induces is offset by a wider
        ``krylov_dmax`` with enough degree to stay accurate.
        """
        V_stack = _build_krylov_basis_jax(
            lambda rhs: solve(rd, ro, rw, rhs),
            X_jax,
            matvec[wkey],
            n,
            k,
            krylov_degree,
        )
        lo, hi = _wall_bounds(other_abs)

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
            X_jax=None,
            solve_at=None,
        )

    @jax.jit
    def gibbs_step(state, key, slice_width):
        beta = state["beta"]
        rd, ro, rw = state["rho_d"], state["rho_o"], state["rho_w"]
        alpha = state["alpha"]

        eta = solve(rd, ro, rw, X_jax @ beta)
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

            # β step needs X̃ = A(ρ_new)⁻¹X at the just-updated (ρ_d,ρ_o,ρ_w);
            # all three moved, so no single basis covers it — one direct solve.
            Xtilde = solve(rd, ro, rw, X_jax)
            beta = _draw_beta(Xtilde, omega, alpha, kb)
            eta = Xtilde @ beta
            if cyc < n_cycles - 1:
                key, kpg2 = jax.random.split(key)
                omega = _draw_omega(y_jax, alpha, eta, kpg2)

        key, ka = jax.random.split(key)
        alpha = _sample_alpha_jax_reduced(eta, y_jax, alpha, alpha_sigma, alpha_nu, ka)

        # Return the fitted latent η so the runner forms the pointwise NB
        # log-likelihood on-device (reusing the sweep's solve) instead of a
        # post-hoc per-draw host-solve loop.
        return {
            "beta": beta,
            "rho_d": rd,
            "rho_o": ro,
            "rho_w": rw,
            "alpha": alpha,
            "omega": omega,
        }, eta

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
    krylov_degree=_KRYLOV_DEGREE_DEFAULT,
    krylov_dmax=_KRYLOV_DMAX_DEFAULT,
    positive=False,
    n_cycles=1,
    jax_seeds=None,
    progressbar=False,
    slice_width=0.4,
):
    """Run the unrestricted flow NB Gibbs sampler on the JAX backend.

    All chains run together under ``jax.vmap`` (like the reduced-form SAR-NB and
    logit paths).  The non-symmetric LU solve goes through ``cholgraph.lu_solve``
    (SuiteSparse KLU) — vmap-safe with numeric factor-reuse under
    ``jit(vmap(...))``.  The three ρ
    slices are Krylov-only (no per-candidate direct solve, which under vmap would
    run for every candidate).  ``W`` is never densified; the exact PG draw uses
    the host callback.

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
    sw = jnp.float64(slice_width)

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

    chains = len(inits)
    if jax_seeds is None:
        jax_seeds = list(range(chains))

    # cholgraph's KLU factor cache must hold each chain's distinct factors live
    # across the sweep's several solves (η, the 3 directional bases, X̃) for the
    # vmapped reuse to land; size generously per chain.
    import cholgraph

    cholgraph.set_lu_cache_size(max(32, 8 * chains))

    # All chains run together under jax.vmap — vmap-safe now that the LU solve is
    # cholgraph.lu_solve (factor-reusing) and the ρ slices are Krylov-only.
    state0 = {
        "beta": jnp.asarray(np.stack([i.beta for i in inits]), dtype=jnp.float64),
        "rho_d": jnp.asarray([float(i.rho_d) for i in inits], dtype=jnp.float64),
        "rho_o": jnp.asarray([float(i.rho_o) for i in inits], dtype=jnp.float64),
        "rho_w": jnp.asarray(
            [float(i.rho_w if i.rho_w is not None else 0.0) for i in inits],
            dtype=jnp.float64,
        ),
        "alpha": jnp.asarray([float(i.alpha) for i in inits], dtype=jnp.float64),
        "omega": jnp.asarray(np.stack([i.omega for i in inits]), dtype=jnp.float64),
    }
    warm_keys = jnp.stack([jax.random.PRNGKey(int(s)) for s in jax_seeds])
    draw_keys = jnp.stack(
        [jax.random.fold_in(jax.random.PRNGKey(int(s)), 1) for s in jax_seeds]
    )

    def _warm_one(s, key):
        def body(_, carry):
            st, kk = carry
            kk, sk = jax.random.split(kk)
            st, _ = gibbs_step(st, sk, sw)
            return (st, kk)

        st, _ = jax.lax.fori_loop(0, tune, body, (s, key))
        return st

    def _draw_one(s, key):
        def body(carry, _):
            st, kk = carry
            kk, sk = jax.random.split(kk)
            st, eta = gibbs_step(st, sk, sw)
            return (st, kk), (
                st["rho_d"],
                st["rho_o"],
                st["rho_w"],
                st["beta"],
                st["alpha"],
                eta,
            )

        _, traces = jax.lax.scan(body, (s, key), None, length=draws)
        return traces

    # One chain per CPU device (pmap) when available, else vmap — see
    # negbin_reduced._jax._run_chains_device_parallel.
    from ._jax import _run_chains_device_parallel

    rd_all, ro_all, rw_all, beta_all, alpha_all, eta_all = _run_chains_device_parallel(
        _warm_one, _draw_one, state0, warm_keys, draw_keys, chains, tune
    )
    sl = slice(None, None, thin) if thin > 1 else slice(None)
    rd_all = np.asarray(rd_all)[:, sl]
    ro_all = np.asarray(ro_all)[:, sl]
    rw_all = np.asarray(rw_all)[:, sl]
    beta_all = np.asarray(beta_all)[:, sl]
    alpha_all = np.asarray(alpha_all)[:, sl]
    eta_all = np.asarray(eta_all)[:, sl]  # (chains, n_keep, N)

    # Pointwise NB log-likelihood from the fitted η collected during sampling —
    # no post-hoc per-draw solves.
    y_np = np.asarray(y, dtype=np.float64)
    results = []
    for c in range(chains):
        alpha_s = alpha_all[c]
        mu = np.exp(np.clip(eta_all[c], -30.0, 30.0))  # (n_keep, N)
        a = alpha_s[:, None]
        log_lik = (
            gammaln(y_np + a)
            - gammaln(a)
            + y_np * np.log(np.maximum(mu / (mu + a), 1e-300))
            + a * np.log(np.maximum(a / (mu + a), 1e-300))
        )
        results.append(
            {
                "rho_d": rd_all[c],
                "rho_o": ro_all[c],
                "rho_w": rw_all[c],
                "beta": beta_all[c],
                "alpha": alpha_s,
                "log_lik": log_lik,
            }
        )
    return results
