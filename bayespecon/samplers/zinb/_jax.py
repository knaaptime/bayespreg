r"""JAX reduced-form Zero-Inflated SAR Negative-Binomial Pólya-Gamma Gibbs.

Composes the two reduced-form jax samplers already built:

* **selection** — reduced-form SAR-**logit** on the *latent* activation ``z``
  (``η_sel = (I − λ W_sel)⁻¹ Z γ``), reusing ``logit_reduced``'s Krylov-only
  ρ-slice density; the working response is ``κ_sel = z − ½`` and ``z`` is redrawn
  every sweep;
* **count** — reduced-form SAR-**NB** on the counts (``η_cnt = (I − ρ W_cnt)⁻¹ X β``),
  reusing ``negbin_reduced``'s Krylov-only density, **z-masked** (structural
  zeros contribute nothing: ``ω_cnt = ε`` and the working ``y`` is set to ``α``
  so ``κ = 0`` there);

linked by the latent indicator ``z``.  Both equations use cholgraph-KLU solves
(never densified), the on-device Pólya-Gamma draw (pgjax), and run each chain on
its own CPU device via ``jax.pmap``.
"""

from __future__ import annotations

import numpy as np

from ..negbin_reduced._core import _KRYLOV_DEGREE_DEFAULT, _KRYLOV_DMAX_DEFAULT


def _make_zinb_gibbs_step(
    y_jax,
    d_jax,
    Z_jax,
    X_jax,
    sel_ctx,
    cnt_ctx,
    n,
    p,
    k,
    priors,
    *,
    krylov_degree,
    krylov_dmax,
):
    """Build a JIT-compiled reduced-form ZINB Gibbs step (9 blocks)."""
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import cho_solve, solve_triangular

    from bayespecon._jax_dispatch import ensure_x64

    from .._utils._jax_slice import jax_slice_sample_1d
    from ..logit_reduced._jax import _rho_log_density_logit
    from ..negbin_reduced._jax import (
        _build_krylov_basis_jax,
        _eval_U_from_basis_jax,
        _make_sparse_solvers,
        _rho_log_density_marginal_jax,
    )

    ensure_x64()

    try:
        import pgjax

        def _draw_pg(hh, zz, kk):
            return pgjax.pg_sample(hh, zz, kk)
    except ImportError:
        from .._utils._jax_polyagamma import jax_polyagamma

        def _draw_pg(hh, zz, kk):
            return jax_polyagamma(hh, zz, key=kk, method="callback")

    _solve_sel, _matvec_Wsel = _make_sparse_solvers(sel_ctx)
    _solve_cnt, _matvec_Wcnt = _make_sparse_solvers(cnt_ctx)

    def _prior_vec(mu, sigma, dim):
        v0 = (
            jnp.full(dim, 1.0 / float(sigma) ** 2)
            if np.isscalar(sigma)
            else 1.0 / jnp.asarray(sigma, dtype=jnp.float64) ** 2
        )
        m0 = (
            jnp.full(dim, float(mu))
            if np.isscalar(mu)
            else jnp.asarray(mu, dtype=jnp.float64)
        )
        return v0, m0

    V0g, mu0g = _prior_vec(priors.gamma_mu, priors.gamma_sigma, p)
    V0b, mu0b = _prior_vec(priors.beta_mu, priors.beta_sigma, k)
    lam_lo, lam_hi = jnp.float64(priors.lam_lower), jnp.float64(priors.lam_upper)
    rho_lo, rho_hi = jnp.float64(priors.rho_lower), jnp.float64(priors.rho_upper)
    a_sigma, a_nu = jnp.float64(priors.alpha_sigma), jnp.float64(priors.alpha_nu)
    dmax = jnp.float64(krylov_dmax)
    _deg = int(krylov_degree)
    _IC = -1  # reparam disabled (target unchanged); simpler/robust

    def _conjugate_normal(Ut, omega, working, V0, mu0, key, dim):
        """Draw β ~ N(Σ (Uᵀ working + V₀⁻¹μ₀), Σ), Σ⁻¹ = UᵀΩU + V₀⁻¹."""
        Uw = Ut * omega[:, None]
        Sig_inv = Uw.T @ Ut + jnp.diag(V0) + 1e-10 * jnp.eye(dim)
        rhs = Ut.T @ working + V0 * mu0
        L = jnp.linalg.cholesky(Sig_inv)
        m = cho_solve((L, True), rhs)
        zc = jax.random.normal(key, shape=(dim,), dtype=jnp.float64)
        return m + solve_triangular(L.T, zc, lower=False)

    @jax.jit
    def gibbs_step(state, key, slice_width):
        gamma = state["gamma"]
        lam = state["lam"]
        beta = state["beta"]
        rho = state["rho"]
        alpha = state["alpha"]
        z = state["z"]  # (n,) float 0/1 from the previous sweep
        (kβ, kγ, kλ, kρ, kα, kzs, kzc, kpg_s, kpg_c) = jax.random.split(key, 9)

        # ─────────── SELECTION (reduced-form logit, response z) ───────────
        eta_sel = _solve_sel(lam, Z_jax @ gamma)
        omega_sel = _draw_pg(jnp.ones(n), jnp.clip(eta_sel, -20.0, 20.0), kpg_s)

        V_sel = _build_krylov_basis_jax(
            lambda rhs: _solve_sel(lam, rhs), Z_jax, _matvec_Wsel, n, p, _deg
        )
        lam_new, _ = jax_slice_sample_1d(
            lambda lv: _rho_log_density_logit(
                lv, V_sel, lam, omega_sel, z, V0g, mu0g, _IC, dmax
            ),
            lam,
            lam_lo,
            lam_hi,
            key=kλ,
            w=slice_width,
        )
        Ztilde = _eval_U_from_basis_jax(V_sel, lam_new - lam)
        gamma = _conjugate_normal(Ztilde, omega_sel, z - 0.5, V0g, mu0g, kγ, p)
        lam = lam_new
        eta_sel = Ztilde @ gamma

        # ─────────────────── ZERO ALLOCATION (z draw) ────────────────────
        eta_cnt = _solve_cnt(rho, X_jax @ beta)
        pi = jax.nn.sigmoid(eta_sel)
        mu_cnt = jnp.exp(jnp.clip(eta_cnt, -30.0, 30.0))
        p_nb0 = jnp.power(alpha / (mu_cnt + alpha), alpha)
        p_z1_if0 = pi * p_nb0 / (pi * p_nb0 + (1.0 - pi) + 1e-300)
        prob = jnp.where(y_jax > 0, 1.0, p_z1_if0)
        z = (jax.random.uniform(kzc, shape=(n,), dtype=jnp.float64) < prob).astype(
            jnp.float64
        )
        z1 = z > 0.5

        # ─────────── COUNT (reduced-form NB, z-masked) ───────────
        h_cnt = jnp.where(z1, jnp.maximum(y_jax + alpha, 1e-3), 1.0)
        omega_cnt = _draw_pg(
            h_cnt, jnp.clip(eta_cnt - jnp.log(alpha), -20.0, 20.0), kpg_c
        )
        omega_cnt = jnp.where(z1, omega_cnt, 1e-300)  # mask structural zeros
        y_for = jnp.where(z1, y_jax, alpha)  # κ = 0.5(y−α) = 0 where z=0

        V_cnt = _build_krylov_basis_jax(
            lambda rhs: _solve_cnt(rho, rhs), X_jax, _matvec_Wcnt, n, k, _deg
        )
        rho_new, _ = jax_slice_sample_1d(
            lambda rv: _rho_log_density_marginal_jax(
                rv, V_cnt, rho, omega_cnt, y_for, alpha, V0b, mu0b, _IC, dmax
            ),
            rho,
            rho_lo,
            rho_hi,
            key=kρ,
            w=slice_width,
        )
        Xtilde = _eval_U_from_basis_jax(V_cnt, rho_new - rho)
        log_alpha = jnp.log(alpha)
        working_cnt = 0.5 * (y_for - alpha) + omega_cnt * log_alpha
        beta = _conjugate_normal(Xtilde, omega_cnt, working_cnt, V0b, mu0b, kβ, k)
        rho = rho_new
        eta_cnt = Xtilde @ beta

        # α | y, η_cnt, z  — slice on log α, z-masked NB log-likelihood.
        from jax.scipy.special import gammaln as _gammaln

        def _alpha_logdens(log_a):
            a = jnp.exp(log_a)
            mu = jnp.exp(jnp.clip(eta_cnt, -30.0, 30.0))
            ll = (
                _gammaln(y_jax + a)
                - _gammaln(a)
                + y_jax * jnp.log(jnp.maximum(mu / (mu + a), 1e-300))
                + a * jnp.log(jnp.maximum(a / (mu + a), 1e-300))
            )
            total = jnp.sum(z * ll)  # only z=1 obs contribute
            log_prior = (
                -0.5 * (a_nu + 1.0) * jnp.log1p((a * a) / (a_nu * a_sigma * a_sigma))
            )
            return log_a + total + log_prior

        log_a_new, _ = jax_slice_sample_1d(
            _alpha_logdens,
            jnp.log(alpha),
            jnp.float64(-4.0),
            jnp.float64(4.0),
            key=kα,
            w=jnp.float64(1.0),
        )
        alpha = jnp.exp(log_a_new)

        new_state = {
            "gamma": gamma,
            "lam": lam,
            "beta": beta,
            "rho": rho,
            "alpha": alpha,
            "z": z,
        }
        return new_state, (lam, gamma, rho, beta, alpha, eta_sel, eta_cnt)

    return gibbs_step


def run_chains_jax_zinb(
    y,
    d,
    Z,
    X,
    W_sel_sparse,
    W_cnt_sparse,
    priors,
    inits,
    draws,
    tune,
    *,
    thin=1,
    krylov_degree=_KRYLOV_DEGREE_DEFAULT,
    krylov_dmax=_KRYLOV_DMAX_DEFAULT,
    slice_width=0.4,
    jax_seeds=None,
    progressbar=False,
):
    """Run the reduced-form ZINB PG-Gibbs sampler (device-parallel).

    Returns one dict per chain with keys ``lam``, ``gamma``, ``rho``, ``beta``,
    ``alpha``, ``log_lik``, ``pi_mean``.
    """
    import jax
    import jax.numpy as jnp
    from scipy.special import gammaln

    from bayespecon._jax_dispatch import ensure_x64

    from ..negbin_reduced._jax import _build_sparse_ctx, _run_chains_device_parallel

    ensure_x64()
    chains = len(inits)
    n, k = X.shape
    p = Z.shape[1]
    y_jax = jnp.asarray(y, dtype=jnp.float64)
    d_jax = jnp.asarray(d, dtype=jnp.float64)
    Z_jax = jnp.asarray(Z, dtype=jnp.float64)
    X_jax = jnp.asarray(X, dtype=jnp.float64)
    sel_ctx = _build_sparse_ctx(W_sel_sparse, n)
    cnt_ctx = _build_sparse_ctx(W_cnt_sparse, n)

    import cholgraph

    # two patterns (W_sel, W_cnt) x chains x a few distinct rho/lam per sweep
    cholgraph.set_lu_cache_size(max(64, 12 * chains))

    slice_width_jax = jnp.float64(slice_width)
    if jax_seeds is None:
        jax_seeds = list(range(chains))

    gibbs_step = _make_zinb_gibbs_step(
        y_jax,
        d_jax,
        Z_jax,
        X_jax,
        sel_ctx,
        cnt_ctx,
        n,
        p,
        k,
        priors,
        krylov_degree=krylov_degree,
        krylov_dmax=krylov_dmax,
    )

    state0 = {
        "gamma": jnp.asarray(np.stack([i.gamma for i in inits]), dtype=jnp.float64),
        "lam": jnp.asarray([float(i.lam) for i in inits], dtype=jnp.float64),
        "beta": jnp.asarray(np.stack([i.beta for i in inits]), dtype=jnp.float64),
        "rho": jnp.asarray([float(i.rho) for i in inits], dtype=jnp.float64),
        "alpha": jnp.asarray([float(i.alpha) for i in inits], dtype=jnp.float64),
        "z": jnp.asarray(np.stack([np.asarray(i.z, dtype=np.float64) for i in inits])),
    }
    warm_keys = jnp.stack([jax.random.PRNGKey(int(s)) for s in jax_seeds])
    draw_keys = jnp.stack(
        [jax.random.fold_in(jax.random.PRNGKey(int(s)), 1) for s in jax_seeds]
    )

    def _warm_one(s, key):
        def body(_, carry):
            st, kk = carry
            kk, sk = jax.random.split(kk)
            st, _ = gibbs_step(st, sk, slice_width_jax)
            return (st, kk)

        st, _ = jax.lax.fori_loop(0, tune, body, (s, key))
        return st

    def _draw_one(s, key):
        def body(carry, _):
            st, kk = carry
            kk, sk = jax.random.split(kk)
            st, tr = gibbs_step(st, sk, slice_width_jax)
            return (st, kk), tr

        _, traces = jax.lax.scan(body, (s, key), None, length=draws)
        return traces

    lam_all, gamma_all, rho_all, beta_all, alpha_all, etasel_all, etacnt_all = (
        _run_chains_device_parallel(
            _warm_one, _draw_one, state0, warm_keys, draw_keys, chains, tune
        )
    )
    lam_all = np.asarray(lam_all)
    gamma_all = np.asarray(gamma_all)
    rho_all = np.asarray(rho_all)
    beta_all = np.asarray(beta_all)
    alpha_all = np.asarray(alpha_all)
    etasel_all = np.asarray(etasel_all)
    etacnt_all = np.asarray(etacnt_all)

    sl = slice(None, None, thin) if thin > 1 else slice(None)
    y_np = np.asarray(y, dtype=np.float64)
    d_np = np.asarray(d, dtype=np.float64)
    results = []
    for c in range(chains):
        eta_sel = etasel_all[c, sl]
        eta_cnt = etacnt_all[c, sl]
        a = alpha_all[c, sl][:, None]
        mu = np.exp(np.clip(eta_cnt, -30.0, 30.0))
        # z=1 obs weight in the NB log-lik: use the count draw's posterior weight
        # pi*pnb0/... — but for stored log-lik we follow the numpy path: NB where
        # activation is likely + logit for all.
        pi = 1.0 / (1.0 + np.exp(-eta_sel))
        ll_nb = (
            gammaln(y_np + a)
            - gammaln(a)
            + y_np * np.log(np.maximum(mu / (mu + a), 1e-300))
            + a * np.log(np.maximum(a / (mu + a), 1e-300))
        )
        ll_sel = d_np * np.log(np.maximum(pi, 1e-300)) + (1 - d_np) * np.log(
            np.maximum(1 - pi, 1e-300)
        )
        results.append(
            {
                "lam": lam_all[c, sl],
                "gamma": gamma_all[c, sl],
                "rho": rho_all[c, sl],
                "beta": beta_all[c, sl],
                "alpha": alpha_all[c, sl],
                "log_lik": np.where(y_np > 0, ll_nb, 0.0) + ll_sel,
                "pi_mean": pi.mean(axis=1),
            }
        )
    return results
