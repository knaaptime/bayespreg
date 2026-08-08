"""Tests + micro-benchmark for the Krylov precision basis (structural Gibbs).

Validates that the shift-invert Krylov basis for P(ρ) = base − ρG1 + ρ²G2
produces solve + logdet values matching the direct CHOLMOD path within
the Krylov radius, and that the per-candidate speedup is real.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from bayespecon.samplers._utils._spatial_normal import (
    CholmodFactor,
    KrylovPrecisionBasis,
    build_precision_krylov_basis,
    eval_precision_logdet_from_basis,
    eval_precision_solve_from_basis,
    lanczos_logdet,
)


def _make_precision_components(n, rho_c, omega, seed=0):
    """Build base, G1, G2 and a CHOLMOD factor for a ring-lattice W."""
    rng = np.random.default_rng(seed)
    W_dense = np.zeros((n, n))
    for i in range(n):
        W_dense[i, (i + 1) % n] = 1.0
        W_dense[i, (i - 1) % n] = 1.0
    row_sums = W_dense.sum(axis=1, keepdims=True)
    W_dense = W_dense / row_sums
    W = sp.csr_matrix(W_dense)
    W_sym = (W + W.T).tocsr()
    WtW = (W.T @ W).tocsr()
    base = (sp.eye(n, format="csr") + sp.diags(omega, format="csr")).tocsr()
    P0 = sp.eye(n, format="csr") + 0.5 * W_sym + 0.25 * WtW
    cholmod_factor = CholmodFactor(P0)
    return base, W_sym, WtW, cholmod_factor


def _direct_solve_logdet(rho, base, W_sym, WtW, rhs, cholmod_factor):
    """Direct CHOLMOD path: factor P(ρ), solve, read logdet."""
    P = (base - rho * W_sym + rho**2 * WtW).tocsc()
    cholmod_factor.factorize(P)
    sol = cholmod_factor.solve(rhs)
    logdet = cholmod_factor.logdet()
    return sol, logdet


class TestKrylovPrecisionBasisEquivalence:
    """Numerical equivalence within the Krylov radius."""

    def test_solve_matches_direct_within_radius(self):
        n, rho_c, degree = 200, 0.3, 12
        omega = np.ones(n) * 0.5
        base, W_sym, WtW, cholmod_factor = _make_precision_components(n, rho_c, omega)
        rng = np.random.default_rng(1)
        rhs = rng.standard_normal((n, 4))

        basis = build_precision_krylov_basis(
            rho_c,
            base,
            W_sym,
            WtW,
            rhs,
            degree=degree,
            cholmod_factor=cholmod_factor,
        )
        # Candidate within the radius
        for drho in [0.001, 0.01, 0.05, 0.1]:
            rho = rho_c + drho
            sol_kry = eval_precision_solve_from_basis(basis, drho)
            sol_dir, _ = _direct_solve_logdet(
                rho, base, W_sym, WtW, rhs, cholmod_factor
            )
            # Relative error should be small (geometric convergence).
            # The Neumann series converges as O((|drho|·‖P_c⁻¹G‖)^{m+1});
            # with degree=12 the truncation is very small for |drho|<0.1,
            # but the tolerance reflects practical MCMC accuracy needs.
            rel_err = np.linalg.norm(sol_kry - sol_dir) / np.linalg.norm(sol_dir)
            # Looser at larger drho — the slice sampler tolerates O(1e-2)
            # error in the log-density surface (MCMC noise dominates).
            assert rel_err < 1e-2, f"drho={drho}: rel_err={rel_err:.2e} > 1e-2"

    def test_logdet_matches_direct_within_radius(self):
        n, rho_c, degree = 200, 0.3, 12
        omega = np.ones(n) * 0.5
        base, W_sym, WtW, cholmod_factor = _make_precision_components(n, rho_c, omega)
        rng = np.random.default_rng(2)
        rhs = rng.standard_normal((n, 2))

        basis = build_precision_krylov_basis(
            rho_c,
            base,
            W_sym,
            WtW,
            rhs,
            degree=degree,
            cholmod_factor=cholmod_factor,
        )
        for drho in [0.001, 0.01, 0.05]:
            rho = rho_c + drho
            logdet_kry = eval_precision_logdet_from_basis(basis, drho, rng=rng)
            _, logdet_dir = _direct_solve_logdet(
                rho, base, W_sym, WtW, rhs, cholmod_factor
            )
            # First-order correction: relative error ~ O(drho^2)
            rel_err = abs(logdet_kry - logdet_dir) / abs(logdet_dir)
            assert rel_err < 0.05, f"drho={drho}: logdet rel_err={rel_err:.3e}"

    def test_basis_carries_solver_and_matvec(self):
        n, rho_c, degree = 100, 0.3, 8
        omega = np.ones(n) * 0.5
        base, W_sym, WtW, cholmod_factor = _make_precision_components(n, rho_c, omega)
        rhs = np.ones((n, 2))
        basis = build_precision_krylov_basis(
            rho_c,
            base,
            W_sym,
            WtW,
            rhs,
            degree=degree,
            cholmod_factor=cholmod_factor,
        )
        assert callable(basis.G_matvec)
        assert callable(basis.solve_at_c)
        # The cached solve_at_c should reproduce V_stack[0]
        v0_check = basis.solve_at_c(rhs)
        assert np.allclose(v0_check, basis.V_stack[0])


class TestKrylovPrecisionBasisSpeedup:
    """Micro-benchmark: per-candidate cost with vs without the basis.

    The Krylov reuse wins when CHOLMOD fill-in makes factorization
    expensive relative to the triangular solves — i.e. on 2-D lattices
    with queen contiguity, not on ring lattices (near-tridiagonal,
    minimal fill-in).  The crossover is around n≈400 on a queen grid.
    """

    def test_slice_step_speedup_queen_lattice(self):
        import libpysal

        n_side = 50  # n = 2500, queen contiguity (high fill-in)
        n = n_side * n_side
        rho_c, degree = 0.3, 12
        omega = np.ones(n) * 0.5
        W = libpysal.weights.lat2W(n_side, n_side, rook=False).sparse
        W = sp.csr_matrix(W / np.asarray(W.sum(1)).ravel())
        W_sym = (W + W.T).tocsr()
        WtW = (W.T @ W).tocsr()
        base = (sp.eye(n, format="csr") + sp.diags(omega, format="csr")).tocsr()
        P0 = sp.eye(n, format="csr") + 0.5 * W_sym + 0.25 * WtW
        cholmod_factor = CholmodFactor(P0)

        rng = np.random.default_rng(3)
        n_candidates = 8
        drhos = rng.uniform(-0.1, 0.1, size=n_candidates)
        rhs = rng.standard_normal((n, 5))

        # --- Direct path: factor + solve + logdet per candidate ---
        t0 = time.perf_counter()
        for drho in drhos:
            rho = rho_c + drho
            _direct_solve_logdet(rho, base, W_sym, WtW, rhs, cholmod_factor)
        t_direct = time.perf_counter() - t0

        # --- Krylov path: build once, Horner per candidate ---
        t0 = time.perf_counter()
        basis = build_precision_krylov_basis(
            rho_c,
            base,
            W_sym,
            WtW,
            rhs,
            degree=degree,
            cholmod_factor=cholmod_factor,
        )
        for drho in drhos:
            eval_precision_solve_from_basis(basis, drho)
            eval_precision_logdet_from_basis(basis, drho, rng=rng)
        t_krylov = time.perf_counter() - t0

        speedup = t_direct / t_krylov
        print(
            f"\n  queen n={n} candidates={n_candidates}: "
            f"direct={t_direct * 1e3:.1f}ms krylov={t_krylov * 1e3:.1f}ms "
            f"speedup={speedup:.2f}x"
        )
        assert speedup > 1.5, (
            f"Expected >1.5x speedup on queen lattice: "
            f"direct={t_direct * 1e3:.1f}ms krylov={t_krylov * 1e3:.1f}ms"
        )

    def test_no_slowdown_on_ring_lattice(self):
        """On a ring lattice (minimal fill-in), Krylov should not be >3x slower.

        The reuse is opt-in; on low-fill-in graphs CHOLMOD is already cheap
        and the basis build costs more than it saves.  This test guards
        against a catastrophic regression — the Krylov path must stay
        within a small factor of direct on the worst-case graph.
        """
        n, rho_c, degree = 2000, 0.3, 12
        omega = np.ones(n) * 0.5
        base, W_sym, WtW, cholmod_factor = _make_precision_components(n, rho_c, omega)
        rng = np.random.default_rng(4)
        n_candidates = 8
        drhos = rng.uniform(-0.1, 0.1, size=n_candidates)
        rhs = rng.standard_normal((n, 5))

        t0 = time.perf_counter()
        for drho in drhos:
            rho = rho_c + drho
            _direct_solve_logdet(rho, base, W_sym, WtW, rhs, cholmod_factor)
        t_direct = time.perf_counter() - t0

        t0 = time.perf_counter()
        basis = build_precision_krylov_basis(
            rho_c,
            base,
            W_sym,
            WtW,
            rhs,
            degree=degree,
            cholmod_factor=cholmod_factor,
        )
        for drho in drhos:
            eval_precision_solve_from_basis(basis, drho)
            eval_precision_logdet_from_basis(basis, drho, rng=rng)
        t_krylov = time.perf_counter() - t0

        slowdown = t_krylov / t_direct
        print(
            f"\n  ring n={n} candidates={n_candidates}: "
            f"direct={t_direct * 1e3:.1f}ms krylov={t_krylov * 1e3:.1f}ms "
            f"slowdown={slowdown:.2f}x"
        )
        # On a ring lattice the basis build (degree+1 solves) can cost more
        # than n_candidates cheap factorizations; allow up to 3x slowdown.
        assert slowdown < 3.0, f"Ring-lattice slowdown {slowdown:.2f}x exceeds 3x guard"


class TestJaxPathUnaffected:
    """The Krylov basis is NumPy-only; the JAX path must still work."""

    def test_jax_import_still_works(self):
        # Smoke test: importing the module should not pull JAX into the
        # Krylov path, and the existing JAX log-density core should still
        # be importable.
        from bayespecon.samplers._utils._spatial_normal import (
            _jax_log_density_core,
        )

        assert callable(_jax_log_density_core)
