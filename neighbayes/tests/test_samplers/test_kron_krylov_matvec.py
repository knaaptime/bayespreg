"""Correctness of the separable-flow Kronecker Krylov basis.

The basis feeds the beta-marginalized rho conditional in the separable NB flow
sampler.  A wrong matvec does not raise — it silently distorts that conditional,
which showed up only as bad mixing (ESS ~3, rhat ~1.8) and a biased rho_o.
These tests pin the operators against explicit Kronecker products.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from neighbayes._ops import kron_solve_matrix
from neighbayes.samplers.negbin_reduced._flow import _build_kron_krylov_basis


@pytest.fixture
def kron_setup():
    rng = np.random.default_rng(0)
    n, k = 6, 3
    A0 = rng.random((n, n)) * (rng.random((n, n)) < 0.6) + 1e-3
    W = sp.csc_matrix(A0 / A0.sum(axis=1, keepdims=True))
    I = sp.eye(n, format="csr")
    rho_d, rho_o = 0.3, 0.2
    Ld = (I - rho_d * W).tocsr()
    Lo = (I - rho_o * W).tocsr()
    X = rng.normal(size=(n * n, k))
    return dict(n=n, W=W, Ld=Ld, Lo=Lo, X=X, rho_d=rho_d, rho_o=rho_o)


def test_solver_uses_kron_lo_ld(kron_setup):
    """Pins the convention the matvec derivations depend on."""
    import scipy.sparse.linalg as spla

    s = kron_setup
    got = kron_solve_matrix(s["Lo"], s["Ld"], s["X"], s["n"])
    want = spla.spsolve(sp.kron(s["Lo"], s["Ld"], format="csc"), s["X"])
    np.testing.assert_allclose(got, want, atol=1e-10)


@pytest.mark.parametrize("direction", ["rho_d", "rho_o"])
def test_basis_first_order_term_matches_explicit_kronecker(kron_setup, direction):
    r"""``V_1`` must equal :math:`A_c^{-1} (\partial A) A_c^{-1} X`.

    With ``A = L_o \otimes L_d``:
    ``dA/drho_d = -(L_o kron W)`` and ``dA/drho_o = -(W kron L_d)``.
    """
    s = kron_setup
    op = (
        sp.kron(s["Lo"], s["W"], format="csr")
        if direction == "rho_d"
        else sp.kron(s["W"], s["Ld"], format="csr")
    )
    basis = _build_kron_krylov_basis(
        s["rho_d"], s["rho_o"], s["X"], s["W"], s["n"], direction, 2
    )
    V0 = kron_solve_matrix(s["Lo"], s["Ld"], s["X"], s["n"])
    np.testing.assert_allclose(basis.V_stack[0], V0, atol=1e-10)
    want = kron_solve_matrix(s["Lo"], s["Ld"], op @ V0, s["n"])
    np.testing.assert_allclose(basis.V_stack[1], want, atol=1e-10)


@pytest.mark.parametrize("direction", ["rho_d", "rho_o"])
def test_basis_reproduces_exact_solve_near_centre(kron_setup, direction):
    """The truncated series must approximate A(rho)^-1 X for small drho."""
    from neighbayes.samplers.negbin_reduced._core import _eval_U_from_basis

    s = kron_setup
    n, I = s["n"], sp.eye(s["n"], format="csr")
    basis = _build_kron_krylov_basis(
        s["rho_d"], s["rho_o"], s["X"], s["W"], n, direction, 12
    )
    drho = 0.02
    if direction == "rho_d":
        rd, ro = s["rho_d"] + drho, s["rho_o"]
    else:
        rd, ro = s["rho_d"], s["rho_o"] + drho
    Ld = (I - rd * s["W"]).tocsr()
    Lo = (I - ro * s["W"]).tocsr()
    exact = kron_solve_matrix(Lo, Ld, s["X"], n)
    approx = _eval_U_from_basis(basis, drho)
    rel = np.abs(approx - exact).max() / np.abs(exact).max()
    assert rel < 1e-6, f"{direction}: relative error {rel:.2e}"


def test_directions_are_not_interchangeable(kron_setup):
    """Guards the original defect: one basis cannot serve both rho's."""
    s = kron_setup
    bd = _build_kron_krylov_basis(
        s["rho_d"], s["rho_o"], s["X"], s["W"], s["n"], "rho_d", 4
    )
    bo = _build_kron_krylov_basis(
        s["rho_d"], s["rho_o"], s["X"], s["W"], s["n"], "rho_o", 4
    )
    assert not np.allclose(bd.V_stack[1], bo.V_stack[1])
    assert bd.rho_basis == pytest.approx(s["rho_d"])
    assert bo.rho_basis == pytest.approx(s["rho_o"])


class TestFlowSystemInvertibility:
    """Spectral admissibility for the unrestricted 3-rho flow system.

    The previous ``|rho_d| + |rho_o| + |rho_w| < 1`` rule is *sufficient* for
    invertibility but not necessary, and it truncated the posterior short of
    the MLE on well-behaved data.
    """

    LAM_MIN, LAM_MAX = -0.529, 1.0

    @pytest.mark.parametrize(
        "rho,expected",
        [
            ((0.35, 0.25, 0.10), True),  # truth on the synthetic DGP
            ((0.487, 0.391, -0.122), True),  # profile MLE; old rule blocked it
            ((0.5, 0.4, -0.2), True),  # sum 1.1, min|eig| 0.300
            ((0.7, 0.6, -0.4), True),  # sum 1.7, min|eig| 0.100
            ((0.8, 0.7, -0.5), False),  # genuinely singular
            ((0.9, 0.9, -0.8), False),  # genuinely singular
            ((0.45, 0.45, 0.30), False),  # near-singular ridge, min|eig| 0.05
        ],
    )
    def test_admissibility(self, rho, expected):
        from neighbayes.samplers.negbin_reduced._flow import (
            flow_system_is_invertible,
        )

        assert flow_system_is_invertible(*rho, self.LAM_MIN, self.LAM_MAX) is expected

    def test_matches_true_spectrum_on_a_real_lattice(self):
        """Cross-check the corner test against explicit eigenvalues of A."""
        from neighbayes.dgp.flows import generate_poisson_flow_data
        from neighbayes.samplers.negbin_reduced._flow import (
            flow_system_is_invertible,
        )

        d = generate_poisson_flow_data(n=9, seed=3)
        n = d["W"].shape[0]
        Wd_ = np.asarray(d["W"].todense())
        lam = np.linalg.eigvals(Wd_).real
        lo, hi = lam.min(), lam.max()
        Wd = sp.kron(sp.eye(n), sp.csr_matrix(Wd_), format="csr")
        Wo = sp.kron(sp.csr_matrix(Wd_), sp.eye(n), format="csr")
        Ww = sp.kron(sp.csr_matrix(Wd_), sp.csr_matrix(Wd_), format="csr")
        I = sp.eye(n * n, format="csr")

        rng = np.random.default_rng(0)
        for _ in range(25):
            rd, ro, rw = rng.uniform(-0.6, 0.6, 3)
            ok = flow_system_is_invertible(rd, ro, rw, lo, hi)
            A = np.asarray((I - rd * Wd - ro * Wo - rw * Ww).todense())
            min_eig = float(np.abs(np.linalg.eigvals(A)).min())
            if ok:
                # Accepting a singular system is the failure that matters.
                assert min_eig > 1e-6, f"accepted near-singular {(rd, ro, rw)}"


class TestUnrestrictedKrylovBasis:
    """Shift-invert basis for the unrestricted 3-rho flow system.

    Without it the rho blocks refactorize A at every slice candidate
    (measured: 22.7 factorizations per sweep against 3 basis builds).
    """

    @staticmethod
    def _setup(n_units=9):
        from neighbayes.dgp.flows import generate_poisson_flow_data

        d = generate_poisson_flow_data(n=n_units, seed=3)
        n = d["W"].shape[0]
        W = sp.csr_matrix(d["W"])
        return dict(
            X=d["X"],
            n=n,
            N=n * n,
            Wd=sp.kron(sp.eye(n), W, format="csr"),
            Wo=sp.kron(W, sp.eye(n), format="csr"),
            Ww=sp.kron(W, W, format="csr"),
        )

    @pytest.mark.parametrize("direction", ["rho_d", "rho_o", "rho_w"])
    @pytest.mark.parametrize("drho", [0.0, 0.01, 0.05])
    def test_matches_exact_solve(self, direction, drho):
        from neighbayes.samplers.negbin_reduced._core import _eval_U_from_basis
        from neighbayes.samplers.negbin_reduced._flow import (
            _assemble_A_unrestricted,
            _solve_A_unrestricted,
            build_unrestricted_krylov_basis,
        )

        s = self._setup()
        centre = [0.30, 0.20, 0.10]
        basis = build_unrestricted_krylov_basis(
            *centre, s["X"], s["Wd"], s["Wo"], s["Ww"], direction, 12
        )
        slot = {"rho_d": 0, "rho_o": 1, "rho_w": 2}[direction]
        rho = list(centre)
        rho[slot] += drho
        A = _assemble_A_unrestricted(*rho, s["Wd"], s["Wo"], s["Ww"], s["N"])
        exact = _solve_A_unrestricted(A, s["X"], T=1)
        approx = _eval_U_from_basis(basis, drho)
        rel = np.abs(approx - exact).max() / np.abs(exact).max()
        assert rel < 1e-9, f"{direction} drho={drho}: rel err {rel:.2e}"

    def test_rho_basis_tracks_the_varied_parameter(self):
        from neighbayes.samplers.negbin_reduced._flow import (
            build_unrestricted_krylov_basis,
        )

        s = self._setup()
        centre = [0.30, 0.20, 0.10]
        for direction, want in zip(("rho_d", "rho_o", "rho_w"), centre):
            b = build_unrestricted_krylov_basis(
                *centre, s["X"], s["Wd"], s["Wo"], s["Ww"], direction, 4
            )
            assert b.rho_basis == pytest.approx(want)
            assert b.safe_dmax > 0

    def test_rejects_unknown_direction(self):
        from neighbayes.samplers.negbin_reduced._flow import (
            build_unrestricted_krylov_basis,
        )

        s = self._setup()
        with pytest.raises(ValueError, match="direction"):
            build_unrestricted_krylov_basis(
                0.3, 0.2, 0.1, s["X"], s["Wd"], s["Wo"], s["Ww"], "rho_x", 4
            )

    def test_directions_produce_different_bases(self):
        from neighbayes.samplers.negbin_reduced._flow import (
            build_unrestricted_krylov_basis,
        )

        s = self._setup()
        bs = [
            build_unrestricted_krylov_basis(
                0.3, 0.2, 0.1, s["X"], s["Wd"], s["Wo"], s["Ww"], d_, 4
            )
            for d_ in ("rho_d", "rho_o", "rho_w")
        ]
        for i in range(3):
            for j in range(i + 1, 3):
                assert not np.allclose(bs[i].V_stack[1], bs[j].V_stack[1])


# ---------------------------------------------------------------------------
# The JAX twin of everything above
# ---------------------------------------------------------------------------
#
# The tests above pin the NumPy separable-flow operators in
# ``negbin_reduced/_flow``.  Its JAX port in ``negbin_reduced/_flow_jax`` was
# never covered, and had drifted in three places at once — the Kronecker solve
# returned ``vec(Hᵀ)`` instead of ``vec(H)`` (a C-order reshape and a missing
# ``transpose(1, 0, 2)``), and both matvecs applied ``Wᵀ`` where ``W`` belongs,
# computing ``(L_oᵀ ⊗ W)`` and ``(L_d ⊗ Wᵀ)``.
#
# None of it raised.  ``SARNegBinFlowSeparable(gibbs_backend="jax")`` simply
# fit a permuted system and reported ρ_d = −0.21 against a true +0.35 with
# R̂ = 1.00 — converged, and wrong.  W must be **asymmetric** here: a
# symmetric W hides every transpose error.

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)


@pytest.fixture
def kron_setup_jax(kron_setup):
    """``kron_setup`` plus the JAX-side operands."""
    from jax.experimental import sparse as jsparse

    from neighbayes.samplers.negbin_reduced._flow_jax import _build_sar_solver_jax

    s = dict(kron_setup)
    W_csc = sp.csc_matrix(s["W"])
    assert (abs(W_csc - W_csc.T)).nnz > 0, "W must be asymmetric to expose transposes"
    s["W_bcoo"] = jsparse.BCOO.from_scipy_sparse(W_csc.tocsr())
    s["solve"] = _build_sar_solver_jax(W_csc, s["n"])
    s["X_jax"] = jnp.asarray(s["X"])
    return s


def _kron_solve_j(s, rho_d, rho_o, B):
    from neighbayes.samplers.negbin_reduced._flow_jax import _kron_solve_jax

    return _kron_solve_jax(
        lambda r: s["solve"](rho_d, r), lambda r: s["solve"](rho_o, r), B, s["n"]
    )


@pytest.mark.parametrize("k", [1, 2, 5])
def test_kron_solve_jax_matches_dense(kron_setup_jax, k):
    """``_kron_solve_jax`` must invert ``kron(Lo, Ld)`` — not its transpose.

    k = 1 is included deliberately: the shipped bug reproduced at k = 1, so a
    multi-column-only test would still have missed the reshape error.
    """
    s = kron_setup_jax
    n, N = s["n"], s["n"] ** 2
    rng = np.random.default_rng(1)
    B = rng.normal(size=(N, k))
    Ld_d, Lo_d = s["Ld"].toarray(), s["Lo"].toarray()

    got = np.asarray(_kron_solve_j(s, s["rho_d"], s["rho_o"], jnp.asarray(B)))
    want = np.linalg.solve(np.kron(Lo_d, Ld_d), B)
    np.testing.assert_allclose(got, want, atol=1e-10)


def test_kron_matvecs_jax_match_explicit_kronecker(kron_setup_jax):
    """Each matvec must be its own Kronecker product, and not the other's."""
    from neighbayes.samplers.negbin_reduced._flow_jax import (
        kron_matvec_d_jax,
        kron_matvec_o_jax,
    )

    s = kron_setup_jax
    n, k = s["n"], s["X"].shape[1]
    rd, ro = s["rho_d"], s["rho_o"]
    W_d, Ld_d, Lo_d = s["W"].toarray(), s["Ld"].toarray(), s["Lo"].toarray()
    rng = np.random.default_rng(2)
    v = jnp.asarray(rng.normal(size=(n * n, k)))

    want_d = np.kron(Lo_d, W_d) @ np.asarray(v)
    want_o = np.kron(W_d, Ld_d) @ np.asarray(v)
    got_d = np.asarray(kron_matvec_d_jax(s["W_bcoo"], ro, v, n, k))
    got_o = np.asarray(kron_matvec_o_jax(s["W_bcoo"], rd, v, n, k))

    np.testing.assert_allclose(got_d, want_d, atol=1e-10)
    np.testing.assert_allclose(got_o, want_o, atol=1e-10)
    # The historical failures were near-misses that swapped a transpose in.
    for bad, label in (
        (np.kron(Lo_d.T, W_d), "kron(Loᵀ, W)"),
        (np.kron(Ld_d, W_d.T), "kron(Ld, Wᵀ)"),
    ):
        assert not np.allclose(got_d, bad @ np.asarray(v)), f"matvec_d is {label}"
        assert not np.allclose(got_o, bad @ np.asarray(v)), f"matvec_o is {label}"


@pytest.mark.parametrize("direction", ["rho_d", "rho_o"])
@pytest.mark.parametrize("drho", [0.05, 0.10])
def test_kron_krylov_series_matches_dense(kron_setup_jax, direction, drho):
    """End-to-end pin: the Krylov series must reproduce the perturbed solve.

    This is the test to change these operators against.  It exercises the
    solve, the matvec and the Horner evaluation together, so it cannot be
    satisfied by two errors that cancel in one of them alone.
    """
    from neighbayes.samplers.negbin_reduced._flow_jax import (
        kron_matvec_d_jax,
        kron_matvec_o_jax,
    )

    s = kron_setup_jax
    n, k = s["n"], s["X"].shape[1]
    rd, ro = s["rho_d"], s["rho_o"]
    W_d, Ld_d, Lo_d = s["W"].toarray(), s["Ld"].toarray(), s["Lo"].toarray()
    I = np.eye(n)

    if direction == "rho_d":
        matvec = lambda V: kron_matvec_d_jax(s["W_bcoo"], ro, V, n, k)  # noqa: E731
        want = np.linalg.solve(np.kron(Lo_d, I - (rd + drho) * W_d), s["X"])
    else:
        matvec = lambda V: kron_matvec_o_jax(s["W_bcoo"], rd, V, n, k)  # noqa: E731
        want = np.linalg.solve(np.kron(I - (ro + drho) * W_d, Ld_d), s["X"])

    V_j = _kron_solve_j(s, rd, ro, s["X_jax"])
    got = np.asarray(V_j).copy()
    for j in range(1, 15):
        V_j = _kron_solve_j(s, rd, ro, matvec(V_j))
        got = got + (drho**j) * np.asarray(V_j)

    rel = np.abs(got - want).max() / np.abs(want).max()
    assert rel < 1e-8, f"{direction} Δρ={drho}: relative error {rel:.2e}"
