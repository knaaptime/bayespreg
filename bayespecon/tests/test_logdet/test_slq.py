"""Tests for Stochastic Lanczos Quadrature (SLQ) logdet."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from bayespecon._logdet import (
    SLQPrecompute,
    make_logdet_fn,
    make_logdet_numpy_fn,
    make_logdet_numpy_vec_fn,
    slq_logdet_eval,
    slq_logdet_eval_vec,
    slq_logdet_precompute,
)
from bayespecon._logdet._slq import _arnoldi_iteration, _batched_lanczos


def _toy_w(n: int = 50, seed: int = 0) -> sp.csr_matrix:
    """Small row-standardized sparse W for testing."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(size=(n, 2))
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    W = (1.0 / d) * (d < np.quantile(d, 0.2))
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    W = W / rs
    return sp.csr_matrix(W)


def _rook_W(side: int) -> sp.csr_matrix:
    """Row-standardized rook-contiguity lattice (undirected, symmetrizable)."""
    n = side * side
    A = sp.lil_matrix((n, n))
    for r in range(side):
        for c in range(side):
            i = r * side + c
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < side and 0 <= cc < side:
                    A[i, rr * side + cc] = 1.0
    A = A.tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    return (sp.diags(1.0 / deg) @ A).tocsr()


def _exact_logdet(rho: float, W: sp.csr_matrix) -> float:
    eigs = np.linalg.eigvals(W.toarray())
    return float(np.sum(np.log(np.abs(1.0 - rho * eigs))))


class TestSLQPrecompute:
    def test_returns_precompute_object(self):
        W = _toy_w(30)
        pre = slq_logdet_precompute(W, n_probes=5, lanczos_deg=10)
        assert isinstance(pre, SLQPrecompute)
        assert pre.n_probes == 5
        assert pre.lanczos_deg == 10
        assert pre.n == 30

    def test_nodes_and_weights_shapes(self):
        W = _toy_w(30)
        pre = slq_logdet_precompute(W, n_probes=5, lanczos_deg=10)
        assert pre.nodes.shape == (5, 10)
        assert pre.weights.shape == (5, 10)


class TestSLQEval:
    def test_eval_returns_float(self):
        W = _toy_w(30)
        pre = slq_logdet_precompute(W, n_probes=10, lanczos_deg=20)
        val = slq_logdet_eval(pre, 0.3)
        assert isinstance(val, float)
        assert np.isfinite(val)

    def test_eval_at_zero_is_zero(self):
        W = _toy_w(30)
        pre = slq_logdet_precompute(W, n_probes=5, lanczos_deg=10)
        assert abs(slq_logdet_eval(pre, 0.0)) < 1e-10

    def test_eval_vec_matches_scalar(self):
        W = _toy_w(30)
        pre = slq_logdet_precompute(W, n_probes=5, lanczos_deg=15)
        rhos = np.array([0.1, 0.3, 0.5])
        vec_result = slq_logdet_eval_vec(pre, rhos)
        scalar_results = np.array([slq_logdet_eval(pre, r) for r in rhos])
        np.testing.assert_allclose(vec_result, scalar_results, rtol=1e-10)

    def test_eval_vec_shape(self):
        W = _toy_w(30)
        pre = slq_logdet_precompute(W, n_probes=5, lanczos_deg=10)
        rhos = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        result = slq_logdet_eval_vec(pre, rhos)
        assert result.shape == (5,)


class TestSLQFactoryIntegration:
    def test_make_logdet_numpy_fn_slq(self):
        W = _toy_w(30)
        fn = make_logdet_numpy_fn(W, eigs=None, method="slq")
        val = fn(0.3)
        assert np.isfinite(val)

    def test_make_logdet_numpy_vec_fn_slq(self):
        W = _toy_w(30)
        fn = make_logdet_numpy_vec_fn(W, eigs=None, method="slq")
        rhos = np.array([0.1, 0.3, 0.5])
        result = fn(rhos)
        assert result.shape == (3,)

    def test_make_logdet_fn_slq_pytensor(self):
        import pytensor

        W = _toy_w(30)
        fn = make_logdet_fn(W, method="slq")
        import pytensor.tensor as pt

        rho_t = pt.dscalar()
        expr = fn(rho_t)
        f = pytensor.function([rho_t], expr)
        val = f(0.3)
        assert np.isfinite(val)

    def test_auto_select_slq_for_large_n(self):
        from bayespecon._logdet import resolve_logdet_method

        # SLQ is opt-in, not auto-selected
        assert resolve_logdet_method(None, n=200000) == "cheb_stochastic"
        assert resolve_logdet_method("slq", n=10000) == "slq"


class TestSLQQuadratureMath:
    """Quadrature correctness, isolated from Monte-Carlo probe noise.

    These pin the math the accuracy of SLQ rests on: the ``n``-scaled Lanczos
    weights form a Gauss rule exact to degree ``2k-1``, and the Arnoldi weights
    use the *bilinear* (biorthogonal) rule required for the non-normal
    Hessenberg — not the symmetric-case ``|V[0, :]|²`` it replaced.
    """

    def test_lanczos_weights_are_n_scaled(self):
        """Σᵢ weights = n per probe (canonical n-scaling, not ‖z‖²)."""
        rng = np.random.default_rng(0)
        n, k = 30, 8
        M = rng.standard_normal((n, n))
        S = (M + M.T) / 2
        Z = rng.standard_normal((n, 4))
        _, weights, _ = _batched_lanczos(lambda Q: S @ Q, n, k, Z, 4)
        np.testing.assert_allclose(weights.sum(axis=1), n, rtol=1e-10)

    def test_gauss_quadrature_exact_to_degree_2k_minus_1(self):
        """Σᵢ v₁ᵢ² θᵢᵖ = q₁ᵀSᵖq₁ for p ≤ 2k-1 (Gauss exactness)."""
        rng = np.random.default_rng(1)
        n, k = 20, 6
        M = rng.standard_normal((n, n))
        S = (M + M.T) / 2
        z = rng.standard_normal(n)
        q = z / np.linalg.norm(z)
        nodes, weights, _ = _batched_lanczos(lambda Q: S @ Q, n, k, z[:, None], 1)
        v1sq = weights[0] / n  # undo the n-scaling → v₁ᵢ²
        Sp = np.eye(n)
        for p in range(2 * k):  # p = 0 .. 2k-1
            quad = float(np.sum(v1sq * nodes[0] ** p))
            truth = float(q @ Sp @ q)
            assert abs(quad - truth) <= 1e-8 * (1 + abs(truth)), f"p={p}"
            Sp = Sp @ S

    def test_arnoldi_bilinear_matches_bilinear_form(self):
        """Full-Krylov Arnoldi: Σᵢ γᵢ f(θᵢ) = q₁ᵀf(A)q₁ for non-normal A.

        The old ``|V[0, :]|²`` rule is wrong here because A's eigenvectors are
        not orthogonal; the biorthogonal γ = V[0,:]·(V⁻¹e₁) is required.
        """
        rng = np.random.default_rng(2)
        n = 12
        A = rng.standard_normal((n, n)) * 0.1  # generic non-symmetric
        z = rng.standard_normal(n)
        q = z / np.linalg.norm(z)
        theta, gamma = _arnoldi_iteration(sp.csr_matrix(A), n, n, z)

        def f(x):
            return np.log(1.0 - 0.5 * x)

        est = np.sum(gamma * f(theta))
        w, V = np.linalg.eig(A)
        fA = V @ np.diag(f(w)) @ np.linalg.inv(V)
        truth = q @ fA @ q
        assert abs(est - truth) < 1e-10

    def test_arnoldi_weights_are_complex_bilinear(self):
        """Arnoldi weights sum to n (n·Σγ = n since Σγ = e₁ᵀe₁ = 1)."""
        rng = np.random.default_rng(3)
        # Directed lattice → asymmetric sparsity → Arnoldi fallback.
        n = 40
        rows, cols = [], []
        for i in range(n):
            rows += [i, i]
            cols += [(i + 1) % n, (i + 2) % n]
        A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        deg = np.asarray(A.sum(axis=1)).ravel()
        W = (sp.diags(1.0 / deg) @ A).tocsr()
        pre = slq_logdet_precompute(W, n_probes=5, lanczos_deg=15)
        assert pre.method == "arnoldi"
        assert np.iscomplexobj(pre.weights)
        np.testing.assert_allclose(pre.weights.sum(axis=1).real, n, rtol=1e-8)


class TestSLQEndToEndAccuracy:
    """SLQ recovers the logdet, though less accurately than cheb_stochastic.

    SLQ estimates the full log-integral per probe and lacks the exact-moment
    control variates ``cheb_stochastic`` uses, so at equal probe counts it is
    several-fold less accurate on flat spatial spectra — an inherent limit of
    the method, not a weight bug.  These tests pin *recovery* (small relative
    error), not competitiveness.
    """

    def test_recovers_logdet_within_relative_tolerance(self):
        W = _rook_W(30)  # n = 900
        for rho in (0.5, 0.9):
            exact = _exact_logdet(rho, W)
            pre = slq_logdet_precompute(
                W, n_probes=50, lanczos_deg=30, rng=np.random.default_rng(0)
            )
            est = slq_logdet_eval(pre, rho)
            rel = abs(est - exact) / abs(exact)
            assert rel < 0.06, f"rho={rho}: rel_err={rel:.4f}"

    def test_directed_arnoldi_recovers_logdet(self):
        """The Arnoldi path recovers a finite logdet on a directed graph."""
        n = 60
        rows, cols = [], []
        for i in range(n):
            rows += [i, i]
            cols += [(i + 1) % n, (i + 2) % n]
        A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        deg = np.asarray(A.sum(axis=1)).ravel()
        W = (sp.diags(1.0 / deg) @ A).tocsr()
        pre = slq_logdet_precompute(W, n_probes=20, lanczos_deg=20)
        est = slq_logdet_eval(pre, 0.3)
        exact = _exact_logdet(0.3, W)
        assert np.isfinite(est)
        # Directed Arnoldi is weak, but should be in the right ballpark.
        assert abs(est - exact) < 0.5 * abs(exact) + 1.0


class TestSLQExactMoments:
    """The exact-moment control variate (``n_exact``).

    The Gauss rule already implies estimates of ``tr(Wʲ)``; replacing them with
    the exact traces removes their sampling error at no extra matrix-vector
    products.  These tests pin the three properties that make that safe: the
    correction is off when asked to be, it is applied identically by the scalar
    and vectorized evaluators, and it reduces error rather than merely changing
    it.
    """

    def _rook(self, side=30):
        import numpy as np
        import scipy.sparse as sp

        n = side * side
        rows, cols = [], []
        for i in range(side):
            for j in range(side):
                k = i * side + j
                if i + 1 < side:
                    rows += [k, (i + 1) * side + j]
                    cols += [(i + 1) * side + j, k]
                if j + 1 < side:
                    rows += [k, i * side + j + 1]
                    cols += [i * side + j + 1, k]
        A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        deg = np.asarray(A.sum(axis=1)).ravel()
        return sp.diags(1.0 / deg) @ A

    def test_n_exact_zero_disables_the_correction(self):
        import numpy as np

        from bayespecon._logdet._slq import slq_logdet_precompute

        W = self._rook()
        pre = slq_logdet_precompute(
            W, n_probes=8, lanczos_deg=15, n_exact=0, rng=np.random.default_rng(0)
        )
        assert pre.cv_coeffs is None
        assert pre.n_exact == 0
        assert pre._cv_correction(0.5) == 0.0

    def test_scalar_and_vector_evaluators_agree_under_correction(self):
        import numpy as np

        from bayespecon._logdet._slq import (
            slq_logdet_eval,
            slq_logdet_eval_vec,
            slq_logdet_precompute,
        )

        W = self._rook()
        pre = slq_logdet_precompute(
            W, n_probes=8, lanczos_deg=15, n_exact=4, rng=np.random.default_rng(0)
        )
        assert pre.n_exact == 4
        rhos = np.linspace(-0.9, 0.9, 11)
        vec = slq_logdet_eval_vec(pre, rhos)
        scalar = np.array([slq_logdet_eval(pre, float(r)) for r in rhos])
        assert np.allclose(vec, scalar, rtol=1e-12, atol=1e-10)

    def test_correction_vanishes_at_rho_zero(self):
        """``log|I - 0·W| = 0`` exactly, so the correction must not perturb it."""
        import numpy as np

        from bayespecon._logdet._slq import slq_logdet_eval, slq_logdet_precompute

        W = self._rook()
        pre = slq_logdet_precompute(
            W, n_probes=8, lanczos_deg=15, n_exact=4, rng=np.random.default_rng(0)
        )
        assert abs(slq_logdet_eval(pre, 0.0)) < 1e-9

    def test_correction_reduces_error_against_the_exact_logdet(self):
        import numpy as np
        import scipy.sparse as sp

        from bayespecon._logdet._slq import (
            slq_logdet_eval_vec,
            slq_logdet_precompute,
        )

        W = self._rook()
        n = W.shape[0]
        eye = sp.eye(n, format="csc")
        rhos = np.linspace(-0.9, 0.9, 9)
        exact = np.array([np.linalg.slogdet((eye - r * W).toarray())[1] for r in rhos])

        def err(depth, seed):
            pre = slq_logdet_precompute(
                W,
                n_probes=30,
                lanczos_deg=15,
                n_exact=depth,
                rng=np.random.default_rng(seed),
            )
            return np.max(np.abs(slq_logdet_eval_vec(pre, rhos) - exact))

        plain = np.median([err(0, s) for s in range(5)])
        corrected = np.median([err(4, s) for s in range(5)])
        assert corrected < plain, f"CV made it worse: {corrected} vs {plain}"
