"""Routing and equivalence tests for the (I − ρW) solvers.

Three routes back the same operator:

* ``_DSymCholSolver`` — Cholesky on the D-symmetrized similar matrix, used
  whenever ``W`` is D-symmetrizable (every row-standardized undirected graph).
* ``KluSarSolver`` — KLU on ``A`` itself, for genuinely directed ``W``.
* ``_CholmodNormalEqSolver`` — Cholesky on ``AᵀA``, the fallback.

They must agree to machine precision; only their cost differs.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from neighbayes.samplers._utils._spatial_normal import CholmodFactor
from neighbayes.samplers.negbin_reduced._core import (
    _CholmodNormalEqSolver,
    _DSymCholSolver,
    _make_cholmod_pattern,
    _symmetrizing_diagonal,
    make_sar_solver,
)


def _row_standardize(W):
    W = sp.csr_matrix(W)
    deg = np.asarray(W.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    return (sp.diags(1.0 / deg) @ W).tocsc()


def _ring(n):
    return _row_standardize(
        sp.diags([np.ones(n - 1), np.ones(n - 1)], [-1, 1], format="csr")
    )


def _queen(side):
    n = side * side
    rows, cols = [], []
    for i in range(side):
        for j in range(side):
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di or dj:
                        a, b = i + di, j + dj
                        if 0 <= a < side and 0 <= b < side:
                            rows.append(i * side + j)
                            cols.append(a * side + b)
    W = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    return _row_standardize(W)


def _directed(n, k=5, seed=0):
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(n), k)
    cols = (rows + rng.integers(1, 40, size=rows.size)) % n
    W = sp.coo_matrix((rng.random(rows.size), (rows, cols)), shape=(n, n)).tocsr()
    return _row_standardize(W)


def _parts(W):
    n = W.shape[0]
    W_sym, WtW, pattern = _make_cholmod_pattern(W, n)
    return CholmodFactor(pattern), W, W_sym, WtW, n


class TestRouting:
    def test_row_standardized_undirected_is_d_symmetrizable(self):
        """Row-standardizing breaks raw symmetry but preserves D-symmetrizability.

        This is why routing keys on the symmetrizing diagonal rather than on
        ``W == Wᵀ``: the latter is false for essentially every real ``W``.
        """
        for W in (_ring(200), _queen(12)):
            assert (abs(W - W.T) > 1e-12).nnz > 0, "expected raw asymmetry"
            assert _symmetrizing_diagonal(W) is not None

    def test_undirected_routes_to_cholesky(self):
        for W in (_ring(200), _queen(12)):
            solver = make_sar_solver(*_parts(W))
            assert isinstance(solver, _DSymCholSolver)

    def test_directed_routes_away_from_dsym(self):
        W = _directed(300)
        assert _symmetrizing_diagonal(W) is None
        solver = make_sar_solver(*_parts(W))
        assert not isinstance(solver, _DSymCholSolver)


class TestEquivalence:
    @pytest.mark.parametrize("rho", [0.0, 0.3, 0.7, -0.4])
    @pytest.mark.parametrize("builder", [_ring, _queen, _directed])
    def test_all_routes_agree(self, builder, rho):
        W = builder(14) if builder is _queen else builder(200)
        n = W.shape[0]
        rng = np.random.default_rng(5)
        B = rng.standard_normal((n, 3))
        A = (sp.eye(n) - rho * W).tocsc()

        forced = ["cholmod", "klu"]
        if _symmetrizing_diagonal(W) is not None:
            forced.append("dsym")

        for tag in forced:
            solver = make_sar_solver(*_parts(W), force=tag)
            solver.factorize(rho)
            x = solver.solve(B)
            assert np.abs(A @ x - B).max() < 1e-9, f"{tag} residual too large"
            # 1-D RHS must round-trip identically to the matching column.
            x1 = solver.solve(B[:, 0])
            assert np.allclose(x1, x[:, 0])

    def test_logdet_matches_dense(self):
        """KLU and D-sym both expose log det(I − ρW); they must agree with dense."""
        for builder, tag in ((_queen, "dsym"), (_directed, "klu")):
            W = builder(10) if builder is _queen else builder(120)
            n = W.shape[0]
            rho = 0.6
            solver = make_sar_solver(*_parts(W), force=tag)
            solver.factorize(rho)
            expected = np.linalg.slogdet((sp.eye(n) - rho * W).toarray())[1]
            assert abs(solver.logdet() - expected) < 1e-8

    def test_normal_equations_still_correct_for_directed(self):
        """The AᵀA fallback is correct for asymmetric W — just more expensive.

        Guards against the misreading that CHOLMOD cannot handle directed W:
        AᵀA is SPD for any non-singular A.
        """
        W = _directed(200)
        n = W.shape[0]
        rho = 0.5
        rng = np.random.default_rng(7)
        b = rng.standard_normal(n)
        solver = _CholmodNormalEqSolver(*_parts(W))
        solver.factorize(rho)
        x = solver.solve(b)
        assert np.abs((sp.eye(n) - rho * W) @ x - b).max() < 1e-9
