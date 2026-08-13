"""Tests for fill-in estimation in :mod:`bayespecon._logdet._fillin`."""

import numpy as np
import pytest
import scipy.sparse as sp

from bayespecon._logdet._fillin import estimate_fillin_ratio


def _rook_2d(n_side: int) -> sp.csr_matrix:
    """Build a 2D rook adjacency (degree ≤ 4) on an n_side × n_side grid."""
    n = n_side * n_side
    rows, cols = [], []
    for i in range(n_side):
        for j in range(n_side):
            idx = i * n_side + j
            if i > 0:
                rows.append(idx)
                cols.append(idx - n_side)
            if i < n_side - 1:
                rows.append(idx)
                cols.append(idx + n_side)
            if j > 0:
                rows.append(idx)
                cols.append(idx - 1)
            if j < n_side - 1:
                rows.append(idx)
                cols.append(idx + 1)
    data = np.ones(len(rows))
    W = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    return W + W.T  # symmetrize


def _star_graph(n: int) -> sp.csr_matrix:
    """Build a star graph: node 0 connected to all others (degree n-1)."""
    rows = list(range(1, n)) + [0] * (n - 1)
    cols = [0] * (n - 1) + list(range(1, n))
    data = np.ones(2 * (n - 1))
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def _knn_graph(n: int, k: int, seed: int = 42) -> sp.csr_matrix:
    """Build an approximate k-NN graph with symmetric edges."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 2))
    # Dense distance matrix — fine for small n
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
    W = np.zeros((n, n))
    for i in range(n):
        nearest = np.argsort(d[i])[: k + 1]  # include self
        for j in nearest:
            if j != i:
                W[i, j] = 1.0
    W = np.maximum(W, W.T)  # symmetrize
    return sp.csr_matrix(W)


class TestEstimateFillinRatio:
    """Tests for ``estimate_fillin_ratio``."""

    def test_rook_lattice_low_ratio(self):
        """2D rook (degree ≤ 4) should give a modest ratio (~4)."""
        W = _rook_2d(10)  # 100 nodes, degree ≤ 4
        ratio = estimate_fillin_ratio(W)
        # Interior nodes have degree 4; corner/edge nodes less.
        # Upper bound ratio ≈ mean degree ≈ 4 (minus boundary effects).
        assert ratio < 10
        assert ratio > 2

    def test_star_graph_high_ratio(self):
        """Star graph (hub degree n-1) should give a high ratio."""
        n = 100
        W = _star_graph(n)
        ratio = estimate_fillin_ratio(W)
        # Hub has degree 99, each neighbor has degree 2 (hub + themselves via sym).
        # Actually each leaf has degree 1 (connected to hub only).
        # Upper bound: for hub row, sum of 99 neighbors each degree 1 = 99.
        # For leaf rows, 1 neighbor (hub) with degree 99 = 99 each, × 99 leaves.
        # Total est = 99 + 99*99 = 9900. nnz = 198. ratio ≈ 50.
        assert ratio > 20

    def test_empty_matrix_returns_inf(self):
        """An all-zeros matrix should return inf."""
        W = sp.csr_matrix((5, 5))
        assert estimate_fillin_ratio(W) == float("inf")

    def test_single_edge(self):
        """A single edge (2×2) should give ratio 1.0 (W² has same structure)."""
        W = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=float))
        ratio = estimate_fillin_ratio(W)
        # Each node has degree 1, neighbor has degree 1.
        # est = 1 + 1 = 2, nnz = 2, ratio = 1.0
        assert ratio == pytest.approx(1.0)

    def test_dense_matrix_high_ratio(self):
        """A dense matrix should give ratio ≈ n."""
        n = 20
        W_dense = np.ones((n, n)) - np.eye(n)
        W = sp.csr_matrix(W_dense)
        ratio = estimate_fillin_ratio(W)
        # Each node has degree n-1, each neighbor has degree n-1.
        # est = n * (n-1) * (n-1), nnz = n*(n-1), ratio = n-1.
        assert ratio == pytest.approx(n - 1)

    def test_knn_graph_ratio_scales_with_k(self):
        """KNN-k graph should give ratio proportional to k."""
        n = 200
        W_k5 = _knn_graph(n, k=5)
        W_k20 = _knn_graph(n, k=20)
        ratio_5 = estimate_fillin_ratio(W_k5)
        ratio_20 = estimate_fillin_ratio(W_k20)
        # KNN-20 should have significantly higher fill-in than KNN-5
        assert ratio_20 > ratio_5
        # KNN-5 should be below the default threshold of 20
        assert ratio_5 < 20
        # KNN-20 should be above (each node has ~20 neighbors each with ~20)
        assert ratio_20 > 15

    def test_matches_actual_w2_for_rook(self):
        """The estimate should upper-bound actual nnz(W²) for a rook lattice."""
        W = _rook_2d(8)  # 64 nodes
        W2 = W @ W
        ratio_est = estimate_fillin_ratio(W)
        actual_ratio = W2.nnz / W.nnz
        # Upper bound property
        assert ratio_est >= actual_ratio
        # Should be reasonably close for a regular graph
        assert ratio_est < 2 * actual_ratio

    def test_pure_python_fallback_matches(self):
        """The pure-Python path should give the same result as Numba (if available)."""
        from bayespecon._logdet._fillin import _estimate_w2_nnz

        if _estimate_w2_nnz is None:
            pytest.skip("Numba not available")

        W = _rook_2d(10)
        n = W.shape[0]
        indptr = W.indptr.astype(np.int64, copy=False)
        indices = W.indices.astype(np.int64, copy=False)

        # Numba path
        est_numba = int(_estimate_w2_nnz(indptr, indices, n))

        # Pure-Python path
        row_degrees = np.diff(indptr)
        est_python = int(row_degrees[indices].sum())

        assert est_numba == est_python
