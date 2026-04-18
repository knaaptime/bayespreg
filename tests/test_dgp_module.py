"""API-level tests for the public bayespecon.dgp module."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from bayespecon import dgp
from tests.helpers import W_to_graph, make_rook_W


def test_dgp_exports_have_gdf_argument() -> None:
    """All exported simulator functions should accept a ``gdf`` argument."""
    funcs = [getattr(dgp, name) for name in dgp.__all__]
    for fn in funcs:
        sig = inspect.signature(fn)
        assert "gdf" in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        ), f"{fn.__name__} must accept gdf directly or through **kwargs"


def test_cross_sectional_generators_run_with_dense_W() -> None:
    """Cross-sectional DGPs should run and return consistent output keys."""
    rng = np.random.default_rng(123)
    W = make_rook_W(4)

    out_sar = dgp.simulate_sar(W=W, rng=rng)
    out_sem = dgp.simulate_sem(W=W, rng=rng)
    out_slx = dgp.simulate_slx(W=W, rng=rng)
    out_sdm = dgp.simulate_sdm(W=W, rng=rng)
    out_sdem = dgp.simulate_sdem(W=W, rng=rng)

    for out in (out_sar, out_sem, out_slx, out_sdm, out_sdem):
        assert set(("y", "X", "W_dense", "W_graph", "params_true")).issubset(out)
        assert out["y"].shape[0] == out["X"].shape[0]


def test_panel_generators_run_with_dense_W() -> None:
    """Panel DGPs should return time-first stacked outputs of length ``N*T``."""
    rng = np.random.default_rng(123)
    N, T = 6, 5
    W = make_rook_W(3)

    panel_outs = [
        dgp.simulate_panel_ols_fe(N=N, T=T, W=W, rng=rng),
        dgp.simulate_panel_sar_fe(N=N, T=T, W=W, rng=rng),
        dgp.simulate_panel_sem_fe(N=N, T=T, W=W, rng=rng),
        dgp.simulate_panel_sdm_fe(N=N, T=T, W=W, rng=rng),
        dgp.simulate_panel_sdem_fe(N=N, T=T, W=W, rng=rng),
        dgp.simulate_panel_ols_re(N=N, T=T, W=W, rng=rng),
        dgp.simulate_panel_sar_re(N=N, T=T, W=W, rng=rng),
        dgp.simulate_panel_sem_re(N=N, T=T, W=W, rng=rng),
        dgp.simulate_panel_dlm_fe(N=N, T=T, W=W, rng=rng),
        dgp.simulate_panel_sdmr_fe(N=N, T=T, W=W, rng=rng),
        dgp.simulate_panel_sdmu_fe(N=N, T=T, W=W, rng=rng),
        dgp.simulate_panel_sar_tobit_fe(N=N, T=T, W=W, rng=rng),
        dgp.simulate_panel_sem_tobit_fe(N=N, T=T, W=W, rng=rng),
    ]

    for out in panel_outs:
        assert out["y"].shape[0] == N * T
        assert out["X"].shape[0] == N * T
        assert out["unit"].shape[0] == N * T
        assert out["time"].shape[0] == N * T


def test_nonlinear_generators_run_with_dense_W() -> None:
    """Nonlinear DGPs should run and return expected keys and shapes."""
    rng = np.random.default_rng(123)
    W = make_rook_W(3)

    out_sar_tobit = dgp.simulate_sar_tobit(W=W, rng=rng)
    out_sem_tobit = dgp.simulate_sem_tobit(W=W, rng=rng)
    out_sdm_tobit = dgp.simulate_sdm_tobit(W=W, rng=rng)
    out_probit = dgp.simulate_spatial_probit(W=W, rng=rng)

    for out in (out_sar_tobit, out_sem_tobit, out_sdm_tobit):
        assert "y_latent" in out and "censored_mask" in out
        assert out["y"].shape == out["y_latent"].shape
        assert out["censored_mask"].shape == out["y"].shape

    assert out_probit["y"].shape[0] == out_probit["X"].shape[0]
    assert out_probit["region_ids"].shape[0] == out_probit["y"].shape[0]


def test_gdf_input_path_for_representative_generators() -> None:
    """Representative simulators should accept GeoDataFrame input.

    This test uses a small regular grid and checks one cross-sectional, one
    panel, and one nonlinear generator.
    """
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    rng = np.random.default_rng(123)
    polys = [box(j, i, j + 1, i + 1) for i in range(3) for j in range(3)]
    gdf = gpd.GeoDataFrame({"id": range(9)}, geometry=polys)

    out_sar = dgp.simulate_sar(gdf=gdf, rng=rng)
    out_panel = dgp.simulate_panel_sar_fe(N=9, T=3, gdf=gdf, rng=rng)
    out_probit = dgp.simulate_spatial_probit(gdf=gdf, rng=rng)

    assert out_sar["W_dense"].shape == (9, 9)
    assert out_panel["y"].shape[0] == 27
    assert out_probit["region_ids"].max() == 8


def test_both_gdf_and_graph_inputs_are_supported() -> None:
    """Providing both gdf and an associated Graph should be supported."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    rng = np.random.default_rng(123)
    polys = [box(j, i, j + 1, i + 1) for i in range(3) for j in range(3)]
    gdf = gpd.GeoDataFrame({"id": range(9)}, geometry=polys)

    W_dense = make_rook_W(3)
    W_graph = W_to_graph(W_dense)

    out = dgp.simulate_sar(W=W_graph, gdf=gdf, rng=rng)
    assert out["W_dense"].shape == (9, 9)


def test_both_gdf_and_graph_inputs_must_match_dimensions() -> None:
    """Passing incompatible W and gdf should raise a clear error."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    polys = [box(j, i, j + 1, i + 1) for i in range(3) for j in range(3)]
    gdf = gpd.GeoDataFrame({"id": range(9)}, geometry=polys)

    W_dense = make_rook_W(2)
    W_graph = W_to_graph(W_dense)

    with pytest.raises(ValueError, match="same number of spatial units"):
        dgp.simulate_sar(W=W_graph, gdf=gdf, seed=123)
