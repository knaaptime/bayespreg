"""Shared utilities for DGP simulation in bayespecon."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from libpysal.graph import Graph


def ensure_rng(rng: np.random.Generator | None = None, seed: int | None = None) -> np.random.Generator:
    """Return a reproducible NumPy random generator.

    Parameters
    ----------
    rng : np.random.Generator, optional
        Existing generator. If provided it takes precedence over ``seed``.
    seed : int, optional
        Seed for constructing a new generator when ``rng`` is not supplied.

    Returns
    -------
    np.random.Generator
        Random number generator used by simulation functions.
    """
    if rng is not None:
        return rng
    return np.random.default_rng(seed)


def row_standardize(W: np.ndarray) -> np.ndarray:
    """Row-standardize a dense weights matrix.

    Parameters
    ----------
    W : np.ndarray
        Dense square matrix.

    Returns
    -------
    np.ndarray
        Row-standardized matrix, with zero-sum rows left unchanged.
    """
    W = np.asarray(W, dtype=float)
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0.0] = 1.0
    return W / rs


def dense_to_graph(W_dense: np.ndarray, row_standardize_weights: bool = False) -> Graph:
    """Convert dense weights matrix to a libpysal Graph.

    Parameters
    ----------
    W_dense : np.ndarray
        Dense square weights matrix.
    row_standardize_weights : bool, default=False
        Whether to row-standardize before conversion.

    Returns
    -------
    Graph
        Graph representation of the same sparse structure.
    """
    W_arr = np.asarray(W_dense, dtype=float)
    if row_standardize_weights:
        W_arr = row_standardize(W_arr)

    n = W_arr.shape[0]
    focal, neighbor, weight = [], [], []
    for i in range(n):
        for j in range(n):
            wij = W_arr[i, j]
            if wij != 0.0:
                focal.append(i)
                neighbor.append(j)
                weight.append(wij)

    g = Graph.from_arrays(
        np.asarray(focal, dtype=int),
        np.asarray(neighbor, dtype=int),
        np.asarray(weight, dtype=float),
    )
    return g.transform("r")


def weights_from_geodataframe(
    gdf: Any,
    contiguity: str = "queen",
    k: int = 4,
    distance_threshold: float | None = None,
) -> Graph:
    """Build a row-standardized Graph from GeoDataFrame geometry.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input geodataframe with geometry column.
    contiguity : {"queen", "rook", "knn", "distance"}, default="queen"
        Rule to build neighbor structure.
    k : int, default=4
        Number of neighbors for ``contiguity='knn'``.
    distance_threshold : float, optional
        Distance cutoff for ``contiguity='distance'``.

    Returns
    -------
    Graph
        Row-standardized graph built from geometry.
    """
    try:
        from libpysal.weights import DistanceBand, KNN, Queen, Rook
    except Exception as exc:  # pragma: no cover
        raise ImportError("libpysal weights builders are required for GeoDataFrame support.") from exc

    if gdf is None:
        raise ValueError("gdf must be provided when building weights from geometry.")
    if not hasattr(gdf, "geometry"):
        raise TypeError("gdf must be a GeoDataFrame-like object with a geometry column.")

    mode = contiguity.lower()
    if mode == "queen":
        w = Queen.from_dataframe(gdf, use_index=False)
    elif mode == "rook":
        w = Rook.from_dataframe(gdf, use_index=False)
    elif mode == "knn":
        w = KNN.from_dataframe(gdf, k=int(k), use_index=False)
    elif mode == "distance":
        if distance_threshold is None:
            raise ValueError("distance_threshold must be supplied when contiguity='distance'.")
        cent = np.column_stack([gdf.geometry.centroid.x.values, gdf.geometry.centroid.y.values])
        w = DistanceBand(cent, threshold=float(distance_threshold), binary=True)
    else:
        raise ValueError("contiguity must be one of {'queen', 'rook', 'knn', 'distance'}." )

    g = Graph.from_W(w)
    return g.transform("r")


def resolve_weights(
    W: Graph | sp.spmatrix | np.ndarray | None = None,
    gdf: Any | None = None,
    contiguity: str = "queen",
    k: int = 4,
    distance_threshold: float | None = None,
) -> tuple[np.ndarray, Graph]:
    """Resolve user-supplied spatial structure to dense matrix and Graph.

    Parameters
    ----------
    W : Graph or sparse/dense matrix, optional
        Explicit spatial structure. If provided together with ``gdf``, the
        matrix/graph is used and checked for dimensional compatibility with
        the GeoDataFrame.
    gdf : geopandas.GeoDataFrame, optional
        Used only when ``W`` is not supplied.
    contiguity : str, default="queen"
        GeoDataFrame neighbor construction mode.
    k : int, default=4
        KNN neighbor count when ``contiguity='knn'``.
    distance_threshold : float, optional
        Distance threshold when ``contiguity='distance'``.

    Returns
    -------
    tuple[np.ndarray, Graph]
        ``(W_dense, W_graph)`` both row-standardized.
    """
    if W is not None:
        gdf_n = len(gdf) if gdf is not None else None
        if isinstance(W, Graph):
            g = W.transform("r")
            Wd = g.sparse.toarray().astype(float)
            if gdf_n is not None and Wd.shape[0] != gdf_n:
                raise ValueError("W and gdf must describe the same number of spatial units.")
            return Wd, g
        if sp.issparse(W):
            Wd = row_standardize(W.toarray().astype(float))
            if gdf_n is not None and Wd.shape[0] != gdf_n:
                raise ValueError("W and gdf must describe the same number of spatial units.")
            return Wd, dense_to_graph(Wd)
        Wd = row_standardize(np.asarray(W, dtype=float))
        if gdf_n is not None and Wd.shape[0] != gdf_n:
            raise ValueError("W and gdf must describe the same number of spatial units.")
        return Wd, dense_to_graph(Wd)

    if gdf is None:
        raise ValueError("Provide either W or gdf.")

    g = weights_from_geodataframe(gdf, contiguity=contiguity, k=k, distance_threshold=distance_threshold)
    return g.sparse.toarray().astype(float), g


def make_design_matrix(rng: np.random.Generator, n: int, k: int = 1, add_intercept: bool = True) -> np.ndarray:
    """Generate synthetic design matrix.

    Parameters
    ----------
    rng : np.random.Generator
        Random generator.
    n : int
        Number of observations.
    k : int, default=1
        Number of non-intercept regressors.
    add_intercept : bool, default=True
        Whether to prepend a constant column.

    Returns
    -------
    np.ndarray
        Design matrix of shape ``(n, k + int(add_intercept))``.
    """
    Z = rng.standard_normal((n, k))
    if add_intercept:
        return np.column_stack([np.ones(n), Z])
    return Z


def panel_index(N: int, T: int) -> pd.DataFrame:
    """Create time-first panel index DataFrame.

    Parameters
    ----------
    N : int
        Number of units.
    T : int
        Number of periods.

    Returns
    -------
    pd.DataFrame
        Columns ``unit`` and ``time`` matching stacked time-first ordering.
    """
    units = np.tile(np.arange(N), T)
    times = np.repeat(np.arange(T), N)
    return pd.DataFrame({"unit": units, "time": times})
