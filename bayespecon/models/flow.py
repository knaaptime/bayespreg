"""Bayesian spatial flow (origin-destination) models.

Implements fully Bayesian SAR-type flow models following
:cite:t:`lesage2008SpatialEconometric`.  The observed variable is an
:math:`n \\times n` flow matrix (or its vectorised form), and the weight
structure uses three Kronecker-product matrices:

.. math::

    W_d = I_n \\otimes W, \\quad W_o = W \\otimes I_n, \\quad W_w = W \\otimes W

so that the model is:

.. math::

    y = \\rho_d W_d y + \\rho_o W_o y + \\rho_w W_w y + X\\beta + \\varepsilon,
    \\quad \\varepsilon \\sim \\mathcal{N}(0, \\sigma^2 I_N)

where :math:`N = n^2`.

Two variants are provided:

* :class:`SAR_Flow` — three free ρ parameters with a Dirichlet stability
  constraint (default) or a quadratic-wall potential when competitive effects
  are needed (``restrict_positive=False``).
* :class:`SAR_Flow_Separable` — constrained :math:`\\rho_w = -\\rho_d \\rho_o`,
  enabling exact eigenvalue-based log-determinant.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import scipy.sparse as sp
from libpysal.graph import Graph

from ..graph import _validate_graph, flow_weight_matrices
from ..logdet import (
    _flow_logdet_poly_coeffs,
    compute_flow_traces,
    flow_logdet_pytensor,
    logdet_eigenvalue,
    make_flow_separable_logdet,
)
from ..ops import kron_solve_vec


class FlowModel(ABC):
    """Abstract base class for Bayesian spatial flow regression models.

    Unlike :class:`~bayespecon.models.base.SpatialModel`, this class works
    with an :math:`N = n^2` vectorised response and three Kronecker-product
    weight matrices constructed from a single n×n graph.  The API mirrors
    :class:`~bayespecon.models.base.SpatialModel` (``fit``, ``summary``,
    ``inference_data``) but the internals are tailored to the flow structure.

    The model accepts a full O-D design matrix *X* of shape ``(n², p)``,
    typically produced by :func:`~bayespecon.graph.flow_design_matrix` or
    :func:`~bayespecon.graph.flow_design_matrix_with_orig`.

    Parameters
    ----------
    y : array-like, shape (n, n) or (N,)
        Observed O-D flow matrix (or its vec-form).  Must be a square
        matrix or a flat vector of length :math:`N = n^2`.
    G : libpysal.graph.Graph
        Row-standardised spatial graph on *n* units.  Validated by
        :func:`~bayespecon.graph._validate_graph`.
    X : np.ndarray or pandas.DataFrame, shape (N, p)
        Full origin-destination design matrix with :math:`N = n^2` rows.
        This is typically produced by
        :func:`~bayespecon.graph.flow_design_matrix` or
        :func:`~bayespecon.graph.flow_design_matrix_with_orig`.
        If a DataFrame, column names are inferred automatically.
    col_names : list[str], optional
        Column labels for *X*.  If *X* is a DataFrame, column names are
        inferred automatically.  Defaults to ``["x0", "x1", ...]``.
    k : int, optional
        Number of regional attribute columns in the design matrix (i.e.,
        the number of destination/origin variable pairs).  When the design
        matrix follows the standard LeSage layout
        ``[intercept, intra_indicator, dest_*, orig_*, intra_*, (dist)]``,
        *k* can be inferred from the column names.  Provide *k* explicitly
        if column names do not follow the ``dest_*``/``orig_*`` convention.
    priors : dict, optional
        Override default priors.  Supported keys vary by subclass.
    logdet_method : str, default "traces"
        How to compute :math:`\\log|I_N - \\rho_d W_d - \\rho_o W_o - \\rho_w W_w|`.
        ``"traces"`` uses Barry-Pace stochastic traces with the multinomial
        Kronecker identity (the default and recommended method).
        ``"separable"`` (SAR_Flow_Separable only) uses eigenvalues of W.
    restrict_positive : bool, default True
        If True, use a ``pm.Dirichlet`` prior that restricts :math:`\\rho_d,
        \\rho_o, \\rho_w \\geq 0` with :math:`\\rho_d + \\rho_o + \\rho_w \\leq 1`.
        This is NUTS-safe and appropriate for most flow applications.
        If False, use three independent ``pm.Uniform(-1, 1)`` priors with a
        differentiable quadratic-wall stability potential.
    miter : int, default 30
        Trace polynomial order for the log-determinant (only used when
        ``logdet_method="traces"``).  Higher values improve accuracy at
        the cost of more precomputation.
    titer : int, default 800
        Geometric tail cutoff for the log-determinant series.
    trace_riter : int, default 50
        Number of Monte Carlo probes for trace estimation.
    trace_seed : int, optional
        Random seed for trace estimation reproducibility.
    """

    def __init__(
        self,
        y: Union[np.ndarray, pd.Series],
        G: Graph,
        X: Union[np.ndarray, pd.DataFrame],
        col_names: Optional[list] = None,
        k: Optional[int] = None,
        priors: Optional[dict] = None,
        logdet_method: str = "traces",
        restrict_positive: bool = True,
        miter: int = 30,
        titer: int = 800,
        trace_riter: int = 50,
        trace_seed: Optional[int] = None,
    ):
        self.priors = priors or {}
        self.logdet_method = logdet_method
        self.restrict_positive = restrict_positive
        self.miter = miter
        self.titer = titer
        self._idata: Optional[az.InferenceData] = None
        self._pymc_model: Optional[pm.Model] = None
        self._approximation = None

        # Validate and extract the n×n weight matrix
        self._W_sparse: sp.csr_matrix = _validate_graph(G)
        self._n: int = self._W_sparse.shape[0]
        self._N: int = self._n * self._n

        # Validate y
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim == 2:
            if y_arr.shape != (self._n, self._n):
                raise ValueError(
                    f"y matrix must be ({self._n}, {self._n}), got {y_arr.shape}."
                )
            self._y_vec = y_arr.ravel()
        elif y_arr.ndim == 1:
            if len(y_arr) != self._N:
                raise ValueError(
                    f"y vector must have length N={self._N} (= n²), got {len(y_arr)}."
                )
            self._y_vec = y_arr
        else:
            raise ValueError("y must be a 1-D or 2-D array.")

        # Validate X and build design matrix
        if isinstance(X, pd.DataFrame):
            if col_names is None:
                col_names = list(X.columns)
            X_arr = X.to_numpy(dtype=np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)

        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        if X_arr.shape[0] != self._N:
            raise ValueError(
                f"X must have {self._N} rows (= n² = {self._n}²), "
                f"got {X_arr.shape[0]}."
            )

        self._X_design: np.ndarray = X_arr  # (N, p)
        if col_names is not None:
            self._feature_names: list[str] = list(col_names)
        elif X_arr.shape[1] == 0:
            self._feature_names = []
        else:
            self._feature_names = [f"x{i}" for i in range(X_arr.shape[1])]

        # Infer k (number of regional attribute columns) for effects computation.
        # Standard LeSage layout: [intercept, intra_indicator, dest_*, orig_*, intra_*, (dist)]
        if k is not None:
            self._k: int = k
        else:
            dest_cols = [name for name in self._feature_names if name.startswith("dest_")]
            self._k = len(dest_cols)
            if self._k == 0:
                # Fallback: cannot infer k from column names; effects decomposition
                # will not be available.  Set k=0 as a sentinel.
                self._k = 0

        # Pre-compute logdet data for separable constraint: log|Lo⊗Ld| = n*f(ρ_d) + n*f(ρ_o).
        # Also keep _W_eigs for backward compatibility.
        self._W_eigs: Optional[np.ndarray] = None
        self._separable_logdet_fn = None
        _SEPARABLE_METHODS = {"separable", "eigenvalue", "chebyshev", "mc_poly"}
        if logdet_method in _SEPARABLE_METHODS:
            _method_key = "eigenvalue" if logdet_method == "separable" else logdet_method
            self._separable_logdet_fn = make_flow_separable_logdet(
                self._W_sparse, self._n, method=_method_key,
                miter=miter, riter=trace_riter, random_state=trace_seed,
            )
            if logdet_method in ("separable", "eigenvalue"):
                self._W_eigs = np.linalg.eigvals(
                    self._W_sparse.toarray().astype(np.float64)).real

        # Pre-compute spatial lags: Wd_y, Wo_y, Ww_y
        wms = flow_weight_matrices(G)
        self._Wd_y: np.ndarray = wms["destination"] @ self._y_vec
        self._Wo_y: np.ndarray = wms["origin"] @ self._y_vec
        self._Ww_y: np.ndarray = wms["network"] @ self._y_vec

        # Keep N×N sparse weight matrices for effects computation
        self._Wd: sp.csr_matrix = wms["destination"]
        self._Wo: sp.csr_matrix = wms["origin"]
        self._Ww: sp.csr_matrix = wms["network"]

        # Pre-compute flow log-det traces (only for "traces" method)
        if logdet_method == "traces":
            self._traces: np.ndarray = compute_flow_traces(
                self._W_sparse, miter=miter, riter=trace_riter, random_state=trace_seed
            )
            result = _flow_logdet_poly_coeffs(self._traces, self._n, miter)
            (
                self._poly_a,
                self._poly_b,
                self._poly_c,
                self._poly_coeffs,
                self._miter_a,
                self._miter_b,
                self._miter_c,
                self._miter_coeffs,
            ) = result
        else:
            self._traces = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_pymc_model(self) -> pm.Model:
        """Construct and return the PyMC model.  Implemented by subclasses."""

    @abstractmethod
    def _compute_spatial_effects_posterior(
        self, draws: Optional[int] = None
    ) -> dict[str, np.ndarray]:
        """Compute posterior spatial effects.  Implemented by subclasses."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _posterior_var_names(
        self,
        model: pm.Model,
        *,
        store_lambda: bool,
    ) -> list[str]:
        names = [rv.name for rv in model.free_RVs]
        names.extend(
            var.name
            for var in model.deterministics
            if store_lambda or var.name != "lambda"
        )
        return list(dict.fromkeys(name for name in names if name is not None))

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: Optional[int] = None,
        store_lambda: bool = False,
        **sample_kwargs,
    ) -> az.InferenceData:
        """Draw samples from the posterior.

        Parameters
        ----------
        draws : int, default 2000
            Number of posterior samples per chain (after tuning).
        tune : int, default 1000
            Number of tuning (warm-up) steps per chain.
        chains : int, default 4
            Number of parallel chains.
        target_accept : float, default 0.9
            Target acceptance rate for NUTS.
        random_seed : int, optional
            Seed for reproducibility.
        store_lambda : bool, default False
            If True, include the high-dimensional fitted mean ``lambda`` in the
            stored posterior. Leaving this False reduces memory and conversion
            overhead for Poisson flow models.
        **sample_kwargs
            Additional keyword arguments forwarded to ``pm.sample``.

        Returns
        -------
        arviz.InferenceData
        """
        model = self._build_pymc_model()
        self._pymc_model = model
        self._approximation = None
        if "var_names" not in sample_kwargs and not store_lambda:
            sample_kwargs["var_names"] = self._posterior_var_names(
                model,
                store_lambda=False,
            )
        with model:
            self._idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                **sample_kwargs,
            )
        return self._idata

    def fit_approx(
        self,
        draws: int = 2000,
        n: int = 10000,
        method: str = "advi",
        random_seed: Optional[int] = None,
        store_lambda: bool = False,
        **fit_kwargs,
    ) -> az.InferenceData:
        """Fit a variational approximation and return posterior draws.

        Parameters
        ----------
        draws : int, default 2000
            Number of samples to draw from the fitted approximation.
        n : int, default 10000
            Number of optimisation iterations for ``pm.fit``.
        method : {"advi", "fullrank_advi"}, default "advi"
            Variational inference family to fit.
        random_seed : int, optional
            Seed for optimisation and posterior sampling.
        store_lambda : bool, default False
            If True, keep the high-dimensional fitted mean ``lambda`` in the
            posterior draws.
        **fit_kwargs
            Additional keyword arguments forwarded to ``pm.fit``.
        """
        method = method.lower()
        if method not in {"advi", "fullrank_advi"}:
            raise ValueError(
                "fit_approx method must be 'advi' or 'fullrank_advi'."
            )

        model = self._build_pymc_model()
        self._pymc_model = model
        with model:
            self._approximation = pm.fit(
                n=n,
                method=method,
                random_seed=random_seed,
                **fit_kwargs,
            )
            self._idata = self._approximation.sample(
                draws=draws,
                random_seed=random_seed,
                return_inferencedata=True,
            )

        if (
            not store_lambda
            and self._idata is not None
            and hasattr(self._idata, "posterior")
            and "lambda" in self._idata.posterior.data_vars
        ):
            self._idata.posterior = self._idata.posterior.drop_vars("lambda")

        return self._idata

    @property
    def inference_data(self) -> Optional[az.InferenceData]:
        """Return ArviZ InferenceData from the most recent fit, or None."""
        return self._idata

    @property
    def pymc_model(self) -> Optional[pm.Model]:
        """Return the PyMC model used for the most recent fit, or None."""
        return self._pymc_model

    @property
    def approximation(self):
        """Return the most recent PyMC variational approximation, if any."""
        return self._approximation

    def summary(
        self,
        var_names: Optional[list] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Return posterior summary table via ArviZ.

        Parameters
        ----------
        var_names : list, optional
            Variable names to include.  Defaults to all parameters.
        **kwargs
            Additional keyword arguments forwarded to ``az.summary``.

        Returns
        -------
        pandas.DataFrame
        """
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")
        return az.summary(self._idata, var_names=var_names, **kwargs)

    def _model_coords(self, extra: Optional[dict] = None) -> dict:
        """Return PyMC coordinate labels for named dimensions."""
        coords: dict = {"coefficient": self._feature_names}
        if extra:
            coords.update(extra)
        return coords

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assemble_A(
        self,
        rho_d: float,
        rho_o: float,
        rho_w: float,
    ) -> sp.csr_matrix:
        """Assemble A = I_N - rho_d*Wd - rho_o*Wo - rho_w*Ww (sparse N×N)."""
        I_N = sp.eye(self._N, format="csr", dtype=np.float64)
        return I_N - rho_d * self._Wd - rho_o * self._Wo - rho_w * self._Ww


# ---------------------------------------------------------------------------
# Model 1: SAR_Flow — unrestricted 3-ρ
# ---------------------------------------------------------------------------


class SAR_Flow(FlowModel):
    """Bayesian SAR flow model with three free spatial autoregressive parameters.

    .. math::

        y = \\rho_d W_d y + \\rho_o W_o y + \\rho_w W_w y + X\\beta + \\varepsilon,
        \\quad \\varepsilon \\sim \\mathcal{N}(0, \\sigma^2 I_N)

    where :math:`W_d = I_n \\otimes W`, :math:`W_o = W \\otimes I_n`,
    :math:`W_w = W \\otimes W`.

    Parameters
    ----------
    y, G, X, col_names, k, priors, logdet_method, miter, titer,
    trace_riter, trace_seed
        See :class:`FlowModel` for descriptions.
    restrict_positive : bool, default True
        If True (default), use ``pm.Dirichlet("rho_simplex", a=ones(4))``
        to enforce :math:`\\rho_d, \\rho_o, \\rho_w \\geq 0` and
        :math:`\\rho_d + \\rho_o + \\rho_w \\leq 1`.  This is NUTS-safe via
        the stick-breaking bijection and is appropriate when competitive
        (negative) spatial spillovers are not expected.

        If False, three independent ``pm.Uniform(-1, 1)`` priors are used
        together with a differentiable quadratic-wall potential to enforce
        the stability constraint.

    Notes
    -----
    The ``priors`` dict supports:

    - ``beta_mu`` (float, default 0): Prior mean for ``beta``.
    - ``beta_sigma`` (float, default 1e6): Prior std for ``beta``.
    - ``sigma_sigma`` (float, default 10): Scale for ``HalfNormal`` prior
      on ``sigma``.
    - ``rho_lower, rho_upper`` (float, default -1, 1): Bounds for
      ``pm.Uniform`` priors on each ρ (only used when
      ``restrict_positive=False``).
    """

    def _build_pymc_model(self) -> pm.Model:
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)

        Wd_y_t = pt.as_tensor_variable(self._Wd_y.astype(np.float64))
        Wo_y_t = pt.as_tensor_variable(self._Wo_y.astype(np.float64))
        Ww_y_t = pt.as_tensor_variable(self._Ww_y.astype(np.float64))
        X_t = pt.as_tensor_variable(self._X_design.astype(np.float64))
        y_t = pt.as_tensor_variable(self._y_vec.astype(np.float64))

        with pm.Model(coords=self._model_coords()) as model:
            if self.restrict_positive:
                # Dirichlet encodes ρ_d, ρ_o, ρ_w ≥ 0 and ρ_d+ρ_o+ρ_w ≤ 1 exactly.
                # rho_simplex[3] is the slack variable (not a spatial parameter).
                rho_simplex = pm.Dirichlet("rho_simplex", a=np.ones(4))
                rho_d = pm.Deterministic("rho_d", rho_simplex[0])
                rho_o = pm.Deterministic("rho_o", rho_simplex[1])
                rho_w = pm.Deterministic("rho_w", rho_simplex[2])
            else:
                rho_lower = self.priors.get("rho_lower", -1.0)
                rho_upper = self.priors.get("rho_upper", 1.0)
                rho_d = pm.Uniform("rho_d", lower=rho_lower, upper=rho_upper)
                rho_o = pm.Uniform("rho_o", lower=rho_lower, upper=rho_upper)
                rho_w = pm.Uniform("rho_w", lower=rho_lower, upper=rho_upper)
                # Quadratic-wall stability potential (differentiable everywhere)
                slack = 1.0 - rho_d - rho_o - rho_w
                pm.Potential(
                    "stability",
                    pt.switch(slack > 0.0, 0.0, -1e6 * slack**2),
                )

            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

            mu = rho_d * Wd_y_t + rho_o * Wo_y_t + rho_w * Ww_y_t + pt.dot(X_t, beta)
            pm.Normal("obs", mu=mu, sigma=sigma, observed=y_t)

            # Jacobian: log|I_N - rho_d*Wd - rho_o*Wo - rho_w*Ww|
            pm.Potential(
                "jacobian",
                flow_logdet_pytensor(
                    rho_d,
                    rho_o,
                    rho_w,
                    self._poly_a,
                    self._poly_b,
                    self._poly_c,
                    self._poly_coeffs,
                    self._miter_a,
                    self._miter_b,
                    self._miter_c,
                    self._miter_coeffs,
                    self.miter,
                    self.titer,
                ),
            )

        return model

    def _compute_spatial_effects_posterior(
        self, draws: Optional[int] = None
    ) -> dict[str, np.ndarray]:
        """Compute posterior origin / destination / intra / network / total effects.

        For each posterior draw of (ρ_d, ρ_o, ρ_w, β_d, β_o), solves the
        scalar summary-statistic system following LeSage (2005) ``calc_effects.m``.

        The shock vector for predictor *p* is
        :math:`z_p = (\\beta_d^{(p)} + \\beta_o^{(p)})\\,\\mathbf{1}_N`, so
        :math:`T_p = A^{-1} z_p = (\\beta_d^{(p)} + \\beta_o^{(p)})\\,A^{-1}\\mathbf{1}_N`.
        Consequently **one N×N solve per draw** suffices for all *k* predictors.

        Parameters
        ----------
        draws : int, optional
            Number of posterior draws to use.  Defaults to all draws.

        Returns
        -------
        dict[str, np.ndarray]
            Keys: ``"origin"``, ``"destination"``, ``"intra"``, ``"network"``,
            ``"total"``.  Each value has shape ``(draws, k)``.
        """
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")

        idata = self._idata
        n = self._n
        N = self._N
        k = self._k

        rho_d_draws = idata.posterior["rho_d"].values.reshape(-1)
        rho_o_draws = idata.posterior["rho_o"].values.reshape(-1)
        rho_w_draws = idata.posterior["rho_w"].values.reshape(-1)
        beta_draws = idata.posterior["beta"].values.reshape(-1, len(self._feature_names))

        dest_start = 2
        orig_start = 2 + k

        n_draws_total = len(rho_d_draws)
        if draws is not None:
            n_draws_total = min(draws, n_draws_total)
            rho_d_draws = rho_d_draws[:n_draws_total]
            rho_o_draws = rho_o_draws[:n_draws_total]
            rho_w_draws = rho_w_draws[:n_draws_total]
            beta_draws = beta_draws[:n_draws_total]

        # Precompute masks (same for all draws)
        intra_mask = np.eye(n, dtype=bool).ravel()  # (N,) diagonal positions
        ones_N = np.ones(N, dtype=np.float64)

        out_origin  = np.zeros((n_draws_total, k), dtype=np.float64)
        out_dest    = np.zeros((n_draws_total, k), dtype=np.float64)
        out_intra   = np.zeros((n_draws_total, k), dtype=np.float64)
        out_network = np.zeros((n_draws_total, k), dtype=np.float64)
        out_total   = np.zeros((n_draws_total, k), dtype=np.float64)

        for idx in range(n_draws_total):
            rd = float(rho_d_draws[idx])
            ro = float(rho_o_draws[idx])
            rw = float(rho_w_draws[idx])
            beta_d_vec = beta_draws[idx, dest_start : dest_start + k]  # (k,)
            beta_o_vec = beta_draws[idx, orig_start : orig_start + k]  # (k,)

            # z_p = (bd[p] + bo[p]) * ones_N for every predictor p, so
            # T_p = A^{-1} z_p = scale[p] * t_ones  where  t_ones = A^{-1} ones_N.
            # One sparse factorisation covers all k predictors.
            A = self._assemble_A(rd, ro, rw)
            t_ones = sp.linalg.spsolve(A, ones_N)  # (N,)

            scale = beta_d_vec + beta_o_vec  # (k,)
            # T_all[:, p] = t_ones * scale[p]
            T_all = t_ones[:, np.newaxis] * scale[np.newaxis, :]  # (N, k)

            totals = T_all.mean(axis=0)                              # (k,)
            intras = T_all[intra_mask, :].sum(axis=0) / N           # (k,)
            # T_dest = T_orig = T_all (dest/orig loops cover all N positions)
            out_total[idx, :]   = totals
            out_intra[idx, :]   = intras
            out_dest[idx, :]    = totals
            out_origin[idx, :]  = totals
            out_network[idx, :] = totals - intras - totals - totals

        return {
            "total":       out_total,
            "destination": out_dest,
            "origin":      out_origin,
            "intra":       out_intra,
            "network":     out_network,
        }


# ---------------------------------------------------------------------------
# Model 2: SAR_Flow_Separable — constrained ρ_w = −ρ_d·ρ_o
# ---------------------------------------------------------------------------


class SAR_Flow_Separable(FlowModel):
    """Bayesian separable SAR flow model with ρ_w = −ρ_d · ρ_o.

    The separability constraint :math:`\\rho_w = -\\rho_d \\rho_o` reduces
    the flow weight matrix to a Kronecker structure whose log-determinant
    factors as:

    .. math::

        \\log|I_N - \\rho_d W_d - \\rho_o W_o + \\rho_d \\rho_o W_w|
        = n \\log|I_n - \\rho_d W| + n \\log|I_n - \\rho_o W|

    enabling exact O(n) log-det evaluation via eigenvalues — no trace
    estimation required.

    Parameters
    ----------
    y, G, X, col_names, k, priors, miter, titer,
    trace_riter, trace_seed
        See :class:`FlowModel` for descriptions.  The ``logdet_method``
        and ``restrict_positive`` parameters are not relevant for this model
        and are ignored.

    Notes
    -----
    The ``priors`` dict supports:

    - ``beta_mu`` (float, default 0): Prior mean for ``beta``.
    - ``beta_sigma`` (float, default 1e6): Prior std for ``beta``.
    - ``sigma_sigma`` (float, default 10): Scale for ``HalfNormal`` prior
      on ``sigma``.
    - ``rho_lower, rho_upper`` (float, default -0.999, 0.999): Bounds for
      ``pm.Uniform`` priors on ``rho_d`` and ``rho_o``.
    """

    def __init__(self, y, G, X, **kwargs):
        # Normalize logdet_method: 'separable' is an alias for 'eigenvalue'.
        # Users may also pass 'chebyshev' or 'mc_poly' for O(m) per-step cost.
        method = kwargs.pop("logdet_method", "eigenvalue")
        if method == "separable":
            method = "eigenvalue"
        _VALID = {"eigenvalue", "chebyshev", "mc_poly"}
        if method not in _VALID:
            raise ValueError(
                f"SAR_Flow_Separable logdet_method must be one of {sorted(_VALID)}; "
                f"got {method!r}."
            )
        kwargs["logdet_method"] = method
        super().__init__(y, G, X, **kwargs)

    def _build_pymc_model(self) -> pm.Model:
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)
        rho_lower = self.priors.get("rho_lower", -0.999)
        rho_upper = self.priors.get("rho_upper", 0.999)

        if self._separable_logdet_fn is None:
            raise RuntimeError(
                "SAR_Flow_Separable requires precomputed logdet data; "
                "initialize with logdet_method='eigenvalue', 'chebyshev', or 'mc_poly'."
            )
        Wd_y_t = pt.as_tensor_variable(self._Wd_y.astype(np.float64))
        Wo_y_t = pt.as_tensor_variable(self._Wo_y.astype(np.float64))
        Ww_y_t = pt.as_tensor_variable(self._Ww_y.astype(np.float64))
        X_t = pt.as_tensor_variable(self._X_design.astype(np.float64))
        y_t = pt.as_tensor_variable(self._y_vec.astype(np.float64))

        with pm.Model(coords=self._model_coords()) as model:
            rho_d = pm.Uniform("rho_d", lower=rho_lower, upper=rho_upper)
            rho_o = pm.Uniform("rho_o", lower=rho_lower, upper=rho_upper)
            # rho_w is deterministic: -rho_d * rho_o  (must appear in posterior)
            rho_w = pm.Deterministic("rho_w", -rho_d * rho_o)

            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

            mu = rho_d * Wd_y_t + rho_o * Wo_y_t + rho_w * Ww_y_t + pt.dot(X_t, beta)
            pm.Normal("obs", mu=mu, sigma=sigma, observed=y_t)

            # Jacobian: n*log|I_n - rho_d*W| + n*log|I_n - rho_o*W|
            # factorisation holds exactly for the separable constraint.
            pm.Potential(
                "jacobian",
                self._separable_logdet_fn(rho_d, rho_o),
            )

        return model

    def _compute_spatial_effects_posterior(
        self, draws: Optional[int] = None
    ) -> dict[str, np.ndarray]:
        """Compute posterior effects using Kronecker-factored solve.

        Identical statistics to :meth:`SAR_Flow._compute_spatial_effects_posterior`
        but exploits :math:`A = L_o \\otimes L_d` to replace the
        :math:`N\\times N` sparse factorisation with two :math:`n\\times n`
        sparse solves via :func:`~bayespecon.ops.kron_solve_vec`.

        Parameters
        ----------
        draws : int, optional
            Number of posterior draws to use.  Defaults to all.

        Returns
        -------
        dict[str, np.ndarray]
            Same keys and shapes as :meth:`SAR_Flow._compute_spatial_effects_posterior`.
        """
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")

        idata = self._idata
        n = self._n
        N = self._N
        k = self._k
        W = self._W_sparse.tocsr()
        I_n = sp.eye(n, format="csr", dtype=np.float64)

        rho_d_draws = idata.posterior["rho_d"].values.reshape(-1)
        rho_o_draws = idata.posterior["rho_o"].values.reshape(-1)
        beta_draws = idata.posterior["beta"].values.reshape(-1, len(self._feature_names))

        dest_start = 2
        orig_start = 2 + k

        n_draws_total = len(rho_d_draws)
        if draws is not None:
            n_draws_total = min(draws, n_draws_total)
            rho_d_draws = rho_d_draws[:n_draws_total]
            rho_o_draws = rho_o_draws[:n_draws_total]
            beta_draws  = beta_draws[:n_draws_total]

        intra_mask = np.eye(n, dtype=bool).ravel()  # (N,) diagonal positions
        ones_N = np.ones(N, dtype=np.float64)

        out_origin  = np.zeros((n_draws_total, k), dtype=np.float64)
        out_dest    = np.zeros((n_draws_total, k), dtype=np.float64)
        out_intra   = np.zeros((n_draws_total, k), dtype=np.float64)
        out_network = np.zeros((n_draws_total, k), dtype=np.float64)
        out_total   = np.zeros((n_draws_total, k), dtype=np.float64)

        for idx in range(n_draws_total):
            rd = float(rho_d_draws[idx])
            ro = float(rho_o_draws[idx])
            beta_d_vec = beta_draws[idx, dest_start : dest_start + k]  # (k,)
            beta_o_vec = beta_draws[idx, orig_start : orig_start + k]  # (k,)

            Ld = (I_n - rd * W).tocsr()
            Lo = (I_n - ro * W).tocsr()
            # Two n×n solves instead of one N×N factorisation
            t_ones = kron_solve_vec(Lo, Ld, ones_N, n)  # (N,)

            scale = beta_d_vec + beta_o_vec  # (k,)
            T_all = t_ones[:, np.newaxis] * scale[np.newaxis, :]  # (N, k)

            totals = T_all.mean(axis=0)                    # (k,)
            intras = T_all[intra_mask, :].sum(axis=0) / N  # (k,)
            out_total[idx, :]   = totals
            out_intra[idx, :]   = intras
            out_dest[idx, :]    = totals
            out_origin[idx, :]  = totals
            out_network[idx, :] = totals - intras - totals - totals

        return {
            "total":       out_total,
            "destination": out_dest,
            "origin":      out_origin,
            "intra":       out_intra,
            "network":     out_network,
        }


# ---------------------------------------------------------------------------
# Model 3: PoissonFlow — Poisson response, unrestricted 3-rho
# ---------------------------------------------------------------------------


class PoissonFlow(SAR_Flow):
    r"""Bayesian SAR flow model with a Poisson observation distribution.

    Models origin-destination flow **counts** :math:`y_{ij} \in \mathbb{N}_0`
    as:

    .. math::

        y_{ij} \sim \operatorname{Poisson}(\lambda_{ij}), \qquad
        \log \boldsymbol{\lambda} = A(\boldsymbol{\rho})^{-1} X\beta

    where the system matrix :math:`A(\rho_d, \rho_o, \rho_w) =
    I_N - \rho_d W_d - \rho_o W_o - \rho_w W_w`.

    The spatial filter is on the **log-mean** scale.  The implicit solve
    :math:`\eta = A^{-1} X\beta` is embedded in the PyMC graph via
    :class:`~bayespecon.ops.SparseFlowSolveOp`, which provides NUTS gradients
    via the adjoint method.

    Parameters
    ----------
    y : array-like, shape (n, n) or (N,)
        Observed non-negative integer flow counts.  Float arrays whose values
        are close to integers are silently rounded.
    G : libpysal.graph.Graph
        Row-standardised spatial graph on *n* units.
    X : np.ndarray or pandas.DataFrame, shape (N, p)
        Full origin-destination design matrix.
    **kwargs
        Passed to :class:`FlowModel`.  ``beta_sigma`` defaults to **10**.

    Notes
    -----
    There is no ``sigma`` parameter; the Poisson variance equals the mean.
    """

    def __init__(self, y, G, X, **kwargs):
        y_arr = np.asarray(y)
        if not np.issubdtype(y_arr.dtype, np.integer):
            y_rounded = np.round(y_arr).astype(np.int64)
            if not np.allclose(y_arr, y_rounded):
                raise ValueError(
                    "PoissonFlow requires integer-valued observations; "
                    f"got dtype {y_arr.dtype} with non-integer values."
                )
            y_arr = y_rounded
        if np.any(y_arr < 0):
            raise ValueError(
                "PoissonFlow requires non-negative integer observations."
            )
        super().__init__(y_arr.astype(np.float64), G, X, **kwargs)
        self._y_int_vec: np.ndarray = y_arr.ravel().astype(np.int64)

    def _build_pymc_model(self) -> pm.Model:
        from ..ops import SparseFlowSolveOp

        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 10.0)

        X_t = pt.as_tensor_variable(self._X_design.astype(np.float64))

        with pm.Model(coords=self._model_coords()) as model:
            if self.restrict_positive:
                rho_simplex = pm.Dirichlet("rho_simplex", a=np.ones(4))
                rho_d = pm.Deterministic("rho_d", rho_simplex[0])
                rho_o = pm.Deterministic("rho_o", rho_simplex[1])
                rho_w = pm.Deterministic("rho_w", rho_simplex[2])
            else:
                rho_lower = self.priors.get("rho_lower", -1.0)
                rho_upper = self.priors.get("rho_upper", 1.0)
                rho_d = pm.Uniform("rho_d", lower=rho_lower, upper=rho_upper)
                rho_o = pm.Uniform("rho_o", lower=rho_lower, upper=rho_upper)
                rho_w = pm.Uniform("rho_w", lower=rho_lower, upper=rho_upper)
                slack = 1.0 - rho_d - rho_o - rho_w
                pm.Potential(
                    "stability",
                    pt.switch(slack > 0.0, 0.0, -1e6 * slack**2),
                )

            beta = pm.Normal(
                "beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient"
            )

            Xb = pt.dot(X_t, beta)
            solve_op = SparseFlowSolveOp(self._Wd, self._Wo, self._Ww)
            eta = solve_op(rho_d, rho_o, rho_w, Xb)
            lam = pm.Deterministic("lambda", pt.exp(eta))

            pm.Poisson("obs", mu=lam, observed=self._y_int_vec)

            pm.Potential(
                "jacobian",
                flow_logdet_pytensor(
                    rho_d,
                    rho_o,
                    rho_w,
                    self._poly_a,
                    self._poly_b,
                    self._poly_c,
                    self._poly_coeffs,
                    self._miter_a,
                    self._miter_b,
                    self._miter_c,
                    self._miter_coeffs,
                    self.miter,
                    self.titer,
                ),
            )

        return model


# ---------------------------------------------------------------------------
# Model 4: PoissonFlow_Separable — Poisson response, rho_w = -rho_d * rho_o
# ---------------------------------------------------------------------------


class PoissonFlow_Separable(SAR_Flow_Separable):
    r"""Bayesian separable SAR flow model with a Poisson observation distribution.

    Combines the Poisson observation model of :class:`PoissonFlow` with the
    separability constraint :math:`\rho_w = -\rho_d \rho_o`, enabling the
    exact eigenvalue-based log-determinant:

    .. math::

        \log|A| = n\,\log|I_n - \rho_d W| + n\,\log|I_n - \rho_o W|

    Parameters
    ----------
    y : array-like, shape (n, n) or (N,)
        Observed non-negative integer flow counts.
    G : libpysal.graph.Graph
        Row-standardised spatial graph on *n* units.
    X : np.ndarray or pandas.DataFrame, shape (N, p)
        Full origin-destination design matrix.
    **kwargs
        Passed to :class:`FlowModel`.  ``beta_sigma`` defaults to **10**.
    """

    def __init__(self, y, G, X, **kwargs):
        y_arr = np.asarray(y)
        if not np.issubdtype(y_arr.dtype, np.integer):
            y_rounded = np.round(y_arr).astype(np.int64)
            if not np.allclose(y_arr, y_rounded):
                raise ValueError(
                    "PoissonFlow_Separable requires integer-valued observations; "
                    f"got dtype {y_arr.dtype} with non-integer values."
                )
            y_arr = y_rounded
        if np.any(y_arr < 0):
            raise ValueError(
                "PoissonFlow_Separable requires non-negative integer observations."
            )
        super().__init__(y_arr.astype(np.float64), G, X, **kwargs)
        self._y_int_vec: np.ndarray = y_arr.ravel().astype(np.int64)

    def _build_pymc_model(self) -> pm.Model:
        from ..ops import KroneckerFlowSolveOp

        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 10.0)
        rho_lower = self.priors.get("rho_lower", -0.999)
        rho_upper = self.priors.get("rho_upper", 0.999)

        n = self._n
        eigs = self._W_eigs.astype(np.float64)
        X_t = pt.as_tensor_variable(self._X_design.astype(np.float64))

        with pm.Model(coords=self._model_coords()) as model:
            rho_d = pm.Uniform("rho_d", lower=rho_lower, upper=rho_upper)
            rho_o = pm.Uniform("rho_o", lower=rho_lower, upper=rho_upper)
            pm.Deterministic("rho_w", -rho_d * rho_o)

            beta = pm.Normal(
                "beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient"
            )

            # KroneckerFlowSolveOp exploits A = L_d ⊗ L_o (two n×n solves
            # instead of one n²×n² solve — O(n³) vs O(n⁶)).
            # rho_w is kept as a Deterministic for trace reporting only.
            Xb = pt.dot(X_t, beta)
            solve_op = KroneckerFlowSolveOp(self._W_sparse, n)
            eta = solve_op(rho_d, rho_o, Xb)
            lam = pm.Deterministic("lambda", pt.exp(eta))

            pm.Poisson("obs", mu=lam, observed=self._y_int_vec)

            pm.Potential(
                "jacobian",
                n * logdet_eigenvalue(rho_d, eigs) + n * logdet_eigenvalue(rho_o, eigs),
            )

        return model
