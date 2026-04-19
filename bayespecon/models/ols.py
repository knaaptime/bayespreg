"""Bayesian OLS (non-spatial) cross-sectional regression model.

y = X @ beta + epsilon,  epsilon ~ N(0, sigma^2 I)

This model contains no spatial structure of its own.  It is the natural
baseline from which spatial specification tests are run to determine which
spatial model — SAR, SEM, SLX, etc. — is most appropriate.  W is optional
at construction time, but several methods (``moran_test``,
``lm_error_test``, ``lm_lag_test``, ``spatial_specification_tests``) require
W to be supplied.
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from .base import SpatialModel
from ..diagnostics import DiagnosticResult


class OLS(SpatialModel):
    """Bayesian ordinary least squares cross-sectional regression.

    .. math::
        y = X\\beta + \\varepsilon, \\quad \\varepsilon \\sim N(0, \\sigma^2 I)

    This model places diffuse Normal priors on the coefficient vector
    :math:`\\beta` and a HalfNormal prior on the noise standard deviation
    :math:`\\sigma`.

    ``W`` is **optional**.  If supplied, Moran's I, LM-error, and LM-lag
    tests can be run on the OLS residuals to guide model selection.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula string, e.g. ``"price ~ poverty + income"``.
        If provided, ``data`` must also be supplied.
    data : pandas.DataFrame or geopandas.GeoDataFrame, optional
        Data source when using formula mode.
    y : array-like, optional
        Dependent variable of shape ``(n,)``.  Required in matrix mode.
    X : array-like or pandas.DataFrame, optional
        Predictor matrix.  Required in matrix mode.
    W : libpysal.graph.Graph or scipy.sparse matrix, optional
        Spatial weights matrix of shape ``(n, n)``.  Not used during
        estimation; required only for the spatial specification tests
        ``moran_test``, ``lm_error_test``, ``lm_lag_test``, and
        ``spatial_specification_tests``.
    priors : dict, optional
        Override default priors.  Supported keys:

        - ``beta_mu`` (float, default 0): Prior mean for :math:`\\beta`.
        - ``beta_sigma`` (float, default 1e6): Prior std for :math:`\\beta`.
        - ``sigma_sigma`` (float, default 10): Scale for HalfNormal prior
          on :math:`\\sigma`.
    logdet_method : str, optional
        Unused for OLS (no spatial lag); kept for API compatibility with
        :class:`SpatialModel`.
    """

    def _build_pymc_model(self) -> pm.Model:
        """Construct the PyMC model for Bayesian OLS regression.

        Returns
        -------
        pymc.Model
            Compiled probabilistic model object with Normal likelihood.
        """
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)

        with pm.Model(coords=self._model_coords()) as model:
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)
            mu = pt.dot(self._X, beta)
            pm.Normal("obs", mu=mu, sigma=sigma, observed=self._y)

        return model

    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        """Not applicable — OLS has no spatial lag structure.

        Raises
        ------
        NotImplementedError
            Always raised; use ``spatial_specification_tests()`` instead to
            assess spatial structure after estimation.
        """
        raise NotImplementedError(
            "OLS has no spatial lag structure and therefore no spatial effects. "
            "Run spatial_specification_tests() to assess which spatial model "
            "is appropriate, then refit with SAR, SEM, SLX, SDM, or SDEM."
        )

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Compute fitted values at posterior mean coefficients.

        Returns
        -------
        np.ndarray
            Posterior-mean fitted values ``X @ E[beta | data]``.
        """
        beta = self._posterior_mean("beta")
        return self._X @ beta

    # ------------------------------------------------------------------
    # Spatial specification tests
    # ------------------------------------------------------------------

    def lm_error_test(self) -> DiagnosticResult:
        """LM test for omitted spatial error autocorrelation.

        Tests whether the OLS residuals exhibit spatial error dependence,
        suggesting that a SEM or SDEM specification may be more appropriate
        than OLS.

        Returns
        -------
        DiagnosticResult
            ``name`` : ``"lm_error"``

            ``statistic`` : float — LM statistic value.

            ``pvalue`` : float — p-value under :math:`\\chi^2(1)` null.

        Notes
        -----
        H\\ :sub:`0`: no spatial error autocorrelation in OLS residuals.
        Under H\\ :sub:`0` the statistic is asymptotically
        :math:`\\chi^2(1)`.

        References
        ----------
        .. [1] Anselin, L. (1988). *Spatial Econometrics: Methods and Models*.
               Kluwer Academic Publishers.
        """
        self._require_W()
        from ..stats.core import lmerror
        raw = lmerror(self._y, self._X, self._W_sparse.toarray())
        return self._wrap_stats_result("lm_error", raw, "lm")

    def lm_lag_test(self) -> DiagnosticResult:
        """LM test for an omitted spatially lagged dependent variable.

        Tests whether a spatially lagged term :math:`\\rho Wy` is missing
        from the OLS specification, suggesting that a SAR or SDM model may
        be more appropriate.

        Returns
        -------
        DiagnosticResult
            ``name`` : ``"lm_lag"``

            ``statistic`` : float — LM statistic value.

            ``pvalue`` : float — p-value under :math:`\\chi^2(1)` null.

        Notes
        -----
        H\\ :sub:`0`: no omitted spatial lag of the dependent variable.
        Under H\\ :sub:`0` the statistic is asymptotically
        :math:`\\chi^2(1)`.

        References
        ----------
        .. [1] Anselin, L. (1988). *Spatial Econometrics: Methods and Models*.
               Kluwer Academic Publishers.
        """
        self._require_W()
        from ..stats.core import lmlag
        raw = lmlag(self._y, self._X, self._W_sparse.toarray())
        return self._wrap_stats_result("lm_lag", raw, "lm")

    def spatial_specification_tests(self) -> dict[str, DiagnosticResult]:
        """Run the full battery of spatial specification tests on OLS residuals.

        Runs Moran's I, the LM-error test, and the LM-lag test.  Together
        these three tests guide model selection: if LM-error dominates,
        prefer SEM; if LM-lag dominates, prefer SAR; if both are significant,
        consider SDM or SDEM.

        Returns
        -------
        dict[str, DiagnosticResult]
            Keys: ``"moran"``, ``"lm_error"``, ``"lm_lag"``.

        See Also
        --------
        moran_test : Moran's I for residual spatial autocorrelation.
        lm_error_test : LM test for spatial error dependence.
        lm_lag_test : LM test for omitted spatial lag.
        """
        return {
            "moran": self.moran_test(),
            "lm_error": self.lm_error_test(),
            "lm_lag": self.lm_lag_test(),
        }
