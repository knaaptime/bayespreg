"""Spatial Autoregressive (SAR / Spatial Lag) Model.

y = rho * W @ y + X @ beta + epsilon,  epsilon ~ N(0, sigma^2 I)

The full likelihood includes the Jacobian |I - rho*W|, added via
pm.Potential so that NUTS samples from the correct posterior.
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from .base import SpatialModel
from ..logdet import make_logdet_fn


class SAR(SpatialModel):
    """Bayesian Spatial Autoregressive (Spatial Lag) model.

    .. math::
        y = \\rho Wy + X\\beta + \\varepsilon, \\quad \\varepsilon \\sim N(0, \\sigma^2 I)

    Parameters
    ----------
    formula, data, y, X, W, priors, logdet_method, w_vars
        See :class:`SpatialModel` for descriptions.

    Notes
    -----
    The ``priors`` dict supports the following keys:

    - ``rho_lower, rho_upper`` (float, default -1, 1): Bounds for the Uniform prior on rho.
    - ``beta_mu`` (float, default 0): Prior mean for beta.
    - ``beta_sigma`` (float, default 1e6): Prior std for beta.
    - ``sigma_sigma`` (float, default 10): Prior std for sigma.
    """

    def _build_pymc_model(self) -> pm.Model:
        """Construct the PyMC model for SAR regression.

        Returns
        -------
        pymc.Model
            Compiled probabilistic model object.
        """
        k = self._X.shape[1]

        rho_lower = self.priors.get("rho_lower", -1.0)
        rho_upper = self.priors.get("rho_upper", 1.0)
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)

        logdet_fn = make_logdet_fn(self._W_eigs.real, method=self.logdet_method,
                       rho_min=rho_lower, rho_max=rho_upper)

        with pm.Model(coords=self._model_coords()) as model:
            rho = pm.Uniform("rho", lower=rho_lower, upper=rho_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

            # mu = rho*Wy + X@beta  (Wy is fixed observed data here)
            mu = rho * self._Wy + pt.dot(self._X, beta)
            pm.Normal("obs", mu=mu, sigma=sigma, observed=self._y)

            # Jacobian: log|I - rho*W|
            pm.Potential("jacobian", logdet_fn(rho))

        return model

    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        """Compute SAR direct/indirect/total effects.

        Returns
        -------
        dict
            Direct, indirect, total effects and feature names.
        """
        # SAR impacts: S(W) = (I - rho*W)^{-1}
        # Direct = mean diagonal of S * beta_k
        # Indirect = (mean row sum of S - 1) * beta_k
        # Total = mean row sum of S * beta_k
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        eigs = self._W_eigs
        mean_diag = float(np.mean((1.0 / (1.0 - rho * eigs)).real))
        if self._is_row_std:
            mean_row_sum = 1.0 / (1.0 - rho)
        else:
            mean_row_sum = float(
                np.linalg.solve(
                    np.eye(self._W_sparse.shape[0]) - rho * self._W_sparse.toarray(),
                    np.ones(self._W_sparse.shape[0]),
                ).mean()
            )
        direct = mean_diag * beta
        total = mean_row_sum * beta
        indirect = total - direct

        return {
            "direct": direct,
            "indirect": indirect,
            "total": total,
            "feature_names": self._feature_names,
        }

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Compute fitted values at posterior mean parameters.

        Returns
        -------
        np.ndarray
            Posterior-mean fitted values.
        """
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        return rho * self._Wy + self._X @ beta

    # ------------------------------------------------------------------
    # Spatial specification tests
    # ------------------------------------------------------------------

    def lm_error_test(self) -> "DiagnosticResult":
        """LM test for omitted spatial error autocorrelation.

        Tests whether residuals from the SAR fit show additional spatial
        error structure (i.e. whether SARAR or SDM might be more
        appropriate than a pure SAR model).

        Returns
        -------
        DiagnosticResult
            ``name`` : ``"lm_error"``

            ``statistic`` : float — LM statistic.

            ``pvalue`` : float — p-value under :math:`\\chi^2(1)` null.

        Notes
        -----
        H\\ :sub:`0`: no spatial error autocorrelation in OLS residuals.

        References
        ----------
        .. [1] Anselin, L. (1988). *Spatial Econometrics: Methods and Models*.
               Kluwer Academic Publishers.
        """
        from ..stats.core import lmerror
        raw = lmerror(self._y, self._X, self._W_sparse.toarray())
        return self._wrap_stats_result("lm_error", raw, "lm")

    def lm_rho_test(self) -> "DiagnosticResult":
        """LM test for the SAR spatial autoregressive parameter :math:`\\rho`.

        Tests the null hypothesis :math:`\\rho = 0` (no spatial lag on y)
        using the LM statistic derived from the OLS score.

        Returns
        -------
        DiagnosticResult
            ``name`` : ``"lm_rho"``

            ``statistic`` : float — LM statistic for :math:`\\rho`.

            ``pvalue`` : float — p-value under :math:`\\chi^2(1)` null.

        References
        ----------
        .. [1] Anselin, L. (1988). *Spatial Econometrics: Methods and Models*.
               Kluwer Academic Publishers.
        """
        from ..stats.core import lmrho
        raw = lmrho(self._y, self._X, self._W_sparse.toarray())
        return self._wrap_stats_result("lm_rho", raw, "lmrho")

    def lm_rho_robust_test(self) -> "DiagnosticResult":
        """Robust LM test for the SAR parameter :math:`\\rho`.

        A robust version of :meth:`lm_rho_test` that accounts for the
        possible presence of spatial error autocorrelation under the
        alternative, reducing size distortion when both spatial
        structures are present.

        Returns
        -------
        DiagnosticResult
            ``name`` : ``"lm_rho_robust"``

            ``statistic`` : float — robust LM statistic.

            ``pvalue`` : float — p-value under :math:`\\chi^2(1)` null.

        References
        ----------
        .. [1] Anselin, L., Bera, A. K., Florax, R., & Yoon, M. J. (1996).
               Simple diagnostic tests for spatial dependence.
               *Regional Science and Urban Economics*, 26(1), 77–104.
        """
        from ..stats.core import lmrhorob
        raw = lmrhorob(self._y, self._X, self._W_sparse.toarray())
        return self._wrap_stats_result("lm_rho_robust", raw, "lmrhorob")

    def spatial_specification_tests(self) -> dict:
        """Run a battery of spatial specification tests on OLS residuals.

        Combines Moran's I, the LM-error test, the LM-:math:`\\rho` test,
        and its robust version.  Useful for post-estimation model checking
        and for deciding whether a more complex spatial specification
        is warranted.

        Returns
        -------
        dict[str, DiagnosticResult]
            Keys: ``"moran"``, ``"lm_error"``, ``"lm_rho"``,
            ``"lm_rho_robust"``.

        See Also
        --------
        moran_test : Moran's I for residual spatial autocorrelation.
        lm_error_test : LM test for spatial error dependence.
        lm_rho_test : LM test for the SAR parameter.
        lm_rho_robust_test : Robust LM test for the SAR parameter.
        """
        return {
            "moran": self.moran_test(),
            "lm_error": self.lm_error_test(),
            "lm_rho": self.lm_rho_test(),
            "lm_rho_robust": self.lm_rho_robust_test(),
        }
