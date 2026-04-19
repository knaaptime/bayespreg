"""Spatial Lag X (SLX) Model.

y = X @ beta1 + W @ X @ beta2 + epsilon,  epsilon ~ N(0, sigma^2 I)

No spatial lag on y, so no Jacobian adjustment is needed and NUTS
converges without difficulty.
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from .base import SpatialModel


class SLX(SpatialModel):
    """Bayesian SLX (Spatial Lag X) model.

    .. math::
        y = X\\beta_1 + WX\\beta_2 + \\varepsilon, \\quad \\varepsilon \\sim N(0, \\sigma^2 I)

    Parameters
    ----------
    formula, data, y, X, W, priors, logdet_method, w_vars
        See :class:`SpatialModel`. Use ``w_vars`` to restrict which X columns
        are spatially lagged.

    Priors (``priors`` dict keys)
    ------------------------------
    beta_mu : float, default 0
        Prior mean for all beta coefficients.
    beta_sigma : float, default 1e6
        Prior std for all beta coefficients (diffuse Normal).
    sigma_sigma : float, default 10
        Scale for HalfNormal prior on sigma.
    """

    def _beta_names(self) -> list[str]:
        return self._feature_names + [f"W*{name}" for name in self._wx_feature_names]

    def _build_pymc_model(self) -> pm.Model:
        """Construct the PyMC model for SLX regression.

        Returns
        -------
        pymc.Model
            Compiled probabilistic model object.
        """
        n, k = self._X.shape
        # Combine X and WX into one design matrix; beta covers both
        Z = np.hstack([self._X, self._WX])  # (n, 2k)

        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)

        with pm.Model(coords=self._model_coords()) as model:
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

            mu = pt.dot(Z, beta)
            pm.Normal("obs", mu=mu, sigma=sigma, observed=self._y)

        return model

    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        """Compute SLX direct/indirect/total effects.

        Returns
        -------
        dict
            Direct, indirect, total effects and feature names.
        """
        k = self._X.shape[1]
        kw = self._WX.shape[1]
        beta = self._posterior_mean("beta")
        beta1 = beta[:k]   # coefficients on X
        beta2 = beta[k:k + kw]   # coefficients on WX (excluding intercept-like terms)

        # For SLX: S_k = d y / d X_k = beta1_k * I + beta2_k * W
        # Direct   = mean(diag(S_k))
        # Total    = mean(row_sums(S_k))
        # Indirect = Total - Direct
        mean_diag_w = float(self._W_sparse.diagonal().mean())
        mean_row_sum_w = float(self._W_sparse.sum() / self._W_sparse.shape[0])

        direct = beta1[self._wx_column_indices] + beta2 * mean_diag_w
        total = beta1[self._wx_column_indices] + beta2 * mean_row_sum_w
        indirect = total - direct

        return {
            "direct": direct,
            "indirect": indirect,
            "total": total,
            "feature_names": self._wx_feature_names,
        }

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Compute fitted values at posterior mean coefficients.

        Returns
        -------
        np.ndarray
            Posterior-mean fitted values.
        """
        k = self._X.shape[1]
        beta = self._posterior_mean("beta")
        Z = np.hstack([self._X, self._WX])
        if beta.shape[0] != 2 * k:
            raise ValueError("Unexpected beta dimension for SLX fitted mean.")
        return Z @ beta

    # ------------------------------------------------------------------
    # Spatial specification tests
    # ------------------------------------------------------------------

    def lm_error_test(self) -> "DiagnosticResult":
        """LM test for omitted spatial error autocorrelation.

        Tests whether the SLX residuals exhibit spatial error dependence,
        suggesting that a SDEM or full SEM specification may be more
        appropriate.

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

    def lm_lag_test(self) -> "DiagnosticResult":
        """LM test for an omitted spatially lagged dependent variable.

        Tests whether adding a lag on *y* (SAR or SDM structure) would
        significantly improve the SLX specification.

        Returns
        -------
        DiagnosticResult
            ``name`` : ``"lm_lag"``

            ``statistic`` : float — LM statistic.

            ``pvalue`` : float — p-value under :math:`\\chi^2(1)` null.

        Notes
        -----
        H\\ :sub:`0`: no omitted spatial lag of the dependent variable.

        References
        ----------
        .. [1] Anselin, L. (1988). *Spatial Econometrics: Methods and Models*.
               Kluwer Academic Publishers.
        """
        from ..stats.core import lmlag
        raw = lmlag(self._y, self._X, self._W_sparse.toarray())
        return self._wrap_stats_result("lm_lag", raw, "lm")

    def spatial_specification_tests(self) -> dict:
        """Run a battery of spatial specification tests on OLS residuals.

        Combines Moran's I, LM-error, and LM-lag tests.

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
