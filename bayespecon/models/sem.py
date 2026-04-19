"""Spatial Error Model (SEM).

y = X @ beta + u,  u = lambda * W @ u + epsilon,  epsilon ~ N(0, sigma^2 I)

Equivalently: (I - lambda*W)(y - X@beta) = epsilon
Likelihood: epsilon ~ N(0, sigma^2 I), plus Jacobian log|I - lambda*W|.
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from .base import SpatialModel
from ..logdet import make_logdet_fn


class SEM(SpatialModel):
    """Bayesian Spatial Error Model.

    .. math::
        y = X\\beta + u, \\quad u = \\lambda Wu + \\varepsilon, \\quad \\varepsilon \\sim N(0, \\sigma^2 I)

    Parameters
    ----------
    formula, data, y, X, W, priors, logdet_method
        See :class:`SpatialModel`.

    Priors (``priors`` dict keys)
    ------------------------------
    lam_lower, lam_upper : float, default -1, 1
        Bounds for the Uniform prior on lambda.
    beta_mu : float, default 0
    beta_sigma : float, default 1e6
    sigma_sigma : float, default 10
    """

    def _build_pymc_model(self) -> pm.Model:
        """Construct the PyMC model for SEM regression.

        Returns
        -------
        pymc.Model
            Compiled probabilistic model object.
        """
        k = self._X.shape[1]

        lam_lower = self.priors.get("lam_lower", -1.0)
        lam_upper = self.priors.get("lam_upper", 1.0)
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)

        logdet_fn = make_logdet_fn(self._W_eigs.real, method=self.logdet_method,
                       rho_min=lam_lower, rho_max=lam_upper)
        W_pt = pt.as_tensor_variable(self._W_dense)

        with pm.Model(coords=self._model_coords()) as model:
            lam = pm.Uniform("lam", lower=lam_lower, upper=lam_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

            # epsilon = (I - lambda*W)(y - X@beta) = resid - lambda*(W @ resid)
            # Avoid forming the dense (I - lam*W) matrix each step.
            resid = self._y - pt.dot(self._X, beta)
            eps = resid - lam * pt.dot(W_pt, resid)
            # PyMC does not allow symbolic expressions as `observed`; encode
            # the transformed-error likelihood directly as a Potential.
            logp_eps = pm.logp(pm.Normal.dist(mu=0.0, sigma=sigma), eps).sum()
            pm.Potential("eps_loglik", logp_eps)

            # Jacobian
            pm.Potential("jacobian", logdet_fn(lam))

        return model

    def _compute_spatial_effects(self) -> dict[str, np.ndarray]:
        """Compute SEM direct/indirect/total effects.

        Returns
        -------
        dict
            Direct, indirect, total effects and feature names.
        """
        # For SEM, spatial multiplier does not apply to X directly.
        # Direct = beta, indirect = 0, total = beta.
        beta = self._posterior_mean("beta")
        return {
            "direct": beta.copy(),
            "indirect": np.zeros_like(beta),
            "total": beta.copy(),
            "feature_names": self._feature_names,
        }

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Compute fitted values at posterior mean coefficients.

        Returns
        -------
        np.ndarray
            Posterior-mean fitted values.
        """
        beta = self._posterior_mean("beta")
        return self._X @ beta

    # ------------------------------------------------------------------
    # Spatial specification tests
    # ------------------------------------------------------------------

    def lm_lag_test(self) -> "DiagnosticResult":
        """LM test for an omitted spatially lagged dependent variable.

        From a SEM perspective, tests whether an additional lag on *y*
        (i.e. a SAR or SARAR structure) is suggested by the data.

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

    def wald_error_test(self) -> "DiagnosticResult":
        """Wald test for spatial error autocorrelation (:math:`\\lambda`).

        Tests H\\ :sub:`0`: :math:`\\lambda = 0` (no spatial error structure)
        using the Wald statistic derived from the SEM concentrated
        maximum likelihood estimate.

        Returns
        -------
        DiagnosticResult
            ``name`` : ``"wald_error"``

            ``statistic`` : float — Wald statistic.

            ``pvalue`` : float — p-value under :math:`\\chi^2(1)` null.

        References
        ----------
        .. [1] Anselin, L. (1988). *Spatial Econometrics: Methods and Models*.
               Kluwer Academic Publishers.
        """
        from ..stats.core import walds
        raw = walds(self._y, self._X, self._W_sparse.toarray())
        return self._wrap_stats_result("wald_error", raw, "wald")

    def lr_ratio_test(self) -> "DiagnosticResult":
        """Likelihood ratio test comparing SEM against OLS.

        Computes twice the log-likelihood difference between the SEM
        concentrated MLE and the OLS model.  A significant result
        supports the SEM specification over plain OLS.

        Returns
        -------
        DiagnosticResult
            ``name`` : ``"lr_ratio"``

            ``statistic`` : float — LR statistic.

            ``pvalue`` : float — p-value under :math:`\\chi^2(1)` null.

        References
        ----------
        .. [1] Anselin, L. (1988). *Spatial Econometrics: Methods and Models*.
               Kluwer Academic Publishers.
        """
        from ..stats.core import lratios
        raw = lratios(self._y, self._X, self._W_sparse.toarray())
        return self._wrap_stats_result("lr_ratio", raw, "lratio")

    def spatial_specification_tests(self) -> dict:
        """Run a battery of spatial specification tests on OLS residuals.

        Combines Moran's I, LM-lag, Wald-error, and LR tests.
        Useful for post-estimation model checking.

        Returns
        -------
        dict[str, DiagnosticResult]
            Keys: ``"moran"``, ``"lm_lag"``, ``"wald_error"``,
            ``"lr_ratio"``.

        See Also
        --------
        moran_test : Moran's I for residual spatial autocorrelation.
        lm_lag_test : LM test for omitted spatial lag.
        wald_error_test : Wald test for spatial error autocorrelation.
        lr_ratio_test : LR test of SEM vs OLS.
        """
        return {
            "moran": self.moran_test(),
            "lm_lag": self.lm_lag_test(),
            "wald_error": self.wald_error_test(),
            "lr_ratio": self.lr_ratio_test(),
        }
