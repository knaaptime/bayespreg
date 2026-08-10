"""Equivalence pins for the PyMC-free bridge-sampling log-posterior.

:func:`~bayespecon.diagnostics._native_log_posterior.native_log_posterior`
replaces a compiled PyTensor graph with closed-form algebra, so it has to
reproduce :func:`~bayespecon.diagnostics.bayesfactor.compile_log_posterior`
*absolutely*, not up to a constant: Bayes factors compare marginal likelihoods
across models, so any offset survives into the comparison.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from bayespecon.diagnostics._native_log_posterior import native_log_posterior
from bayespecon.diagnostics.bayesfactor import compile_log_posterior
from bayespecon.models.cross_section.ols import OLS
from bayespecon.models.cross_section.sar import SAR
from bayespecon.models.cross_section.sdem import SDEM
from bayespecon.models.cross_section.sdm import SDM
from bayespecon.models.cross_section.sem import SEM
from bayespecon.models.cross_section.slx import SLX
from bayespecon.tests.helpers import make_rook_W


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    W = sp.csr_matrix(make_rook_W(5).astype(float))
    n = W.shape[0]
    X = rng.normal(size=(n, 2))
    y = sp.linalg.spsolve(
        sp.eye(n, format="csc") - 0.5 * W,
        1.0 + 2.0 * X[:, 0] - X[:, 1] + rng.normal(scale=0.5, size=n),
    )
    return y, X, W


def _fit(cls, data, **kwargs):
    y, X, W = data
    ctor = dict(y=y, X=X, **kwargs)
    if cls is not OLS:
        ctor["W"] = W
    model = cls(**ctor)
    sampler = "nuts" if kwargs.get("robust") or cls is OLS else "gibbs"
    model.fit(
        draws=120, tune=80, chains=2, sampler=sampler, progressbar=False, random_seed=1
    )
    return model


@pytest.mark.parametrize("cls", [OLS, SLX, SAR, SDM, SEM, SDEM])
class TestMatchesPyMC:
    def test_parameter_names_and_order_match(self, cls, data):
        """A different order would make the bridge estimator read blocks
        transposed, silently."""
        model = _fit(cls, data)
        _, pymc_names, pymc_info, _ = compile_log_posterior(model.pymc_model)
        _, native_names, native_info, _ = native_log_posterior(model)
        assert native_names == pymc_names
        assert native_info == pymc_info

    def test_unconstrained_draws_match(self, cls, data):
        model = _fit(cls, data)
        _, _, _, pymc_to_u = compile_log_posterior(model.pymc_model)
        _, _, _, native_to_u = native_log_posterior(model)
        post = model.inference_data.posterior
        assert np.allclose(native_to_u(post), pymc_to_u(post), rtol=0, atol=1e-12)

    def test_log_posterior_matches_absolutely(self, cls, data):
        """Machine precision, not "up to a constant" — constants matter here."""
        model = _fit(cls, data)
        pymc_fn, _, _, pymc_to_u = compile_log_posterior(model.pymc_model)
        native_fn, _, _, _ = native_log_posterior(model)
        theta = pymc_to_u(model.inference_data.posterior)[:60]
        diff = np.array([native_fn(t) - pymc_fn(t) for t in theta])
        assert np.abs(diff).max() < 1e-9


@pytest.mark.parametrize("cls", [SAR, SEM])
def test_robust_matches_within_gammaln_accuracy(cls, data):
    """Student-t agrees to ~6e-10 per observation.

    The residual is not in the algebra: it is ``scipy.special.gammaln`` versus
    PyTensor's log-gamma at the same arguments, with scipy's being the more
    accurate.  It appears as a constant offset, so it moves a log marginal
    likelihood by ``n * 6e-10`` — far below bridge sampling's Monte-Carlo error.
    """
    model = _fit(cls, data, robust=True)
    pymc_fn, _, _, pymc_to_u = compile_log_posterior(model.pymc_model)
    native_fn, _, _, _ = native_log_posterior(model)
    theta = pymc_to_u(model.inference_data.posterior)[:60]
    diff = np.array([native_fn(t) - pymc_fn(t) for t in theta])
    n = model._y.shape[0]
    assert np.abs(diff).max() < n * 1e-8
    # Constant offset, so it cancels in any ratio of posterior densities.
    assert diff.std() < 1e-12


def test_unsupported_family_returns_none():
    """``None`` means "no closed form here", so callers fall back to PyMC
    rather than treating it as a failure."""

    class NotAGaussianModel:
        pass

    assert native_log_posterior(NotAGaussianModel()) is None


def test_bridge_uses_native_path_without_building_a_pymc_graph(data):
    """The point of the exercise: a Gibbs fit must not need a PyTensor graph."""
    from bayespecon.diagnostics.bayesfactor import bayes_factor_compare_models

    models = [_fit(cls, data) for cls in (SAR, SEM)]
    assert all(m._pymc_model is None for m in models)

    bayes_factor_compare_models(models, model_labels=["SAR", "SEM"], method="bridge")

    assert all(m._pymc_model is None for m in models), (
        "bridge sampling built a PyMC graph; the native log-posterior was bypassed"
    )
