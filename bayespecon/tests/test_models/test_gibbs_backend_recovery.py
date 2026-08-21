"""Parameterized backend recovery tests for Gibbs-sampler models.

Every model that supports ``gibbs_backend`` ∈ {numpy, jax} is exercised
**on both backends** via ``pytest.parametrize``, ensuring the JAX and
NumPy paths produce statistically equivalent posteriors that recover the
true DGP parameters.  Grid sizes, true parameters, tolerances, and
sampling settings match the existing per-model recovery test files
exactly — these are **not** loose.

The bug this catches: the JAX Krylov path previously diverged near the
ρ boundary (|ρ| ≈ 1) because it used a fixed ``krylov_dmax`` that
exceeded the Neumann series' convergence radius, producing spurious
density barriers that trapped the sampler — yielding posteriors that
disagreed with the NumPy path.

Run with::

    pytest -m recovery -v                        # recovery tests only
    pytest test_gibbs_backend_recovery.py -v     # this file only
"""

from __future__ import annotations

import numpy as np
import pytest

from bayespecon import dgp
from bayespecon.tests.helpers import (
    W_to_graph,
    make_rook_W,
    make_sar_logit_data,
    make_sar_logit_structural_data,
    make_sem_logit_data,
)

pytestmark = [pytest.mark.slow, pytest.mark.recovery]

BACKENDS = ["numpy", "jax"]


# ---------------------------------------------------------------------------
# Logit models: 30×30 grid (n=900), matching test_spatial_logit_recovery.py
# NB/ZINB models: 32×32 grid (n=1024), matching test_sar_negbin_latent.py
# ---------------------------------------------------------------------------
SIDE_LOGIT = 30
SIDE_NB = 32

# True parameters — logit (matching test_spatial_logit_recovery.py)
RHO_LOGIT = 0.4
LAM_LOGIT = 0.4
BETA_LOGIT = np.array([0.3, 1.0])

# True parameters — NB (matching test_sar_negbin_latent.py)
RHO_NB = 0.4
BETA_NB = np.array([0.5, 0.8])
ALPHA_NB = 2.0
SIGMA2_NB = 0.5

# True parameters — ZINB (matching test_sar_zinb_recovery.py)
RHO_ZINB = 0.4
LAM_ZINB = 0.3
BETA_ZINB = np.array([1.0, 0.6])
GAMMA_ZINB = np.array([0.3, 1.0])
ALPHA_ZINB = 2.0

# Sampling
DRAWS = 1000
TUNE = 1000
CHAINS = 2

# Tolerances — matching the per-model recovery tests exactly
TOL_SPATIAL_LOGIT = 0.15
TOL_BETA_LOGIT = 0.25
TOL_BETA_INTERCEPT_SEM = 0.50

TOL_SPATIAL_NB = 0.20
TOL_BETA_NB = 0.40
TOL_ALPHA_NB = 1.5

TOL_SPATIAL_ZINB = 0.20
TOL_LAM_ZINB = 0.25
TOL_BETA_ZINB = 0.40
TOL_GAMMA_ZINB = 0.40
TOL_ALPHA_ZINB = 1.5


def _fit_kwargs(backend):
    """Common fit kwargs, with n_jobs only for the NumPy backend."""
    kw = dict(
        draws=DRAWS,
        tune=TUNE,
        chains=CHAINS,
        random_seed=42,
        progressbar=False,
    )
    if backend == "numpy":
        kw["n_jobs"] = 1
    return kw


# ======================================================================
# SARLogit (reduced-form)
# ======================================================================


@pytest.fixture(scope="module")
def sar_logit_data():
    """Reduced-form SAR-logit data (noise-free DGP) on a 30×30 rook grid."""
    rng = np.random.default_rng(42)
    W_dense = make_rook_W(SIDE_LOGIT)
    y, X = make_sar_logit_data(rng, W_dense, rho=RHO_LOGIT, beta=BETA_LOGIT)
    return y, X, W_to_graph(W_dense)


@pytest.mark.parametrize("backend", BACKENDS)
def test_sar_logit_backend_recovery(sar_logit_data, backend):
    """SARLogit recovers ρ and β on both backends."""
    from bayespecon.models import SARLogit

    y, X, W = sar_logit_data
    idata = SARLogit(y=y, X=X, W=W).fit(gibbs_backend=backend, **_fit_kwargs(backend))

    rho_hat = float(idata.posterior["rho"].mean())
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values

    assert abs(rho_hat - RHO_LOGIT) < TOL_SPATIAL_LOGIT, (
        f"SARLogit[{backend}] rho: {rho_hat:.3f} vs {RHO_LOGIT}"
    )
    for j, (bh, bt) in enumerate(zip(beta_hat, BETA_LOGIT)):
        assert abs(bh - bt) < TOL_BETA_LOGIT, (
            f"SARLogit[{backend}] beta[{j}]: {bh:.3f} vs {bt}"
        )


# ======================================================================
# SARLogitStructural (latent-field)
# ======================================================================


@pytest.fixture(scope="module")
def sar_logit_struct_data():
    """Structural-form SAR-logit data (with noise) on a 30×30 rook grid."""
    rng = np.random.default_rng(42)
    W_dense = make_rook_W(SIDE_LOGIT)
    y, X = make_sar_logit_structural_data(rng, W_dense, rho=RHO_LOGIT, beta=BETA_LOGIT)
    return y, X, W_to_graph(W_dense)


@pytest.mark.parametrize("backend", BACKENDS)
def test_sar_logit_struct_backend_recovery(sar_logit_struct_data, backend):
    """SARLogitStructural recovers ρ and β on both backends."""
    from bayespecon.models import SARLogitStructural

    y, X, W = sar_logit_struct_data
    idata = SARLogitStructural(y=y, X=X, W=W).fit(
        gibbs_backend=backend, **_fit_kwargs(backend)
    )

    rho_hat = float(idata.posterior["rho"].mean())
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values

    assert abs(rho_hat - RHO_LOGIT) < TOL_SPATIAL_LOGIT, (
        f"SARLogitStructural[{backend}] rho: {rho_hat:.3f} vs {RHO_LOGIT}"
    )
    for j, (bh, bt) in enumerate(zip(beta_hat, BETA_LOGIT)):
        assert abs(bh - bt) < TOL_BETA_LOGIT, (
            f"SARLogitStructural[{backend}] beta[{j}]: {bh:.3f} vs {bt}"
        )


# ======================================================================
# SEMLogit (spatial error)
# ======================================================================


@pytest.fixture(scope="module")
def sem_logit_data():
    """SEM-logit data on a 30×30 rook grid."""
    rng = np.random.default_rng(42)
    W_dense = make_rook_W(SIDE_LOGIT)
    y, X = make_sem_logit_data(rng, W_dense, lam=LAM_LOGIT, beta=BETA_LOGIT)
    return y, X, W_to_graph(W_dense)


@pytest.mark.parametrize("backend", BACKENDS)
def test_sem_logit_backend_recovery(sem_logit_data, backend):
    """SEMLogit recovers λ and β on both backends."""
    from bayespecon.models import SEMLogit

    y, X, W = sem_logit_data
    idata = SEMLogit(y=y, X=X, W=W).fit(gibbs_backend=backend, **_fit_kwargs(backend))

    lam_hat = float(idata.posterior["lam"].mean())
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values

    assert abs(lam_hat - LAM_LOGIT) < TOL_SPATIAL_LOGIT, (
        f"SEMLogit[{backend}] lam: {lam_hat:.3f} vs {LAM_LOGIT}"
    )
    for j, (bh, bt) in enumerate(zip(beta_hat, BETA_LOGIT)):
        tol = TOL_BETA_INTERCEPT_SEM if j == 0 else TOL_BETA_LOGIT
        assert abs(bh - bt) < tol, f"SEMLogit[{backend}] beta[{j}]: {bh:.3f} vs {bt}"


# ======================================================================
# SARNegBin (reduced-form, sigma2=0)
# ======================================================================


@pytest.fixture(scope="module")
def sar_negbin_data():
    """Reduced-form SAR-NB data (deterministic latent, sigma2=0) on a 32×32 rook grid."""
    rng = np.random.default_rng(42)
    W = W_to_graph(make_rook_W(SIDE_NB))
    return dgp.simulate_sar_negbin(
        W=W,
        rho=RHO_NB,
        beta=BETA_NB,
        alpha=ALPHA_NB,
        sigma2=0.0,
        rng=rng,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_sar_negbin_backend_recovery(sar_negbin_data, backend):
    """SARNegBin recovers ρ, β, and α on both backends."""
    from bayespecon.models import SARNegBin

    d = sar_negbin_data
    W = W_to_graph(make_rook_W(SIDE_NB))
    idata = SARNegBin(y=d["y"], X=d["X"], W=W).fit(
        gibbs_backend=backend, **_fit_kwargs(backend)
    )

    rho_hat = float(idata.posterior["rho"].mean())
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    alpha_hat = float(idata.posterior["alpha"].mean())

    assert abs(rho_hat - RHO_NB) < TOL_SPATIAL_NB, (
        f"SARNegBin[{backend}] rho: {rho_hat:.3f} vs {RHO_NB}"
    )
    for j, (bh, bt) in enumerate(zip(beta_hat, BETA_NB)):
        assert abs(bh - bt) < TOL_BETA_NB, (
            f"SARNegBin[{backend}] beta[{j}]: {bh:.3f} vs {bt}"
        )
    assert abs(alpha_hat - ALPHA_NB) < TOL_ALPHA_NB, (
        f"SARNegBin[{backend}] alpha: {alpha_hat:.3f} vs {ALPHA_NB}"
    )


# ======================================================================
# SARNegBinStructural (latent-field, sigma2>0)
# ======================================================================


@pytest.fixture(scope="module")
def sar_negbin_struct_data():
    """Structural-form SAR-NB data (with latent noise, sigma2=0.5) on a 32×32 rook grid."""
    rng = np.random.default_rng(42)
    W = W_to_graph(make_rook_W(SIDE_NB))
    return dgp.simulate_sar_negbin(
        W=W,
        rho=RHO_NB,
        beta=BETA_NB,
        alpha=ALPHA_NB,
        sigma2=SIGMA2_NB,
        rng=rng,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_sar_negbin_struct_backend_recovery(sar_negbin_struct_data, backend):
    """SARNegBinStructural recovers ρ, β, α on both backends."""
    from bayespecon.models import SARNegBinStructural

    d = sar_negbin_struct_data
    W = W_to_graph(make_rook_W(SIDE_NB))
    idata = SARNegBinStructural(y=d["y"], X=d["X"], W=W).fit(
        gibbs_backend=backend, **_fit_kwargs(backend)
    )

    rho_hat = float(idata.posterior["rho"].mean())
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    alpha_hat = float(idata.posterior["alpha"].mean())

    assert abs(rho_hat - RHO_NB) < TOL_SPATIAL_NB, (
        f"SARNegBinStructural[{backend}] rho: {rho_hat:.3f} vs {RHO_NB}"
    )
    for j, (bh, bt) in enumerate(zip(beta_hat, BETA_NB)):
        assert abs(bh - bt) < TOL_BETA_NB, (
            f"SARNegBinStructural[{backend}] beta[{j}]: {bh:.3f} vs {bt}"
        )
    assert abs(alpha_hat - ALPHA_NB) < TOL_ALPHA_NB, (
        f"SARNegBinStructural[{backend}] alpha: {alpha_hat:.3f} vs {ALPHA_NB}"
    )


# ======================================================================
# SARZINB (zero-inflated NB)
# ======================================================================


@pytest.fixture(scope="module")
def sar_zinb_data():
    """ZINB data on a 32×32 rook grid."""
    rng = np.random.default_rng(42)
    W = W_to_graph(make_rook_W(SIDE_NB))
    return dgp.simulate_sar_zinb(
        W=W,
        rho=RHO_ZINB,
        lam=LAM_ZINB,
        beta=BETA_ZINB,
        gamma=GAMMA_ZINB,
        alpha=ALPHA_ZINB,
        rng=rng,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_sar_zinb_backend_recovery(sar_zinb_data, backend):
    """SARZINB recovers ρ, λ, β, γ, α on both backends."""
    from bayespecon.models import SARZINB

    d = sar_zinb_data
    W = W_to_graph(make_rook_W(SIDE_NB))
    idata = SARZINB(y=d["y"], X=d["X"], Z=d["Z"], W=W).fit(
        gibbs_backend=backend, **_fit_kwargs(backend)
    )

    rho_hat = float(idata.posterior["rho"].mean())
    lam_hat = float(idata.posterior["lam"].mean())
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    gamma_hat = idata.posterior["gamma"].mean(("chain", "draw")).values
    alpha_hat = float(idata.posterior["alpha"].mean())

    assert abs(rho_hat - RHO_ZINB) < TOL_SPATIAL_ZINB, (
        f"SARZINB[{backend}] rho: {rho_hat:.3f} vs {RHO_ZINB}"
    )
    assert abs(lam_hat - LAM_ZINB) < TOL_LAM_ZINB, (
        f"SARZINB[{backend}] lam: {lam_hat:.3f} vs {LAM_ZINB}"
    )
    for j, (bh, bt) in enumerate(zip(beta_hat, BETA_ZINB)):
        assert abs(bh - bt) < TOL_BETA_ZINB, (
            f"SARZINB[{backend}] beta[{j}]: {bh:.3f} vs {bt}"
        )
    for j, (gh, gt) in enumerate(zip(gamma_hat, GAMMA_ZINB)):
        assert abs(gh - gt) < TOL_GAMMA_ZINB, (
            f"SARZINB[{backend}] gamma[{j}]: {gh:.3f} vs {gt}"
        )
    assert abs(alpha_hat - ALPHA_ZINB) < TOL_ALPHA_ZINB, (
        f"SARZINB[{backend}] alpha: {alpha_hat:.3f} vs {ALPHA_ZINB}"
    )


# ======================================================================
# Cross-backend mixing quality (catches the original Krylov divergence bug)
# ======================================================================


@pytest.mark.parametrize("backend", BACKENDS)
def test_sar_logit_backend_mixing(sar_logit_data, backend):
    """SARLogit ρ has good ESS and R̂ on both backends (not stuck at boundary)."""
    import arviz as az

    from bayespecon.models import SARLogit

    y, X, W = sar_logit_data
    idata = SARLogit(y=y, X=X, W=W).fit(gibbs_backend=backend, **_fit_kwargs(backend))

    summ = az.summary(idata, var_names=["rho"])
    ess = float(summ.loc["rho", "ess_bulk"])
    rhat = float(summ.loc["rho", "r_hat"])

    # The JAX bug caused ESS ≈ 7 and R̂ ≈ 1.5 (stuck at ρ ≈ −0.9).
    assert ess > 100, f"SARLogit[{backend}] ESS={ess:.0f} too low (sampler stuck?)"
    assert rhat < 1.1, f"SARLogit[{backend}] R̂={rhat:.3f} too high (chains not mixed)"
