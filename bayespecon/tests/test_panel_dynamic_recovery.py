"""Parameter recovery tests for dynamic panel model classes.

Each test generates balanced panel data from known parameters, fits the
model, and asserts the posterior mean is within tolerance of the true value.

**Design notes**

Dynamic panel models with ``model=1`` (unit FE demeaning) suffer from
Nickell bias: the demeaned lagged dependent variable is correlated with
the demeaned error, biasing φ toward zero.  To obtain clean parameter
recovery, we use ``model=0`` (pooled) and generate data **without** unit
effects (``sigma_alpha=0``), so the DGP matches the model specification
exactly.

Run with::

    pytest tests/test_panel_dynamic_recovery.py -m slow -v
"""

from __future__ import annotations

import numpy as np
import pytest

from bayespecon import (
    DLMPanelFE,
    SDMRPanelFE,
    SDMUPanelFE,
    SARPanelDEDynamic,
    SEMPanelDEDynamic,
    SDEMPanelDEDynamic,
    SLXPanelDEDynamic,
)
from .helpers import (
    PANEL_N,
    PANEL_T,
    SAMPLE_KWARGS,
    make_panel_dlm_data,
    make_panel_sdmr_data,
    make_panel_sdmu_data,
    make_panel_sar_dynamic_data,
    make_panel_sem_dynamic_data,
    make_panel_sdem_dynamic_data,
    make_panel_slx_dynamic_data,
)

pytestmark = pytest.mark.slow

# True parameters
PHI_TRUE = 0.4
RHO_TRUE = 0.3
LAM_TRUE = 0.3
THETA_TRUE = -0.1
BETA_TRUE = np.array([1.0, 2.0])
SIGMA_TRUE = 1.0
# sigma_alpha=0 so DGP matches the pooled (model=0) specification exactly
SIGMA_ALPHA_TRUE = 0.0

# Tolerances — dynamic panels with pooled specification
ABS_TOL_PHI = 0.25
ABS_TOL_SPATIAL = 0.35
ABS_TOL_BETA = 0.50
ABS_TOL_THETA = 0.40


# ---------------------------------------------------------------------------
# DLM Panel FE  (non-spatial dynamic)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_dlm_panel_fe_recovers_phi(rng, W_panel_dense, W_panel_graph):
    """DLMPanelFE posterior mean of phi should be close to the true value."""
    y, X, _ = make_panel_dlm_data(
        rng, W_panel_dense, PANEL_N, PANEL_T,
        phi=PHI_TRUE, beta=BETA_TRUE, sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = DLMPanelFE(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    phi_hat = float(idata.posterior["phi"].mean())
    assert abs(phi_hat - PHI_TRUE) < ABS_TOL_PHI, (
        f"DLMPanelFE phi: expected ≈{PHI_TRUE}, got {phi_hat:.3f}"
    )


@pytest.mark.slow
def test_dlm_panel_fe_recovers_beta(rng, W_panel_dense, W_panel_graph):
    """DLMPanelFE posterior means of beta should match truth."""
    y, X, _ = make_panel_dlm_data(
        rng, W_panel_dense, PANEL_N, PANEL_T,
        phi=PHI_TRUE, beta=BETA_TRUE, sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = DLMPanelFE(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    for j, (bhat, btrue) in enumerate(zip(beta_hat, BETA_TRUE)):
        assert abs(bhat - btrue) < ABS_TOL_BETA, (
            f"DLMPanelFE beta[{j}]: expected ≈{btrue}, got {bhat:.3f}"
        )


# ---------------------------------------------------------------------------
# SDMR Panel FE  (restricted SDM dynamic)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_sdmr_panel_fe_recovers_rho(rng, W_panel_dense, W_panel_graph):
    """SDMRPanelFE posterior mean of rho should be close to the true value."""
    y, X, _ = make_panel_sdmr_data(
        rng, W_panel_dense, PANEL_N, PANEL_T,
        rho=RHO_TRUE, phi=PHI_TRUE, beta=BETA_TRUE, sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SDMRPanelFE(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    rho_hat = float(idata.posterior["rho"].mean())
    assert abs(rho_hat - RHO_TRUE) < ABS_TOL_SPATIAL, (
        f"SDMRPanelFE rho: expected ≈{RHO_TRUE}, got {rho_hat:.3f}"
    )


@pytest.mark.slow
def test_sdmr_panel_fe_recovers_phi(rng, W_panel_dense, W_panel_graph):
    """SDMRPanelFE posterior mean of phi should be close to the true value."""
    y, X, _ = make_panel_sdmr_data(
        rng, W_panel_dense, PANEL_N, PANEL_T,
        rho=RHO_TRUE, phi=PHI_TRUE, beta=BETA_TRUE, sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SDMRPanelFE(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    phi_hat = float(idata.posterior["phi"].mean())
    assert abs(phi_hat - PHI_TRUE) < ABS_TOL_PHI, (
        f"SDMRPanelFE phi: expected ≈{PHI_TRUE}, got {phi_hat:.3f}"
    )


@pytest.mark.slow
def test_sdmr_panel_fe_recovers_beta(rng, W_panel_dense, W_panel_graph):
    """SDMRPanelFE posterior means of beta should match truth."""
    y, X, _ = make_panel_sdmr_data(
        rng, W_panel_dense, PANEL_N, PANEL_T,
        rho=RHO_TRUE, phi=PHI_TRUE, beta=BETA_TRUE, sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SDMRPanelFE(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    for j, (bhat, btrue) in enumerate(zip(beta_hat, BETA_TRUE)):
        assert abs(bhat - btrue) < ABS_TOL_BETA, (
            f"SDMRPanelFE beta[{j}]: expected ≈{btrue}, got {bhat:.3f}"
        )


# ---------------------------------------------------------------------------
# SDMU Panel FE  (unrestricted SDM dynamic)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_sdmu_panel_fe_recovers_rho(rng, W_panel_dense, W_panel_graph):
    """SDMUPanelFE posterior mean of rho should be close to the true value."""
    y, X, _ = make_panel_sdmu_data(
        rng, W_panel_dense, PANEL_N, PANEL_T,
        rho=RHO_TRUE, phi=PHI_TRUE, theta=THETA_TRUE,
        beta=BETA_TRUE, sigma=SIGMA_TRUE, sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SDMUPanelFE(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    rho_hat = float(idata.posterior["rho"].mean())
    assert abs(rho_hat - RHO_TRUE) < ABS_TOL_SPATIAL, (
        f"SDMUPanelFE rho: expected ≈{RHO_TRUE}, got {rho_hat:.3f}"
    )


@pytest.mark.slow
def test_sdmu_panel_fe_recovers_phi(rng, W_panel_dense, W_panel_graph):
    """SDMUPanelFE posterior mean of phi should be close to the true value."""
    y, X, _ = make_panel_sdmu_data(
        rng, W_panel_dense, PANEL_N, PANEL_T,
        rho=RHO_TRUE, phi=PHI_TRUE, theta=THETA_TRUE,
        beta=BETA_TRUE, sigma=SIGMA_TRUE, sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SDMUPanelFE(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    phi_hat = float(idata.posterior["phi"].mean())
    assert abs(phi_hat - PHI_TRUE) < ABS_TOL_PHI, (
        f"SDMUPanelFE phi: expected ≈{PHI_TRUE}, got {phi_hat:.3f}"
    )


@pytest.mark.slow
def test_sdmu_panel_fe_recovers_theta(rng, W_panel_dense, W_panel_graph):
    """SDMUPanelFE posterior mean of theta should be close to the true value."""
    y, X, _ = make_panel_sdmu_data(
        rng, W_panel_dense, PANEL_N, PANEL_T,
        rho=RHO_TRUE, phi=PHI_TRUE, theta=THETA_TRUE,
        beta=BETA_TRUE, sigma=SIGMA_TRUE, sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SDMUPanelFE(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    theta_hat = float(idata.posterior["theta"].mean())
    assert abs(theta_hat - THETA_TRUE) < ABS_TOL_THETA, (
        f"SDMUPanelFE theta: expected ≈{THETA_TRUE}, got {theta_hat:.3f}"
    )


@pytest.mark.slow
def test_sdmu_panel_fe_recovers_beta(rng, W_panel_dense, W_panel_graph):
    """SDMUPanelFE posterior means of beta should match truth."""
    y, X, _ = make_panel_sdmu_data(
        rng, W_panel_dense, PANEL_N, PANEL_T,
        rho=RHO_TRUE, phi=PHI_TRUE, theta=THETA_TRUE,
        beta=BETA_TRUE, sigma=SIGMA_TRUE, sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SDMUPanelFE(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    for j, (bhat, btrue) in enumerate(zip(beta_hat, BETA_TRUE)):
        assert abs(bhat - btrue) < ABS_TOL_BETA, (
            f"SDMUPanelFE beta[{j}]: expected ≈{btrue}, got {bhat:.3f}"
        )


# ---------------------------------------------------------------------------
# SAR Panel DE Dynamic  (SAR with lagged DV, no WX)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_sar_panel_de_dynamic_recovers_rho(rng, W_panel_dense, W_panel_graph):
    """SARPanelDEDynamic posterior mean of rho should be close to the true value."""
    y, X, _ = make_panel_sar_dynamic_data(
        rng, W_panel_dense, PANEL_N, PANEL_T,
        rho=RHO_TRUE, phi=PHI_TRUE, beta=BETA_TRUE, sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SARPanelDEDynamic(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    rho_hat = float(idata.posterior["rho"].mean())
    assert abs(rho_hat - RHO_TRUE) < ABS_TOL_SPATIAL, (
        f"SARPanelDEDynamic rho: expected ≈{RHO_TRUE}, got {rho_hat:.3f}"
    )


@pytest.mark.slow
def test_sar_panel_de_dynamic_recovers_phi(rng, W_panel_dense, W_panel_graph):
    """SARPanelDEDynamic posterior mean of phi should be close to the true value."""
    y, X, _ = make_panel_sar_dynamic_data(
        rng, W_panel_dense, PANEL_N, PANEL_T,
        rho=RHO_TRUE, phi=PHI_TRUE, beta=BETA_TRUE, sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SARPanelDEDynamic(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    phi_hat = float(idata.posterior["phi"].mean())
    assert abs(phi_hat - PHI_TRUE) < ABS_TOL_PHI, (
        f"SARPanelDEDynamic phi: expected ≈{PHI_TRUE}, got {phi_hat:.3f}"
    )


@pytest.mark.slow
def test_sar_panel_de_dynamic_recovers_beta(rng, W_panel_dense, W_panel_graph):
    """SARPanelDEDynamic posterior means of beta should match truth."""
    y, X, _ = make_panel_sar_dynamic_data(
        rng, W_panel_dense, PANEL_N, PANEL_T,
        rho=RHO_TRUE, phi=PHI_TRUE, beta=BETA_TRUE, sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SARPanelDEDynamic(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    for j, (bhat, btrue) in enumerate(zip(beta_hat, BETA_TRUE)):
        assert abs(bhat - btrue) < ABS_TOL_BETA, (
            f"SARPanelDEDynamic beta[{j}]: expected ≈{btrue}, got {bhat:.3f}"
        )


# ---------------------------------------------------------------------------
# SEM Panel DE Dynamic  (SEM with lagged DV)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_sem_panel_de_dynamic_recovers_lam(rng, W_panel_dense, W_panel_graph):
    """SEMPanelDEDynamic posterior mean of lam should be close to the true value."""
    y, X, _ = make_panel_sem_dynamic_data(
        rng, W_panel_dense, PANEL_N, PANEL_T,
        lam=LAM_TRUE, phi=PHI_TRUE, beta=BETA_TRUE, sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SEMPanelDEDynamic(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    lam_hat = float(idata.posterior["lam"].mean())
    assert abs(lam_hat - LAM_TRUE) < ABS_TOL_SPATIAL, (
        f"SEMPanelDEDynamic lam: expected ≈{LAM_TRUE}, got {lam_hat:.3f}"
    )


@pytest.mark.slow
def test_sem_panel_de_dynamic_recovers_phi(rng, W_panel_dense, W_panel_graph):
    """SEMPanelDEDynamic posterior mean of phi should be close to the true value."""
    y, X, _ = make_panel_sem_dynamic_data(
        rng, W_panel_dense, PANEL_N, PANEL_T,
        lam=LAM_TRUE, phi=PHI_TRUE, beta=BETA_TRUE, sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SEMPanelDEDynamic(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    phi_hat = float(idata.posterior["phi"].mean())
    assert abs(phi_hat - PHI_TRUE) < ABS_TOL_PHI, (
        f"SEMPanelDEDynamic phi: expected ≈{PHI_TRUE}, got {phi_hat:.3f}"
    )


@pytest.mark.slow
def test_sem_panel_de_dynamic_recovers_beta(rng, W_panel_dense, W_panel_graph):
    """SEMPanelDEDynamic posterior means of beta should match truth."""
    y, X, _ = make_panel_sem_dynamic_data(
        rng, W_panel_dense, PANEL_N, PANEL_T,
        lam=LAM_TRUE, phi=PHI_TRUE, beta=BETA_TRUE, sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SEMPanelDEDynamic(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    for j, (bhat, btrue) in enumerate(zip(beta_hat, BETA_TRUE)):
        assert abs(bhat - btrue) < ABS_TOL_BETA, (
            f"SEMPanelDEDynamic beta[{j}]: expected ≈{btrue}, got {bhat:.3f}"
        )


# ---------------------------------------------------------------------------
# SDEM Panel DE Dynamic  (SDEM with lagged DV)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_sdem_panel_de_dynamic_recovers_lam(rng, W_panel_dense, W_panel_graph):
    """SDEMPanelDEDynamic posterior mean of lam should be close to the true value."""
    y, X, _ = make_panel_sdem_dynamic_data(
        rng, W_panel_dense, PANEL_N, PANEL_T,
        lam=LAM_TRUE, phi=PHI_TRUE, beta=BETA_TRUE, sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SDEMPanelDEDynamic(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    lam_hat = float(idata.posterior["lam"].mean())
    assert abs(lam_hat - LAM_TRUE) < ABS_TOL_SPATIAL, (
        f"SDEMPanelDEDynamic lam: expected ≈{LAM_TRUE}, got {lam_hat:.3f}"
    )


@pytest.mark.slow
def test_sdem_panel_de_dynamic_recovers_phi(rng, W_panel_dense, W_panel_graph):
    """SDEMPanelDEDynamic posterior mean of phi should be close to the true value."""
    y, X, _ = make_panel_sdem_dynamic_data(
        rng, W_panel_dense, PANEL_N, PANEL_T,
        lam=LAM_TRUE, phi=PHI_TRUE, beta=BETA_TRUE, sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SDEMPanelDEDynamic(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    phi_hat = float(idata.posterior["phi"].mean())
    assert abs(phi_hat - PHI_TRUE) < ABS_TOL_PHI, (
        f"SDEMPanelDEDynamic phi: expected ≈{PHI_TRUE}, got {phi_hat:.3f}"
    )


@pytest.mark.slow
def test_sdem_panel_de_dynamic_recovers_beta(rng, W_panel_dense, W_panel_graph):
    """SDEMPanelDEDynamic posterior means of beta should match truth."""
    y, X, _ = make_panel_sdem_dynamic_data(
        rng, W_panel_dense, PANEL_N, PANEL_T,
        lam=LAM_TRUE, phi=PHI_TRUE, beta=BETA_TRUE, sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SDEMPanelDEDynamic(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    for j, (bhat, btrue) in enumerate(zip(beta_hat, BETA_TRUE)):
        assert abs(bhat - btrue) < ABS_TOL_BETA, (
            f"SDEMPanelDEDynamic beta[{j}]: expected ≈{btrue}, got {bhat:.3f}"
        )


# ---------------------------------------------------------------------------
# SLX Panel DE Dynamic  (SLX with lagged DV)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_slx_panel_de_dynamic_recovers_phi(rng, W_panel_dense, W_panel_graph):
    """SLXPanelDEDynamic posterior mean of phi should be close to the true value."""
    y, X, _ = make_panel_slx_dynamic_data(
        rng, W_panel_dense, PANEL_N, PANEL_T,
        phi=PHI_TRUE, beta=BETA_TRUE, sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SLXPanelDEDynamic(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    phi_hat = float(idata.posterior["phi"].mean())
    assert abs(phi_hat - PHI_TRUE) < ABS_TOL_PHI, (
        f"SLXPanelDEDynamic phi: expected ≈{PHI_TRUE}, got {phi_hat:.3f}"
    )


@pytest.mark.slow
def test_slx_panel_de_dynamic_recovers_beta(rng, W_panel_dense, W_panel_graph):
    """SLXPanelDEDynamic posterior means of beta should match truth."""
    y, X, _ = make_panel_slx_dynamic_data(
        rng, W_panel_dense, PANEL_N, PANEL_T,
        phi=PHI_TRUE, beta=BETA_TRUE, sigma=SIGMA_TRUE,
        sigma_alpha=SIGMA_ALPHA_TRUE,
    )
    model = SLXPanelDEDynamic(y=y, X=X, W=W_panel_graph, N=PANEL_N, T=PANEL_T, model=0)
    idata = model.fit(**SAMPLE_KWARGS)
    beta_hat = idata.posterior["beta"].mean(("chain", "draw")).values
    for j, (bhat, btrue) in enumerate(zip(beta_hat, BETA_TRUE)):
        assert abs(bhat - btrue) < ABS_TOL_BETA, (
            f"SLXPanelDEDynamic beta[{j}]: expected ≈{btrue}, got {bhat:.3f}"
        )