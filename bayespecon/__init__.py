"""Bayesian spatial econometric models and diagnostics.

The package exposes cross-sectional and panel spatial regression model
classes and Bayesian specification tests.

Examples
--------
Import a model class directly from the package namespace::

        from bayespecon import SAR
"""

from . import dgp
from .graph import (
    destination_weights,
    flow_design_matrix,
    flow_design_matrix_with_orig,
    flow_weight_matrices,
    FlowDesignMatrix,
    network_weights,
    origin_weights,
)
from .models.flow import SAR_Flow, SAR_Flow_Separable, PoissonFlow, PoissonFlow_Separable
from .dgp.flows import generate_flow_data, generate_poisson_flow_data
from .ops import SparseFlowSolveOp, kron_solve_matrix, kron_solve_vec
from .diagnostics import (
    BayesianLMTestResult,
    bayesian_lm_lag_test,
    bayesian_lm_error_test,
    bayesian_lm_wx_test,
    bayesian_lm_sdm_joint_test,
    bayesian_lm_slx_error_joint_test,
    bayesian_robust_lm_lag_sdm_test,
    bayesian_robust_lm_wx_test,
    bayesian_robust_lm_error_sdem_test,
    # Panel LM tests
    bayesian_panel_lm_lag_test,
    bayesian_panel_lm_error_test,
    bayesian_panel_robust_lm_lag_test,
    bayesian_panel_robust_lm_error_test,
    bayesian_panel_lm_wx_test,
    bayesian_panel_lm_sdm_joint_test,
    bayesian_panel_lm_slx_error_joint_test,
    bayesian_panel_robust_lm_lag_sdm_test,
    bayesian_panel_robust_lm_wx_test,
    bayesian_panel_robust_lm_error_sdem_test,
    bayes_factor_compare_models,
    bic_to_bf,
    compile_log_posterior,
    post_prob,
)
from .models import (
    OLS,
    SAR,
    SDEM,
    SDM,
    SEM,
    SLX,
    DLMPanelFE,
    SARPanelDEDynamic,
    SEMPanelDEDynamic,
    SDEMPanelDEDynamic,
    SLXPanelDEDynamic,
    OLSPanelFE,
    OLSPanelRE,
    SARPanelFE,
    SARPanelRE,
    SARPanelTobit,
    SARTobit,
    SDEMPanelFE,
    SLXPanelFE,
    SDMPanelFE,
    SDMRPanelFE,
    SDMTobit,
    SDMUPanelFE,
    SEMPanelFE,
    SEMPanelRE,
    SEMPanelTobit,
    SAR_Flow_Panel,
    SAR_Flow_Separable_Panel,
    PoissonFlow_Panel,
    PoissonFlow_Separable_Panel,
    SEMTobit,
    SpatialProbit,
)

__all__ = [
    "SLX",
    "OLS",
    "SAR",
    "SEM",
    "SDM",
    "SDEM",
    "SARTobit",
    "SEMTobit",
    "SDMTobit",
    "SpatialProbit",
    "OLSPanelFE",
    "SARPanelFE",
    "SEMPanelFE",
    "SDMPanelFE",
    "SDEMPanelFE",
    "SLXPanelFE",
    "DLMPanelFE",
    "SDMRPanelFE",
    "SDMUPanelFE",
    "SARPanelDEDynamic",
    "SEMPanelDEDynamic",
    "SDEMPanelDEDynamic",
    "SLXPanelDEDynamic",
    "OLSPanelRE",
    "SARPanelRE",
    "SEMPanelRE",
    "SARPanelTobit",
    "SEMPanelTobit",
    "dgp",
    "SAR_Flow",
    "SAR_Flow_Separable",
    "PoissonFlow",
    "PoissonFlow_Separable",
    "SAR_Flow_Panel",
    "SAR_Flow_Separable_Panel",
    "PoissonFlow_Panel",
    "PoissonFlow_Separable_Panel",
    "SparseFlowSolveOp",
    "destination_weights",
    "origin_weights",
    "network_weights",
    "flow_weight_matrices",
    "flow_design_matrix",
    "flow_design_matrix_with_orig",
    "FlowDesignMatrix",
    "generate_flow_data",
    "generate_poisson_flow_data",
    "bayes_factor_compare_models",
    "bic_to_bf",
    "BayesianLMTestResult",
    "bayesian_lm_lag_test",
    "bayesian_lm_error_test",
    "bayesian_lm_wx_test",
    "bayesian_lm_sdm_joint_test",
    "bayesian_lm_slx_error_joint_test",
    "bayesian_robust_lm_lag_sdm_test",
    "bayesian_robust_lm_wx_test",
    "bayesian_robust_lm_error_sdem_test",
    # Panel LM tests
    "bayesian_panel_lm_lag_test",
    "bayesian_panel_lm_error_test",
    "bayesian_panel_robust_lm_lag_test",
    "bayesian_panel_robust_lm_error_test",
    "bayesian_panel_lm_wx_test",
    "bayesian_panel_lm_sdm_joint_test",
    "bayesian_panel_lm_slx_error_joint_test",
    "bayesian_panel_robust_lm_lag_sdm_test",
    "bayesian_panel_robust_lm_wx_test",
    "bayesian_panel_robust_lm_error_sdem_test",
]


import contextlib
from importlib.metadata import PackageNotFoundError, version

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("bayespecon")
