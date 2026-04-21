"""Bayesian spatial econometric models and diagnostics.

The package exposes cross-sectional and panel spatial regression model
classes, residual diagnostics, and MATLAB-style test/statistics helpers.

Examples
--------
Import a model class directly from the package namespace::

        from bayespecon import SAR

Import MATLAB-style statistics helpers from the stats submodule::

        from bayespecon import stats
"""

from . import dgp, stats
from .diagnostics import (
    DiagnosticResult,
    arch_test,
    bpagan_test,
    hausman_fe_re_test,
    ljung_box_q,
    panel_residual_structure,
    pesaran_cd_test,
    rdiagnose_like,
)
from .diagnostics_bayesfactor import bayes_factor_compare_models
from .models import (
    OLS,
    SAR,
    SDEM,
    SDM,
    SEM,
    SLX,
    DLMPanelFE,
    OLSPanelFE,
    OLSPanelRE,
    SARPanelFE,
    SARPanelRE,
    SARPanelTobit,
    SARTobit,
    SDEMPanelFE,
    SDMPanelFE,
    SDMRPanelFE,
    SDMTobit,
    SDMUPanelFE,
    SEMPanelFE,
    SEMPanelRE,
    SEMPanelTobit,
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
    "DLMPanelFE",
    "SDMRPanelFE",
    "SDMUPanelFE",
    "OLSPanelRE",
    "SARPanelRE",
    "SEMPanelRE",
    "SARPanelTobit",
    "SEMPanelTobit",
    "rdiagnose_like",
    "bpagan_test",
    "arch_test",
    "hausman_fe_re_test",
    "ljung_box_q",
    "panel_residual_structure",
    "pesaran_cd_test",
    "stats",
    "dgp",
    "bayes_factor_compare_models",
]


import contextlib
from importlib.metadata import PackageNotFoundError, version

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("bayespecon")
