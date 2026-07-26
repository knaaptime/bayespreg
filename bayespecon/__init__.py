"""Bayesian spatial econometric models and diagnostics.

The package exposes cross-sectional and panel spatial regression model
classes and Bayesian specification tests.

Submodules and attributes are loaded lazily following SPEC 1
(https://scientific-python.org/specs/spec-0001/) so that ``import bayespecon``
is cheap and does not eagerly import ``pymc``/``pytensor``/``arviz``. The
public API surface is declared in the sibling ``__init__.pyi`` stub for
static type checkers and IDE autocomplete.

Examples
--------
Import a model class from the ``models`` submodule::

        from bayespecon.models import SAR
"""

import contextlib
import os as _os
import sys as _sys
from importlib.metadata import PackageNotFoundError, version


def _auto_configure_cpu_devices() -> None:
    """Expose multiple CPU devices so the JAX Gibbs backends can run each chain
    on its own device (``pmap``) for true multi-core parallelism.

    JAX's CPU backend defaults to a *single* device, so ``vmap``-over-chains
    executes on one core and loses to the NumPy path's joblib *processes*.
    Mapping one chain per CPU device (``pmap``) is the analogue of those
    processes — and, with JAX's per-core efficiency, it *beats* the NumPy
    backend.  The device count must be chosen before JAX initialises, via
    ``--xla_force_host_platform_device_count``; we set it at import unless JAX is
    already loaded or the user configured it explicitly.
    """
    if "jax" in _sys.modules:
        return  # too late: JAX already initialised — the Gibbs path falls back to vmap
    flags = _os.environ.get("XLA_FLAGS", "")
    if "xla_force_host_platform_device_count" in flags:
        return  # respect an explicit user setting
    n = min(_os.cpu_count() or 4, 16)
    _os.environ["XLA_FLAGS"] = (
        f"{flags} --xla_force_host_platform_device_count={n}".strip()
    )


_auto_configure_cpu_devices()

import lazy_loader as _lazy

__getattr__, __dir__, __all__ = _lazy.attach_stub(__name__, __file__)

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("bayespecon")
