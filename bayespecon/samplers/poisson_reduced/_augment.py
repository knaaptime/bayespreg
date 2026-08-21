r"""Auxiliary-mixture augmentation for Poisson log-link regression.

Implements the Frühwirth-Schnatter & Wagner (2006) data augmentation, which
represents a Poisson observation through the inter-arrival times of the
underlying Poisson process.  Unlike Pólya–Gamma — which has no exact Poisson
representation, and whose NB-limit approximation degenerates as the working
precision outruns the Fisher information — this scheme is exact and its working
precision converges to the Poisson information :math:`\mu`.

Construction
------------
Read :math:`y_i \sim \operatorname{Poisson}(\lambda_i)`,
:math:`\lambda_i = e^{\eta_i}`, as the number of jumps in :math:`[0, 1]` of a
Poisson process with intensity :math:`\lambda_i`.  Let :math:`t_k` be the
arrival times.  Augment with

- :math:`\tau_{i2} = t_{y_i}`, the last arrival at or before 1 (only when
  :math:`y_i > 0`), which is :math:`\operatorname{Gamma}(y_i, \lambda_i)`
  truncated to :math:`(0, 1)`;
- :math:`\xi_{i1}`, the inter-arrival time from :math:`\tau_{i2}` (or from 0
  when :math:`y_i = 0`) to the first arrival *after* 1, which by
  memorylessness is :math:`(1 - \tau_{i2}) + \operatorname{Exp}(\lambda_i)`.

Taking negative logs turns each into a location model in :math:`\eta_i`:

.. math::

    -\log \xi_{i1} = \eta_i + \varepsilon, \qquad
      \varepsilon \sim -\log \operatorname{Gamma}(1, 1) \\
    -\log \tau_{i2} = \eta_i + \varepsilon', \qquad
      \varepsilon' \sim -\log \operatorname{Gamma}(y_i, 1)

Each error is approximated by a finite normal mixture (see :mod:`._mixture`).
Conditional on the component indicator :math:`r`, the augmented observation is
exactly Gaussian with working response :math:`s = -\log(\cdot) - m_r` and
working precision :math:`\omega = 1/v_r` — the same ``(s, omega)`` contract the
Pólya–Gamma samplers hand to the shared β and ρ blocks.

So every observation contributes one augmented row, plus a second row when
:math:`y_i > 0`.
"""

from __future__ import annotations

import numpy as np

from ._mixture import mixture_for_shape, mixture_for_unit_shape

__all__ = ["AugmentedDesign", "draw_augmentation", "build_augmented_index"]


class AugmentedDesign:
    """Row bookkeeping for the ragged augmented design.

    The augmented design stacks the ``N`` inter-arrival rows on top of the
    ``N_pos`` last-arrival rows (one per strictly positive count), so the
    working design is ``U_aug = U[rows]`` with ``rows`` given by :attr:`rows`.

    Parameters
    ----------
    y : ndarray, shape (N,)
        Integer response vector.

    Attributes
    ----------
    N : int
        Number of observations.
    pos : ndarray of int
        Indices with ``y > 0``.
    rows : ndarray of int, shape (N + N_pos,)
        Index into the ``N``-row design for each augmented row.
    """

    __slots__ = ("N", "pos", "rows", "y_pos")

    def __init__(self, y: np.ndarray):
        y = np.asarray(y)
        self.N = int(y.shape[0])
        self.pos = np.flatnonzero(y > 0).astype(np.intp)
        self.y_pos = y[self.pos].astype(np.float64)
        self.rows = np.concatenate([np.arange(self.N, dtype=np.intp), self.pos])

    @property
    def n_aug(self) -> int:
        """Total number of augmented rows (``N + N_pos``)."""
        return int(self.rows.shape[0])


def build_augmented_index(y: np.ndarray) -> AugmentedDesign:
    """Build the augmented row index for a response vector."""
    return AugmentedDesign(y)


def _draw_truncated_gamma_below_one(
    shape: np.ndarray,
    rate: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""Draw :math:`\operatorname{Gamma}(\text{shape}, \text{rate})` truncated to (0, 1).

    Inverse-CDF on the truncated support.  When the untruncated mass below 1 is
    numerically zero (which happens transiently during warmup, if the current
    :math:`\lambda` is far below the count it must explain), the inverse CDF is
    ill-conditioned; we fall back to a draw from the *conditional* mode region
    by sampling uniformly in a shrunken interval below 1, which keeps the chain
    valid-by-construction rather than returning NaN.
    """
    from scipy.stats import gamma as _gamma

    scale = 1.0 / rate
    upper_mass = _gamma.cdf(1.0, a=shape, scale=scale)
    u = rng.random(shape.shape)
    safe = upper_mass > 1e-12
    out = np.empty_like(shape, dtype=np.float64)

    if np.any(safe):
        out[safe] = _gamma.ppf(
            u[safe] * upper_mass[safe], a=shape[safe], scale=scale[safe]
        )
    if np.any(~safe):
        # Degenerate branch: essentially all mass is above 1, so the truncated
        # draw concentrates just below 1.  Sample from (1-eps, 1).
        out[~safe] = 1.0 - 1e-8 * rng.random((~safe).sum())

    return np.clip(out, 1e-300, 1.0 - 1e-15)


def draw_augmentation(
    y: np.ndarray,
    eta: np.ndarray,
    design: AugmentedDesign,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Draw the auxiliary variables and return the Gaussian working data.

    Parameters
    ----------
    y : ndarray, shape (N,)
        Integer responses.
    eta : ndarray, shape (N,)
        Current linear predictor, :math:`\log \lambda`.
    design : AugmentedDesign
        Row bookkeeping from :func:`build_augmented_index`.
    rng : numpy.random.Generator
        Random state.

    Returns
    -------
    s : ndarray, shape (N + N_pos,)
        Working response for each augmented row.
    omega : ndarray, shape (N + N_pos,)
        Working precision (``1 / v_r``) for each augmented row.
    """
    eta = np.clip(np.asarray(eta, dtype=np.float64), -30.0, 30.0)
    lam = np.exp(eta)
    pos = design.pos

    # --- tau_2: last arrival at or before 1, only for y > 0 -------------
    tau2 = np.empty(pos.shape[0], dtype=np.float64)
    if pos.size:
        tau2 = _draw_truncated_gamma_below_one(design.y_pos, lam[pos], rng)

    # --- xi_1: inter-arrival from tau_2 (or 0) to the first arrival > 1 --
    # Memorylessness: the residual wait past (1 - tau_2) is a fresh Exp(lam).
    start = np.zeros(design.N, dtype=np.float64)
    if pos.size:
        start[pos] = tau2
    xi1 = (1.0 - start) + rng.exponential(1.0, size=design.N) / lam

    # --- turn each into a Gaussian working observation -------------------
    # Row block 1: -log(xi1) = eta + eps,  eps ~ -log Gamma(1, 1)
    w1, m1, v1 = mixture_for_unit_shape()
    s1, o1 = _mixture_working_data(-np.log(xi1), eta, w1, m1, v1, rng)

    if pos.size == 0:
        return s1, o1

    # Row block 2: -log(tau2) = eta + eps', eps' ~ -log Gamma(y_i, 1)
    s2, o2 = _mixture_working_data_by_shape(-np.log(tau2), eta[pos], design.y_pos, rng)
    return np.concatenate([s1, s2]), np.concatenate([o1, o2])


def _mixture_working_data(
    z: np.ndarray,
    eta: np.ndarray,
    w: np.ndarray,
    m: np.ndarray,
    v: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the mixture indicator, then return ``(s, omega)``.

    ``P(r = j | z, eta) ∝ w_j N(z - eta; m_j, v_j)``.
    """
    resid = z - eta
    # log w_j + log N(resid; m_j, v_j), shape (n, K)
    lp = (
        np.log(w)[None, :]
        - 0.5 * np.log(v)[None, :]
        - 0.5 * (resid[:, None] - m[None, :]) ** 2 / v[None, :]
    )
    lp -= lp.max(axis=1, keepdims=True)
    p = np.exp(lp)
    p /= p.sum(axis=1, keepdims=True)
    # Vectorised categorical draw via the inverse-CDF trick.
    idx = (p.cumsum(axis=1) < rng.random((resid.shape[0], 1))).sum(axis=1)
    idx = np.clip(idx, 0, w.shape[0] - 1)
    return z - m[idx], 1.0 / v[idx]


def _mixture_working_data_by_shape(
    z: np.ndarray,
    eta: np.ndarray,
    shapes: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """As :func:`_mixture_working_data`, with a per-observation shape.

    Groups by integer shape so each distinct ``y`` value costs one vectorised
    mixture step.  Counts above the tabulated cutoff use the single-normal
    limit, which needs no indicator draw at all.
    """
    s = np.empty(z.shape[0], dtype=np.float64)
    o = np.empty(z.shape[0], dtype=np.float64)
    for shape_val in np.unique(shapes):
        sel = shapes == shape_val
        w, m, v = mixture_for_shape(float(shape_val))
        if w.shape[0] == 1:
            s[sel] = z[sel] - m[0]
            o[sel] = 1.0 / v[0]
        else:
            s[sel], o[sel] = _mixture_working_data(z[sel], eta[sel], w, m, v, rng)
    return s, o
