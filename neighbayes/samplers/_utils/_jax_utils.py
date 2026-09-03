"""Shared JAX utility helpers for Gibbs samplers.

Centralises small boilerplate — JAX/equinox availability checks, BCOO
construction, Pólya-Gamma draw factories, and the conjugate-normal β
draw — that was previously copy-pasted across six sampler sub-packages.

Not part of the public API.
"""

from __future__ import annotations

import importlib.util


def check_jax_available(*, require_equinox: bool = False) -> None:
    """Raise ``ImportError`` if JAX (and optionally equinox) is not installed.

    Parameters
    ----------
    require_equinox : bool, default False
        If ``True``, also check for the ``equinox`` package.
    """
    if importlib.util.find_spec("jax") is None:
        raise ImportError(
            "JAX is required for the JAX Gibbs sampler. Install with: pip install jax"
        )
    if require_equinox and importlib.util.find_spec("equinox") is None:
        raise ImportError(
            "equinox is required for the JAX Gibbs sampler. "
            "Install with: pip install equinox"
        )


def build_w_bcoo(W_sparse):
    """Build ``(W, Wᵀ)`` as JAX BCOO sparse matrices — never densify W.

    Parameters
    ----------
    W_sparse : scipy.sparse.spmatrix
        Row-standardised spatial weights matrix.

    Returns
    -------
    tuple[jax.experimental.sparse.BCOO, jax.experimental.sparse.BCOO]
        ``(W_bcoo, Wt_bcoo)`` — W and its transpose as JAX BCOO matrices.
    """
    from jax.experimental import sparse as jsparse

    W_bcoo = jsparse.BCOO.from_scipy_sparse(W_sparse.tocsr())
    Wt_bcoo = jsparse.BCOO.from_scipy_sparse(W_sparse.T.tocsr())
    return W_bcoo, Wt_bcoo


def make_pg_draw():
    """Return a JAX-compatible Pólya-Gamma draw function.

    Prefers ``pgjax.pg_sample`` (exact Devroye sampler, on-device, no
    host round-trip) when installed — it dominates every alternative on
    speed, works inside ``jax.lax.scan``, and is exact for any ``h``
    (integer or non-integer).  Falls back to the ``polyagamma`` C extension
    via ``jax.pure_callback``; this is slower (host round-trip per call)
    but still correct for any ``h``.

    Returns
    -------
    callable
        ``draw_pg(h, z, key) -> jnp.ndarray`` — vectorised PG draw.
    """
    try:
        import pgjax

        def _draw_pg(h, z, key):
            return pgjax.pg_sample(h, z, key)

        return _draw_pg
    except ImportError:
        pass

    # Fallback: numpy polyagamma via pure_callback (slower but correct).
    # Numpy polyagamma beats the old on-device "exp" approximation at every
    # size tested, so there is no reason to keep the truncated-series path.
    def _draw_pg(h, z, key):
        import jax
        import jax.numpy as jnp
        import numpy as np

        from ._polyagamma import sample_polyagamma

        h_j = jnp.asarray(h, dtype=jnp.float64)
        z_j = jnp.asarray(z, dtype=jnp.float64)
        scalar_input = h_j.ndim == 0
        if scalar_input:
            h_j = h_j[None]
            z_j = z_j[None]

        result_shape = jnp.empty_like(h_j)
        key, cb_key = jax.random.split(key)
        cb_seed = jax.random.key_data(cb_key)[0].astype(jnp.int64) % (2**31)

        def _callback(h_np, z_np, seed_np):
            rng = np.random.default_rng(int(seed_np))
            return sample_polyagamma(
                np.asarray(h_np, dtype=np.float64),
                np.asarray(z_np, dtype=np.float64),
                rng=rng,
            )

        result = jax.pure_callback(_callback, result_shape, h_j, z_j, cb_seed)
        result = jnp.maximum(result, 1e-6)
        if scalar_input:
            result = result[0]
        return result

    return _draw_pg


def conjugate_normal(Ut, omega, working, V0, mu0, key, dim):
    """Draw β ~ N(Σ (Uᵀ working + V₀⁻¹μ₀), Σ), Σ⁻¹ = UᵀΩU + V₀⁻¹.

    Shared conjugate-normal posterior draw used by all JAX Gibbs samplers.

    Parameters
    ----------
    Ut : jnp.ndarray, shape (n, dim)
        Design matrix (pre-computed, possibly transformed).
    omega : jnp.ndarray, shape (n,)
        Pólya-Gamma precision weights.
    working : jnp.ndarray, shape (n,)
        Working response (z = κ/ω or similar).
    V0 : jnp.ndarray, shape (dim,)
        Prior precision vector (diagonal of V₀⁻¹).
    mu0 : jnp.ndarray, shape (dim,)
        Prior mean vector.
    key : jax.Array
        PRNG key.
    dim : int
        Dimension of β (passed explicitly to avoid re-deriving).

    Returns
    -------
    jnp.ndarray, shape (dim,)
        Posterior draw of β.
    """
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import cho_solve, solve_triangular

    Uw = Ut * omega[:, None]
    Sig_inv = Uw.T @ Ut + jnp.diag(V0) + 1e-10 * jnp.eye(dim)
    rhs = Ut.T @ working + V0 * mu0
    L = jnp.linalg.cholesky(Sig_inv)
    m = cho_solve((L, True), rhs)
    zc = jax.random.normal(key, shape=(dim,), dtype=jnp.float64)
    return m + solve_triangular(L.T, zc, lower=False)
