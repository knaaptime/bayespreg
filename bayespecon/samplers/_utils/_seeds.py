"""Seed derivation helpers for Gibbs samplers.

Centralises the pattern of spawning per-chain ``SeedSequence`` children from a
user-supplied ``random_seed`` (or fresh OS entropy when ``None``).

Historically each call site did::

    child_seeds = parent_ss.spawn(chains)
    seeds = [int(s.generate_state(1)[0]) for s in child_seeds]

which collapses a 128-bit ``SeedSequence`` into a 32-bit ``int`` (via
``generate_state(1)``), discarding most of the entropy.  The helpers here
return the ``SeedSequence`` children directly so that ``np.random.default_rng``
receives the full 128-bit seed.  For JAX paths that require an integer key,
:func:`seed_sequence_to_int` extracts 63 bits (to fit ``np.int64``).
"""

from __future__ import annotations

import numpy as np


def spawn_chain_seeds(
    random_seed: int | None,
    chains: int,
    *,
    extra: int = 0,
) -> list[np.random.SeedSequence]:
    """Spawn per-chain ``SeedSequence`` children from a user seed.

    Parameters
    ----------
    random_seed : int or None
        User-supplied seed.  ``None`` draws fresh OS entropy (non-reproducible).
    chains : int
        Number of parallel chains.
    extra : int, default 0
        Additional ``SeedSequence`` children to spawn (e.g. for a scouting
        phase).  The returned list has length ``chains + extra``; the caller
        is responsible for slicing off the extra seeds.

    Returns
    -------
    list of np.random.SeedSequence
        ``chains + extra`` independent child seeds, each carrying full
        128-bit entropy.
    """
    parent = (
        np.random.SeedSequence(random_seed)
        if random_seed is not None
        else np.random.SeedSequence()
    )
    return list(parent.spawn(chains + extra))


def seed_sequence_to_int(seed: np.random.SeedSequence) -> int:
    """Convert a ``SeedSequence`` to a Python ``int`` for JAX ``PRNGKey``.

    Uses ``generate_state(2)`` to produce a 64-bit value (two uint32 words),
    preserving more entropy than the previous ``int(s.generate_state(1)[0])``
    (32-bit) pattern.  The result is masked to signed-64-bit range so JAX's
    ``PRNGKey`` (which calls ``np.int64(seed)``) does not overflow.
    """
    val = int(seed.generate_state(2).view(np.uint64)[0])
    # Mask to 63 bits so it fits in np.int64 (JAX requirement).
    return val & ((1 << 63) - 1)
