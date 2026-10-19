"""Host-side scalar and state diagnostics for experiment artifacts."""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp


def finite_float(value: Any) -> float | None:
    """Convert a scalar to a JSON-safe finite float, or ``None``."""

    if value is None:
        return None
    selected = float(value)
    return selected if math.isfinite(selected) else None


@jax.jit
def _tree_nonfinite_count(tree: Any):
    counts = [
        jnp.count_nonzero(~jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(tree)
    ]
    if not counts:
        return jnp.asarray(0, dtype=jnp.int32)
    return jnp.sum(jnp.stack(counts))


def tree_nonfinite_count(tree: Any) -> int:
    """Count nonfinite scalar elements without transferring the full pytree."""

    count = _tree_nonfinite_count(tree)
    count.block_until_ready()
    return int(count)
