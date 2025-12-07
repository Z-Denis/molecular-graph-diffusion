"""Dataset helper utilities."""

from __future__ import annotations

import flax
import jax.numpy as jnp


@flax.struct.dataclass
class GraphBatch:
    """Container for batched graph features."""

    atom_type: jnp.ndarray
    hybrid: jnp.ndarray
    cont: jnp.ndarray
    edges: jnp.ndarray
    node_mask: jnp.ndarray
    pair_mask: jnp.ndarray


__all__ = ["GraphBatch"]
