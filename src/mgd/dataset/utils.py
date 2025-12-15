"""Dataset helper utilities."""

from __future__ import annotations

import flax
import jax
import jax.numpy as jnp


@flax.struct.dataclass
class GraphBatch:
    """Container for batched graph features."""

    atom_type: jnp.ndarray
    hybrid: jnp.ndarray
    cont: jnp.ndarray
    bond_type: jnp.ndarray
    dknn: jnp.ndarray
    node_mask: jnp.ndarray
    pair_mask: jnp.ndarray

    def stop_gradient(self) -> "GraphBatch":
        """Return a copy with gradients stopped on all fields."""
        return GraphBatch(
            atom_type=jax.lax.stop_gradient(self.atom_type),
            hybrid=jax.lax.stop_gradient(self.hybrid),
            cont=jax.lax.stop_gradient(self.cont),
            bond_type=jax.lax.stop_gradient(self.bond_type),
            dknn=jax.lax.stop_gradient(self.dknn),
            node_mask=jax.lax.stop_gradient(self.node_mask),
            pair_mask=jax.lax.stop_gradient(self.pair_mask),
        )


__all__ = ["GraphBatch"]
