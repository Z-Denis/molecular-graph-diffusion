"""Flax-based molecular graph diffusion denoiser."""

from __future__ import annotations

from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from flax import linen as nn

from .backbone import MPNNBackbone


class MPNNDenoiser(nn.Module):
    node_dim: int   # hidden_dim for nodes
    edge_dim: int   # hidden_dim for edges
    mess_dim: int   # message hidden dim
    time_dim: int   # time hidden dim

    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    n_layers: int = 1
    param_dtype: DTypeLike = "float32"

    @nn.compact
    def __call__(
        self,
        nodes: jnp.ndarray,
        edges: jnp.ndarray,
        time: jnp.ndarray,
        *,
        node_mask: jnp.ndarray,
        pair_mask: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        backbone = MPNNBackbone(
            self.node_dim, self.edge_dim, self.mess_dim, self.time_dim,
            activation=self.activation,
            param_dtype=self.param_dtype,
            n_layers=self.n_layers,
            name='backbone',
        )

        # Evaluate heads
        nodes, edges = backbone(nodes, edges, time, 
                                node_mask=node_mask, pair_mask=pair_mask)

        # Latent to noise space
        eps_nodes = nn.Dense(
            self.node_dim, param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(1e-6),
            name="node_head",
        )(nodes)
        eps_edges = nn.Dense(
            self.edge_dim, param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(1e-6),
            name="edge_head",
        )(edges)

        # Mask out invalid atomic positions
        node_mask = jax.lax.stop_gradient(node_mask)
        pair_mask = jax.lax.stop_gradient(pair_mask)
        eps_nodes = eps_nodes * node_mask[..., None]
        eps_edges = eps_edges * pair_mask[..., None]

        return eps_nodes, eps_edges

__all__ = ["MPNNDenoiser"]
