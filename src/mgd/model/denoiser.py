"""Flax-based molecular graph diffusion denoiser."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from flax import linen as nn

from ..latent import GraphLatent, GraphLatentSpace
from .backbone import MPNNBackbone


class MPNNDenoiser(nn.Module):
    space: GraphLatentSpace
    
    mess_dim: int   # message hidden dim
    time_dim: int   # time hidden dim

    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    n_layers: int = 1
    param_dtype: DTypeLike = "float32"

    @nn.compact
    def __call__(
        self,
        xt: GraphLatent,
        time: jnp.ndarray,
        *,
        node_mask: jnp.ndarray,
        pair_mask: jnp.ndarray,
    ) -> GraphLatent:
        backbone = MPNNBackbone(
            space=self.space,
            mess_dim=self.mess_dim,
            time_dim=self.time_dim,
            activation=self.activation,
            param_dtype=self.param_dtype,
            n_layers=self.n_layers,
            name="backbone",
        )

        # Evaluate heads
        nodes, edges = backbone(xt.node, xt.edge, time,
                                node_mask=node_mask, pair_mask=pair_mask)

        # Latent to noise space
        eps_nodes = nn.Dense(
            self.space.node_dim, param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(1e-6),
            name="node_head",
        )(nodes)
        eps_edges = nn.Dense(
            self.space.edge_dim, param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(1e-6),
            name="edge_head",
        )(edges)

        # Mask out invalid atomic positions
        node_mask = jax.lax.stop_gradient(node_mask)
        pair_mask = jax.lax.stop_gradient(pair_mask)
        eps_nodes = eps_nodes * node_mask[..., None]
        eps_edges = eps_edges * pair_mask[..., None]

        return GraphLatent(eps_nodes, eps_edges)

__all__ = ["MPNNDenoiser"]
