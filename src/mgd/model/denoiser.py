"""Flax-based molecular graph diffusion denoiser."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from flax import linen as nn

from ..latent import GraphLatent
from .backbone import MPNNBackbone


class MPNNDenoiser(nn.Module):
    node_dim: int
    edge_dim: int

    node_vocab: int
    edge_vocab: int
    
    mess_dim: int   # message hidden dim
    time_dim: int   # time hidden dim
    node_count_dim: int | None = None

    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    n_layers: int = 1
    param_dtype: DTypeLike = "float32"

    @nn.compact
    def __call__(
        self,
        xt: GraphLatent,
        log_sigma: jnp.ndarray,
        *,
        node_mask: jnp.ndarray,
        pair_mask: jnp.ndarray,
    ) -> GraphLatent:
        backbone = MPNNBackbone(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            mess_dim=self.mess_dim,
            time_dim=self.time_dim,
            node_count_dim=self.node_count_dim,
            activation=self.activation,
            param_dtype=self.param_dtype,
            n_layers=self.n_layers,
            name="backbone",
        )

        # Evaluate heads
        nodes, edges = backbone(
            xt.node,
            xt.edge,
            log_sigma,
            node_mask=node_mask,
            pair_mask=pair_mask,
        )

        # Latent to noise space
        eps_nodes = nn.Dense(
            self.node_vocab, param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(1e-6),
            name="node_head",
        )(nodes)
        eps_edges = nn.Dense(
            self.edge_vocab, param_dtype=self.param_dtype,
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
