"""Flax Linen modules for the backbone of a denoiser."""

from __future__ import annotations

from typing import Callable, Tuple
from functools import partial

import jax.numpy as jnp
from jax.typing import DTypeLike
from flax import linen as nn

from ..latent import GraphLatentSpace
from .embeddings import TimeEmbedding
from .gnn_layers import MessagePassingLayer


class MPNNBackbone(nn.Module):
    space: GraphLatentSpace
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
        times: jnp.ndarray,
        *,
        node_mask: jnp.ndarray,
        pair_mask: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        node_dim, edge_dim = nodes.shape[-1], edges.shape[-1]
        if node_dim != self.space.node_dim or edge_dim != self.space.edge_dim:
            raise ValueError(
                f"Backbone received node/edge dims {(node_dim, edge_dim)} "
                f"but was initialized with {(self.space.node_dim, self.space.edge_dim)}."
            )
        time_emb = TimeEmbedding(
            self.time_dim,
            self.space.node_dim,
            self.space.edge_dim,
            self.activation,
            param_dtype=self.param_dtype,
            name="time_embedding",
        )
        mpnn = partial(
            MessagePassingLayer,
            self.space.node_dim,
            self.space.edge_dim,
            self.mess_dim,
            activation=self.activation,
            param_dtype=self.param_dtype,
        )
        
        t_nodes, t_edges = time_emb(times)
        nodes = nodes + t_nodes[..., None, :]
        edges = edges + t_edges[..., None, None, :]

        for i in range(self.n_layers):
            nodes, edges = mpnn(name=f"mpnn_{i}")(
                nodes, edges,
                node_mask=node_mask,
                pair_mask=pair_mask,
            )
        
        return nodes, edges
