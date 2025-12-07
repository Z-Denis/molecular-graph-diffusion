"""Flax Linen modules for the backbone of a denoiser."""

from __future__ import annotations

from typing import Callable
from functools import partial

import jax.numpy as jnp
from jax.typing import DTypeLike
from flax import linen as nn

from .embeddings import TimeEmbedding
from .gnn_layers import MessagePassingLayer


class MPNNBackbone(nn.Module):
    node_dim: int   # hidden_dim for nodes
    edge_dim: int   # hidden_dim for edges
    mess_dim: int   # message hidden dim
    time_dim: int   # time hidden dim

    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    n_layers: int = 1
    param_dtype: DTypeLike = "float32"

    @nn.compact
    def __call__(self, nodes: jnp.ndarray, edges: jnp.ndarray, times: jnp.ndarray, *,
                 node_mask: jnp.ndarray, pair_mask: jnp.ndarray):
        time_emb = TimeEmbedding(
            self.time_dim,
            self.node_dim,
            self.edge_dim,
            self.activation,
            param_dtype=self.param_dtype,
            name="time_embedding",
        )
        mpnn = partial(
            MessagePassingLayer,
            self.node_dim,
            self.edge_dim,
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
