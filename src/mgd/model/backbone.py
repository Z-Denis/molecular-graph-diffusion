"""Flax Linen modules for the backbone of a denoiser."""

from __future__ import annotations

from typing import Callable
from functools import partial

import jax.numpy as jnp
from jax.typing import DTypeLike
from flax import linen as nn

from ..dataset.encoding import ATOM_VOCAB_SIZE, HYBRID_VOCAB_SIZE, BOND_VOCAB_SIZE
from ..dataset.utils import GraphBatch
from .embeddings import NodeEmbedder, EdgeEmbedder, TimeEmbedding
from .gnn_layers import MessagePassingLayer


class MPNNBackbone(nn.Module):
    atom_dim: int 
    hybrid_dim: int
    cont_dim: int

    node_dim: int   # hidden_dim for nodes
    edge_dim: int   # hidden_dim for edges
    mess_dim: int   # message hidden dim
    time_dim: int   # time hidden dim

    atom_vocab_dim: int = ATOM_VOCAB_SIZE
    hybrid_vocab_dim: int = HYBRID_VOCAB_SIZE
    bond_vocab_dim: int = BOND_VOCAB_SIZE

    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    n_layers: int = 1
    param_dtype: DTypeLike = "float32"

    @nn.compact
    def __call__(self, graph: GraphBatch, times: jnp.ndarray):
        time_emb = TimeEmbedding(
            self.time_dim,
            self.node_dim,
            self.edge_dim,
            self.activation,
            param_dtype=self.param_dtype,
            name="time_embedding",
        )
        node_emb = NodeEmbedder(
            self.atom_vocab_dim,
            self.hybrid_vocab_dim,
            self.atom_dim, 
            self.hybrid_dim, 
            self.cont_dim, 
            self.node_dim,
            self.activation,
            param_dtype=self.param_dtype,
            name="node_embedding",
        )
        edge_emb = EdgeEmbedder(
            self.bond_vocab_dim,
            self.edge_dim,
            self.edge_dim,
            self.activation,
            param_dtype=self.param_dtype,
            name="edge_embedding",
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
        nodes = node_emb(
            graph.atom_type,
            graph.hybrid,
            graph.cont,
        )
        nodes = t_nodes[..., None, :] + nodes
        edges = t_edges[..., None, None, :] + edge_emb(graph.edges)

        for i in range(self.n_layers):
            nodes, edges = mpnn(name=f"mpnn_{i}")(
                nodes, edges,
                node_mask=graph.node_mask,
                pair_mask=graph.pair_mask,
            )
        
        return nodes, edges
