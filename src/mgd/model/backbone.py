"""Flax Linen modules for the backbone of a denoiser."""

from __future__ import annotations

from typing import Callable, Tuple
from functools import partial

import jax.numpy as jnp
from jax.typing import DTypeLike
from flax import linen as nn

from ..latent import GraphLatentSpace
from .embeddings import TimeEmbedding, NodeCountEmbedding
from .gnn_layers import MessagePassingLayer, TransformerBlock


class MPNNBackbone(nn.Module):
    space: GraphLatentSpace
    mess_dim: int   # message hidden dim
    time_dim: int   # time hidden dim
    node_count_dim: int | None = None

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
        count_dim = self.time_dim if self.node_count_dim is None else self.node_count_dim
        count_emb = NodeCountEmbedding(
            embed_dim=count_dim,
            node_dim=self.space.node_dim,
            edge_dim=self.space.edge_dim,
            activation=self.activation,
            param_dtype=self.param_dtype,
            name="node_count_embedding",
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
        n_nodes = jnp.sum(node_mask, axis=-1).astype(jnp.int32)
        c_nodes, c_edges = count_emb(n_nodes)
        nodes = nodes + t_nodes[..., None, :]
        edges = edges + t_edges[..., None, None, :]
        nodes = nodes + c_nodes[..., None, :]
        edges = edges + c_edges[..., None, None, :]

        for i in range(self.n_layers):
            nodes, edges = mpnn(name=f"mpnn_{i}")(
                nodes, edges,
                node_mask=node_mask,
                pair_mask=pair_mask,
            )
        
        return nodes, edges


class TransformerBackbone(nn.Module):
    space: GraphLatentSpace
    node_dim: int
    edge_dim: int

    time_dim: int   # time hidden dim
    node_count_dim: int | None = None

    n_layers: int = 1
    n_heads: int = 8
    alpha_node: int = 4
    alpha_edge: int = 4
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    param_dtype: DTypeLike = "float32"
    symmetrize: bool = True
    use_mul: bool = True

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
        proj_n = nn.Dense(self.node_dim, param_dtype=self.param_dtype)
        proj_e = nn.Dense(self.edge_dim, param_dtype=self.param_dtype)
        time_emb = TimeEmbedding(
            time_dim=self.time_dim,
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            activation=self.activation,
            param_dtype=self.param_dtype,
            name="time_embedding",
        )
        count_dim = self.time_dim if self.node_count_dim is None else self.node_count_dim
        count_emb = NodeCountEmbedding(
            embed_dim=count_dim,
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            activation=self.activation,
            param_dtype=self.param_dtype,
            name="node_count_embedding",
        )
        tf = partial(
            TransformerBlock,
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            n_heads=self.n_heads,
            alpha_node=self.alpha_node,
            alpha_edge=self.alpha_edge,
            activation=self.activation,
            param_dtype=self.param_dtype,
            symmetrize=self.symmetrize,
            use_mul=self.use_mul,
        )

        t_nodes, t_edges = time_emb(times)
        n_nodes = jnp.sum(node_mask, axis=-1).astype(jnp.int32)   # we should start providing conditioning externally
        c_nodes, c_edges = count_emb(n_nodes)
        nodes = proj_n(nodes) + t_nodes[..., None, :]
        edges = proj_e(edges) + t_edges[..., None, None, :]
        nodes = nodes + c_nodes[..., None, :]
        edges = edges + c_edges[..., None, None, :]

        for i in range(self.n_layers):
            nodes, edges = tf(name=f"block_{i}")(
                nodes, edges,
                node_mask=node_mask,
                pair_mask=pair_mask,
            )
        
        return nodes, edges
