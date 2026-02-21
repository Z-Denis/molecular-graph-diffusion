"""Flax Linen modules for the backbone of a denoiser."""

from __future__ import annotations

from typing import Callable, Tuple
from functools import partial

import jax.numpy as jnp
from jax.typing import DTypeLike
from flax import linen as nn

from ..dataset.chemistry import DEFAULT_CHEMISTRY
from .embeddings import TimeEmbedding, NodeCountEmbedding
from .gnn_layers import MessagePassingLayer, TransformerBlock


class MPNNBackbone(nn.Module):
    node_dim: int
    edge_dim: int
    mess_dim: int   # message hidden dim
    time_dim: int   # time hidden dim
    node_count_dim: int | None = None
    max_nodes: int = DEFAULT_CHEMISTRY.max_nodes
    use_routing_gate: bool = False
    routing_beta_min: float = 0.0
    routing_beta_max: float = 1.0
    routing_beta_k: float = 3.0
    routing_sigma_data: float = 1.0

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
        time_emb = TimeEmbedding(
            self.time_dim,
            self.node_dim,
            self.edge_dim,
            self.activation,
            param_dtype=self.param_dtype,
            name="time_embedding",
        )
        count_dim = self.time_dim if self.node_count_dim is None else self.node_count_dim
        count_emb = NodeCountEmbedding(
            embed_dim=count_dim,
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            max_nodes=self.max_nodes,
            activation=self.activation,
            param_dtype=self.param_dtype,
            name="node_count_embedding",
        )
        mpnn = partial(
            MessagePassingLayer,
            self.node_dim,
            self.edge_dim,
            self.mess_dim,
            activation=self.activation,
            param_dtype=self.param_dtype,
            use_routing_gate=self.use_routing_gate,
            routing_beta_min=self.routing_beta_min,
            routing_beta_max=self.routing_beta_max,
            routing_beta_k=self.routing_beta_k,
            routing_sigma_data=self.routing_sigma_data,
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
                nodes,
                edges,
                times,
                node_mask=node_mask,
                pair_mask=pair_mask,
            )
        
        return nodes, edges


class TransformerBackbone(nn.Module):
    node_dim: int
    edge_dim: int

    time_dim: int   # time hidden dim
    node_count_dim: int | None = None
    max_nodes: int = DEFAULT_CHEMISTRY.max_nodes

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
            max_nodes=self.max_nodes,
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
