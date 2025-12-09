"""Flax Linen modules for graph neural network layers (message passing, readout)."""

from __future__ import annotations

from typing import Callable, Tuple
from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from flax import linen as nn

from .utils import MLP, aggregate_node_edge


class MessagePassingLayer(nn.Module):
    """Single message passing block with simple MLP updates."""

    node_dim: int
    edge_dim: int
    mess_dim: int
    residual_connections: bool = True
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    param_dtype: DTypeLike = "float32"

    @nn.compact
    def __call__(
        self,
        nodes: jnp.ndarray,
        edges: jnp.ndarray,
        *,
        node_mask: jnp.ndarray,
        pair_mask: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Updates nodes and edges.

        nodes: (batch..., n_atoms, node_dim)
        edges: (batch..., n_atoms, n_atoms, edge_dim)
        node_mask: (batch..., n_atoms)
        pair_mask: (batch..., n_atoms, n_atoms)
        """
        pair_mask = jax.lax.stop_gradient(pair_mask)
        node_mask = jax.lax.stop_gradient(node_mask)

        conc = aggregate_node_edge  # Defaults to aggregation by concatenation
        mlp = partial(MLP, param_dtype=self.param_dtype, n_layers=2, activation=self.activation)
        edge_update = mlp(features=self.edge_dim, name='edge_mlp')
        mess_update = mlp(features=self.mess_dim, name='mess_mlp')
        node_update = mlp(features=self.node_dim, name='node_mlp')

        nodes_n = nn.LayerNorm(param_dtype=self.param_dtype)(nodes)
        edges_n = nn.LayerNorm(param_dtype=self.param_dtype)(edges)

        m_ij = mess_update(conc(node_j=nodes_n, edge_ij=edges_n))
        m_ij = m_ij * pair_mask[..., None]
        m_i = jnp.sum(m_ij, axis=-2)
        m_i_n = nn.LayerNorm(param_dtype=self.param_dtype)(m_i)

        edges_up = edge_update(conc(node_i=nodes_n, node_j=nodes_n, edge_ij=edges_n))
        edges_up = edges_up * pair_mask[..., None]
        if self.residual_connections:
            edges = edges + edges_up
        else:
            edges = edges_up

        nodes_up = node_update(m_i_n) * node_mask[..., None]
        if self.residual_connections:
            nodes = nodes + nodes_up
        else:
            nodes = nodes_up

        return nodes, edges

__all__ = ["MessagePassingLayer"]
