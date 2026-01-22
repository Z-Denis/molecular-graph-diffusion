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
    symmetrize: bool = True

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
        if self.symmetrize:
            edges = 0.5 * (edges + edges.swapaxes(-2, -3))

        return nodes, edges


class SelfAttention(nn.Module):
    n_features: int
    n_heads: int = 1

    param_dtype: DTypeLike = "float32"

    @nn.compact
    def __call__(
        self,
        nodes: jnp.ndarray,
        edges: jnp.ndarray,
        *,
        node_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        assert self.n_features % self.n_heads == 0
        nodes_shape = nodes.shape
        d_head = self.n_features // self.n_heads

        node_mask = jax.lax.stop_gradient(node_mask).astype(nodes.dtype)
        attn_mask = node_mask[:, :, None] * node_mask[:, None, :]
        big_neg = jnp.array(-1e9, dtype=self.param_dtype)
        log_attn_mask = (1.0 - attn_mask) * big_neg
        log_attn_mask = log_attn_mask[:, None, :, :]  # (B,1,N,N)
        diag = jnp.eye(nodes_shape[-1], dtype=nodes.dtype)[None, None, :, :]  # (1,1,N,N)
        
        qkv = nn.Dense(3 * self.n_features, param_dtype=self.param_dtype)(nodes)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(q.shape[:-1], self.n_heads, d_head).swapaxes(-2, -3)  # (B,H,N,Dh)
        k = k.reshape(k.shape[:-1], self.n_heads, d_head).swapaxes(-2, -3)  # (B,H,N,Dh)
        v = v.reshape(v.shape[:-1], self.n_heads, d_head).swapaxes(-2, -3)  # (B,H,N,Dh)
        edges = edges * (1 - diag)  # remove diagonal garbage
        bias = nn.Dense(self.n_heads, param_dtype=self.param_dtype)(edges)  # (B,N,N,H)
        bias = bias.swapaxes(-1, -3)    # (B,H,N,N)
        log_w = jnp.einsum('...id,...jd->...ij', q, k) / jnp.sqrt(d_head)
        log_w = log_w + bias + log_attn_mask
        w = jax.nn.softmax(log_w, axis=-1)
        out = jnp.einsum('...ij,...jd->...id', w, v)
        
        out = out.swapaxes(-2, -3).reshape(nodes_shape)
        out = nn.Dense(self.n_features, param_dtype=self.param_dtype)(out)
        return out * node_mask[..., None]


class TransformerBlock(nn.Module):
    node_dim: int
    edge_dim: int
    n_heads: int = 8
    alpha_node: int = 4
    alpha_edge: int = 4
    param_dtype: DTypeLike = "float32"
    symmetrize: bool = True
    use_mul: bool = True

    @nn.compact
    def __call__(
        self,
        nodes: jnp.ndarray,
        edges: jnp.ndarray,
        *,
        node_mask: jnp.ndarray,
        pair_mask: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        node_mask = jax.lax.stop_gradient(node_mask).astype(nodes.dtype)
        pair_mask = jax.lax.stop_gradient(pair_mask).astype(edges.dtype)

        ln = partial(nn.LayerNorm, param_dtype=self.param_dtype)
        att = SelfAttention(
                self.node_dim, 
                n_heads=self.n_heads, 
                param_dtype=self.param_dtype,
            )
        conc = aggregate_node_edge

        h = ln()(nodes)
        h = att(h, edges, node_mask=node_mask)
        nodes = nodes + h
        h = ln()(nodes)
        h = MLP(
            (self.alpha_node * self.node_dim, self.node_dim),
            param_dtype=self.param_dtype,
        )(h)
        nodes = nodes + h
        nodes = nodes * node_mask[..., None]

        ni = nodes[:, :, None, :]
        nj = nodes[:, None, :, :]
        parts = [ni, nj]
        if self.use_mul:
            parts.append(ni * nj)
        
        hp = ln()(edges)
        parts.append(hp)
        pair_in = jnp.concatenate(parts, axis=-1)

        hp = MLP(
            (self.alpha_edge * self.edge_dim, self.edge_dim),
            param_dtype=self.param_dtype,
        )(pair_in)
        edges = edges + hp
        if self.symmetrize:
            edges = 0.5 * (edges + edges.swapaxes(-2, -3))
        edges = edges * pair_mask

        return nodes, edges


__all__ = ["MessagePassingLayer"]
