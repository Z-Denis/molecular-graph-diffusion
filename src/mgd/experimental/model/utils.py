"""Utility modules and functions for graph models (legacy helpers)."""

from __future__ import annotations

from typing import Callable, Sequence

import jax.numpy as jnp
from mgd.model.utils import MLP, aggregate_node_edge

def estimate_latent_stats_masked(ae_state, loader, num_batches: int = 200):
    """Estimate latent mean/std using masks from batches to ignore padding."""
    m_node = 0.0
    m_edge = 0.0
    s_node = 0.0
    s_edge = 0.0
    w_node = 0.0
    w_edge = 0.0

    # Mean
    for _, batch in zip(range(num_batches), loader):
        lat = ae_state.encode(batch, apply_norm=False)
        node_mask = batch.node_mask[..., None]
        edge_mask = batch.pair_mask[..., None]
        m_node += (lat.node * node_mask).sum(axis=(0, 1))
        m_edge += (lat.edge * edge_mask).sum(axis=(0, 1, 2))
        w_node += node_mask.sum()
        w_edge += edge_mask.sum()
    mean = {
        "node": m_node / w_node,
        "edge": m_edge / w_edge,
    }

    # Variance
    for _, batch in zip(range(num_batches), loader):
        lat = ae_state.encode(batch, apply_norm=False)
        node_mask = batch.node_mask[..., None]
        edge_mask = batch.pair_mask[..., None]
        s_node += (jnp.square(lat.node - mean["node"]) * node_mask).sum(axis=(0, 1))
        s_edge += (jnp.square(lat.edge - mean["edge"]) * edge_mask).sum(axis=(0, 1, 2))
    std = {
        "node": jnp.sqrt(s_node / w_node + 1e-8),
        "edge": jnp.sqrt(s_edge / w_edge + 1e-8),
    }
    return mean, std


def bond_bias_initializer(p_exist: float, p_types: Sequence[float] | None = None, eps: float = 1e-6) -> Callable:
    """Initializer for decoder visible layer bias using empirical bond probabilities.

    The bias vector is:
        [logit(p_exist), log(p_single), log(p_double), log(p_triple), log(p_aromatic), ...]
    where ``p_types`` provides the per-type probabilities (already normalized as you prefer).
    If ``p_types`` is None, type logits are initialized to zeros.
    """
    p_exist = jnp.clip(p_exist, eps, 1.0 - eps)
    exist_bias = jnp.log(p_exist) - jnp.log1p(-p_exist)

    if p_types is None:
        def init(key, shape, dtype=jnp.float32):
            if shape[0] < 1:
                raise ValueError(f"bond_bias_initializer expected shape >=1, got {shape}")
            bias = jnp.zeros(shape, dtype=dtype).at[0].set(exist_bias.astype(dtype))
            return bias
        return init
    
    p_types_arr = jnp.asarray(p_types, dtype=jnp.float32)
    p_types_arr = jnp.clip(p_types_arr, eps, 1.0)
    type_bias = jnp.log(p_types_arr)

    bias_vec = jnp.concatenate([jnp.array([exist_bias], dtype=jnp.float32), type_bias])
    expected_shape = (bias_vec.shape[0],)

    def init(key, shape, dtype=jnp.float32):
        if tuple(shape) != expected_shape:
            raise ValueError(f"bond_bias_initializer expected shape {expected_shape}, got {shape}")
        return bias_vec.astype(dtype)

    return init


__all__ = [
    "MLP",
    "aggregate_node_edge",
    "bond_bias_initializer",
    "estimate_latent_stats_masked",
]
