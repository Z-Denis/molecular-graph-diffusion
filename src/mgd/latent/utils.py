"""Latent-related utilities."""

from __future__ import annotations

import jax.numpy as jnp

from mgd.latent.space import GraphLatent


def center_logits(logits: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Center logits per node/edge across classes using the provided mask."""
    weights = mask[..., None]
    denom = jnp.maximum(weights.sum(axis=-1, keepdims=True), 1.0)
    mean = (logits * weights).sum(axis=-1, keepdims=True) / denom
    return logits - mean


def symmetrize_edge(edge: jnp.ndarray) -> jnp.ndarray:
    """Symmetrize edge logits by averaging (i,j) and (j,i)."""
    return 0.5 * (edge + edge.swapaxes(-2, -3))


def symmetrize_latent(
    latent: GraphLatent,
    node_mask: jnp.ndarray | None = None,
    pair_mask: jnp.ndarray | None = None,
) -> GraphLatent:
    """Return a latent with symmetric edges, optionally masked."""
    edge = symmetrize_edge(latent.edge)
    out = GraphLatent(latent.node, edge)
    if node_mask is not None and pair_mask is not None:
        out = out.masked(node_mask, pair_mask)
    return out
