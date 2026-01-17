"""Latent-related utilities."""

from __future__ import annotations

import jax.numpy as jnp

from mgd.latent.space import GraphLatent


def normalize_embeddings(vectors: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Normalize embeddings to a radius sqrt(d) hypersphere."""
    dim = vectors.shape[-1]
    norm = jnp.linalg.norm(vectors, axis=-1, keepdims=True)
    return jnp.sqrt(dim) * vectors / (norm + eps)


def weighted_embeddings(probs: jnp.ndarray, embeddings: jnp.ndarray) -> jnp.ndarray:
    """Compute probability-weighted embeddings."""
    return jnp.einsum("...k,kd->...d", probs, embeddings)


def latent_from_probs(
    node_probs: jnp.ndarray,
    edge_probs: jnp.ndarray,
    node_embeddings: jnp.ndarray,
    edge_embeddings: jnp.ndarray,
) -> GraphLatent:
    """Map categorical probabilities to expected latent embeddings."""
    node = weighted_embeddings(node_probs, node_embeddings)
    edge = weighted_embeddings(edge_probs, edge_embeddings)
    return GraphLatent(node=node, edge=edge)


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
