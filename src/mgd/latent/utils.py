"""Latent-related utilities."""

from __future__ import annotations

import jax.numpy as jnp


def center_logits(logits: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Center logits per node/edge across classes using the provided mask."""
    weights = mask[..., None]
    denom = jnp.maximum(weights.sum(axis=-1, keepdims=True), 1.0)
    mean = (logits * weights).sum(axis=-1, keepdims=True) / denom
    return logits - mean
