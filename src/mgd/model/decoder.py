"""Flax Linen modules for node-wise and edge-wise decoding."""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
from flax import linen as nn
from jax.typing import DTypeLike
from jax.nn.initializers import zeros

# Use absolute import to avoid any relative import ambiguity
from mgd.model.utils import MLP


class NodeCategoricalDecoder(nn.Module):
    """Decode node latents into categorical logits."""

    hidden_dim: int
    n_layers: int
    n_categories: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    param_dtype: DTypeLike = "float32"

    @nn.compact
    def __call__(self, node_latent: jnp.ndarray) -> jnp.ndarray:
        """Return node logits of shape (..., n_atoms, n_categories)."""
        ln = nn.LayerNorm(param_dtype=self.param_dtype)
        mlp = MLP(
            self.hidden_dim,
            self.n_layers,
            activation=self.activation,
            param_dtype=self.param_dtype,
        )
        head = nn.Dense(self.n_categories,
                          param_dtype=self.param_dtype,
                          )

        h = ln(node_latent)
        h = mlp(h)
        h = self.activation(h)
        logits = head(h)

        return logits


class EdgeCategoricalDecoder(nn.Module):
    """Decode edge latents into symmetric categorical logits."""

    hidden_dim: int
    n_layers: int
    n_categories: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    bond_bias_init: Callable = zeros
    param_dtype: DTypeLike = "float32"

    @nn.compact
    def __call__(self, edge_latent: jnp.ndarray) -> jnp.ndarray:
        """Return symmetric edge logits of shape (..., n_atoms, n_atoms, n_categories)."""
        ln = nn.LayerNorm(param_dtype=self.param_dtype)
        mlp = MLP(
            self.hidden_dim,
            self.n_layers,
            activation=self.activation,
            param_dtype=self.param_dtype,
        )
        head = nn.Dense(
            self.n_categories, param_dtype=self.param_dtype, bias_init=self.bond_bias_init
            )

        h = ln(edge_latent)
        h = mlp(h)
        h = self.activation(h)
        logits = head(h)
        logits_sym = 0.5 * (logits + jnp.swapaxes(logits, -3, -2))

        return logits_sym


__all__ = ["NodeCategoricalDecoder", "EdgeCategoricalDecoder"]
