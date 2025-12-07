"""Flax Linen modules for atom and bond embeddings."""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
from flax import linen as nn


class NodeEmbedder(nn.Module):
    """Embed atom categorical + continuous features into a shared hidden space.

    Args:
        atom_vocab: Vocabulary size for atom embeddings (include padding slot).
        hybrid_vocab: Vocabulary size for hybridization embeddings (include padding slot).
        atom_dim: Output size for atom-type embedding.
        hybrid_dim: Output size for hybridization embedding.
        cont_dim: Output size for continuous projection.
        hidden_dim: Final hidden dimension after fusion.
        activation: Nonlinearity applied before the last projection.
    """

    atom_vocab: int
    hybrid_vocab: int
    atom_dim: int
    hybrid_dim: int
    cont_dim: int
    hidden_dim: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu

    @nn.compact
    def __call__(self, atom_ids: jnp.ndarray, hybrid_ids: jnp.ndarray, node_continuous: jnp.ndarray) -> jnp.ndarray:
        """Returns (batch, n_atoms, hidden_dim).

        Example:
            >>> import jax
            >>> import jax.numpy as jnp
            >>> model = NodeEmbedder(atom_vocab=6, hybrid_vocab=4, atom_dim=8, hybrid_dim=4, cont_dim=8, hidden_dim=32)
            >>> variables = model.init(
            ...     jax.random.PRNGKey(0),
            ...     jnp.zeros((2, 29), dtype=jnp.int32),
            ...     jnp.zeros((2, 29), dtype=jnp.int32),
            ...     jnp.zeros((2, 29, 4)),
            ... )
            >>> out = model.apply(
            ...     variables,
            ...     jnp.zeros((2, 29), dtype=jnp.int32),
            ...     jnp.zeros((2, 29), dtype=jnp.int32),
            ...     jnp.zeros((2, 29, 4)),
            ... )
            >>> out.shape
            (2, 29, 32)
        """
        atom_emb = nn.Embed(self.atom_vocab, self.atom_dim, name="atom_embedding")(atom_ids)
        hybrid_emb = nn.Embed(self.hybrid_vocab, self.hybrid_dim, name="hybrid_embedding")(hybrid_ids)
        cont_emb = nn.Dense(self.cont_dim, name="cont_projection")(node_continuous)

        fused = jnp.concatenate([atom_emb, hybrid_emb, cont_emb], axis=-1)
        h = nn.Dense(self.hidden_dim, name="fuse")(fused)
        h = self.activation(h)
        h = nn.Dense(self.hidden_dim, name="output")(h)
        return h


class EdgeEmbedder(nn.Module):
    """Embed bond categorical features into a shared hidden space."""

    edge_vocab: int
    edge_dim: int
    hidden_dim: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu

    @nn.compact
    def __call__(self, edge_types: jnp.ndarray) -> jnp.ndarray:
        """Returns (batch, n_atoms, n_atoms, hidden_dim)."""
        emb = nn.Embed(self.edge_vocab, self.edge_dim, name="edge_embedding")(edge_types)
        h = nn.Dense(self.hidden_dim, name="fuse")(emb)
        h = self.activation(h)
        h = nn.Dense(self.hidden_dim, name="output")(h)
        return h


__all__ = ["NodeEmbedder", "EdgeEmbedder"]
