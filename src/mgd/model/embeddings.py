"""Flax Linen modules for atom and bond embeddings."""

from __future__ import annotations

from typing import Callable

from jax.typing import DTypeLike
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
    param_dtype: DTypeLike = "float32"

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
        atom_emb = nn.Embed(self.atom_vocab, self.atom_dim, name="atom_embedding", param_dtype=self.param_dtype)(atom_ids)
        hybrid_emb = nn.Embed(self.hybrid_vocab, self.hybrid_dim, name="hybrid_embedding", param_dtype=self.param_dtype)(hybrid_ids)
        cont_emb = nn.Dense(self.cont_dim, name="cont_projection", param_dtype=self.param_dtype)(node_continuous)

        fused = jnp.concatenate([atom_emb, hybrid_emb, cont_emb], axis=-1)
        h = nn.Dense(self.hidden_dim, name="fuse", param_dtype=self.param_dtype)(fused)
        h = self.activation(h)
        h = nn.Dense(self.hidden_dim, name="output", param_dtype=self.param_dtype)(h)
        return h


class EdgeEmbedder(nn.Module):
    """Embed bond categorical features into a shared hidden space."""

    edge_vocab: int
    edge_dim: int
    hidden_dim: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    param_dtype: DTypeLike = "float32"

    @nn.compact
    def __call__(self, edge_types: jnp.ndarray) -> jnp.ndarray:
        """Returns (batch, n_atoms, n_atoms, hidden_dim)."""
        emb = nn.Embed(self.edge_vocab, self.edge_dim, name="edge_embedding", param_dtype=self.param_dtype)(edge_types)
        h = nn.Dense(self.hidden_dim, name="fuse", param_dtype=self.param_dtype)(emb)
        h = self.activation(h)
        h = nn.Dense(self.hidden_dim, name="output", param_dtype=self.param_dtype)(h)
        return h


def sinusoidal_time_embedding(timesteps: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Sinusoidal positional embedding for diffusion timesteps.

    Args:
        timesteps: int32/float32 array of shape (batch,) or with leading batch dims.
        dim: embedding dimension (must be even).
    Returns:
        Array shaped like timesteps + (dim,)

    Formula (i < dim/2):
        emb[..., 2*i]   = sin(t * 10000^{-2i/(dim-2)})
        emb[..., 2*i+1] = cos(t * 10000^{-2i/(dim-2)})
    """
    half = dim // 2
    if dim % 2 != 0 or half < 1:
        raise ValueError("sinusoidal_time_embedding requires even dim >= 2.")
    timesteps = timesteps.astype(jnp.float32)
    if half == 1:
        freqs = jnp.array([1.0], dtype=jnp.float32)
    else:
        freqs = jnp.exp(
            -jnp.log(10000.0) * jnp.arange(half, dtype=jnp.float32) / float(half - 1)
        )
    args = timesteps[..., None] * freqs
    emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
    return emb


class TimeEmbedding(nn.Module):
    time_dim: int
    node_dim: int
    edge_dim: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    param_dtype: DTypeLike = "float32"

    @nn.compact
    def __call__(self, times):
        emb = sinusoidal_time_embedding(times, self.time_dim)
        h = nn.Dense(features=self.time_dim, param_dtype=self.param_dtype)(emb)
        h = self.activation(h)
        t_nodes = nn.Dense(features=self.node_dim, param_dtype=self.param_dtype)(h)
        t_edges = nn.Dense(features=self.edge_dim, param_dtype=self.param_dtype)(h)
        
        return t_nodes, t_edges

__all__ = ["NodeEmbedder", "EdgeEmbedder", "sinusoidal_time_embedding", "TimeEmbedding"]
