"""Flax Linen modules for atom and bond embeddings."""

from __future__ import annotations

from typing import Callable, Tuple

import jax
from jax.typing import DTypeLike
import jax.numpy as jnp
from flax import linen as nn

from ..dataset.encoding import ATOM_VOCAB_SIZE, HYBRID_VOCAB_SIZE, BOND_VOCAB_SIZE
from ..dataset.utils import GraphBatch
from ..latent import GraphLatent, GraphLatentSpace


class NodeEmbedder(nn.Module):
    """Embed atom categorical + continuous features into a shared hidden space (no mask!).

    Args:
        atom_vocab: Vocabulary size for atom embeddings (include padding slot).
        hybrid_vocab: Vocabulary size for hybridization embeddings (include padding slot).
        atom_embed_dim: Output size for atom-type embedding.
        hybrid_embed_dim: Output size for hybridization embedding.
        cont_embed_dim: Output size for continuous projection.
        hidden_dim: Final hidden dimension after fusion.
        activation: Nonlinearity applied before the last projection.
    """

    atom_vocab: int
    hybrid_vocab: int
    atom_embed_dim: int
    hybrid_embed_dim: int
    cont_embed_dim: int
    hidden_dim: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    scale: float = 1.0
    param_dtype: DTypeLike = "float32"

    @nn.compact
    def __call__(self, atom_ids: jnp.ndarray, hybrid_ids: jnp.ndarray, node_continuous: jnp.ndarray) -> jnp.ndarray:
        """Returns (batch, n_atoms, hidden_dim).

        Example:
            >>> import jax
            >>> import jax.numpy as jnp
            >>> model = NodeEmbedder(atom_vocab=6, hybrid_vocab=4, atom_embed_dim=8, hybrid_embed_dim=4, cont_embed_dim=8, hidden_dim=32)
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
        atom_emb = nn.Embed(self.atom_vocab, self.atom_embed_dim, name="atom_embedding", param_dtype=self.param_dtype)(atom_ids)
        hybrid_emb = nn.Embed(self.hybrid_vocab, self.hybrid_embed_dim, name="hybrid_embedding", param_dtype=self.param_dtype)(hybrid_ids)
        cont_emb = nn.Dense(self.cont_embed_dim, name="cont_projection", param_dtype=self.param_dtype)(node_continuous)

        fused = jnp.concatenate([atom_emb, hybrid_emb, cont_emb], axis=-1)
        h = nn.Dense(self.hidden_dim, name="fuse", param_dtype=self.param_dtype)(fused)
        h = self.activation(h)
        h = nn.Dense(
            self.hidden_dim,
            name="output",
            use_bias=False,
            kernel_init=jax.nn.initializers.he_normal(),
            param_dtype=self.param_dtype,
        )(h)
        h = nn.LayerNorm()(h)
        return self.scale * h


class EdgeEmbedder(nn.Module):
    """Embed bond categorical features into a shared hidden space (no mask!)."""

    edge_vocab: int
    edge_embed_dim: int
    hidden_dim: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    scale: float = 1.0
    param_dtype: DTypeLike = "float32"

    @nn.compact
    def __call__(self, edge_types: jnp.ndarray) -> jnp.ndarray:
        """Returns (batch, n_atoms, n_atoms, hidden_dim)."""
        emb = nn.Embed(self.edge_vocab, self.edge_embed_dim, name="edge_embedding", param_dtype=self.param_dtype)(edge_types)
        h = nn.Dense(self.hidden_dim, name="fuse", param_dtype=self.param_dtype)(emb)
        h = self.activation(h)
        h = nn.Dense(
            self.hidden_dim,
            name="output",
            use_bias=False,
            kernel_init=jax.nn.initializers.he_normal(),
            param_dtype=self.param_dtype,
        )(h)
        h = nn.LayerNorm()(h)
        return self.scale * h


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
    def __call__(self, times: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        emb = sinusoidal_time_embedding(times, self.time_dim)
        h = nn.Dense(features=self.time_dim, param_dtype=self.param_dtype)(emb)
        h = self.activation(h)
        t_nodes = nn.Dense(features=self.node_dim, param_dtype=self.param_dtype, use_bias=False)(h)
        t_edges = nn.Dense(features=self.edge_dim, param_dtype=self.param_dtype, use_bias=False)(h)
        
        return t_nodes, t_edges


class GraphEmbedder(nn.Module):
    """Embed raw graph categorical/continuous features into latent node/edge tensors."""

    space: GraphLatentSpace
    
    atom_embed_dim: int
    hybrid_embed_dim: int
    cont_embed_dim: int
    edge_embed_dim: int    # embed dim for edges

    atom_vocab_dim: int = ATOM_VOCAB_SIZE
    hybrid_vocab_dim: int = HYBRID_VOCAB_SIZE
    edge_vocab_dim: int = BOND_VOCAB_SIZE

    node_scale: float = 1.0
    edge_scale: float = 1.0

    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    param_dtype: DTypeLike = "float32"

    @nn.compact
    def __call__(
        self, 
        graph: GraphBatch, 
        node_mask: jnp.ndarray, 
        pair_mask: jnp.ndarray
        ) -> GraphLatent:
        node_emb = NodeEmbedder(
            self.atom_vocab_dim,
            self.hybrid_vocab_dim,
            self.atom_embed_dim,
            self.hybrid_embed_dim,
            self.cont_embed_dim,
            self.space.node_dim,
            self.activation,
            param_dtype=self.param_dtype,
            scale=self.node_scale,
            name="node_embedding",
        )
        edge_emb = EdgeEmbedder(
            self.edge_vocab_dim,
            self.edge_embed_dim,
            self.space.edge_dim,
            self.activation,
            param_dtype=self.param_dtype,
            scale=self.edge_scale,
            name="edge_embedding",
        )

        nodes = node_emb(
            graph.atom_type,
            graph.hybrid,
            graph.cont,
        )
        edges = edge_emb(graph.edges)

        return GraphLatent(nodes, edges).masked(node_mask, pair_mask)


__all__ = ["NodeEmbedder", "EdgeEmbedder", "GraphEmbedder", "sinusoidal_time_embedding", "TimeEmbedding"]
