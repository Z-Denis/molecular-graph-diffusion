"""Flax Linen modules for atom and bond embeddings."""

from __future__ import annotations

from typing import Callable, Tuple

import jax
from jax.typing import DTypeLike
import jax.numpy as jnp
from flax import linen as nn

from ..dataset.qm9 import ATOM_VOCAB_SIZE, HYBRID_VOCAB_SIZE, BOND_VOCAB_SIZE
from ..dataset.utils import GraphBatch
from ..latent import GraphLatent, GraphLatentSpace
from ..latent.utils import latent_from_probs, normalize_embeddings


class OneHotGraphEmbedder(nn.Module):
    """Parameter-free embedder that returns masked one-hot latents."""

    space: GraphLatentSpace

    @nn.compact
    def __call__(
        self,
        graph: GraphBatch,
        node_mask: jnp.ndarray | None = None,
        pair_mask: jnp.ndarray | None = None,
    ) -> GraphLatent:
        if node_mask is None:
            node_mask = graph.node_mask
        if pair_mask is None:
            pair_mask = graph.pair_mask
        node_onehot = jax.nn.one_hot(graph.atom_type, self.space.node_dim, dtype=self.space.dtype)
        edge_onehot = jax.nn.one_hot(graph.bond_type, self.space.edge_dim, dtype=self.space.dtype)
        node_onehot = node_onehot * node_mask[..., None]
        edge_onehot = edge_onehot * pair_mask[..., None]
        return GraphLatent(node=node_onehot, edge=edge_onehot)


class CategoricalLatentEmbedder(nn.Module):
    """Trainable categorical embedder with hypersphere-normalized embeddings."""

    space: GraphLatentSpace
    node_vocab: int
    edge_vocab: int
    eps: float = 1e-8
    param_dtype: DTypeLike = "float32"

    def setup(self) -> None:
        self._v_node = self.param(
            "node_embeddings",
            nn.initializers.normal(1.0),
            (self.node_vocab, self.space.node_dim),
        )
        self._v_edge = self.param(
            "edge_embeddings",
            nn.initializers.normal(1.0),
            (self.edge_vocab, self.space.edge_dim),
        )

    def embeddings(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        node = normalize_embeddings(self._v_node, eps=self.eps)
        edge = normalize_embeddings(self._v_edge, eps=self.eps)
        return node, edge

    def __call__(
        self,
        node_labels: jnp.ndarray,
        edge_labels: jnp.ndarray,
        *,
        node_mask: jnp.ndarray | None = None,
        pair_mask: jnp.ndarray | None = None,
    ) -> GraphLatent:
        node_emb, edge_emb = self.embeddings()
        nodes = node_emb[node_labels]
        edges = edge_emb[edge_labels]
        if node_mask is not None:
            nodes = nodes * node_mask[..., None]
        if pair_mask is not None:
            edges = edges * pair_mask[..., None]
        return GraphLatent(node=nodes, edge=edges)

    def probs_to_latent(self, node_probs: jnp.ndarray, edge_probs: jnp.ndarray) -> GraphLatent:
        node_emb, edge_emb = self.embeddings()
        return latent_from_probs(node_probs, edge_probs, node_emb, edge_emb)


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


class PairEmbedder(nn.Module):
    """Embed bond categorical + continuous features into a shared hidden space (no mask!)."""

    bond_vocab: int

    bond_embed_dim: int
    bond_cont_dim: int
    
    hidden_dim: int

    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    scale: float = 1.0
    param_dtype: DTypeLike = "float32"

    @nn.compact
    def __call__(self, bond_types: jnp.ndarray, bond_cont: jnp.ndarray | None = None) -> jnp.ndarray:
        """Returns (batch, n_atoms, n_atoms, hidden_dim)."""
        emb = nn.Embed(self.bond_vocab, self.bond_embed_dim, name="bond_embedding", param_dtype=self.param_dtype)(bond_types)
        proj = nn.Dense(
            self.bond_cont_dim or self.bond_embed_dim,
            name="pair_cont_projection",
            param_dtype=self.param_dtype,
        )(bond_cont)
        h = jnp.concatenate([emb, proj], axis=-1)
        h = nn.Dense(self.hidden_dim, name="fuse", param_dtype=self.param_dtype)(h)
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
    atom_cont_embed_dim: int
    
    bond_embed_dim: int
    bond_cont_embed_dim: int

    atom_vocab_dim: int = ATOM_VOCAB_SIZE
    hybrid_vocab_dim: int = HYBRID_VOCAB_SIZE
    bond_vocab_dim: int = BOND_VOCAB_SIZE

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
            self.atom_cont_embed_dim,
            self.space.node_dim,
            self.activation,
            param_dtype=self.param_dtype,
            scale=self.node_scale,
            name="node_embedding",
        )
        edge_emb = PairEmbedder(
            bond_vocab=self.bond_vocab_dim,
            bond_embed_dim=self.bond_embed_dim,
            bond_cont_dim=self.bond_cont_embed_dim,
            hidden_dim=self.space.edge_dim,
            activation=self.activation,
            param_dtype=self.param_dtype,
            scale=self.edge_scale,
            name="edge_embedding",
        )

        nodes = node_emb(
            graph.atom_type,
            graph.hybrid,
            graph.cont,
        )
        edges = edge_emb(graph.bond_type, graph.dknn)

        return GraphLatent(nodes, edges).masked(node_mask, pair_mask)


__all__ = ["NodeEmbedder", "PairEmbedder", "GraphEmbedder", "sinusoidal_time_embedding", "TimeEmbedding"]
