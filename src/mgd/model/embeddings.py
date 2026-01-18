"""Flax Linen modules for categorical embeddings and time conditioning."""

from __future__ import annotations

from typing import Callable, Tuple

import jax.numpy as jnp
from jax.typing import DTypeLike
from flax import linen as nn

from ..latent import GraphLatent, GraphLatentSpace
from ..latent.utils import latent_from_probs, normalize_embeddings
from ..dataset.qm9 import MAX_NODES


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


def sinusoidal_time_embedding(timesteps: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Sinusoidal positional embedding for diffusion timesteps."""
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


class NodeCountEmbedding(nn.Module):
    embed_dim: int
    node_dim: int
    edge_dim: int
    max_nodes: int = MAX_NODES
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    param_dtype: DTypeLike = "float32"

    @nn.compact
    def __call__(self, n_nodes: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        n_nodes = jnp.asarray(n_nodes, dtype=jnp.int32)
        n_nodes = jnp.clip(n_nodes, 0, self.max_nodes)

        emb = nn.Embed(
            num_embeddings=self.max_nodes + 1,
            features=self.embed_dim,
            param_dtype=self.param_dtype,
        )(n_nodes)

        h = nn.Dense(self.embed_dim, param_dtype=self.param_dtype)(emb)
        h = self.activation(h)

        n_nodes_nodes = nn.Dense(self.node_dim, param_dtype=self.param_dtype)(h)
        n_nodes_edges = nn.Dense(self.edge_dim, param_dtype=self.param_dtype)(h)

        return n_nodes_nodes, n_nodes_edges


__all__ = [
    "CategoricalLatentEmbedder",
    "sinusoidal_time_embedding",
    "TimeEmbedding",
    "NodeCountEmbedding",
]
