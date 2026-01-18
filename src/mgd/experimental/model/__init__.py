"""Experimental model package (legacy + prototypes)."""

from .autoencoder import GraphAutoencoder
from .autoencoder_onehot import OneHotAutoencoder
from .decoder import GraphDecoder, NodeCategoricalDecoder, EdgeCategoricalDecoder
from .embeddings import (
    GraphEmbedder,
    OneHotGraphEmbedder,
    NodeEmbedder,
    PairEmbedder,
    TimeEmbedding,
    sinusoidal_time_embedding,
)
from .utils import MLP, aggregate_node_edge, bond_bias_initializer, estimate_latent_stats_masked

__all__ = [
    "GraphAutoencoder",
    "OneHotAutoencoder",
    "GraphDecoder",
    "NodeCategoricalDecoder",
    "EdgeCategoricalDecoder",
    "GraphEmbedder",
    "OneHotGraphEmbedder",
    "NodeEmbedder",
    "PairEmbedder",
    "TimeEmbedding",
    "sinusoidal_time_embedding",
    "MLP",
    "aggregate_node_edge",
    "bond_bias_initializer",
    "estimate_latent_stats_masked",
]
