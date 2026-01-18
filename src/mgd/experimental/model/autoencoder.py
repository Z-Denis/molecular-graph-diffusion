"""Graph autoencoder module combining encoder and decoder components."""

from __future__ import annotations

from flax import linen as nn

from mgd.experimental.dataset.utils import GraphBatch
from mgd.latent import GraphLatent
from mgd.experimental.model.embeddings import GraphEmbedder
from mgd.experimental.model.decoder import GraphDecoder


class GraphAutoencoder(nn.Module):
    """Simple autoencoder wrapping an encoder (GraphEmbedder) and a decoder.

    The encoder produces a :class:`GraphLatent`, the decoder consumes it and
    outputs reconstructed features (e.g., logits). The module exposes explicit
    ``encode`` and ``decode`` methods to make freezing/normalization outside
    the module straightforward.
    """

    embedder: GraphEmbedder
    decoder: GraphDecoder

    def encode(self, graph: GraphBatch) -> GraphLatent:
        """Encode raw graph features into latent node/edge tensors."""
        return self.embedder(graph, graph.node_mask, graph.pair_mask)

    def decode(self, latents: GraphLatent):
        """Decode latent node/edge tensors into reconstructed outputs."""
        return self.decoder(latents)

    def __call__(self, graph: GraphBatch):
        """Run encode->decode and return (reconstruction, latents)."""
        latents = self.encode(graph)
        recon = self.decode(latents)
        return recon, latents


__all__ = ["GraphAutoencoder"]
