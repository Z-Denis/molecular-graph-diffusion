"""Trivial autoencoder for one-hot latents: identity decode."""

from __future__ import annotations

import flax.linen as nn
from mgd.latent import GraphLatent
from mgd.model.embeddings import OneHotGraphEmbedder
from mgd.dataset.utils import GraphBatch


class OneHotAutoencoder(nn.Module):
    """Autoencoder where encode is one-hot and decode is identity."""

    embedder: OneHotGraphEmbedder

    def encode(self, batch: GraphBatch):
        return self.embedder(batch, batch.node_mask, batch.pair_mask)

    def decode(self, latents: GraphLatent):
        return latents

    def __call__(self, batch: GraphBatch):
        latents = self.encode(batch)
        recon = self.decode(latents)
        return recon, latents


__all__ = ["OneHotAutoencoder"]
