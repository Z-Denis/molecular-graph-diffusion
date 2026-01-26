"""Categorical diffusion model with continuous latent embeddings."""

from __future__ import annotations

from typing import Dict, Protocol

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from flax import linen as nn

from ..dataset.utils import GraphBatch
from ..latent import GraphLatent, edge_probs_from_logits, latent_from_scalar, symmetrize_latent
from .embeddings import CategoricalLatentEmbedder


class DenoiserLike(Protocol):
    def __call__(
        self,
        xt: GraphLatent,
        log_sigma: jnp.ndarray,
        *,
        node_mask: jnp.ndarray,
        pair_mask: jnp.ndarray,
    ) -> GraphLatent: ...


class GraphDiffusionModel(nn.Module):
    denoiser: DenoiserLike              # predicts categorical logits
    embedder: CategoricalLatentEmbedder
    sigma_data_node: float              # RMS of node latents (masked)
    sigma_data_edge: float              # RMS of edge latents (masked)
    sigma_min: float                    # training/sampling lower sigma
    sigma_max: float                    # training/sampling upper sigma
    param_dtype: DTypeLike = "float32"

    def _input_scale(self, sigma: jnp.ndarray) -> GraphLatent:
        """Compute input scaling for noisy embeddings."""
        s2 = jnp.square(sigma)
        sd_node2 = jnp.square(self.sigma_data_node)
        sd_edge2 = jnp.square(self.sigma_data_edge)

        expand_node = lambda x: x[..., None, None]
        expand_edge = lambda x: x[..., None, None, None]
        return GraphLatent(
            expand_node(1.0 / jnp.sqrt(s2 + sd_node2)),
            expand_edge(1.0 / jnp.sqrt(s2 + sd_edge2)),
        )

    def predict_logits(
        self,
        x_in: GraphLatent,
        log_sigma: jnp.ndarray,
        *,
        node_mask: jnp.ndarray,
        pair_mask: jnp.ndarray,
    ) -> GraphLatent:
        """Run the denoiser on preconditioned inputs."""
        return self.denoiser(
            x_in,
            log_sigma,
            node_mask=node_mask,
            pair_mask=pair_mask,
        )

    def encode(self, batch: GraphBatch, *, node_mask: jnp.ndarray, pair_mask: jnp.ndarray) -> GraphLatent:
        """Encode categorical graph data into clean latent embeddings."""
        return self.embedder(
            batch.atom_type,
            batch.bond_type,
            node_mask=node_mask,
            pair_mask=pair_mask,
        )

    def _probs_from_logits(
        self,
        logits: GraphLatent,
    ) -> GraphLatent:
        node = jax.nn.softmax(logits.node, axis=-1)
        edge = edge_probs_from_logits(logits.edge)
        return GraphLatent(node=node, edge=edge)

    def _xhat_from_probs(self, probs: GraphLatent) -> GraphLatent:
        return self.embedder.probs_to_latent(probs.node, probs.edge)

    def logits_to_latent(
        self,
        logits: GraphLatent,
    ) -> GraphLatent:
        """Convert categorical logits to expected latent embeddings."""
        probs = self._probs_from_logits(logits)
        return self._xhat_from_probs(probs)

    def denoise(
        self,
        x_noisy: GraphLatent,
        sigma: jnp.ndarray,
        *,
        node_mask: jnp.ndarray,
        pair_mask: jnp.ndarray,
    ) -> Dict[str, GraphLatent]:
        """Predict categorical logits and return embeddings derived from probabilities."""
        c_in = self._input_scale(sigma)
        # assert c_in.node.shape == (sigma.shape[0], 1, 1)
        # assert c_in.edge.shape == (sigma.shape[0], 1, 1, 1)
        log_sigma = jnp.log(sigma + 1e-12)
        logits = self.predict_logits(
            c_in * x_noisy,
            log_sigma,
            node_mask=node_mask,
            pair_mask=pair_mask,
        )
        probs = self._probs_from_logits(logits)
        x_hat = self._xhat_from_probs(probs).masked(node_mask, pair_mask)
        return {"x_hat": x_hat, "logits": logits, "probs": probs}

    def __call__(
        self,
        x0: GraphLatent,
        sigma: jnp.ndarray,
        *,
        node_mask: jnp.ndarray,
        pair_mask: jnp.ndarray,
        noise: GraphLatent | None = None,
    ) -> Dict[str, object]:
        """Forward pass: add noise at level sigma and predict denoised x_hat."""
        if noise is None:
            rng = self.make_rng("noise")
            noise = self.embedder.space.noise_from_masks(rng, node_mask, pair_mask)
        x_noisy = (x0 + latent_from_scalar(sigma) * noise).masked(node_mask, pair_mask)
        x_noisy = symmetrize_latent(x_noisy, node_mask, pair_mask)
        denoise_out = self.denoise(
            x_noisy,
            sigma,
            node_mask=node_mask,
            pair_mask=pair_mask,
        )
        return {
            **denoise_out,
            "noise": noise,
            "noisy": x_noisy,
            "clean": x0,
            "sigma": sigma,
        }


__all__ = ["GraphDiffusionModel"]
