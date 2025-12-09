"""Graph diffusion wrapper combining embedding, noise injection, and denoising."""

from __future__ import annotations

from typing import Dict, Optional

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from flax import linen as nn

from ..dataset.utils import GraphBatch
from ..diffusion.schedules import DiffusionSchedule
from ..latent import GraphLatent, latent_from_scalar
from .embeddings import GraphEmbedder
from .denoiser import MPNNDenoiser


class GraphDiffusionModel(nn.Module):
    """Latent graph diffusion with embedding, forward noise, and denoising.

    The embedder is only needed when a clean graph is provided (e.g. training).
    """
    # TODO: Make API for AbstractDenoiser and AbstractEmbedder and check LatentSpace consistency
    embedder: GraphEmbedder
    denoiser: MPNNDenoiser
    schedule: DiffusionSchedule
    param_dtype: DTypeLike = "float32"

    def encode(self, graph: GraphBatch) -> GraphLatent:
        """Embed raw graph features into latent node/edge tensors."""
        return self.embedder(graph)

    def q_sample(
        self,
        x0: GraphLatent,
        t: jnp.ndarray,
        noise: GraphLatent,
    ) -> GraphLatent:
        """Diffusion forward step: x_t = sqrt(ab_t) x0 + sqrt(1-ab_t) eps."""
        alpha_bar_t = jnp.take(self.schedule.alpha_bar, t)
        sqrt_ab = jnp.sqrt(alpha_bar_t)
        sqrt_om = jnp.sqrt(1.0 - alpha_bar_t)
        scale_ab = latent_from_scalar(sqrt_ab)
        scale_om = latent_from_scalar(sqrt_om)
        return scale_ab * x0 + scale_om * noise

    def predict_eps(
        self,
        xt: GraphLatent,
        t: jnp.ndarray,
        *,
        node_mask: jnp.ndarray,
        pair_mask: jnp.ndarray,
    ) -> GraphLatent:
        """Run the denoiser to estimate noise."""
        return self.denoiser(
            xt,
            t,
            node_mask=node_mask,
            pair_mask=pair_mask,
        )

    def __call__(
        self,
        graph: GraphBatch,
        t: jnp.ndarray,
        noise_nodes: Optional[jnp.ndarray] = None,
        noise_edges: Optional[jnp.ndarray] = None,
    ) -> Dict[str, GraphLatent]:
        """Training forward pass returning noisy latents and predicted noise."""
        x0 = self.encode(graph)
        if noise_nodes is None or noise_edges is None:
            rng = self.make_rng("noise")
            rng_n, rng_e = jax.random.split(rng)
            noise_nodes = jax.random.normal(rng_n, x0.node.shape, dtype=x0.node.dtype)
            noise_edges = jax.random.normal(rng_e, x0.edge.shape, dtype=x0.edge.dtype)

        noise = GraphLatent(noise_nodes, noise_edges)
        xt = self.q_sample(x0, t, noise)
        eps = self.predict_eps(
            xt,
            t,
            node_mask=graph.node_mask,
            pair_mask=graph.pair_mask,
        )
        return {
            "eps_pred": eps,
            "noise": noise,
            "noisy": xt,
            "clean": x0,
        }


__all__ = ["GraphDiffusionModel"]
