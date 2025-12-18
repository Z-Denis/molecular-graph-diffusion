"""EDM-style latent diffusion with external preconditioning.

Forward noise: x_sigma = x + sigma * eps, eps ~ N(0, I).
Prediction: x_hat = c_skip * x_sigma + c_out * f_theta(c_in * x_sigma, log_sigma),
where
    c_in   = 1 / sqrt(sigma^2 + sigma_data^2)
    c_skip = sigma_data^2 / (sigma^2 + sigma_data^2)
    c_out  = sigma * sigma_data / sqrt(sigma^2 + sigma_data^2)
Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models", 2022.
"""

from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from flax import linen as nn

from ..latent import GraphLatent, latent_from_scalar
from .denoiser import MPNNDenoiser


class GraphDiffusionModel(nn.Module):
    denoiser: MPNNDenoiser              # outputs f_theta
    sigma_data_node: float              # RMS of node latents (masked)
    sigma_data_edge: float              # RMS of edge latents (masked)
    sigma_min: float                    # training/sampling lower sigma
    sigma_max: float                    # training/sampling upper sigma
    param_dtype: DTypeLike = "float32"

    def _coeffs(self, sigma: jnp.ndarray):
        """Compute EDM scaling coefficients."""
        s2 = jnp.square(sigma)
        sd_node2 = jnp.square(self.sigma_data_node)
        sd_edge2 = jnp.square(self.sigma_data_edge)

        expand_node = lambda x: x[..., None, None]
        expand_edge = lambda x: x[..., None, None, None]
        c_in = GraphLatent(expand_node(1.0 / jnp.sqrt(s2 + sd_node2)), expand_edge(1.0 / jnp.sqrt(s2 + sd_edge2)))
        c_skip = GraphLatent(expand_node(sd_node2 / (s2 + sd_node2)), expand_edge(sd_edge2 / (s2 + sd_edge2)))
        c_out = GraphLatent(
            expand_node(sigma * self.sigma_data_node / jnp.sqrt(s2 + sd_node2)),
            expand_edge(sigma * self.sigma_data_edge / jnp.sqrt(s2 + sd_edge2)),
        )

        return c_in, c_skip, c_out

    def predict_f(
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

    def denoise(
        self,
        x_noisy: GraphLatent,
        sigma: jnp.ndarray,
        *,
        node_mask: jnp.ndarray,
        pair_mask: jnp.ndarray,
    ) -> GraphLatent:
        """Apply EDM preconditioning and return x_hat."""
        c_in, c_skip, c_out = self._coeffs(sigma)
        # assert c_in.node.shape == (sigma.shape[0], 1, 1)
        # assert c_in.edge.shape == (sigma.shape[0], 1, 1, 1)
        log_sigma = jnp.log(sigma + 1e-12)
        f = self.predict_f(
            c_in * x_noisy,
            log_sigma,
            node_mask=node_mask,
            pair_mask=pair_mask,
        )
        x_hat = c_skip * x_noisy + c_out * f
        x_hat = x_hat.masked(node_mask, pair_mask)
        return x_hat

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
            rng_n, rng_e = jax.random.split(rng)
            noise = GraphLatent(
                jax.random.normal(rng_n, x0.node.shape, dtype=x0.node.dtype),
                jax.random.normal(rng_e, x0.edge.shape, dtype=x0.edge.dtype),
            )
        x_noisy = (x0 + latent_from_scalar(sigma) * noise).masked(node_mask, pair_mask)
        x_hat = self.denoise(
            x_noisy,
            sigma,
            node_mask=node_mask,
            pair_mask=pair_mask,
        )
        return {
            "x_hat": x_hat,
            "noise": noise,
            "noisy": x_noisy,
            "clean": x0,
            "sigma": sigma,
        }


__all__ = ["GraphDiffusionModel"]
