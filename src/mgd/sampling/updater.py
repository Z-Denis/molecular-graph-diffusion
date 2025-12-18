"""Reverse-step policies for diffusion sampling."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import jax
import jax.numpy as jnp

from mgd.diffusion.schedules import DiffusionSchedule
from mgd.latent import GraphLatent, latent_from_scalar


class BaseUpdater(ABC):
    """Abstract base class for reverse diffusion updates."""

    def __init__(self, schedule: DiffusionSchedule):
        self.schedule = schedule

    @abstractmethod
    def step(
        self,
        xt: GraphLatent,
        eps_pred: GraphLatent,
        t: jnp.ndarray,
        node_mask: jnp.ndarray,
        pair_mask: jnp.ndarray,
        rng: Optional[jax.Array] = None,
    ) -> GraphLatent:
        raise NotImplementedError


class DDPMUpdater(BaseUpdater):
    """Standard DDPM reverse update."""

    def step(
        self,
        xt: GraphLatent,
        eps_pred: GraphLatent,
        t: jnp.ndarray,
        node_mask: jnp.ndarray,
        pair_mask: jnp.ndarray,
        rng: Optional[jax.Array] = None,
    ) -> GraphLatent:
        beta_t = jnp.take(self.schedule.betas, t)
        alpha_t = jnp.take(self.schedule.alphas, t)
        alpha_bar_t = jnp.take(self.schedule.alpha_bar, t)
        alpha_bar_prev = jnp.take(self.schedule.alpha_bar, jnp.maximum(t - 1, 0))

        coef1 = 1.0 / jnp.sqrt(alpha_t)
        coef2 = beta_t / jnp.sqrt(1.0 - alpha_bar_t)

        coef1_latent = latent_from_scalar(coef1)
        coef2_latent = latent_from_scalar(coef2)
        mean = coef1_latent * (xt - coef2_latent * eps_pred)

        # Use posterior variance beta_tilde (Ho et al. 2020)
        beta_tilde = beta_t * (1.0 - alpha_bar_prev) / jnp.maximum(1.0 - alpha_bar_t, 1e-8)
        sigma = jnp.where(t > 0, jnp.sqrt(beta_tilde), jnp.zeros_like(beta_t))
        sigma_latent = latent_from_scalar(sigma)

        rng_n, rng_e = jax.random.split(rng)
        noise_nodes = jax.random.normal(rng_n, xt.node.shape, dtype=xt.node.dtype)
        noise_edges = jax.random.normal(rng_e, xt.edge.shape, dtype=xt.edge.dtype)
        noise_latent = GraphLatent(noise_nodes, noise_edges)

        x_prev = mean + sigma_latent * noise_latent
        return x_prev.masked(node_mask, pair_mask)


class DDIMUpdater(BaseUpdater):
    """Deterministic DDIM-style update (eta controls stochasticity; default 0)."""

    def __init__(self, schedule: DiffusionSchedule, eta: float = 0.0):
        super().__init__(schedule)
        self.eta = eta

    def step(
        self,
        xt: GraphLatent,
        eps_pred: GraphLatent,
        t: jnp.ndarray,
        node_mask: jnp.ndarray,
        pair_mask: jnp.ndarray,
        rng: Optional[jax.Array] = None,
    ) -> GraphLatent:
        # DDIM update (eta=0 -> deterministic). See Song et al. 2021.
        alpha_bar_t = jnp.take(self.schedule.alpha_bar, t)
        alpha_bar_prev = jnp.take(self.schedule.alpha_bar, jnp.maximum(t - 1, 0))
        sqrt_ab_t = jnp.sqrt(alpha_bar_t)
        sqrt_ab_prev = jnp.sqrt(alpha_bar_prev)
        sigma = self.eta * jnp.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))

        eps_scale = latent_from_scalar(sigma)
        noise_latent = GraphLatent(jnp.zeros_like(xt.node), jnp.zeros_like(xt.edge))
        if self.eta != 0.0 and rng is not None:
            rng_n, rng_e = jax.random.split(rng)
            noise_latent = GraphLatent(
                jax.random.normal(rng_n, xt.node.shape, dtype=xt.node.dtype),
                jax.random.normal(rng_e, xt.edge.shape, dtype=xt.edge.dtype),
            )

        xt_minus = (
            latent_from_scalar(sqrt_ab_prev) * (xt - latent_from_scalar(jnp.sqrt(1 - alpha_bar_t)) * eps_pred) / sqrt_ab_t
            + eps_scale * noise_latent
        )
        return xt_minus.masked(node_mask, pair_mask)


__all__ = ["BaseUpdater", "DDPMUpdater", "DDIMUpdater"]
