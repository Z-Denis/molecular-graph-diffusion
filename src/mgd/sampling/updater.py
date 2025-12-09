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

        coef1 = 1.0 / jnp.sqrt(alpha_t)
        coef2 = beta_t / jnp.sqrt(1.0 - alpha_bar_t)

        coef1_latent = latent_from_scalar(coef1)
        coef2_latent = latent_from_scalar(coef2)
        mean = coef1_latent * (xt - coef2_latent * eps_pred)

        sigma = jnp.where(t > 0, jnp.sqrt(beta_t), jnp.zeros_like(beta_t))
        sigma_latent = latent_from_scalar(sigma)

        if (sigma > 0).any() and rng is None:
            raise ValueError("rng is required for stochastic DDPM updates when t > 0.")

        if rng is None:
            noise_latent = GraphLatent(jnp.zeros_like(xt.node), jnp.zeros_like(xt.edge))
        else:
            rng_n, rng_e = jax.random.split(rng)
            noise_nodes = jax.random.normal(rng_n, xt.node.shape, dtype=xt.node.dtype)
            noise_edges = jax.random.normal(rng_e, xt.edge.shape, dtype=xt.edge.dtype)
            noise_latent = GraphLatent(noise_nodes, noise_edges)

        x_prev = mean + sigma_latent * noise_latent
        return x_prev.masked(node_mask, pair_mask)


class DDIMUpdater(BaseUpdater):
    """Placeholder for deterministic DDIM updates."""

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
        # TODO: implement DDIM update; delegating to DDPM for now.
        beta_t = jnp.take(self.schedule.betas, t)
        alpha_t = jnp.take(self.schedule.alphas, t)
        alpha_bar_t = jnp.take(self.schedule.alpha_bar, t)

        coef1 = 1.0 / jnp.sqrt(alpha_t)
        coef2 = beta_t / jnp.sqrt(1.0 - alpha_bar_t)

        coef1_latent = latent_from_scalar(coef1)
        coef2_latent = latent_from_scalar(coef2)
        mean = coef1_latent * (xt - coef2_latent * eps_pred)
        return mean.masked(node_mask, pair_mask)


__all__ = ["BaseUpdater", "DDPMUpdater", "DDIMUpdater"]
