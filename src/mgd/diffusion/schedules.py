"""Noise schedule utilities for diffusion models."""

from __future__ import annotations

import jax.numpy as jnp
from jax.typing import DTypeLike
import flax


@flax.struct.dataclass
class DiffusionSchedule:
    """Container for betas, alphas, and cumulative alphas."""

    betas: jnp.ndarray
    alphas: jnp.ndarray
    alpha_bar: jnp.ndarray


def cosine_beta_schedule(
    timesteps: int,
    s: float = 0.008,
    dtype: DTypeLike = jnp.float32,
    clip_min: float = 1e-8,
    clip_max: float = 0.999,
) -> DiffusionSchedule:
    """Cosine schedule from Nichol & Dhariwal (2021).

    Alpha_hat(t) = cos^2(((t/T + s)/(1+s)) * pi/2) / cos^2(s/(1+s) * pi/2)
    betas_t = 1 - Alpha_hat(t+1)/Alpha_hat(t)
    """
    steps = jnp.arange(timesteps + 1, dtype=dtype)
    f = lambda t: jnp.cos(((t / timesteps) + s) / (1.0 + s) * jnp.pi / 2.0) ** 2
    alpha_bar = f(steps) / f(0)
    betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
    betas = jnp.clip(betas, clip_min, clip_max)
    alphas = 1.0 - betas
    alpha_bar = jnp.cumprod(alphas)
    return DiffusionSchedule(betas=betas, alphas=alphas, alpha_bar=alpha_bar)


__all__ = ["DiffusionSchedule", "cosine_beta_schedule"]
