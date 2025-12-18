"""EDM-style sigma schedules and samplers.

Forward noise: x_sigma = x + sigma * eps, eps ~ N(0, I).
Sampling grid (Karras et al. 2022):
    sigma_i = (sigma_max^(1/rho) + i/(N-1) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def sample_sigma(rng, shape, sigma_min: float, sigma_max: float) -> jnp.ndarray:
    """Log-uniform sigma for training."""
    u = jax.random.uniform(rng, shape, minval=0.0, maxval=1.0)
    log_sigma = jnp.log(sigma_min) + u * (jnp.log(sigma_max) - jnp.log(sigma_min))
    return jnp.exp(log_sigma)


def sample_sigma_mixture(
    rng,
    shape,
    sigma_min: float,
    sigma_max: float,
    *,
    p_low: float = 0.3,
    k: float = 3.0,
) -> jnp.ndarray:
    """Mixture sigma sampler that oversamples low sigmas.

    With prob ``p_low``: log(sigma) ~ U(log sigma_min, log(k * sigma_min));
    else:                log(sigma) ~ U(log sigma_min, log sigma_max).
    """
    rng_flag, rng_low, rng_full = jax.random.split(rng, 3)
    take_low = jax.random.bernoulli(rng_flag, p=p_low, shape=shape)
    log_low = jax.random.uniform(rng_low, shape, minval=jnp.log(sigma_min), maxval=jnp.log(k * sigma_min))
    log_full = jax.random.uniform(rng_full, shape, minval=jnp.log(sigma_min), maxval=jnp.log(sigma_max))
    log_sigma = jnp.where(take_low, log_low, log_full)
    return jnp.exp(log_sigma)


def make_sigma_schedule(
    sigma_min: float,
    sigma_max: float,
    rho: float = 7.0,
    num_steps: int = 40,
) -> jnp.ndarray:
    """Power-law (EDM) decreasing sigma grid."""
    i = jnp.arange(num_steps, dtype=jnp.float32)
    inv_rho = 1.0 / rho
    sigmas = (sigma_max ** inv_rho + i / jnp.maximum(num_steps - 1, 1) * (sigma_min ** inv_rho - sigma_max ** inv_rho)) ** rho
    return sigmas


__all__ = ["sample_sigma", "sample_sigma_mixture", "make_sigma_schedule"]
