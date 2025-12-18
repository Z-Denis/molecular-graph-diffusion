"""Evaluation utilities for diffusion models."""

from __future__ import annotations

from functools import partial
from typing import Callable, Dict, Iterable

import jax
import jax.numpy as jnp

from mgd.dataset.utils import GraphBatch
from mgd.training.losses import edm_masked_mse
from mgd.diffusion.schedules import sample_sigma


@partial(jax.jit, static_argnames=["apply_fn", "eval_fn"])
def eval_step(
    params,
    apply_fn,
    latents,
    node_mask: jnp.ndarray,
    pair_mask: jnp.ndarray,
    rng: jax.Array,
    sigma_min: float,
    sigma_max: float,
    eval_fn: Callable,
) -> Dict[str, jnp.ndarray]:
    """Single evaluation step with masked EDM x0-prediction loss."""
    rng_s, rng_noise = jax.random.split(rng)
    batch_size = latents.node.shape[0]
    sigma = sample_sigma(rng_s, (batch_size,), sigma_min, sigma_max)
    outputs = apply_fn(
        {"params": params},
        latents,
        sigma,
        node_mask=node_mask,
        pair_mask=pair_mask,
        rngs={"noise": rng_noise},
    )
    loss, parts = eval_fn(outputs["x_hat"], outputs["clean"], node_mask, pair_mask, sigma=sigma)
    return {"loss": loss, "sigma_mean": sigma.mean(), **parts}


def evaluate(
    state,
    loader: Iterable[GraphBatch],
    rng: jax.Array,
    eval_fn: Callable = edm_masked_mse,
) -> Dict[str, jnp.ndarray]:
    """Run evaluation over a loader and return mean metrics."""
    sigma_min = getattr(state.model, "sigma_min")
    sigma_max = getattr(state.model, "sigma_max")
    params = state.params
    apply_fn = state.apply_fn

    metrics = []
    for batch in loader:
        rng, step_rng = jax.random.split(rng)
        latents = state.encode(batch)
        metrics.append(
            eval_step(
                params,
                apply_fn,
                latents,
                batch.node_mask,
                batch.pair_mask,
                step_rng,
                sigma_min,
                sigma_max,
                eval_fn,
            )
        )
    stacked = {k: jnp.stack([m[k] for m in metrics]) for k in metrics[0]}
    return {k: v.mean() for k, v in stacked.items()}


__all__ = ["eval_step", "evaluate"]
