"""Evaluation utilities for diffusion models."""

from __future__ import annotations

from functools import partial
from typing import Iterable, Dict

import jax
import jax.numpy as jnp

from mgd.dataset.utils import GraphBatch
from mgd.training.losses import masked_mse


@partial(jax.jit, static_argnames="apply_fn")
def eval_step(
    params,
    apply_fn,
    batch: GraphBatch,
    rng: jax.Array,
    num_steps: int,
) -> Dict[str, jnp.ndarray]:
    """Single evaluation step with masked noise-prediction loss."""
    rng_t, rng_noise = jax.random.split(rng)
    t = jax.random.randint(rng_t, (batch.atom_type.shape[0],), 0, num_steps)
    outputs = apply_fn({"params": params}, batch, t, rngs={"noise": rng_noise})
    loss, parts = masked_mse(outputs["eps_pred"], outputs["noise"], batch.node_mask, batch.pair_mask)
    return {"loss": loss, **parts}


def evaluate(
    state,
    loader: Iterable[GraphBatch],
    rng: jax.Array,
) -> Dict[str, jnp.ndarray]:
    """Run evaluation over a loader and return mean metrics."""
    num_steps = state.model.schedule.betas.shape[0]
    params = state.params
    apply_fn = state.apply_fn

    metrics = []
    for batch in loader:
        rng, step_rng = jax.random.split(rng)
        metrics.append(eval_step(params, apply_fn, batch, step_rng, num_steps))
    stacked = {k: jnp.stack([m[k] for m in metrics]) for k in metrics[0]}
    return {k: v.mean() for k, v in stacked.items()}


__all__ = ["eval_step", "evaluate"]
