"""Evaluation utilities for diffusion models."""

from __future__ import annotations

from functools import partial
from typing import Callable, Dict, Iterable, Optional, Union

import jax
import jax.numpy as jnp

from mgd.dataset.utils import GraphBatch
from mgd.training.losses import masked_mse


@partial(jax.jit, static_argnames=["apply_fn", "eval_fn"])
def eval_step(
    params,
    apply_fn,
    latents,
    node_mask: jnp.ndarray,
    pair_mask: jnp.ndarray,
    rng: jax.Array,
    num_steps: int,
    eval_fn: Callable,
    time: Optional[Union[jax.Array, int]] = None,
) -> Dict[str, jnp.ndarray]:
    """Single evaluation step with masked noise-prediction loss."""
    rng_t, rng_noise = jax.random.split(rng)
    batch_size = latents.node.shape[0]
    if time is None:
        t = jax.random.randint(rng_t, (batch_size,), 0, num_steps)
    else:
        t = jnp.asarray(time)
        if t.ndim == 0:
            t = jnp.full((batch_size,), t, dtype=jnp.int32)
    outputs = apply_fn(
        {"params": params},
        latents,
        t,
        node_mask=node_mask,
        pair_mask=pair_mask,
        rngs={"noise": rng_noise},
    )
    loss, parts = eval_fn(outputs["eps_pred"], outputs["noise"], node_mask, pair_mask)
    return {"loss": loss, **parts}


def evaluate(
    state,
    loader: Iterable[GraphBatch],
    rng: jax.Array,
    eval_fn: Callable = masked_mse,
    time: Optional[Union[jax.Array, int]] = None,
) -> Dict[str, jnp.ndarray]:
    """Run evaluation over a loader and return mean metrics.

    ``time`` can be None (random per-example timesteps) or a scalar/array of timesteps.
    """
    num_steps = state.model.schedule.betas.shape[0]
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
                num_steps,
                eval_fn,
                time=time,
            )
        )
    stacked = {k: jnp.stack([m[k] for m in metrics]) for k in metrics[0]}
    return {k: v.mean() for k, v in stacked.items()}


__all__ = ["eval_step", "evaluate"]
