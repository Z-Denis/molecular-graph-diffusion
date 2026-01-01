"""Minimal training loop orchestration for diffusion."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import jax
import jax.numpy as jnp

from mgd.dataset.utils import GraphBatch
from mgd.training.checkpoints import restore_checkpoint, save_checkpoint
from mgd.training.train_step import DiffusionTrainState, train_step
from mgd.utils.logging import Logger

from tqdm import tqdm


def _mean_metrics(history: List[Dict[str, jnp.ndarray]]) -> Dict[str, jnp.ndarray]:
    if not history:
        return {}
    stacked = {k: jnp.stack([m[k] for m in history]) for k in history[0]}
    return {k: v.mean() for k, v in stacked.items()}


def train_loop(
    state: DiffusionTrainState,
    loader: Iterable[GraphBatch],
    *,
    n_steps: int,
    rng: jax.Array,
    logger: Logger,
) -> Tuple[DiffusionTrainState, List[Dict[str, jnp.ndarray]]]:
    """Step-based training loop (streaming over the loader as needed).

    A ``Logger`` must be provided; its settings control logging and checkpoints.
    """
    step_fn = jax.jit(
        lambda st, b, r: train_step(
            st,
            b,
            r,
        )
    )
    history: List[Dict[str, jnp.ndarray]] = []

    metrics_buffer: List[Dict[str, jnp.ndarray]] = []
    loader_iter = iter(loader)

    with tqdm(total=n_steps) as pbar:
        for step in range(1, n_steps + 1):
            rng, step_rng = jax.random.split(rng)
            batch = next(loader_iter)
            state, metrics = step_fn(state, batch, step_rng)
            metrics_buffer.append(metrics)
            pbar.update(1)
            if logger.log_every and (step % logger.log_every == 0):
                # Show all recorded metrics in the progress bar
                postfix = {k: f"{float(v):.4f}" for k, v in metrics.items()}
                pbar.set_postfix(**postfix)
            if logger.maybe_log(step, metrics_buffer):
                history.append(logger.data[-1])
                metrics_buffer = []
            logger.maybe_checkpoint(step, state)

    if metrics_buffer:
        history.append(_mean_metrics(metrics_buffer))
        logger.data.append(history[-1])

    return state, history


__all__ = ["train_loop", "restore_checkpoint", "save_checkpoint", "Logger"]
