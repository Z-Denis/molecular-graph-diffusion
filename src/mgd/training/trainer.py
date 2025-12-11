"""Minimal training loop orchestration for diffusion."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import jax
import jax.numpy as jnp

from mgd.dataset.utils import GraphBatch
from mgd.training.checkpoints import restore_checkpoint, save_checkpoint
from mgd.training.train_step import DiffusionTrainState, train_step

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
    log_every: int = 0,
    ckpt_dir: str | None = None,
    ckpt_every: int | None = None,
) -> Tuple[DiffusionTrainState, List[Dict[str, jnp.ndarray]]]:
    """Step-based training loop (streaming over the loader as needed)."""
    step_fn = jax.jit(train_step)
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
            if log_every and (step % log_every == 0):
                loss_val = float(metrics["loss"])
                pbar.set_postfix(loss=f"{loss_val:.4f}")
                history.append(_mean_metrics(metrics_buffer))
                metrics_buffer = []

            if ckpt_dir and ckpt_every and (step % ckpt_every == 0):
                save_checkpoint(ckpt_dir, state, step=step)

    if metrics_buffer:
        history.append(_mean_metrics(metrics_buffer))

    return state, history


__all__ = ["train_loop", "restore_checkpoint", "save_checkpoint"]
