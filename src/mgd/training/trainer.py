"""Minimal training loop orchestration for diffusion."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import jax
import jax.numpy as jnp

from mgd.dataset.utils import GraphBatch
from mgd.training.checkpoints import restore_checkpoint, save_checkpoint
from mgd.training.train_step import DiffusionTrainState, train_step


def _mean_metrics(history: List[Dict[str, jnp.ndarray]]) -> Dict[str, jnp.ndarray]:
    if not history:
        return {}
    stacked = {k: jnp.stack([m[k] for m in history]) for k in history[0]}
    return {k: v.mean() for k, v in stacked.items()}


def train_loop(
    state: DiffusionTrainState,
    loader: Iterable[GraphBatch],
    num_epochs: int,
    rng: jax.Array,
    log_every: int = 0,
    ckpt_dir: str | None = None,
    ckpt_every: int | None = None,
) -> Tuple[DiffusionTrainState, List[Dict[str, jnp.ndarray]]]:
    """Simple epoch-based training loop."""
    step_fn = jax.jit(train_step)
    history: List[Dict[str, jnp.ndarray]] = []

    for epoch in range(num_epochs):
        rng, epoch_rng = jax.random.split(rng)
        epoch_metrics: List[Dict[str, jnp.ndarray]] = []
        for step, batch in enumerate(loader):
            epoch_rng, step_rng = jax.random.split(epoch_rng)
            state, metrics = step_fn(state, batch, step_rng)
            epoch_metrics.append(metrics)
            if log_every and (step + 1) % log_every == 0:
                loss_val = float(metrics["loss"])
                print(f"epoch {epoch+1} step {step+1}: loss={loss_val:.4f}")

        mean_metrics = _mean_metrics(epoch_metrics)
        history.append(mean_metrics)

        if ckpt_dir and ckpt_every and (epoch + 1) % ckpt_every == 0:
            save_checkpoint(ckpt_dir, state, step=epoch + 1)

    return state, history


__all__ = ["train_loop", "restore_checkpoint", "save_checkpoint"]
