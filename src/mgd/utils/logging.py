"""Simple logging utility for training loops."""

from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp

from mgd.utils.checkpoints import save_checkpoint
from mgd.training.train_step import DiffusionTrainState


def _mean_metrics(history: List[Dict[str, jnp.ndarray]]) -> Dict[str, jnp.ndarray]:
    if not history:
        return {}
    stacked = {k: jnp.stack([m[k] for m in history]) for k in history[0]}
    return {k: v.mean() for k, v in stacked.items()}


class Logger:
    """Simple logger to accumulate metrics and optionally handle checkpoints."""

    def __init__(self, log_every: int = 0, ckpt_dir: str | None = None, ckpt_every: int | None = None):
        self.log_every = log_every
        self.ckpt_dir = ckpt_dir
        self.ckpt_every = ckpt_every
        self.data: List[Dict[str, jnp.ndarray]] = []

    def maybe_log(self, step: int, metrics_buffer: List[Dict[str, jnp.ndarray]]):
        if self.log_every and (step % self.log_every == 0) and metrics_buffer:
            self.data.append(_mean_metrics(metrics_buffer))
            return True
        return False

    def maybe_checkpoint(self, step: int, state: DiffusionTrainState):
        if self.ckpt_dir and self.ckpt_every and (step % self.ckpt_every == 0):
            save_checkpoint(self.ckpt_dir, state, step=step)


__all__ = ["Logger"]
