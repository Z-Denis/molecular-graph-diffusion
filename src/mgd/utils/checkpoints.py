"""Checkpoint save/restore helpers."""

from __future__ import annotations

import os
from typing import Any

from flax.training import checkpoints


def save_checkpoint(ckpt_dir: str, state: Any, step: int, keep: int = 3, prefix: str = "ckpt_") -> str:
    """Save a checkpoint for the given state."""
    ckpt_dir = os.path.abspath(os.path.expanduser(ckpt_dir))
    os.makedirs(ckpt_dir, exist_ok=True)
    return checkpoints.save_checkpoint(ckpt_dir, state, step, overwrite=True, keep=keep, prefix=prefix)


def restore_checkpoint(ckpt_dir: str, state: Any, prefix: str = "ckpt_") -> Any:
    """Restore the latest checkpoint if present, otherwise return the input state."""
    ckpt_dir = os.path.abspath(os.path.expanduser(ckpt_dir))
    return checkpoints.restore_checkpoint(ckpt_dir, target=state, prefix=prefix)


__all__ = ["save_checkpoint", "restore_checkpoint"]
