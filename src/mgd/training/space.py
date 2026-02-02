"""Diffusion-space interfaces and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Protocol, Tuple

import jax.numpy as jnp

from mgd.dataset.utils import GraphBatch
from mgd.training.losses import categorical_ce_loss


class DiffusionSpace(Protocol):
    """Interface for mapping batches into diffusion space and scoring outputs."""

    def loss(self, outputs: Dict[str, object], batch: GraphBatch) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Compute loss/metrics from model outputs."""


@dataclass(frozen=True)
class CategoricalDiffusionSpace:
    """Categorical diffusion space using logits and CE losses."""

    loss_fn: Callable[..., Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]] = categorical_ce_loss
    node_class_weights: jnp.ndarray | None = None
    edge_exist_weights: jnp.ndarray | None = None
    edge_type_weights: jnp.ndarray | None = None
    label_smoothing: float | None = None

    def loss(self, outputs: Dict[str, object], batch: GraphBatch) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        return self.loss_fn(
            outputs["logits"],
            batch,
            batch.node_mask,
            batch.pair_mask,
            node_class_weights=self.node_class_weights,
            edge_exist_weights=self.edge_exist_weights,
            edge_type_weights=self.edge_type_weights,
            label_smoothing=self.label_smoothing,
        )


__all__ = [
    "DiffusionSpace",
    "CategoricalDiffusionSpace",
]
