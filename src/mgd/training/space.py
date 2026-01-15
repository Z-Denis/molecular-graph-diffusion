"""Diffusion-space interfaces and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Protocol, Tuple

import jax
import jax.numpy as jnp

from mgd.dataset.utils import GraphBatch
from mgd.latent import GraphLatent, GraphLatentSpace, center_logits
from mgd.training.losses import categorical_ce_loss, edm_masked_mse


class DiffusionSpace(Protocol):
    """Interface for mapping batches into diffusion space and scoring outputs."""

    def loss(self, outputs: Dict[str, object], batch: GraphBatch) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Compute loss/metrics from model outputs."""


def _identity(latent: GraphLatent) -> GraphLatent:
    return latent


@dataclass(frozen=True)
class LatentDiffusionSpace:
    """EDM latent diffusion space with configurable encoder and normalization."""

    encoder: Callable[[GraphBatch], GraphLatent]
    normalizer: Callable[[GraphLatent], GraphLatent] = _identity
    loss_fn: Callable[..., Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]] = edm_masked_mse

    def encode(self, batch: GraphBatch) -> GraphLatent:
        return self.normalizer(self.encoder(batch))

    def loss(self, outputs: Dict[str, object], batch: GraphBatch) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        return self.loss_fn(
            outputs["x_hat"],
            outputs["clean"],
            batch.node_mask,
            batch.pair_mask,
            sigma=outputs["sigma"],
        )


@dataclass(frozen=True)
class OneHotLogitDiffusionSpace:
    """Logit diffusion space using one-hot graph targets."""

    space: GraphLatentSpace
    loss_fn: Callable[..., Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]] = edm_masked_mse
    gauge_fix: bool = True
    alpha: float = 2.0

    def encode(self, batch: GraphBatch) -> GraphLatent:
        node_oh = jax.nn.one_hot(batch.atom_type, self.space.node_dim, dtype=self.space.dtype)
        edge_oh = jax.nn.one_hot(batch.bond_type, self.space.edge_dim, dtype=self.space.dtype)
        node = (2.0 * node_oh - 1.0) * self.alpha
        edge = (2.0 * edge_oh - 1.0) * self.alpha
        return GraphLatent(
            node=node * batch.node_mask[..., None],
            edge=edge * batch.pair_mask[..., None],
        )

    def loss(self, outputs: Dict[str, object], batch: GraphBatch) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        x_hat = outputs["x_hat"]
        clean = outputs["clean"]
        if self.gauge_fix:
            x_hat = GraphLatent(
                center_logits(x_hat.node, batch.node_mask),
                center_logits(x_hat.edge, batch.pair_mask),
            )
            clean = GraphLatent(
                center_logits(clean.node, batch.node_mask),
                center_logits(clean.edge, batch.pair_mask),
            )
        return self.loss_fn(x_hat, clean, batch.node_mask, batch.pair_mask, sigma=outputs["sigma"])


@dataclass(frozen=True)
class CategoricalDiffusionSpace:
    """Categorical diffusion space using logits and CE losses."""

    loss_fn: Callable[..., Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]] = categorical_ce_loss
    node_class_weights: jnp.ndarray | None = None
    edge_class_weights: jnp.ndarray | None = None
    label_smoothing: float | None = None

    def loss(self, outputs: Dict[str, object], batch: GraphBatch) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        return self.loss_fn(
            outputs["logits"],
            batch,
            batch.node_mask,
            batch.pair_mask,
            node_class_weights=self.node_class_weights,
            edge_class_weights=self.edge_class_weights,
            label_smoothing=self.label_smoothing,
        )


__all__ = [
    "DiffusionSpace",
    "LatentDiffusionSpace",
    "OneHotLogitDiffusionSpace",
    "CategoricalDiffusionSpace",
]
