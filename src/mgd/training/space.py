"""Diffusion-space interfaces and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Protocol, Tuple

import jax.numpy as jnp

from mgd.dataset.chemistry import ChemistrySpec, DEFAULT_CHEMISTRY
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
    spec: ChemistrySpec = DEFAULT_CHEMISTRY
    valence_weight: float = 0.0
    valence_beta_min: float = 1.0
    valence_beta_max: float = 1.0
    valence_beta_k: float = 3.0
    valence_sigma_data: float = 1.0
    valence_sigma_cutoff_mult: float = 10.0
    snr_reweight_ce: bool = False
    snr_sigma_data: float = 1.0

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
            valence_weight=self.valence_weight,
            max_valence=jnp.asarray(self.spec.valence_table),
            bond_orders=jnp.asarray(self.spec.bond_orders),
            sigma=outputs.get("sigma"),
            valence_beta_min=self.valence_beta_min,
            valence_beta_max=self.valence_beta_max,
            valence_beta_k=self.valence_beta_k,
            valence_sigma_data=self.valence_sigma_data,
            valence_sigma_cutoff_mult=self.valence_sigma_cutoff_mult,
            snr_reweight_ce=self.snr_reweight_ce,
            snr_sigma_data=self.snr_sigma_data,
        )


__all__ = [
    "DiffusionSpace",
    "CategoricalDiffusionSpace",
]
