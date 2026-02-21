"""Loss utilities for categorical diffusion training."""

from __future__ import annotations

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import optax

from mgd.dataset.chemistry import DEFAULT_CHEMISTRY
from mgd.latent import GraphLatent, edge_probs_from_logits


def _apply_weights(
    loss: jnp.ndarray, labels: jnp.ndarray, weights: jnp.ndarray | None
) -> jnp.ndarray:
    if weights is None:
        return loss
    w = jnp.take(weights, labels)
    return w * loss


def _apply_binary_weights(
    loss: jnp.ndarray, labels: jnp.ndarray, weights: jnp.ndarray | None
) -> jnp.ndarray:
    if weights is None:
        return loss
    if weights.ndim == 0:
        return weights * loss
    w = jnp.take(weights, labels.astype(jnp.int32))
    return w * loss


def _apply_mask_mean(loss: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    masked = loss * mask
    return masked.sum() / jnp.maximum(mask.sum(), 1.0)


def _apply_mask_mean_per_graph(loss: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    masked = loss * mask
    denom = jnp.maximum(mask.sum(axis=tuple(range(1, mask.ndim))), 1.0)
    numer = masked.sum(axis=tuple(range(1, masked.ndim)))
    return (numer / denom).mean()


def _softmax_ce(
    logits: jnp.ndarray, targets: jnp.ndarray, label_smoothing: float | None
) -> jnp.ndarray:
    if label_smoothing is None:
        return optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    if not (0.0 <= label_smoothing < 1.0):
        raise ValueError("label_smoothing must be in [0, 1).")
    n_classes = logits.shape[-1]
    one_hot = jax.nn.one_hot(targets, n_classes)
    smooth = label_smoothing / n_classes
    target_probs = (1.0 - label_smoothing) * one_hot + smooth
    return optax.softmax_cross_entropy(logits, target_probs)


def masked_cross_entropy(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    mask: jnp.ndarray,
    class_weights: jnp.ndarray | None = None,
    use_label_smoothing: float | None = None,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    ce = _softmax_ce(logits, targets, use_label_smoothing)
    ce = _apply_weights(ce, targets, class_weights)
    loss = _apply_mask_mean(ce, mask)
    return loss, {"loss": loss}


def masked_cross_entropy_per_graph(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    mask: jnp.ndarray,
    class_weights: jnp.ndarray | None = None,
    use_label_smoothing: float | None = None,
    sample_weights: jnp.ndarray | None = None,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    ce = _softmax_ce(logits, targets, use_label_smoothing)
    ce = _apply_weights(ce, targets, class_weights)
    masked = ce * mask
    denom = jnp.maximum(mask.sum(axis=tuple(range(1, mask.ndim))), 1.0)
    numer = masked.sum(axis=tuple(range(1, masked.ndim)))
    per_graph = numer / denom
    if sample_weights is None:
        loss = per_graph.mean()
    else:
        w = jnp.asarray(sample_weights, dtype=per_graph.dtype)
        loss = (per_graph * w).sum() / jnp.maximum(w.sum(), 1e-8)
    return loss, {"loss": loss}


def masked_binary_ce_per_graph(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    mask: jnp.ndarray,
    weights: jnp.ndarray | None = None,
    sample_weights: jnp.ndarray | None = None,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    bce = optax.sigmoid_binary_cross_entropy(logits, targets)
    bce = _apply_binary_weights(bce, targets, weights)
    masked = bce * mask
    denom = jnp.maximum(mask.sum(axis=tuple(range(1, mask.ndim))), 1.0)
    numer = masked.sum(axis=tuple(range(1, masked.ndim)))
    per_graph = numer / denom
    if sample_weights is None:
        loss = per_graph.mean()
    else:
        w = jnp.asarray(sample_weights, dtype=per_graph.dtype)
        loss = (per_graph * w).sum() / jnp.maximum(w.sum(), 1e-8)
    return loss, {"loss": loss}


def _valence_over_penalty_per_graph(
    probs: GraphLatent,
    node_mask: jnp.ndarray,
    pair_mask: jnp.ndarray,
    *,
    max_valence: jnp.ndarray,
    bond_orders: jnp.ndarray,
) -> jnp.ndarray:
    bo = jnp.asarray(bond_orders, dtype=probs.edge.dtype)
    if bo.shape[0] == probs.edge.shape[-1] - 1:
        bo = jnp.concatenate([jnp.zeros((1,), dtype=bo.dtype), bo], axis=0)
    if bo.shape[0] != probs.edge.shape[-1]:
        raise ValueError(
            f"bond_orders length ({bo.shape[0]}) must match edge probs channels "
            f"({probs.edge.shape[-1]}) or be type-only ({probs.edge.shape[-1] - 1})."
        )
    vmax_table = jnp.asarray(max_valence, dtype=probs.node.dtype)
    if vmax_table.shape[0] == probs.node.shape[-1] - 1:
        vmax_table = jnp.concatenate(
            [jnp.zeros((1,), dtype=vmax_table.dtype), vmax_table],
            axis=0,
        )
    if vmax_table.shape[0] != probs.node.shape[-1]:
        raise ValueError(
            f"max_valence length ({vmax_table.shape[0]}) must match node probs channels "
            f"({probs.node.shape[-1]}) or be type-only ({probs.node.shape[-1] - 1})."
        )

    v_hat = (probs.edge * bo).sum(axis=-1) * pair_mask
    v_hat = v_hat.sum(axis=-1)
    vmax = (probs.node * vmax_table).sum(axis=-1)
    over = jnp.maximum(v_hat - vmax, 0.0)

    node_mask = node_mask.astype(over.dtype)
    numer = (jnp.square(over) * node_mask).sum(axis=-1)
    denom = jnp.maximum(node_mask.sum(axis=-1), 1.0)
    return numer / denom


def categorical_ce_loss(
    logits: GraphLatent,
    batch,
    node_mask: jnp.ndarray,
    pair_mask: jnp.ndarray,
    node_class_weights: jnp.ndarray | None = None,
    edge_exist_weights: jnp.ndarray | None = None,
    edge_type_weights: jnp.ndarray | None = None,
    label_smoothing: float | None = None,
    valence_weight: float = 0.0,
    max_valence: jnp.ndarray | None = None,
    bond_orders: jnp.ndarray | None = None,
    sigma: jnp.ndarray | None = None,
    valence_beta_min: float = 1.0,
    valence_beta_max: float = 1.0,
    valence_beta_k: float = 3.0,
    valence_sigma_data: float = 1.0,
    valence_sigma_cutoff_mult: float = 10.0,
    snr_reweight_ce: bool = False,
    snr_sigma_data: float = 1.0,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Cross-entropy on categorical node logits and factorized edge logits."""
    sample_weights = None
    if snr_reweight_ce:
        if sigma is None:
            raise ValueError("sigma must be provided when snr_reweight_ce=True.")
        sd2 = jnp.square(jnp.asarray(snr_sigma_data, dtype=logits.node.dtype))
        snr = sd2 / jnp.maximum(jnp.square(sigma), 1e-12)
        sample_weights = snr / (1.0 + snr)

    node_loss, _ = masked_cross_entropy_per_graph(
        logits.node,
        batch.atom_type,
        node_mask,
        class_weights=node_class_weights,
        use_label_smoothing=label_smoothing,
        sample_weights=sample_weights,
    )
    edge_exist = (batch.bond_type > 0).astype(logits.edge.dtype)
    exist_loss, _ = masked_binary_ce_per_graph(
        logits.edge[..., 0],
        edge_exist,
        pair_mask,
        weights=edge_exist_weights,
        sample_weights=sample_weights,
    )
    type_targets = jnp.maximum(batch.bond_type - 1, 0)
    type_mask = pair_mask * (batch.bond_type > 0)
    type_loss, _ = masked_cross_entropy_per_graph(
        logits.edge[..., 1:],
        type_targets,
        type_mask,
        class_weights=edge_type_weights,
        use_label_smoothing=label_smoothing,
        sample_weights=sample_weights,
    )
    edge_loss = exist_loss + type_loss
    total = node_loss + edge_loss

    valence_loss = jnp.asarray(0.0, dtype=total.dtype)
    valence_loss_raw = jnp.asarray(0.0, dtype=total.dtype)
    if valence_weight:
        probs = GraphLatent(
            node=jax.nn.softmax(logits.node, axis=-1),
            edge=edge_probs_from_logits(logits.edge, pair_mask=pair_mask),
        )
        bo = (
            jnp.asarray(DEFAULT_CHEMISTRY.bond_orders)
            if bond_orders is None
            else jnp.asarray(bond_orders)
        )
        vmax = (
            jnp.asarray(DEFAULT_CHEMISTRY.valence_table)
            if max_valence is None
            else jnp.asarray(max_valence)
        )
        per_graph = _valence_over_penalty_per_graph(
            probs,
            node_mask,
            pair_mask,
            max_valence=vmax,
            bond_orders=bo,
        )
        valence_loss_raw = per_graph.mean()
        if sigma is not None:
            log_sigma_data = jnp.log(jnp.asarray(valence_sigma_data, dtype=per_graph.dtype))
            beta = jax.nn.sigmoid(
                valence_beta_k * (log_sigma_data - jnp.log(jnp.maximum(sigma, 1e-12)))
            )
            beta = valence_beta_min + (valence_beta_max - valence_beta_min) * beta
            sigma_cutoff = jnp.asarray(valence_sigma_cutoff_mult * valence_sigma_data, dtype=per_graph.dtype)
            beta = beta * (sigma <= sigma_cutoff).astype(beta.dtype)
            per_graph = per_graph * beta
        valence_loss = per_graph.mean()
        total = total + valence_weight * valence_loss

    return total, {
        "node_loss": node_loss,
        "edge_loss": edge_loss,
        "edge_exist_loss": exist_loss,
        "edge_type_loss": type_loss,
        "valence_over_loss": valence_loss,
        "valence_over_loss_raw": valence_loss_raw,
    }


__all__ = [
    "masked_cross_entropy",
    "masked_cross_entropy_per_graph",
    "masked_binary_ce_per_graph",
    "categorical_ce_loss",
]
