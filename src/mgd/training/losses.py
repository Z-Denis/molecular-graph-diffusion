"""Loss utilities for categorical diffusion training."""

from __future__ import annotations

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import optax

from mgd.latent import GraphLatent


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
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    ce = _softmax_ce(logits, targets, use_label_smoothing)
    ce = _apply_weights(ce, targets, class_weights)
    loss = _apply_mask_mean_per_graph(ce, mask)
    return loss, {"loss": loss}


def masked_binary_ce_per_graph(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    mask: jnp.ndarray,
    weights: jnp.ndarray | None = None,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    bce = optax.sigmoid_binary_cross_entropy(logits, targets)
    bce = _apply_binary_weights(bce, targets, weights)
    loss = _apply_mask_mean_per_graph(bce, mask)
    return loss, {"loss": loss}


def categorical_ce_loss(
    logits: GraphLatent,
    batch,
    node_mask: jnp.ndarray,
    pair_mask: jnp.ndarray,
    node_class_weights: jnp.ndarray | None = None,
    edge_exist_weights: jnp.ndarray | None = None,
    edge_type_weights: jnp.ndarray | None = None,
    label_smoothing: float | None = None,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Cross-entropy on categorical node logits and factorized edge logits."""
    node_loss, _ = masked_cross_entropy_per_graph(
        logits.node,
        batch.atom_type,
        node_mask,
        class_weights=node_class_weights,
        use_label_smoothing=label_smoothing,
    )
    edge_exist = (batch.bond_type > 0).astype(logits.edge.dtype)
    exist_loss, _ = masked_binary_ce_per_graph(
        logits.edge[..., 0],
        edge_exist,
        pair_mask,
        weights=edge_exist_weights,
    )
    type_targets = jnp.maximum(batch.bond_type - 1, 0)
    type_mask = pair_mask * (batch.bond_type > 0)
    type_loss, _ = masked_cross_entropy_per_graph(
        logits.edge[..., 1:],
        type_targets,
        type_mask,
        class_weights=edge_type_weights,
        use_label_smoothing=label_smoothing,
    )
    edge_loss = exist_loss + type_loss
    total = node_loss + edge_loss
    return total, {
        "node_loss": node_loss,
        "edge_loss": edge_loss,
        "edge_exist_loss": exist_loss,
        "edge_type_loss": type_loss,
    }


__all__ = [
    "masked_cross_entropy",
    "masked_cross_entropy_per_graph",
    "masked_binary_ce_per_graph",
    "categorical_ce_loss",
]
