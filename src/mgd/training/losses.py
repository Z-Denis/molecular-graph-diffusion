"""Loss utilities for diffusion training."""

from __future__ import annotations

from typing import Dict, Tuple

import jax.numpy as jnp
import optax
import jax

from mgd.latent import GraphLatent


def _apply_weights(
    loss: jnp.ndarray, labels: jnp.ndarray, weights: jnp.ndarray | None
) -> jnp.ndarray:
    """Optionally reweight losses per class/label."""
    if weights is None:
        return loss
    w = jnp.take(weights, labels)
    return w * loss


def _apply_mask_mean(loss: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Apply mask and normalize by the number of valid entries."""
    masked = loss * mask
    return masked.sum() / jnp.maximum(mask.sum(), 1.0)


def _softmax_ce(
    logits: jnp.ndarray, targets: jnp.ndarray, label_smoothing: float | None
) -> jnp.ndarray:
    """Cross-entropy with optional label smoothing."""
    if label_smoothing is None:
        return optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    if not (0.0 <= label_smoothing < 1.0):
        raise ValueError("label_smoothing must be in [0, 1).")
    n_classes = logits.shape[-1]
    one_hot = jax.nn.one_hot(targets, n_classes)
    smooth = label_smoothing / n_classes
    target_probs = (1.0 - label_smoothing) * one_hot + smooth
    return optax.softmax_cross_entropy(logits, target_probs)


def masked_mse(
    pred: GraphLatent,
    target: GraphLatent,
    node_mask: jnp.ndarray,
    pair_mask: jnp.ndarray,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Mean squared error over valid nodes and edges.

    Mathematically (eps_pred vs eps_target):
        L_node = sum_{b,i} m_i ||eps_pred_{b,i} - eps_{b,i}||^2 / sum_{b,i} m_i
        L_edge = sum_{b,i,j} M_{ij} ||eps_pred_{b,ij} - eps_{b,ij}||^2 / sum_{b,i,j} M_{ij}
        L = L_node + L_edge
    where m_i = node_mask, M_{ij} = pair_mask.
    """
    node_w = node_mask[..., None]
    edge_w = pair_mask[..., None]

    node_err = jnp.square(pred.node - target.node) * node_w
    edge_err = jnp.square(pred.edge - target.edge) * edge_w

    node_norm = jnp.maximum(node_w.sum(), 1.0)
    edge_norm = jnp.maximum(edge_w.sum(), 1.0)

    node_loss = node_err.sum() / node_norm
    edge_loss = edge_err.sum() / edge_norm
    total = node_loss + edge_loss
    return total, {"node_loss": node_loss, "edge_loss": edge_loss}


def masked_cosine_similarity(
    pred: GraphLatent,
    target: GraphLatent,
    node_mask: jnp.ndarray,
    pair_mask: jnp.ndarray,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Mean cosine similarity over valid nodes and edges."""

    def mask_flat_norm(x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        masked = x * mask[..., None]
        flat = masked.reshape(masked.shape[0], -1)
        norm = jnp.linalg.norm(flat, ord=2, axis=-1, keepdims=True)
        norm = jnp.maximum(norm, 1e-8)
        return flat / norm

    node_pred = mask_flat_norm(pred.node, node_mask)
    node_targ = mask_flat_norm(target.node, node_mask)
    edge_pred = mask_flat_norm(pred.edge, pair_mask)
    edge_targ = mask_flat_norm(target.edge, pair_mask)

    node_sim = jnp.sum(node_pred * node_targ, axis=-1).mean()
    edge_sim = jnp.sum(edge_pred * edge_targ, axis=-1).mean()
    total = 0.5 * (node_sim + edge_sim)

    return total, {"node_sim": node_sim, "edge_sim": edge_sim}


def masked_cross_entropy(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    mask: jnp.ndarray,
    class_weights: jnp.ndarray | None = None,
    use_label_smoothing: float | None = None,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Masked categorical cross-entropy for arbitrary graph logits.

    Works for node logits (e.g., ``(B, N, C)``) or edge logits
    (e.g., ``(B, N, N, C)``) so long as ``mask`` matches ``targets``.
    Loss is averaged over unmasked entries:
        L = sum m * CE(logits, y) / sum m
    where CE is softmax cross-entropy and m is the provided mask.
    """
    ce = _softmax_ce(logits, targets, use_label_smoothing)
    ce = _apply_weights(ce, targets, class_weights)
    loss = _apply_mask_mean(ce, mask)
    return loss, {"loss": loss}


def _bond_reconstruction_loss(
    exist_logits: jnp.ndarray,
    type_logits: jnp.ndarray,
    exists_label: jnp.ndarray,
    type_label: jnp.ndarray,
    mask: jnp.ndarray,
    existence_weights: jnp.ndarray | None = None,
    bond_type_weights: jnp.ndarray | None = None,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Two-head bond loss: existence (binary) + type (categorical).

    Existence loss is applied to all masked edges. Type loss is only applied
    where a bond exists (``exists_label==1``) and is also masked.
    Both terms are normalized per-molecule by the number of valid edges and
    averaged over the batch.
    """
    mask = mask.astype(jnp.float32)
    valid_edges = jnp.maximum(mask.sum(axis=(1, 2)), 1.0)

    # Existence: sigmoid BCE on a single logit per edge.
    exist_loss = optax.sigmoid_binary_cross_entropy(exist_logits, exists_label)
    exist_loss = _apply_weights(exist_loss, exists_label.astype(jnp.int32), existence_weights)
    exist_loss = (exist_loss * mask).sum(axis=(1, 2)) / valid_edges

    # Type: categorical CE on edges that exist.
    type_loss = optax.softmax_cross_entropy_with_integer_labels(type_logits, type_label)
    type_loss = _apply_weights(type_loss, type_label, bond_type_weights)
    type_loss = (type_loss * exists_label * mask).sum(axis=(1, 2)) / valid_edges

    total = exist_loss + type_loss
    loss = total.mean()

    metrics = {
        "loss": loss,
        "loss_exist": exist_loss.mean(),
        "loss_type": type_loss.mean(),
    }
    return loss, metrics


def bond_reconstruction_loss(
    logits: jnp.ndarray,
    type_label: jnp.ndarray,
    mask: jnp.ndarray,
    existence_weights: jnp.ndarray | None = None,
    bond_type_weights: jnp.ndarray | None = None,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Bond loss where logits pack existence (channel 0) and type (rest).

    Parameters follow the decoder output:
        logits: [..., 1 + n_types] where channel 0 is the existence logit.
        type_label: integer bond-type labels (0 means no bond).
        mask: pair mask for valid edges.
    Existence labels are inferred as ``type_label != 0``.
    """
    if logits.shape[-1] < 2:
        raise ValueError("Concatenated logits must have at least 2 channels.")
    exist_logits = logits[..., 0]
    type_logits = logits[..., 1:]
    exists_label = (type_label != 0).astype(jnp.float32)
    return _bond_reconstruction_loss(
        exist_logits,
        type_logits,
        exists_label,
        type_label,
        mask,
        existence_weights=existence_weights,
        bond_type_weights=bond_type_weights,
    )

__all__ = [
    "masked_mse",
    "masked_cosine_similarity",
    "masked_cross_entropy",
    "bond_reconstruction_loss",
]
