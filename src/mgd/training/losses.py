"""Loss utilities for diffusion training."""

from __future__ import annotations

from typing import Dict, Tuple

import jax.numpy as jnp
import optax
import jax

from mgd.latent import GraphLatent


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
    if use_label_smoothing is not None:
        if not (0.0 <= use_label_smoothing < 1.0):
            raise ValueError("use_label_smoothing must be in [0, 1).")
        n_classes = logits.shape[-1]
        one_hot = jax.nn.one_hot(targets, n_classes)
        smooth = use_label_smoothing / n_classes
        target_probs = (1.0 - use_label_smoothing) * one_hot + smooth
        ce = optax.softmax_cross_entropy(logits, target_probs)
    else:
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    if class_weights is not None:
        weights = jnp.take(class_weights, targets)
        ce = weights * ce
    masked = ce * mask
    loss = masked.sum() / jnp.maximum(mask.sum(), 1.0)
    return loss, {"loss": loss}


__all__ = ["masked_mse", "masked_cosine_similarity", "masked_cross_entropy"]
