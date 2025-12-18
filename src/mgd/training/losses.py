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
    node_weight: jnp.ndarray | None = None,
    edge_weight: jnp.ndarray | None = None,
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
    if node_weight is not None:
        node_w = node_w * node_weight[..., None, None]  # (B, N, 1)
    if edge_weight is not None:
        edge_w = edge_w * edge_weight[..., None, None, None]  # (B, N, N, 1)

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
    pos_weight: float | None = None,
    type_loss_scale: float = 1.0,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Two-head bond loss: existence (binary) + type (categorical).

    Existence loss is applied to all masked edges. Type loss is only applied
    where a bond exists (``exists_label==1``) and is also masked.
    Both terms are normalized per-molecule by the number of valid edges and
    averaged over the batch.
    """
    mask = mask.astype(jnp.float32)
    exists_label = exists_label.astype(jnp.float32)

    valid_edges = jnp.maximum(mask.sum(axis=(1, 2)), 1.0)
    
    # sigmoid BCE on a single logit per edge.
    exist_loss = optax.sigmoid_binary_cross_entropy(
        exist_logits, 
        exists_label,
        )
    if pos_weight is not None:
        pos_weight = jnp.asarray(pos_weight, dtype=exist_loss.dtype)
        exist_loss = jnp.where(
            exists_label == 1.0,
            pos_weight * exist_loss,
            exist_loss,
        )
    exist_loss = _apply_weights(exist_loss, exists_label.astype(jnp.int32), existence_weights)
    exist_loss = (exist_loss * mask).sum(axis=(1, 2)) / valid_edges

    # categorical CE on edges that exist.
    type_loss = optax.softmax_cross_entropy_with_integer_labels(
        type_logits, 
        type_label,
        )
    type_loss = _apply_weights(type_loss, type_label, bond_type_weights)
    type_loss = (type_loss * exists_label * mask).sum(axis=(1, 2)) / valid_edges

    total = exist_loss + type_loss_scale * type_loss
    loss = total.mean()

    metrics = {
        "loss": loss,
        "loss_exist": exist_loss.mean(),
        "loss_type": type_loss.mean(),
        "existence_rate": (mask * exists_label).sum() / valid_edges.sum()
    }
    return loss, metrics


def bond_reconstruction_loss(
    logits: jnp.ndarray,
    type_label: jnp.ndarray,
    mask: jnp.ndarray,
    existence_weights: jnp.ndarray | None = None,
    bond_type_weights: jnp.ndarray | None = None,
    pos_weight: float | None = None,
    type_loss_scale: float = 1.0,
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
    loss, metrics = _bond_reconstruction_loss(
        exist_logits,
        type_logits,
        exists_label,
        type_label,
        mask,
        existence_weights=existence_weights,
        bond_type_weights=bond_type_weights,
        pos_weight=pos_weight,
        type_loss_scale=type_loss_scale,
    )

    # # Precision/recall on existence (using sigmoid > 0.5)
    # exist_prob = jax.nn.sigmoid(exist_logits)
    # pred_exist = (exist_prob > 0.2).astype(jnp.float32)

    # mask_f = mask.astype(jnp.float32)
    # frac_pred_pos = (pred_exist * mask_f).sum()
    # tp = (pred_exist * exists_label * mask_f).sum()
    # fp = (pred_exist * (1.0 - exists_label) * mask_f).sum()
    # fn = ((1.0 - pred_exist) * exists_label * mask_f).sum()
    # precision = tp / jnp.maximum(tp + fp, 1e-8)
    # recall = tp / jnp.maximum(tp + fn, 1e-8)
    # prob_mean = (exist_prob * mask_f).sum() / jnp.maximum(mask_f.sum(), 1.0)
    # metrics.update(
    #     {
    #         "precision": precision,
    #         "recall": recall,
    #         "frac_pred_pos": frac_pred_pos,
    #         "exist_prob_mean": prob_mean,
    #     }
    # )
    pred_exists = (jax.nn.sigmoid(exist_logits) > 0.5)
    pred_rate = (mask * pred_exists).sum() / mask.sum()
    mean_prob = (mask * jax.nn.sigmoid(exist_logits)).sum() / mask.sum()
    metrics.update(
        {
            "pred_rate": pred_rate,
            "mean_prob": mean_prob,
        }
    )
    return loss, metrics


def bond_reconstruction_focal_loss(
    logits: jnp.ndarray,
    type_label: jnp.ndarray,
    mask: jnp.ndarray,
    existence_weights: jnp.ndarray | None = None,
    bond_type_weights: jnp.ndarray | None = None,
    alpha: float = 0.25,
    gamma: float = 2.0,
    eps: float = 1e-8,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Bond loss with focal loss on existence and CE on types."""
    if logits.shape[-1] < 2:
        raise ValueError("Concatenated logits must have at least 2 channels.")
    exist_logits = logits[..., 0]
    type_logits = logits[..., 1:]
    exists_label = (type_label != 0).astype(jnp.float32)

    p = jax.nn.sigmoid(exist_logits)
    p = jnp.clip(p, eps, 1.0 - eps)
    pt = jnp.where(exists_label == 1.0, p, 1.0 - p)
    alpha_t = jnp.where(exists_label == 1.0, alpha, 1.0 - alpha)
    focal = -alpha_t * ((1.0 - pt) ** gamma) * jnp.log(pt)
    if existence_weights is not None:
        focal = _apply_weights(focal, exists_label.astype(jnp.int32), existence_weights)
    mask_f = mask.astype(focal.dtype)
    valid_edges = jnp.maximum(mask_f.sum(axis=(1, 2)), 1.0)
    loss_exist = (focal * mask_f).sum(axis=(1, 2)) / valid_edges

    type_loss = optax.softmax_cross_entropy_with_integer_labels(type_logits, type_label)
    type_loss = _apply_weights(type_loss, type_label, bond_type_weights)
    type_loss = (type_loss * exists_label * mask_f).sum(axis=(1, 2)) / valid_edges

    total = loss_exist + type_loss
    loss = total.mean()
    metrics = {
        "loss": loss,
        "loss_exist": loss_exist.mean(),
        "loss_type": type_loss.mean(),
    }
    return loss, metrics


def graph_reconstruction_loss(
    recon: Dict[str, jnp.ndarray],
    batch,
    latents=None,
    *,
    atom_class_weights: jnp.ndarray | None = None,
    node_loss_scale: float = 1.0,
    bond_loss_scale: float = 1.0,
    bond_loss_fn=bond_reconstruction_loss,
    bond_loss_kwargs: Dict | None = None,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Combine node cross-entropy and bond reconstruction losses.

    Args:
        recon: dict with ``"node"`` logits (B, N, n_atoms) and ``"edge"`` logits (B, N, N, 1+n_bond_types).
        batch: GraphBatch containing ``atom_type``, ``bond_type``, ``node_mask``, ``pair_mask``.
        atom_class_weights: optional class weights for atom CE.
        node_loss_scale: scalar multiplier for the node CE term.
        bond_loss_scale: scalar multiplier for the bond loss term.
        bond_loss_fn: bond loss function (defaults to ``bond_reconstruction_loss``).
        bond_loss_kwargs: extra kwargs forwarded to the bond loss.
    """
    bond_loss_kwargs = bond_loss_kwargs or {}

    node_loss, node_metrics = masked_cross_entropy(
        recon["node"],
        batch.atom_type,
        batch.node_mask,
        class_weights=atom_class_weights,
    )
    bond_loss_val, bond_metrics = bond_loss_fn(
        recon["edge"],
        batch.bond_type,
        batch.pair_mask,
        **bond_loss_kwargs,
    )

    total = node_loss_scale * node_loss + bond_loss_scale * bond_loss_val
    metrics = {
        "loss": total,
        "loss_node": node_loss,
        "loss_bond": bond_loss_val,
    }
    metrics.update({f"node_{k}": v for k, v in node_metrics.items() if k != "loss"})
    metrics.update({f"bond_{k}": v for k, v in bond_metrics.items() if k != "loss"})
    return total, metrics

__all__ = [
    "masked_mse",
    "masked_cosine_similarity",
    "masked_cross_entropy",
    "bond_reconstruction_loss",
    "bond_reconstruction_focal_loss",
    "graph_reconstruction_loss",
]
