"""Loss utilities for diffusion training."""

from __future__ import annotations

from typing import Dict, Tuple

import jax.numpy as jnp

from mgd.model.utils import GraphLatent


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


__all__ = ["masked_mse"]
