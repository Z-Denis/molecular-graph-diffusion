"""Energy-guidance utilities for categorical-logit diffusion sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from mgd.dataset.chemistry import ChemistrySpec, DEFAULT_CHEMISTRY
from mgd.latent import GraphLatent, center_logits, symmetrize_edge


def _edge_probs(edge_logits: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.softmax(edge_logits, axis=-1)


def _atom_probs(atom_logits: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.softmax(atom_logits, axis=-1)


def _expected_bond_order_sum(
    edge_logits: jnp.ndarray,
    pair_mask: jnp.ndarray,
    bond_orders: jnp.ndarray,
) -> jnp.ndarray:
    p = _edge_probs(edge_logits)
    bo = jnp.asarray(bond_orders, dtype=edge_logits.dtype)
    if bo.shape[0] != p.shape[-1]:
        raise ValueError("bond_orders length must match edge logits channels.")
    expected = (p * bo).sum(axis=-1)
    expected = expected * pair_mask
    return expected.sum(axis=-1)


def _expected_degree(edge_logits: jnp.ndarray, pair_mask: jnp.ndarray) -> jnp.ndarray:
    p = _edge_probs(edge_logits)
    p_exist = 1.0 - p[..., 0]
    return (p_exist * pair_mask).sum(axis=-1)


def valence_over_penalty(
    logits: GraphLatent,
    node_mask: jnp.ndarray,
    pair_mask: jnp.ndarray,
    *,
    max_valence: jnp.ndarray = jnp.asarray(DEFAULT_CHEMISTRY.valence_table),
    bond_orders: jnp.ndarray = jnp.asarray(DEFAULT_CHEMISTRY.bond_orders),
) -> jnp.ndarray:
    v_hat = _expected_bond_order_sum(logits.edge, pair_mask, bond_orders)
    p_atom = _atom_probs(logits.node)
    vmax = (p_atom * jnp.asarray(max_valence, dtype=v_hat.dtype)).sum(axis=-1)
    over = jnp.maximum(v_hat - vmax, 0.0)
    mask = node_mask.astype(v_hat.dtype)
    return (jnp.square(over) * mask).sum() / jnp.maximum(mask.sum(), 1.0)


def degree_mse_penalty(
    logits: GraphLatent,
    node_mask: jnp.ndarray,
    pair_mask: jnp.ndarray,
    *,
    target_degree: jnp.ndarray,
) -> jnp.ndarray:
    deg_hat = _expected_degree(logits.edge, pair_mask)
    mask = node_mask.astype(deg_hat.dtype)
    return (jnp.square(deg_hat - target_degree) * mask).sum() / jnp.maximum(mask.sum(), 1.0)


@dataclass(frozen=True)
class LogitGuidanceConfig:
    spec: ChemistrySpec = DEFAULT_CHEMISTRY
    valence_weight: float = 0.0
    degree_weight: float = 0.0
    gauge_fix: bool = True
    bond_orders: jnp.ndarray | None = None
    max_valence: jnp.ndarray | None = None


def make_logit_guidance(
    config: LogitGuidanceConfig,
    *,
    weight_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[dict, jnp.ndarray, jnp.ndarray, jnp.ndarray], GraphLatent]:
    """Return a guidance function to apply to x_hat during sampling."""
    bond_orders = (
        jnp.asarray(config.spec.bond_orders)
        if config.bond_orders is None
        else jnp.asarray(config.bond_orders)
    )
    max_valence = (
        jnp.asarray(config.spec.valence_table)
        if config.max_valence is None
        else jnp.asarray(config.max_valence)
    )

    def energy(logits, node_mask, pair_mask):
        if config.gauge_fix:
            logits = GraphLatent(
                center_logits(logits.node, node_mask),
                center_logits(logits.edge, pair_mask),
            )
        total = jnp.array(0.0, dtype=logits.node.dtype)
        if config.valence_weight:
            total = total + config.valence_weight * valence_over_penalty(
                logits,
                node_mask,
                pair_mask,
                max_valence=max_valence,
                bond_orders=bond_orders,
            )
        if config.degree_weight:
            raise ValueError("degree_weight is not supported without explicit targets.")
        return total

    def guide(pred, node_mask, pair_mask, sigma):
        logits = pred["logits"]
        w = 1.0 if weight_fn is None else weight_fn(sigma)
        grad_fn = jax.grad(lambda x: energy(x, node_mask, pair_mask))
        g = grad_fn(logits)
        sigma2 = jnp.square(sigma)
        guided = logits - g * (sigma2 * w)
        guided = GraphLatent(guided.node, symmetrize_edge(guided.edge))
        return guided.masked(node_mask, pair_mask)

    return guide


__all__ = [
    "LogitGuidanceConfig",
    "make_logit_guidance",
    "valence_over_penalty",
    "degree_mse_penalty",
]
