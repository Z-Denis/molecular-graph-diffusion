"""Energy-guidance utilities for categorical-logit diffusion sampling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.numpy as jnp

from mgd.latent import (
    GraphLatent,
    center_edge_type_logits,
    center_logits,
    edge_probs_from_logits,
    symmetrize_edge_probs,
)
from mgd.dataset.qm9 import BOND_ORDERS, VALENCE_TABLE


def _expected_bond_order_sum(
    edge_probs: jnp.ndarray,
    pair_mask: jnp.ndarray,
    bond_orders: jnp.ndarray,
) -> jnp.ndarray:
    p = edge_probs
    bo = jnp.asarray(bond_orders, dtype=edge_probs.dtype)
    if bo.shape[0] != p.shape[-1]:
        raise ValueError("bond_orders length must match edge logits channels.")
    expected = (p * bo).sum(axis=-1)
    expected = expected * pair_mask
    return expected.sum(axis=-1)


def _expected_degree(edge_probs: jnp.ndarray, pair_mask: jnp.ndarray) -> jnp.ndarray:
    p_exist = 1.0 - edge_probs[..., 0]
    return (p_exist * pair_mask).sum(axis=-1)


def valence_over_penalty(
    probs: GraphLatent,
    node_mask: jnp.ndarray,
    pair_mask: jnp.ndarray,
    *,
    max_valence: jnp.ndarray = VALENCE_TABLE,
    bond_orders: jnp.ndarray = BOND_ORDERS,
) -> jnp.ndarray:
    v_hat = _expected_bond_order_sum(probs.edge, pair_mask, bond_orders)
    vmax = (probs.node * jnp.asarray(max_valence, dtype=v_hat.dtype)).sum(axis=-1)
    over = jnp.maximum(v_hat - vmax, 0.0)
    mask = node_mask.astype(v_hat.dtype)
    return (jnp.square(over) * mask).sum() / jnp.maximum(mask.sum(), 1.0)


def degree_mse_penalty(
    probs: GraphLatent,
    node_mask: jnp.ndarray,
    pair_mask: jnp.ndarray,
    *,
    target_degree: jnp.ndarray,
) -> jnp.ndarray:
    deg_hat = _expected_degree(probs.edge, pair_mask)
    mask = node_mask.astype(deg_hat.dtype)
    return (jnp.square(deg_hat - target_degree) * mask).sum() / jnp.maximum(mask.sum(), 1.0)


@dataclass(frozen=True)
class LogitGuidanceConfig:
    valence_weight: float = 0.0
    degree_weight: float = 0.0
    gauge_fix: bool = True
    symmetrize_probs: bool = False
    mode: str = "state"  # "state" (default) or "logit"
    bond_orders: jnp.ndarray = field(default_factory=lambda: BOND_ORDERS)
    max_valence: jnp.ndarray = field(default_factory=lambda: VALENCE_TABLE)


def make_logit_guidance(
    config: LogitGuidanceConfig,
    *,
    weight_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[dict, jnp.ndarray, jnp.ndarray, jnp.ndarray], GraphLatent]:
    """Return a guidance function to apply to x_hat during sampling."""

    def energy(probs, node_mask, pair_mask):
        total = jnp.array(0.0, dtype=probs.node.dtype)
        if config.valence_weight:
            total = total + config.valence_weight * valence_over_penalty(
                probs,
                node_mask,
                pair_mask,
                max_valence=config.max_valence,
                bond_orders=config.bond_orders,
            )
        if config.degree_weight:
            raise ValueError("degree_weight is not supported without explicit targets.")
        return total

    def probs_from_logits(logits: GraphLatent) -> GraphLatent:
        probs = GraphLatent(
            jax.nn.softmax(logits.node, axis=-1),
            edge_probs_from_logits(logits.edge),
        )
        if config.symmetrize_probs:
            probs = GraphLatent(probs.node, symmetrize_edge_probs(probs.edge))
        return probs

    def guide(pred, xt, sigma, node_mask, pair_mask, predict_fn, logits_to_latent):
        w = 1.0 if weight_fn is None else weight_fn(sigma)
        sigma2 = jnp.square(sigma)

        def apply_gauge_fix(logits: GraphLatent) -> GraphLatent:
            if not config.gauge_fix:
                return logits
            return GraphLatent(
                center_logits(logits.node, node_mask),
                center_edge_type_logits(logits.edge, pair_mask),
            )

        if config.mode == "state":
            def energy_from_x(x):
                out = predict_fn(x, sigma, node_mask, pair_mask)
                logits = apply_gauge_fix(out["logits"])
                probs = probs_from_logits(logits)
                return energy(probs, node_mask, pair_mask)

            g = jax.grad(energy_from_x)(xt)
            scale = sigma2 * w
            x_hat = pred["x_hat"]
            guided = GraphLatent(
                x_hat.node + g.node * scale[..., None, None],
                x_hat.edge + g.edge * scale[..., None, None, None],
            )
            return guided.masked(node_mask, pair_mask)

        if config.mode == "logit":
            logits = apply_gauge_fix(pred["logits"])

            def energy_from_logits(x):
                probs = probs_from_logits(x)
                return energy(probs, node_mask, pair_mask)

            g = jax.grad(energy_from_logits)(logits)
            guided_logits = logits - g * (sigma2 * w)
            guided = logits_to_latent(guided_logits)
            return guided.masked(node_mask, pair_mask)

        raise ValueError(f"Unknown guidance mode: {config.mode}")

    return guide


__all__ = [
    "LogitGuidanceConfig",
    "make_logit_guidance",
    "valence_over_penalty",
    "degree_mse_penalty",
]
