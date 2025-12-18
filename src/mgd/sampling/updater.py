"""EDM reverse-step policies (deterministic Heun)."""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp

from mgd.latent import GraphLatent


class HeunUpdater:
    """Deterministic Heun (predictor-corrector) for dx/dsigma = (x - x_hat)/sigma."""

    def step(
        self,
        xt: GraphLatent,
        x_hat: GraphLatent,
        x_pred: GraphLatent,
        x_hat_next: GraphLatent,
        sigma: jnp.ndarray,
        sigma_next: jnp.ndarray,
        node_mask: jnp.ndarray,
        pair_mask: jnp.ndarray,
        rng: Optional[jax.Array] = None,
    ) -> GraphLatent:
        del rng  # deterministic
        sigma_safe = jnp.maximum(sigma, 1e-12)
        sigma_next_safe = jnp.maximum(sigma_next, 1e-12)
        ds = sigma_next - sigma
        dxt = GraphLatent(
            (xt.node - x_hat.node) / sigma_safe[..., None, None],
            (xt.edge - x_hat.edge) / sigma_safe[..., None, None, None],
        )
        # Corrector uses x_hat_next recomputed at (x_pred, sigma_next)
        dxt_next = GraphLatent(
            (x_pred.node - x_hat_next.node) / sigma_next_safe[..., None, None],
            (x_pred.edge - x_hat_next.edge) / sigma_next_safe[..., None, None, None],
        )
        # If sigma_next is effectively zero, fall back to Euler (use dxt)
        is_last = (sigma_next <= 1e-12)
        dxt_next = GraphLatent(
            jnp.where(is_last[..., None, None], dxt.node, dxt_next.node),
            jnp.where(is_last[..., None, None, None], dxt.edge, dxt_next.edge),
        )
        dxt_avg_node = 0.5 * (dxt.node + dxt_next.node)
        dxt_avg_edge = 0.5 * (dxt.edge + dxt_next.edge)
        x_next = GraphLatent(
            xt.node + ds[..., None, None] * dxt_avg_node,
            xt.edge + ds[..., None, None, None] * dxt_avg_edge,
        )
        return x_next.masked(node_mask, pair_mask)


__all__ = ["HeunUpdater"]
