"""
Quick diffusion diagnostics on a trained model.

For selected timesteps t, this script:
  - takes a real batch x0
  - generates x_t with the forward process
  - predicts noise eps_hat = eps_theta(x_t, t)
  - reconstructs x0_hat = (x_t - sqrt(1 - alpha_bar_t) * eps_hat) / sqrt(alpha_bar_t)
  - reports masked mean/std drift and MSE vs. ground truth noise

Prints a table and saves a small plot of MSE vs. t.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from mgd.dataset import GraphBatchLoader
from mgd.utils import restore_checkpoint
from mgd.training.losses import masked_mse
from mgd.latent import GraphLatent, latent_from_scalar


def masked_mean_std(latent: GraphLatent, node_mask: jnp.ndarray, pair_mask: jnp.ndarray):
    node_w = node_mask.astype(latent.node.dtype)[..., None]
    edge_w = pair_mask.astype(latent.edge.dtype)[..., None]
    node_mean = (latent.node * node_w).sum() / jnp.maximum(node_w.sum(), 1)
    edge_mean = (latent.edge * edge_w).sum() / jnp.maximum(edge_w.sum(), 1)
    node_std = jnp.sqrt(((latent.node - node_mean) ** 2 * node_w).sum() / jnp.maximum(node_w.sum(), 1))
    edge_std = jnp.sqrt(((latent.edge - edge_mean) ** 2 * edge_w).sum() / jnp.maximum(edge_w.sum(), 1))
    return (node_mean, node_std), (edge_mean, edge_std)


def run_diagnostics(state, loader, timesteps: Sequence[int], clamp_eps: float = 1e-6):
    batch = next(iter(loader))
    node_mask, pair_mask = batch.node_mask, batch.pair_mask
    x0 = state.encode(batch)

    results = []

    for t_int in timesteps:
        t = jnp.full((batch.atom_type.shape[0],), t_int, dtype=jnp.int32)
        rng = jax.random.PRNGKey(0)
        rng_n, rng_e = jax.random.split(rng)
        noise_nodes = jax.random.normal(rng_n, x0.node.shape, dtype=x0.node.dtype)
        noise_edges = jax.random.normal(rng_e, x0.edge.shape, dtype=x0.edge.dtype)
        noise = GraphLatent(noise_nodes, noise_edges)

        alpha_bar_t = jnp.take(state.model.schedule.alpha_bar, t)
        scale_ab = latent_from_scalar(jnp.sqrt(alpha_bar_t))
        scale_om = latent_from_scalar(jnp.sqrt(1.0 - alpha_bar_t))

        xt = scale_ab * x0 + scale_om * noise
        eps_hat = state.predict_eps(xt, t, node_mask=node_mask, pair_mask=pair_mask)

        # MSE on noise
        mse, _ = masked_mse(eps_hat, noise, node_mask, pair_mask)

        # Reconstruct x0_hat with optional clamp for stability
        den = latent_from_scalar(jnp.sqrt(jnp.maximum(alpha_bar_t, clamp_eps)))
        x0_hat = (xt - scale_om * eps_hat) / den

        (m_node, s_node), (m_edge, s_edge) = masked_mean_std(x0, node_mask, pair_mask)
        (mh_node, sh_node), (mh_edge, sh_edge) = masked_mean_std(x0_hat, node_mask, pair_mask)

        results.append(
            {
                "t": t_int,
                "mse": float(mse),
                "x0_node_mean": float(m_node),
                "x0_node_std": float(s_node),
                "x0_edge_mean": float(m_edge),
                "x0_edge_std": float(s_edge),
                "x0hat_node_mean": float(mh_node),
                "x0hat_node_std": float(sh_node),
                "x0hat_edge_mean": float(mh_edge),
                "x0hat_edge_std": float(sh_edge),
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="Diffusion diagnostics.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to diffusion checkpoint.")
    parser.add_argument("--data", type=str, required=True, help="Path to processed data npz.")
    parser.add_argument("--splits", type=str, required=True, help="Path to splits npz.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--timesteps", type=str, default="1,50,100,250,500,750")
    args = parser.parse_args()

    # Load data/splits
    splits = dict(np.load(args.splits))
    data = dict(np.load(args.data))
    loader = GraphBatchLoader(
        data,
        indices=splits["train"],
        batch_size=args.batch_size,
        key=jax.random.PRNGKey(0),
    )

    # Build a dummy state to restore (assumes you know how to build your model)
    # You must construct `state` (DiffusionTrainState) matching your checkpoint before this script.
    raise SystemExit("Construct your DiffusionTrainState as `state` before running this script.")


if __name__ == "__main__":
    main()
