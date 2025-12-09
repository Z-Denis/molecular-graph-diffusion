"""Sampling loop that runs reverse diffusion using an updater."""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp

from mgd.latent import GraphLatent, AbstractLatentSpace
from mgd.sampling.updater import BaseUpdater
from mgd.training.train_step import DiffusionTrainState


class GraphSampler:
    """Run the reverse diffusion chain with a provided updater."""

    def __init__(self, space: AbstractLatentSpace, state: DiffusionTrainState, updater: BaseUpdater):
        """
        Build a sampler from a latent space, a train state and an instantiated updater.
        Args:
            space: Latent space
            state: Diffusion train state
            updater: Reverse-step policy (e.g., DDPMUpdater(schedule))
        """
        self.space = space
        predict_fn = lambda xt, t, node_mask, pair_mask: state.predict_eps(
            xt,
            t,
            node_mask=node_mask,
            pair_mask=pair_mask,
        )
        self.predict_fn = predict_fn
        self.updater = updater

    def sample(
        self,
        rng: jax.Array,
        *,
        n_atoms: int,
        n_steps: int | None = None,
        batch_size: int = 1,
        node_mask: Optional[jnp.ndarray] = None,
        pair_mask: Optional[jnp.ndarray] = None,
        snapshot_steps: Optional[jnp.ndarray] = None,
    ):
        """Iteratively sample x_0 from noise using the provided updater.

        If snapshot_steps is provided, returns (x0, snapshots) where snapshots
        has shape (n_snaps, ...) ordered by the provided steps (descending t).
        """
        if node_mask is None:
            node_mask = jnp.ones((batch_size, n_atoms), dtype=self.space.dtype)
        if pair_mask is None:
            pair_mask = node_mask[..., :, None] * node_mask[..., None, :]
        xt = self.space.random_latent(rng, batch_size, n_atoms, node_mask=node_mask, pair_mask=pair_mask)

        if n_steps is None:
            n_steps = len(self.updater.schedule.betas) - 1
        times = jnp.arange(n_steps, 0, -1, dtype=jnp.int32)

        record = snapshot_steps is not None
        if record:
            snapshot_steps = jnp.asarray(snapshot_steps)
            snapshot_steps = jnp.sort(snapshot_steps)[::-1]  # descending to match loop
            snaps_init = (
                GraphLatent(
                    node=jnp.zeros((snapshot_steps.shape[0],) + xt.node.shape, dtype=xt.node.dtype),
                    edge=jnp.zeros((snapshot_steps.shape[0],) + xt.edge.shape, dtype=xt.edge.dtype),
                )
            )
        else:
            snaps_init = None

        def body(carry, t):
            if record:
                xt_c, snaps_c = carry
            else:
                xt_c = carry
            step_rng = jax.random.fold_in(rng, t)
            eps = self.predict_fn(xt_c, t, node_mask, pair_mask)
            xt_next = self.updater.step(xt_c, eps, t, node_mask, pair_mask, rng=step_rng)
            if record:
                # Write snapshot if t is in snapshot_steps
                match = jnp.where(snapshot_steps == t, size=1, fill_value=-1)[0][0]
                snaps = GraphLatent(
                    node=jax.lax.select(
                        match >= 0,
                        snaps_c.node.at[match].set(xt_next.node),
                        snaps_c.node,
                    ),
                    edge=jax.lax.select(
                        match >= 0,
                        snaps_c.edge.at[match].set(xt_next.edge),
                        snaps_c.edge,
                    ),
                )
                return (xt_next, snaps), None
            return xt_next, None

        if record:
            (xt_final, snaps_out), _ = jax.lax.scan(body, (xt, snaps_init), times)
            return xt_final, snaps_out
        xt_final, _ = jax.lax.scan(body, xt, times)
        return xt_final


__all__ = ["GraphSampler"]
