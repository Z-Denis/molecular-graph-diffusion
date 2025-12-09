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
    ) -> GraphLatent:
        """Iteratively sample x_0 from noise using the provided updater."""
        if node_mask is None:
            node_mask = jnp.ones((batch_size, n_atoms), dtype=self.space.dtype)
        if pair_mask is None:
            pair_mask = node_mask[..., :, None] * node_mask[..., None, :]
        xt = self.space.random_latent(rng, batch_size, n_atoms, node_mask=node_mask, pair_mask=pair_mask)

        if n_steps is None:
            n_steps = len(self.updater.schedule.betas)
        for step in range(n_steps, 0, -1):
            t = jnp.full(xt.node.shape[:-2], step, dtype=jnp.int32)
            rng, step_rng = jax.random.split(rng)
            eps = self.predict_fn(xt, t, node_mask, pair_mask)
            xt = self.updater.step(xt, eps, t, node_mask, pair_mask, rng=step_rng)

        return xt


__all__ = ["GraphSampler"]
