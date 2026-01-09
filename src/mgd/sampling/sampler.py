"""Sampling loop that runs reverse diffusion using an updater."""

from __future__ import annotations

from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp

from mgd.latent import GraphLatent, AbstractLatentSpace, symmetrize_latent
from mgd.sampling.updater import HeunUpdater
from mgd.training.train_step import DiffusionTrainState


def _prepare_masks(
    n_atoms: Union[int, jnp.ndarray],
    batch_size: int,
    space_dtype,
    node_mask: Optional[jnp.ndarray],
    pair_mask: Optional[jnp.ndarray],
    max_atoms: int,
):
    if isinstance(n_atoms, int):
        n_atoms_arr = jnp.full((batch_size,), n_atoms, dtype=jnp.int32)
    else:
        n_atoms_arr = jnp.asarray(n_atoms, dtype=jnp.int32)
        batch_size = int(n_atoms_arr.shape[0])

    if max_atoms < int(jnp.max(n_atoms_arr)):
        raise ValueError("max_atoms must be >= max(n_atoms) for this batch.")
    max_atoms = int(max_atoms)

    if node_mask is None:
        arange = jnp.arange(max_atoms, dtype=space_dtype)
        node_mask = (arange[None, :] < n_atoms_arr[:, None]).astype(space_dtype)
    if pair_mask is None:
        pair_mask = node_mask[..., :, None] * node_mask[..., None, :]
        eye = jnp.eye(max_atoms, dtype=pair_mask.dtype)
        pair_mask = pair_mask * (1.0 - eye)

    return batch_size, max_atoms, node_mask, pair_mask


class LatentSampler:
    """Run the reverse EDM chain with a provided updater."""

    def __init__(self, space: AbstractLatentSpace, state: DiffusionTrainState, updater: HeunUpdater | None = None):
        self.space = space
        predict_fn = lambda xt, sigma, node_mask, pair_mask: state.denoise(
            xt,
            sigma,
            node_mask=node_mask,
            pair_mask=pair_mask,
        )
        self.predict_fn = predict_fn
        self.updater = updater or HeunUpdater()

    def sample(
        self,
        rng: jax.Array,
        *,
        n_atoms: Union[int, jnp.ndarray],
        sigma_schedule: jnp.ndarray,
        batch_size: int = 1,
        node_mask: Optional[jnp.ndarray] = None,
        pair_mask: Optional[jnp.ndarray] = None,
        snapshot_steps: Optional[jnp.ndarray] = None,
        max_atoms: int = None,
        guidance_fn: Optional[
            Callable[[GraphLatent, jnp.ndarray, jnp.ndarray, jnp.ndarray], GraphLatent]
        ] = None,
    ):
        """Iteratively sample x_0 from noise using the provided updater.

        If snapshot_steps is provided (indices into the sigma_schedule array),
        returns (x0, snapshots) where snapshots has shape (n_snaps, ...).
        """
        if max_atoms is None:
            raise ValueError("max_atoms must be provided to sample.")
        batch_size, max_atoms, node_mask, pair_mask = _prepare_masks(
            n_atoms, batch_size, self.space.dtype, node_mask, pair_mask, max_atoms
        )
        sigma_schedule = jnp.asarray(sigma_schedule)
        xt = self.space.random_latent(
            rng,
            batch_size,
            max_atoms,
            node_mask=node_mask,
            pair_mask=pair_mask,
        )
        xt = symmetrize_latent(xt, node_mask, pair_mask) * sigma_schedule[0]

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

        def body(carry, idx):
            if record:
                xt_c, snaps_c, rng_c = carry
            else:
                xt_c, rng_c = carry
            rng_c, step_rng = jax.random.split(rng_c)
            sigma = sigma_schedule[idx]
            sigma_next = sigma_schedule[jnp.minimum(idx + 1, sigma_schedule.shape[0] - 1)]
            x_hat = self.predict_fn(xt_c, sigma, node_mask, pair_mask)
            x_hat = symmetrize_latent(x_hat, node_mask, pair_mask)
            if guidance_fn is not None:
                x_hat = guidance_fn(x_hat, node_mask, pair_mask, sigma)
                x_hat = symmetrize_latent(x_hat, node_mask, pair_mask)
            ds = sigma_next - sigma
            slope = GraphLatent(
                (xt_c.node - x_hat.node) / sigma[..., None, None],
                (xt_c.edge - x_hat.edge) / sigma[..., None, None, None],
            )
            x_pred = GraphLatent(
                xt_c.node + ds[..., None, None] * slope.node,
                xt_c.edge + ds[..., None, None, None] * slope.edge,
            ).masked(node_mask, pair_mask)
            x_hat_next = self.predict_fn(x_pred, sigma_next, node_mask, pair_mask)
            xt_next = self.updater.step(
                xt_c,
                x_hat,
                x_pred,
                x_hat_next,
                sigma,
                sigma_next,
                node_mask,
                pair_mask,
                rng=step_rng,
            )
            xt_next = symmetrize_latent(xt_next, node_mask, pair_mask)
            if record:
                match = jnp.where(snapshot_steps == idx, size=1, fill_value=-1)[0][0]
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
                return (xt_next, snaps, rng_c), None
            return (xt_next, rng_c), None

        if record:
            (xt_final, snaps_out, _), _ = jax.lax.scan(
                body, (xt, snaps_init, rng), jnp.arange(sigma_schedule.shape[0] - 1)
            )
            return xt_final, snaps_out
        (xt_final, _), _ = jax.lax.scan(body, (xt, rng), jnp.arange(sigma_schedule.shape[0] - 1))
        return xt_final


__all__ = ["LatentSampler"]
