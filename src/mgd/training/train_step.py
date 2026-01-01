"""Single training step logic for diffusion."""

from __future__ import annotations

from typing import Callable, Tuple

import flax
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from mgd.dataset.utils import GraphBatch
from mgd.model.diffusion_model import GraphDiffusionModel
from mgd.training.space import DiffusionSpace
from mgd.diffusion.schedules import sample_sigma


class DiffusionTrainState(train_state.TrainState):
    """Train state carrying the model reference."""

    model: GraphDiffusionModel = flax.struct.field(pytree_node=False)
    space: DiffusionSpace = flax.struct.field(pytree_node=False)
    sigma_sampler: Callable[[jax.Array, tuple, float, float], jnp.ndarray] = flax.struct.field(pytree_node=False)

    def encode(self, batch: GraphBatch):
        """Encode a graph batch to latents using the configured diffusion space."""
        return self.space.encode(batch)

    def predict_xhat(
        self,
        xt,
        sigma,
        *,
        node_mask,
        pair_mask,
    ):
        """Return x_hat = denoise(xt, sigma)."""
        return self.denoise(xt, sigma, node_mask=node_mask, pair_mask=pair_mask)

    def denoise(self, xt, sigma, *, node_mask, pair_mask):
        return self.model.apply(
            {"params": self.params},
            xt,
            sigma,
            node_mask=node_mask,
            pair_mask=pair_mask,
            method=self.model.denoise,
        )


def create_train_state(
    model: GraphDiffusionModel,
    sample_latent,
    node_mask: jnp.ndarray,
    pair_mask: jnp.ndarray,
    tx: optax.GradientTransformation,
    rng: jax.Array,
    *,
    space: DiffusionSpace,
    sigma_sampler: Callable[[jax.Array, tuple, float, float], jnp.ndarray] = sample_sigma,
) -> DiffusionTrainState:
    """Initialize model parameters and return a train state."""
    rng_params, rng_noise = jax.random.split(rng)
    sigma0 = jnp.ones((sample_latent.node.shape[0],), dtype=sample_latent.node.dtype)
    variables = model.init(
        {"params": rng_params, "noise": rng_noise},
        sample_latent,
        sigma0,
        node_mask=node_mask,
        pair_mask=pair_mask,
    )
    return DiffusionTrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        model=model,
        space=space,
        sigma_sampler=sigma_sampler,
    )


def train_step(
    state: DiffusionTrainState,
    batch: GraphBatch,
    rng: jax.Array,
) -> Tuple[DiffusionTrainState, dict]:
    """One optimization step with EDM denoising loss."""

    def loss_inner(params):
        rng_sigma, rng_noise = jax.random.split(rng)
        x0 = state.encode(batch)
        sigma = state.sigma_sampler(
            rng_sigma,
            (batch.atom_type.shape[0],),
            state.model.sigma_min,
            state.model.sigma_max,
        )
        outputs = state.model.apply(
            {"params": params},
            x0,
            sigma,
            node_mask=batch.node_mask,
            pair_mask=batch.pair_mask,
            rngs={"noise": rng_noise},
        )
        loss, parts = state.space.loss(outputs, batch)
        parts["sigma_mean"] = jnp.mean(sigma)
        return loss, parts

    (loss, metrics), grads = jax.value_and_grad(loss_inner, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = {"loss": loss, **metrics}
    return state, metrics


__all__ = ["DiffusionTrainState", "create_train_state", "train_step"]
