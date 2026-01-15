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
        return self.model.apply(
            {"params": self.params},
            batch,
            node_mask=batch.node_mask,
            pair_mask=batch.pair_mask,
            method=self.model.encode,
        )

    def predict_xhat(
        self,
        xt,
        sigma,
        *,
        node_mask,
        pair_mask,
    ):
        """Return x_hat = denoise(xt, sigma)."""
        return self.denoise(xt, sigma, node_mask=node_mask, pair_mask=pair_mask)["x_hat"]

    def denoise(self, xt, sigma, *, node_mask, pair_mask):
        return self.model.apply(
            {"params": self.params},
            xt,
            sigma,
            node_mask=node_mask,
            pair_mask=pair_mask,
            method=self.model.denoise,
        )

    def logits_to_latent(self, logits):
        return self.model.apply(
            {"params": self.params},
            logits,
            method=self.model.logits_to_latent,
        )


def create_train_state(
    model: GraphDiffusionModel,
    batch: GraphBatch,
    tx: optax.GradientTransformation,
    rng: jax.Array,
    *,
    space: DiffusionSpace,
    sigma_sampler: Callable[[jax.Array, tuple, float, float], jnp.ndarray] = sample_sigma,
) -> DiffusionTrainState:
    """Initialize model parameters from a batch and return a train state."""
    node_mask = batch.node_mask
    pair_mask = batch.pair_mask
    rng_params, rng_noise = jax.random.split(rng)

    dummy_latent = model.denoiser.space.zeros_from_masks(node_mask, pair_mask)
    sigma0 = jnp.ones((dummy_latent.node.shape[0],), dtype=dummy_latent.node.dtype)
    variables = model.init(
        {"params": rng_params, "noise": rng_noise},
        dummy_latent,
        sigma0,
        node_mask=node_mask,
        pair_mask=pair_mask,
    )
    embed_vars = model.init(
        {"params": rng_params},
        batch,
        node_mask=node_mask,
        pair_mask=pair_mask,
        method=model.encode,
    )
    params = flax.core.unfreeze(variables["params"])
    params.update(flax.core.unfreeze(embed_vars["params"]))
    params = flax.core.freeze(params)
    return DiffusionTrainState.create(
        apply_fn=model.apply,
        params=params,
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
