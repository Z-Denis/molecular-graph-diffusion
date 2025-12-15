"""Single training step logic for diffusion."""

from __future__ import annotations

from typing import Tuple

import flax
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from mgd.dataset.utils import GraphBatch
from mgd.model.diffusion_model import GraphDiffusionModel
from mgd.training.losses import masked_mse


class DiffusionTrainState(train_state.TrainState):
    """Train state carrying the model reference."""

    model: GraphDiffusionModel = flax.struct.field(pytree_node=False)

    def encode(self, batch: GraphBatch):
        """Encode a graph batch to latents using current params."""
        return self.model.apply({"params": self.params}, batch, method=self.model.encode)

    def predict_eps(
        self,
        xt,
        t,
        *,
        node_mask,
        pair_mask,
    ):
        """Alias to model.predict_eps bound with current params."""
        return self.model.apply(
            {"params": self.params},
            xt,
            t,
            node_mask=node_mask,
            pair_mask=pair_mask,
            method=self.model.predict_eps,
        )


def create_train_state(
    model: GraphDiffusionModel,
    batch: GraphBatch,
    tx: optax.GradientTransformation,
    rng: jax.Array,
) -> DiffusionTrainState:
    """Initialize model parameters and return a train state."""
    rng_params, rng_noise = jax.random.split(rng)
    t0 = jnp.zeros((batch.atom_type.shape[0],), dtype=jnp.int32)
    variables = model.init({"params": rng_params, "noise": rng_noise}, batch, t0)
    return DiffusionTrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        model=model,
    )


def train_step(
    state: DiffusionTrainState,
    batch: GraphBatch,
    rng: jax.Array,
    *,
    loss_fn=masked_mse,
    loss_kwargs: dict | None = None,
    use_p2: bool = False,
    p2_exponent: float = 0.5,
) -> Tuple[DiffusionTrainState, dict]:
    """One optimization step with noise prediction loss.

    If ``use_p2`` is True, applies SNR-based p2 weighting (as in Imagen/DDPM++) with
    weight = (SNR / (SNR + 1)) ** p2_exponent.
    """
    num_steps = state.model.schedule.betas.shape[0]
    loss_kwargs = loss_kwargs or {}

    def loss_inner(params):
        rng_t, rng_noise = jax.random.split(rng)
        t = jax.random.randint(rng_t, shape=(batch.atom_type.shape[0],), minval=0, maxval=num_steps)
        outputs = state.model.apply({"params": params}, batch, t, rngs={"noise": rng_noise})
        if use_p2:
            snr_t = state.model.schedule.snr(t)
            weight = (snr_t / (snr_t + 1.0)) ** p2_exponent
            node_w = weight
            edge_w = weight
        else:
            node_w = None
            edge_w = None

        loss, parts = loss_fn(
            outputs["eps_pred"],
            outputs["noise"],
            batch.node_mask,
            batch.pair_mask,
            node_weight=node_w,
            edge_weight=edge_w,
            **loss_kwargs,
        )
        parts["t_mean"] = jnp.mean(t)
        return loss, parts

    (loss, metrics), grads = jax.value_and_grad(loss_inner, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = {"loss": loss, **metrics}
    return state, metrics


__all__ = ["DiffusionTrainState", "create_train_state", "train_step"]
