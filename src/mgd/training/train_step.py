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
) -> Tuple[DiffusionTrainState, dict]:
    """One optimization step with noise prediction loss."""
    num_steps = state.model.schedule.betas.shape[0]

    def loss_fn(params):
        rng_t, rng_noise = jax.random.split(rng)
        t = jax.random.randint(rng_t, shape=(batch.atom_type.shape[0],), minval=0, maxval=num_steps)
        outputs = state.model.apply({"params": params}, batch, t, rngs={"noise": rng_noise})
        loss, parts = masked_mse(
            outputs["eps_pred"],
            outputs["noise"],
            batch.node_mask,
            batch.pair_mask,
        )
        parts["t_mean"] = jnp.mean(t)
        return loss, parts

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = {"loss": loss, **metrics}
    return state, metrics


__all__ = ["DiffusionTrainState", "create_train_state", "train_step"]
