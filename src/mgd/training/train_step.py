"""Single training step logic for diffusion."""

from __future__ import annotations

from typing import Any, Callable, Tuple

import flax
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from mgd.dataset.utils import GraphBatch
from mgd.model.diffusion_model import GraphDiffusionModel
from mgd.training.autoencoder import normalize_latent
from mgd.training.losses import masked_mse


class DiffusionTrainState(train_state.TrainState):
    """Train state carrying the model reference."""

    model: GraphDiffusionModel = flax.struct.field(pytree_node=False)
    encoder_apply_fn: Callable[..., Any] | None = flax.struct.field(pytree_node=False, default=None)
    encoder_params: Any = None
    encoder_method: Any = flax.struct.field(pytree_node=False, default=None)
    latent_mean: Any = None
    latent_std: Any = None

    def encode(self, batch: GraphBatch):
        """Encode a graph batch to latents using the frozen encoder parameters."""
        if hasattr(batch, "node") and hasattr(batch, "edge"):
            latents = batch  # already a GraphLatent
        else:
            if self.encoder_apply_fn is None or self.encoder_params is None:
                raise ValueError("encoder_apply_fn and encoder_params must be set for encoding.")
            # Accept either a module with .apply/.encode, or an apply_fn + explicit encoder_method
            if hasattr(self.encoder_apply_fn, "apply") and hasattr(self.encoder_apply_fn, "encode"):
                apply_fn = self.encoder_apply_fn.apply
                method = self.encoder_apply_fn.encode
            elif self.encoder_method is not None:
                apply_fn = self.encoder_apply_fn
                method = self.encoder_method
            else:
                raise ValueError(
                    "encoder_apply_fn must be a module with .apply/.encode or provide encoder_method explicitly."
                )
            latents = apply_fn(
                {"params": self.encoder_params},
                batch,
                method=method,
            )
        return normalize_latent(latents, self.latent_mean, self.latent_std)

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
    sample_latent,
    node_mask: jnp.ndarray,
    pair_mask: jnp.ndarray,
    tx: optax.GradientTransformation,
    rng: jax.Array,
    *,
    encoder_state: Any | None = None,
    encoder_apply_fn: Callable[..., Any] | None = None,
    encoder_params: Any = None,
    encoder_method: Any = None,
    latent_mean: Any = None,
    latent_std: Any = None,
) -> DiffusionTrainState:
    """Initialize model parameters and return a train state.

    If ``encoder_state`` (AutoencoderTrainState) is provided, encoder settings
    are inferred automatically (apply_fn=model of the AE, params=ae.params['embedder'],
    method=model.encode, latent_mean/std from the AE).
    """
    if encoder_state is not None:
        encoder_apply_fn = encoder_state.model
        encoder_params = encoder_state.params  # pass full tree so submodule names match
        encoder_method = encoder_state.model.encode
        latent_mean = encoder_state.latent_mean
        latent_std = encoder_state.latent_std

    rng_params, rng_noise = jax.random.split(rng)
    t0 = jnp.zeros((sample_latent.node.shape[0],), dtype=jnp.int32)
    variables = model.init(
        {"params": rng_params, "noise": rng_noise},
        sample_latent,
        t0,
        node_mask=node_mask,
        pair_mask=pair_mask,
    )
    return DiffusionTrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        model=model,
        encoder_apply_fn=encoder_apply_fn,
        encoder_params=encoder_params,
        encoder_method=encoder_method,
        latent_mean=latent_mean,
        latent_std=latent_std,
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
        x0 = state.encode(batch)
        t = jax.random.randint(rng_t, shape=(batch.atom_type.shape[0],), minval=0, maxval=num_steps)
        outputs = state.model.apply(
            {"params": params},
            x0,
            t,
            node_mask=batch.node_mask,
            pair_mask=batch.pair_mask,
            rngs={"noise": rng_noise},
        )
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
