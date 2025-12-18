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
from mgd.training.losses import edm_masked_mse
from mgd.diffusion.schedules import sample_sigma, sample_sigma_mixture


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
    loss_fn=edm_masked_mse,
    loss_kwargs: dict | None = None,
) -> Tuple[DiffusionTrainState, dict]:
    """One optimization step with EDM denoising loss."""
    loss_kwargs = loss_kwargs or {}

    def loss_inner(params):
        rng_sigma, rng_noise = jax.random.split(rng)
        x0 = state.encode(batch)
        cfg = dict(loss_kwargs)  # avoid mutating caller
        sigma_min = cfg.pop("sigma_min", state.model.sigma_min)
        sigma_max = cfg.pop("sigma_max", state.model.sigma_max)
        p_low = cfg.pop("sigma_p_low", None)
        k_low = cfg.pop("sigma_k_low", 3.0)
        loss_kwargs_local = cfg
        if p_low is not None:
            sigma = sample_sigma_mixture(
                rng_sigma,
                (batch.atom_type.shape[0],),
                sigma_min,
                sigma_max,
                p_low=p_low,
                k=k_low,
            )
        else:
            sigma = sample_sigma(rng_sigma, (batch.atom_type.shape[0],), sigma_min, sigma_max)
        outputs = state.model.apply(
            {"params": params},
            x0,
            sigma,
            node_mask=batch.node_mask,
            pair_mask=batch.pair_mask,
            rngs={"noise": rng_noise},
        )
        loss, parts = loss_fn(
            outputs["x_hat"],
            outputs["clean"],
            batch.node_mask,
            batch.pair_mask,
            sigma=sigma,
            **loss_kwargs_local,
        )
        parts["sigma_mean"] = jnp.mean(sigma)
        return loss, parts

    (loss, metrics), grads = jax.value_and_grad(loss_inner, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = {"loss": loss, **metrics}
    return state, metrics


__all__ = ["DiffusionTrainState", "create_train_state", "train_step"]
