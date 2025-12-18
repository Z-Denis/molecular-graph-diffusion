"""Training utilities for a graph autoencoder with latent normalization support."""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple, Iterable

import flax
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from mgd.latent import GraphLatent
from tqdm import tqdm


def _stat_for(value: Any, part: str):
    """Extract a node/edge statistic from a GraphLatent, mapping, or scalar/array."""
    if isinstance(value, GraphLatent):
        return getattr(value, part)
    if isinstance(value, dict):
        return value[part]
    return value


def normalize_latent(latent: GraphLatent, mean: Any, std: Any) -> GraphLatent:
    """Apply (latent - mean) / std per component if stats are provided."""
    if mean is None or std is None:
        return latent
    return GraphLatent(
        node=(latent.node - _stat_for(mean, "node")) / _stat_for(std, "node"),
        edge=(latent.edge - _stat_for(mean, "edge")) / _stat_for(std, "edge"),
    )


def denormalize_latent(latent: GraphLatent, mean: Any, std: Any) -> GraphLatent:
    """Apply latent * std + mean per component if stats are provided."""
    if mean is None or std is None:
        return latent
    return GraphLatent(
        node=latent.node * _stat_for(std, "node") + _stat_for(mean, "node"),
        edge=latent.edge * _stat_for(std, "edge") + _stat_for(mean, "edge"),
    )


class AutoencoderTrainState(train_state.TrainState):
    """Train state carrying the autoencoder module and latent normalization stats."""

    model: flax.linen.Module = flax.struct.field(pytree_node=False)
    latent_mean: Any = None
    latent_std: Any = None

    def normalize(self, mean: Any, std: Any) -> "AutoencoderTrainState":
        """Return a new state with stored latent normalization stats."""
        return self.replace(latent_mean=mean, latent_std=std)

    def encode(self, batch, *, apply_norm: bool = True):
        latents = self.model.apply({"params": self.params}, batch, method=self.model.encode)
        return normalize_latent(latents, self.latent_mean, self.latent_std) if apply_norm else latents

    def decode(self, latents: GraphLatent, *, denormalize: bool = True):
        latents_in = denormalize_latent(latents, self.latent_mean, self.latent_std) if denormalize else latents
        return self.model.apply({"params": self.params}, latents_in, method=self.model.decode)

    def reconstruct(self, batch):
        latents = self.encode(batch, apply_norm=True)
        recon = self.decode(latents, denormalize=True)
        return recon, latents


def create_autoencoder_state(
    model: flax.linen.Module,
    batch,
    tx: optax.GradientTransformation,
    rng: jax.Array,
) -> AutoencoderTrainState:
    """Initialize autoencoder parameters and return a train state."""
    variables = model.init({"params": rng}, batch)
    return AutoencoderTrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        model=model,
    )


def autoencoder_train_step(
    state: AutoencoderTrainState,
    batch,
    *,
    loss_fn: Callable,
    loss_kwargs: Dict | None = None,
) -> Tuple[AutoencoderTrainState, Dict[str, jnp.ndarray]]:
    """One optimization step for the autoencoder with a user-provided loss_fn.

    ``loss_fn`` must accept ``recon``, ``batch``, and ``latents`` keywords and
    return ``(loss, metrics_dict)``.
    """
    loss_kwargs = loss_kwargs or {}

    def _loss(params):
        recon, latents = state.model.apply({"params": params}, batch)
        loss, metrics = loss_fn(recon=recon, batch=batch, latents=latents, **loss_kwargs)
        return loss, metrics

    (loss, metrics), grads = jax.value_and_grad(_loss, has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads)
    metrics = {"loss": loss, **metrics}
    return new_state, metrics


def _mean_metrics(history):
    if not history:
        return {}
    stacked = {k: jnp.stack([m[k] for m in history]) for k in history[0]}
    return {k: v.mean() for k, v in stacked.items()}


def autoencoder_train_loop(
    state: AutoencoderTrainState,
    loader: Iterable,
    *,
    n_steps: int,
    logger,
    loss_fn: Callable,
    loss_kwargs: Dict | None = None,
):
    """Step-based training loop for the autoencoder."""
    loss_kwargs = loss_kwargs or {}
    step_fn = jax.jit(lambda st, b: autoencoder_train_step(st, b, loss_fn=loss_fn, loss_kwargs=loss_kwargs))

    history = []
    metrics_buffer = []
    loader_iter = iter(loader)

    with tqdm(total=n_steps) as pbar:
        for step in range(1, n_steps + 1):
            batch = next(loader_iter)
            state, metrics = step_fn(state, batch)
            metrics_buffer.append(metrics)
            pbar.update(1)
            if logger.log_every and (step % logger.log_every == 0):
                postfix = {k: f"{float(v):.4f}" for k, v in metrics.items()}
                pbar.set_postfix(**postfix)
            if logger.maybe_log(step, metrics_buffer):
                history.append(logger.data[-1])
                metrics_buffer = []
            logger.maybe_checkpoint(step, state)

    if metrics_buffer:
        history.append(_mean_metrics(metrics_buffer))
        logger.data.append(history[-1])

    return state, history


__all__ = [
    "AutoencoderTrainState",
    "create_autoencoder_state",
    "autoencoder_train_step",
    "normalize_latent",
    "denormalize_latent",
    "autoencoder_train_loop",
]
