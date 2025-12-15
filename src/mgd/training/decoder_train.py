"""Training helpers for decoders (node/edge classifiers)."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import flax
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from mgd.training.losses import masked_cross_entropy
from mgd.utils.logging import Logger
from .train_step import DiffusionTrainState
from tqdm import tqdm

class DecoderTrainState(train_state.TrainState):
    """Train state carrying the decoder module reference."""

    model: flax.linen.Module = flax.struct.field(pytree_node=False)

    def predict_logits(self, inputs: jnp.ndarray, **model_kwargs):
        """Apply the decoder with current params."""
        return self.model.apply({"params": self.params}, inputs, **model_kwargs)


def create_decoder_state(
    model: flax.linen.Module,
    sample_latent: jnp.ndarray,
    tx: optax.GradientTransformation,
    rng: jax.Array,
    model_kwargs: Optional[Dict] = None,
) -> DecoderTrainState:
    """Initialize decoder parameters and return a train state."""
    model_kwargs = model_kwargs or {}
    variables = model.init(rng, sample_latent, **model_kwargs)
    return DecoderTrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        model=model,
    )


def decoder_train_step(
    state: DecoderTrainState,
    latents: jnp.ndarray,
    targets: jnp.ndarray,
    mask: jnp.ndarray,
    *,
    loss_fn: Callable = masked_cross_entropy,
    loss_kwargs: Optional[Dict] = None,
) -> Tuple[DecoderTrainState, Dict[str, jnp.ndarray]]:
    """One optimization step for a decoder with a configurable loss."""
    loss_kwargs = loss_kwargs or {}

    def _loss(params):
        logits = state.model.apply({"params": params}, latents)
        loss, metrics = loss_fn(logits, targets, mask, **loss_kwargs)
        return loss, metrics

    (loss, metrics), grads = jax.value_and_grad(_loss, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = {"loss": loss, **metrics}
    return state, metrics


def decoder_train_loop(
    diffusion_state: DiffusionTrainState,
    decoder_state: DecoderTrainState,
    loader,
    *,
    n_steps: int,
    feature_type: str = "node",
    loss_fn: Callable = masked_cross_entropy,
    loss_kwargs: Optional[Dict] = None,
    logger: Logger,
):
    """Step-based training loop for decoders.

    ``loader`` is expected to yield GraphBatch.
    feature_type: "node" (uses atom_type/node_mask) or "edge" (uses edges/pair_mask).
    ``loss_fn`` must follow the signature ``loss_fn(logits, targets, mask, **kwargs)``.
    A ``Logger`` must be provided; it controls logging and checkpoints.
    """
    from mgd.training.checkpoints import save_checkpoint  # local import to avoid cycles

    def _mean_metrics(history):
        if not history:
            return {}
        stacked = {k: jnp.stack([m[k] for m in history]) for k in history[0]}
        return {k: v.mean() for k, v in stacked.items()}

    loss_kwargs = loss_kwargs or {}
    step_fn = jax.jit(
        lambda st, l, y, m: decoder_train_step(
            st,
            l,
            y,
            m,
            loss_fn=loss_fn,
            loss_kwargs=loss_kwargs,
        ),
        static_argnames=(),
    )
    history = []
    metrics_buffer = []
    loader_iter = iter(loader)

    with tqdm(total=n_steps) as pbar:
        for step in range(1, n_steps + 1):
            batch = next(loader_iter)
            latents_full = diffusion_state.encode(batch)  # GraphLatent, frozen backbone
            if feature_type == "node":
                latents = latents_full.node
                targets = batch.atom_type  # integer labels
                mask = batch.node_mask
            elif feature_type == "edge":
                latents = latents_full.edge
                targets = batch.edges  # integer bond types
                mask = batch.pair_mask
            else:
                raise ValueError(f"feature_type must be 'node' or 'edge', got {feature_type}")
            decoder_state, metrics = step_fn(
                decoder_state,
                latents,
                targets,
                mask,
            )
            metrics_buffer.append(metrics)
            pbar.update(1)
            if logger.log_every and (step % logger.log_every == 0):
                loss_val = float(metrics["loss"])
                pbar.set_postfix(loss=f"{loss_val:.4f}")
            if logger.maybe_log(step, metrics_buffer):
                history.append(logger.data[-1])
                metrics_buffer = []
            logger.maybe_checkpoint(step, decoder_state)
    if metrics_buffer:
        history.append(_mean_metrics(metrics_buffer))
        logger.data.append(history[-1])
    return decoder_state, history


__all__ = [
    "DecoderTrainState",
    "create_decoder_state",
    "decoder_train_step",
    "decoder_train_loop",
]
