"""Training helpers for decoders (node/edge classifiers)."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from functools import partial

import flax
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from mgd.training.losses import masked_cross_entropy
from .train_step import DiffusionTrainState

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
    model_kwargs: Optional[Dict] = None,
    class_weights: Optional[jnp.ndarray] = None,
    label_smoothing: Optional[float] = None,
) -> Tuple[DecoderTrainState, Dict[str, jnp.ndarray]]:
    """One optimization step for a decoder with masked cross-entropy."""
    model_kwargs = model_kwargs or {}

    def loss_fn(params):
        logits = state.model.apply({"params": params}, latents, **model_kwargs)
        loss, metrics = masked_cross_entropy(
            logits,
            targets,
            mask,
            class_weights=class_weights,
            use_label_smoothing=label_smoothing,
        )
        return loss, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = {"loss": loss, **metrics}
    return state, metrics


def decoder_train_loop(
    diffusion_state: DiffusionTrainState,
    decoder_state: DecoderTrainState,
    loader,
    num_epochs: int,
    log_every: int = 0,
    ckpt_dir: str | None = None,
    ckpt_every: int | None = None,
    class_weights: Optional[jnp.ndarray] = None,
    label_smoothing: Optional[float] = None,
    feature_type: str = "node",
):
    """Epoch-based training loop for decoders.

    ``loader`` is expected to yield GraphBatch.
    feature_type: "node" (uses atom_type/node_mask) or "edge" (uses edges/pair_mask).
    """
    from mgd.training.checkpoints import save_checkpoint  # local import to avoid cycles
    from tqdm import tqdm

    def _mean_metrics(history):
        if not history:
            return {}
        stacked = {k: jnp.stack([m[k] for m in history]) for k in history[0]}
        return {k: v.mean() for k, v in stacked.items()}

    step_fn = jax.jit(decoder_train_step, static_argnames=("label_smoothing",))
    ls_value = label_smoothing
    history = []
    for epoch in tqdm(range(num_epochs)):
        epoch_metrics = []
        for step, batch in enumerate(loader):
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
                class_weights=class_weights,
                label_smoothing=ls_value,
            )
            epoch_metrics.append(metrics)
            if log_every and (step + 1) % log_every == 0:
                print(f"epoch {epoch+1} step {step+1}: loss={float(metrics['loss']):.4f}")
        mean_metrics = _mean_metrics(epoch_metrics)
        history.append(mean_metrics)
        if ckpt_dir and ckpt_every and (epoch + 1) % ckpt_every == 0:
            save_checkpoint(ckpt_dir, decoder_state, step=epoch + 1)
    return decoder_state, history


__all__ = [
    "DecoderTrainState",
    "create_decoder_state",
    "decoder_train_step",
    "decoder_train_loop",
]
