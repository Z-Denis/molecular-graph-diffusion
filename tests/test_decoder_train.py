import jax
import jax.numpy as jnp
import optax

from mgd.model.decoder import NodeCategoricalDecoder
from mgd.training.decoder_train import (
    DecoderTrainState,
    create_decoder_state,
    decoder_train_step,
)
from mgd.training.losses import masked_cross_entropy


def test_decoder_train_step_reduces_loss():
    rng = jax.random.PRNGKey(0)
    latent = jnp.zeros((2, 3, 4))  # (batch, atoms, features)
    mask = jnp.ones((2, 3))
    # Imbalanced labels to ensure non-symmetric gradients
    targets = jnp.zeros((2, 3), dtype=jnp.int32)

    model = NodeCategoricalDecoder(hidden_dim=8, n_layers=1, n_categories=3)
    tx = optax.sgd(0.5)

    state = create_decoder_state(model, latent, mask, tx, rng)

    def compute_loss(params):
        logits = state.model.apply({"params": params}, latent)
        loss, _ = masked_cross_entropy(logits, targets, mask)
        return loss

    loss_before = compute_loss(state.params)
    for _ in range(3):
        state, metrics = decoder_train_step(
            state,
            latent,
            targets,
            mask,
        )
    loss_after = compute_loss(state.params)

    assert loss_after < loss_before
    assert "loss" in metrics


def test_predict_logits_matches_apply():
    rng = jax.random.PRNGKey(0)
    latent = jnp.zeros((1, 2, 4))
    mask = jnp.ones((1, 2))
    targets = jnp.zeros((1, 2), dtype=jnp.int32)
    model = NodeCategoricalDecoder(hidden_dim=4, n_layers=1, n_categories=3)
    tx = optax.sgd(0.1)
    state = create_decoder_state(model, latent, mask, tx, rng)

    logits_apply = model.apply({"params": state.params}, latent)
    logits_state = state.predict_logits(latent)
    assert jnp.allclose(logits_apply, logits_state)


def test_predict_logits_accepts_graphlatent():
    rng = jax.random.PRNGKey(0)
    latent = jnp.zeros((1, 2, 4))
    mask = jnp.ones((1, 2))
    targets = jnp.zeros((1, 2), dtype=jnp.int32)
    model = NodeCategoricalDecoder(hidden_dim=4, n_layers=1, n_categories=3)
    tx = optax.sgd(0.1)
    state = create_decoder_state(model, latent, mask, tx, rng)

    # expects array input
    logits = state.predict_logits(latent)
    assert logits.shape == (1, 2, 3)
