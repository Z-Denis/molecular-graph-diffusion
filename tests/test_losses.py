import jax.numpy as jnp

from mgd.training.losses import masked_cross_entropy


def test_masked_cross_entropy_node_only():
    logits = jnp.zeros((1, 2, 3))
    labels = jnp.array([[1, 2]])
    mask = jnp.array([[1.0, 0.0]])

    total, metrics = masked_cross_entropy(logits, labels, mask)

    # Only the first node contributes: CE = -log(1/3) = log(3)
    expected = jnp.log(3.0)
    assert jnp.allclose(total, expected)
    assert jnp.allclose(metrics["loss"], expected)


def test_masked_cross_entropy_with_edges():
    # Shape (B, N, N, C)
    edge_logits = jnp.zeros((1, 2, 2, 2))
    edge_labels = jnp.array([[[0, 1], [1, 0]]])
    pair_mask = jnp.array([[[1.0, 1.0], [1.0, 0.0]]])

    total, metrics = masked_cross_entropy(edge_logits, edge_labels, pair_mask)

    # For uniform logits, CE = log(2) on each unmasked entry.
    expected = jnp.log(2.0)
    assert jnp.allclose(metrics["loss"], expected)


def test_masked_cross_entropy_label_smoothing():
    logits = jnp.array([[[2.0, 0.0, 0.0]]])  # favors class 0
    targets = jnp.array([[1]])  # true class 1
    mask = jnp.array([[1.0]])
    eps = 0.2

    total_smooth, _ = masked_cross_entropy(logits, targets, mask, use_label_smoothing=eps)
    total_hard, _ = masked_cross_entropy(logits, targets, mask, use_label_smoothing=None)

    # Smoothing should reduce the loss relative to the hard label.
    assert total_smooth < total_hard
