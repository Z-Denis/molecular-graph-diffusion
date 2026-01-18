import jax.numpy as jnp

from mgd.training.losses import masked_cross_entropy
from mgd.experimental.training.losses import graph_reconstruction_loss, bond_reconstruction_loss


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


def test_graph_reconstruction_loss_matches_components():
    # Node part
    node_logits = jnp.zeros((1, 2, 3))
    atom_type = jnp.array([[1, 0]], dtype=jnp.int32)
    node_mask = jnp.array([[1.0, 0.0]])
    # Edge part: simple 2x2, mask zero to isolate node loss
    edge_logits = jnp.zeros((1, 2, 2, 2))  # existence+1 bond type but shape still >=2
    bond_type = jnp.zeros((1, 2, 2), dtype=jnp.int32)
    pair_mask = jnp.zeros((1, 2, 2))

    class Batch:
        pass
    batch = Batch()
    batch.atom_type = atom_type
    batch.bond_type = bond_type
    batch.node_mask = node_mask
    batch.pair_mask = pair_mask

    recon = {"node": node_logits, "edge": edge_logits}

    total, metrics = graph_reconstruction_loss(recon, batch)

    node_loss, _ = masked_cross_entropy(node_logits, atom_type, node_mask)
    bond_loss_val, _ = bond_reconstruction_loss(edge_logits, bond_type, pair_mask)

    assert jnp.allclose(total, node_loss + bond_loss_val)
    assert jnp.allclose(metrics["loss_node"], node_loss)
    assert jnp.allclose(metrics["loss_bond"], bond_loss_val)
