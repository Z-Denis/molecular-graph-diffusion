import jax
import jax.numpy as jnp
import optax

from mgd.experimental.dataset.utils import GraphBatch
from mgd.latent import GraphLatentSpace, GraphLatent
from mgd.experimental.model.autoencoder import GraphAutoencoder
from mgd.experimental.model.embeddings import GraphEmbedder
from mgd.experimental.model.decoder import EdgeCategoricalDecoder, NodeCategoricalDecoder, GraphDecoder
from mgd.experimental.training.autoencoder import create_autoencoder_state


def _small_batch():
    atom_type = jnp.array([[1, 2]], dtype=jnp.int32)
    hybrid = jnp.zeros_like(atom_type)
    cont = jnp.zeros((1, 2, 1), dtype=jnp.float32)
    bond_type = jnp.zeros((1, 2, 2), dtype=jnp.int32)
    dknn = jnp.zeros((1, 2, 2, 1), dtype=jnp.float32)
    node_mask = jnp.ones_like(atom_type, dtype=jnp.float32)
    pair_mask = jnp.ones_like(bond_type, dtype=jnp.float32)
    return GraphBatch(
        atom_type=atom_type,
        hybrid=hybrid,
        cont=cont,
        bond_type=bond_type,
        dknn=dknn,
        node_mask=node_mask,
        pair_mask=pair_mask,
    )


def test_autoencoder_normalization_round_trip():
    space = GraphLatentSpace(node_dim=3, edge_dim=2, dtype=jnp.float32)
    embedder = GraphEmbedder(
        space=space,
        atom_embed_dim=2,
        hybrid_embed_dim=2,
        atom_cont_embed_dim=2,
        bond_embed_dim=2,
        bond_cont_embed_dim=2,
    )
    decoder = GraphDecoder(
        node_decoder=NodeCategoricalDecoder(hidden_dim=4, n_layers=1, n_categories=3),
        edge_decoder=EdgeCategoricalDecoder(hidden_dim=4, n_layers=1, n_categories=3),
    )
    model = GraphAutoencoder(embedder=embedder, decoder=decoder)

    batch = _small_batch()
    tx = optax.identity()
    state = create_autoencoder_state(model, batch, tx, jax.random.PRNGKey(0))

    # Compute raw latents then store simple normalization stats
    raw_latent = state.encode(batch, apply_norm=False)
    mean = GraphLatent(node=jnp.ones_like(raw_latent.node), edge=jnp.ones_like(raw_latent.edge))
    std = GraphLatent(node=2.0 * jnp.ones_like(raw_latent.node), edge=2.0 * jnp.ones_like(raw_latent.edge))
    normed_state = state.normalize(mean, std)

    # Encoding with normalization should apply (x - mean) / std
    latents_norm = normed_state.encode(batch)
    assert jnp.allclose(latents_norm.node, (raw_latent.node - 1.0) / 2.0)
    assert jnp.allclose(latents_norm.edge, (raw_latent.edge - 1.0) / 2.0)

    # Decoding normalized latents should internally denormalize back
    decoded_from_norm = normed_state.decode(latents_norm)
    decoded_from_raw = normed_state.decode(raw_latent, denormalize=False)
    assert jnp.allclose(decoded_from_norm["edge"], decoded_from_raw["edge"])
    assert jnp.allclose(decoded_from_norm["node"], decoded_from_raw["node"])

    # Reconstruct uses normalized encode + denormalized decode
    recon, used_latents = normed_state.reconstruct(batch)
    assert jnp.allclose(used_latents.node, latents_norm.node)
    assert jnp.allclose(recon["edge"], decoded_from_raw["edge"])
    assert jnp.allclose(recon["node"], decoded_from_raw["node"])
