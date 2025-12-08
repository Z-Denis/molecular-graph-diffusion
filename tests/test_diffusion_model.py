import jax
import jax.numpy as jnp

from mgd.dataset.utils import GraphBatch
from mgd.diffusion.schedules import cosine_beta_schedule
from mgd.model.denoiser import MPNNDenoiser
from mgd.model.diffusion_model import GraphDiffusionModel
from mgd.model.embeddings import GraphEmbedder


def _tiny_batch():
    # Two graphs, three nodes (third node is padded)
    atom_type = jnp.array([[1, 2, 0], [0, 1, 0]], dtype=jnp.int32)
    hybrid = jnp.zeros_like(atom_type)
    cont = jnp.zeros((2, 3, 1), dtype=jnp.float32)
    edges = jnp.zeros((2, 3, 3), dtype=jnp.int32)
    node_mask = (atom_type > 0).astype(jnp.float32)
    pair_mask = (node_mask[..., :, None] * node_mask[..., None, :]).astype(jnp.float32)
    return GraphBatch(
        atom_type=atom_type,
        hybrid=hybrid,
        cont=cont,
        edges=edges,
        node_mask=node_mask,
        pair_mask=pair_mask,
    )


def _tiny_model():
    embedder = GraphEmbedder(
        atom_embed_dim=4,
        hybrid_embed_dim=3,
        cont_embed_dim=2,
        node_hidden_dim=5,
        edge_embed_dim=3,
        edge_hidden_dim=4,
    )
    denoiser = MPNNDenoiser(
        node_dim=5,
        edge_dim=4,
        mess_dim=6,
        time_dim=8,
    )
    schedule = cosine_beta_schedule(timesteps=10)
    return GraphDiffusionModel(embedder=embedder, denoiser=denoiser, schedule=schedule)


def test_training_forward_shapes_and_keys():
    batch = _tiny_batch()
    model = _tiny_model()
    rng = {"params": jax.random.PRNGKey(0), "noise": jax.random.PRNGKey(1)}
    times = jnp.array([1, 2], dtype=jnp.int32)
    variables = model.init(rng, batch, times)
    outputs = model.apply(variables, batch, times, rngs={"noise": jax.random.PRNGKey(2)})
    for key in ("eps_pred", "noise", "noisy", "clean"):
        assert key in outputs
        assert isinstance(outputs[key].node, jnp.ndarray)
        assert isinstance(outputs[key].edge, jnp.ndarray)
    assert outputs["eps_pred"].node.shape == batch.cont.shape[:2] + (5,)
    assert outputs["eps_pred"].edge.shape == batch.pair_mask.shape + (4,)


def test_sample_masks_and_shapes_reproducible():
    batch = _tiny_batch()
    model = _tiny_model()
    rngs = {"params": jax.random.PRNGKey(0), "noise": jax.random.PRNGKey(1)}
    variables = model.init(rngs, batch, jnp.array([1, 2], dtype=jnp.int32))
    init_latent = model.apply(variables, batch, method=model.encode)

    base_rng = jax.random.PRNGKey(42)
    out1 = model.apply(
        variables,
        method=model.sample,
        rng=base_rng,
        num_steps=3,
        init_latent=init_latent,
        node_mask=batch.node_mask,
        pair_mask=batch.pair_mask,
    )
    out2 = model.apply(
        variables,
        method=model.sample,
        rng=base_rng,
        num_steps=3,
        init_latent=init_latent,
        node_mask=batch.node_mask,
        pair_mask=batch.pair_mask,
    )

    assert jnp.allclose(out1.node, out2.node)
    assert jnp.allclose(out1.edge, out2.edge)
    # Masked regions should stay zero
    assert jnp.allclose(out1.node * (1.0 - batch.node_mask)[..., None], 0.0)
    assert jnp.allclose(out1.edge * (1.0 - batch.pair_mask)[..., None], 0.0)
