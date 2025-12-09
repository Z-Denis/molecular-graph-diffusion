import jax
import jax.numpy as jnp

import optax

from mgd.dataset.utils import GraphBatch
from mgd.diffusion.schedules import cosine_beta_schedule
from mgd.latent import GraphLatentSpace
from mgd.model.denoiser import MPNNDenoiser
from mgd.model.diffusion_model import GraphDiffusionModel
from mgd.model.embeddings import GraphEmbedder
from mgd.sampling import DDPMUpdater, GraphSampler
from mgd.training.train_step import DiffusionTrainState


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
    space = GraphLatentSpace(node_dim=5, edge_dim=4, dtype=jnp.float32)
    embedder = GraphEmbedder(
        space=space,
        atom_embed_dim=4,
        hybrid_embed_dim=3,
        cont_embed_dim=2,
        edge_embed_dim=3,
    )
    denoiser = MPNNDenoiser(
        space=space,
        mess_dim=6,
        time_dim=8,
    )
    schedule = cosine_beta_schedule(timesteps=10)
    return GraphDiffusionModel(embedder=embedder, denoiser=denoiser, schedule=schedule)


def _make_state(model, params):
    return DiffusionTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.identity(),
        model=model,
    )


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
    assert outputs["eps_pred"].node.shape == batch.cont.shape[:2] + (model.embedder.space.node_dim,)
    assert outputs["eps_pred"].edge.shape == batch.pair_mask.shape + (model.embedder.space.edge_dim,)


def test_sample_masks_and_shapes_reproducible():
    batch = _tiny_batch()
    model = _tiny_model()
    rngs = {"params": jax.random.PRNGKey(0), "noise": jax.random.PRNGKey(1)}
    variables = model.init(rngs, batch, jnp.array([1, 2], dtype=jnp.int32))

    updater = DDPMUpdater(model.schedule)
    sampler = GraphSampler(space=model.embedder.space, state=_make_state(model, variables["params"]), updater=updater)
    base_rng = jax.random.PRNGKey(42)
    out1 = sampler.sample(
        base_rng,
        n_steps=3,
        batch_size=batch.atom_type.shape[0],
        n_atoms=batch.atom_type.shape[1],
        node_mask=batch.node_mask,
        pair_mask=batch.pair_mask,
    )
    out2 = sampler.sample(
        base_rng,
        n_steps=3,
        batch_size=batch.atom_type.shape[0],
        n_atoms=batch.atom_type.shape[1],
        node_mask=batch.node_mask,
        pair_mask=batch.pair_mask,
    )

    assert jnp.allclose(out1.node, out2.node)
    assert jnp.allclose(out1.edge, out2.edge)
    # Masked regions should stay zero
    assert jnp.allclose(out1.node * (1.0 - batch.node_mask)[..., None], 0.0)
    assert jnp.allclose(out1.edge * (1.0 - batch.pair_mask)[..., None], 0.0)
