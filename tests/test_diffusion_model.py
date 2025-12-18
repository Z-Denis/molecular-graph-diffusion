import jax
import jax.numpy as jnp

import optax

from mgd.dataset.utils import GraphBatch
from mgd.diffusion.schedules import make_sigma_schedule
from mgd.latent import GraphLatentSpace, GraphLatent
from mgd.model.denoiser import MPNNDenoiser
from mgd.model.diffusion_model import GraphDiffusionModel
from mgd.sampling import HeunUpdater, LatentSampler
from mgd.training.train_step import DiffusionTrainState


def _tiny_batch():
    # Two graphs, three nodes (third node is padded)
    atom_type = jnp.array([[1, 2, 0], [0, 1, 0]], dtype=jnp.int32)
    hybrid = jnp.zeros_like(atom_type)
    cont = jnp.zeros((2, 3, 1), dtype=jnp.float32)
    edge_type = jnp.zeros((2, 3, 3), dtype=jnp.int32)
    dknn = jnp.zeros((2, 3, 3, 1), dtype=jnp.float32)
    node_mask = (atom_type > 0).astype(jnp.float32)
    pair_mask = (node_mask[..., :, None] * node_mask[..., None, :]).astype(jnp.float32)
    return GraphBatch(
        atom_type=atom_type,
        hybrid=hybrid,
        cont=cont,
        bond_type=edge_type,
        dknn=dknn,
        node_mask=node_mask,
        pair_mask=pair_mask,
    )


def _tiny_model():
    space = GraphLatentSpace(node_dim=5, edge_dim=4, dtype=jnp.float32)
    denoiser = MPNNDenoiser(
        space=space,
        mess_dim=6,
        time_dim=8,
    )
    model = GraphDiffusionModel(
        denoiser=denoiser,
        sigma_data_node=1.0,
        sigma_data_edge=1.0,
        sigma_min=0.005,
        sigma_max=8.0,
    )
    return space, model


def _make_state(model, params):
    return DiffusionTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.identity(),
        model=model,
    )


def test_training_forward_shapes_and_keys():
    batch = _tiny_batch()
    space, model = _tiny_model()
    rng = {"params": jax.random.PRNGKey(0), "noise": jax.random.PRNGKey(1)}
    latents = GraphLatent(
        jax.random.normal(jax.random.PRNGKey(123), batch.cont.shape[:2] + (space.node_dim,)),
        jax.random.normal(jax.random.PRNGKey(456), batch.pair_mask.shape + (space.edge_dim,)),
    )
    sigma = jnp.array([0.5, 0.7], dtype=jnp.float32)
    variables = model.init(rng, latents, sigma, node_mask=batch.node_mask, pair_mask=batch.pair_mask)
    outputs = model.apply(
        variables,
        latents,
        sigma,
        node_mask=batch.node_mask,
        pair_mask=batch.pair_mask,
        rngs={"noise": jax.random.PRNGKey(2)},
    )
    for key in ("x_hat", "noise", "noisy", "clean"):
        assert key in outputs
        assert isinstance(outputs[key].node, jnp.ndarray)
        assert isinstance(outputs[key].edge, jnp.ndarray)
    assert outputs["x_hat"].node.shape == batch.cont.shape[:2] + (space.node_dim,)
    assert outputs["x_hat"].edge.shape == batch.pair_mask.shape + (space.edge_dim,)


def test_sample_masks_and_shapes_reproducible():
    batch = _tiny_batch()
    space, model = _tiny_model()
    rngs = {"params": jax.random.PRNGKey(0), "noise": jax.random.PRNGKey(1)}
    latents = GraphLatent(
        jax.random.normal(jax.random.PRNGKey(123), batch.cont.shape[:2] + (space.node_dim,)),
        jax.random.normal(jax.random.PRNGKey(456), batch.pair_mask.shape + (space.edge_dim,)),
    )
    sigma = jnp.array([0.5, 0.7], dtype=jnp.float32)
    variables = model.init(
        rngs,
        latents,
        sigma,
        node_mask=batch.node_mask,
        pair_mask=batch.pair_mask,
    )

    updater = HeunUpdater()
    sampler = LatentSampler(space=space, state=_make_state(model, variables["params"]), updater=updater)
    sigma_schedule = make_sigma_schedule(model.sigma_min, model.sigma_max, num_steps=5)
    base_rng = jax.random.PRNGKey(42)
    out1 = sampler.sample(
        base_rng,
        sigma_schedule=sigma_schedule,
        batch_size=batch.atom_type.shape[0],
        n_atoms=batch.atom_type.shape[1],
        node_mask=batch.node_mask,
        pair_mask=batch.pair_mask,
        max_atoms=batch.atom_type.shape[1],
    )
    out2 = sampler.sample(
        base_rng,
        sigma_schedule=sigma_schedule,
        batch_size=batch.atom_type.shape[0],
        n_atoms=batch.atom_type.shape[1],
        node_mask=batch.node_mask,
        pair_mask=batch.pair_mask,
        max_atoms=batch.atom_type.shape[1],
    )

    assert jnp.allclose(out1.node, out2.node)
    assert jnp.allclose(out1.edge, out2.edge)
    # Masked regions should stay zero
    assert jnp.allclose(out1.node * (1.0 - batch.node_mask)[..., None], 0.0)
    assert jnp.allclose(out1.edge * (1.0 - batch.pair_mask)[..., None], 0.0)
