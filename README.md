# Molecular Graph Diffusion (WIP)

This JAX/Flax (linen) project implements a diffusion-based generative framework for molecular graphs, with a focus on the QM9 dataset. It supports two complementary diffusion modes: (i) logit-level diffusion, where continuous-time diffusion is applied directly to node and edge logits as a relaxation of discrete molecular graphs, and (ii) latent-level diffusion, where diffusion operates in the latent space of a graph autoencoder. Both modes are built on a unified Elucidated Diffusion Model (EDM) backbone and share common sampling, guidance, and evaluation infrastructure. The project emphasises careful treatment of categorical variables, entropy and scale control, and inference-time guidance for enforcing soft chemical constraints. 

The codebase is a research sandbox to investigate the prospects of continuous relaxation of discrete graphs and study how representation choice (logits vs latents) affects stability, validity, and controllability in molecular diffusion models.

## Related work

This approach combines ideas from continuous-time diffusion models (EDM), discrete graph diffusion, and molecular generative modeling. At the logit level, our method can be interpreted as a continuous relaxation of discrete graph diffusion approaches such as DiGress, replacing categorical transitions with Gaussian diffusion in logit space. Compared to discrete diffusion, this offers greater flexibility for energy-based guidance but requires careful control of entropy, scale, and gauge symmetries.

We also explore latent diffusion via graph autoencoders, following the general philosophy of latent diffusion models, but find that the absence of a canonical Euclidean geometry in graph latents introduces additional challenges compared to image domains.

## Setup
Create a virtualenv, install dependencies, and install the package in editable mode:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Logit-level diffusion

First process molecules from the QM9 dataset into dense adjacency format and create training-test splits:
```bash
python3 scripts/preprocess_qm9.py --input data/raw/gdb9.sdf --output data/processed/qm9_dense.npz --dtype float32 --feature_style separate
python3 scripts/create_splits.py --num_samples 131970 --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --output data/processed/qm9_splits.npz --seed 42
```

This showcases the interface to train a diffusion model operating at logit level:
```python
import jax, numpy as np
from mgd.dataset import GraphBatchLoader
from mgd.model import OneHotGraphEmbedder
from mgd.latent import GraphLatentSpace
from mgd.model import OneHotAutoencoder
from mgd.training import OneHotLogitDiffusionSpace
from mgd.dataset.qm9 import BOND_VOCAB_SIZE, ATOM_VOCAB_SIZE

# Data
batch_size = 64
splits = dict(np.load("../data/processed/qm9_splits.npz"))
data = dict(np.load("../data/processed/qm9_dense.npz"))
train_loader = GraphBatchLoader(data, indices=splits["train"], batch_size=batch_size, key=jax.random.PRNGKey(0))

# Model
space = GraphLatentSpace(node_dim=ATOM_VOCAB_SIZE, edge_dim=BOND_VOCAB_SIZE)
onehot = OneHotGraphEmbedder(space=space)
ae_model = OneHotAutoencoder(embedder=onehot)

# Diffusion space: choose logit latents
diff_space = OneHotLogitDiffusionSpace(space=space)
# Init encoder params and a sample latent
rng = jax.random.PRNGKey(0)
batch = next(iter(train_loader))
enc_vars = onehot.init(rng, batch, batch.node_mask, batch.pair_mask)
sample_latent = onehot.apply(enc_vars, batch, batch.node_mask, batch.pair_mask)
# Second atom of the first molecule in the minibatch
enc_vars, sample_latent.node[0, 1]
```
```python-repl
({}, Array([0., 0., 0., 1., 0., 0.], dtype=float32))
```

```python
from mgd.model import MPNNDenoiser
from mgd.model import GraphDiffusionModel
from mgd.training import create_train_state
from mgd.training import train_loop
from mgd.diffusion import sample_sigma_mixture
from mgd.utils import Logger

from pathlib import Path
from functools import partial
import optax

def make_decoder_lr_schedule():
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=1e-3,
        warmup_steps=1_000,
        decay_steps=5_000,
        end_value=1e-4,
    )

# Hyperparameters
lr = 5e-4
mess_dim = 256 // 2
time_dim = 256 // 2
n_layers = 4
rho = 7.0
num_steps = 40

# Sigma scales from latent stats (masked RMS from the AE)
sigma_data_node = 1.0#float(jnp.linalg.norm(ae_state.latent_std["node"]) / jnp.sqrt(ae_state.latent_std["node"].size))
sigma_data_edge = 1.0#float(jnp.linalg.norm(ae_state.latent_std["edge"]) / jnp.sqrt(ae_state.latent_std["edge"].size))
sigma_max = 8.0 * max(sigma_data_node, sigma_data_edge)
sigma_min = 0.005 * max(sigma_data_node, sigma_data_edge)

# Build model
denoiser = MPNNDenoiser(
    space=space,
    mess_dim=mess_dim,
    time_dim=time_dim,   # takes log(sigma)
    n_layers=n_layers,
)
diff_model = GraphDiffusionModel(
    denoiser=denoiser,
    sigma_data_node=sigma_data_node,
    sigma_data_edge=sigma_data_edge,
    sigma_min=sigma_min,
    sigma_max=sigma_max,
)

# Init train state
rng = jax.random.PRNGKey(0)
batch = next(iter(train_loader))
enc_vars = onehot.init(rng, batch, batch.node_mask, batch.pair_mask)
enc_params = enc_vars.get("params", {})  # empty FrozenDict
init_latent = onehot.apply(enc_vars, batch, batch.node_mask, batch.pair_mask)

tx = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=make_decoder_lr_schedule()),
)

# Add additional sampling weight at low sigma
sigma_sampler = partial(sample_sigma_mixture, p_low=0.3, k=3.0)

diff_state = create_train_state(
    diff_model,
    init_latent,
    batch.node_mask,
    batch.pair_mask,
    tx,
    rng,
    space=diff_space,
    sigma_sampler=sigma_sampler,
)

# Logger/checkpoint setup
ckpt_dir = Path(f"../checkpoints/diffusion_edm_smin0p01_{n_layers}layer_onehot").resolve()
logger = Logger(log_every=10, ckpt_dir=ckpt_dir, ckpt_every=1000)

# Train
diff_state, history = train_loop(
    diff_state,
    train_loader,
    n_steps=3_000,
    rng=jax.random.PRNGKey(1),
    logger=logger,
)

# import matplotlib.pyplot as plt

# plt.plot([h['loss'] for h in logger.data])
# plt.plot([h['edge_loss'] for h in logger.data])
# plt.plot([h['node_loss'] for h in logger.data])
# plt.ylim(0, 5)

print("Last metrics:", history[-1] if history else logger.data[-1])
```
```python-repl
100%|██████████| 3000/3000 [11:31<00:00,  4.34it/s, edge_loss=0.4117, loss=1.0517, node_loss=0.6401, sigma_mean=0.4541]
Last metrics: {'edge_loss': Array(0.46702132, dtype=float32), 'loss': Array(1.1933677, dtype=float32), 'node_loss': Array(0.72634643, dtype=float32), 'sigma_mean': Array(0.6563614, dtype=float32)}
```

Once the diffusion model is trained, sampling is performed as follows:
```python
from mgd.dataset.qm9 import MAX_NODES
from mgd.training import compute_class_weights, compute_occupation_log_weights, mask_logits
from mgd.sampling import LatentSampler, HeunUpdater, LogitGuidanceConfig, make_logit_guidance
from mgd.sampling.sampler import _prepare_masks

# Config
batch_size = 1024
max_atoms = MAX_NODES
rho = 7.0
num_steps = 40
rng = jax.random.PRNGKey(0)

# Generate random molecule sizes (n_atoms) for each element of the batch with the
# same statistics as the dataset
occup_log_weights = compute_occupation_log_weights(train_loader, MAX_NODES, max_batches=1024)
rng, rng_size = jax.random.split(rng)
n_atoms = jax.random.categorical(rng_size, occup_log_weights, shape=(batch_size,))
_, _, node_mask, pair_mask = _prepare_masks(n_atoms, batch_size, space.dtype, None, None, max_atoms=MAX_NODES)

# Build sigma schedule
sigma_sched = make_sigma_schedule(
    diff_state.model.sigma_min,
    diff_state.model.sigma_max,
    rho=rho,
    num_steps=num_steps,
)

# Prepare sampler
space = diff_space.space  # latent space used by diffusion space
sampler = LatentSampler(space=space, state=diff_state, updater=HeunUpdater())

# Sample atom types for guidance (from empirical distribution)
_, atom_counts = compute_class_weights(
    train_loader,
    ATOM_VOCAB_SIZE,
    pad_value=0,
    return_counts=True,
)
atom_probs = jnp.asarray(atom_counts / atom_counts.sum())

# Setup soft guidance
guidance_cfg = LogitGuidanceConfig(valence_weight=2.0, aromatic_weight=0.1)
guidance_fn = make_logit_guidance(
    guidance_cfg,
    weight_fn = lambda s: 2.0 * (1.0 - s / diff_state.model.sigma_max),
)

# Sample latents
rng, sample_rng = jax.random.split(rng)
latents = sampler.sample(
    sample_rng,
    sigma_schedule=sigma_sched,
    batch_size=batch_size,
    n_atoms=n_atoms,
    node_mask=node_mask,
    pair_mask=pair_mask,
    max_atoms=max_atoms,
    guidance_fn=guidance_fn,
)

# Symmetrise edges
latents_edge = 0.5 * (latents.edge + latents.edge.swapaxes(-2, -3))

# Convert logits to discrete predictions
recon = {"node": latents.node, "edge": latents_edge}
# recon["node"]: (B, N, n_atom_types), recon["edge"]: (B, N, N, 1+n_bond_types)
atom_logits = mask_logits(recon["node"], node_mask, 0)
atom_pred = jnp.argmax(atom_logits, axis=-1)            # (B, N)
bond_pred = jnp.argmax(recon["edge"], axis=-1)        # class 0 == no bond
bond_pred = bond_pred * pair_mask.astype(bond_pred.dtype)

print("atom_pred shape:", atom_pred.shape)
print("bond_pred shape:", bond_pred.shape)
```
```python-repl
atom_pred shape: (1024, 29)
bond_pred shape: (1024, 29, 29)
```

## Latent-level diffusion

This involves an extra step, where an autoencoder is trained as to produce smooth latents in a space of desired dimension. The decoder decodes these latents into a set of logits. In particular, for bonds, we predict one extra logit that determines the probability of existence of a bond of any type between any two atoms; this masks the bond-type predictions made from the usual logits.

This procedure generically leads to the collapse of the latent space into a sharp low-dimensional manifold that breaks diffusion. To mitigate it, the autoencoder is trained under strong noise injection at the latent level. In practice, the magnitude of the noise is chosen as large as possible while ensuring that the trained denoising autoencoder is able to achieve perfect F1 scores for both atom and bond types. Early stopping is also enforced with the same condition.