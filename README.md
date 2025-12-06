# Molecular Graph Diffusion (WIP)

JAX/Flax (linen) project that trains a diffusion model on molecular graphs (e.g., QM9) and supports sampling and evaluation pipelines.

## Layout
- `data/`: raw/processed storage plus QM9 download stub.
- `configs/`: YAML configs for experiments, model, diffusion, and sampling.
- `src/mgd`: package namespace with dataset, diffusion, model, training, sampling, eval, and utility modules (WIP).
- `scripts/`: entrypoints for preprocessing, training, sampling, and evaluation (WIP).
- `notebooks/`: notebooks for data inspection, training demo, and sampling visualization (WIP).

## Setup
Create a virtualenv, install dependencies, and install the package in editable mode:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```
