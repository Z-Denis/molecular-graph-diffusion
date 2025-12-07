"""Flax-based molecular graph diffusion denoiser."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from flax import linen as nn

from ..dataset.encoding import ATOM_VOCAB_SIZE, HYBRID_VOCAB_SIZE, BOND_VOCAB_SIZE
from ..dataset.utils import GraphBatch
from .backbone import MPNNBackbone


class MPNNDenoiser(nn.Module):
    atom_dim: int 
    hybrid_dim: int
    cont_dim: int

    node_dim: int   # hidden_dim for nodes
    edge_dim: int   # hidden_dim for edges
    mess_dim: int   # message hidden dim
    time_dim: int   # time hidden dim

    atom_vocab_dim: int = ATOM_VOCAB_SIZE
    hybrid_vocab_dim: int = HYBRID_VOCAB_SIZE
    bond_vocab_dim: int = BOND_VOCAB_SIZE

    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    n_layers: int = 1
    param_dtype: DTypeLike = "float32"

    @nn.compact
    def __call__(self, graph: GraphBatch, time: jnp.ndarray):
        backbone = MPNNBackbone(
            self.atom_dim, self.hybrid_dim, self.cont_dim,
            self.node_dim, self.edge_dim, self.mess_dim, self.time_dim,
            atom_vocab_dim=self.atom_vocab_dim,
            hybrid_vocab_dim=self.hybrid_vocab_dim,
            bond_vocab_dim=self.bond_vocab_dim,
            activation=self.activation,
            param_dtype=self.param_dtype,
            n_layers=self.n_layers,
            name='backbone',
        )

        # Evaluate heads
        nodes, edges = backbone(graph, time)

        # Latent to noise space
        eps_nodes = nn.Dense(
            self.node_dim, param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(1e-6),
            name="node_head",
        )(nodes)
        eps_edges = nn.Dense(
            self.edge_dim, param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(1e-6),
            name="edge_head",
        )(edges)

        # Mask out invalid atomic positions
        node_mask = jax.lax.stop_gradient(graph.node_mask)
        pair_mask = jax.lax.stop_gradient(graph.pair_mask)
        eps_nodes = eps_nodes * node_mask[..., None]
        eps_edges = eps_edges * pair_mask[..., None]

        return eps_nodes, eps_edges

__all__ = ["MPNNDenoiser"]
