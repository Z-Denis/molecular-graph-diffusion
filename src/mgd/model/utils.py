"""Utility modules and functions for graph models."""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence, Tuple, Union

from jax.typing import DTypeLike
import jax.numpy as jnp
from flax import linen as nn

FeaturesArg = Union[int, Sequence[int]]


class MLP(nn.Module):
    """Simple MLP with configurable layers and activation.

    Use either:
        MLP(name='mlp', features=(n1, n2, ...), param_dtype='float32')
    or:
        MLP(name='mlp', features=hidden_size, n_layers=2, param_dtype='float32')
    """

    features: FeaturesArg
    n_layers: Optional[int] = None
    activation: Callable = nn.gelu
    param_dtype: DTypeLike = "float32"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if isinstance(self.features, Iterable) and not isinstance(self.features, int):
            layer_sizes: Tuple[int, ...] = tuple(self.features)
        else:
            if self.n_layers is None:
                raise ValueError("n_layers must be provided when features is an int.")
            layer_sizes = tuple([int(self.features)] * int(self.n_layers))

        for i, size in enumerate(layer_sizes):
            x = nn.Dense(features=size, param_dtype=self.param_dtype, name=f"dense_{i}")(x)
            if i < len(layer_sizes) - 1:
                x = self.activation(x)
        return x


def aggregate_node_edge(
    node_i: Optional[jnp.ndarray] = None,
    node_j: Optional[jnp.ndarray] = None,
    edge_ij: Optional[jnp.ndarray] = None,
    reducer: Optional[Callable[[Sequence[jnp.ndarray]], jnp.ndarray]] = None,
) -> jnp.ndarray:
    """Aggregate broadcasted node/edge inputs to (batch, n_atoms, n_atoms, feat).

    Nodes are broadcast across pair dimensions:
        node_i -> expand dim=-2 then broadcast over j
        node_j -> expand dim=-3 then broadcast over i
    Edge features are assumed already shaped (batch, n_atoms, n_atoms, d_e).

    reducer: function applied to list of broadcasted parts.
             Defaults to concatenation along the last axis.
             Can be a custom callable like `lambda parts: jnp.sum(jnp.stack(parts, axis=-1), axis=-1)`.
    """
    if node_i is None and node_j is None and edge_ij is None:
        raise ValueError("At least one of node_i, node_j, or edge_ij must be provided.")

    # Infer n_atoms
    n_atoms = None
    batch_shape = None
    if edge_ij is not None:
        batch_shape = edge_ij.shape[:-3]
        n_atoms = edge_ij.shape[-3]
    if n_atoms is None and node_i is not None:
        batch_shape = node_i.shape[:-2]
        n_atoms = node_i.shape[-2]
    if n_atoms is None and node_j is not None:
        batch_shape = node_j.shape[:-2]
        n_atoms = node_j.shape[-2]
    if n_atoms is None or batch_shape is None:
        raise ValueError("Could not infer batch or n_atoms from provided inputs.")

    parts = []
    if node_i is not None:
        ni = jnp.expand_dims(node_i, axis=-2)
        ni = jnp.broadcast_to(ni, batch_shape + (n_atoms, n_atoms, ni.shape[-1]))
        parts.append(ni)
    if node_j is not None:
        nj = jnp.expand_dims(node_j, axis=-3)
        nj = jnp.broadcast_to(nj, batch_shape + (n_atoms, n_atoms, nj.shape[-1]))
        parts.append(nj)
    if edge_ij is not None:
        parts.append(edge_ij)
    if reducer is None:
        return jnp.concatenate(parts, axis=-1)
    return reducer(parts)


__all__ = ["MLP", "aggregate_node_edge"]
