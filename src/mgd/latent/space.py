"""Latent space utilities and containers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import flax
import jax
import jax.numpy as jnp


@flax.struct.dataclass
class GraphLatent:
    """Container for node/edge latent tensors with basic arithmetic support."""

    node: jnp.ndarray
    edge: jnp.ndarray
    __array_priority__ = 1000

    def masked(self, node_mask, pair_mask):
        return GraphLatent(
            self.node * node_mask[..., None],
            self.edge * pair_mask[..., None],
        )

    def __add__(self, other: Any) -> "GraphLatent":
        if isinstance(other, GraphLatent):
            return type(self)(jnp.add(self.node, other.node), jnp.add(self.edge, other.edge))
        return type(self)(jnp.add(self.node, other), jnp.add(self.edge, other))

    def __radd__(self, other: Any) -> "GraphLatent":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "GraphLatent":
        if isinstance(other, GraphLatent):
            return type(self)(jnp.subtract(self.node, other.node), jnp.subtract(self.edge, other.edge))
        return type(self)(jnp.subtract(self.node, other), jnp.subtract(self.edge, other))

    def __rsub__(self, other: Any) -> "GraphLatent":
        if isinstance(other, GraphLatent):
            return type(self)(jnp.subtract(other.node, self.node), jnp.subtract(other.edge, self.edge))
        return type(self)(jnp.subtract(other, self.node), jnp.subtract(other, self.edge))

    def __mul__(self, other: Any) -> "GraphLatent":
        if isinstance(other, GraphLatent):
            return type(self)(jnp.multiply(self.node, other.node), jnp.multiply(self.edge, other.edge))
        return type(self)(jnp.multiply(self.node, other), jnp.multiply(self.edge, other))

    def __rmul__(self, other: Any) -> "GraphLatent":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "GraphLatent":
        if isinstance(other, GraphLatent):
            return type(self)(jnp.divide(self.node, other.node), jnp.divide(self.edge, other.edge))
        return type(self)(jnp.divide(self.node, other), jnp.divide(self.edge, other))

    def __rtruediv__(self, other: Any) -> "GraphLatent":
        if isinstance(other, GraphLatent):
            return type(self)(jnp.divide(other.node, self.node), jnp.divide(other.edge, self.edge))
        return type(self)(jnp.divide(other, self.node), jnp.divide(other, self.edge))

    def __neg__(self) -> "GraphLatent":
        return type(self)(-self.node, -self.edge)


def latent_from_scalar(value: jnp.ndarray, node_ndim: int = 2, edge_ndim: int = 3) -> GraphLatent:
    """Broadcast a scalar/array to node and edge shapes with trailing ones.

    Examples:
        latent_from_scalar(s)  -> node shape (..., 1, 1), edge shape (..., 1, 1, 1)
        latent_from_scalar(s, node_ndim=3, edge_ndim=4) -> (..., 1,1,1) and (...,1,1,1,1)
    """
    node = jnp.reshape(value, value.shape + (1,) * node_ndim)
    edge = jnp.reshape(value, value.shape + (1,) * edge_ndim)
    return GraphLatent(node=node, edge=edge)


class AbstractLatentSpace(ABC):
    """Abstract interface exposing latent dimensionality and helpers."""

    @property
    @abstractmethod
    def node_dim(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def edge_dim(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def dtype(self):
        raise NotImplementedError

    @abstractmethod
    def zeros_from_masks(self, node_mask: jnp.ndarray, pair_mask: jnp.ndarray) -> GraphLatent:
        raise NotImplementedError

    @abstractmethod
    def noise_from_masks(self, rng: jax.Array, node_mask: jnp.ndarray, pair_mask: jnp.ndarray) -> GraphLatent:
        raise NotImplementedError

    @abstractmethod
    def random_latent(
        self,
        rng: jax.Array,
        batch_size: int,
        n_atoms: int,
        node_mask: jnp.ndarray | None = None,
        pair_mask: jnp.ndarray | None = None,
    ) -> GraphLatent:
        """Sample a latent with optional masks; defaults to full ones if masks are None."""
        raise NotImplementedError


class GraphLatentSpace(AbstractLatentSpace):
    """Concrete latent space definition for graph node/edge latents."""

    def __init__(self, node_dim: int, edge_dim: int, dtype: jnp.dtype = jnp.float32):
        self._node_dim = int(node_dim)
        self._edge_dim = int(edge_dim)
        self._dtype = dtype

    # Abstract property implementations
    @property
    def node_dim(self) -> int:  # type: ignore[override]
        return self._node_dim

    @property
    def edge_dim(self) -> int:  # type: ignore[override]
        return self._edge_dim

    @property
    def dtype(self):  # type: ignore[override]
        return self._dtype

    def zeros_from_masks(self, node_mask: jnp.ndarray, pair_mask: jnp.ndarray) -> GraphLatent:
        node_shape = node_mask.shape + (self.node_dim,)
        edge_shape = pair_mask.shape + (self.edge_dim,)
        return GraphLatent(
            jnp.zeros(node_shape, dtype=self.dtype),
            jnp.zeros(edge_shape, dtype=self.dtype),
        ).masked(node_mask, pair_mask)

    def noise_from_masks(self, rng: jax.Array, node_mask: jnp.ndarray, pair_mask: jnp.ndarray) -> GraphLatent:
        rng_n, rng_e = jax.random.split(rng)
        node_shape = node_mask.shape + (self.node_dim,)
        edge_shape = pair_mask.shape + (self.edge_dim,)
        nodes = jax.random.normal(rng_n, node_shape, dtype=self.dtype)
        edges = jax.random.normal(rng_e, edge_shape, dtype=self.dtype)
        edges = (edges + edges.swapaxes(-2, -3)) / jnp.sqrt(2.0)
        n = edges.shape[-2]
        edges = edges * (1.0 - jnp.eye(n, dtype=edges.dtype)[..., None])
        return GraphLatent(nodes, edges).masked(node_mask, pair_mask)

    def random_latent(
        self,
        rng: jax.Array,
        batch_size: int,
        n_atoms: int,
        node_mask: jnp.ndarray | None = None,
        pair_mask: jnp.ndarray | None = None,
    ) -> GraphLatent:
        if node_mask is None:
            node_mask = jnp.ones((batch_size, n_atoms), dtype=self.dtype)
        if pair_mask is None:
            pair_mask = node_mask[..., :, None] * node_mask[..., None, :]
        return self.noise_from_masks(rng, node_mask, pair_mask)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(node_dim={self.node_dim}, "
            f"edge_dim={self.edge_dim}, dtype={self.dtype})"
        )


__all__ = [
    "GraphLatent",
    "latent_from_scalar",
    "AbstractLatentSpace",
    "GraphLatentSpace",
]
