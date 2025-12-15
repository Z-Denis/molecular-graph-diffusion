"""Lightweight JAX data loader for QM9 tensors."""

from __future__ import annotations

from typing import Dict, Iterator, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from mgd.dataset.utils import GraphBatch


class GraphBatchLoader:
    """Iterates over preprocessed QM9 arrays with JAX-friendly batches.

    Args:
        data: Mapping of array name -> np.ndarray/jnp.ndarray, all with leading batch dim.
        indices: 1D array of indices for this split.
        batch_size: Number of samples per batch.
        key: PRNGKey for shuffling.
        shuffle: Whether to shuffle each pass.
        stop_gradient: If True, stop gradients on batch tensors.
        n_batches: Optional finite number of batches to yield; None for infinite stream.

    Example:
        >>> import jax, numpy as np
        >>> splits = dict(np.load("data/processed/qm9_splits.npz"))
        >>> data = dict(np.load("data/processed/qm9_dense.npz"))
        >>> loader = GraphBatchLoader(data, indices=splits["train"], batch_size=64, key=jax.random.PRNGKey(0))
        >>> batch = next(iter(loader))
        >>> batch.node_mask.shape
        (64, 29)
    """

    def __init__(
        self,
        data: Dict[str, np.ndarray],
        indices: np.ndarray,
        batch_size: int,
        key: jax.Array,
        shuffle: bool = True,
        stop_gradient: bool = True,
        n_batches: int | None = None,
    ) -> None:
        required = [
            "atom_ids",
            "hybrid_ids",
            "node_continuous",
            "bond_types",
            "dknn",
            "node_mask",
            "pair_mask",
        ]
        missing = [k for k in required if k not in data]
        if missing:
            raise KeyError(f"Missing required keys in data: {missing}. Provided keys: {list(data.keys())}")
        self.data = {k: jnp.asarray(v) for k, v in data.items()}
        self.indices = jnp.asarray(indices)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self._key = key
        self._stop_gradient = stop_gradient
        self._n_batches = n_batches
        
        for name, arr in self.data.items():
            if arr.shape[0] <= int(self.indices.max()):
                raise ValueError(
                    f"Array '{name}' has leading dim {arr.shape[0]} but indices include up to {int(self.indices.max())}"
                )
        

    def __len__(self) -> int:
        """Approximate number of batches per pass (or the fixed limit if set)."""
        if self._n_batches is not None:
            return self._n_batches
        n = self.indices.shape[0]
        return (n + self.batch_size - 1) // self.batch_size

    def _ordered_indices(self) -> jnp.ndarray:
        if not self.shuffle:
            return self.indices
        self._key, sub = jax.random.split(self._key)
        order = jax.random.permutation(sub, self.indices.shape[0])
        return self.indices[order]

    def __iter__(self) -> Iterator[GraphBatch]:
        """Yield batches indefinitely, reshuffling indices each pass if enabled."""
        yielded = 0
        while self._n_batches is None or yielded < self._n_batches:
            idx = self._ordered_indices()
            n = idx.shape[0]
            for start in range(0, n, self.batch_size):
                if self._n_batches is not None and yielded >= self._n_batches:
                    break
                end = start + self.batch_size
                batch_idx = idx[start:end]
                batch = {name: arr[batch_idx] for name, arr in self.data.items()}
                graph = _to_graph_batch(batch)
                if self._stop_gradient:
                    graph = graph.stop_gradient()
                yield graph
                yielded += 1


def _to_graph_batch(batch: Dict[str, jnp.ndarray]) -> GraphBatch:
    graph = GraphBatch(
        atom_type=batch["atom_ids"],
        hybrid=batch["hybrid_ids"],
        cont=batch["node_continuous"],
        bond_type=batch["bond_types"],
        dknn=batch["dknn"],
        node_mask=batch["node_mask"],
        pair_mask=batch["pair_mask"],
    )
    return graph


__all__ = ["GraphBatchLoader"]
