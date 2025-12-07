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
        shuffle: Whether to shuffle each epoch.
        drop_last: Drop final partial batch if it is smaller than batch_size.

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
        drop_last: bool = False,
    ) -> None:
        required = ["atom_ids", "hybrid_ids", "node_continuous", "edge_types", "node_mask", "pair_mask"]
        missing = [k for k in required if k not in data]
        if missing:
            raise KeyError(f"Missing required keys in data: {missing}. Provided keys: {list(data.keys())}")
        self.data = {k: jnp.asarray(v) for k, v in data.items()}
        self.indices = jnp.asarray(indices)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._key = key
        
        for name, arr in self.data.items():
            if arr.shape[0] <= int(self.indices.max()):
                raise ValueError(
                    f"Array '{name}' has leading dim {arr.shape[0]} but indices include up to {int(self.indices.max())}"
                )
        

    def __len__(self) -> int:
        """Number of batches per epoch (ceil unless drop_last)."""
        n = self.indices.shape[0]
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def _ordered_indices(self) -> jnp.ndarray:
        if not self.shuffle:
            return self.indices
        self._key, sub = jax.random.split(self._key)
        order = jax.random.permutation(sub, self.indices.shape[0])
        return self.indices[order]

    def __iter__(self) -> Iterator[GraphBatch]:
        idx = self._ordered_indices()
        n = idx.shape[0]
        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            if end > n and self.drop_last:
                break
            batch_idx = idx[start:end]
            batch = {name: arr[batch_idx] for name, arr in self.data.items()}
            yield _to_graph_batch(batch)


def _to_graph_batch(batch: Dict[str, jnp.ndarray]) -> GraphBatch:
    graph = GraphBatch(
        atom_type=batch["atom_ids"],
        hybrid=batch["hybrid_ids"],
        cont=batch["node_continuous"],
        edges=batch["edge_types"],
        node_mask=batch["node_mask"],
        pair_mask=batch["pair_mask"],
    )
    return graph


__all__ = ["GraphBatchLoader"]
