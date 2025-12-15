"""Utilities for training helpers (e.g., class weights)."""

from __future__ import annotations

from typing import Callable, Iterable, Optional
import itertools

import numpy as np

from mgd.dataset.utils import GraphBatch


def compute_class_weights(
    loader: Iterable[GraphBatch],
    n_classes: int,
    *,
    pad_value: int | None = None,
    label_getter: Callable[[GraphBatch], np.ndarray] = lambda batch: batch.atom_type,
    mask_getter: Callable[[GraphBatch], np.ndarray] = lambda batch: batch.node_mask,
    scheme: str = "sqrt_inv",
    eps: float = 1e-8,
    max_batches: Optional[int] = None,
    return_counts: bool = False,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Estimate class weights from a masked loader and return raw counts.

    Args:
        loader: iterable of GraphBatch.
        n_classes: total number of classes (including padding).
        pad_value: label used for padding/unknown that should not drive weights.
        label_getter: function extracting integer labels from a batch.
        mask_getter: function extracting a mask aligned with labels (1=valid).
        scheme: "inv" (1/freq) or "sqrt_inv" (1/sqrt(freq)).
        eps: small constant to avoid division by zero.
        max_batches: optional cap on number of batches to consume (useful for streaming loaders).

    Returns:
        weights: np.ndarray of shape (n_classes,) with mean-normalized weights.
        counts: np.ndarray of shape (n_classes,) with raw counts.
    """
    counts = np.zeros((n_classes,), dtype=np.float64)

    if max_batches is None:
        try:
            max_batches = len(loader)
        except TypeError:
            max_batches = None

    iterator = itertools.islice(loader, max_batches) if max_batches is not None else loader

    for batch in iterator:
        labels = np.asarray(label_getter(batch))
        mask = np.asarray(mask_getter(batch)) > 0.5
        valid_labels = labels[mask]
        if valid_labels.size == 0:
            continue
        counts += np.bincount(valid_labels, minlength=n_classes)

    if counts.sum() == 0:
        raise ValueError("No valid labels found when computing class weights.")

    if pad_value is not None and 0 <= pad_value < n_classes:
        counts[pad_value] = 0.0

    if scheme == "inv":
        weights = 1.0 / (counts + eps)
    elif scheme == "sqrt_inv":
        weights = 1.0 / np.sqrt(counts + eps)
    else:
        raise ValueError(f"Unknown scheme '{scheme}', expected 'inv' or 'sqrt_inv'.")
    weights[counts == 0] = 0
    positive = weights[counts > 0]
    weights = weights / positive.mean()
    if return_counts:
        return weights, counts
    return weights


def compute_occupation_log_weights(
    loader: Iterable[GraphBatch],
    n_atom_max: int,
    *,
    max_batches: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute log-weights over molecule sizes (number of atoms) from a loader.

    Returns:
        log_weights: np.ndarray of log(count / total) of length n_atom_max+1
    """
    occupation_counts = np.zeros(n_atom_max + 1, dtype=np.int64)

    if max_batches is None:
        try:
            max_batches = len(loader)
        except TypeError:
            max_batches = None

    iterator = itertools.islice(loader, max_batches) if max_batches is not None else loader

    for batch in iterator:
        # node_mask has shape (batch, n_atoms); sum gives actual atom count
        occupation = np.array(batch.node_mask.astype(int)).sum(axis=-1)
        max_occ = occupation.max(initial=0)
        if max_occ > n_atom_max:
            raise ValueError(
                f"Observed occupation {max_occ} exceeds n_atom_max={n_atom_max}. "
                "Increase n_atom_max to cover your data."
            )
        occupation_counts += np.bincount(occupation, minlength=n_atom_max + 1)

    total = occupation_counts.sum()
    if total == 0:
        raise ValueError("No valid samples found when computing occupation weights.")
    log_weights = np.log(occupation_counts + 1e-12) - np.log(float(total))
    return log_weights


def mask_logits(
    logits: np.ndarray,
    mask: np.ndarray,
    pad_class: int | None = 0,
) -> np.ndarray:
    """Mask logits with optional padding class handling.

    - If pad_class is None: all classes are set to -inf where mask == 0.
    - Otherwise: logits for ``pad_class`` are always set to -inf, and all classes
      are set to -inf where mask == 0.
    """
    masked = np.array(logits, copy=True)
    valid = np.array(mask, copy=False).astype(bool)
    n_classes = logits.shape[-1]

    if pad_class is None:
        invalid_idx = ~valid
        masked = np.where(
            np.broadcast_to(invalid_idx[..., None], masked.shape),
            -np.inf,
            masked,
        )
        return masked

    if pad_class < 0 or pad_class >= n_classes:
        raise ValueError(f"pad_class {pad_class} out of range for logits dim {n_classes}")

    expand_shape = valid.shape + (n_classes,)
    # Always block pad_class
    pad_mask = np.broadcast_to(np.arange(n_classes) == pad_class, expand_shape)
    masked = np.where(pad_mask, -np.inf, masked)
    # Mask all classes where invalid
    invalid_idx = ~valid
    masked = np.where(
        np.broadcast_to(invalid_idx[..., None], expand_shape),
        -np.inf,
        masked,
    )
    return masked


__all__ = ["compute_class_weights", "compute_occupation_log_weights", "mask_logits"]
