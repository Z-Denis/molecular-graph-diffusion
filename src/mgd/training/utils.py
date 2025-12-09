"""Utilities for training helpers (e.g., class weights)."""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

from mgd.dataset.utils import GraphBatch


def compute_class_weights(
    loader: Iterable[GraphBatch],
    num_classes: int,
    *,
    pad_value: int = 0,
    label_getter: Callable[[GraphBatch], np.ndarray] = lambda batch: batch.atom_type,
    mask_getter: Callable[[GraphBatch], np.ndarray] = lambda batch: batch.node_mask,
    scheme: str = "sqrt_inv",
    eps: float = 1e-8,
) -> np.ndarray:
    """Estimate class weights from a masked loader.

    Args:
        loader: iterable of GraphBatch.
        num_classes: total number of classes (including padding).
        pad_value: label used for padding/unknown that should not drive weights.
        label_getter: function extracting integer labels from a batch.
        mask_getter: function extracting a mask aligned with labels (1=valid).
        scheme: "inv" (1/freq) or "sqrt_inv" (1/sqrt(freq)).
        eps: small constant to avoid division by zero.

    Returns:
        np.ndarray of shape (num_classes,) with mean-normalized weights.
    """
    counts = np.zeros((num_classes,), dtype=np.float64)

    for batch in loader:
        labels = np.asarray(label_getter(batch))
        mask = np.asarray(mask_getter(batch)) > 0.5
        valid_labels = labels[mask]
        if valid_labels.size == 0:
            continue
        counts += np.bincount(valid_labels, minlength=num_classes)

    if counts.sum() == 0:
        raise ValueError("No valid labels found when computing class weights.")
    
    if scheme == "inv":
        weights = 1.0 / (counts + eps)
    elif scheme == "sqrt_inv":
        weights = 1.0 / np.sqrt(counts + eps)
    else:
        raise ValueError(f"Unknown scheme '{scheme}', expected 'inv' or 'sqrt_inv'.")
    weights[counts == 0] = 0
    positive = weights[counts > 0]
    weights = weights / positive.mean()
    return weights


def compute_occupation_log_weights(
    loader: Iterable[GraphBatch],
    n_atom_max: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute log-weights over molecule sizes (number of atoms) from a loader.

    Returns:
        log_weights: np.ndarray of log(count / total) of length n_atom_max+1
    """
    occupation_counts = np.zeros(n_atom_max + 1, dtype=np.int64)

    for batch in loader:
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
    pad_class: int = 0,
) -> np.ndarray:
    """Mask logits so padded positions cannot select non-padding classes.

    For positions where mask == 0, logits for all classes except ``pad_class`` are set to -inf.
    Args:
        logits: array of shape (..., num_classes)
        mask: broadcastable to logits[..., 0]; entries 1=valid, 0=pad
        pad_class: index of padding class
    """
    masked = np.array(logits, copy=True)
    valid = np.array(mask, copy=False).astype(bool)
    if pad_class < 0 or pad_class >= logits.shape[-1]:
        raise ValueError(f"pad_class {pad_class} out of range for logits dim {logits.shape[-1]}")
    # Build a mask over classes
    class_mask = np.ones(logits.shape[-1], dtype=bool)
    class_mask[pad_class] = False
    # Apply only where invalid positions; broadcast to classes
    invalid_idx = ~valid
    if invalid_idx.any():
        expand_shape = invalid_idx.shape + (logits.shape[-1],)
        class_mask_full = np.broadcast_to(class_mask, expand_shape)
        masked = np.where(
            np.broadcast_to(invalid_idx[..., None], expand_shape) & class_mask_full,
            -np.inf,
            masked,
        )
    return masked


__all__ = ["compute_class_weights", "compute_occupation_log_weights", "mask_logits"]
