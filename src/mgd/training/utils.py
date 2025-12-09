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


__all__ = ["compute_class_weights"]
