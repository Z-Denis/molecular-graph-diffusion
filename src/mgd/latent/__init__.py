"""Latent space containers and utilities."""

from .space import (
    AbstractLatentSpace,
    GraphLatent,
    GraphLatentSpace,
    latent_from_scalar,
)
from .utils import center_logits

__all__ = [
    "AbstractLatentSpace",
    "GraphLatentSpace",
    "GraphLatent",
    "latent_from_scalar",
    "center_logits",
]
