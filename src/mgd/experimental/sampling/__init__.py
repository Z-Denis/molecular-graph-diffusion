"""Experimental sampling package (legacy + prototypes)."""

from .guidance import (
    LogitGuidanceConfig,
    make_logit_guidance,
    valence_over_penalty,
    degree_mse_penalty,
    aromatic_coherence_penalty,
)

__all__ = [
    "LogitGuidanceConfig",
    "make_logit_guidance",
    "valence_over_penalty",
    "degree_mse_penalty",
    "aromatic_coherence_penalty",
]
