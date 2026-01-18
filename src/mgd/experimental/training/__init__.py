"""Experimental training package (legacy + prototypes)."""

from .autoencoder import (
    AutoencoderTrainState,
    create_autoencoder_state,
    autoencoder_train_loop,
)
from .losses import (
    masked_mse,
    edm_masked_mse,
    bond_reconstruction_loss,
    aromatic_coherence_penalty,
)
from .space import (
    LatentDiffusionSpace,
    OneHotLogitDiffusionSpace,
    CategoricalDiffusionSpace,
)

__all__ = [
    "AutoencoderTrainState",
    "create_autoencoder_state",
    "autoencoder_train_loop",
    "masked_mse",
    "edm_masked_mse",
    "bond_reconstruction_loss",
    "aromatic_coherence_penalty",
    "LatentDiffusionSpace",
    "OneHotLogitDiffusionSpace",
    "CategoricalDiffusionSpace",
]
