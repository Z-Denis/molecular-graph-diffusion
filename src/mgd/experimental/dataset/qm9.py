"""QM9-specific categorical vocabularies and feature tables (legacy)."""

from __future__ import annotations

import numpy as np
from rdkit import Chem

MAX_NODES = 29

ATOM_TYPES = ["H", "C", "N", "O", "F"]
HYBRIDIZATIONS = [
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
]

# Pauling electronegativity (approximate)
ELECTRONEGATIVITY = {
    "H": 2.20,
    "C": 2.55,
    "N": 3.04,
    "O": 3.44,
    "F": 3.98,
}

# Reserve 0 for padding/unknown; categories start at 1
ATOM_TO_ID = {sym: i + 1 for i, sym in enumerate(ATOM_TYPES)}
HYBRID_TO_ID = {hyb: i + 1 for i, hyb in enumerate(HYBRIDIZATIONS)}

BOND_TO_ID = {
    "no_bond": 0,
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
    Chem.BondType.AROMATIC: 4,
}

# Valence lookup aligned with ATOM_TO_ID (index 0 is pad/unknown).
VALENCE_TABLE = np.array([0, 1, 4, 3, 2, 1], dtype=np.float32)  # pad, H, C, N, O, F

BOND_ORDERS = np.array(
    [0.0, 1.0, 2.0, 3.0, 1.5], dtype=np.float32
)  # pad/no-bond, single, double, triple, aromatic

ATOM_VOCAB_SIZE = len(ATOM_TYPES) + 1
HYBRID_VOCAB_SIZE = len(HYBRIDIZATIONS) + 1
BOND_VOCAB_SIZE = max(BOND_TO_ID.values()) + 1

__all__ = [
    "MAX_NODES",
    "ATOM_TYPES",
    "HYBRIDIZATIONS",
    "ELECTRONEGATIVITY",
    "ATOM_TO_ID",
    "HYBRID_TO_ID",
    "BOND_TO_ID",
    "VALENCE_TABLE",
    "BOND_ORDERS",
    "ATOM_VOCAB_SIZE",
    "HYBRID_VOCAB_SIZE",
    "BOND_VOCAB_SIZE",
]
