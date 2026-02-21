"""QM9-specific categorical vocabularies and feature tables."""

from __future__ import annotations

import numpy as np
from rdkit import Chem

from .chemistry import ChemistrySpec

QM9_IMPLICIT_H = ChemistrySpec(
    name="qm9_implicit_h",
    max_nodes=9,
    atom_types=("C", "N", "O", "F"),
    bond_to_id={
        "no_bond": 0,
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
    },
    bond_orders=np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
    valence_table=np.array([0.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32),  # pad, C, N, O, F
    allowed_valences=((4,), (3,), (1, 2), (1,)),  # C, N, O, F
    k_caps_by_index={1: 4, 2: 3, 3: 2, 4: 1},
    explicit_h=False,
)

QM9_EXPLICIT_H = ChemistrySpec(
    name="qm9_explicit_h",
    max_nodes=29,
    atom_types=("H", "C", "N", "O", "F"),
    bond_to_id={
        "no_bond": 0,
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
    },
    bond_orders=np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
    valence_table=np.array([0.0, 1.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32),  # pad, H, C, N, O, F
    allowed_valences=((1,), (4,), (3,), (1, 2), (1,)),  # H, C, N, O, F
    k_caps_by_index={1: 1, 2: 4, 3: 3, 4: 2, 5: 1},
    explicit_h=True,
)

MAX_NODES = QM9_IMPLICIT_H.max_nodes
ATOM_TYPES = list(QM9_IMPLICIT_H.atom_types)
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
ATOM_TO_ID = QM9_IMPLICIT_H.atom_to_id
HYBRID_TO_ID = {hyb: i + 1 for i, hyb in enumerate(HYBRIDIZATIONS)}

BOND_TO_ID = dict(QM9_IMPLICIT_H.bond_to_id)

# Valence lookup aligned with ATOM_TO_ID (index 0 is pad/unknown).
VALENCE_TABLE = np.asarray(QM9_IMPLICIT_H.valence_table, dtype=np.float32)

BOND_ORDERS = np.asarray(QM9_IMPLICIT_H.bond_orders, dtype=np.float32)

ATOM_VOCAB_SIZE = QM9_IMPLICIT_H.atom_vocab_size
HYBRID_VOCAB_SIZE = len(HYBRIDIZATIONS) + 1
BOND_VOCAB_SIZE = QM9_IMPLICIT_H.bond_vocab_size

__all__ = [
    "MAX_NODES",
    "QM9_IMPLICIT_H",
    "QM9_EXPLICIT_H",
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
