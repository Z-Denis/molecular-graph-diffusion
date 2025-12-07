"""Featurization helpers for molecular graphs (QM9-style).

Provides utilities to turn an RDKit molecule into dense node/edge features with
fixed padding. Unknown categorical values are encoded as 0 (pad/unknown).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from numpy.typing import DTypeLike
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

ATOM_VOCAB_SIZE = len(ATOM_TYPES) + 1
HYBRID_VOCAB_SIZE = len(HYBRIDIZATIONS) + 1
BOND_VOCAB_SIZE = max(BOND_TO_ID.values()) + 1


def _one_hot_from_ids(ids: np.ndarray, depth: int, dtype: DTypeLike, zero_as_unknown: bool = True) -> np.ndarray:
    out = np.zeros(ids.shape + (depth,), dtype=dtype)
    flat_idx = ids.reshape(-1)
    mask = flat_idx > 0 if zero_as_unknown else flat_idx >= 0
    rows = np.nonzero(mask)[0]
    cols = flat_idx[mask]
    out.reshape(-1, depth)[rows, cols] = 1.0
    return out


def featurize_components(
    mol: Chem.Mol, dtype: DTypeLike = "float32"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return categorical ids, continuous scalars, and masks for a single molecule."""
    mol = Chem.AddHs(mol, addCoords=False)
    n_atoms = mol.GetNumAtoms()
    if n_atoms > MAX_NODES:
        raise ValueError(f"Molecule has {n_atoms} atoms, exceeds MAX_NODES={MAX_NODES}")

    atom_ids = np.zeros((MAX_NODES,), dtype=np.int32)
    hybrid_ids = np.zeros((MAX_NODES,), dtype=np.int32)
    node_cont = np.zeros((MAX_NODES, 4), dtype=dtype)  # electronegativity, degree/4, formal_valence/4, aromaticity
    edge_types = np.zeros((MAX_NODES, MAX_NODES), dtype=np.int32)
    node_mask = np.zeros((MAX_NODES,), dtype=dtype)
    bond_mask = np.zeros((MAX_NODES, MAX_NODES), dtype=dtype)

    for i, atom in enumerate(mol.GetAtoms()):
        symbol = atom.GetSymbol()
        atom_ids[i] = ATOM_TO_ID.get(symbol, 0)
        hyb = atom.GetHybridization()
        if symbol != "H":
            if hyb in HYBRID_TO_ID:
                hybrid_ids[i] = HYBRID_TO_ID[hyb]
            else:
                print(
                    f"Encountered unsupported hybridization {hyb} for atom {symbol}; "
                    "encoded as zeros."
                )
        node_cont[i, 0] = ELECTRONEGATIVITY.get(symbol, 0.0)
        node_cont[i, 1] = atom.GetDegree() / 4.0
        node_cont[i, 2] = atom.GetFormalCharge() / 4.0
        node_cont[i, 3] = 1.0 if atom.GetIsAromatic() else 0.0
        node_mask[i] = 1.0

    for i in range(n_atoms):
        for j in range(n_atoms):
            bond = None if i == j else mol.GetBondBetweenAtoms(i, j)
            bond_type = bond.GetBondType() if bond is not None else "no_bond"
            edge_types[i, j] = BOND_TO_ID.get(bond_type, 0)
            if bond is not None:
                bond_mask[i, j] = 1.0

    pair_mask = node_mask[:, None] * node_mask[None, :]
    return atom_ids, hybrid_ids, node_cont, edge_types, node_mask, pair_mask, bond_mask


def build_flat_features(
    atom_ids: np.ndarray, hybrid_ids: np.ndarray, node_cont: np.ndarray, dtype: DTypeLike
) -> np.ndarray:
    """Concatenate categorical one-hots with continuous scalars."""
    atom_one_hot = _one_hot_from_ids(atom_ids, ATOM_VOCAB_SIZE, dtype)
    hybrid_one_hot = _one_hot_from_ids(hybrid_ids, HYBRID_VOCAB_SIZE, dtype)
    return np.concatenate([atom_one_hot, hybrid_one_hot, node_cont], axis=-1)


def encode_molecule(
    mol: Chem.Mol, feature_style: str = "flat", dtype: DTypeLike = "float32"
) -> Dict[str, np.ndarray]:
    """Encode an RDKit molecule into dense node/edge features.

    Example:
        >>> mol = Chem.MolFromSmiles("CCO")
        >>> features = encode_molecule(mol, feature_style="flat")
    """
    atom_ids, hybrid_ids, node_cont, edge_types, node_mask, pair_mask, bond_mask = featurize_components(
        mol, dtype=dtype
    )

    if feature_style == "flat":
        nodes = build_flat_features(atom_ids, hybrid_ids, node_cont, dtype=dtype)
        edge_one_hot = _one_hot_from_ids(edge_types, BOND_VOCAB_SIZE, dtype, zero_as_unknown=False)
        return {
            "nodes": nodes.astype(dtype),
            "edges": edge_one_hot.astype(dtype),
            "node_mask": node_mask.astype(dtype),
            "pair_mask": pair_mask.astype(dtype),
            "bond_mask": bond_mask.astype(dtype),
        }

    if feature_style != "separate":
        raise ValueError(f"feature_style must be 'flat' or 'separate', got {feature_style}")

    return {
        "atom_ids": atom_ids.astype(np.int32),
        "hybrid_ids": hybrid_ids.astype(np.int32),
        "node_continuous": node_cont.astype(dtype),
        "edge_types": edge_types.astype(np.int32),
        "node_mask": node_mask.astype(dtype),
        "pair_mask": pair_mask.astype(dtype),
        "bond_mask": bond_mask.astype(dtype),
    }


__all__ = [
    "MAX_NODES",
    "ATOM_TYPES",
    "HYBRIDIZATIONS",
    "ELECTRONEGATIVITY",
    "ATOM_TO_ID",
    "HYBRID_TO_ID",
    "BOND_TO_ID",
    "ATOM_VOCAB_SIZE",
    "HYBRID_VOCAB_SIZE",
    "BOND_VOCAB_SIZE",
    "featurize_components",
    "build_flat_features",
    "encode_molecule",
]
