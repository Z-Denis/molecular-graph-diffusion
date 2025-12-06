"""Featurization helpers for molecular graphs (QM9-style).

Exposes utilities to turn an RDKit molecule into dense node/edge features with
fixed padding. Unknown categorical values are encoded as all-zero vectors.
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

BOND_TYPES = {
    "no_bond": 0,
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
    Chem.BondType.AROMATIC: 4,
}


def featurize_components(
    mol: Chem.Mol, dtype: DTypeLike = "float32"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return categorical one-hots, continuous scalars, and masks for a single molecule."""
    mol = Chem.AddHs(mol, addCoords=False)
    n_atoms = mol.GetNumAtoms()
    if n_atoms > MAX_NODES:
        raise ValueError(f"Molecule has {n_atoms} atoms, exceeds MAX_NODES={MAX_NODES}")

    node_cont = np.zeros((MAX_NODES, 4), dtype=dtype)  # electronegativity, degree/4, formal_valence/4, aromaticity
    atom_one_hot = np.zeros((MAX_NODES, len(ATOM_TYPES)), dtype=dtype)
    hybrid_one_hot = np.zeros((MAX_NODES, len(HYBRIDIZATIONS)), dtype=dtype)
    edge_one_hot = np.zeros((MAX_NODES, MAX_NODES, len(BOND_TYPES)), dtype=dtype)
    node_mask = np.zeros((MAX_NODES,), dtype=dtype)
    bond_mask = np.zeros((MAX_NODES, MAX_NODES), dtype=dtype)

    for i, atom in enumerate(mol.GetAtoms()):
        symbol = atom.GetSymbol()
        if symbol in ATOM_TYPES:
            atom_one_hot[i, ATOM_TYPES.index(symbol)] = 1.0
        hyb = atom.GetHybridization()
        if symbol != "H":
            if hyb in HYBRIDIZATIONS:
                hybrid_one_hot[i, HYBRIDIZATIONS.index(hyb)] = 1.0
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
            idx = BOND_TYPES.get(bond_type, 0)
            edge_one_hot[i, j, idx] = 1.0
            if bond is not None:
                bond_mask[i, j] = 1.0

    pair_mask = node_mask[:, None] * node_mask[None, :]
    return atom_one_hot, hybrid_one_hot, node_cont, edge_one_hot, node_mask, pair_mask, bond_mask


def build_flat_features(
    atom_one_hot: np.ndarray, hybrid_one_hot: np.ndarray, node_cont: np.ndarray
) -> np.ndarray:
    """Concatenate categorical one-hots with continuous scalars."""
    return np.concatenate([atom_one_hot, hybrid_one_hot, node_cont], axis=-1)


def encode_molecule(
    mol: Chem.Mol, feature_style: str = "flat", dtype: DTypeLike = "float32"
) -> Dict[str, np.ndarray]:
    """Encode an RDKit molecule into dense node/edge features.

    Example:
        >>> mol = Chem.MolFromSmiles("CCO")
        >>> features = encode_molecule(mol, feature_style="flat")
    """
    atom_oh, hybrid_oh, node_cont, edge_oh, node_mask, pair_mask, bond_mask = featurize_components(mol, dtype=dtype)

    if feature_style == "flat":
        nodes = build_flat_features(atom_oh, hybrid_oh, node_cont)
        return {
            "nodes": nodes.astype(dtype),
            "edges": edge_oh.astype(dtype),
            "node_mask": node_mask.astype(dtype),
            "pair_mask": pair_mask.astype(dtype),
            "bond_mask": bond_mask.astype(dtype),
        }

    if feature_style != "separate":
        raise ValueError(f"feature_style must be 'flat' or 'separate', got {feature_style}")

    return {
        "atom_one_hot": atom_oh.astype(dtype),
        "hybrid_one_hot": hybrid_oh.astype(dtype),
        "node_continuous": node_cont.astype(dtype),
        "edge_one_hot": edge_oh.astype(dtype),
        "node_mask": node_mask.astype(dtype),
        "pair_mask": pair_mask.astype(dtype),
        "bond_mask": bond_mask.astype(dtype),
    }


__all__ = [
    "MAX_NODES",
    "ATOM_TYPES",
    "HYBRIDIZATIONS",
    "BOND_TYPES",
    "featurize_components",
    "build_flat_features",
    "encode_molecule",
]
