"""Featurization helpers for molecular graphs (QM9-style, categorical only)."""

from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy as np
from numpy.typing import DTypeLike
from rdkit import Chem

from mgd.dataset.utils import GraphBatch
from mgd.dataset.qm9 import (
    MAX_NODES,
    ATOM_TO_ID,
    BOND_TO_ID,
)


def featurize_components(
    mol: Chem.Mol,
    dtype: DTypeLike = "float32",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return categorical ids, edge types, and masks for a single molecule."""
    mol = Chem.AddHs(mol, addCoords=False)
    n_atoms = mol.GetNumAtoms()
    if n_atoms > MAX_NODES:
        raise ValueError(f"Molecule has {n_atoms} atoms, exceeds MAX_NODES={MAX_NODES}")

    atom_ids = np.zeros((MAX_NODES,), dtype=np.int32)
    bond_types = np.zeros((MAX_NODES, MAX_NODES), dtype=np.int32)
    node_mask = np.zeros((MAX_NODES,), dtype=dtype)

    for i, atom in enumerate(mol.GetAtoms()):
        symbol = atom.GetSymbol()
        atom_ids[i] = ATOM_TO_ID.get(symbol, 0)
        node_mask[i] = 1.0

    for i in range(n_atoms):
        for j in range(n_atoms):
            bond = None if i == j else mol.GetBondBetweenAtoms(i, j)
            bond_type = bond.GetBondType() if bond is not None else "no_bond"
            bond_types[i, j] = BOND_TO_ID.get(bond_type, 0)

    pair_mask = node_mask[:, None] * node_mask[None, :]
    pair_mask = pair_mask * (1.0 - np.eye(MAX_NODES, dtype=dtype))
    return atom_ids, bond_types, node_mask, pair_mask


def encode_molecule(
    mol: Chem.Mol,
    *,
    dtype: DTypeLike = "float32",
    as_batch: bool = False,
) -> Union[Dict[str, np.ndarray], GraphBatch]:
    """Encode an RDKit molecule into dense node/edge features (categorical only)."""
    atom_ids, bond_types, node_mask, pair_mask = featurize_components(mol, dtype=dtype)

    feats = {
        "atom_type": atom_ids.astype(np.int32),
        "bond_type": bond_types.astype(np.int32),
        "node_mask": node_mask.astype(dtype),
        "pair_mask": pair_mask.astype(dtype),
    }
    if as_batch:
        return GraphBatch(
            atom_type=feats["atom_type"],
            bond_type=feats["bond_type"],
            node_mask=feats["node_mask"],
            pair_mask=feats["pair_mask"],
        )
    return feats


__all__ = [
    "featurize_components",
    "encode_molecule",
]
