"""Featurization helpers for molecular graphs (QM9-style, categorical only)."""

from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy as np
from numpy.typing import DTypeLike
from rdkit import Chem

from mgd.dataset.chemistry import ChemistrySpec, DEFAULT_CHEMISTRY
from mgd.dataset.utils import GraphBatch


def featurize_components(
    mol: Chem.Mol,
    dtype: DTypeLike = "float32",
    spec: ChemistrySpec = DEFAULT_CHEMISTRY,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return categorical ids, edge types, and masks for a single molecule."""
    if spec.explicit_h:
        mol = Chem.AddHs(mol, addCoords=False)
    else:
        # In implicit-H mode we want a strict heavy-atom graph.
        # RemoveHs can keep special hydrogens; RemoveAllHs enforces full stripping.
        mol = Chem.RemoveAllHs(mol, sanitize=True)

    n_atoms = mol.GetNumAtoms()
    if n_atoms > spec.max_nodes:
        raise ValueError(f"Molecule has {n_atoms} atoms, exceeds max_nodes={spec.max_nodes}")

    atom_to_id = spec.atom_to_id
    bond_to_id = spec.bond_to_id
    atom_ids = np.zeros((spec.max_nodes,), dtype=np.int32)
    bond_types = np.zeros((spec.max_nodes, spec.max_nodes), dtype=np.int32)
    node_mask = np.zeros((spec.max_nodes,), dtype=dtype)

    for i, atom in enumerate(mol.GetAtoms()):
        symbol = atom.GetSymbol()
        atom_ids[i] = atom_to_id.get(symbol, 0)
        node_mask[i] = 1.0

    for i in range(n_atoms):
        for j in range(n_atoms):
            bond = None if i == j else mol.GetBondBetweenAtoms(i, j)
            bond_type = bond.GetBondType() if bond is not None else "no_bond"
            bond_types[i, j] = bond_to_id.get(bond_type, 0)

    pair_mask = node_mask[:, None] * node_mask[None, :]
    pair_mask = pair_mask * (1.0 - np.eye(spec.max_nodes, dtype=dtype))
    return atom_ids, bond_types, node_mask, pair_mask


def encode_molecule(
    mol: Chem.Mol,
    *,
    dtype: DTypeLike = "float32",
    as_batch: bool = False,
    spec: ChemistrySpec = DEFAULT_CHEMISTRY,
) -> Union[Dict[str, np.ndarray], GraphBatch]:
    """Encode an RDKit molecule into dense node/edge features (categorical only)."""
    atom_ids, bond_types, node_mask, pair_mask = featurize_components(mol, dtype=dtype, spec=spec)

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
