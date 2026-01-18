"""Featurization helpers for molecular graphs (QM9-style).

Provides utilities to turn an RDKit molecule into dense node/edge features with
fixed padding. Unknown categorical values are encoded as 0 (pad/unknown).
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy as np
from numpy.typing import DTypeLike
from rdkit import Chem

from mgd.experimental.dataset.utils import GraphBatch
from mgd.experimental.dataset.qm9 import (
    MAX_NODES,
    ATOM_TYPES,
    HYBRIDIZATIONS,
    ELECTRONEGATIVITY,
    ATOM_TO_ID,
    HYBRID_TO_ID,
    BOND_TO_ID,
    VALENCE_TABLE,
    BOND_ORDERS,
    ATOM_VOCAB_SIZE,
    HYBRID_VOCAB_SIZE,
    BOND_VOCAB_SIZE,
)

DEFAULT_DKNN_K = 5
DEFAULT_DKNN_ALPHA = 5.0



def compute_dknn(
    coords: np.ndarray,
    atom_mask: np.ndarray,
    *,
    k_max: int,
    alpha: float,
    symmetrize: bool = True,
) -> np.ndarray:
    """Compute multi-scale soft kNN distance features."""
    coords = np.asarray(coords)
    atom_mask = np.asarray(atom_mask).astype(bool)

    B, N, _ = coords.shape
    k_eff = max(0, min(k_max, N - 1))

    pair_mask = atom_mask[:, :, None] & atom_mask[:, None, :]

    # Pairwise distances
    diff = coords[:, :, None, :] - coords[:, None, :, :]
    dist = np.linalg.norm(diff, axis=-1)

    # Invalidate padded atoms and diagonal
    large = 1e5
    dist = np.where(pair_mask, dist, large)
    dist = dist + np.eye(N)[None] * large

    # Count valid neighbors per atom
    num_neighbors = pair_mask.sum(axis=-1) - 1  # exclude self

    dknn = np.zeros((B, N, N, k_max), dtype=coords.dtype)

    for k in range(1, k_eff + 1):
        dk = np.partition(dist, kth=k - 1, axis=-1)[..., k - 1]

        valid_k = num_neighbors >= k  # (B, N)

        delta = np.maximum(dist - dk[..., None], 0.0)
        feat = np.exp(-alpha * delta)

        # Zero out atoms without enough neighbors
        feat *= valid_k[..., None]

        if symmetrize:
            feat = np.maximum(feat, feat.transpose(0, 2, 1))

        dknn[..., k - 1] = feat

    # Final masking
    dknn *= pair_mask[..., None]
    dknn *= (1.0 - np.eye(N)[None, :, :, None])

    return dknn


def featurize_components(
    mol: Chem.Mol,
    dtype: DTypeLike = "float32",
    *,
    dknn_k: int = DEFAULT_DKNN_K,
    dknn_alpha: float = DEFAULT_DKNN_ALPHA,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return categorical ids, continuous scalars, edge types, dknn, and masks for a single molecule."""
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule has no conformers; cannot compute coordinates for dknn.")
    mol = Chem.AddHs(mol, addCoords=False)
    n_atoms = mol.GetNumAtoms()
    if n_atoms > MAX_NODES:
        raise ValueError(f"Molecule has {n_atoms} atoms, exceeds MAX_NODES={MAX_NODES}")

    atom_ids = np.zeros((MAX_NODES,), dtype=np.int32)
    hybrid_ids = np.zeros((MAX_NODES,), dtype=np.int32)
    node_cont = np.zeros((MAX_NODES, 4), dtype=dtype)  # electronegativity, degree/4, formal_valence/4, aromaticity
    bond_types = np.zeros((MAX_NODES, MAX_NODES), dtype=np.int32)
    node_mask = np.zeros((MAX_NODES,), dtype=dtype)
    coords = np.zeros((MAX_NODES, 3), dtype=dtype)

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
        pos = mol.GetConformer().GetAtomPosition(i)
        coords[i] = np.array([pos.x, pos.y, pos.z], dtype=dtype)

    for i in range(n_atoms):
        for j in range(n_atoms):
            bond = None if i == j else mol.GetBondBetweenAtoms(i, j)
            bond_type = bond.GetBondType() if bond is not None else "no_bond"
            bond_types[i, j] = BOND_TO_ID.get(bond_type, 0)

    pair_mask = node_mask[:, None] * node_mask[None, :]
    # remove self-interactions
    pair_mask = pair_mask * (1.0 - np.eye(MAX_NODES, dtype=dtype))
    dknn = compute_dknn(coords[None], node_mask[None], k_max=dknn_k, alpha=dknn_alpha)[0]
    return atom_ids, hybrid_ids, node_cont, bond_types, dknn, node_mask, pair_mask


def encode_molecule(
    mol: Chem.Mol,
    *,
    dtype: DTypeLike = "float32",
    as_batch: bool = False,
    dknn_k: int = DEFAULT_DKNN_K,
    dknn_alpha: float = DEFAULT_DKNN_ALPHA,
) -> Union[Dict[str, np.ndarray], GraphBatch]:
    """Encode an RDKit molecule into dense node/edge features (no flat mode).

    Example:
        >>> mol = Chem.MolFromSmiles("CCO")
        >>> features = encode_molecule(mol)
    """
    atom_ids, hybrid_ids, node_cont, bond_types, dknn, node_mask, pair_mask = featurize_components(
        mol, dtype=dtype, dknn_k=dknn_k, dknn_alpha=dknn_alpha
    )

    feats = {
        "atom_ids": atom_ids.astype(np.int32),
        "hybrid_ids": hybrid_ids.astype(np.int32),
        "node_continuous": node_cont.astype(dtype),
        "bond_types": bond_types.astype(np.int32),
        "dknn": dknn.astype(dtype),
        "node_mask": node_mask.astype(dtype),
        "pair_mask": pair_mask.astype(dtype),
    }
    if as_batch:
        batch = GraphBatch(
            atom_type=feats["atom_ids"],
            hybrid=feats["hybrid_ids"],
            cont=feats["node_continuous"],
            bond_type=feats["bond_types"],
            dknn=feats["dknn"],
            node_mask=feats["node_mask"],
            pair_mask=feats["pair_mask"],
        )
        return batch
    return feats


__all__ = [
    "DEFAULT_DKNN_K",
    "DEFAULT_DKNN_ALPHA",
    "compute_dknn",
    "featurize_components",
    "encode_molecule",
]
