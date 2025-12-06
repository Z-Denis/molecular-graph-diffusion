"""Preprocess QM9 into dense adjacency tensors with explicit hydrogens.

For each molecule, produce fixed-size arrays (max_nodes=28):
- Categorical one-hots for atoms and hybridizations (unknown -> zeros).
- Continuous scalars for electronegativity, degree/4, formal_valence/4, aromaticity.
- Edge bond-type one-hots (includes no-bond).
- Masks: node_mask, pair_mask, bond_mask.

Outputs a single .npz file with arrays stacked over molecules. Choose feature
representation with --feature_style (flat|separate). Flat concatenates node
one-hots + continuous scalars; separate saves each component. Unknown categories
are encoded as all-zero vectors.

Example:
    python scripts/preprocess_qm9.py \\
        --input data/raw/gdb9.sdf \\
        --output data/processed/qm9_dense.npz \\
        --dtype float32 \\
        --feature_style flat
    # or
    python scripts/preprocess_qm9.py --feature_style separate
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import DTypeLike
from rdkit import Chem

MAX_NODES = 29

ATOM_TYPES = ["H", "C", "N", "O", "F"]
HYBRIDIZATIONS = [
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    # Chem.HybridizationType.SP3D,    # Should not be in QM9
    # Chem.HybridizationType.SP3D2,   # Should not be in QM9
]

# Pauling electronegativity (approximate)
ELECTRONEGATIVITY = {
    "H": 2.20,
    "C": 2.55,
    "N": 3.04,
    "O": 3.44,
    "F": 3.98,
}

# Bond type mapping to indices
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

    node_cont = np.zeros((MAX_NODES, 4), dtype=dtype)  # electroneg, degree/4, formal_valence/4, aromaticity
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
                print(f'Hybridization type {hyb} was encountered for atom {symbol} in',
                      f'molecule {mol}. You might want to update `BOND_TYPES` at',
                      f'`scripts/preprocess_qm9.py` and rerun the preprocessing.')
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


def load_qm9_sdf(path: Path) -> List[Chem.Mol]:
    suppl = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
    return [mol for mol in suppl if mol is not None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess QM9 into dense adjacency format.")
    parser.add_argument("--input", type=Path, default=Path("data/raw/gdb9.sdf"), help="Path to QM9 SDF file.")
    parser.add_argument("--output", type=Path, default=Path("data/processed/qm9_dense.npz"), help="Output .npz file path.")
    parser.add_argument("--dtype", type=str, default="float32", help="Floating dtype for output arrays.")
    parser.add_argument(
        "--feature_style",
        type=str,
        default="flat",
        choices=["flat", "separate"],
        help="Whether to output flat concatenated features or split categorical vs continuous arrays.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = np.dtype(args.dtype)

    mols = load_qm9_sdf(args.input)
    print(f"Loaded {len(mols)} molecules from {args.input}")

    node_list: List[np.ndarray] = []
    edge_list: List[np.ndarray] = []
    atom_oh_list: List[np.ndarray] = []
    hybrid_oh_list: List[np.ndarray] = []
    node_cont_list: List[np.ndarray] = []
    edge_oh_list: List[np.ndarray] = []
    node_mask_list: List[np.ndarray] = []
    pair_mask_list: List[np.ndarray] = []
    bond_mask_list: List[np.ndarray] = []

    for idx, mol in enumerate(mols):
        try:
            atom_oh, hybrid_oh, node_cont, edge_oh, node_mask, pair_mask, bond_mask = featurize_components(
                mol, dtype=dtype
            )
        except ValueError as exc:
            print(f"Skipping molecule {idx} ({exc})")
            continue
        if args.feature_style == "flat":
            nodes = build_flat_features(atom_oh, hybrid_oh, node_cont)
            node_list.append(nodes.astype(dtype))
            edge_list.append(edge_oh.astype(dtype))
        else:
            atom_oh_list.append(atom_oh.astype(dtype))
            hybrid_oh_list.append(hybrid_oh.astype(dtype))
            node_cont_list.append(node_cont.astype(dtype))
            edge_oh_list.append(edge_oh.astype(dtype))
        node_mask_list.append(node_mask.astype(dtype))
        pair_mask_list.append(pair_mask.astype(dtype))
        bond_mask_list.append(bond_mask.astype(dtype))
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{len(mols)} molecules...")

    arrays: Dict[str, np.ndarray] = {
        "node_mask": np.stack(node_mask_list, axis=0),
        "pair_mask": np.stack(pair_mask_list, axis=0),
        "bond_mask": np.stack(bond_mask_list, axis=0),
    }
    if args.feature_style == "flat":
        arrays["nodes"] = np.stack(node_list, axis=0)
        arrays["edges"] = np.stack(edge_list, axis=0)
    else:
        arrays.update(
            {
                "atom_one_hot": np.stack(atom_oh_list, axis=0),
                "hybrid_one_hot": np.stack(hybrid_oh_list, axis=0),
                "node_continuous": np.stack(node_cont_list, axis=0),
                "edge_one_hot": np.stack(edge_oh_list, axis=0),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(f"Saved preprocessed data to {args.output}")


if __name__ == "__main__":
    main()
