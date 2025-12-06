"""Preprocess QM9 into dense adjacency tensors with explicit hydrogens.

For each molecule, produce fixed-size arrays (max_nodes=28):
- nodes: (max_nodes, d_node) with atom encodings and scalar features
- edges: (max_nodes, max_nodes, d_edge) one-hot bond types (incl. no-bond)
- node_mask: (max_nodes,) 1 for real atoms, 0 for padding
- pair_mask: (max_nodes, max_nodes) 1 when both atoms exist
- bond_mask: (max_nodes, max_nodes) 1 when a bond exists

Outputs a single .npz file with arrays stacked over molecules.

Example:
    python scripts/preprocess_qm9.py \\
        --input data/raw/gdb9.sdf \\
        --output data/processed/qm9_dense.npz \\
        --dtype float32
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import DTypeLike
from rdkit import Chem

MAX_NODES = 29

# Allowed categorical vocabularies
ATOM_TYPES = ["H", "C", "N", "O", "F"]
HYBRIDIZATIONS = [
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D,
    Chem.HybridizationType.SP3D2,
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


def one_hot(value, choices: List, dtype: DTypeLike = "float32") -> np.ndarray:
    vec = np.zeros(len(choices), dtype=dtype)
    if value in choices:
        vec[choices.index(value)] = 1.0
    return vec


def encode_atom(atom: Chem.Atom, dtype: DTypeLike = "float32") -> np.ndarray:
    symbol = atom.GetSymbol()
    atom_oh = one_hot(symbol, ATOM_TYPES, dtype=dtype)
    electroneg = np.array([ELECTRONEGATIVITY.get(symbol, 0.0)], dtype=dtype)
    hybrid = one_hot(atom.GetHybridization(), HYBRIDIZATIONS, dtype=dtype)
    degree = np.array([atom.GetDegree() / 4.0], dtype=dtype)
    formal_valence = np.array([atom.GetFormalCharge() / 4.0], dtype=dtype)
    aromatic = np.array([1.0 if atom.GetIsAromatic() else 0.0], dtype=dtype)
    return np.concatenate([atom_oh, electroneg, hybrid, degree, formal_valence, aromatic])


def encode_bond_type(bond: Chem.Bond | None, dtype: DTypeLike = "float32") -> np.ndarray:
    """One-hot for bond type, including no-bond."""
    if bond is None:
        idx = BOND_TYPES["no_bond"]
    else:
        idx = BOND_TYPES.get(bond.GetBondType(), 0)
    vec = np.zeros(len(BOND_TYPES), dtype=dtype)
    vec[idx] = 1.0
    return vec


def featurize_mol(mol: Chem.Mol, dtype: DTypeLike = "float32") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return dense node/edge features and masks for a single molecule."""
    mol = Chem.AddHs(mol, addCoords=False)
    n_atoms = mol.GetNumAtoms()
    if n_atoms > MAX_NODES:
        raise ValueError(f"Molecule has {n_atoms} atoms, exceeds MAX_NODES={MAX_NODES}")

    d_node = len(ATOM_TYPES) + 1 + len(HYBRIDIZATIONS) + 1 + 1 + 1
    d_edge = len(BOND_TYPES)

    nodes = np.zeros((MAX_NODES, d_node), dtype=dtype)
    edges = np.zeros((MAX_NODES, MAX_NODES, d_edge), dtype=dtype)
    node_mask = np.zeros((MAX_NODES,), dtype=dtype)
    pair_mask = np.zeros((MAX_NODES, MAX_NODES), dtype=dtype)
    bond_mask = np.zeros((MAX_NODES, MAX_NODES), dtype=dtype)

    # Nodes
    for i, atom in enumerate(mol.GetAtoms()):
        nodes[i] = encode_atom(atom, dtype=dtype)
        node_mask[i] = 1.0

    # Pair mask
    pair_mask = node_mask[:, None] * node_mask[None, :]

    # Edges and bond mask
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                edges[i, j] = encode_bond_type(None, dtype=dtype)
                continue
            bond = mol.GetBondBetweenAtoms(i, j)
            edges[i, j] = encode_bond_type(bond, dtype=dtype)
            if bond is not None:
                bond_mask[i, j] = 1.0

    return nodes, edges, node_mask, pair_mask, bond_mask


def load_qm9_sdf(path: Path) -> List[Chem.Mol]:
    suppl = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
    return [mol for mol in suppl if mol is not None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess QM9 into dense adjacency format.")
    parser.add_argument("--input", type=Path, default=Path("data/raw/gdb9.sdf"), help="Path to QM9 SDF file.")
    parser.add_argument("--output", type=Path, default=Path("data/processed/qm9_dense.npz"), help="Output .npz file path.")
    parser.add_argument("--dtype", type=str, default="float32", help="Floating dtype for output arrays.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = np.dtype(args.dtype)

    mols = load_qm9_sdf(args.input)
    print(f"Loaded {len(mols)} molecules from {args.input}")

    node_list: List[np.ndarray] = []
    edge_list: List[np.ndarray] = []
    node_mask_list: List[np.ndarray] = []
    pair_mask_list: List[np.ndarray] = []
    bond_mask_list: List[np.ndarray] = []

    for idx, mol in enumerate(mols):
        try:
            nodes, edges, node_mask, pair_mask, bond_mask = featurize_mol(mol, dtype=dtype)
        except ValueError as exc:
            print(f"Skipping molecule {idx} ({exc})")
            continue
        node_list.append(nodes.astype(dtype))
        edge_list.append(edges.astype(dtype))
        node_mask_list.append(node_mask.astype(dtype))
        pair_mask_list.append(pair_mask.astype(dtype))
        bond_mask_list.append(bond_mask.astype(dtype))
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{len(mols)} molecules...")

    arrays: Dict[str, np.ndarray] = {
        "nodes": np.stack(node_list, axis=0),
        "edges": np.stack(edge_list, axis=0),
        "node_mask": np.stack(node_mask_list, axis=0),
        "pair_mask": np.stack(pair_mask_list, axis=0),
        "bond_mask": np.stack(bond_mask_list, axis=0),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(f"Saved preprocessed data to {args.output}")


if __name__ == "__main__":
    main()
