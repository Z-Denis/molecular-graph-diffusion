"""Preprocess QM9 into dense adjacency tensors with explicit hydrogens.

For each molecule, produce fixed-size arrays (max_nodes=29):
- Categorical atoms (integer ids).
- Edges: bond-type ids (0 = no bond).
- Masks: node_mask, pair_mask (1 when both nodes exist).

Outputs a single .npz file with arrays stacked over molecules.

Example:
    python scripts/preprocess_qm9.py \\
        --input data/raw/gdb9.sdf \\
        --output data/processed/qm9_dense.npz \\
        --dtype float32 \\
        --kekulize
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
from rdkit import Chem

from mgd.dataset import encode_molecule


def load_qm9_sdf(path: Path) -> List[Chem.Mol]:
    suppl = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
    return [mol for mol in suppl if mol is not None]


def kekulize_molecule(mol: Chem.Mol) -> Chem.Mol | None:
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:
        return None
    return mol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess QM9 into dense adjacency format.")
    parser.add_argument("--input", type=Path, default=Path("data/raw/gdb9.sdf"), help="Path to QM9 SDF file.")
    parser.add_argument("--output", type=Path, default=Path("data/processed/qm9_dense.npz"), help="Output .npz file path.")
    parser.add_argument("--dtype", type=str, default="float32", help="Floating dtype for output arrays.")
    parser.add_argument(
        "--kekulize",
        dest="kekulize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Kekulize molecules before encoding. Failures are dropped.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = np.dtype(args.dtype)

    mols = load_qm9_sdf(args.input)
    print(f"Loaded {len(mols)} molecules from {args.input}")

    atom_id_list: List[np.ndarray] = []
    bond_type_list: List[np.ndarray] = []
    node_mask_list: List[np.ndarray] = []
    pair_mask_list: List[np.ndarray] = []

    processed = 0
    dropped_kek = 0
    for idx, mol in enumerate(mols):
        if args.kekulize:
            mol = kekulize_molecule(mol)
            if mol is None:
                dropped_kek += 1
                continue
        try:
            features = encode_molecule(
                mol,
                dtype=dtype,
            )
        except ValueError as exc:
            print(f"Skipping molecule {idx} ({exc})")
            continue
        atom_id_list.append(features["atom_type"])
        bond_type_list.append(features["bond_type"])
        node_mask_list.append(features["node_mask"])
        pair_mask_list.append(features["pair_mask"])
        processed += 1
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{len(mols)} molecules...")

    if processed == 0:
        raise RuntimeError("No molecules were successfully processed; check input data.")
    if args.kekulize and dropped_kek:
        print(f"Dropped {dropped_kek} molecules that failed kekulization.")

    arrays: Dict[str, np.ndarray] = {
        "atom_type": np.stack(atom_id_list, axis=0),
        "bond_type": np.stack(bond_type_list, axis=0),
        "node_mask": np.stack(node_mask_list, axis=0),
        "pair_mask": np.stack(pair_mask_list, axis=0),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(f"Saved preprocessed data to {args.output}")


if __name__ == "__main__":
    main()
