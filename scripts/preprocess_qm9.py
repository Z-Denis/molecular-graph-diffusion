"""Preprocess QM9 into dense adjacency tensors with explicit hydrogens.

For each molecule, produce fixed-size arrays (max_nodes=29):
- Categorical atoms/hybridizations (one-hot for flat; integer ids for separate).
- Continuous scalars: electronegativity, degree/4, formal_valence/4, aromaticity.
- Edges: bond-type (one-hot for flat; integer ids for separate, 0 = no bond).
- Masks: node_mask, pair_mask (1 when both nodes exist).

Outputs a single .npz file with arrays stacked over molecules. Choose feature
representation with --feature_style (flat|separate). Flat concatenates node
one-hots + continuous scalars; separate saves categorical ids + continuous arrays.
Unknown categories map to id 0.

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
from typing import Dict, List

import numpy as np
from rdkit import Chem

from mgd.dataset.encoding import encode_molecule


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
    atom_id_list: List[np.ndarray] = []
    hybrid_id_list: List[np.ndarray] = []
    node_cont_list: List[np.ndarray] = []
    edge_type_list: List[np.ndarray] = []
    node_mask_list: List[np.ndarray] = []
    pair_mask_list: List[np.ndarray] = []

    for idx, mol in enumerate(mols):
        try:
            features = encode_molecule(mol, feature_style=args.feature_style, dtype=dtype)
        except ValueError as exc:
            print(f"Skipping molecule {idx} ({exc})")
            continue
        if args.feature_style == "flat":
            node_list.append(features["nodes"])
            edge_list.append(features["edges"])
        else:
            atom_id_list.append(features["atom_ids"])
            hybrid_id_list.append(features["hybrid_ids"])
            node_cont_list.append(features["node_continuous"])
            edge_type_list.append(features["edge_types"])
        node_mask_list.append(features["node_mask"])
        pair_mask_list.append(features["pair_mask"])
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{len(mols)} molecules...")

    arrays: Dict[str, np.ndarray] = {
        "node_mask": np.stack(node_mask_list, axis=0),
        "pair_mask": np.stack(pair_mask_list, axis=0),
    }
    if args.feature_style == "flat":
        arrays["nodes"] = np.stack(node_list, axis=0)
        arrays["edges"] = np.stack(edge_list, axis=0)
    else:
        arrays.update(
            {
                "atom_ids": np.stack(atom_id_list, axis=0),
                "hybrid_ids": np.stack(hybrid_id_list, axis=0),
                "node_continuous": np.stack(node_cont_list, axis=0),
                "edge_types": np.stack(edge_type_list, axis=0),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(f"Saved preprocessed data to {args.output}")


if __name__ == "__main__":
    main()
