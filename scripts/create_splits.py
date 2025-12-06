"""Generate train/val/test split indices for a dataset.

Creates disjoint random splits given ratios and number of samples, and saves
the indices to a compressed .npz file (default: data/processed/splits.npz).

Example:
    python scripts/create_splits.py \\
        --num_samples 131970 \\
        --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 \\
        --output data/processed/qm9_splits.npz \\
        --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create random dataset splits.")
    parser.add_argument("--num_samples", type=int, required=True, help="Total number of samples to split.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Proportion for training split.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Proportion for validation split.")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Proportion for test split.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/splits.npz"),
        help="Path to save split indices (.npz).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(total_ratio, 1.0, atol=1e-6):
        raise ValueError(f"Ratios must sum to 1.0; got {total_ratio}")

    n = args.num_samples
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)

    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    n_test = n - n_train - n_val

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, train=train_idx, val=val_idx, test=test_idx, seed=args.seed)
    print(
        f"Saved splits to {args.output} "
        f"(train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)})"
    )


if __name__ == "__main__":
    main()
