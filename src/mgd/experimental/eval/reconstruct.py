"""Helpers to rebuild RDKit molecules from decoded atom/bond classes (legacy)."""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
from rdkit import Chem

from mgd.dataset.qm9 import ATOM_TYPES, BOND_TO_ID

# Invert bond mapping (skip "no_bond")
_ID_TO_BOND = {v: k for k, v in BOND_TO_ID.items()}


def build_molecules(
    atom_ids: np.ndarray,
    bond_ids: np.ndarray,
    *,
    n_atoms: Optional[np.ndarray] = None,
    node_mask: Optional[np.ndarray] = None,
    sanitize: bool = True,
) -> List[Optional[Chem.Mol]]:
    """Reconstruct RDKit molecules from integer atom and bond classes.

    Args:
        atom_ids: array (batch, max_atoms) with categorical atom IDs (0 = pad/unknown).
        bond_ids: array (batch, max_atoms, max_atoms) with bond type IDs (0 = no bond).
        n_atoms: optional (batch,) number of real atoms per example; if None, inferred from node_mask.
        node_mask: optional mask to infer n_atoms when n_atoms is None.
        sanitize: whether to sanitize molecules after construction.

    Returns:
        List of RDKit Mol or None if construction failed for an example.
    """
    atom_ids = np.asarray(atom_ids)
    bond_ids = np.asarray(bond_ids)
    batch_size = atom_ids.shape[0]
    if bond_ids.shape[0] != batch_size:
        raise ValueError("atom_ids and bond_ids must share batch dimension.")
    if atom_ids.shape[1] != bond_ids.shape[1]:
        raise ValueError("atom_ids and bond_ids must share atom dimension.")

    if n_atoms is None:
        if node_mask is None:
            raise ValueError("Provide n_atoms or node_mask to infer atom counts.")
        n_atoms_arr = np.asarray(node_mask).sum(axis=-1).astype(int)
    else:
        n_atoms_arr = np.asarray(n_atoms).astype(int)
        if n_atoms_arr.shape[0] != batch_size:
            raise ValueError("n_atoms must match batch dimension.")

    results: List[Optional[Chem.Mol]] = []
    max_atoms = atom_ids.shape[1]

    for b in range(batch_size):
        count = int(n_atoms_arr[b])
        if count > max_atoms:
            results.append(None)
            continue
        try:
            rw = Chem.RWMol()
            for i in range(count):
                aid = int(atom_ids[b, i])
                if 0 < aid <= len(ATOM_TYPES):
                    symbol = ATOM_TYPES[aid - 1]
                else:
                    symbol = "C"  # fallback for unknown/padding
                rw.AddAtom(Chem.Atom(symbol))

            for i in range(count):
                for j in range(i + 1, count):
                    bid = int(bond_ids[b, i, j])
                    btype = _ID_TO_BOND.get(bid)
                    if btype in (None, "no_bond"):
                        continue
                    rw.AddBond(i, j, btype)

            if sanitize:
                Chem.SanitizeMol(rw)
            results.append(rw.GetMol())
        except Exception:
            results.append(None)

    return results


__all__ = ["build_molecules"]
