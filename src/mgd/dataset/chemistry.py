"""Generic chemistry schema and preset lookup table."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ChemistrySpec:
    """Canonical chemistry tables used across data, guidance, and decoding."""

    name: str
    max_nodes: int
    atom_types: tuple[str, ...]
    bond_to_id: Mapping[object, int]
    bond_orders: np.ndarray
    valence_table: np.ndarray
    allowed_valences: tuple[tuple[int, ...], ...]
    k_caps_by_index: Mapping[int, int]
    explicit_h: bool = False

    @property
    def atom_to_id(self) -> dict[str, int]:
        return {symbol: i + 1 for i, symbol in enumerate(self.atom_types)}

    @property
    def atom_vocab_size(self) -> int:
        return len(self.atom_types) + 1

    @property
    def bond_vocab_size(self) -> int:
        return max(self.bond_to_id.values()) + 1


from . import qm9 as _qm9  # noqa: E402

CHEMISTRIES: dict[str, ChemistrySpec] = {
    _qm9.QM9_IMPLICIT_H.name: _qm9.QM9_IMPLICIT_H,
    _qm9.QM9_EXPLICIT_H.name: _qm9.QM9_EXPLICIT_H,
}

DEFAULT_CHEMISTRY: ChemistrySpec = _qm9.QM9_IMPLICIT_H


def get_chemistry(name: str) -> ChemistrySpec:
    """Resolve a chemistry preset by name."""
    try:
        return CHEMISTRIES[name]
    except KeyError as exc:
        available = ", ".join(sorted(CHEMISTRIES)) or "<empty>"
        raise KeyError(f"Unknown chemistry '{name}'. Available: {available}") from exc


__all__ = [
    "ChemistrySpec",
    "CHEMISTRIES",
    "DEFAULT_CHEMISTRY",
    "get_chemistry",
]
