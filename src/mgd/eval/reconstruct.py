"""Greedy decoding and RDKit reconstruction helpers."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from rdkit import Chem
from rdkit.Chem import rdchem, rdmolops

from mgd.dataset.chemistry import ChemistrySpec, DEFAULT_CHEMISTRY


def build_allowed_valences_by_index(
    spec: ChemistrySpec = DEFAULT_CHEMISTRY,
) -> list[Tuple[int, ...]]:
    allowed: list[Tuple[int, ...]] = [(0,)]
    allowed.extend(tuple(vals) for vals in spec.allowed_valences)
    return allowed


ALLOWED_BY_INDEX = build_allowed_valences_by_index(DEFAULT_CHEMISTRY)
K_CAPS_BY_INDEX = dict(DEFAULT_CHEMISTRY.k_caps_by_index)


def decode_greedy_valence_single(
    edge_logits: jnp.ndarray,
    atom_pred: jnp.ndarray,
    node_mask: jnp.ndarray,
    pair_mask: jnp.ndarray,
    *,
    bond_orders: np.ndarray = DEFAULT_CHEMISTRY.bond_orders,
    k_caps_by_index: Dict[int, int] | np.ndarray | None = None,  # TODO: find better name
    allowed_by_index: list[Tuple[int, ...]] | None = None,
    topk_margin: int = 1,                 # TODO: find better name
    stage1_top_m: int | None = None,      # TODO: find better name
    cleanup: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    """Greedy valence-constrained decoding for a single graph."""
    c = edge_logits.shape[-1]
    if c != 4:
        raise ValueError("Expected 4 bond classes (0..3).")

    # softmax over classes
    p = jax.nn.softmax(edge_logits, axis=-1)

    # symmetrize in probability space
    p = 0.5 * (p + jnp.swapaxes(p, -2, -3))

    # renormalize per pair
    z = p.sum(axis=-1, keepdims=True)
    p = p / (z + eps)

    # force diagonal to none-bond type
    n = p.shape[0]
    diag = jnp.eye(n, dtype=p.dtype)
    p = p * (1.0 - diag[..., None])
    p = p.at[jnp.arange(n), jnp.arange(n), 0].set(1.0)

    # existence / type probs
    p_none = p[..., 0]
    p_exist = 1.0 - p_none
    p_pos = p[..., 1:]
    p_pos_sum = p_pos.sum(axis=-1, keepdims=True)
    p_type = p_pos / (p_pos_sum + eps)

    # type + score
    k_star = jnp.argmax(p_pos, axis=-1) + 1  # 1..3
    score = p_exist * jnp.max(p_type, axis=-1)

    # to numpy for greedy
    p_exist = np.array(p_exist)
    k_star = np.array(k_star)
    score = np.array(score)
    atom_pred = np.array(atom_pred)
    node_mask = np.array(node_mask).astype(bool)
    pair_mask = np.array(pair_mask).astype(bool)

    # padding convention: atom_type==0 => invalid
    active = node_mask & (atom_pred > 0)

    # Optional k-neighbour pruning pruning. Per node, it:
    # - looks at p_exist[i, j] (bond existence probability),
    # - keeps only the top‑k neighbours (where k = k_caps_by_index[atom_type] + topk_margin),
    # - ignores invalid pairs (padding/diagonal).
    # This yields an undirected mask keep_undir used to filter candidate edges. It reduces 
    # the candidate list to the most plausible neighbours and improves the stability and 
    # efficiency of the greedy decoding.
    if k_caps_by_index is None:
        k_caps_by_index = K_CAPS_BY_INDEX
    if allowed_by_index is None:
        allowed_by_index = ALLOWED_BY_INDEX

    keep_undir = np.ones_like(pair_mask, dtype=bool)
    if k_caps_by_index is not None:
        keep_dir = np.zeros_like(pair_mask, dtype=bool)
        for i in range(n):
            if not active[i]:
                continue
            a = int(atom_pred[i])
            cap = (
                k_caps_by_index[a]
                if not isinstance(k_caps_by_index, dict)
                else k_caps_by_index.get(a, 0)
            )
            k = int(cap) + int(topk_margin)
            if k <= 0:
                continue
            vals = p_exist[i].copy()
            vals[i] = -np.inf
            vals[~pair_mask[i]] = -np.inf
            vals[~active] = -np.inf

            valid = np.isfinite(vals)
            m = int(valid.sum())
            k_eff = min(int(k), m)
            if k_eff <= 0:
                continue
            idx = np.argpartition(vals, -k_eff)[-k_eff:]
            keep_dir[i, idx] = True
        keep_undir = np.logical_or(keep_dir, keep_dir.T)
        keep_undir = np.logical_and(keep_undir, pair_mask)

    # valence caps
    def _allowed_for(a: int) -> Tuple[int, ...]:
        if 0 <= a < len(allowed_by_index):
            return allowed_by_index[a]
        return (0,)

    # initialise per-atom valence budget
    vcap = np.array([max(_allowed_for(a)) for a in atom_pred], dtype=float)
    v_sum = np.zeros(n, dtype=float)

    # candidate list
    candidates = []
    for i in range(n):
        if not active[i]:
            continue
        for j in range(i + 1, n):
            if not active[j]:
                continue
            if not keep_undir[i, j]:
                continue
            if not pair_mask[i, j]:
                continue
            k = int(k_star[i, j])
            order = float(bond_orders[k])
            s = float(score[i, j])
            candidates.append((s, i, j, k, order))

    candidates.sort(reverse=True, key=lambda x: x[0])

    # two-stage greedy
    if stage1_top_m is None:
        stage1_top_m = 2 * int(active.sum())  # default: 2N
    stage1 = candidates[:stage1_top_m]
    stage2 = candidates[stage1_top_m:]

    bond_type = np.zeros((n, n), dtype=np.int32)

    def try_accept(cands):
        nonlocal v_sum, bond_type
        for _s, i, j, k, order in cands:
            if v_sum[i] + order <= vcap[i] and v_sum[j] + order <= vcap[j]:
                bond_type[i, j] = k
                bond_type[j, i] = k
                v_sum[i] += order
                v_sum[j] += order

    try_accept(stage1)
    try_accept(stage2)

    # cleanup: prune weakest edges until v_sum in allowed set
    if cleanup:
        inc: list[list[Tuple[float, int, float]]] = [[] for _ in range(n)]
        for s, i, j, _k, order in candidates:
            if bond_type[i, j] != 0:
                inc[i].append((s, j, order))
                inc[j].append((s, i, order))
        for i in range(n):
            if not active[i]:
                continue
            allowed_i = _allowed_for(int(atom_pred[i]))
            if v_sum[i] in allowed_i:
                continue
            inc[i].sort(key=lambda x: x[0])  # weakest first
            for _s, j, order in inc[i]:
                if v_sum[i] in allowed_i:
                    break
                if bond_type[i, j] == 0:
                    continue
                bond_type[i, j] = 0
                bond_type[j, i] = 0
                v_sum[i] -= order
                v_sum[j] -= order

    np.fill_diagonal(bond_type, 0)
    return bond_type


def decode_greedy_valence_batch(
    edge_logits: jnp.ndarray,
    atom_pred: jnp.ndarray,
    node_mask: jnp.ndarray,
    pair_mask: jnp.ndarray,
    *,
    spec: ChemistrySpec = DEFAULT_CHEMISTRY,
    **kwargs,
) -> jnp.ndarray:
    """Batch wrapper for decode_greedy_valence_single."""
    kwargs.setdefault("bond_orders", spec.bond_orders)
    kwargs.setdefault("k_caps_by_index", dict(spec.k_caps_by_index))
    kwargs.setdefault("allowed_by_index", build_allowed_valences_by_index(spec))
    bsz = edge_logits.shape[0]
    out = []
    for b in range(bsz):
        out.append(
            decode_greedy_valence_single(
                edge_logits[b],
                atom_pred[b],
                node_mask[b],
                pair_mask[b],
                **kwargs,
            )
        )
    return jnp.asarray(np.stack(out, axis=0))


BOND_TYPE_MAP = {
    1: rdchem.BondType.SINGLE,
    2: rdchem.BondType.DOUBLE,
    3: rdchem.BondType.TRIPLE,
}


def mol_from_predictions(
    atom_pred: np.ndarray,
    bond_pred: np.ndarray,
    node_mask: np.ndarray | None = None,
    *,
    spec: ChemistrySpec = DEFAULT_CHEMISTRY,
    keep_largest_component: bool = True,
) -> Chem.Mol:
    """Build an RDKit molecule from predicted atom/bond classes."""
    atom_pred = list(map(int, atom_pred))
    n = len(atom_pred)
    if node_mask is None:
        node_mask = [a > 0 for a in atom_pred]
    else:
        node_mask = list(map(bool, node_mask))

    mol = Chem.RWMol()
    idx_map: Dict[int, int] = {}
    for i, a in enumerate(atom_pred):
        if not node_mask[i] or a == 0:
            continue
        if a > len(spec.atom_types):
            continue
        sym = spec.atom_types[a - 1]
        atom = Chem.Atom(sym)
        idx_map[i] = mol.AddAtom(atom)

    for i in range(n):
        if i not in idx_map:
            continue
        for j in range(i + 1, n):
            if j not in idx_map:
                continue
            b = int(bond_pred[i, j])
            if b == 0:
                continue
            bt = BOND_TYPE_MAP.get(b)
            if bt is None:
                continue
            mol.AddBond(idx_map[i], idx_map[j], bt)

    mol = mol.GetMol()

    # drop smaller unconnected chunks
    if keep_largest_component:
        frags = rdmolops.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
        if frags:
            frag_sizes = [len(f) for f in frags]
            keep = set(frags[int(np.argmax(frag_sizes))])
            to_remove = [i for i in range(mol.GetNumAtoms()) if i not in keep]
            if to_remove:
                em = Chem.EditableMol(mol)
                for idx in sorted(to_remove, reverse=True):
                    em.RemoveAtom(idx)
                mol = em.GetMol()

    return mol


def build_molecule_and_check(
    atom_pred: np.ndarray,
    bond_pred: np.ndarray,
    node_mask: np.ndarray | None = None,
    *,
    spec: ChemistrySpec = DEFAULT_CHEMISTRY,
    keep_largest_component: bool = True,
) -> Tuple[bool, Chem.Mol | Exception]:
    """Build a molecule and sanitize it, returning (is_valid, mol_or_error)."""
    mol = mol_from_predictions(
        atom_pred,
        bond_pred,
        node_mask,
        spec=spec,
        keep_largest_component=keep_largest_component,
    )
    try:
        Chem.SanitizeMol(mol)
        return True, mol
    except Exception as exc:
        return False, exc
