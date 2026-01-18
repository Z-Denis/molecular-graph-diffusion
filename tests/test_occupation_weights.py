import numpy as np

from mgd.dataset.utils import GraphBatch
from mgd.training.utils import compute_occupation_log_weights


def test_compute_occupation_log_weights():
    batch = GraphBatch(
        atom_type=np.zeros((2, 4), dtype=np.int32),
        bond_type=np.zeros((2, 4, 4), dtype=np.int32),
        node_mask=np.array([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=np.float32),
        pair_mask=np.ones((2, 4, 4), dtype=np.float32),
    )
    log_weights = compute_occupation_log_weights([batch], n_atom_max=4)
    # occupations: first sample has 2 atoms, second has 3 atoms
    assert log_weights.shape == (5,)
    assert np.isfinite(log_weights).all()
