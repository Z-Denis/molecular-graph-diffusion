import numpy as np

from mgd.dataset.utils import GraphBatch
from mgd.training.utils import compute_class_weights


def test_compute_class_weights_masks_padding():
    # labels: two batches, with padding label=0
    batch = GraphBatch(
        atom_type=np.array([[0, 1, 2], [0, 1, 1]], dtype=np.int32),
        hybrid=np.zeros((2, 3), dtype=np.int32),
        cont=np.zeros((2, 3, 1), dtype=np.float32),
        edges=np.zeros((2, 3, 3), dtype=np.int32),
        node_mask=np.ones((2, 3), dtype=np.float32),
        pair_mask=np.ones((2, 3, 3), dtype=np.float32),
    )
    loader = [batch]
    weights = compute_class_weights(loader, num_classes=3, pad_value=0)
    # padding weight should not blow up, and weights are mean-normalized
    assert weights.shape == (3,)
    assert np.isfinite(weights).all()
    assert np.isclose(weights.mean(), 1.0)
