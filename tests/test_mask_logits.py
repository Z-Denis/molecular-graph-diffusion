import numpy as np

from mgd.training.utils import mask_logits


def test_mask_logits():
    logits = np.zeros((2, 2, 3))
    mask = np.array([[1, 0], [0, 1]])
    masked = mask_logits(logits, mask, pad_class=0)
    # Check invalid positions: only pad_class remains finite
    assert np.isneginf(masked[0, 1, 1:]).all()
    assert np.isneginf(masked[1, 0, 1:]).all()
    # Valid positions unchanged
    assert np.all(masked[0, 0] == logits[0, 0])
