import numpy as np

from mgd.training.utils import mask_logits


def test_mask_logits():
    logits = np.zeros((2, 2, 3))
    mask = np.array([[1, 0], [0, 1]])
    masked = mask_logits(logits, mask, pad_class=0)
    # Invalid positions: all classes -inf
    assert np.isneginf(masked[0, 1]).all()
    assert np.isneginf(masked[1, 0]).all()
    # pad_class always masked
    assert np.isneginf(masked[..., 0]).all()
    # Valid positions for non-pad classes remain finite
    assert np.all(masked[0, 0, 1:] == logits[0, 0, 1:])


def test_mask_logits_pad_kept_on_valid():
    logits = np.zeros((1, 2, 3))
    mask = np.array([[1, 0]])
    masked = mask_logits(logits, mask, pad_class=0)
    # pad masked everywhere
    assert np.isneginf(masked[..., 0]).all()
    # non-pad forbidden on invalid
    assert np.isneginf(masked[0, 1, 1:]).all()
    # valid position non-pad unchanged
    assert np.all(masked[0, 0, 1:] == logits[0, 0, 1:])
