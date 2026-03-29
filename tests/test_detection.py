"""Basic tests for the detection pipeline."""

import numpy as np


def test_placeholder() -> None:
    """Placeholder — replace with real unit tests."""
    assert 1 + 1 == 2


def test_numpy_available() -> None:
    """Verify numpy (core dependency) is importable."""
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    assert arr.shape == (10, 10, 3)
