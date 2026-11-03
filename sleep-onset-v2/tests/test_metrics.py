import numpy as np

from evaluate import binned_mae, recording_weighted_bmae


def test_exact_bin_boundaries_and_weights():
    y = np.array([0.0, 39.9, 40.0, 89.9, 90.0, 299.9, 300.0, 600.0])
    prediction = y + np.array([1, 1, 2, 2, 3, 3, 4, 4])
    assert np.isclose(binned_mae(y, prediction), 2.5)
    expected = (10 * 1 + 5 * 2 + 3 * 3 + 1 * 4) / 19
    assert np.isclose(binned_mae(y, prediction, (10, 5, 3, 1)), expected)


def test_recordings_are_macro_averaged():
    y = np.array([10.0, 10.0, 10.0])
    prediction = np.array([10.0, 10.0, 20.0])
    ids = np.array([0, 0, 1])
    assert np.isclose(recording_weighted_bmae(y, prediction, ids), 5.0)
