from types import SimpleNamespace

import numpy as np

from data import ChronologicalSequenceDataset, OfficialWindowDataset


class FakeSegmentDataset:
    def __init__(self):
        self.segments = [
            SimpleNamespace(timeline="a", start=0.0),
            SimpleNamespace(timeline="a", start=5.0),
            SimpleNamespace(timeline="b", start=0.0),
        ]

    def __len__(self):
        return 3

    def __getitem__(self, index):
        target = [595.0, 590.0, 100.0][index]
        return SimpleNamespace(
            data={
                "neuro": np.zeros((1, 4, 640), dtype=np.float32),
                "target": np.array([target], dtype=np.float32),
                "subject_id": np.array([index // 2]),
            }
        )


def test_shapes_targets_order_and_recording_boundaries():
    windows = OfficialWindowDataset(FakeSegmentDataset())
    x, y = windows[0]
    assert x.shape == (4, 640)
    assert float(y) == 595.0
    sequences = ChronologicalSequenceDataset(windows, sequence_length=2)
    assert len(sequences) == 2
    first_x, first_y, first_valid = sequences[0]
    second_x, second_y, second_valid = sequences[1]
    assert first_x.shape == second_x.shape == (2, 4, 640)
    assert first_y.tolist() == [595.0, 590.0]
    assert second_y[0] == 100.0
    assert first_valid.tolist() == [True, True]
    assert second_valid.tolist() == [True, False]
