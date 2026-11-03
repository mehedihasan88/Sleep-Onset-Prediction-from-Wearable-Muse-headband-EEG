import copy
import sys
import types
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from data import ChronologicalSequenceDataset, FullRecordingDataset, OfficialWindowDataset
from evaluate import BIN_WEIGHTS, bin_indices, recording_weighted_bmae
from features import SpectralFeatureExtractor
from model import DualPathSleepNet
from package_submission import package_submission
from train import (
    EXPECTED_PARAMETERS,
    build_accumulation_groups,
    weighted_l1_parts,
)


def model():
    torch.manual_seed(42)
    return DualPathSleepNet(dropout=0.0).eval()


def test_architecture_and_features():
    network = model()
    assert sum(p.numel() for p in network.parameters() if p.requires_grad) == (
        EXPECTED_PARAMETERS
    )
    values = SpectralFeatureExtractor().raw_features(torch.randn(3, 4, 640))
    assert values.shape == (3, 48)
    assert torch.isfinite(values).all()


def test_causality_streaming_and_recording_isolation():
    network = model()
    x = torch.randn(1, 8, 4, 640)
    changed = x.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 100
    with torch.inference_mode():
        full, _ = network(x)
        altered, _ = network(changed)
        state, streamed = None, []
        for step in range(x.shape[1]):
            prediction, state = network(x[:, step], state)
            streamed.append(prediction)
    assert torch.allclose(full[:, :4], altered[:, :4], atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        full, torch.stack(streamed, dim=1), atol=1e-5, rtol=1e-5
    )
    untouched = copy.deepcopy(network)
    network.predict(torch.randn(1, 4, 640))
    network.reset_state()
    recording = torch.randn(4, 4, 640)
    after = torch.cat([network.predict(window[None]) for window in recording])
    untouched.reset_state()
    direct = torch.cat([untouched.predict(window[None]) for window in recording])
    assert torch.allclose(after, direct, atol=1e-5, rtol=1e-5)


def test_loss_boundaries_and_accumulation():
    target = torch.tensor([[0.0, 39.9, 40.0, 89.9, 90.0, 299.9, 300.0, 600.0]])
    prediction = target + 2
    numerator, denominator = weighted_l1_parts(prediction, target)
    bins = torch.as_tensor(bin_indices(target.numpy()))
    weights = torch.tensor(BIN_WEIGHTS)[bins]
    expected = ((prediction - target).abs() * weights).sum() / weights.sum()
    assert torch.equal(numerator / denominator, expected)
    groups = build_accumulation_groups([2, 0, 3, 1], [100, 200, 250, 20], 384)
    assert [index for group in groups for index in group] == [2, 0, 3, 1]


def test_fp16_embeddings_keep_recurrence_and_gradients_finite():
    network = model().train()
    original = network.window_embeddings

    def half_embeddings(windows, return_diagnostics=False):
        return original(windows, return_diagnostics=False).half()

    network.window_embeddings = half_embeddings
    prediction, state = network(torch.randn(1, 30, 4, 640))
    assert prediction.dtype == torch.float32
    assert state.dtype == torch.float32
    prediction.sum().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in network.parameters()
    )


def test_gradient_propagates_beyond_24_steps_without_retained_state():
    network = model().train()
    x = torch.randn(1, 26, 4, 640)
    fused = network.window_embeddings(x.flatten(0, 1)).reshape(1, 26, -1)
    fused.retain_grad()
    recurrent, _ = network.gru(fused)
    prediction = torch.sigmoid(network.head(recurrent[:, -1])).sum()
    prediction.backward()
    assert torch.isfinite(fused.grad).all()
    assert fused.grad[:, 0].abs().sum() > 0
    assert network._state is None


class FakeSegmentDataset:
    segments = [
        SimpleNamespace(timeline="a", start=0.0),
        SimpleNamespace(timeline="a", start=5.0),
        SimpleNamespace(timeline="b", start=0.0),
    ]

    def __len__(self):
        return 3

    def __getitem__(self, index):
        return SimpleNamespace(
            data={
                "neuro": np.zeros((1, 4, 640), dtype=np.float32),
                "target": np.asarray([[595.0, 590.0, 100.0][index]], dtype=np.float32),
                "subject_id": np.asarray([index // 2]),
            }
        )


def test_recording_dataset_keeps_boundaries_and_order():
    windows = OfficialWindowDataset(FakeSegmentDataset())
    chunks = ChronologicalSequenceDataset(windows, 2)
    recordings = FullRecordingDataset(windows)
    assert len(chunks) == len(recordings) == 2
    assert recordings[0]["targets"].tolist() == [595.0, 590.0]
    assert recordings[1]["targets"].tolist() == [100.0]
    assert recordings.lengths == {0: 2, 1: 1}


def test_official_recording_metric():
    target = np.asarray([10, 50, 100, 400, 20, 60, 200, 500])
    prediction = target + np.asarray([1, 2, 3, 4, 2, 4, 6, 8])
    ids = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    assert np.isclose(recording_weighted_bmae(target, prediction, ids), 99 / 38)


def submission_namespace():
    base = types.ModuleType("benchmark_utils.base_solver")
    base.CompetSolver = object
    sys.modules["benchmark_utils"] = types.ModuleType("benchmark_utils")
    sys.modules["benchmark_utils.base_solver"] = base
    namespace = {}
    source = Path("submission.py").read_text(encoding="utf-8")
    exec(compile(source, "submission.py", "exec"), namespace)
    return namespace


def test_submission_features_loader_and_zip(tmp_path):
    network = model()
    submitted = submission_namespace()["SpectralFeatureExtractor"]()
    x = torch.randn(2, 4, 640)
    assert torch.allclose(
        network.features.raw_features(x), submitted.raw_features(x), atol=1e-6, rtol=1e-6
    )
    checkpoint = {
        "model_state_dict": network.state_dict(),
        "model_config": {
            "n_chans": 4,
            "cnn_embedding": 128,
            "spectral_embedding": 64,
            "gru_hidden": 128,
            "gru_layers": 1,
            "dropout": 0.0,
            "sampling_rate": 128.0,
            "n_times": 640,
        },
        "spectral_feature_scaler": {
            "mean": network.features.feature_mean,
            "std": network.features.feature_std,
        },
    }
    best = tmp_path / "best.pt"
    output = tmp_path / "submission.zip"
    torch.save(checkpoint, best)
    package_submission(best, Path("submission.py"), output)
    with zipfile.ZipFile(output) as archive:
        assert sorted(archive.namelist()) == ["model_weights.pt", "submission.py"]
        assert "V3-FullRecording" in archive.read("submission.py").decode()
        unpacked = tmp_path / "unpacked"
        archive.extractall(unpacked)
    loaded = submission_namespace()["Solver"]().load_model(
        {"submission_dir": unpacked, "device": "cpu", "n_chans": 4}
    )
    loaded.predict(torch.randn(1, 4, 640))
    clone = copy.deepcopy(loaded)
    clone.reset_state()
    assert torch.isfinite(clone.predict(torch.randn(1, 4, 640))).all()
