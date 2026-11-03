import copy
import sys
import types
import zipfile

import torch

from features import SpectralFeatureExtractor
from model import DualPathSleepNet
from package_submission import package_submission


def test_deepcopy_after_prediction():
    model = DualPathSleepNet(dropout=0.0).eval()
    model.predict(torch.randn(1, 4, 640))
    clone = copy.deepcopy(model)
    clone.reset_state()
    output = clone.predict(torch.randn(1, 4, 640))
    assert output.shape == (1,)
    assert torch.isfinite(output).all()


def _submission_namespace():
    base = types.ModuleType("benchmark_utils.base_solver")
    base.CompetSolver = object
    sys.modules["benchmark_utils"] = types.ModuleType("benchmark_utils")
    sys.modules["benchmark_utils.base_solver"] = base
    namespace = {}
    exec(
        compile(
            open("submission.py", encoding="utf-8").read(),
            "submission.py",
            "exec",
        ),
        namespace,
    )
    return namespace


def test_training_and_submission_features_match():
    namespace = _submission_namespace()
    training = SpectralFeatureExtractor()
    submitted = namespace["SpectralFeatureExtractor"]()
    x = torch.randn(3, 4, 640)
    assert torch.allclose(
        training.raw_features(x), submitted.raw_features(x), atol=1e-6, rtol=1e-6
    )


def test_submission_zip_and_loader(tmp_path):
    model = DualPathSleepNet(dropout=0.0).eval()
    checkpoint = {
        "model_state_dict": model.state_dict(),
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
            "mean": model.features.feature_mean.cpu(),
            "std": model.features.feature_std.cpu(),
        },
        "optimizer_state_dict": {},
        "scheduler_state_dict": {},
        "epoch": 1,
        "best_validation_wbmae": 1.0,
        "config": {},
        "seed": 42,
    }
    best = tmp_path / "best.pt"
    torch.save(checkpoint, best)
    output = tmp_path / "submission.zip"
    package_submission(best, __import__("pathlib").Path("submission.py"), output)
    with zipfile.ZipFile(output) as archive:
        assert sorted(archive.namelist()) == ["model_weights.pt", "submission.py"]
        weights = archive.read("model_weights.pt")

    namespace = _submission_namespace()
    submission_dir = tmp_path / "unpacked"
    submission_dir.mkdir()
    (submission_dir / "model_weights.pt").write_bytes(weights)
    loaded = namespace["Solver"]().load_model(
        {"submission_dir": submission_dir, "device": "cpu", "n_chans": 4}
    )
    assert loaded.predict(torch.randn(1, 4, 640)).shape == (1,)
