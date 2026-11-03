"""Create the minimal Codabench submission archive."""

from __future__ import annotations

import zipfile
from pathlib import Path

import torch


def package_submission(best_checkpoint: Path, submission_py: Path, output: Path) -> None:
    checkpoint = torch.load(best_checkpoint, map_location="cpu", weights_only=True)
    weights = output.parent / "model_weights.pt"
    torch.save(
        {
            "model_state_dict": checkpoint["model_state_dict"],
            "model_config": checkpoint["model_config"],
            "spectral_feature_scaler": checkpoint["spectral_feature_scaler"],
        },
        weights,
    )
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(submission_py, "submission.py")
        archive.write(weights, "model_weights.pt")
    weights.unlink()
    with zipfile.ZipFile(output) as archive:
        if sorted(archive.namelist()) != ["model_weights.pt", "submission.py"]:
            raise RuntimeError("Invalid submission ZIP layout.")
