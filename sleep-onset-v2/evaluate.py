"""Exact official metric plus detailed chronological streaming evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

BIN_EDGES = (0.0, 40.0, 90.0, 300.0, 600.0)
BIN_WEIGHTS = (10.0, 5.0, 3.0, 1.0)
BIN_NAMES = ("0_40", "40_90", "90_300", "300_600")


def bin_indices(y: np.ndarray) -> np.ndarray:
    return np.digitize(np.asarray(y), BIN_EDGES[1:-1], right=False)


def per_bin_mae(y_true, y_pred) -> list[float]:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    bins = bin_indices(y_true)
    return [
        float(np.abs(y_pred[bins == i] - y_true[bins == i]).mean())
        if np.any(bins == i)
        else float("nan")
        for i in range(4)
    ]


def binned_mae(y_true, y_pred, weights=None) -> float:
    maes = np.asarray(per_bin_mae(y_true, y_pred))
    present = np.isfinite(maes)
    if not present.any():
        return float("nan")
    w = np.ones(4) if weights is None else np.asarray(weights, dtype=np.float64)
    return float(np.sum(w[present] * maes[present]) / w[present].sum())


def recording_weighted_bmae(y_true, y_pred, record_ids) -> float:
    ids = np.asarray(record_ids)
    return float(
        np.mean(
            [
                binned_mae(
                    np.asarray(y_true)[ids == record_id],
                    np.asarray(y_pred)[ids == record_id],
                    BIN_WEIGHTS,
                )
                for record_id in np.unique(ids)
            ]
        )
    )


@dataclass
class EvaluationResult:
    summary: dict[str, float | int]
    predictions: list[dict]
    per_recording: list[dict]


@torch.inference_mode()
def evaluate_streaming(
    model: torch.nn.Module,
    records: Iterable,
    device: torch.device,
) -> EvaluationResult:
    model.eval()
    prediction_rows: list[dict] = []
    recording_rows: list[dict] = []
    all_y, all_pred, all_ids = [], [], []
    for record_id, participant_id, onsets, windows, labels in records:
        model.reset_state()
        predictions = np.asarray(
            [
                float(model.predict(window[None].to(device)).item())
                for window in windows
            ],
            dtype=np.float64,
        )
        targets = labels.cpu().numpy().astype(np.float64)
        bins = bin_indices(targets)
        maes = per_bin_mae(targets, predictions)
        rec_wbmae = binned_mae(targets, predictions, BIN_WEIGHTS)
        recording_rows.append(
            {
                "recording_id": int(record_id),
                "participant_id": int(participant_id),
                "number_of_windows": len(targets),
                "recording_wbmae": rec_wbmae,
                "recording_plain_mae": float(np.abs(predictions - targets).mean()),
                **{f"bin_{name}_mae": maes[i] for i, name in enumerate(BIN_NAMES)},
                "mean_prediction": float(predictions.mean()),
                "mean_target": float(targets.mean()),
                "recording_minutes": float(len(targets) * 5.0 / 60.0),
            }
        )
        for i, (onset, target, prediction, bin_id) in enumerate(
            zip(onsets, targets, predictions, bins)
        ):
            prediction_rows.append(
                {
                    "recording_id": int(record_id),
                    "participant_id": int(participant_id),
                    "window_index": i,
                    "window_end_seconds": float(onset + 5.0),
                    "target_seconds": float(target),
                    "predicted_seconds": float(prediction),
                    "absolute_error": float(abs(prediction - target)),
                    "signed_error": float(prediction - target),
                    "target_bin": BIN_NAMES[int(bin_id)],
                }
            )
        all_y.append(targets)
        all_pred.append(predictions)
        all_ids.append(np.full(len(targets), record_id))
    y = np.concatenate(all_y)
    pred = np.concatenate(all_pred)
    ids = np.concatenate(all_ids)
    maes = per_bin_mae(y, pred)
    summary: dict[str, float | int] = {
        "wbmae": recording_weighted_bmae(y, pred, ids),
        "bmae": binned_mae(y, pred),
        "mae": float(np.abs(pred - y).mean()),
        "n_recordings": len(recording_rows),
        "n_windows": len(y),
        **{f"bin_{name}_mae": maes[i] for i, name in enumerate(BIN_NAMES)},
    }
    for i, name in enumerate(BIN_NAMES):
        mask = bin_indices(y) == i
        summary[f"bin_{name}_bias"] = float((pred[mask] - y[mask]).mean())
    return EvaluationResult(summary, prediction_rows, recording_rows)
