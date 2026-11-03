"""Official Track 3 warm-up metric and causal streaming evaluation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

BIN_EDGES = (0.0, 40.0, 90.0, 300.0, 600.0)
BIN_WEIGHTS = (10.0, 5.0, 3.0, 1.0)


def binned_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bin_edges: tuple[float, ...] = BIN_EDGES,
    bin_weights: tuple[float, ...] | None = None,
) -> float:
    """Match ``benchmark_utils.metrics.binned_mae`` from the competition."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    edges = np.asarray(bin_edges, dtype=np.float64)
    weights = np.ones(len(edges) - 1) if bin_weights is None else np.asarray(
        bin_weights, dtype=np.float64
    )
    index = np.digitize(y_true, edges[1:-1], right=False)
    in_range = (y_true >= edges[0]) & (y_true <= edges[-1])
    error = np.abs(y_pred - y_true)
    present = [
        b for b in range(len(edges) - 1) if np.any(in_range & (index == b))
    ]
    if not present:
        return float("nan")
    maes = np.asarray([error[in_range & (index == b)].mean() for b in present])
    selected_weights = weights[present]
    return float(np.sum(selected_weights * maes) / selected_weights.sum())


def recording_weighted_bmae(
    y_true: np.ndarray, y_pred: np.ndarray, record_ids: np.ndarray
) -> float:
    """Official warm-up score: per-recording W-bMAE, then recording mean."""
    record_ids = np.asarray(record_ids).ravel()
    scores = [
        binned_mae(
            np.asarray(y_true)[record_ids == record_id],
            np.asarray(y_pred)[record_ids == record_id],
            bin_weights=BIN_WEIGHTS,
        )
        for record_id in np.unique(record_ids)
    ]
    return float(np.mean(scores))


@torch.inference_mode()
def evaluate_streaming(
    model: torch.nn.Module,
    records: Iterable[tuple[int, torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, float]:
    """Evaluate records one window at a time, resetting state at boundaries."""
    model.eval()
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    for record_id, windows, labels in records:
        model.reset_state()
        record_predictions = []
        for window in windows:
            prediction = model.predict(window[None].to(device))
            record_predictions.append(float(prediction.item()))
        labels_np = labels.cpu().numpy().astype(np.float64)
        targets.append(labels_np)
        predictions.append(np.asarray(record_predictions, dtype=np.float64))
        ids.append(np.full(len(labels_np), record_id, dtype=np.int64))

    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)
    record_ids = np.concatenate(ids)
    return {
        "wbmae": recording_weighted_bmae(y_true, y_pred, record_ids),
        "bmae": binned_mae(y_true, y_pred),
        "mae": float(np.abs(y_pred - y_true).mean()),
    }
