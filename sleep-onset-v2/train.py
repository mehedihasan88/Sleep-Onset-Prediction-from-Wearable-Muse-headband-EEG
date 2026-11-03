"""Fresh V2 training, streaming validation, diagnostics and packaging."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data import load_official_datasets
from diagnostics import (
    DiagnosticLogger,
    component_gradient_norms,
    feature_snapshot,
)
from evaluate import BIN_EDGES, BIN_NAMES, BIN_WEIGHTS, evaluate_streaming
from features import fit_feature_scaler
from model import DualPathSleepNet
from package_submission import package_submission


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def target_weights(target: torch.Tensor) -> torch.Tensor:
    boundaries = torch.tensor(BIN_EDGES[1:-1], device=target.device)
    bins = torch.bucketize(target, boundaries, right=True)
    return torch.tensor(BIN_WEIGHTS, device=target.device)[bins]


def weighted_l1(prediction, target, valid):
    weights = target_weights(target) * valid
    return ((prediction - target).abs() * weights).sum() / weights.sum().clamp_min(1)


def model_config(config: dict) -> dict:
    model = config["model"]
    return {
        "n_chans": model["n_chans"],
        "cnn_embedding": model["cnn_embedding"],
        "spectral_embedding": model["spectral_embedding"],
        "gru_hidden": model["gru_hidden"],
        "gru_layers": model["gru_layers"],
        "dropout": model["dropout"],
        "sampling_rate": model["sampling_rate"],
        "n_times": model["n_times"],
    }


def save_checkpoint(
    path, model, optimizer, scheduler, epoch, best, config, seed
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_validation_wbmae": best,
            "config": config,
            "model_config": model_config(config),
            "seed": seed,
            "spectral_feature_scaler": {
                "mean": model.features.feature_mean.detach().cpu(),
                "std": model.features.feature_std.detach().cpu(),
            },
        },
        path,
    )


def elapsed_time_diagnostic(train_windows, evaluation) -> dict:
    elapsed, targets = [], []
    for record_id, indices in train_windows.by_record.items():
        del record_id
        for index in indices:
            _x, y = train_windows[index]
            elapsed.append(float(train_windows.onsets[index] + 5.0))
            targets.append(float(y))
    design = np.column_stack([np.ones(len(elapsed)), np.asarray(elapsed)])
    intercept, slope = np.linalg.lstsq(design, np.asarray(targets), rcond=None)[0]
    rows = evaluation.predictions
    predicted = np.clip(
        intercept + slope * np.asarray([row["window_end_seconds"] for row in rows]),
        0,
        600,
    )
    y = np.asarray([row["target_seconds"] for row in rows])
    ids = np.asarray([row["recording_id"] for row in rows])
    from evaluate import recording_weighted_bmae

    frame_errors = np.abs(
        np.asarray([row["predicted_seconds"] for row in rows]) - y
    )
    midpoint = np.asarray([row["window_index"] for row in rows]) < np.asarray(
        [
            next(
                rec["number_of_windows"]
                for rec in evaluation.per_recording
                if rec["recording_id"] == row["recording_id"]
            )
            / 2
            for row in rows
        ]
    )
    rec_w = np.asarray([row["recording_wbmae"] for row in evaluation.per_recording])
    rec_len = np.asarray([row["recording_minutes"] for row in evaluation.per_recording])
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "validation_wbmae": recording_weighted_bmae(y, predicted, ids),
        "validation_plain_mae": float(np.abs(predicted - y).mean()),
        "model_early_half_mae": float(frame_errors[midpoint].mean()),
        "model_late_half_mae": float(frame_errors[~midpoint].mean()),
        "prediction_std": float(
            np.std([row["predicted_seconds"] for row in rows])
        ),
        "recording_length_wbmae_correlation": float(np.corrcoef(rec_len, rec_w)[0, 1]),
        "label": "train-fitted elapsed-time-only diagnostic; not submitted",
    }


def train(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    train_cfg = config["training"]
    if args.epochs is not None:
        train_cfg["max_epochs"] = args.epochs
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(args.output_dir)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    logger = DiagnosticLogger(output, config_path)
    logger.message(f"Device: {device}")
    sequence_data, train_windows, val_windows, _test, metadata = load_official_datasets(
        args.data_dir,
        sequence_length=config["data"]["sequence_length"],
        batch_size=train_cfg["batch_size"],
        seed=seed,
        download=args.download,
    )
    logger.message(json.dumps(metadata))
    model = DualPathSleepNet(**model_config(config)).to(device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.message(f"Fresh V2 parameters: {parameters:,}")

    scaler_loader = DataLoader(sequence_data, batch_size=train_cfg["batch_size"])
    mean, std, scaler_windows = fit_feature_scaler(
        model.features, scaler_loader, device
    )
    logger.message(f"Fitted spectral scaler on {scaler_windows} training windows.")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise RuntimeError("Non-finite training spectral scaler.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["max_epochs"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch, best, stale = 1, float("inf"), 0
    last_path, best_path = checkpoints / "last_model.pt", checkpoints / "best_model.pt"
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = int(state["epoch"]) + 1
        best = float(state["best_validation_wbmae"])
        logger.message(f"Resumed at epoch {start_epoch}.")

    batch_size = int(train_cfg["batch_size"])
    diagnostic_loader = DataLoader(sequence_data, batch_size=min(2, batch_size))
    diagnostic_batch = next(iter(diagnostic_loader))[0]
    started = time.monotonic()
    deadline = started + float(train_cfg["time_budget_hours"]) * 3600
    stop_reason = "maximum epochs"
    peak_memory = 0.0

    for epoch in range(start_epoch, train_cfg["max_epochs"] + 1):
        epoch_started = time.monotonic()
        while True:
            loader = DataLoader(
                sequence_data,
                batch_size=batch_size,
                shuffle=True,
                num_workers=train_cfg["num_workers"],
                pin_memory=device.type == "cuda",
            )
            model.train()
            losses, maes, gradient_norms = [], [], []
            last_component_grads = {}
            try:
                for windows, targets, valid in loader:
                    windows = windows.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    valid = valid.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast(
                        device_type=device.type, enabled=device.type == "cuda"
                    ):
                        predictions, _ = model(windows)
                        loss = weighted_l1(predictions, targets, valid)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    last_component_grads = component_gradient_norms(model)
                    grad = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), train_cfg["gradient_clip"]
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    losses.append(float(loss.detach()))
                    maes.append(
                        float(
                            (
                                ((predictions - targets).abs() * valid).sum()
                                / valid.sum().clamp_min(1)
                            ).detach()
                        )
                    )
                    gradient_norms.append(float(grad))
                break
            except torch.OutOfMemoryError:
                if batch_size <= 1:
                    raise
                batch_size //= 2
                torch.cuda.empty_cache()
                logger.message(f"CUDA OOM: retrying epoch with batch size {batch_size}.")

        training_seconds = time.monotonic() - epoch_started
        validation_started = time.monotonic()
        evaluation = evaluate_streaming(model, val_windows.iter_records(), device)
        validation_seconds = time.monotonic() - validation_started
        scheduler.step()
        if device.type == "cuda":
            peak_memory = max(peak_memory, torch.cuda.max_memory_allocated() / 2**20)
        feature_row = {
            **feature_snapshot(model, diagnostic_batch, epoch),
            **last_component_grads,
        }
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_mae": float(np.mean(maes)),
            "validation_wbmae": evaluation.summary["wbmae"],
            "validation_bmae": evaluation.summary["bmae"],
            "validation_plain_mae": evaluation.summary["mae"],
            **{
                f"validation_bin_{name}_mae": evaluation.summary[f"bin_{name}_mae"]
                for name in BIN_NAMES
            },
            "learning_rate": optimizer.param_groups[0]["lr"],
            "gradient_norm_mean": float(np.mean(gradient_norms)),
            "training_seconds": training_seconds,
            "validation_seconds": validation_seconds,
            "gpu_peak_memory_mb": peak_memory,
            "train_windows": scaler_windows,
            "validation_windows": evaluation.summary["n_windows"],
            "validation_recordings": evaluation.summary["n_recordings"],
            "batch_size": batch_size,
        }
        logger.log_epoch(row, feature_row)
        logger.message(
            f"Epoch {epoch:02d} | loss {row['train_loss']:.4f} | "
            f"val W-bMAE {row['validation_wbmae']:.4f}"
        )
        improved = float(evaluation.summary["wbmae"]) < best
        if improved:
            best = float(evaluation.summary["wbmae"])
            stale = 0
            save_checkpoint(
                best_path, model, optimizer, scheduler, epoch, best, config, seed
            )
        else:
            stale += 1
        save_checkpoint(last_path, model, optimizer, scheduler, epoch, best, config, seed)
        if stale >= train_cfg["early_stopping_patience"]:
            stop_reason = f"early stopping after {stale} non-improving epochs"
            break
        if time.monotonic() > deadline - 15 * 60:
            stop_reason = "wall-clock budget (15-minute reserve)"
            break

    best_state = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(best_state["model_state_dict"])
    final_evaluation = evaluate_streaming(model, val_windows.iter_records(), device)
    elapsed = elapsed_time_diagnostic(train_windows, final_evaluation)
    summary = {
        "seed": seed,
        "parameters": parameters,
        "training_runtime_seconds": time.monotonic() - started,
        "device": str(device),
        "stop_reason": stop_reason,
        "peak_gpu_memory_mb": peak_memory,
    }
    logger.save_final(final_evaluation, model, summary, elapsed)
    package_submission(
        best_path, Path(__file__).with_name("submission.py"), output / "submission.zip"
    )
    logger.message(f"Best validation W-bMAE: {best:.4f}")
    logger.message(f"Submission: {output / 'submission.zip'}")
    logger.message(f"Diagnostics: {output / 'v2_diagnostics.zip'}")
    logger.archive(output / "v2_diagnostics.zip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.yaml")
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
