"""V3 controlled experiment: complete-recording BPTT with the V2 model."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data import FullRecordingDataset, load_official_datasets
from diagnostics import ExperimentLogger, component_gradient_norms, feature_snapshot
from evaluate import BIN_EDGES, BIN_NAMES, BIN_WEIGHTS, evaluate_streaming
from features import fit_feature_scaler
from model import DualPathSleepNet
from package_submission import package_submission

EXPECTED_PARAMETERS = 143_777


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_config(config: dict) -> dict:
    model = config["model"]
    return {
        key: model[key]
        for key in (
            "n_chans",
            "cnn_embedding",
            "spectral_embedding",
            "gru_hidden",
            "gru_layers",
            "dropout",
            "sampling_rate",
            "n_times",
        )
    }


def target_weights(target: torch.Tensor) -> torch.Tensor:
    boundaries = torch.tensor(
        BIN_EDGES[1:-1], device=target.device, dtype=target.dtype
    )
    bins = torch.bucketize(target, boundaries, right=True)
    return torch.tensor(BIN_WEIGHTS, device=target.device, dtype=target.dtype)[bins]


def weighted_l1_parts(prediction, target) -> tuple[torch.Tensor, torch.Tensor]:
    weights = target_weights(target)
    return ((prediction - target).abs() * weights).sum(), weights.sum()


def build_accumulation_groups(
    order: list[int], lengths: list[int], target_windows: int
) -> list[list[int]]:
    if target_windows < 1:
        raise ValueError("target_windows must be positive")
    groups, current, windows = [], [], 0
    for index in order:
        if lengths[index] < 1:
            raise ValueError("A training recording is empty")
        current.append(index)
        windows += lengths[index]
        if windows >= target_windows:
            groups.append(current)
            current, windows = [], 0
    if current:
        groups.append(current)
    return groups


def save_checkpoint(path, model, optimizer, scheduler, epoch, best, config, seed):
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


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(model, dataset, device, config) -> dict:
    """Measure full BPTT on median, p90, and longest recordings."""
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    lengths = np.asarray([dataset.lengths[r] for r in dataset.record_ids])
    order = np.argsort(lengths)
    selected = {
        "median": int(order[len(order) // 2]),
        "long_p90": int(order[int(0.9 * (len(order) - 1))]),
        "longest": int(order[-1]),
    }
    trial = copy.deepcopy(model).to(device).train()
    cfg = config["training"]
    optimizer = torch.optim.AdamW(
        trial.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    amp = bool(cfg["mixed_precision"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    rows = []
    try:
        for label, index in selected.items():
            record = dataset[index]
            x = record["windows"].unsqueeze(0).to(device)
            y = record["targets"].unsqueeze(0).to(device)
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            _sync(device)
            start = time.perf_counter()
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                prediction, _ = trial(x)
                numerator, denominator = weighted_l1_parts(prediction, y)
                loss = numerator / denominator
            _sync(device)
            forward = time.perf_counter() - start
            start = time.perf_counter()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trial.parameters(), cfg["gradient_clip"])
            _sync(device)
            backward = time.perf_counter() - start
            start = time.perf_counter()
            scaler.step(optimizer)
            scaler.update()
            _sync(device)
            step = time.perf_counter() - start
            rows.append(
                {
                    "label": label,
                    "recording_id": int(record["recording_id"]),
                    "windows": len(record["targets"]),
                    "forward_seconds": forward,
                    "backward_seconds": backward,
                    "optimizer_step_seconds": step,
                    "total_seconds": forward + backward + step,
                    "peak_gpu_memory_mb": (
                        torch.cuda.max_memory_allocated(device) / 2**20
                        if device.type == "cuda"
                        else 0.0
                    ),
                }
            )
    except torch.OutOfMemoryError as error:
        raise RuntimeError(
            "Full-recording BPTT is infeasible on this GPU. No automatic "
            "truncated-BPTT fallback was applied."
        ) from error
    finally:
        del trial, optimizer, scaler
        if device.type == "cuda":
            torch.cuda.empty_cache()
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    seconds_per_window = np.mean(
        [row["total_seconds"] / row["windows"] for row in rows]
    )
    estimated_epoch = float(seconds_per_window * lengths.sum())
    return {
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "total_training_recordings": len(dataset),
        "total_training_windows": int(lengths.sum()),
        "longest_recording_length": int(lengths.max()),
        "median_recording_length": float(np.median(lengths)),
        "recording_benchmarks": rows,
        "peak_gpu_memory_mb": max(row["peak_gpu_memory_mb"] for row in rows),
        "estimated_training_seconds_per_epoch": estimated_epoch,
        "estimated_total_training_seconds": estimated_epoch * cfg["max_epochs"],
    }


def validate_split(config, metadata, scaler_windows):
    expected = config["data"]
    checks = (
        ("n_train_records", "expected_train_records"),
        ("n_val_records", "expected_validation_records"),
    )
    for actual_key, expected_key in checks:
        if metadata[actual_key] != expected[expected_key]:
            raise RuntimeError(
                f"Expected {expected_key}={expected[expected_key]}, "
                f"got {metadata[actual_key]}"
            )
    if scaler_windows != expected["expected_train_windows"]:
        raise RuntimeError(
            f"Expected {expected['expected_train_windows']} training windows, "
            f"got {scaler_windows}"
        )


def evaluation_diagnostics(evaluation) -> dict:
    predictions = np.asarray([row["predicted_seconds"] for row in evaluation.predictions])
    errors = np.asarray([row["absolute_error"] for row in evaluation.predictions])
    lengths = {
        row["recording_id"]: row["number_of_windows"]
        for row in evaluation.per_recording
    }
    early = np.asarray(
        [
            row["window_index"] < lengths[row["recording_id"]] / 2
            for row in evaluation.predictions
        ]
    )
    worst = max(evaluation.per_recording, key=lambda row: row["recording_wbmae"])
    return {
        "validation_prediction_mean": float(predictions.mean()),
        "validation_prediction_std": float(predictions.std()),
        "validation_early_half_mae": float(errors[early].mean()),
        "validation_late_half_mae": float(errors[~early].mean()),
        "worst_recording_id": worst["recording_id"],
        "worst_recording_wbmae": worst["recording_wbmae"],
    }


def train(args):
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.epochs is not None:
        config["training"]["max_epochs"] = args.epochs
    cfg, seed = config["training"], int(config["experiment"]["seed"])
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(args.output_dir)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    logger = ExperimentLogger(output, config_path, config)

    chunks, train_windows, val_windows, _test, metadata = load_official_datasets(
        args.data_dir,
        sequence_length=config["data"]["sequence_length"],
        batch_size=cfg["batch_size"],
        seed=seed,
        download=args.download,
    )
    full_data = FullRecordingDataset(train_windows)
    model = DualPathSleepNet(**model_config(config)).to(device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if parameters != EXPECTED_PARAMETERS:
        raise RuntimeError(f"Expected {EXPECTED_PARAMETERS:,} parameters, got {parameters:,}")
    logger.message(f"Device: {device}; fresh V3 parameters: {parameters:,}")

    scaler_loader = DataLoader(
        chunks, batch_size=cfg["batch_size"], shuffle=False, num_workers=0
    )
    mean, std, scaler_windows = fit_feature_scaler(
        model.features, scaler_loader, device
    )
    validate_split(config, metadata, scaler_windows)
    torch.save(
        {"mean": mean, "std": std, "count": scaler_windows},
        logger.logs / "feature_scaler.pt",
    )
    logger.message(f"Recomputed scaler from {scaler_windows} training windows.")

    measured = benchmark(model, full_data, device, config)
    logger.save_benchmark(measured)
    for label, value in (
        ("GPU name", measured["gpu_name"]),
        ("Total training recordings", measured["total_training_recordings"]),
        ("Total training windows", measured["total_training_windows"]),
        ("Longest recording length", measured["longest_recording_length"]),
        ("Median recording length", measured["median_recording_length"]),
        ("Forward/backward benchmark", json.dumps(measured["recording_benchmarks"])),
        ("Peak GPU memory", f"{measured['peak_gpu_memory_mb']:.1f} MB"),
        (
            "Estimated time per epoch",
            f"{measured['estimated_training_seconds_per_epoch'] / 60:.2f} minutes",
        ),
        (
            "Estimated total training duration",
            f"{measured['estimated_total_training_seconds'] / 3600:.2f} hours",
        ),
    ):
        logger.message(f"{label}: {value}")
    if args.benchmark_only:
        return
    logger.message(
        "Mixed precision remains enabled for the CNN. The GRU, regression "
        "head, and loss run in fp32 because full-recording backpropagation "
        "overflowed fp16 gradients."
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["max_epochs"]
    )
    amp = bool(cfg["mixed_precision"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    diagnostic = full_data[0]["windows"][:24].unsqueeze(0)
    lengths = [full_data.lengths[r] for r in full_data.record_ids]
    started = time.monotonic()
    deadline = started + cfg["time_budget_hours"] * 3600
    best, stale, peak = float("inf"), 0, 0.0
    stop_reason, budget_reached = "maximum epochs", False
    best_path, last_path = (
        checkpoints / "best_model.pt",
        checkpoints / "last_model.pt",
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    consecutive_nonfinite = 0

    for epoch in range(1, cfg["max_epochs"] + 1):
        epoch_start = time.monotonic()
        order = torch.randperm(
            len(full_data), generator=torch.Generator().manual_seed(seed + epoch)
        ).tolist()
        groups = build_accumulation_groups(
            order, lengths, cfg["target_windows_per_step"]
        )
        model.train()
        total_num = total_den = absolute_sum = 0.0
        total_windows, grad_norms, step_windows, step_rows = 0, [], [], []
        component_grads = {}
        skipped_nonfinite = 0
        for step_index, group in enumerate(groups, 1):
            records = [full_data[index] for index in group]
            n_windows = sum(len(record["targets"]) for record in records)
            denominator = sum(
                float(target_weights(record["targets"]).sum()) for record in records
            )
            optimizer.zero_grad(set_to_none=True)
            numerator_value = absolute_value = 0.0
            before = (
                torch.cuda.memory_allocated(device) / 2**20
                if device.type == "cuda"
                else 0.0
            )
            for record in records:
                x = record["windows"].unsqueeze(0).to(device)
                y = record["targets"].unsqueeze(0).to(device)
                with torch.amp.autocast(device_type=device.type, enabled=amp):
                    prediction, _state = model(x)
                prediction = prediction.float()
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError("Non-finite training prediction")
                numerator, _ = weighted_l1_parts(prediction, y)
                contribution = numerator / denominator
                scaler.scale(contribution).backward()
                numerator_value += float(numerator.detach())
                absolute_value += float((prediction.detach() - y).abs().sum())
                del x, y, prediction, numerator, contribution, _state
            scaler.unscale_(optimizer)
            finite = all(
                p.grad is None or bool(torch.isfinite(p.grad).all())
                for p in model.parameters()
            )
            if finite:
                component_grads = component_gradient_norms(model)
                grad = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["gradient_clip"]
                )
                consecutive_nonfinite = 0
            else:
                grad = float("inf")
                skipped_nonfinite += 1
                consecutive_nonfinite += 1
                names = [
                    name
                    for name, parameter in model.named_parameters()
                    if parameter.grad is not None
                    and not torch.isfinite(parameter.grad).all()
                ]
                logger.message(
                    f"Skipped non-finite gradient at epoch {epoch}, "
                    f"step {step_index}; GradScaler scale is now reduced. "
                    f"Parameters: {', '.join(names[:8])}"
                )
                if consecutive_nonfinite >= 30:
                    raise FloatingPointError(
                        "Gradients stayed non-finite for 30 optimizer steps. "
                        "Full-recording training is numerically unstable."
                    )
            # Always call step/update: non-finite gradients are skipped and the
            # loss scale is reduced, which is the normal mixed-precision behavior.
            scaler.step(optimizer)
            scaler.update()
            after = (
                torch.cuda.memory_allocated(device) / 2**20
                if device.type == "cuda"
                else 0.0
            )
            current_peak = (
                torch.cuda.max_memory_allocated(device) / 2**20
                if device.type == "cuda"
                else 0.0
            )
            peak = max(peak, current_peak)
            total_num += numerator_value
            total_den += denominator
            absolute_sum += absolute_value
            total_windows += n_windows
            grad_norms.append(float(grad))
            step_windows.append(n_windows)
            step_rows.append(
                {
                    "epoch": epoch,
                    "optimizer_step": step_index,
                    "recordings": len(group),
                    "windows": n_windows,
                    "weighted_l1": numerator_value / denominator,
                    "gradient_norm": float(grad),
                    "gradients_finite": finite,
                    "allocated_before_mb": before,
                    "allocated_after_mb": after,
                    "memory_increase_mb": after - before,
                    "unexpected_memory_growth": step_index > 1 and after - before > 50,
                    "peak_allocated_mb": current_peak,
                    "nan_inf_count": 0,
                }
            )
            if time.monotonic() >= deadline:
                budget_reached = True
                stop_reason = "four-hour wall-clock training budget"
                break
        logger.log_optimization(step_rows)
        training_seconds = time.monotonic() - epoch_start
        validation_start = time.monotonic()
        evaluation = evaluate_streaming(model, val_windows.iter_records(), device)
        validation_seconds = time.monotonic() - validation_start
        scheduler.step()
        feature_row = {
            **feature_snapshot(model, diagnostic, epoch),
            **component_grads,
        }
        row = {
            "epoch": epoch,
            "train_loss": total_num / total_den,
            "train_mae": absolute_sum / total_windows,
            "validation_wbmae": evaluation.summary["wbmae"],
            "validation_bmae": evaluation.summary["bmae"],
            "validation_plain_mae": evaluation.summary["mae"],
            **{
                f"validation_bin_{name}_mae": evaluation.summary[f"bin_{name}_mae"]
                for name in BIN_NAMES
            },
            **{
                f"validation_bin_{name}_bias": evaluation.summary[f"bin_{name}_bias"]
                for name in BIN_NAMES
            },
            **evaluation_diagnostics(evaluation),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "gradient_norm": float(
                np.mean([value for value in grad_norms if np.isfinite(value)])
                if any(np.isfinite(value) for value in grad_norms)
                else float("inf")
            ),
            "optimizer_steps": len(step_windows),
            "skipped_nonfinite_steps": skipped_nonfinite,
            "mean_windows_per_optimizer_step": float(np.mean(step_windows)),
            "training_seconds": training_seconds,
            "validation_seconds": validation_seconds,
            "peak_gpu_memory_mb": peak,
            "train_windows": total_windows,
            "validation_windows": evaluation.summary["n_windows"],
            "validation_recordings": evaluation.summary["n_recordings"],
            "gradients_finite": skipped_nonfinite == 0,
            "nan_inf_count": feature_row["nan_inf_count"],
        }
        logger.log_epoch(row, feature_row)
        logger.message(
            f"Epoch {epoch:02d} | loss {row['train_loss']:.4f} | "
            f"val W-bMAE {row['validation_wbmae']:.4f}"
        )
        if row["validation_wbmae"] < best:
            best, stale = row["validation_wbmae"], 0
            save_checkpoint(
                best_path, model, optimizer, scheduler, epoch, best, config, seed
            )
        else:
            stale += 1
        save_checkpoint(last_path, model, optimizer, scheduler, epoch, best, config, seed)
        if budget_reached:
            break
        if stale >= cfg["early_stopping_patience"]:
            stop_reason = f"early stopping after {stale} non-improving epochs"
            break

    state = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    final = evaluate_streaming(model, val_windows.iter_records(), device)
    logger.save_final(
        final,
        model,
        {
            "seed": seed,
            "parameters": parameters,
            "scaler_windows": scaler_windows,
            "training_runtime_seconds": time.monotonic() - started,
            "device": str(device),
            "stop_reason": stop_reason,
            "peak_gpu_memory_mb": peak,
        },
    )
    package_submission(best_path, Path(__file__).with_name("submission.py"), output / "submission.zip")
    logger.archive(output / "v3_diagnostics.zip")
    logger.message(f"Best validation W-bMAE: {best:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.yaml")
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--benchmark-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
