"""Diagnostics and reporting for V3-FullRecording."""

from __future__ import annotations

import json
import platform
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from evaluate import BIN_NAMES, EvaluationResult


def feature_snapshot(model, windows: torch.Tensor, epoch: int) -> dict:
    model.eval()
    with torch.inference_mode():
        flat = windows[:2].flatten(0, 1).to(next(model.parameters()).device)
        _fused, values = model.window_embeddings(flat, return_diagnostics=True)
    result = {"epoch": epoch, "nan_inf_count": 0}
    for name, tensor in values.items():
        result["nan_inf_count"] += int((~torch.isfinite(tensor)).sum())
        safe = torch.nan_to_num(tensor)
        result[f"{name}_mean"] = float(safe.mean())
        result[f"{name}_std"] = float(safe.std())
        result[f"{name}_norm"] = float(safe.norm(dim=1).mean())
    result["correction_to_cnn_ratio"] = result["correction_norm"] / max(
        result["cnn_norm"], 1e-8
    )
    model.train()
    return result


def component_gradient_norms(model) -> dict[str, float]:
    groups = {
        "cnn": list(model.cnn_branches.parameters()) + list(model.cnn_fusion.parameters()),
        "spectral_mlp": list(model.spectral_mlp.parameters()),
        "spectral_projection": list(model.spectral_projection.parameters()),
        "gru": list(model.gru.parameters()),
        "head": list(model.head.parameters()),
    }
    return {
        f"gradient_norm_{name}": float(
            sum(
                parameter.grad.detach().square().sum()
                for parameter in parameters
                if parameter.grad is not None
            )
            .sqrt()
            .cpu()
        )
        for name, parameters in groups.items()
    }


class ExperimentLogger:
    def __init__(self, root: Path, config_path: Path, config: dict) -> None:
        self.root = root
        self.logs = root / "logs"
        self.plots = root / "plots"
        self.trajectories = self.plots / "prediction_trajectories"
        for folder in (self.logs, self.plots, self.trajectories):
            folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(config_path, root / "config.yaml")
        self.config = config
        self.history: list[dict] = []
        self.optimization: list[dict] = []
        self.features: list[dict] = []
        self.training_log = self.logs / "training_log.txt"

    def message(self, text: str) -> None:
        print(text, flush=True)
        with self.training_log.open("a", encoding="utf-8") as stream:
            stream.write(text + "\n")

    def save_benchmark(self, value: dict) -> None:
        (self.logs / "runtime_benchmark.json").write_text(
            json.dumps(value, indent=2), encoding="utf-8"
        )

    def log_optimization(self, rows: list[dict]) -> None:
        self.optimization.extend(rows)
        pd.DataFrame(self.optimization).to_csv(
            self.logs / "optimization_diagnostics.csv", index=False
        )

    def log_epoch(self, row: dict, feature_row: dict) -> None:
        self.history.append(row)
        self.features.append(feature_row)
        pd.DataFrame(self.history).to_csv(self.logs / "epoch_metrics.csv", index=False)
        (self.logs / "feature_diagnostics.json").write_text(
            json.dumps(self.features, indent=2), encoding="utf-8"
        )

    def _curve(self, columns: list[str], title: str, filename: str) -> None:
        frame = pd.DataFrame(self.history)
        fig, axis = plt.subplots(figsize=(7, 4))
        for column in columns:
            axis.plot(frame.epoch, frame[column], marker="o", label=column)
        axis.set(title=title, xlabel="Epoch", ylabel="Seconds")
        axis.grid(alpha=0.25)
        if len(columns) > 1:
            axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(self.plots / filename, dpi=140)
        plt.close(fig)

    def _plots(self, predictions: pd.DataFrame, recordings: pd.DataFrame) -> None:
        self._curve(["train_loss"], "Training weighted L1", "training_loss.png")
        self._curve(["validation_wbmae"], "Validation W-bMAE", "validation_wbmae.png")
        self._curve(
            [f"validation_bin_{name}_mae" for name in BIN_NAMES],
            "Validation MAE by target bin",
            "per_bin_mae.png",
        )
        for record_id in recordings.nlargest(10, "recording_wbmae").recording_id:
            frame = predictions[predictions.recording_id == record_id]
            fig, axis = plt.subplots(figsize=(8, 4))
            axis.plot(frame.window_end_seconds, frame.target_seconds, label="Target")
            axis.plot(frame.window_end_seconds, frame.predicted_seconds, label="Prediction")
            axis.set(xlabel="Elapsed seconds", ylabel="Seconds to N2")
            axis.legend()
            fig.tight_layout()
            fig.savefig(self.trajectories / f"recording_{record_id}.png", dpi=120)
            plt.close(fig)

    def save_final(
        self, result: EvaluationResult, model: torch.nn.Module, run: dict
    ) -> None:
        predictions = pd.DataFrame(result.predictions)
        recordings = pd.DataFrame(result.per_recording)
        predictions.to_csv(self.logs / "validation_predictions.csv", index=False)
        recordings.to_csv(self.logs / "per_recording_metrics.csv", index=False)
        self._plots(predictions, recordings)
        (self.root / "model_summary.txt").write_text(
            f"{model}\n\nTrainable parameters: {run['parameters']:,}\n",
            encoding="utf-8",
        )
        (self.root / "environment.txt").write_text(
            "\n".join(
                [
                    f"python: {sys.version}",
                    f"torch: {torch.__version__}",
                    f"platform: {platform.platform()}",
                    f"cuda: {torch.version.cuda}",
                    "gpu: "
                    + (
                        torch.cuda.get_device_name(0)
                        if torch.cuda.is_available()
                        else "None"
                    ),
                ]
            ),
            encoding="utf-8",
        )
        self._summary(result, predictions, recordings, run)

    def _summary(
        self,
        result: EvaluationResult,
        predictions: pd.DataFrame,
        recordings: pd.DataFrame,
        run: dict,
    ) -> None:
        refs = self.config["references"]
        best = min(self.history, key=lambda row: row["validation_wbmae"])
        final = self.history[-1]
        positions = predictions.groupby("recording_id").cumcount()
        lengths = predictions.groupby("recording_id")["recording_id"].transform("size")
        early = positions < lengths / 2
        early_mae = float(predictions.loc[early, "absolute_error"].mean())
        late_mae = float(predictions.loc[~early, "absolute_error"].mean())
        improved_bins = [
            name
            for name in BIN_NAMES
            if result.summary[f"bin_{name}_mae"] < refs[f"v2_bin_{name}_mae"]
        ]
        worth = (
            "Yes"
            if best["validation_wbmae"] < refs["v2_best_validation_wbmae"]
            and result.summary["bin_300_600_mae"]
            <= 1.1 * refs["v2_bin_300_600_mae"]
            else "Not yet"
        )
        metric_rows = "\n".join(
            f"| {name} MAE | {refs[f'v2_bin_{name}_mae']:.3f} | "
            f"{result.summary[f'bin_{name}_mae']:.3f} |"
            for name in BIN_NAMES
        )
        bias_rows = "\n".join(
            f"- {name}: V2 {refs[f'v2_bin_{name}_bias']:+.3f}; "
            f"V3 {result.summary[f'bin_{name}_bias']:+.3f} s"
            for name in BIN_NAMES
        )
        text = f"""# V3-FullRecording experiment summary

## Controlled change
- Complete-recording BPTT replaces V2's independent 24-window training chunks.
- Architecture, seed, preprocessing, scaler mathematics, weighted L1, split,
  optimizer, scheduler, dropout, validation, and submission interface remain unchanged.
- Numerical exception: the CNN uses mixed precision, while the GRU, regression
  head, and loss use fp32. Full-recording fp16 backpropagation overflowed.
- Scaler was recomputed from {run['scaler_windows']:,} training-only windows;
  no V2 checkpoint or scaler was loaded.
- Parameters: {run['parameters']:,} (V2: 143,777)

## Runtime
- Date: {datetime.now(timezone.utc).isoformat()}
- Device: {run['device']}
- Runtime: {run['training_runtime_seconds'] / 60:.2f} minutes
- Peak GPU memory: {run['peak_gpu_memory_mb']:.1f} MB
- Stop reason: {run['stop_reason']}
- Best/final epoch: {best['epoch']} / {final['epoch']}
- Optimizer steps at best epoch: {best['optimizer_steps']}
- Mean windows/step at best epoch: {best['mean_windows_per_optimizer_step']:.2f}

## V2 versus V3
| Metric | V2 | V3 |
|---|---:|---:|
| Best validation W-bMAE | {refs['v2_best_validation_wbmae']:.4f} | {best['validation_wbmae']:.4f} |
| Validation bMAE | {refs['v2_validation_bmae']:.4f} | {result.summary['bmae']:.4f} |
| Validation plain MAE | {refs['v2_validation_plain_mae']:.4f} | {result.summary['mae']:.4f} |
{metric_rows}
| Training duration (min) | {refs['v2_training_duration_minutes']:.2f} | {run['training_runtime_seconds'] / 60:.2f} |
| Peak GPU memory (MB) | {refs['v2_peak_gpu_memory_mb']:.1f} | {run['peak_gpu_memory_mb']:.1f} |

- Early/late MAE: {early_mae:.3f} / {late_mae:.3f} s
- Improved bins: {', '.join(improved_bins) if improved_bins else 'none'}
- Prediction mean/std: {predictions.predicted_seconds.mean():.3f} /
  {predictions.predicted_seconds.std(ddof=0):.3f} s
- Worst recording: {int(recordings.loc[recordings.recording_wbmae.idxmax(), 'recording_id'])}

## Bias
{bias_rows}

## Decision
- Worth submitting: {worth}.
- A single seed is not statistical proof. Indivisible recordings also change
  optimizer-step count, which complicates attribution.
"""
        (self.root / "experiment_summary.md").write_text(text, encoding="utf-8")

    def archive(self, output: Path) -> None:
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(self.root.rglob("*")):
                if path.is_file() and path != output:
                    archive.write(
                        path, Path("v3_full_recording") / path.relative_to(self.root)
                    )
