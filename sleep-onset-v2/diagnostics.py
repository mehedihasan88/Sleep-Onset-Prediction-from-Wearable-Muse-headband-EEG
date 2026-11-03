"""Compact but comprehensive experiment logging for V2."""

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
        finite = torch.isfinite(tensor)
        result["nan_inf_count"] += int((~finite).sum())
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


class DiagnosticLogger:
    def __init__(self, output_root: Path, config_path: Path) -> None:
        self.root = Path(output_root)
        self.logs = self.root / "logs"
        self.plots = self.root / "plots"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.plots.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(config_path, self.root / "config.yaml")
        self.history: list[dict] = []
        self.feature_history: list[dict] = []
        self.training_log = self.logs / "training_log.txt"

    def message(self, text: str) -> None:
        print(text, flush=True)
        with self.training_log.open("a", encoding="utf-8") as stream:
            stream.write(text + "\n")

    def log_epoch(self, row: dict, feature_row: dict) -> None:
        self.history.append(row)
        self.feature_history.append(feature_row)
        pd.DataFrame(self.history).to_csv(self.logs / "epoch_metrics.csv", index=False)
        (self.logs / "feature_diagnostics.json").write_text(
            json.dumps(self.feature_history, indent=2), encoding="utf-8"
        )

    def _curve(self, columns, title, ylabel, filename) -> None:
        frame = pd.DataFrame(self.history)
        fig, axis = plt.subplots(figsize=(7, 4))
        for column in columns:
            if column in frame:
                axis.plot(frame["epoch"], frame[column], marker="o", label=column)
        axis.set(title=title, xlabel="Epoch", ylabel=ylabel)
        axis.grid(alpha=0.25)
        if len(columns) > 1:
            axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(self.plots / filename, dpi=140)
        plt.close(fig)

    def save_curves(self) -> None:
        self._curve(["train_loss"], "Training loss", "Weighted L1", "training_loss.png")
        self._curve(
            ["validation_wbmae"], "Validation W-bMAE", "Seconds", "validation_wbmae.png"
        )
        self._curve(
            ["train_mae", "validation_plain_mae"],
            "Training and validation MAE",
            "Seconds",
            "train_validation_mae.png",
        )
        self._curve(
            [f"validation_bin_{name}_mae" for name in BIN_NAMES],
            "Validation MAE by target bin",
            "Seconds",
            "per_bin_mae.png",
        )
        self._curve(["gradient_norm_mean"], "Gradient norm", "L2 norm", "gradient_norm.png")
        self._curve(["learning_rate"], "Learning rate", "Learning rate", "learning_rate.png")

    def save_final(
        self,
        result: EvaluationResult,
        model: torch.nn.Module,
        summary: dict,
        elapsed_diagnostic: dict,
    ) -> None:
        predictions = pd.DataFrame(result.predictions)
        recordings = pd.DataFrame(result.per_recording)
        predictions.to_csv(self.logs / "validation_predictions.csv", index=False)
        recordings.to_csv(self.logs / "per_recording_metrics.csv", index=False)
        participant = (
            recordings.groupby("participant_id")
            .agg(
                number_of_recordings=("recording_id", "size"),
                number_of_windows=("number_of_windows", "sum"),
                mean_recording_wbmae=("recording_wbmae", "mean"),
                mean_recording_plain_mae=("recording_plain_mae", "mean"),
            )
            .reset_index()
        )
        participant.to_csv(self.logs / "per_participant_metrics.csv", index=False)
        (self.logs / "elapsed_time_diagnostic.json").write_text(
            json.dumps(elapsed_diagnostic, indent=2), encoding="utf-8"
        )
        self._prediction_distribution(predictions)
        self._recording_plots(predictions, recordings)
        self.save_curves()
        model_text = (
            f"{model}\n\nTrainable parameters: "
            f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n"
        )
        (self.root / "model_summary.txt").write_text(model_text, encoding="utf-8")
        environment = {
            "python": sys.version,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        (self.root / "environment.txt").write_text(
            "\n".join(f"{key}: {value}" for key, value in environment.items()),
            encoding="utf-8",
        )
        self._summary(summary, result, elapsed_diagnostic)

    def _prediction_distribution(self, frame: pd.DataFrame) -> None:
        fig, axis = plt.subplots(figsize=(7, 4))
        bins = np.linspace(0, 600, 31)
        axis.hist(frame.target_seconds, bins=bins, alpha=0.55, label="Target")
        axis.hist(frame.predicted_seconds, bins=bins, alpha=0.55, label="Prediction")
        axis.set(xlabel="Seconds", ylabel="Windows", title="Prediction distribution")
        axis.legend()
        fig.tight_layout()
        fig.savefig(self.plots / "prediction_distribution.png", dpi=140)
        plt.close(fig)

    def _recording_plots(self, predictions: pd.DataFrame, recordings: pd.DataFrame) -> None:
        for kind, selected in (
            ("worst_recordings", recordings.nlargest(10, "recording_wbmae")),
            ("best_recordings", recordings.nsmallest(5, "recording_wbmae")),
        ):
            folder = self.plots / kind
            folder.mkdir(exist_ok=True)
            for record_id in selected.recording_id:
                frame = predictions[predictions.recording_id == record_id]
                fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
                axes[0].plot(frame.window_end_seconds, frame.target_seconds, label="Target")
                axes[0].plot(
                    frame.window_end_seconds, frame.predicted_seconds, label="Prediction"
                )
                axes[0].axvline(
                    frame.window_end_seconds.max(),
                    color="black",
                    linestyle="--",
                    alpha=0.5,
                    label="First N2",
                )
                axes[0].legend()
                axes[0].set(ylabel="Seconds remaining")
                axes[1].plot(
                    frame.window_end_seconds, frame.absolute_error, color="tab:red"
                )
                axes[1].set(xlabel="Elapsed recording seconds", ylabel="Absolute error")
                fig.suptitle(f"Recording {record_id}")
                fig.tight_layout()
                fig.savefig(folder / f"recording_{record_id}.png", dpi=120)
                plt.close(fig)

    def _summary(self, summary: dict, result: EvaluationResult, elapsed: dict) -> None:
        best = min(self.history, key=lambda row: row["validation_wbmae"])
        final = self.history[-1]
        feature = self.feature_history[best["epoch"] - 1]
        worst = max(result.per_recording, key=lambda row: row["recording_wbmae"])
        text = f"""# V2 experiment summary

## Experiment identity
- Model: Dual-Path Spectral-Temporal CNN-GRU
- Date: {datetime.now(timezone.utc).isoformat()}
- Seed: {summary['seed']}
- Split: official Muse 412 train / 88 validation / 40 public test
- Trainable parameters: {summary['parameters']:,}
- Training runtime: {summary['training_runtime_seconds'] / 60:.2f} minutes
- Device: {summary['device']}
- Python: {platform.python_version()}
- PyTorch: {torch.__version__}

## Training outcomes
- Best epoch: {best['epoch']}
- Final epoch: {final['epoch']}
- Best validation W-bMAE: {best['validation_wbmae']:.4f} s
- Final validation W-bMAE: {final['validation_wbmae']:.4f} s
- Training loss at best epoch: {best['train_loss']:.4f}
- Final training loss: {final['train_loss']:.4f}
- Stop reason: {summary['stop_reason']}
- Peak GPU memory: {summary['peak_gpu_memory_mb']:.1f} MB

## Best-checkpoint error analysis
- Plain MAE: {result.summary['mae']:.4f} s
- Unweighted bMAE: {result.summary['bmae']:.4f} s
- Bin MAEs: {', '.join(f"{name}={result.summary[f'bin_{name}_mae']:.3f}" for name in BIN_NAMES)}
- Bin biases: {', '.join(f"{name}={result.summary[f'bin_{name}_bias']:.3f}" for name in BIN_NAMES)}
- Worst recording: {worst['recording_id']} ({worst['number_of_windows']} windows,
  W-bMAE {worst['recording_wbmae']:.3f} s)
- Prediction standard deviation: {elapsed['prediction_std']:.3f} s
- Early/late MAE: {elapsed['model_early_half_mae']:.3f} /
  {elapsed['model_late_half_mae']:.3f} s
- Recording-length/W-bMAE correlation: {elapsed['recording_length_wbmae_correlation']:.3f}

## Feature fusion
- CNN norm: {feature['cnn_norm']:.4f}
- Spectral embedding norm: {feature['spectral_norm']:.4f}
- Spectral correction norm: {feature['correction_norm']:.4f}
- Correction/CNN ratio: {feature['correction_to_cnn_ratio']:.4f}
- NaN/Inf count: {feature['nan_inf_count']}
- Spectral MLP gradient norm: {feature.get('gradient_norm_spectral_mlp', float('nan')):.6f}
- Spectral projection gradient norm: {feature.get('gradient_norm_spectral_projection', float('nan')):.6f}

## Elapsed-time-only diagnostic
- Train-fitted rule: y = {elapsed['intercept']:.4f} + {elapsed['slope']:.4f} * elapsed_seconds
- Validation W-bMAE: {elapsed['validation_wbmae']:.4f} s
- This diagnostic is not the submitted model.

## V1 comparison
- V1 validation W-bMAE: 43.0692 s
- V1 Codabench W-bMAE: 55.0472 s
- V2 values must only be compared on the same validation or Codabench split.

## Generalization limitations
- This is one seed on the official development split.
- The sealed cohort includes unseen sleepers and may shift substantially.
- Training backpropagates through 120-second chunks while inference retains
  state across a full recording.
"""
        (self.root / "experiment_summary.md").write_text(text, encoding="utf-8")
        ratio = feature["correction_to_cnn_ratio"]
        largest_bias_bin = max(
            BIN_NAMES, key=lambda name: abs(result.summary[f"bin_{name}_bias"])
        )
        fusion_advice = (
            "Path B is weak relative to the CNN; investigate spectral scaling or fusion."
            if ratio < 0.1
            else "Path B dominates the CNN; investigate projection regularization."
            if ratio > 1.0
            else "Path B has a non-trivial, non-dominant contribution; test its value with a future ablation."
        )
        conclusions = f"""# Diagnostic conclusions

1. {fusion_advice}
2. The largest absolute prediction bias is in `{largest_bias_bin}`; prioritize
   calibration or sampling changes for that measured failure mode.
3. Compare the elapsed-only W-bMAE ({elapsed['validation_wbmae']:.3f}) with
   V2 before changing temporal context. If they are close, reduce reliance on
   elapsed time using recording-start augmentation or varied stream starts.

These are evidence-driven proposals only; no V3 changes were applied.
"""
        (self.root / "diagnostic_conclusions.md").write_text(
            conclusions, encoding="utf-8"
        )

    def archive(self, output_zip: Path) -> None:
        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(self.root.rglob("*")):
                if path.is_file() and path != output_zip:
                    relative = path.relative_to(self.root)
                    if relative.parts[0] == "logs":
                        relative = Path(relative.name)
                    archive.write(path, Path("v2_diagnostics") / relative)
