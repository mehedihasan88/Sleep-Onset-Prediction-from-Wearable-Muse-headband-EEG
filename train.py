"""Train one CNN-GRU model on the official Track 3 data."""

from __future__ import annotations

import argparse
import json
import random
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import load_official_datasets
from evaluate import BIN_EDGES, BIN_WEIGHTS, evaluate_streaming
from model import SleepOnsetNet


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def target_weights(target: torch.Tensor) -> torch.Tensor:
    """Per-window severity weights matching the official target bins."""
    boundaries = torch.tensor(BIN_EDGES[1:-1], device=target.device)
    bins = torch.bucketize(target, boundaries, right=True)
    weights = torch.tensor(BIN_WEIGHTS, device=target.device)
    return weights[bins]


def weighted_l1(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    weights = target_weights(target)
    weights = weights * valid
    return ((prediction - target).abs() * weights).sum() / weights.sum().clamp_min(1)


def package_submission(
    checkpoint: Path, submission_source: Path, output_zip: Path
) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(submission_source, "submission.py")
        archive.write(checkpoint, "best_model.pt")
    print(f"Submission ZIP: {output_zip.resolve()}")


def train(args: argparse.Namespace) -> Path:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_data, val_data, _, metadata = load_official_datasets(
        args.data_dir,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        seed=args.seed,
        download=args.download,
    )
    print(json.dumps(metadata, indent=2))
    loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    model = SleepOnsetNet(
        n_chans=metadata["n_chans"],
        embedding_size=args.embedding_size,
        gru_hidden_size=args.gru_hidden_size,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-3
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    checkpoint = Path(args.output_dir) / "best_model.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_score = float("inf")
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches = 0
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.detach())
            batches += 1

        scores = evaluate_streaming(model, val_data.iter_records(), device)
        mean_loss = epoch_loss / max(batches, 1)
        print(
            f"Epoch {epoch:02d} | Training loss: {mean_loss:.4f} | "
            f"Validation W-bMAE: {scores['wbmae']:.4f}"
        )
        if scores["wbmae"] < best_score:
            best_score = scores["wbmae"]
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": {
                        "n_chans": metadata["n_chans"],
                        "embedding_size": args.embedding_size,
                        "gru_hidden_size": args.gru_hidden_size,
                    },
                    "validation_wbmae": best_score,
                    "epoch": epoch,
                },
                checkpoint,
            )
            print(f"Best checkpoint: {checkpoint.resolve()}")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    package_submission(
        checkpoint,
        Path(__file__).with_name("submission.py"),
        Path(args.output_dir).parent / "submission.zip",
    )
    return checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--embedding-size", type=int, default=128)
    parser.add_argument("--gru-hidden-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=33)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
