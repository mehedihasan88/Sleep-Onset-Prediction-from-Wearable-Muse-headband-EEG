"""Official NeuralBench Muse loader and chronological sequence datasets."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

TASK = "_sleep_onset_stream"
STUDY = "Interaxon2026Muse"


def _official_config(data_dir: Path, batch_size: int, seed: int) -> Any:
    """Build the same merged data config as the official competition adapter."""
    from neuralbench.experiment_config import merge_task_config

    cfg = merge_task_config("eeg", TASK, None)["data"]
    cfg.update(
        {
            "study.source.path": str(data_dir),
            "study.source.infra.folder": str(data_dir / "cache"),
            "neuro.infra.cluster": None,
            "neuro.infra.folder": str(data_dir / "cache"),
            "batch_size": batch_size,
            "seed": seed,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        }
    )
    return cfg


def download_official_data(data_dir: str | Path) -> None:
    """Download NEMAR nm000287 through the official NeuralSet source."""
    import neuralset as ns

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg = _official_config(data_dir, batch_size=16, seed=33)
    source = dict(cfg["study"]["source"])
    source["path"] = data_dir / source["name"]
    print(f"Downloading/preparing {STUDY} under {data_dir} ...")
    ns.Study(**source).download()


class OfficialWindowDataset(Dataset):
    """Expose a NeuralBench SegmentDataset as plain tensors plus recording IDs."""

    def __init__(self, segment_dataset: Any) -> None:
        self.segment_dataset = segment_dataset
        records: dict[Any, int] = {}
        self.record_ids = np.asarray(
            [
                records.setdefault(segment.timeline, len(records))
                for segment in segment_dataset.segments
            ],
            dtype=np.int64,
        )
        self.onsets = np.asarray(
            [float(segment.start) for segment in segment_dataset.segments],
            dtype=np.float64,
        )
        self.by_record: dict[int, list[int]] = defaultdict(list)
        for index, record_id in enumerate(self.record_ids):
            self.by_record[int(record_id)].append(index)
        for indices in self.by_record.values():
            indices.sort(key=self.onsets.__getitem__)

    def __len__(self) -> int:
        return len(self.segment_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.segment_dataset[int(index)].data
        x = torch.as_tensor(np.asarray(data["neuro"]), dtype=torch.float32)
        if x.ndim == 3 and x.shape[0] == 1:
            x = x[0]
        y = torch.as_tensor(np.asarray(data["target"]), dtype=torch.float32)
        y = y.reshape(-1)[0]
        if x.ndim != 2:
            raise ValueError(f"Expected a (C, T) EEG window, got {tuple(x.shape)}")
        return x, y

    def iter_records(self) -> Iterator[tuple[int, torch.Tensor, torch.Tensor]]:
        for record_id, indices in sorted(self.by_record.items()):
            windows, labels = zip(*(self[index] for index in indices))
            yield record_id, torch.stack(windows), torch.stack(labels)


class ChronologicalSequenceDataset(Dataset):
    """Fixed-length, within-recording sequences; never crosses a boundary."""

    def __init__(
        self, windows: OfficialWindowDataset, sequence_length: int = 12
    ) -> None:
        self.windows = windows
        self.sequence_length = sequence_length
        self.sequences: list[list[int]] = []
        for indices in windows.by_record.values():
            for start in range(0, len(indices), sequence_length):
                self.sequences.append(indices[start : start + sequence_length])

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = self.sequences[index]
        windows, targets = zip(*(self.windows[i] for i in indices))
        x = torch.stack(windows)
        y = torch.stack(targets)
        valid = torch.ones(len(indices), dtype=torch.bool)
        padding = self.sequence_length - len(indices)
        if padding:
            x = torch.cat([x, x[-1:].expand(padding, -1, -1)], dim=0)
            y = torch.cat([y, y[-1:].expand(padding)], dim=0)
            valid = torch.cat([valid, torch.zeros(padding, dtype=torch.bool)])
        return x, y, valid


def load_official_datasets(
    data_dir: str | Path,
    sequence_length: int = 12,
    batch_size: int = 16,
    seed: int = 33,
    download: bool = False,
) -> tuple[
    ChronologicalSequenceDataset,
    OfficialWindowDataset,
    OfficialWindowDataset,
    dict[str, Any],
]:
    """Prepare/load official train, validation and public-test splits."""
    from neuralbench.data import Data

    logging.getLogger("neuralset").setLevel(logging.WARNING)
    logging.getLogger("exca").setLevel(logging.WARNING)
    data_dir = Path(data_dir)
    if download:
        download_official_data(data_dir)
    if not (data_dir / STUDY).exists():
        raise FileNotFoundError(
            f"{STUDY} is absent under {data_dir}. Re-run with --download."
        )

    print("Loading the official NeuralBench task (first run builds its cache) ...")
    loaders = Data(**_official_config(data_dir, batch_size, seed)).prepare()
    train_windows = OfficialWindowDataset(loaders["train"].dataset)
    val_windows = OfficialWindowDataset(loaders["val"].dataset)
    test_windows = OfficialWindowDataset(loaders["test"].dataset)
    x0, _ = train_windows[0]
    metadata = {
        "n_chans": int(x0.shape[0]),
        "n_times": int(x0.shape[1]),
        "sfreq": float(x0.shape[1] / 5.0),
        "n_train_records": len(train_windows.by_record),
        "n_val_records": len(val_windows.by_record),
        "n_test_records": len(test_windows.by_record),
    }
    return (
        ChronologicalSequenceDataset(train_windows, sequence_length),
        val_windows,
        test_windows,
        metadata,
    )
