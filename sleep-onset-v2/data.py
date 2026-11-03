"""Official NeuralBench 0.4.0 Muse data adapter for V2."""

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
EXPECTED_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]


def _official_config(data_dir: Path, batch_size: int, seed: int) -> Any:
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


def download_official_data(data_dir: Path) -> None:
    import neuralset as ns

    data_dir.mkdir(parents=True, exist_ok=True)
    source = dict(_official_config(data_dir, 16, 42)["study"]["source"])
    source["path"] = data_dir / source["name"]
    ns.Study(**source).download()


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class OfficialWindowDataset(Dataset):
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
        self._participants: dict[int, int] = {}

    def __len__(self) -> int:
        return len(self.segment_dataset)

    def _read(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        data = self.segment_dataset[int(index)].data
        x = torch.as_tensor(_numpy(data["neuro"]), dtype=torch.float32)
        if x.ndim == 3 and x.shape[0] == 1:
            x = x[0]
        y = torch.as_tensor(_numpy(data["target"]), dtype=torch.float32).reshape(-1)[0]
        subject = data.get("subject_id", -1)
        participant = int(_numpy(subject).reshape(-1)[0]) if subject is not None else -1
        if x.shape != (4, 640):
            raise ValueError(f"Official Muse window must be (4,640), got {tuple(x.shape)}")
        return x, y, participant

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y, _ = self._read(index)
        return x, y

    def participant(self, record_id: int) -> int:
        if record_id not in self._participants:
            first = self.by_record[record_id][0]
            self._participants[record_id] = self._read(first)[2]
        return self._participants[record_id]

    def iter_records(
        self,
    ) -> Iterator[tuple[int, int, np.ndarray, torch.Tensor, torch.Tensor]]:
        for record_id, indices in sorted(self.by_record.items()):
            windows, labels = zip(*(self[index] for index in indices))
            yield (
                record_id,
                self.participant(record_id),
                self.onsets[indices].copy(),
                torch.stack(windows),
                torch.stack(labels),
            )


class ChronologicalSequenceDataset(Dataset):
    def __init__(self, windows: OfficialWindowDataset, sequence_length: int = 24):
        self.windows = windows
        self.sequence_length = sequence_length
        self.sequences: list[list[int]] = []
        for indices in windows.by_record.values():
            for start in range(0, len(indices), sequence_length):
                self.sequences.append(indices[start : start + sequence_length])

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int):
        indices = self.sequences[index]
        windows, targets = zip(*(self.windows[i] for i in indices))
        x, y = torch.stack(windows), torch.stack(targets)
        valid = torch.ones(len(indices), dtype=torch.bool)
        padding = self.sequence_length - len(indices)
        if padding:
            x = torch.cat([x, x[-1:].expand(padding, -1, -1)], 0)
            y = torch.cat([y, y[-1:].expand(padding)], 0)
            valid = torch.cat([valid, torch.zeros(padding, dtype=torch.bool)])
        return x, y, valid


def load_official_datasets(
    data_dir: str | Path,
    sequence_length: int = 24,
    batch_size: int = 16,
    seed: int = 42,
    download: bool = False,
):
    from neuralbench.data import Data

    logging.getLogger("neuralset").setLevel(logging.WARNING)
    logging.getLogger("exca").setLevel(logging.WARNING)
    data_dir = Path(data_dir)
    if download:
        download_official_data(data_dir)
    if not (data_dir / STUDY).exists():
        raise FileNotFoundError(f"{STUDY} missing under {data_dir}; use --download.")
    loaders = Data(**_official_config(data_dir, batch_size, seed)).prepare()
    train = OfficialWindowDataset(loaders["train"].dataset)
    val = OfficialWindowDataset(loaders["val"].dataset)
    test = OfficialWindowDataset(loaders["test"].dataset)
    x0, _ = train[0]
    extractor = loaders["train"].dataset.extractors["neuro"]
    channels = getattr(extractor, "_channels", None)
    channel_names = (
        [name for name, _ in sorted(channels.items(), key=lambda item: item[1])]
        if channels
        else EXPECTED_CHANNELS
    )
    if channel_names != EXPECTED_CHANNELS:
        raise ValueError(f"Expected channels {EXPECTED_CHANNELS}, got {channel_names}")
    metadata = {
        "n_chans": x0.shape[0],
        "n_times": x0.shape[1],
        "sfreq": x0.shape[1] / 5.0,
        "channel_names": channel_names,
        "n_train_records": len(train.by_record),
        "n_val_records": len(val.by_record),
        "n_test_records": len(test.by_record),
    }
    if metadata["sfreq"] != 128.0:
        raise ValueError(f"Expected 128 Hz, got {metadata['sfreq']}")
    return ChronologicalSequenceDataset(train, sequence_length), train, val, test, metadata
