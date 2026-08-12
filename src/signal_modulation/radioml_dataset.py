"""Memory-conscious PyTorch access to validated RadioML dictionary groups."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset

from signal_modulation.radioml import RadioMLDataset as RawRadioMLDataset
from signal_modulation.radioml import audit_radioml_dataset
from signal_modulation.splitting import DatasetSplit, stratified_split_indices


class RadioMLTorchDataset(Dataset):
    """Expose grouped RadioML arrays as individual signal/label/SNR samples."""

    def __init__(self, grouped_signals: RawRadioMLDataset) -> None:
        summary = audit_radioml_dataset(grouped_signals)
        self.grouped_signals = grouped_signals
        self.modulations = summary.modulations
        self.class_to_index = {
            modulation: index for index, modulation in enumerate(self.modulations)
        }
        self.group_keys = tuple(
            sorted(grouped_signals, key=lambda key: (key[0], int(key[1])))
        )

        offsets = [0]
        label_parts: list[np.ndarray] = []
        snr_parts: list[np.ndarray] = []
        for modulation, snr in self.group_keys:
            group_size = int(grouped_signals[(modulation, snr)].shape[0])
            offsets.append(offsets[-1] + group_size)
            label_parts.append(
                np.full(group_size, self.class_to_index[modulation], dtype=np.int64)
            )
            snr_parts.append(np.full(group_size, int(snr), dtype=np.int16))

        self._offsets = tuple(offsets)
        self.labels = np.concatenate(label_parts)
        self.snrs = np.concatenate(snr_parts)

    def __len__(self) -> int:
        return self._offsets[-1]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if not isinstance(index, (int, np.integer)):
            raise TypeError("sample index must be an integer")
        normalized_index = int(index)
        if normalized_index < 0 or normalized_index >= len(self):
            raise IndexError("sample index is outside the dataset")

        group_index = bisect_right(self._offsets, normalized_index) - 1
        group_start = self._offsets[group_index]
        within_group_index = normalized_index - group_start
        key = self.group_keys[group_index]
        signal = self.grouped_signals[key][within_group_index]
        return {
            "signal": torch.from_numpy(signal),
            "label": torch.tensor(self.labels[normalized_index], dtype=torch.int64),
            "snr": torch.tensor(self.snrs[normalized_index], dtype=torch.float32),
        }


@dataclass(frozen=True, slots=True)
class RadioMLPartitions:
    """One shared dataset plus mutually exclusive train/validation/test views."""

    dataset: RadioMLTorchDataset
    indices: DatasetSplit
    train: Subset
    validation: Subset
    test: Subset


def create_radioml_partitions(
    dataset: RadioMLTorchDataset,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    seed: int = 20260811,
) -> RadioMLPartitions:
    """Stratify every modulation/SNR group without copying signal arrays."""

    split = stratified_split_indices(
        dataset.labels,
        dataset.snrs,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    return RadioMLPartitions(
        dataset=dataset,
        indices=split,
        train=Subset(dataset, split.train),
        validation=Subset(dataset, split.validation),
        test=Subset(dataset, split.test),
    )
