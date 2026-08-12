"""Validated PyTorch dataset and reproducible data-loader helpers."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class IQSignalDataset(Dataset):
    """In-memory I/Q samples with modulation labels and SNR metadata."""

    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        snrs: np.ndarray,
    ) -> None:
        signal_values = np.asarray(signals)
        label_values = np.asarray(labels)
        snr_values = np.asarray(snrs)

        if signal_values.ndim != 3 or signal_values.shape[1] != 2:
            raise ValueError("signals must have shape (samples, 2, length)")
        if signal_values.shape[0] == 0 or signal_values.shape[2] == 0:
            raise ValueError("signals must contain at least one non-empty sample")
        if np.iscomplexobj(signal_values) or not np.issubdtype(signal_values.dtype, np.number):
            raise ValueError("signals must contain real-valued I/Q numbers")
        if not np.all(np.isfinite(signal_values)):
            raise ValueError("signals must contain only finite values")

        sample_count = signal_values.shape[0]
        if label_values.ndim != 1 or label_values.size != sample_count:
            raise ValueError("labels must be one-dimensional and match the sample count")
        if not np.issubdtype(label_values.dtype, np.integer) or np.any(label_values < 0):
            raise ValueError("labels must contain non-negative integer class indices")

        if snr_values.ndim != 1 or snr_values.size != sample_count:
            raise ValueError("snrs must be one-dimensional and match the sample count")
        if (
            np.iscomplexobj(snr_values)
            or not np.issubdtype(snr_values.dtype, np.number)
            or not np.all(np.isfinite(snr_values))
        ):
            raise ValueError("snrs must contain finite real numbers")

        self.signals = torch.from_numpy(
            np.ascontiguousarray(signal_values, dtype=np.float32)
        )
        self.labels = torch.from_numpy(
            np.ascontiguousarray(label_values, dtype=np.int64)
        )
        self.snrs = torch.from_numpy(
            np.ascontiguousarray(snr_values, dtype=np.float32)
        )

    def __len__(self) -> int:
        return self.signals.shape[0]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {
            "signal": self.signals[index],
            "label": self.labels[index],
            "snr": self.snrs[index],
        }


def create_data_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int = 20260811,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Build a deterministic loader suitable for the initial Windows setup."""

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if not isinstance(seed, int) or not math.isfinite(float(seed)):
        raise ValueError("seed must be an integer")

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        drop_last=False,
    )
