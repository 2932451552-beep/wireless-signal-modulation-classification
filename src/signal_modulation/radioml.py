"""Restricted loading and schema validation for the legacy RadioML pickle."""

from __future__ import annotations

import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy._core.multiarray import _reconstruct


RadioMLKey = tuple[str, int]
RadioMLDataset = dict[RadioMLKey, np.ndarray]


@dataclass(frozen=True, slots=True)
class RadioMLSummary:
    """Validated structural facts about a loaded RadioML dataset."""

    group_count: int
    total_samples: int
    modulations: tuple[str, ...]
    snrs: tuple[int, ...]
    shapes: tuple[tuple[int, int, int], ...]
    dtypes: tuple[str, ...]
    samples_per_modulation: tuple[tuple[str, int], ...]
    samples_per_snr: tuple[tuple[int, int], ...]


class RestrictedNumpyUnpickler(pickle.Unpickler):
    """Allow only the NumPy constructors required by the audited dataset."""

    _allowed_globals: dict[tuple[str, str], Any] = {
        ("numpy", "dtype"): np.dtype,
        ("numpy", "ndarray"): np.ndarray,
        ("numpy.core.multiarray", "_reconstruct"): _reconstruct,
        ("numpy._core.multiarray", "_reconstruct"): _reconstruct,
    }

    def find_class(self, module: str, name: str) -> Any:
        try:
            return self._allowed_globals[(module, name)]
        except KeyError as error:
            raise pickle.UnpicklingError(
                f"pickle global is not allowed: {module}.{name}"
            ) from error

    def persistent_load(self, persistent_id: object) -> Any:
        raise pickle.UnpicklingError(
            f"persistent pickle references are not allowed: {persistent_id!r}"
        )


def load_restricted_radioml_pickle(path: str | Path) -> RadioMLDataset:
    """Load a legacy Python-2-compatible pickle through the strict allowlist."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)

    with source.open("rb") as file_handle:
        unpickler = RestrictedNumpyUnpickler(
            file_handle,
            fix_imports=True,
            encoding="latin1",
            errors="strict",
        )
        loaded = unpickler.load()
        if file_handle.read(1):
            raise ValueError("pickle contains trailing data after the first object")

    if not isinstance(loaded, dict):
        raise ValueError("RadioML pickle must contain a dictionary")
    return loaded


def audit_radioml_dataset(dataset: RadioMLDataset) -> RadioMLSummary:
    """Validate all labels, array shapes, dtypes, and finite I/Q values."""

    if not dataset:
        raise ValueError("RadioML dataset cannot be empty")

    modulations: set[str] = set()
    snrs: set[int] = set()
    shapes: set[tuple[int, int, int]] = set()
    dtypes: set[str] = set()
    modulation_counts: Counter[str] = Counter()
    snr_counts: Counter[int] = Counter()
    total_samples = 0

    for key, signals in dataset.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("every RadioML key must be a (modulation, snr) tuple")
        modulation, snr = key
        if not isinstance(modulation, str) or not modulation:
            raise ValueError("modulation labels must be non-empty strings")
        if not isinstance(snr, (int, np.integer)):
            raise ValueError("SNR labels must be integers")
        if not isinstance(signals, np.ndarray):
            raise ValueError("every RadioML group must contain a NumPy array")
        if signals.ndim != 3 or signals.shape[1:] != (2, 128):
            raise ValueError("RadioML arrays must have shape (samples, 2, 128)")
        if signals.shape[0] == 0:
            raise ValueError("RadioML groups cannot be empty")
        if signals.dtype != np.float32:
            raise ValueError("RadioML I/Q arrays must use float32 values")
        if not np.isfinite(signals).all():
            raise ValueError("RadioML I/Q arrays must contain only finite values")

        normalized_snr = int(snr)
        sample_count = int(signals.shape[0])
        modulations.add(modulation)
        snrs.add(normalized_snr)
        shapes.add(tuple(int(value) for value in signals.shape))
        dtypes.add(str(signals.dtype))
        modulation_counts[modulation] += sample_count
        snr_counts[normalized_snr] += sample_count
        total_samples += sample_count

    return RadioMLSummary(
        group_count=len(dataset),
        total_samples=total_samples,
        modulations=tuple(sorted(modulations)),
        snrs=tuple(sorted(snrs)),
        shapes=tuple(sorted(shapes)),
        dtypes=tuple(sorted(dtypes)),
        samples_per_modulation=tuple(sorted(modulation_counts.items())),
        samples_per_snr=tuple(sorted(snr_counts.items())),
    )


def validate_radioml_2016_10a_profile(summary: RadioMLSummary) -> None:
    """Require the published 11-modulation, 20-SNR, 220-group profile."""

    expected_snrs = tuple(range(-20, 20, 2))
    if len(summary.modulations) != 11:
        raise ValueError("RML2016.10A must contain 11 modulation classes")
    if summary.snrs != expected_snrs:
        raise ValueError("RML2016.10A must contain SNR levels from -20 to 18 dB")
    if summary.group_count != 220:
        raise ValueError("RML2016.10A must contain 220 modulation/SNR groups")
    if summary.total_samples != 220_000:
        raise ValueError("RML2016.10A must contain 220,000 samples")
    if summary.shapes != ((1000, 2, 128),):
        raise ValueError("every RML2016.10A group must have shape (1000, 2, 128)")
