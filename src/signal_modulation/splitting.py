"""Deterministic dataset splitting by modulation label and SNR stratum."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """Integer sample indices for three mutually exclusive dataset partitions."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def _validate_inputs(labels: np.ndarray, snrs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    label_values = np.asarray(labels)
    snr_values = np.asarray(snrs)
    if label_values.ndim != 1 or snr_values.ndim != 1:
        raise ValueError("labels and snrs must be one-dimensional")
    if label_values.size == 0 or label_values.size != snr_values.size:
        raise ValueError("labels and snrs must have the same non-zero length")
    if not np.issubdtype(label_values.dtype, np.integer):
        raise ValueError("labels must contain integer class indices")
    if not np.issubdtype(snr_values.dtype, np.number) or not np.all(np.isfinite(snr_values)):
        raise ValueError("snrs must contain finite numeric values")
    return label_values.astype(np.int64), snr_values


def _validate_fractions(train_fraction: float, validation_fraction: float) -> float:
    test_fraction = 1.0 - train_fraction - validation_fraction
    fractions = (train_fraction, validation_fraction, test_fraction)
    if not all(math.isfinite(value) and 0.0 < value < 1.0 for value in fractions):
        raise ValueError("train, validation, and test fractions must all be between 0 and 1")
    return test_fraction


def assert_split_integrity(split: DatasetSplit, sample_count: int) -> None:
    """Ensure every sample appears exactly once across the three partitions."""

    if sample_count <= 0:
        raise ValueError("sample_count must be greater than zero")
    partitions = (split.train, split.validation, split.test)
    if any(values.ndim != 1 for values in partitions):
        raise ValueError("split indices must be one-dimensional")

    combined = np.concatenate(partitions)
    if combined.size != sample_count:
        raise ValueError("split does not contain the expected number of samples")
    if not np.issubdtype(combined.dtype, np.integer):
        raise ValueError("split indices must be integers")
    if np.any(combined < 0) or np.any(combined >= sample_count):
        raise ValueError("split contains an out-of-range sample index")
    if np.unique(combined).size != sample_count:
        raise ValueError("split contains duplicate or missing sample indices")


def stratified_split_indices(
    labels: np.ndarray,
    snrs: np.ndarray,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    seed: int = 20260811,
) -> DatasetSplit:
    """Split every modulation/SNR stratum with the same ratios and random seed."""

    label_values, snr_values = _validate_inputs(labels, snrs)
    _validate_fractions(train_fraction, validation_fraction)

    strata: dict[tuple[int, float], list[int]] = {}
    for index, (label, snr) in enumerate(zip(label_values, snr_values, strict=True)):
        strata.setdefault((int(label), float(snr)), []).append(index)

    generator = np.random.default_rng(seed)
    train_groups: list[np.ndarray] = []
    validation_groups: list[np.ndarray] = []
    test_groups: list[np.ndarray] = []

    for key in sorted(strata):
        shuffled = generator.permutation(np.asarray(strata[key], dtype=np.int64))
        group_size = shuffled.size
        train_count = int(group_size * train_fraction)
        validation_count = int(group_size * validation_fraction)
        test_count = group_size - train_count - validation_count
        if min(train_count, validation_count, test_count) == 0:
            raise ValueError(f"stratum {key} is too small for the requested fractions")

        train_groups.append(shuffled[:train_count])
        validation_groups.append(shuffled[train_count : train_count + validation_count])
        test_groups.append(shuffled[train_count + validation_count :])

    split = DatasetSplit(
        train=generator.permutation(np.concatenate(train_groups)),
        validation=generator.permutation(np.concatenate(validation_groups)),
        test=generator.permutation(np.concatenate(test_groups)),
    )
    assert_split_integrity(split, sample_count=label_values.size)
    return split


def stratified_subsample_indices(
    candidate_indices: np.ndarray,
    labels: np.ndarray,
    snrs: np.ndarray,
    *,
    samples_per_stratum: int,
    allowed_snrs: tuple[int, ...] | None = None,
    seed: int = 20260811,
) -> np.ndarray:
    """Select an equal, reproducible count from each eligible label/SNR stratum."""

    label_values, snr_values = _validate_inputs(labels, snrs)
    candidates = np.asarray(candidate_indices)
    if candidates.ndim != 1 or candidates.size == 0:
        raise ValueError("candidate_indices must be a non-empty one-dimensional array")
    if not np.issubdtype(candidates.dtype, np.integer):
        raise ValueError("candidate_indices must contain integers")
    candidates = candidates.astype(np.int64, copy=False)
    if np.any(candidates < 0) or np.any(candidates >= label_values.size):
        raise ValueError("candidate_indices contain an out-of-range value")
    if np.unique(candidates).size != candidates.size:
        raise ValueError("candidate_indices cannot contain duplicates")
    if samples_per_stratum <= 0:
        raise ValueError("samples_per_stratum must be greater than zero")

    allowed = None if allowed_snrs is None else {int(value) for value in allowed_snrs}
    if allowed is not None and not allowed:
        raise ValueError("allowed_snrs cannot be empty")

    strata: dict[tuple[int, float], list[int]] = {}
    for index in candidates:
        snr = float(snr_values[index])
        if allowed is not None and int(snr) not in allowed:
            continue
        key = (int(label_values[index]), snr)
        strata.setdefault(key, []).append(int(index))
    if not strata:
        raise ValueError("no candidate samples match the requested SNR levels")

    generator = np.random.default_rng(seed)
    selected_groups: list[np.ndarray] = []
    for key in sorted(strata):
        group = np.asarray(strata[key], dtype=np.int64)
        if group.size < samples_per_stratum:
            raise ValueError(
                f"stratum {key} has fewer than {samples_per_stratum} candidates"
            )
        selected_groups.append(generator.permutation(group)[:samples_per_stratum])
    return generator.permutation(np.concatenate(selected_groups))
