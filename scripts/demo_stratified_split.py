"""Demonstrate leakage-free splitting with small synthetic metadata."""

from __future__ import annotations

from collections import Counter

import numpy as np

from signal_modulation.splitting import stratified_split_indices


def _counts(indices: np.ndarray, labels: np.ndarray, snrs: np.ndarray) -> Counter:
    return Counter((int(labels[index]), int(snrs[index])) for index in indices)


def main() -> None:
    labels: list[int] = []
    snrs: list[int] = []
    for label in (0, 1):
        for snr in (-10, 0, 10):
            labels.extend([label] * 10)
            snrs.extend([snr] * 10)

    label_array = np.asarray(labels, dtype=np.int64)
    snr_array = np.asarray(snrs, dtype=np.int64)
    split = stratified_split_indices(
        label_array,
        snr_array,
        train_fraction=0.60,
        validation_fraction=0.20,
        seed=42,
    )

    print(f"total_samples={label_array.size}")
    print(f"train_samples={split.train.size}")
    print(f"validation_samples={split.validation.size}")
    print(f"test_samples={split.test.size}")
    print(f"train_counts={dict(sorted(_counts(split.train, label_array, snr_array).items()))}")
    print(
        "validation_counts="
        f"{dict(sorted(_counts(split.validation, label_array, snr_array).items()))}"
    )
    print(f"test_counts={dict(sorted(_counts(split.test, label_array, snr_array).items()))}")


if __name__ == "__main__":
    main()
