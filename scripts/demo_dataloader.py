"""Create synthetic I/Q samples and move one DataLoader batch to the GPU."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Subset

from signal_modulation.dataset import IQSignalDataset, create_data_loader
from signal_modulation.splitting import stratified_split_indices
from signal_modulation.synthetic import (
    add_awgn,
    complex_to_iq,
    generate_bpsk,
    generate_qpsk,
)


def _build_synthetic_dataset() -> IQSignalDataset:
    generator = np.random.default_rng(20260811)
    signals: list[np.ndarray] = []
    labels: list[int] = []
    snrs: list[int] = []

    for label in (0, 1):
        for snr_db in (-10, 0, 10):
            for _ in range(16):
                if label == 0:
                    symbols = generator.integers(0, 2, size=128, dtype=np.int64)
                    clean = generate_bpsk(symbols)
                else:
                    symbols = generator.integers(0, 4, size=128, dtype=np.int64)
                    clean = generate_qpsk(symbols)
                noisy = add_awgn(clean, snr_db=snr_db, rng=generator)
                signals.append(complex_to_iq(noisy))
                labels.append(label)
                snrs.append(snr_db)

    return IQSignalDataset(
        np.stack(signals),
        np.asarray(labels, dtype=np.int64),
        np.asarray(snrs, dtype=np.int64),
    )


def main() -> None:
    dataset = _build_synthetic_dataset()
    split = stratified_split_indices(
        dataset.labels.numpy(),
        dataset.snrs.numpy(),
        train_fraction=0.50,
        validation_fraction=0.25,
        seed=42,
    )
    training_subset = Subset(dataset, split.train.tolist())
    loader = create_data_loader(
        training_subset,
        batch_size=8,
        shuffle=True,
        seed=42,
    )
    batch = next(iter(loader))

    print(f"dataset_samples={len(dataset)}")
    print(f"training_samples={len(training_subset)}")
    print(f"signal_batch_shape={tuple(batch['signal'].shape)}")
    print(f"label_batch_shape={tuple(batch['label'].shape)}")
    print(f"snr_batch_shape={tuple(batch['snr'].shape)}")
    print(f"loader_device={batch['signal'].device}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_signals = batch["signal"].to(device)
    print(f"training_device={gpu_signals.device}")


if __name__ == "__main__":
    main()
