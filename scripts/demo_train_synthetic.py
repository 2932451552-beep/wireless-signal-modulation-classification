"""Train the small CNN on a synthetic BPSK/QPSK sanity-check dataset."""

from __future__ import annotations

import numpy as np
import torch

from signal_modulation.dataset import IQSignalDataset, create_data_loader
from signal_modulation.model import SimpleCNN1D
from signal_modulation.splitting import stratified_split_indices
from signal_modulation.synthetic import add_awgn, complex_to_iq, generate_bpsk, generate_qpsk
from signal_modulation.training import fit


def make_synthetic_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create balanced BPSK/QPSK samples for a pipeline sanity check only."""

    rng = np.random.default_rng(20260811)
    length = 128
    samples_per_group = 80
    snr_levels = (5.0, 10.0, 15.0)
    signals: list[np.ndarray] = []
    labels: list[int] = []
    snrs: list[float] = []

    for label in (0, 1):
        for snr in snr_levels:
            for _ in range(samples_per_group):
                symbol_count = 2 if label == 0 else 4
                symbols = rng.integers(0, symbol_count, size=length)
                clean = generate_bpsk(symbols) if label == 0 else generate_qpsk(symbols)
                signals.append(complex_to_iq(add_awgn(clean, snr, rng)))
                labels.append(label)
                snrs.append(snr)

    return (
        np.asarray(signals, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(snrs, dtype=np.float32),
    )


def main() -> None:
    torch.manual_seed(20260811)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260811)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    signals, labels, snrs = make_synthetic_dataset()
    split = stratified_split_indices(labels, snrs, seed=20260811)
    train_dataset = IQSignalDataset(signals[split.train], labels[split.train], snrs[split.train])
    validation_dataset = IQSignalDataset(
        signals[split.validation], labels[split.validation], snrs[split.validation]
    )
    train_loader = create_data_loader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        seed=20260811,
        pin_memory=device.type == "cuda",
    )
    validation_loader = create_data_loader(
        validation_dataset,
        batch_size=64,
        shuffle=False,
        seed=20260811,
        pin_memory=device.type == "cuda",
    )

    model = SimpleCNN1D(num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    print(f"device={device.type}")
    print(f"train_samples={len(train_dataset)} validation_samples={len(validation_dataset)}")
    result = fit(
        model,
        train_loader,
        validation_loader,
        optimizer,
        device=device,
        epochs=5,
        patience=2,
    )
    for record in result.history:
        print(
            f"epoch={record.epoch} "
            f"train_loss={record.train.loss:.4f} "
            f"train_accuracy={record.train.accuracy:.4f} "
            f"validation_loss={record.validation.loss:.4f} "
            f"validation_accuracy={record.validation.accuracy:.4f}"
        )

    print(f"best_epoch={result.best_epoch} stopped_early={result.stopped_early}")
    print("result_scope=synthetic_bpsk_qpsk_pipeline_check_only")
    print("radioml_accuracy_not_reported=true")


if __name__ == "__main__":
    main()
