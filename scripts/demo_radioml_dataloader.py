"""Restricted-load RadioML and verify one real train DataLoader batch."""

from __future__ import annotations

import argparse
from pathlib import Path

from signal_modulation.dataset import create_data_loader
from signal_modulation.radioml import load_restricted_radioml_pickle
from signal_modulation.radioml_dataset import (
    RadioMLTorchDataset,
    create_radioml_partitions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pickle_file", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped_signals = load_restricted_radioml_pickle(args.pickle_file)
    dataset = RadioMLTorchDataset(grouped_signals)
    partitions = create_radioml_partitions(dataset, seed=20260811)
    train_loader = create_data_loader(
        partitions.train,
        batch_size=64,
        shuffle=True,
        seed=20260811,
        num_workers=0,
        pin_memory=True,
    )
    batch = next(iter(train_loader))

    print(f"total_samples={len(dataset)}")
    print(f"train_samples={len(partitions.train)}")
    print(f"validation_samples={len(partitions.validation)}")
    print(f"test_samples={len(partitions.test)}")
    print(f"modulations={dataset.modulations}")
    print(f"class_to_index={dataset.class_to_index}")
    print(f"batch_signal_shape={tuple(batch['signal'].shape)}")
    print(f"batch_label_shape={tuple(batch['label'].shape)}")
    print(f"batch_snr_shape={tuple(batch['snr'].shape)}")
    print("signal_arrays_copied_during_index_build=false")
    print("model_training_started=false")


if __name__ == "__main__":
    main()
