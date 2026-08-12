"""Run a bounded high-SNR RadioML GPU training smoke test, not a benchmark."""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import torch
from torch.utils.data import Subset

from signal_modulation.dataset import create_data_loader
from signal_modulation.evaluation import evaluate_classifier
from signal_modulation.model import SimpleCNN1D
from signal_modulation.radioml import load_restricted_radioml_pickle
from signal_modulation.radioml_dataset import (
    RadioMLTorchDataset,
    create_radioml_partitions,
)
from signal_modulation.reproducibility import configure_reproducibility
from signal_modulation.splitting import stratified_subsample_indices
from signal_modulation.training import fit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pickle_file", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_reproducibility(20260812)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    grouped_signals = load_restricted_radioml_pickle(args.pickle_file)
    dataset = RadioMLTorchDataset(grouped_signals)
    partitions = create_radioml_partitions(dataset, seed=20260811)
    high_snr_levels = (10, 12, 14, 16, 18)
    train_indices = stratified_subsample_indices(
        partitions.indices.train,
        dataset.labels,
        dataset.snrs,
        samples_per_stratum=100,
        allowed_snrs=high_snr_levels,
        seed=20260812,
    )
    validation_indices = stratified_subsample_indices(
        partitions.indices.validation,
        dataset.labels,
        dataset.snrs,
        samples_per_stratum=30,
        allowed_snrs=high_snr_levels,
        seed=20260812,
    )
    train_loader = create_data_loader(
        Subset(dataset, train_indices),
        batch_size=64,
        shuffle=True,
        seed=20260812,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = create_data_loader(
        Subset(dataset, validation_indices),
        batch_size=128,
        shuffle=False,
        seed=20260812,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = SimpleCNN1D(num_classes=len(dataset.modulations)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as temporary_directory:
        checkpoint_path = Path(temporary_directory) / "smoke_best.pt"
        result = fit(
            model,
            train_loader,
            validation_loader,
            optimizer,
            device=device,
            epochs=3,
            patience=2,
            checkpoint_path=checkpoint_path,
        )
        checkpoint_saved = checkpoint_path.is_file()
        evaluation = evaluate_classifier(
            model,
            validation_loader,
            device=device,
            num_classes=len(dataset.modulations),
        )
    elapsed_seconds = time.perf_counter() - started

    print(f"device={device.type}")
    print(f"snr_scope={high_snr_levels}")
    print(f"train_samples={train_indices.size}")
    print(f"validation_samples={validation_indices.size}")
    for record in result.history:
        print(
            f"epoch={record.epoch} "
            f"train_loss={record.train.loss:.4f} "
            f"train_accuracy={record.train.accuracy:.4f} "
            f"validation_loss={record.validation.loss:.4f} "
            f"validation_accuracy={record.validation.accuracy:.4f}"
        )
    print(f"best_epoch={result.best_epoch}")
    print(f"restored_validation_loss={evaluation.loss:.4f}")
    print(f"restored_validation_accuracy={evaluation.classification.accuracy:.4f}")
    print(f"macro_precision={evaluation.classification.macro_precision:.4f}")
    print(f"macro_recall={evaluation.classification.macro_recall:.4f}")
    print(f"macro_f1={evaluation.classification.macro_f1:.4f}")
    print(f"confusion_matrix_size={len(evaluation.classification.confusion_matrix)}x{len(evaluation.classification.confusion_matrix)}")
    print(
        "accuracy_by_snr="
        + str(tuple((item.snr, round(item.accuracy, 4)) for item in evaluation.by_snr))
    )
    print(f"elapsed_seconds={elapsed_seconds:.2f}")
    if device.type == "cuda":
        print(f"peak_cuda_memory_mb={torch.cuda.max_memory_allocated() / 1024**2:.2f}")
    print(f"temporary_checkpoint_saved={str(checkpoint_saved).lower()}")
    print("result_scope=high_snr_balanced_smoke_test_only")
    print("test_set_used=false")
    print("benchmark_accuracy_reported=false")


if __name__ == "__main__":
    main()
