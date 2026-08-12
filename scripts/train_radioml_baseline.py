"""Train the first full RadioML baseline while keeping the test split sealed."""

from __future__ import annotations

import argparse
import platform
import time
from dataclasses import asdict
from pathlib import Path

import torch

from signal_modulation.data_integrity import sha256_file
from signal_modulation.dataset import create_data_loader
from signal_modulation.evaluation import evaluate_classifier
from signal_modulation.experiment import (
    BaselineExperimentConfig,
    prepare_new_run_directory,
    write_json_atomic,
)
from signal_modulation.model import (
    SimpleCNN1D,
    TemporalCNN1D,
    count_trainable_parameters,
)
from signal_modulation.radioml import load_restricted_radioml_pickle
from signal_modulation.radioml_dataset import (
    RadioMLTorchDataset,
    create_radioml_partitions,
)
from signal_modulation.reproducibility import configure_reproducibility
from signal_modulation.training import EpochRecord, fit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pickle_file", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--model-variant",
        choices=("simple", "temporal"),
        default="simple",
        help="CNN architecture to train; simple preserves the original baseline.",
    )
    return parser.parse_args()


def create_model(variant: str, *, num_classes: int) -> torch.nn.Module:
    """Build an explicitly named architecture for a controlled comparison."""

    if variant == "simple":
        return SimpleCNN1D(num_classes=num_classes)
    if variant == "temporal":
        return TemporalCNN1D(num_classes=num_classes)
    raise ValueError(f"unsupported model variant: {variant}")


def print_epoch(record: EpochRecord) -> None:
    print(
        f"epoch={record.epoch} "
        f"train_loss={record.train.loss:.4f} "
        f"train_accuracy={record.train.accuracy:.4f} "
        f"validation_loss={record.validation.loss:.4f} "
        f"validation_accuracy={record.validation.accuracy:.4f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    config = BaselineExperimentConfig()
    output_directory = prepare_new_run_directory(args.output_directory)
    configure_reproducibility(config.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    grouped_signals = load_restricted_radioml_pickle(args.pickle_file)
    dataset = RadioMLTorchDataset(grouped_signals)
    partitions = create_radioml_partitions(dataset, seed=config.seed)
    train_loader = create_data_loader(
        partitions.train,
        batch_size=config.train_batch_size,
        shuffle=True,
        seed=config.seed,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = create_data_loader(
        partitions.validation,
        batch_size=config.evaluation_batch_size,
        shuffle=False,
        seed=config.seed,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = create_model(
        args.model_variant,
        num_classes=len(dataset.modulations),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    checkpoint_filename = (
        "baseline_best.pt"
        if args.model_variant == "simple"
        else "temporal_cnn_best.pt"
    )
    checkpoint_path = output_directory / checkpoint_filename
    started = time.perf_counter()
    result = fit(
        model,
        train_loader,
        validation_loader,
        optimizer,
        device=device,
        epochs=config.epochs,
        patience=config.patience,
        min_delta=config.min_delta,
        checkpoint_path=checkpoint_path,
        on_epoch=print_epoch,
    )
    validation = evaluate_classifier(
        model,
        validation_loader,
        device=device,
        num_classes=len(dataset.modulations),
    )
    elapsed_seconds = time.perf_counter() - started

    history = [
        {
            "epoch": record.epoch,
            "train": asdict(record.train),
            "validation": asdict(record.validation),
        }
        for record in result.history
    ]
    payload = {
        "experiment": f"radioml_2016_10a_{args.model_variant}_cnn_v1",
        "scope": "full_train_and_validation_only",
        "test_set_used": False,
        "controlled_change": "model_architecture_only",
        "config": asdict(config),
        "dataset": {
            "pickle_sha256": sha256_file(args.pickle_file),
            "total_samples": len(dataset),
            "train_samples": len(partitions.train),
            "validation_samples": len(partitions.validation),
            "sealed_test_samples": len(partitions.test),
            "modulations": dataset.modulations,
            "class_to_index": dataset.class_to_index,
        },
        "model": {
            "name": type(model).__name__,
            "trainable_parameters": count_trainable_parameters(model),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "elapsed_seconds": elapsed_seconds,
            "peak_cuda_memory_mb": (
                torch.cuda.max_memory_allocated() / 1024**2
                if device.type == "cuda"
                else None
            ),
        },
        "training": {
            "history": history,
            "best_epoch": result.best_epoch,
            "best_validation_loss": result.best_validation_loss,
            "stopped_early": result.stopped_early,
        },
        "validation": {
            "loss": validation.loss,
            "sample_count": validation.sample_count,
            "accuracy": validation.classification.accuracy,
            "macro_precision": validation.classification.macro_precision,
            "macro_recall": validation.classification.macro_recall,
            "macro_f1": validation.classification.macro_f1,
            "per_class_precision": validation.classification.per_class_precision,
            "per_class_recall": validation.classification.per_class_recall,
            "per_class_f1": validation.classification.per_class_f1,
            "confusion_matrix": validation.classification.confusion_matrix,
            "by_snr": [asdict(item) for item in validation.by_snr],
        },
    }
    result_path = write_json_atomic(output_directory / "validation_result.json", payload)

    print(f"best_epoch={result.best_epoch}")
    print(f"validation_accuracy={validation.classification.accuracy:.4f}")
    print(f"validation_macro_f1={validation.classification.macro_f1:.4f}")
    print(f"elapsed_seconds={elapsed_seconds:.2f}")
    print(f"checkpoint={checkpoint_path}")
    print(f"result_json={result_path}")
    print("test_set_used=false")
    print("final_benchmark_reported=false")


if __name__ == "__main__":
    main()
