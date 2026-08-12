"""Evaluate the frozen TemporalCNN once on the sealed RadioML test split."""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from signal_modulation.data_integrity import sha256_file
from signal_modulation.dataset import create_data_loader
from signal_modulation.evaluation import evaluate_classifier
from signal_modulation.experiment import prepare_new_run_directory, write_json_atomic
from signal_modulation.final_evaluation import (
    load_frozen_model_checkpoint,
    verify_file_sha256,
)
from signal_modulation.model import TemporalCNN1D, count_trainable_parameters
from signal_modulation.radioml import load_restricted_radioml_pickle
from signal_modulation.radioml_dataset import (
    RadioMLTorchDataset,
    create_radioml_partitions,
)
from signal_modulation.reproducibility import configure_reproducibility


FROZEN_SEED = 20260812
FROZEN_EPOCH = 13
FROZEN_DATASET_SHA256 = "b29ccc25b00d0718cd3b70ffa9158662ec83f6d9b63ffd845c7bcbe3b3096e8c"
FROZEN_CHECKPOINT_SHA256 = "424bd247b767d0ab23c7d2217445dea62d351d9cc54ec1fe0f9bacbb770349b3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pickle_file", type=Path)
    parser.add_argument("checkpoint_file", type=Path)
    parser.add_argument("output_directory", type=Path)
    return parser.parse_args()


def build_test_result(
    *,
    pickle_file: Path,
    checkpoint_file: Path,
    output_directory: Path,
) -> Path:
    """Verify frozen inputs, evaluate exactly the test subset, and persist metrics."""

    dataset_sha256 = verify_file_sha256(pickle_file, FROZEN_DATASET_SHA256)
    model = TemporalCNN1D(num_classes=11)
    checkpoint = load_frozen_model_checkpoint(
        model,
        checkpoint_file,
        expected_sha256=FROZEN_CHECKPOINT_SHA256,
        expected_epoch=FROZEN_EPOCH,
    )
    destination = prepare_new_run_directory(output_directory)
    configure_reproducibility(FROZEN_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    grouped_signals = load_restricted_radioml_pickle(pickle_file)
    dataset = RadioMLTorchDataset(grouped_signals)
    if len(dataset.modulations) != 11:
        raise ValueError("frozen protocol requires exactly 11 modulation classes")
    partitions = create_radioml_partitions(dataset, seed=FROZEN_SEED)
    test_loader = create_data_loader(
        partitions.test,
        batch_size=512,
        shuffle=False,
        seed=FROZEN_SEED,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model.to(device)
    started = time.perf_counter()
    evaluation = evaluate_classifier(
        model,
        test_loader,
        device=device,
        num_classes=len(dataset.modulations),
    )
    elapsed_seconds = time.perf_counter() - started
    if evaluation.sample_count != len(partitions.test):
        raise RuntimeError("test evaluation did not consume the complete frozen split")

    payload = {
        "experiment": "radioml_2016_10a_temporal_cnn_final_test_v1",
        "scope": "frozen_test_only",
        "test_set_used": True,
        "model_selection_after_test": False,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "seed": FROZEN_SEED,
            "selected_epoch": FROZEN_EPOCH,
            "selection_criterion": "lowest_validation_loss",
            "dataset_sha256": dataset_sha256,
            "checkpoint_sha256": sha256_file(checkpoint_file),
            "shuffle": False,
            "gradient_enabled": False,
        },
        "dataset": {
            "total_samples": len(dataset),
            "train_samples": len(partitions.train),
            "validation_samples": len(partitions.validation),
            "test_samples": len(partitions.test),
            "modulations": dataset.modulations,
            "class_to_index": dataset.class_to_index,
        },
        "model": {
            "name": type(model).__name__,
            "trainable_parameters": count_trainable_parameters(model),
            "checkpoint_epoch": int(checkpoint["epoch"]),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "elapsed_seconds": elapsed_seconds,
        },
        "test": {
            "loss": evaluation.loss,
            "sample_count": evaluation.sample_count,
            "accuracy": evaluation.classification.accuracy,
            "macro_precision": evaluation.classification.macro_precision,
            "macro_recall": evaluation.classification.macro_recall,
            "macro_f1": evaluation.classification.macro_f1,
            "per_class_precision": evaluation.classification.per_class_precision,
            "per_class_recall": evaluation.classification.per_class_recall,
            "per_class_f1": evaluation.classification.per_class_f1,
            "confusion_matrix": evaluation.classification.confusion_matrix,
            "by_snr": [asdict(item) for item in evaluation.by_snr],
        },
    }
    return write_json_atomic(destination / "final_test_result.json", payload)


def main() -> None:
    args = parse_args()
    result_path = build_test_result(
        pickle_file=args.pickle_file,
        checkpoint_file=args.checkpoint_file,
        output_directory=args.output_directory,
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    print(f"test_accuracy={payload['test']['accuracy']:.4f}")
    print(f"test_macro_f1={payload['test']['macro_f1']:.4f}")
    print(f"test_samples={payload['test']['sample_count']}")
    print(f"result_json={result_path}")
    print("model_selection_after_test=false")


if __name__ == "__main__":
    main()
