"""Tests for frozen-artifact checks used before the final test pass."""

import tempfile
import unittest
from pathlib import Path

import torch

from signal_modulation.data_integrity import sha256_file
from signal_modulation.final_evaluation import (
    load_frozen_model_checkpoint,
    verify_file_sha256,
)
from signal_modulation.model import TemporalCNN1D


class FinalEvaluationSafetyTests(unittest.TestCase):
    def test_hash_verification_accepts_exact_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.bin"
            path.write_bytes(b"frozen")

            actual = verify_file_sha256(path, sha256_file(path))

            self.assertEqual(actual, sha256_file(path))

    def test_hash_verification_rejects_modified_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.bin"
            path.write_bytes(b"modified")

            with self.assertRaisesRegex(ValueError, "mismatch"):
                verify_file_sha256(path, "0" * 64)

    def test_checkpoint_loader_requires_frozen_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            model = TemporalCNN1D(num_classes=11)
            torch.save({"epoch": 12, "model_state_dict": model.state_dict()}, path)

            with self.assertRaisesRegex(ValueError, "epoch"):
                load_frozen_model_checkpoint(
                    TemporalCNN1D(num_classes=11),
                    path,
                    expected_sha256=sha256_file(path),
                    expected_epoch=13,
                )

    def test_checkpoint_loader_restores_exact_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            source = TemporalCNN1D(num_classes=11)
            torch.save({"epoch": 13, "model_state_dict": source.state_dict()}, path)
            target = TemporalCNN1D(num_classes=11)

            checkpoint = load_frozen_model_checkpoint(
                target,
                path,
                expected_sha256=sha256_file(path),
                expected_epoch=13,
            )

            self.assertEqual(checkpoint["epoch"], 13)
            for source_parameter, target_parameter in zip(
                source.parameters(), target.parameters(), strict=True
            ):
                torch.testing.assert_close(source_parameter, target_parameter)
