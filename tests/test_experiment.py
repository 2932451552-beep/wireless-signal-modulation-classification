"""Tests for frozen experiment settings and safe JSON persistence."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from signal_modulation.experiment import (
    BaselineExperimentConfig,
    prepare_new_run_directory,
    write_json_atomic,
)


class _TrainScriptImportTests(unittest.TestCase):
    def test_model_factory_preserves_explicit_variants(self) -> None:
        from scripts.train_radioml_baseline import create_model

        simple = create_model("simple", num_classes=11)
        temporal = create_model("temporal", num_classes=11)

        self.assertEqual(simple(torch.randn(2, 2, 128)).shape, (2, 11))
        self.assertEqual(temporal(torch.randn(2, 2, 128)).shape, (2, 11))

    def test_model_factory_rejects_unknown_variant(self) -> None:
        from scripts.train_radioml_baseline import create_model

        with self.assertRaisesRegex(ValueError, "unsupported"):
            create_model("unknown", num_classes=11)


class ExperimentTests(unittest.TestCase):
    def test_default_baseline_configuration_is_valid(self) -> None:
        config = BaselineExperimentConfig()

        self.assertEqual(config.epochs, 20)
        self.assertEqual(config.train_batch_size, 256)
        self.assertEqual(config.learning_rate, 0.001)

    def test_invalid_learning_rate_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            BaselineExperimentConfig(learning_rate=0.0)

    def test_non_empty_run_directory_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory) / "existing"
            run_directory.mkdir()
            (run_directory / "result.json").write_text("keep", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                prepare_new_run_directory(run_directory)

            self.assertEqual(
                (run_directory / "result.json").read_text(encoding="utf-8"),
                "keep",
            )

    def test_json_result_is_written_as_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_path = Path(temporary_directory) / "result.json"

            write_json_atomic(result_path, {"scope": "验证集", "accuracy": 0.5})
            loaded = json.loads(result_path.read_text(encoding="utf-8"))

            self.assertEqual(loaded, {"scope": "验证集", "accuracy": 0.5})
            self.assertFalse(result_path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
