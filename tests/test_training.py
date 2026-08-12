"""Tests for the explicit classification training loop."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from signal_modulation.dataset import IQSignalDataset, create_data_loader
from signal_modulation.model import SimpleCNN1D
from signal_modulation.training import EpochMetrics, evaluate, fit, train_one_epoch


def _separable_loader(*, shuffle: bool = False):
    signals = np.zeros((16, 2, 32), dtype=np.float32)
    labels = np.arange(16, dtype=np.int64) % 2
    signals[labels == 0, 0, :] = 1.0
    signals[labels == 1, 1, :] = 1.0
    snrs = np.full(16, 20.0, dtype=np.float32)
    dataset = IQSignalDataset(signals, labels, snrs)
    return create_data_loader(dataset, batch_size=4, shuffle=shuffle, seed=7)


class TrainingTests(unittest.TestCase):
    def test_training_updates_parameters_and_reports_metrics(self) -> None:
        torch.manual_seed(7)
        model = SimpleCNN1D(num_classes=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        before = model.classifier.weight.detach().clone()

        metrics = train_one_epoch(
            model,
            _separable_loader(shuffle=True),
            optimizer,
            device=torch.device("cpu"),
        )

        self.assertEqual(metrics.sample_count, 16)
        self.assertGreaterEqual(metrics.accuracy, 0.0)
        self.assertLessEqual(metrics.accuracy, 1.0)
        self.assertTrue(np.isfinite(metrics.loss))
        self.assertFalse(torch.equal(before, model.classifier.weight.detach()))

    def test_evaluation_does_not_change_parameters(self) -> None:
        model = SimpleCNN1D(num_classes=2)
        before = [parameter.detach().clone() for parameter in model.parameters()]

        metrics = evaluate(
            model,
            _separable_loader(),
            device=torch.device("cpu"),
        )

        self.assertEqual(metrics.sample_count, 16)
        self.assertFalse(model.training)
        for old, new in zip(before, model.parameters(), strict=True):
            torch.testing.assert_close(old, new)

    def test_empty_loader_is_rejected(self) -> None:
        model = SimpleCNN1D(num_classes=2)

        with self.assertRaisesRegex(ValueError, "no samples"):
            evaluate(model, [], device=torch.device("cpu"))

    def test_malformed_batch_is_rejected(self) -> None:
        model = SimpleCNN1D(num_classes=2)
        malformed = [{"signal": torch.randn(2, 2, 8)}]

        with self.assertRaisesRegex(ValueError, "signal and label"):
            evaluate(model, malformed, device=torch.device("cpu"))

    def test_fit_records_history_and_saves_best_checkpoint(self) -> None:
        torch.manual_seed(7)
        model = SimpleCNN1D(num_classes=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        with TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "best.pt"
            result = fit(
                model,
                _separable_loader(shuffle=True),
                _separable_loader(),
                optimizer,
                device=torch.device("cpu"),
                epochs=2,
                checkpoint_path=checkpoint_path,
            )
            checkpoint = torch.load(checkpoint_path, weights_only=True)

        self.assertEqual(len(result.history), 2)
        self.assertIn(result.best_epoch, (1, 2))
        self.assertEqual(checkpoint["epoch"], result.best_epoch)
        self.assertIn("model_state_dict", checkpoint)
        self.assertIn("optimizer_state_dict", checkpoint)

    def test_fit_stops_after_configured_non_improving_epochs(self) -> None:
        model = SimpleCNN1D(num_classes=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        train_metrics = EpochMetrics(loss=0.4, accuracy=0.8, sample_count=16)
        validation_metrics = [
            EpochMetrics(loss=0.5, accuracy=0.8, sample_count=16),
            EpochMetrics(loss=0.6, accuracy=0.7, sample_count=16),
            EpochMetrics(loss=0.7, accuracy=0.6, sample_count=16),
        ]

        with (
            patch("signal_modulation.training.train_one_epoch", return_value=train_metrics),
            patch("signal_modulation.training.evaluate", side_effect=validation_metrics),
        ):
            result = fit(
                model,
                [],
                [],
                optimizer,
                device=torch.device("cpu"),
                epochs=10,
                patience=2,
            )

        self.assertTrue(result.stopped_early)
        self.assertEqual(len(result.history), 3)
        self.assertEqual(result.best_epoch, 1)

    def test_fit_rejects_invalid_control_values(self) -> None:
        model = SimpleCNN1D(num_classes=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        with self.assertRaises(ValueError):
            fit(model, [], [], optimizer, device=torch.device("cpu"), epochs=0)
        with self.assertRaises(ValueError):
            fit(
                model,
                [],
                [],
                optimizer,
                device=torch.device("cpu"),
                epochs=1,
                patience=0,
            )


if __name__ == "__main__":
    unittest.main()
