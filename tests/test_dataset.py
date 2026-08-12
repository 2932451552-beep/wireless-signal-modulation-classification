"""Tests for the PyTorch I/Q Dataset and DataLoader."""

import unittest

import numpy as np
import torch

from signal_modulation.dataset import IQSignalDataset, create_data_loader


def _example_arrays(sample_count: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signals = np.zeros((sample_count, 2, 8), dtype=np.float32)
    signals[:, 0, 0] = np.arange(sample_count, dtype=np.float32)
    labels = np.arange(sample_count, dtype=np.int64) % 2
    snrs = (np.arange(sample_count, dtype=np.int64) % 3) * 10 - 10
    return signals, labels, snrs


class IQSignalDatasetTests(unittest.TestCase):
    def test_single_item_has_expected_shapes_and_dtypes(self) -> None:
        signals, labels, snrs = _example_arrays()
        dataset = IQSignalDataset(signals, labels, snrs)

        item = dataset[0]

        self.assertEqual(len(dataset), 12)
        self.assertEqual(item["signal"].shape, (2, 8))
        self.assertEqual(item["signal"].dtype, torch.float32)
        self.assertEqual(item["label"].dtype, torch.int64)
        self.assertEqual(item["snr"].dtype, torch.float32)

    def test_loader_combines_samples_into_batches(self) -> None:
        signals, labels, snrs = _example_arrays()
        dataset = IQSignalDataset(signals, labels, snrs)
        loader = create_data_loader(dataset, batch_size=4, shuffle=False)

        batch = next(iter(loader))

        self.assertEqual(batch["signal"].shape, (4, 2, 8))
        self.assertEqual(batch["label"].shape, (4,))
        self.assertEqual(batch["snr"].shape, (4,))

    def test_shuffling_is_reproducible_with_the_same_seed(self) -> None:
        signals, labels, snrs = _example_arrays()
        dataset = IQSignalDataset(signals, labels, snrs)
        first_loader = create_data_loader(dataset, batch_size=12, shuffle=True, seed=99)
        second_loader = create_data_loader(dataset, batch_size=12, shuffle=True, seed=99)

        first_ids = next(iter(first_loader))["signal"][:, 0, 0]
        second_ids = next(iter(second_loader))["signal"][:, 0, 0]

        torch.testing.assert_close(first_ids, second_ids)

    def test_invalid_signal_shape_is_rejected(self) -> None:
        _, labels, snrs = _example_arrays()
        invalid_signals = np.zeros((12, 8), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "shape"):
            IQSignalDataset(invalid_signals, labels, snrs)

    def test_non_finite_signal_is_rejected(self) -> None:
        signals, labels, snrs = _example_arrays()
        signals[0, 0, 0] = np.nan

        with self.assertRaisesRegex(ValueError, "finite"):
            IQSignalDataset(signals, labels, snrs)

    def test_label_count_must_match_signal_count(self) -> None:
        signals, labels, snrs = _example_arrays()

        with self.assertRaisesRegex(ValueError, "match"):
            IQSignalDataset(signals, labels[:-1], snrs)

    def test_invalid_batch_size_is_rejected(self) -> None:
        signals, labels, snrs = _example_arrays()
        dataset = IQSignalDataset(signals, labels, snrs)

        with self.assertRaises(ValueError):
            create_data_loader(dataset, batch_size=0, shuffle=False)


if __name__ == "__main__":
    unittest.main()
