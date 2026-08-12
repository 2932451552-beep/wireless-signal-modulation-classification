"""Tests for memory-conscious RadioML sample access and partitions."""

import unittest

import numpy as np
import torch

from signal_modulation.dataset import create_data_loader
from signal_modulation.radioml_dataset import (
    RadioMLTorchDataset,
    create_radioml_partitions,
)
from signal_modulation.splitting import assert_split_integrity


def _grouped_dataset(samples_per_group: int = 20):
    grouped = {}
    for modulation_index, modulation in enumerate(("BPSK", "QPSK")):
        for snr in (-10, 0):
            signals = np.zeros((samples_per_group, 2, 128), dtype=np.float32)
            signals[:, 0, 0] = modulation_index
            signals[:, 1, 0] = snr
            grouped[(modulation, snr)] = signals
    return grouped


class RadioMLDatasetTests(unittest.TestCase):
    def test_sample_mapping_returns_signal_class_and_snr(self) -> None:
        grouped = _grouped_dataset()
        dataset = RadioMLTorchDataset(grouped)

        first = dataset[0]

        self.assertEqual(len(dataset), 80)
        self.assertEqual(dataset.modulations, ("BPSK", "QPSK"))
        self.assertEqual(first["signal"].shape, (2, 128))
        self.assertEqual(first["signal"].dtype, torch.float32)
        self.assertEqual(first["label"].item(), 0)
        self.assertEqual(first["snr"].item(), -10.0)

    def test_signal_tensor_shares_the_original_numpy_storage(self) -> None:
        grouped = _grouped_dataset()
        dataset = RadioMLTorchDataset(grouped)

        tensor = dataset[0]["signal"]

        self.assertTrue(np.shares_memory(tensor.numpy(), grouped[("BPSK", -10)]))

    def test_partitions_are_complete_disjoint_and_stratified(self) -> None:
        dataset = RadioMLTorchDataset(_grouped_dataset())
        partitions = create_radioml_partitions(dataset, seed=99)

        self.assertEqual(len(partitions.train), 56)
        self.assertEqual(len(partitions.validation), 12)
        self.assertEqual(len(partitions.test), 12)
        assert_split_integrity(partitions.indices, sample_count=len(dataset))

        for label in (0, 1):
            for snr in (-10, 0):
                mask = (dataset.labels == label) & (dataset.snrs == snr)
                group_indices = set(np.flatnonzero(mask))
                self.assertEqual(len(group_indices & set(partitions.indices.train)), 14)
                self.assertEqual(len(group_indices & set(partitions.indices.validation)), 3)
                self.assertEqual(len(group_indices & set(partitions.indices.test)), 3)

    def test_partition_data_loader_produces_training_batch_shape(self) -> None:
        dataset = RadioMLTorchDataset(_grouped_dataset())
        partitions = create_radioml_partitions(dataset)
        loader = create_data_loader(
            partitions.train,
            batch_size=16,
            shuffle=False,
        )

        batch = next(iter(loader))

        self.assertEqual(batch["signal"].shape, (16, 2, 128))
        self.assertEqual(batch["label"].shape, (16,))
        self.assertEqual(batch["snr"].shape, (16,))

    def test_invalid_sample_index_is_rejected(self) -> None:
        dataset = RadioMLTorchDataset(_grouped_dataset())

        with self.assertRaises(IndexError):
            _ = dataset[-1]
        with self.assertRaises(IndexError):
            _ = dataset[len(dataset)]


if __name__ == "__main__":
    unittest.main()
