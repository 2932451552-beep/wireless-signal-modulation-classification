"""Tests for deterministic modulation/SNR stratified splitting."""

import unittest
from collections import Counter

import numpy as np

from signal_modulation.splitting import (
    DatasetSplit,
    assert_split_integrity,
    stratified_split_indices,
    stratified_subsample_indices,
)


def _small_balanced_metadata() -> tuple[np.ndarray, np.ndarray]:
    labels: list[int] = []
    snrs: list[int] = []
    for label in (0, 1):
        for snr in (-10, 0, 10):
            labels.extend([label] * 10)
            snrs.extend([snr] * 10)
    return np.asarray(labels, dtype=np.int64), np.asarray(snrs, dtype=np.int64)


def _stratum_counts(
    indices: np.ndarray,
    labels: np.ndarray,
    snrs: np.ndarray,
) -> Counter[tuple[int, int]]:
    return Counter((int(labels[index]), int(snrs[index])) for index in indices)


class StratifiedSplitTests(unittest.TestCase):
    def test_every_stratum_uses_requested_ratios(self) -> None:
        labels, snrs = _small_balanced_metadata()

        split = stratified_split_indices(
            labels,
            snrs,
            train_fraction=0.60,
            validation_fraction=0.20,
            seed=42,
        )

        self.assertEqual(set(_stratum_counts(split.train, labels, snrs).values()), {6})
        self.assertEqual(set(_stratum_counts(split.validation, labels, snrs).values()), {2})
        self.assertEqual(set(_stratum_counts(split.test, labels, snrs).values()), {2})

    def test_split_is_complete_and_has_no_overlap(self) -> None:
        labels, snrs = _small_balanced_metadata()
        split = stratified_split_indices(labels, snrs, seed=7)

        assert_split_integrity(split, sample_count=labels.size)

        self.assertTrue(set(split.train).isdisjoint(split.validation))
        self.assertTrue(set(split.train).isdisjoint(split.test))
        self.assertTrue(set(split.validation).isdisjoint(split.test))

    def test_same_seed_reproduces_same_indices(self) -> None:
        labels, snrs = _small_balanced_metadata()

        first = stratified_split_indices(labels, snrs, seed=123)
        second = stratified_split_indices(labels, snrs, seed=123)

        np.testing.assert_array_equal(first.train, second.train)
        np.testing.assert_array_equal(first.validation, second.validation)
        np.testing.assert_array_equal(first.test, second.test)

    def test_different_seed_changes_training_indices(self) -> None:
        labels, snrs = _small_balanced_metadata()

        first = stratified_split_indices(labels, snrs, seed=1)
        second = stratified_split_indices(labels, snrs, seed=2)

        self.assertFalse(np.array_equal(first.train, second.train))

    def test_invalid_fractions_are_rejected(self) -> None:
        labels, snrs = _small_balanced_metadata()

        with self.assertRaises(ValueError):
            stratified_split_indices(
                labels,
                snrs,
                train_fraction=0.90,
                validation_fraction=0.20,
            )

    def test_duplicate_indices_fail_integrity_check(self) -> None:
        split = DatasetSplit(
            train=np.asarray([0, 1], dtype=np.int64),
            validation=np.asarray([1], dtype=np.int64),
            test=np.asarray([2], dtype=np.int64),
        )

        with self.assertRaisesRegex(ValueError, "duplicate or missing"):
            assert_split_integrity(split, sample_count=4)

    def test_too_small_stratum_is_rejected(self) -> None:
        labels = np.asarray([0, 0], dtype=np.int64)
        snrs = np.asarray([0, 0], dtype=np.int64)

        with self.assertRaisesRegex(ValueError, "too small"):
            stratified_split_indices(labels, snrs)

    def test_stratified_subsample_is_balanced_and_reproducible(self) -> None:
        labels, snrs = _small_balanced_metadata()
        candidates = np.arange(labels.size, dtype=np.int64)

        first = stratified_subsample_indices(
            candidates,
            labels,
            snrs,
            samples_per_stratum=3,
            allowed_snrs=(0, 10),
            seed=5,
        )
        second = stratified_subsample_indices(
            candidates,
            labels,
            snrs,
            samples_per_stratum=3,
            allowed_snrs=(0, 10),
            seed=5,
        )

        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.size, 12)
        self.assertEqual(set(_stratum_counts(first, labels, snrs).values()), {3})
        self.assertEqual(set(snrs[first]), {0, 10})

    def test_stratified_subsample_rejects_insufficient_candidates(self) -> None:
        labels, snrs = _small_balanced_metadata()
        candidates = np.arange(labels.size, dtype=np.int64)

        with self.assertRaisesRegex(ValueError, "fewer than"):
            stratified_subsample_indices(
                candidates,
                labels,
                snrs,
                samples_per_stratum=11,
            )


if __name__ == "__main__":
    unittest.main()
