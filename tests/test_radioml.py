"""Tests for restricted RadioML loading and schema validation."""

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from signal_modulation.radioml import (
    RadioMLSummary,
    audit_radioml_dataset,
    load_restricted_radioml_pickle,
    validate_radioml_2016_10a_profile,
)


class _ForbiddenPickleObject:
    def __reduce__(self):
        return eval, ("1 + 1",)


class RadioMLTests(unittest.TestCase):
    def test_restricted_loader_accepts_expected_numpy_arrays(self) -> None:
        dataset = {
            ("BPSK", 0): np.zeros((3, 2, 128), dtype=np.float32),
            ("QPSK", 0): np.ones((3, 2, 128), dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            pickle_path = Path(temporary_directory) / "dataset.pkl"
            pickle_path.write_bytes(pickle.dumps(dataset, protocol=4))

            loaded = load_restricted_radioml_pickle(pickle_path)
            summary = audit_radioml_dataset(loaded)

        self.assertEqual(summary.group_count, 2)
        self.assertEqual(summary.total_samples, 6)
        self.assertEqual(summary.modulations, ("BPSK", "QPSK"))

    def test_restricted_loader_rejects_unapproved_global(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            pickle_path = Path(temporary_directory) / "unsafe.pkl"
            pickle_path.write_bytes(pickle.dumps(_ForbiddenPickleObject(), protocol=2))

            with self.assertRaisesRegex(pickle.UnpicklingError, "not allowed"):
                load_restricted_radioml_pickle(pickle_path)

    def test_schema_rejects_wrong_signal_shape(self) -> None:
        dataset = {("BPSK", 0): np.zeros((3, 128), dtype=np.float32)}

        with self.assertRaisesRegex(ValueError, "shape"):
            audit_radioml_dataset(dataset)

    def test_schema_rejects_non_finite_values(self) -> None:
        signals = np.zeros((3, 2, 128), dtype=np.float32)
        signals[0, 0, 0] = np.nan

        with self.assertRaisesRegex(ValueError, "finite"):
            audit_radioml_dataset({("BPSK", 0): signals})

    def test_official_profile_rejects_incomplete_summary(self) -> None:
        incomplete = RadioMLSummary(
            group_count=1,
            total_samples=1,
            modulations=("BPSK",),
            snrs=(0,),
            shapes=((1, 2, 128),),
            dtypes=("float32",),
            samples_per_modulation=(("BPSK", 1),),
            samples_per_snr=((0, 1),),
        )

        with self.assertRaisesRegex(ValueError, "11 modulation"):
            validate_radioml_2016_10a_profile(incomplete)


if __name__ == "__main__":
    unittest.main()
