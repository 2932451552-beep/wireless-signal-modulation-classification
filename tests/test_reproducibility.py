"""Tests for central experiment reproducibility controls."""

import random
import unittest

import numpy as np
import torch

from signal_modulation.reproducibility import configure_reproducibility


class ReproducibilityTests(unittest.TestCase):
    def test_same_seed_repeats_python_numpy_and_torch_values(self) -> None:
        configure_reproducibility(123)
        first = (random.random(), np.random.random(), torch.rand(3))

        configure_reproducibility(123)
        second = (random.random(), np.random.random(), torch.rand(3))

        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        torch.testing.assert_close(first[2], second[2], rtol=0.0, atol=0.0)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())

    def test_invalid_seed_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            configure_reproducibility(-1)


if __name__ == "__main__":
    unittest.main()
