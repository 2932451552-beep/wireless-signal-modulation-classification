"""Unit tests for the first I/Q and SNR learning stage."""

import unittest

import numpy as np

from signal_modulation.synthetic import (
    add_awgn,
    complex_to_iq,
    estimate_snr_db,
    generate_bpsk,
    generate_qpsk,
    iq_to_complex,
)


class SyntheticSignalTests(unittest.TestCase):
    def test_bpsk_maps_bits_to_positive_and_negative_real_symbols(self) -> None:
        actual = generate_bpsk(np.asarray([0, 1, 1, 0], dtype=np.int64))

        np.testing.assert_array_equal(
            actual,
            np.asarray([1.0, -1.0, -1.0, 1.0], dtype=np.complex64),
        )

    def test_qpsk_constellation_has_unit_average_power(self) -> None:
        signal = generate_qpsk(np.asarray([0, 1, 2, 3], dtype=np.int64))

        self.assertAlmostEqual(float(np.mean(np.abs(signal) ** 2)), 1.0, places=6)

    def test_complex_iq_round_trip_preserves_signal(self) -> None:
        signal = generate_qpsk(np.asarray([0, 3, 1, 2], dtype=np.int64))

        iq = complex_to_iq(signal)
        restored = iq_to_complex(iq)

        self.assertEqual(iq.shape, (2, 4))
        self.assertEqual(iq.dtype, np.float32)
        np.testing.assert_allclose(restored, signal)

    def test_awgn_is_close_to_requested_snr(self) -> None:
        rng = np.random.default_rng(20260811)
        symbols = rng.integers(0, 2, size=50_000, dtype=np.int64)
        clean = generate_bpsk(symbols)

        noisy = add_awgn(clean, snr_db=5.0, rng=rng)

        self.assertAlmostEqual(estimate_snr_db(clean, noisy), 5.0, delta=0.15)

    def test_higher_snr_produces_less_noise_with_equal_random_samples(self) -> None:
        clean = generate_bpsk(np.zeros(4_096, dtype=np.int64))
        low_snr = add_awgn(clean, -10.0, np.random.default_rng(7))
        high_snr = add_awgn(clean, 20.0, np.random.default_rng(7))

        low_error = float(np.mean(np.abs(low_snr - clean) ** 2))
        high_error = float(np.mean(np.abs(high_snr - clean) ** 2))

        self.assertLess(high_error, low_error)

    def test_invalid_symbol_index_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            generate_bpsk(np.asarray([0, 2], dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
