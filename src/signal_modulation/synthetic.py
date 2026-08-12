"""Small, deterministic signal helpers used before downloading a dataset."""

from __future__ import annotations

import math

import numpy as np


def _validate_symbol_indices(symbols: np.ndarray, class_count: int) -> np.ndarray:
    values = np.asarray(symbols)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("symbols must be a non-empty one-dimensional array")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("symbols must contain integer class indices")
    if np.any(values < 0) or np.any(values >= class_count):
        raise ValueError(f"symbols must be in the range [0, {class_count - 1}]")
    return values


def generate_bpsk(symbols: np.ndarray) -> np.ndarray:
    """Map binary symbols to the BPSK constellation: 0 -> +1, 1 -> -1."""

    values = _validate_symbol_indices(symbols, class_count=2)
    return (1.0 - 2.0 * values.astype(np.float32)).astype(np.complex64)


def generate_qpsk(symbols: np.ndarray) -> np.ndarray:
    """Map four symbol indices to a unit-power Gray-ordered QPSK constellation."""

    values = _validate_symbol_indices(symbols, class_count=4)
    scale = np.float32(1.0 / math.sqrt(2.0))
    constellation = np.asarray(
        [1.0 + 1.0j, -1.0 + 1.0j, -1.0 - 1.0j, 1.0 - 1.0j],
        dtype=np.complex64,
    )
    return constellation[values] * scale


def add_awgn(
    signal: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add complex additive white Gaussian noise at the requested SNR in dB."""

    values = np.asarray(signal, dtype=np.complex64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("signal must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("signal must contain only finite values")
    if not math.isfinite(snr_db):
        raise ValueError("snr_db must be finite")

    signal_power = float(np.mean(np.abs(values) ** 2))
    if signal_power <= 0.0:
        raise ValueError("signal power must be greater than zero")

    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    component_std = math.sqrt(noise_power / 2.0)
    noise = component_std * (
        rng.standard_normal(values.shape) + 1j * rng.standard_normal(values.shape)
    )
    return (values + noise).astype(np.complex64)


def estimate_snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Estimate the measured SNR between a clean signal and its noisy version."""

    clean_values = np.asarray(clean, dtype=np.complex64)
    noisy_values = np.asarray(noisy, dtype=np.complex64)
    if clean_values.shape != noisy_values.shape or clean_values.size == 0:
        raise ValueError("clean and noisy signals must have the same non-empty shape")

    signal_power = float(np.mean(np.abs(clean_values) ** 2))
    noise_power = float(np.mean(np.abs(noisy_values - clean_values) ** 2))
    if signal_power <= 0.0:
        raise ValueError("signal power must be greater than zero")
    if noise_power == 0.0:
        return math.inf
    return 10.0 * math.log10(signal_power / noise_power)


def complex_to_iq(signal: np.ndarray) -> np.ndarray:
    """Convert a complex sequence with shape (length,) to I/Q shape (2, length)."""

    values = np.asarray(signal, dtype=np.complex64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("signal must be a non-empty one-dimensional array")
    return np.stack((values.real, values.imag), axis=0).astype(np.float32)


def iq_to_complex(iq: np.ndarray) -> np.ndarray:
    """Convert an I/Q array with shape (2, length) back to a complex sequence."""

    values = np.asarray(iq, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != 2 or values.shape[1] == 0:
        raise ValueError("iq must have shape (2, length) with a non-zero length")
    return (values[0] + 1j * values[1]).astype(np.complex64)
