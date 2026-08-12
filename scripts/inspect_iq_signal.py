"""Print a small, reproducible I/Q and SNR demonstration."""

from __future__ import annotations

import numpy as np

from signal_modulation.synthetic import (
    add_awgn,
    complex_to_iq,
    estimate_snr_db,
    generate_bpsk,
)


def main() -> None:
    """Generate one BPSK example and compare three noise levels."""

    rng = np.random.default_rng(20260811)
    symbols = rng.integers(0, 2, size=128, dtype=np.int64)
    clean = generate_bpsk(symbols)

    print(f"sample_length={clean.size}")
    print(f"clean_iq_shape={complex_to_iq(clean).shape}")
    print(f"first_8_bits={symbols[:8].tolist()}")
    print(f"first_8_clean_symbols={clean[:8].real.astype(int).tolist()}")

    for snr_db in (20.0, 0.0, -10.0):
        noisy = add_awgn(clean, snr_db=snr_db, rng=rng)
        iq = complex_to_iq(noisy)
        measured = estimate_snr_db(clean, noisy)
        print(
            f"requested_snr_db={snr_db:>5.1f} "
            f"measured_snr_db={measured:>6.2f} "
            f"iq_shape={iq.shape} "
            f"first_iq=({iq[0, 0]:.3f}, {iq[1, 0]:.3f})"
        )


if __name__ == "__main__":
    main()
