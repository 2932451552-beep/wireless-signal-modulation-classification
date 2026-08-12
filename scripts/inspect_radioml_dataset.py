"""Restricted-load and validate the official RML2016.10A dataset structure."""

from __future__ import annotations

import argparse
from pathlib import Path

from signal_modulation.radioml import (
    audit_radioml_dataset,
    load_restricted_radioml_pickle,
    validate_radioml_2016_10a_profile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pickle_file", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_restricted_radioml_pickle(args.pickle_file)
    summary = audit_radioml_dataset(dataset)
    validate_radioml_2016_10a_profile(summary)

    print(f"group_count={summary.group_count}")
    print(f"total_samples={summary.total_samples}")
    print(f"modulations={summary.modulations}")
    print(f"snrs={summary.snrs}")
    print(f"shapes={summary.shapes}")
    print(f"dtypes={summary.dtypes}")
    print(f"samples_per_modulation={summary.samples_per_modulation}")
    print(f"samples_per_snr={summary.samples_per_snr}")
    print("all_values_finite=true")
    print("official_profile_valid=true")


if __name__ == "__main__":
    main()
