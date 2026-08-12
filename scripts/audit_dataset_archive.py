"""Audit a downloaded dataset archive without extracting or loading pickle data."""

from __future__ import annotations

import argparse
from pathlib import Path

from signal_modulation.data_integrity import audit_tar_bz2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Path to RML2016.10a.tar.bz2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = audit_tar_bz2(args.archive)

    print(f"archive={args.archive.resolve()}")
    print(f"size_bytes={audit.size_bytes}")
    print(f"sha256={audit.sha256}")
    print(f"member_count={len(audit.members)}")
    for member in audit.members:
        print(f"member={member}")


if __name__ == "__main__":
    main()
