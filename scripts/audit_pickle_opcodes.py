"""Scan pickle opcodes without executing or reconstructing dataset objects."""

from __future__ import annotations

import argparse
from pathlib import Path

from signal_modulation.data_integrity import audit_pickle_opcodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pickle_file", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = audit_pickle_opcodes(args.pickle_file)
    print(f"pickle={args.pickle_file.resolve()}")
    print(f"size_bytes={audit.size_bytes}")
    print(f"sha256={audit.sha256}")
    print(f"opcode_count={audit.opcode_count}")
    print(f"protocols={audit.protocols}")
    print(f"global_references={audit.global_references}")
    print(f"constructor_opcodes={audit.constructor_opcodes}")
    print("objects_constructed=false")
    print("audit_is_not_a_safety_proof=true")


if __name__ == "__main__":
    main()
