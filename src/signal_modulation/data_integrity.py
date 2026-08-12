"""Read-only integrity checks for dataset archives before extraction."""

from __future__ import annotations

import hashlib
import pickletools
import tarfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True, slots=True)
class ArchiveAudit:
    """Summary of a dataset archive that passed structural checks."""

    size_bytes: int
    sha256: str
    members: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PickleOpcodeAudit:
    """Opcode metadata collected without constructing any pickled objects."""

    size_bytes: int
    sha256: str
    opcode_count: int
    protocols: tuple[int, ...]
    global_references: tuple[str, ...]
    constructor_opcodes: tuple[tuple[str, int], ...]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a file SHA-256 digest without loading the whole file into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_member_name(name: str) -> None:
    normalized = name.replace("\\", "/")
    member_path = PurePosixPath(normalized)
    if not normalized or member_path.is_absolute() or ".." in member_path.parts:
        raise ValueError(f"unsafe archive member path: {name!r}")
    if member_path.parts and ":" in member_path.parts[0]:
        raise ValueError(f"unsafe archive member path: {name!r}")


def audit_tar_bz2(path: Path) -> ArchiveAudit:
    """Inspect a bzip2 tar archive without extracting or unpickling its contents."""

    if not path.is_file():
        raise FileNotFoundError(path)

    member_names: list[str] = []
    with tarfile.open(path, mode="r:bz2") as archive:
        for member in archive.getmembers():
            _validate_member_name(member.name)
            if member.issym() or member.islnk():
                raise ValueError(f"archive links are not allowed: {member.name!r}")
            if not member.isfile() and not member.isdir():
                raise ValueError(f"unsupported archive member type: {member.name!r}")
            member_names.append(member.name)

    return ArchiveAudit(
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
        members=tuple(member_names),
    )


def audit_pickle_opcodes(path: Path) -> PickleOpcodeAudit:
    """Scan pickle opcodes without importing globals or reconstructing objects.

    This is a risk-reduction check, not a proof that a pickle is safe to load.
    """

    if not path.is_file():
        raise FileNotFoundError(path)

    opcode_count = 0
    protocols: set[int] = set()
    global_references: set[str] = set()
    constructor_counts: Counter[str] = Counter()
    constructor_names = {
        "BUILD",
        "INST",
        "NEWOBJ",
        "NEWOBJ_EX",
        "OBJ",
        "REDUCE",
        "STACK_GLOBAL",
        "PERSID",
        "BINPERSID",
    }

    with path.open("rb") as file_handle:
        for opcode, argument, _position in pickletools.genops(file_handle):
            opcode_count += 1
            if opcode.name == "PROTO":
                protocols.add(int(argument))
            elif opcode.name == "GLOBAL":
                global_references.add(str(argument).replace("\n", " "))
            if opcode.name in constructor_names:
                constructor_counts[opcode.name] += 1

    return PickleOpcodeAudit(
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
        opcode_count=opcode_count,
        protocols=tuple(sorted(protocols)),
        global_references=tuple(sorted(global_references)),
        constructor_opcodes=tuple(sorted(constructor_counts.items())),
    )
