"""Tests for read-only dataset archive auditing."""

import io
import pickle
import tarfile
import tempfile
import unittest
from pathlib import Path

from signal_modulation.data_integrity import (
    audit_pickle_opcodes,
    audit_tar_bz2,
    sha256_file,
)


class DataIntegrityTests(unittest.TestCase):
    def test_valid_archive_is_audited_without_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive_path = root / "dataset.tar.bz2"
            payload = b"safe test payload"

            with tarfile.open(archive_path, mode="w:bz2") as archive:
                member = tarfile.TarInfo("RML2016.10a_dict.pkl")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))

            audit = audit_tar_bz2(archive_path)

            self.assertEqual(audit.members, ("RML2016.10a_dict.pkl",))
            self.assertEqual(audit.sha256, sha256_file(archive_path))
            self.assertGreater(audit.size_bytes, 0)
            self.assertFalse((root / "RML2016.10a_dict.pkl").exists())

    def test_parent_directory_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "unsafe.tar.bz2"
            payload = b"unsafe"

            with tarfile.open(archive_path, mode="w:bz2") as archive:
                member = tarfile.TarInfo("../outside.txt")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))

            with self.assertRaisesRegex(ValueError, "unsafe archive member path"):
                audit_tar_bz2(archive_path)

    def test_archive_links_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "link.tar.bz2"

            with tarfile.open(archive_path, mode="w:bz2") as archive:
                member = tarfile.TarInfo("dataset-link")
                member.type = tarfile.SYMTYPE
                member.linkname = "RML2016.10a_dict.pkl"
                archive.addfile(member)

            with self.assertRaisesRegex(ValueError, "archive links are not allowed"):
                audit_tar_bz2(archive_path)

    def test_invalid_hash_chunk_size_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            file_path = Path(temporary_directory) / "file.bin"
            file_path.write_bytes(b"content")

            with self.assertRaises(ValueError):
                sha256_file(file_path, chunk_size=0)

    def test_pickle_opcode_audit_does_not_construct_objects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            pickle_path = Path(temporary_directory) / "sample.pkl"
            pickle_path.write_bytes(pickle.dumps(len, protocol=2))

            audit = audit_pickle_opcodes(pickle_path)

            self.assertGreater(audit.opcode_count, 0)
            self.assertIn(2, audit.protocols)
            self.assertTrue(any("len" in reference for reference in audit.global_references))

    def test_missing_pickle_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing_path = Path(temporary_directory) / "missing.pkl"

            with self.assertRaises(FileNotFoundError):
                audit_pickle_opcodes(missing_path)


if __name__ == "__main__":
    unittest.main()
