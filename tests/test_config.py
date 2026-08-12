"""Tests for the dependency-free Phase 0 skeleton."""

import unittest
from pathlib import Path

from signal_modulation import ProjectPaths


class ProjectPathsTests(unittest.TestCase):
    def test_expected_directories_are_resolved_under_project_root(self) -> None:
        paths = ProjectPaths(root=Path("project-root"))

        self.assertEqual(paths.raw_data, Path("project-root/data/raw"))
        self.assertEqual(paths.processed_data, Path("project-root/data/processed"))
        self.assertEqual(paths.artifacts, Path("project-root/artifacts"))

    def test_configuration_does_not_create_directories(self) -> None:
        paths = ProjectPaths(root=Path("directory-that-should-not-be-created"))

        _ = paths.raw_data

        self.assertFalse(paths.root.exists())


if __name__ == "__main__":
    unittest.main()
