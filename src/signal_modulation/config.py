"""Dependency-free project path configuration for the initial skeleton."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Resolve project directories without creating or modifying them."""

    root: Path

    @classmethod
    def from_source_file(cls) -> "ProjectPaths":
        return cls(root=Path(__file__).resolve().parents[2])

    @property
    def raw_data(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed_data(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"
