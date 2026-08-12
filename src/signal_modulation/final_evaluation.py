"""Safety checks for the one-time final evaluation of a frozen model."""

from __future__ import annotations

import hmac
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from signal_modulation.data_integrity import sha256_file


def verify_file_sha256(path: str | Path, expected_sha256: str) -> str:
    """Verify a frozen artifact before any test metrics are calculated."""

    if len(expected_sha256) != 64:
        raise ValueError("expected SHA-256 must contain 64 hexadecimal characters")
    try:
        int(expected_sha256, 16)
    except ValueError as error:
        raise ValueError("expected SHA-256 must be hexadecimal") from error
    actual_sha256 = sha256_file(path)
    if not hmac.compare_digest(actual_sha256.lower(), expected_sha256.lower()):
        raise ValueError(f"SHA-256 mismatch for frozen artifact: {Path(path).name}")
    return actual_sha256


def load_frozen_model_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    expected_sha256: str,
    expected_epoch: int,
) -> Mapping[str, Any]:
    """Load only tensor weights after checking hash and selected epoch."""

    if expected_epoch <= 0:
        raise ValueError("expected_epoch must be greater than zero")
    verify_file_sha256(checkpoint_path, expected_sha256)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a dictionary")
    if checkpoint.get("epoch") != expected_epoch:
        raise ValueError("checkpoint epoch does not match the frozen protocol")
    model_state = checkpoint.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise ValueError("checkpoint is missing model_state_dict")
    model.load_state_dict(model_state, strict=True)
    return checkpoint
