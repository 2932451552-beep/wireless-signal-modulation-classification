"""Central random-seed and deterministic-computation controls."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def configure_reproducibility(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch and optionally require deterministic kernels."""

    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)
