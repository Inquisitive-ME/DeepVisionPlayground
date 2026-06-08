"""Shared pytest fixtures and config for the DeepVisionPlayground test suite.

The suite is meant to run fast on CPU: small images (32x32–64x64), tiny
batches, no plotting. Anything that takes more than a couple of seconds
on a laptop belongs in a separate, opt-in benchmark suite — not here.
"""
from __future__ import annotations

import os

import matplotlib
import pytest
import torch

# Use a non-interactive matplotlib backend so importing visualization
# modules in tests doesn't open windows or hang in CI.
matplotlib.use("Agg")
os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture(autouse=True)
def deterministic_torch():
    """Seed torch and limit threads so tests are reproducible and quick."""
    torch.manual_seed(0)
    torch.set_num_threads(1)
    yield


@pytest.fixture
def tiny_image_size() -> tuple[int, int]:
    # Big enough that the default shape_size_range=(20, 90) and the lines
    # dataset's default min_spacing=30 still fit, small enough that tests
    # are fast on CPU.
    return (64, 64)
