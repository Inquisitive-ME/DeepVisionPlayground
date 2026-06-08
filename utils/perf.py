"""Small helpers that turn on the obvious GPU performance knobs.

These default to "fast and slightly nondeterministic", which is the right
choice when the goal is comparing architectures by speed and final metric.
Call ``configure_for_speed()`` once at process start.
"""
from __future__ import annotations

import torch


def configure_for_speed() -> None:
    """Enable cuDNN autotuner and TF32 matmul on Ampere+ GPUs.

    - ``cudnn.benchmark = True``: picks the fastest conv kernel for fixed
      input shapes, which is exactly our case (256x256 RGB tensors).
    - ``set_float32_matmul_precision("high")``: lets PyTorch run matmuls in
      TF32 on Ampere/Hopper, ~2x speedup on linear / FC layers with
      negligible accuracy impact.
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        # TF32 matmuls give a free win on Ampere (compute capability 8.x).
        torch.set_float32_matmul_precision("high")


def pick_device() -> torch.device:
    """Cuda if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
