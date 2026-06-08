"""TensorBoard logger for training runs.

Falls back to a no-op writer if ``tensorboard`` isn't installed, so the
training scripts run unchanged even without the optional dependency.
"""
from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriter

    _TB_AVAILABLE = True
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dep
    _SummaryWriter = None  # type: ignore[assignment, misc]
    _TB_AVAILABLE = False


class _NoOpWriter:
    """Stand-in when tensorboard isn't installed; methods are no-ops."""

    def add_scalar(self, *args: Any, **kwargs: Any) -> None: ...
    def add_scalars(self, *args: Any, **kwargs: Any) -> None: ...
    def add_image(self, *args: Any, **kwargs: Any) -> None: ...
    def add_figure(self, *args: Any, **kwargs: Any) -> None: ...
    def add_hparams(self, *args: Any, **kwargs: Any) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


class TrainingLogger:
    """Thin wrapper around SummaryWriter that knows how to log our metrics.

    Use as a context manager so the writer is closed cleanly:

        with TrainingLogger("runs", "train_multi") as logger:
            logger.log_scalar("train/loss", loss, step=epoch)
    """

    def __init__(
        self,
        root: str | Path = "runs",
        run_name: str | None = None,
        run_suffix: str | None = None,
    ) -> None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        parts = [p for p in (run_name, ts, run_suffix) if p]
        self.run_dir = Path(root) / "_".join(parts)
        if _TB_AVAILABLE and _SummaryWriter is not None:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.writer: Any = _SummaryWriter(log_dir=str(self.run_dir))
        else:
            self.writer = _NoOpWriter()
        self.tb_available = _TB_AVAILABLE

    def __enter__(self) -> "TrainingLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, float(value), step)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        for tag, value in metrics.items():
            self.log_scalar(tag, value, step)

    def log_hparams(
        self,
        hparams: dict[str, Any],
        final_metrics: dict[str, float],
    ) -> None:
        # SummaryWriter requires scalars/strings — coerce common types.
        clean = {
            k: (v if isinstance(v, (int, float, str, bool)) else str(v))
            for k, v in hparams.items()
        }
        self.writer.add_hparams(clean, final_metrics)

    def log_figure(self, tag: str, figure: Any, step: int) -> None:
        """Log a matplotlib Figure. No-op if TB isn't available."""
        if not self.tb_available:
            return
        # Convert via PNG buffer so we don't depend on tb.utils internals.
        buf = io.BytesIO()
        figure.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        from PIL import Image

        with Image.open(buf) as img:
            arr = np.asarray(img.convert("RGB"))
        # SummaryWriter expects (C, H, W) uint8.
        self.writer.add_image(tag, torch.from_numpy(arr).permute(2, 0, 1), step)

    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()
