"""Tests for TrainingLogger."""
from __future__ import annotations

import os

from utils.training_logger import TrainingLogger


def test_logger_writes_events_when_tb_available(tmp_path):
    with TrainingLogger(root=tmp_path, run_name="t") as logger:
        logger.log_scalar("foo/bar", 0.5, step=0)
        logger.log_metrics({"foo/baz": 1.0, "foo/qux": 2.0}, step=1)
    files = sorted(os.listdir(logger.run_dir))
    if logger.tb_available:
        assert any(f.startswith("events.out.tfevents") for f in files)
    else:
        # In environments without tensorboard installed, the writer is a
        # no-op; the run_dir may not even be created.
        pass


def test_logger_noop_path_does_not_crash(tmp_path):
    """Logger should accept all calls regardless of TB availability."""
    with TrainingLogger(root=tmp_path, run_name="t") as logger:
        logger.log_scalar("a", 1.0, 0)
        logger.log_metrics({"a": 1.0}, 0)
        logger.log_hparams({"lr": 1e-3, "model": "X"}, {"a": 1.0})
        logger.flush()


def test_run_dir_is_unique_per_call(tmp_path):
    a = TrainingLogger(root=tmp_path, run_name="t", run_suffix="v1")
    b = TrainingLogger(root=tmp_path, run_name="t", run_suffix="v2")
    try:
        assert a.run_dir != b.run_dir
    finally:
        a.close()
        b.close()
