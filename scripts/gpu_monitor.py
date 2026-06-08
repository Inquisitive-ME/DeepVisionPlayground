"""Lightweight GPU sampler.

Polls ``nvidia-smi`` at ~1 Hz, writes a CSV row per sample, and prints a
periodic summary. Stop with Ctrl-C; on exit it prints final stats so a
background invocation can be tail-grepped.

    python -m scripts.gpu_monitor --out gpu_log.csv --interval 1.0
"""
from __future__ import annotations

import argparse
import csv
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

QUERY_FIELDS = (
    "timestamp",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
)


def sample_once() -> dict[str, str] | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(QUERY_FIELDS)}",
                "--format=csv,noheader,nounits",
                "-i", "0",
            ],
            stderr=subprocess.STDOUT, text=True, timeout=2.0,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None
    parts = [p.strip() for p in out.strip().split(",")]
    if len(parts) != len(QUERY_FIELDS):
        return None
    return dict(zip(QUERY_FIELDS, parts))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("gpu_log.csv"))
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--print-every", type=int, default=10,
                   help="Print a summary line every N samples")
    args = p.parse_args()

    samples_util: list[float] = []
    samples_mem: list[float] = []

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stop = False

    def _stop(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=QUERY_FIELDS)
        writer.writeheader()
        sample_count = 0
        while not stop:
            row = sample_once()
            if row is not None:
                writer.writerow(row)
                f.flush()
                try:
                    samples_util.append(float(row["utilization.gpu"]))
                    samples_mem.append(float(row["memory.used"]))
                except ValueError:
                    pass
                sample_count += 1
                if sample_count % args.print_every == 0 and samples_util:
                    util_recent = samples_util[-args.print_every:]
                    mem_recent = samples_mem[-args.print_every:]
                    print(
                        f"[{sample_count}] util mean={statistics.mean(util_recent):.0f}%"
                        f" max={max(util_recent):.0f}%"
                        f" mem mean={statistics.mean(mem_recent):.0f}MiB"
                        f" max={max(mem_recent):.0f}MiB",
                        flush=True,
                    )
            time.sleep(args.interval)

    if samples_util:
        print()
        print(f"=== GPU monitor summary ({len(samples_util)} samples) ===")
        print(f"util gpu : mean={statistics.mean(samples_util):.1f}%"
              f" median={statistics.median(samples_util):.1f}%"
              f" max={max(samples_util):.0f}%"
              f" >50%: {100*sum(1 for u in samples_util if u > 50)/len(samples_util):.1f}%")
        print(f"mem used : mean={statistics.mean(samples_mem):.0f}MiB"
              f" max={max(samples_mem):.0f}MiB")
        print(f"csv      : {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
