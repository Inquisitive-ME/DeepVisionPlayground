"""Render the bucketed multi-object metrics from a trained model into a table.

Reads the ``results.json`` written by ``scripts.run_training`` for a multi
task, pulls out the ``final_metrics_by_count`` and ``final_metrics_by_size``
sweeps, and prints a clean Markdown-style matrix:

- (shape count vs pixel threshold) for recall@T
- (shape count vs metric) for the headline metrics
- (shape size range vs metric) for the same

Optionally writes the same as a CSV for spreadsheet pasting.

    python -m scripts.sweep_buckets runs/<run_dir> [--csv out.csv]

If the run was a single-model training we also display the within-run
``by_size_*`` and ``by_count_*`` buckets, which slice the *same* val
distribution by per-image and per-object characteristics — i.e. for a
val set generated with num_shapes=(0,5), how does the model do on
images that happened to draw n=4-5 vs n=1-2.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

HEADLINE_KEYS = (
    ("multi/mean_matched_center_px", "mean_px"),
    ("multi/median_matched_center_px", "median_px"),
    ("multi/matched_class_accuracy", "class_acc"),
    ("multi/cardinality_error", "card_err"),
    ("multi/map_center", "map_center"),
    ("multi/pearson_cx", "pearson_cx"),
    ("multi/pearson_cy", "pearson_cy"),
)
RECALL_THRESHOLDS = (2, 4, 8, 16)


def fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def render_matrix(rows: list[list[str]]) -> str:
    """Format a rectangular array of strings as a fixed-width table."""
    if not rows:
        return ""
    widths = [max(len(str(r[c])) for r in rows) for c in range(len(rows[0]))]
    out = []
    for i, row in enumerate(rows):
        line = "  ".join(str(c).ljust(widths[j]) for j, c in enumerate(row))
        out.append(line)
        if i == 0:
            out.append("  ".join("-" * w for w in widths))
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path, help="Run directory under runs/")
    p.add_argument("--csv", type=Path, help="Optional CSV output path")
    args = p.parse_args()

    results_path = args.run_dir / "results.json"
    if not results_path.exists():
        print(f"!! no results.json in {args.run_dir}")
        return 1
    with open(results_path) as f:
        results = json.load(f)

    cfg = results.get("config", {})
    print(f"=== Sweep summary for {args.run_dir} ===")
    print(f"task={cfg.get('task')}  encoder={cfg.get('encoder')}  "
          f"epochs={cfg.get('epochs')}  n_params={results.get('n_params'):,}")
    print()

    by_count = results.get("final_metrics_by_count", {})
    by_size = results.get("final_metrics_by_size", {})
    final = results.get("final_metrics", {})

    # Within-run buckets sliced from the training val set.
    print("--- Within-run by-count buckets (training val distribution) ---")
    inrun_count_rows = [["bucket", "n_gt", "mean_px", "recall@2px", "recall@4px", "recall@8px", "recall@16px"]]
    for tag in ("n0", "n1", "n2", "n3-5", "n6-10", "n11-20", "n21+"):
        n_gt = final.get(f"multi/by_count/{tag}/n_gt")
        if n_gt is None or n_gt == 0:
            continue
        inrun_count_rows.append([
            tag,
            f"{int(n_gt)}",
            fmt(final.get(f"multi/by_count/{tag}/mean_center_px")),
            *[fmt(final.get(f"multi/by_count/{tag}/recall@{t}px")) for t in RECALL_THRESHOLDS],
        ])
    print(render_matrix(inrun_count_rows))
    print()

    print("--- Within-run by-size buckets (training val distribution) ---")
    inrun_size_rows = [["bucket", "n_gt", "mean_px", "recall@2px", "recall@4px", "recall@8px", "recall@16px"]]
    for tag in ("xs", "sm", "md", "lg"):
        n_gt = final.get(f"multi/by_size/{tag}/n_gt")
        if n_gt is None or n_gt == 0:
            continue
        inrun_size_rows.append([
            tag,
            f"{int(n_gt)}",
            fmt(final.get(f"multi/by_size/{tag}/mean_center_px")),
            *[fmt(final.get(f"multi/by_size/{tag}/recall@{t}px")) for t in RECALL_THRESHOLDS],
        ])
    print(render_matrix(inrun_size_rows))
    print()

    # Cross-distribution sweeps.
    if by_count:
        print("--- Cross-distribution shape-count sweep (separate val sets) ---")
        rows = [["count", *[k[1] for k in HEADLINE_KEYS]]]
        for n_str in sorted(by_count, key=lambda x: int(x)):
            entry = by_count[n_str]
            rows.append([f"n={n_str}", *[fmt(entry.get(k[0])) for k in HEADLINE_KEYS]])
        print(render_matrix(rows))
        print()

    if by_size:
        print("--- Cross-distribution shape-size sweep (separate val sets) ---")
        rows = [["size_range", *[k[1] for k in HEADLINE_KEYS]]]
        for key in sorted(by_size, key=lambda s: int(s.split("-")[0])):
            entry = by_size[key]
            rows.append([f"{key} px", *[fmt(entry.get(k[0])) for k in HEADLINE_KEYS]])
        print(render_matrix(rows))
        print()

    by_overlap = results.get("final_metrics_by_overlap", {})
    if by_overlap:
        print("--- Cross-distribution overlap sweep (separate val sets) ---")
        rows = [["max_overlap", *[k[1] for k in HEADLINE_KEYS]]]
        for key in sorted(by_overlap, key=float):
            entry = by_overlap[key]
            rows.append([f"{key}", *[fmt(entry.get(k[0])) for k in HEADLINE_KEYS]])
        print(render_matrix(rows))
        print()

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["section", "row", *[k[1] for k in HEADLINE_KEYS]])
            for n_str, entry in sorted(by_count.items(), key=lambda x: int(x[0])):
                w.writerow(["count", f"n={n_str}", *[entry.get(k[0]) for k in HEADLINE_KEYS]])
            for key in sorted(by_size, key=lambda s: int(s.split("-")[0])):
                entry = by_size[key]
                w.writerow(["size", key, *[entry.get(k[0]) for k in HEADLINE_KEYS]])
        print(f"csv written to {args.csv}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
