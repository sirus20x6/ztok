#!/usr/bin/env python3
"""Sweep semantic thresholds for one contextualized dataset exchange."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_dataset_mapping import benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, action="append", required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    rows = []
    for threshold in args.threshold:
        metrics, _ = benchmark(payload, cosine_threshold=threshold)
        rows.append(metrics)
    report = {
        "schema": "ztok.dataset_superposition.threshold_sweep.v1",
        "dataset_id": payload.get("dataset_id"),
        "embedding_adapter": payload.get("embedding_adapter"),
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sweep.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Dataset mapping threshold sweep",
        "",
        "| Threshold | Precision | Recall | Predicted pairs | Forbidden merges | Unique retention | Superbatches |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['cosine_threshold']:.3f} | {row['pair_precision']:.3f} | "
            f"{row['pair_recall']:.3f} | {row['predicted_pair_count']} | "
            f"{row['forbidden_false_merges']} | {row['unique_detail_retention']:.3f} | "
            f"{row['superbatch_count']} |"
        )
    lines.extend(
        [
            "",
            "This curated sweep validates the mapper and selected teacher layer, not automatic role parsing or corpus-scale retrieval.",
        ]
    )
    (args.output_dir / "sweep.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
