#!/usr/bin/env python3
"""Score dataset-superposition mapping against curated cross-example pairs."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

from map_dataset import load_exchange
from superposition.dataset_mapping import (
    DatasetMappingConfig,
    build_dataset_superposition_plan,
)


def _identity(raw: dict[str, Any]) -> tuple[int, str]:
    return int(raw["example_id"]), str(raw["span_id"])


def _pair(
    left: tuple[int, str], right: tuple[int, str]
) -> tuple[tuple[int, str], tuple[int, str]]:
    return tuple(sorted((left, right)))  # type: ignore[return-value]


def benchmark(
    payload: dict[str, Any], *, cosine_threshold: float | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    spans, edges, embeddings = load_exchange(payload)
    plan = build_dataset_superposition_plan(
        spans,
        edges,
        embeddings,
        DatasetMappingConfig(
            cosine_threshold=(
                cosine_threshold
                if cosine_threshold is not None
                else float(payload.get("cosine_threshold", 0.92))
            ),
            minimum_aligned_spans=int(payload.get("minimum_aligned_spans", 2)),
            minimum_alignment_coverage=float(
                payload.get("minimum_alignment_coverage", 0.75)
            ),
            verbose_diagnostics=True,
        ),
    )
    predicted = set()
    for cluster in plan["clusters"]:
        members = [
            (int(member["example_id"]), str(member["span_id"]))
            for member in cluster["members"]
        ]
        predicted.update(_pair(left, right) for left, right in combinations(members, 2))
    expected = {
        _pair(_identity(item["left"]), _identity(item["right"]))
        for item in payload.get("expected_pairs", [])
    }
    forbidden = {
        _pair(_identity(item["left"]), _identity(item["right"]))
        for item in payload.get("forbidden_pairs", [])
    }
    true_positive = len(predicted & expected)
    false_positive = len(predicted - expected)
    false_negative = len(expected - predicted)
    precision = true_positive / len(predicted) if predicted else 1.0
    recall = true_positive / len(expected) if expected else 1.0
    expected_unique = {_identity(item) for item in payload.get("expected_unique", [])}
    retained_unique = {
        (int(item["example_id"]), str(item["span_id"])) for item in plan["unique_spans"]
    }
    unique_retention = (
        len(expected_unique & retained_unique) / len(expected_unique)
        if expected_unique
        else 1.0
    )
    metrics = {
        "schema": "ztok.dataset_superposition.benchmark.v1",
        "dataset_id": payload.get("dataset_id"),
        "cosine_threshold": (
            cosine_threshold
            if cosine_threshold is not None
            else float(payload.get("cosine_threshold", 0.92))
        ),
        "embedding_adapter": payload.get("embedding_adapter"),
        "expected_pair_count": len(expected),
        "predicted_pair_count": len(predicted),
        "true_positive_pairs": true_positive,
        "false_positive_pairs": false_positive,
        "false_negative_pairs": false_negative,
        "pair_precision": precision,
        "pair_recall": recall,
        "forbidden_false_merges": len(predicted & forbidden),
        "expected_unique_count": len(expected_unique),
        "retained_unique_count": len(expected_unique & retained_unique),
        "unique_detail_retention": unique_retention,
        "superbatch_count": len(plan["superbatches"]),
    }
    return metrics, plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cosine-threshold", type=float)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    metrics, plan = benchmark(payload, cosine_threshold=args.cosine_threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n"
    )
    markdown = [
        "# Dataset semantic-superposition mapping benchmark",
        "",
        f"- Dataset: `{metrics['dataset_id']}`",
        f"- Cosine threshold: {metrics['cosine_threshold']:.3f}",
        f"- Pair precision: {metrics['pair_precision']:.3f}",
        f"- Pair recall: {metrics['pair_recall']:.3f}",
        f"- Forbidden false merges: {metrics['forbidden_false_merges']}",
        f"- Unique-detail retention: {metrics['unique_detail_retention']:.3f}",
        f"- Superbatches produced: {metrics['superbatch_count']}",
        "",
        "This is a hand-authored structural plumbing benchmark. It does not validate a real contextual encoder.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(markdown) + "\n")
    print("\n".join(markdown))


if __name__ == "__main__":
    main()
