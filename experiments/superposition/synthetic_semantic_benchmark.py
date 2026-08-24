#!/usr/bin/env python3
"""Deterministic CCSS safety benchmark over hand-authored caption spans."""

from __future__ import annotations

import argparse
import itertools
import json
import resource
import subprocess
import time
from pathlib import Path

from ztok.superposition import (
    SemanticSpan,
    SemanticSuperpositionConfig,
    build_semantic_plan,
)


def pair(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))


def git_state() -> tuple[str, bool]:
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ).strip()
    )
    return commit, dirty


def corpus() -> tuple[list[SemanticSpan], list[list[float]], set[tuple[str, str]], set[tuple[str, str]], set[str]]:
    records = [
        # True synonym / near-equivalent sets.
        ("bright", 0, "lighting_intensity", "light", [1.0, 0.0, 0.0], "attribute", ()),
        ("brilliant", 1, "lighting_intensity", "light", [0.999, 0.02, 0.0], "attribute", ()),
        ("intense", 2, "lighting_intensity", "light", [0.997, 0.04, 0.0], "attribute", ()),
        ("white", 0, "color", "light", [0.0, 1.0, 0.0], "attribute", ()),
        ("pale", 2, "color", "light", [0.02, 0.999, 0.0], "attribute", ()),
        ("luminous", 0, "emission", "light", [0.0, 0.0, 1.0], "attribute", ()),
        ("glowing", 1, "emission", "light", [0.01, 0.0, 0.999], "attribute", ()),
        ("red-dress", 3, "color", "dress", [0.2, 0.98, 0.0], "attribute", ()),
        ("crimson-gown", 4, "color", "dress", [0.22, 0.975, 0.0], "attribute", ()),
        ("over-shoulder", 3, "gaze_action", "person", [0.45, 0.1, 0.887], "action", ()),
        ("glancing-backward", 4, "gaze_action", "person", [0.46, 0.11, 0.881], "action", ()),
        # Contradictory/protected and entity-separated details.
        ("standing", 0, "posture", "person", [0.7, 0.0, 0.7], "action", ("seated",)),
        ("seated", 1, "posture", "person", [-0.7, 0.0, -0.7], "action", ()),
        ("left", 0, "left_right", "person", [0.6, 0.8, 0.0], "spatial_connector", ("right",)),
        ("right", 1, "left_right", "person", [-0.6, -0.8, 0.0], "spatial_connector", ()),
        ("one-person", 0, "count", "person", [0.3, 0.4, 0.866], "quantifier", ("two-people",)),
        ("two-people", 1, "count", "person", [-0.3, -0.4, -0.866], "quantifier", ()),
        ("bright-room", 3, "lighting_intensity", "room", [1.0, 0.0, 0.0], "attribute", ()),
        ("bright-dress", 4, "lighting_intensity", "dress", [1.0, 0.0, 0.0], "attribute", ()),
    ]
    spans: list[SemanticSpan] = []
    embeddings: list[list[float]] = []
    for index, (name, caption, role, entity, vector, kind, contradicts) in enumerate(records):
        spans.append(
            SemanticSpan(
                name,
                caption_id=caption,
                token_start=index,
                token_end=index + 1,
                byte_start=index * 2,
                byte_end=index * 2 + 1,
                text=name,
                role=role,
                entity_id=entity,
                kind=kind,
                contradicts=contradicts,
            )
        )
        embeddings.append(vector)

    expected_clusters = [
        ("bright", "brilliant", "intense"),
        ("white", "pale"),
        ("luminous", "glowing"),
        ("red-dress", "crimson-gown"),
        ("over-shoulder", "glancing-backward"),
    ]
    expected = {
        pair(left, right)
        for cluster in expected_clusters
        for left, right in itertools.combinations(cluster, 2)
    }
    forbidden = {
        pair("bright", "white"),
        pair("bright", "luminous"),
        pair("standing", "seated"),
        pair("left", "right"),
        pair("one-person", "two-people"),
        pair("bright-room", "bright-dress"),
        pair("bright", "bright-room"),
        pair("bright", "bright-dress"),
    }
    required_unique = {
        "standing", "seated", "left", "right", "one-person", "two-people",
        "bright-room", "bright-dress",
    }
    return spans, embeddings, expected, forbidden, required_unique


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()

    spans, embeddings, expected, forbidden, required_unique = corpus()
    config = SemanticSuperpositionConfig(
        default_cosine_threshold=0.94,
        minimum_score=0.94,
        never_fuse_roles=("count", "posture", "left_right"),
        verbose_diagnostics=True,
    )
    started = time.perf_counter()
    plan = build_semantic_plan(
        spans,
        embeddings,
        config,
        image_id="synthetic-semantic-safety-v1",
        original_token_count=len(spans),
    )
    elapsed = time.perf_counter() - started

    actual: set[tuple[str, str]] = set()
    examples = []
    for unit in plan["units"]:
        members = unit.get("members", [])
        actual.update(pair(a, b) for a, b in itertools.combinations(members, 2))
        if members:
            examples.append(
                {
                    "members": members,
                    "role": unit.get("role"),
                    "minimum_similarity": unit.get("minimum_pair_similarity"),
                    "support_fraction": unit.get("support_fraction"),
                }
            )

    true_positive = len(actual & expected)
    false_positive = len(actual - expected)
    false_negative = len(expected - actual)
    precision = true_positive / len(actual) if actual else 1.0
    recall = true_positive / len(expected) if expected else 1.0
    forbidden_merged = sorted(actual & forbidden)
    singleton_ids = {
        unit["span_id"]
        for unit in plan["units"]
        if "span_id" in unit and unit["kind"] in {"unique", "relation"}
    }
    retained_unique = required_unique & singleton_ids
    commit, dirty = git_state()
    report = {
        "schema": "ztok.superposition_benchmark.v1",
        "benchmark": "synthetic_semantic_safety",
        "git_commit": commit,
        "git_dirty": dirty,
        "tokenizer_identifier": "hand-authored token/span offsets",
        "vocabulary_identifier": "synthetic semantic span IDs",
        "embedding_model": "hand-authored 3D safety vectors",
        "embedding_layer": "n/a",
        "span_extraction_policy": "hand-authored word and phrase spans",
        "span_count": len(spans),
        "expected_pair_count": len(expected),
        "actual_pair_count": len(actual),
        "pair_precision": precision,
        "pair_recall": recall,
        "cluster_purity": precision,
        "contradiction_false_merge_rate": len(forbidden_merged) / len(forbidden),
        "retained_unique_detail_rate": len(retained_unique) / len(required_unique),
        "false_positive_pairs": sorted(actual - expected),
        "missed_merge_pairs": sorted(expected - actual),
        "forbidden_merged_pairs": forbidden_merged,
        "cluster_examples": examples,
        "runtime_seconds": elapsed,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "pass": (
            precision == 1.0
            and not forbidden_merged
            and len(retained_unique) == len(required_unique)
        ),
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# CCSS synthetic semantic safety benchmark",
        "",
        f"- Git commit: `{commit}` ({'dirty' if dirty else 'clean'})",
        f"- Pair precision: **{precision:.3%}**",
        f"- Pair recall: **{recall:.3%}**",
        f"- Cluster purity: **{precision:.3%}**",
        f"- Protected/contradiction false-merge rate: **{report['contradiction_false_merge_rate']:.3%}**",
        f"- High-confidence unique-detail retention: **{report['retained_unique_detail_rate']:.3%}**",
        f"- Runtime: {elapsed:.6f} seconds",
        f"- Result: **{'PASS' if report['pass'] else 'FAIL'}**",
        "",
        "## Qualitative clusters",
        "",
    ]
    for example in examples:
        lines.append(
            f"- `{example['role']}`: {', '.join(example['members'])} "
            f"(min cosine {example['minimum_similarity']:.4f})"
        )
    lines.extend(
        [
            "",
            "## Errors",
            "",
            f"- False merges: `{report['false_positive_pairs']}`",
            f"- Missed merges: `{report['missed_merge_pairs']}`",
            f"- Forbidden merges: `{report['forbidden_merged_pairs']}`",
            "",
        ]
    )
    args.markdown_output.write_text("\n".join(lines))
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
