#!/usr/bin/env python3
"""Run auditable CCSS compression/ablation benchmarks on exchange documents.

Input is JSONL with one ``ztok.semantic_spans.v1`` document per image, or a
single JSON document/array. Contextual embeddings must already be present.
This script intentionally does not run Qwen or an image-grounding model.
"""

from __future__ import annotations

import argparse
import collections
import json
import resource
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

from ztok.superposition import build_semantic_plan_from_exchange


def git_state(root: Path) -> tuple[str, bool]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ).strip()
    )
    return commit, dirty


def documents(path: Path) -> Iterable[dict[str, Any]]:
    text = path.read_text()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        yield from parsed
        return
    if isinstance(parsed, dict):
        yield parsed
        return
    for line in text.splitlines():
        if line.strip():
            yield json.loads(line)


def source_stats(document: dict[str, Any]) -> tuple[int, int]:
    captions = document.get("captions", [])
    span_count = sum(len(caption.get("spans", [])) for caption in captions)
    token_count = document.get("original_token_count")
    if token_count is None:
        token_count = sum(len(caption.get("tokens", [])) for caption in captions)
    return int(token_count), span_count


def evaluate(
    docs: list[dict[str, Any]],
    config: dict[str, Any],
    qualitative_limit: int = 12,
) -> dict[str, Any]:
    started = time.perf_counter()
    totals: collections.Counter[str] = collections.Counter()
    cluster_sizes: collections.Counter[int] = collections.Counter()
    support_buckets: collections.Counter[str] = collections.Counter()
    qualitative: list[dict[str, Any]] = []
    contradiction_cases: list[dict[str, Any]] = []
    for document in docs:
        plan = build_semantic_plan_from_exchange(document, config)
        token_count, span_count = source_stats(document)
        consensus = [unit for unit in plan["units"] if unit["kind"] == "consensus"]
        unique = [unit for unit in plan["units"] if unit["kind"] == "unique"]
        relations = [unit for unit in plan["units"] if unit["kind"] == "relation"]
        alternatives = [
            unit for unit in plan["units"]
            if unit["kind"] == "uncertainty" and unit.get("alternatives")
        ]
        totals.update(
            images=1,
            captions=len(document.get("captions", [])),
            source_tokens=token_count,
            semantic_spans=span_count,
            consensus_groups=len(consensus),
            unique_units=len(unique),
            relation_units=len(relations),
            uncertainty_units=len(alternatives),
            output_units=len(plan["units"]),
        )
        for unit in consensus:
            cluster_sizes[len(unit["members"])] += 1
            fraction = float(unit["support_fraction"])
            bucket = (
                "100%" if fraction == 1.0 else
                ">=50%" if fraction >= 0.5 else
                "<50%"
            )
            support_buckets[bucket] += 1
            if len(qualitative) < qualitative_limit:
                qualitative.append(
                    {
                        "image_id": document.get("image_id", ""),
                        "role": unit.get("role"),
                        "members": unit["members"],
                        "weights": unit["weights"],
                        "minimum_similarity": unit["minimum_pair_similarity"],
                        "support_fraction": fraction,
                    }
                )
        for unit in alternatives:
            if len(contradiction_cases) < qualitative_limit:
                contradiction_cases.append(
                    {
                        "image_id": document.get("image_id", ""),
                        "alternatives": unit["alternatives"],
                        "reason": unit.get("reason"),
                    }
                )

    elapsed = time.perf_counter() - started
    compression_ratio = (
        totals["output_units"] / totals["source_tokens"]
        if totals["source_tokens"] else 0.0
    )
    return {
        **totals,
        "compression_ratio": compression_ratio,
        "compression_factor": 1.0 / compression_ratio if compression_ratio else 0.0,
        "cluster_size_distribution": {
            str(key): value for key, value in sorted(cluster_sizes.items())
        },
        "support_distribution": dict(sorted(support_buckets.items())),
        "runtime_seconds": elapsed,
        "images_per_second": totals["images"] / elapsed if elapsed else 0.0,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "qualitative_clusters": qualitative,
        "contradiction_cases": contradiction_cases,
    }


def ablations(docs: list[dict[str, Any]], base: dict[str, Any]) -> list[dict[str, Any]]:
    runs: list[tuple[str, dict[str, Any]]] = []
    for threshold in (0.85, 0.88, 0.90, 0.92, 0.94, 0.96):
        runs.append((f"threshold_{threshold:.2f}", {**base, "default_cosine_threshold": threshold, "minimum_score": threshold}))
    for fusion in ("mean", "weighted-mean", "norm-preserving-mean"):
        runs.append((f"fusion_{fusion}", {**base, "fusion": fusion}))
    runs.extend(
        [
            ("constraints_cosine_only", {**base, "require_role_match": False, "require_entity_match": False, "grounding_weight": 0.0}),
            ("constraints_role", {**base, "require_role_match": True, "require_entity_match": False, "grounding_weight": 0.0}),
            ("constraints_role_entity", {**base, "require_role_match": True, "require_entity_match": True, "grounding_weight": 0.0}),
            ("constraints_role_entity_grounding", {**base, "require_role_match": True, "require_entity_match": True, "grounding_weight": 0.1}),
            ("support_2", {**base, "minimum_support_count": 2, "minimum_support_fraction": 0.0}),
            ("support_3", {**base, "minimum_support_count": 3, "minimum_support_fraction": 0.0}),
            ("support_50pct", {**base, "minimum_support_count": 1, "minimum_support_fraction": 0.5}),
        ]
    )
    results = []
    for name, config in runs:
        metrics = evaluate(docs, config, qualitative_limit=2)
        results.append({"name": name, "config": config, "metrics": metrics})
    # Learned set fusion is downstream and cannot be represented by a static
    # embedding operation; retain an explicit row rather than silently omit it.
    results.append(
        {
            "name": "fusion_learned_set_downstream",
            "status": "requires downstream trainable adapter",
        }
    )
    return results


def markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        "# CCSS offline benchmark",
        "",
        f"- Git commit: `{report['git_commit']}` ({'dirty' if report['git_dirty'] else 'clean'})",
        f"- Tokenizer/vocabulary: `{report['tokenizer_identifier']}` / `{report['vocabulary_identifier']}`",
        f"- Embedding model/layer: `{report['embedding_model']}` / `{report['embedding_layer']}`",
        f"- Span extraction: `{report['span_extraction_policy']}`",
        f"- Images/captions: {metrics['images']} / {metrics['captions']}",
        f"- Source tokens / spans / output units: {metrics['source_tokens']} / {metrics['semantic_spans']} / {metrics['output_units']}",
        f"- Consensus / unique / relation units: {metrics['consensus_groups']} / {metrics['unique_units']} / {metrics['relation_units']}",
        f"- Compression: **{metrics['compression_factor']:.3f}x** ({metrics['compression_ratio']:.5f} output units/source token)",
        f"- Runtime: {metrics['runtime_seconds']:.3f}s; peak RSS: {metrics['peak_rss_kib']} KiB",
        "",
        "## Cluster examples",
        "",
    ]
    for cluster in metrics["qualitative_clusters"]:
        lines.append(
            f"- `{cluster['image_id']}` / `{cluster['role']}`: "
            f"{', '.join(cluster['members'])} "
            f"(min cosine {cluster['minimum_similarity']:.4f}, "
            f"support {cluster['support_fraction']:.1%})"
        )
    lines.extend(
        [
            "",
            "## Go/no-go gates",
            "",
            f"- At least 2x compression: `{report['go_no_go']['compression_2x']}`",
            f"- Protected false-merge rate below 1%: `{report['go_no_go']['protected_false_merge_below_1pct']}`",
            f"- Unique-detail retention at least 95%: `{report['go_no_go']['unique_detail_retention_95pct']}`",
            f"- Beats mean pooling on reconstruction/detail: `{report['go_no_go']['beats_mean_pooling']}`",
            f"- No significant learned-resampler degradation: `{report['go_no_go']['resampler_non_degradation']}`",
            "",
            "Unknown gates require expected-pair/detail annotations and downstream decoder/resampler metrics; they are never inferred from compression alone.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tokenizer-identifier", required=True)
    parser.add_argument("--vocabulary-identifier", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--embedding-layer", required=True)
    parser.add_argument("--span-extraction-policy", required=True)
    parser.add_argument("--run-ablations", action="store_true")
    args = parser.parse_args()

    docs = list(documents(args.input))
    if args.limit:
        docs = docs[: args.limit]
    base = json.loads(args.config.read_text()) if args.config else {}
    root = Path(__file__).resolve().parents[2]
    commit, dirty = git_state(root)
    metrics = evaluate(docs, base)
    report: dict[str, Any] = {
        "schema": "ztok.ccss_benchmark.v1",
        "git_commit": commit,
        "git_dirty": dirty,
        "tokenizer_identifier": args.tokenizer_identifier,
        "vocabulary_identifier": args.vocabulary_identifier,
        "embedding_model": args.embedding_model,
        "embedding_layer": args.embedding_layer,
        "span_extraction_policy": args.span_extraction_policy,
        "threshold_config": base,
        "metrics": metrics,
        "go_no_go": {
            "compression_2x": metrics["compression_factor"] >= 2.0,
            "protected_false_merge_below_1pct": None,
            "unique_detail_retention_95pct": None,
            "beats_mean_pooling": None,
            "resampler_non_degradation": None,
            "stable_single_caption_inference": None,
        },
    }
    if args.run_ablations:
        report["ablations"] = ablations(docs, base)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2) + "\n")
    args.markdown_output.write_text(markdown(report))
    print(markdown(report))


if __name__ == "__main__":
    main()
