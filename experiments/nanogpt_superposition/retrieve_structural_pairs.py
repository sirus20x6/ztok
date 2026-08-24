#!/usr/bin/env python3
"""Retrieve structurally compatible dataset examples before semantic mapping."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def structural_features(example: dict[str, Any]) -> Counter[str]:
    spans = {span["span_id"]: span for span in example["spans"]}
    eligible = {
        span_id: span
        for span_id, span in spans.items()
        if not span.get("protected", False) and span.get("semantic_role") != "syntax"
    }
    features: Counter[str] = Counter()
    for span in eligible.values():
        features[
            "node:"
            + ":".join(
                (
                    str(span.get("grammatical_role") or ""),
                    str(span.get("semantic_role") or ""),
                    str(span.get("entity_type") or ""),
                    str(span.get("binding_type") or ""),
                )
            )
        ] += 1
    for edge in example["relations"]:
        source = eligible.get(edge["source_span_id"])
        target = eligible.get(edge["target_span_id"])
        if source is None or target is None:
            continue
        features[
            "edge:"
            + ":".join(
                (
                    str(source.get("semantic_role") or ""),
                    str(edge["relation"]),
                    str(target.get("semantic_role") or ""),
                )
            )
        ] += 1
    return features


def weighted_jaccard(left: Counter[str], right: Counter[str]) -> float:
    keys = left.keys() | right.keys()
    union = sum(max(left[key], right[key]) for key in keys)
    if not union:
        return 0.0
    return sum(min(left[key], right[key]) for key in keys) / union


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=32)
    parser.add_argument("--minimum-score", type=float, default=0.55)
    parser.add_argument("--minimum-content-spans", type=int, default=3)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    examples = payload["examples"]
    features = [structural_features(example) for example in examples]
    candidates = []
    for left in range(len(examples)):
        if sum(features[left].values()) < args.minimum_content_spans:
            continue
        for right in range(left + 1, len(examples)):
            if sum(features[right].values()) < args.minimum_content_spans:
                continue
            score = weighted_jaccard(features[left], features[right])
            if score >= args.minimum_score:
                candidates.append((score, left, right))
    selected = []
    used: set[int] = set()
    for score, left, right in sorted(
        candidates,
        key=lambda item: (
            -item[0],
            examples[item[1]]["example_id"],
            examples[item[2]]["example_id"],
        ),
    ):
        if left in used or right in used:
            continue
        used.update((left, right))
        selected.append(
            {
                "left_example_id": examples[left]["example_id"],
                "right_example_id": examples[right]["example_id"],
                "structural_score": score,
                "left_feature_count": sum(features[left].values()),
                "right_feature_count": sum(features[right].values()),
            }
        )
        if len(selected) >= args.max_pairs:
            break
    output = {
        **payload,
        "dataset_id": f"{payload.get('dataset_id', args.input.stem)}-retrieved-{len(selected)}",
        "examples": [
            example for index, example in enumerate(examples) if index in used
        ],
        "retrieval_adapter": {
            "kind": "weighted_jaccard_of_content_node_and_dependency_edge_signatures",
            "minimum_score": args.minimum_score,
            "minimum_content_spans": args.minimum_content_spans,
            "candidate_pair_count": len(candidates),
            "selected_pair_count": len(selected),
            "disjoint_pairs": True,
        },
        "retrieval_pairs": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                "source_examples": len(examples),
                "candidate_pairs": len(candidates),
                "selected_pairs": len(selected),
                "selected_examples": len(used),
            }
        )
    )


if __name__ == "__main__":
    main()
