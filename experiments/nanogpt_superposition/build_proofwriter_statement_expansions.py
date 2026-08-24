#!/usr/bin/env python3
"""Expand individual ProofWriter statements into valid one-slot superpositions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ztok
from benchmark_proofwriter_superposition import (
    Unit,
    entities_from_statements,
    single_token_id,
    statement_units,
)
from datasets import load_dataset

SCHEMA = "ztok.proofwriter_statement_expansions.v1"
EXPANDABLE_KINDS = frozenset({"entity", "predicate", "variable"})


@dataclass(frozen=True)
class StatementRecord:
    statement_id: str
    kind: str
    truth_class: str
    text: str
    row_indices: tuple[int, ...]
    units: tuple[Unit, ...]


def context_id(facts: list[str], rules: list[str]) -> str:
    canonical = json.dumps(
        {"facts": facts, "rules": rules}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:20]


def statement_records(
    facts: list[str],
    rules: list[str],
    questions: dict[tuple[str, str], list[int]],
    all_row_indices: tuple[int, ...],
) -> list[StatementRecord]:
    entities = entities_from_statements(facts)
    records = []
    sources = [
        *(("fact", "asserted", fact, all_row_indices) for fact in dict.fromkeys(facts)),
        *(("rule", "rule", rule, all_row_indices) for rule in dict.fromkeys(rules)),
        *(
            ("question", answer, question, tuple(rows))
            for (question, answer), rows in sorted(questions.items())
        ),
    ]
    for index, (kind, truth, text, rows) in enumerate(sources):
        units = statement_units(
            text,
            entities,
            rename_entities=False,
            rename_predicates=False,
        )
        records.append(
            StatementRecord(
                statement_id=f"{kind}-{index}",
                kind=kind,
                truth_class=truth,
                text=text,
                row_indices=tuple(rows),
                units=tuple(units),
            )
        )
    return records


def build_context_expansions(
    identifier: str,
    records: list[StatementRecord],
    pipeline: Any,
    *,
    maximum_alternatives: int,
) -> list[dict[str, Any]]:
    candidates: dict[tuple[Any, ...], list[StatementRecord]] = defaultdict(list)
    for record in records:
        signatures = tuple(unit.signature for unit in record.units)
        for slot_index, unit in enumerate(record.units):
            if unit.kind not in EXPANDABLE_KINDS:
                continue
            masked = (
                signatures[:slot_index]
                + (f"{unit.kind}:<EXPAND>",)
                + signatures[slot_index + 1 :]
            )
            candidates[(record.kind, record.truth_class, slot_index, masked)].append(
                record
            )

    expansions = []
    for (kind, truth, slot_index, masked), members in sorted(candidates.items()):
        by_value: dict[tuple[str, int], list[StatementRecord]] = defaultdict(list)
        for member in members:
            unit = member.units[slot_index]
            token_id = single_token_id(pipeline, unit.surface)
            if token_id is not None:
                by_value[(unit.surface, token_id)].append(member)
        if len(by_value) < 2:
            continue
        ordered_values = sorted(
            by_value.items(),
            key=lambda item: (-len(item[1]), item[0][0].casefold(), item[0][0]),
        )[:maximum_alternatives]
        if len(ordered_values) < 2:
            continue
        selected_members = [member for _, values in ordered_values for member in values]
        source_units = sum(len(member.units) for member in selected_members)
        output_units = len(selected_members[0].units)
        digest = hashlib.sha256(
            (identifier + "\n" + "\n".join(masked)).encode()
        ).hexdigest()[:20]
        expansions.append(
            {
                "expansion_id": digest,
                "context_id": identifier,
                "statement_kind": kind,
                "truth_class": truth,
                "expanded_slot_index": slot_index,
                "expanded_slot_kind": selected_members[0].units[slot_index].kind,
                "logical_signature": list(masked),
                "alternatives": [
                    {
                        "surface": surface,
                        "token_id": token_id,
                        "statement_ids": [member.statement_id for member in values],
                        "statements": [member.text for member in values],
                        "row_indices": sorted(
                            {row for member in values for row in member.row_indices}
                        ),
                        "weight": 1.0 / len(ordered_values),
                    }
                    for (surface, token_id), values in ordered_values
                ],
                "fusion": "norm_preserving_mean",
                "recommended_original_weight": 0.7,
                "branch_policy": "one_logical_slot_only",
                "source_logical_units": source_units,
                "output_logical_units": output_units,
                "compression_ratio": output_units / source_units,
            }
        )
    return expansions


def build_inventory(
    dataset: Any,
    pipeline: Any,
    *,
    maximum_alternatives: int,
    max_rows: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    contexts: dict[str, dict[str, Any]] = {}
    row_count = min(len(dataset), max_rows or len(dataset))
    answer_counts: Counter[str] = Counter()
    for row_index, row in enumerate(dataset):
        if row_index >= row_count:
            break
        facts = list(row["facts"])
        rules = list(row["rules"])
        identifier = context_id(facts, rules)
        context = contexts.setdefault(
            identifier,
            {
                "facts": facts,
                "rules": rules,
                "row_indices": [],
                "questions": defaultdict(list),
            },
        )
        context["row_indices"].append(row_index)
        context["questions"][(str(row["question"]), str(row["answer"]))].append(
            row_index
        )
        answer_counts[str(row["answer"])] += 1

    expansions = []
    eligible_rows = set()
    statement_counts: Counter[str] = Counter()
    expanded_statement_kinds: Counter[str] = Counter()
    expanded_slot_kinds: Counter[str] = Counter()
    alternative_counts: Counter[int] = Counter()
    source_units = 0
    output_units = 0
    for identifier, context in sorted(contexts.items()):
        records = statement_records(
            context["facts"],
            context["rules"],
            context["questions"],
            tuple(context["row_indices"]),
        )
        statement_counts.update(record.kind for record in records)
        context_expansions = build_context_expansions(
            identifier,
            records,
            pipeline,
            maximum_alternatives=maximum_alternatives,
        )
        if context_expansions:
            eligible_rows.update(context["row_indices"])
        for expansion in context_expansions:
            expanded_statement_kinds[expansion["statement_kind"]] += 1
            expanded_slot_kinds[expansion["expanded_slot_kind"]] += 1
            alternative_counts[len(expansion["alternatives"])] += 1
            source_units += expansion["source_logical_units"]
            output_units += expansion["output_logical_units"]
        expansions.extend(context_expansions)

    return expansions, {
        "row_count": row_count,
        "context_count": len(contexts),
        "answer_counts": dict(sorted(answer_counts.items())),
        "statement_counts": dict(sorted(statement_counts.items())),
        "expansion_count": len(expansions),
        "eligible_row_count": len(eligible_rows),
        "eligible_row_fraction": len(eligible_rows) / row_count if row_count else 0.0,
        "expanded_statement_kinds": dict(sorted(expanded_statement_kinds.items())),
        "expanded_slot_kinds": dict(sorted(expanded_slot_kinds.items())),
        "alternative_count_distribution": {
            str(size): count for size, count in sorted(alternative_counts.items())
        },
        "expansion_inventory_source_logical_units": source_units,
        "expansion_inventory_output_logical_units": output_units,
        "expansion_inventory_compression_ratio": (
            output_units / source_units if source_units else 1.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="wentingzhao/proofwriter")
    parser.add_argument(
        "--revision", default="c468a2e9a467cc3ed8a90c2caa318e620fe59f41"
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-alternatives", type=int, default=8)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()
    if args.maximum_alternatives < 2:
        raise SystemExit("--maximum-alternatives must be at least two")
    dataset = load_dataset(
        args.dataset,
        split=args.split,
        revision=args.revision,
    )
    with ztok.Pipeline.from_path(args.tokenizer) as pipeline:
        expansions, metrics = build_inventory(
            dataset,
            pipeline,
            maximum_alternatives=args.maximum_alternatives,
            max_rows=args.max_rows,
        )
    summary = {
        "schema": SCHEMA,
        "dataset": args.dataset,
        "revision": args.revision,
        "split": args.split,
        "tokenizer": str(args.tokenizer.resolve()),
        "maximum_alternatives": args.maximum_alternatives,
        "safety": {
            "same_context_only": True,
            "same_statement_kind": True,
            "same_truth_class": True,
            "one_logical_slot_only": True,
            "single_ztok_token_alternatives_only": True,
        },
        **metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "expansions.jsonl").open("w") as output:
        for expansion in expansions:
            output.write(json.dumps(expansion, sort_keys=True) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
