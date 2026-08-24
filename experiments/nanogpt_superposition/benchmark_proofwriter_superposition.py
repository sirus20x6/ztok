#!/usr/bin/env python3
"""Measure role-safe and logical-isomorphism superposition on ProofWriter."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ztok
from datasets import load_dataset
from nltk.stem import WordNetLemmatizer

SCHEMA = "ztok.proofwriter_superposition.v1"
UNARY_FACT = re.compile(r"^(.+?) is (not )?([A-Za-z][A-Za-z -]*?)\.$")
BINARY_FACT = re.compile(
    r"^(.+?) (does not )?([A-Za-z]+) ((?:[Tt]he .+)|(?:[A-Z][A-Za-z]*(?: .+)?))\.$"
)
TOKEN = re.compile(r"<E:[^>]+>|[A-Za-z]+(?:'[A-Za-z]+)?|[,.;:!?-]")
VARIABLES = frozenset(
    {"someone", "somebody", "something", "they", "them", "their", "it", "its"}
)
FUNCTION_WORDS = frozenset(
    {
        "a",
        "all",
        "an",
        "and",
        "are",
        "do",
        "does",
        "if",
        "is",
        "people",
        "person",
        "some",
        "that",
        "the",
        "then",
        "thing",
        "things",
        "who",
    }
)
OPERATORS = {"if": "IF", "then": "THEN", "and": "AND", "not": "NOT"}


@dataclass(frozen=True)
class Unit:
    kind: str
    key: str
    surface: str

    @property
    def signature(self) -> str:
        return f"{self.kind}:{self.key}"


def parse_fact(sentence: str) -> tuple[str, str, str | None]:
    match = UNARY_FACT.fullmatch(sentence)
    if match:
        return match.group(1), match.group(3).casefold(), None
    match = BINARY_FACT.fullmatch(sentence)
    if match:
        return match.group(1), match.group(3).casefold(), match.group(4)
    raise ValueError(f"unsupported ProofWriter fact: {sentence!r}")


def entities_from_statements(statements: list[str]) -> list[str]:
    entities = []
    seen = set()
    for statement in statements:
        try:
            subject, _, object_ = parse_fact(statement)
        except ValueError:
            continue
        for value in (subject, object_):
            if value is None or value.casefold() in VARIABLES:
                continue
            folded = value.casefold()
            if folded not in seen:
                seen.add(folded)
                entities.append(value)
    return entities


def mark_entities(text: str, entities: list[str]) -> str:
    result = text
    for entity in sorted(entities, key=lambda value: (-len(value), value.casefold())):
        result = re.sub(
            rf"(?<![A-Za-z]){re.escape(entity)}(?![A-Za-z])",
            f"<E:{entity}>",
            result,
            flags=re.IGNORECASE,
        )
    return result


def statement_units(
    statement: str,
    entities: list[str],
    *,
    rename_entities: bool,
    rename_predicates: bool,
    entity_ids: dict[str, str] | None = None,
    predicate_ids: dict[str, str] | None = None,
) -> list[Unit]:
    entity_ids = entity_ids if entity_ids is not None else {}
    predicate_ids = predicate_ids if predicate_ids is not None else {}
    lemmatizer = WordNetLemmatizer()
    units = []
    marked = mark_entities(statement, entities)
    for raw in TOKEN.findall(marked):
        if raw.startswith("<E:"):
            surface = raw[3:-1]
            folded = surface.casefold()
            key = (
                entity_ids.setdefault(folded, f"E{len(entity_ids)}")
                if rename_entities
                else folded
            )
            units.append(Unit("entity", key, surface))
            continue
        folded = raw.casefold()
        if folded in VARIABLES:
            units.append(
                Unit(
                    "variable",
                    "V0" if rename_predicates else folded,
                    folded,
                )
            )
        elif folded in OPERATORS:
            units.append(Unit("operator", OPERATORS[folded], folded))
        elif folded in FUNCTION_WORDS or not folded.isalpha():
            units.append(Unit("syntax", folded, folded))
        else:
            lemma = lemmatizer.lemmatize(folded, "v")
            key = (
                predicate_ids.setdefault(lemma, f"P{len(predicate_ids)}")
                if rename_predicates
                else lemma
            )
            units.append(Unit("predicate", key, folded))
    return units


def canonicalize_trace(
    facts: list[str],
    rules: list[str],
    question: str,
    answer: str,
    *,
    rename_predicates: bool,
) -> list[Unit]:
    entities = entities_from_statements([*facts, question])
    entity_ids: dict[str, str] = {}
    predicate_ids: dict[str, str] = {}
    units: list[Unit] = []

    def append_statement(kind: str, statement: str) -> None:
        units.append(Unit("separator", kind, kind))
        units.extend(
            statement_units(
                statement,
                entities,
                rename_entities=True,
                rename_predicates=rename_predicates,
                entity_ids=entity_ids,
                predicate_ids=predicate_ids,
            )
        )

    for fact in facts:
        append_statement("FACT", fact)
    for rule in rules:
        append_statement("RULE", rule)
    append_statement("QUESTION", question)
    units.append(Unit("answer", answer.casefold(), answer.casefold()))
    return units


def proof_traces(row: dict[str, Any], row_index: int) -> list[dict[str, Any]]:
    if row["answer"] == "Unknown" or not row["used_facts"]:
        return []
    if len(row["used_facts"]) != len(row["used_rules"]):
        raise ValueError(f"proof alternatives differ at row {row_index}")
    return [
        {
            "row_index": row_index,
            "proof_index": proof_index,
            "facts": list(facts),
            "rules": list(rules),
            "question": str(row["question"]),
            "answer": str(row["answer"]),
            "depth": int(row["depth"]),
        }
        for proof_index, (facts, rules) in enumerate(
            zip(row["used_facts"], row["used_rules"], strict=True)
        )
    ]


def single_token_id(pipeline: ztok.Pipeline, surface: str) -> int | None:
    ids = pipeline.encode(" " + surface)
    return int(ids[0]) if len(ids) == 1 else None


def build_groups(
    traces: list[dict[str, Any]],
    pipeline: ztok.Pipeline,
    *,
    rename_predicates: bool,
    maximum_group_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_signature: dict[tuple[str, ...], list[tuple[dict[str, Any], list[Unit]]]] = (
        defaultdict(list)
    )
    total_units = 0
    for trace in traces:
        units = canonicalize_trace(
            trace["facts"],
            trace["rules"],
            trace["question"],
            trace["answer"],
            rename_predicates=rename_predicates,
        )
        total_units += len(units)
        by_signature[tuple(unit.signature for unit in units)].append((trace, units))

    groups = []
    grouped_trace_count = 0
    grouped_source_units = 0
    primer_output_units = 0
    actual_superposition_slots = 0
    actual_superposition_occurrences = 0
    for signature, members in sorted(by_signature.items(), key=lambda item: item[0]):
        for start in range(0, len(members), maximum_group_size):
            chunk = members[start : start + maximum_group_size]
            if len(chunk) < 2:
                continue
            slot_payload = []
            for slot_index in range(len(signature)):
                slot_units = [units[slot_index] for _, units in chunk]
                values: dict[tuple[str, int | None], int] = defaultdict(int)
                for unit in slot_units:
                    values[(unit.surface, single_token_id(pipeline, unit.surface))] += 1
                distinct = len(values) > 1
                superposable = (
                    distinct
                    and slot_units[0].kind
                    in {
                        "entity",
                        "predicate",
                        "variable",
                    }
                    and all(token_id is not None for _, token_id in values)
                )
                if superposable:
                    actual_superposition_slots += 1
                    actual_superposition_occurrences += len(slot_units)
                slot_payload.append(
                    {
                        "slot_index": slot_index,
                        "kind": slot_units[0].kind,
                        "logical_key": slot_units[0].key,
                        "superposable": superposable,
                        "values": [
                            {
                                "surface": surface,
                                "token_id": token_id,
                                "count": count,
                                "weight": count / len(slot_units),
                            }
                            for (surface, token_id), count in sorted(values.items())
                        ],
                    }
                )
            if not any(slot["superposable"] for slot in slot_payload):
                continue
            grouped_trace_count += len(chunk)
            grouped_source_units += len(signature) * len(chunk)
            primer_output_units += len(signature)
            digest = hashlib.sha256("\n".join(signature).encode()).hexdigest()[:20]
            groups.append(
                {
                    "group_id": f"{digest}-{start // maximum_group_size}",
                    "member_count": len(chunk),
                    "members": [
                        {
                            "row_index": trace["row_index"],
                            "proof_index": trace["proof_index"],
                            "depth": trace["depth"],
                        }
                        for trace, _ in chunk
                    ],
                    "answer": chunk[0][0]["answer"],
                    "logical_signature": list(signature),
                    "slots": slot_payload,
                }
            )
    return groups, {
        "proof_trace_count": len(traces),
        "logical_unit_count": total_units,
        "group_count": len(groups),
        "grouped_trace_count": grouped_trace_count,
        "trace_coverage": grouped_trace_count / len(traces) if traces else 0.0,
        "grouped_source_units": grouped_source_units,
        "primer_output_units": primer_output_units,
        "eligible_primer_compression_ratio": (
            primer_output_units / grouped_source_units if grouped_source_units else 1.0
        ),
        "superposition_slot_count": actual_superposition_slots,
        "superposition_source_occurrences": actual_superposition_occurrences,
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
    parser.add_argument("--maximum-group-size", type=int, default=8)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()
    if args.maximum_group_size < 2:
        raise SystemExit("--maximum-group-size must be at least two")

    dataset = load_dataset(
        args.dataset,
        split=args.split,
        revision=args.revision,
    )
    traces = []
    answer_counts: dict[str, int] = defaultdict(int)
    row_count = min(len(dataset), args.max_rows or len(dataset))
    for row_index in range(row_count):
        row = dataset[row_index]
        answer_counts[str(row["answer"])] += 1
        traces.extend(proof_traces(row, row_index))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    modes = {
        "role_safe": False,
        "logical_isomorphism_ceiling": True,
    }
    summary = {
        "schema": SCHEMA,
        "dataset": args.dataset,
        "revision": args.revision,
        "split": args.split,
        "row_count": row_count,
        "answer_counts": dict(sorted(answer_counts.items())),
        "proof_rows_exclude_unknown": True,
        "maximum_group_size": args.maximum_group_size,
        "tokenizer": str(args.tokenizer.resolve()),
        "modes": {},
    }
    with ztok.Pipeline.from_path(args.tokenizer) as pipeline:
        for mode, rename_predicates in modes.items():
            groups, metrics = build_groups(
                traces,
                pipeline,
                rename_predicates=rename_predicates,
                maximum_group_size=args.maximum_group_size,
            )
            summary["modes"][mode] = metrics
            path = args.output_dir / f"{mode}.jsonl"
            with path.open("w") as output:
                for group in groups:
                    output.write(json.dumps(group, sort_keys=True) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
