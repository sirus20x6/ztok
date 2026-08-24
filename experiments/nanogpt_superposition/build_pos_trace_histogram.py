#!/usr/bin/env python3
"""Build train-only POS histograms and local grammatical relation traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import spacy
from prepare_data import _iter_documents, _split

SCHEMA = "ztok.pos_logical_traces.v1"
TARGET_POS = frozenset({"NOUN", "VERB", "ADV"})
MORPH_FEATURES = frozenset({"Degree", "Mood", "Number", "Person", "Tense", "VerbForm"})
PROTECTED_LEMMAS = frozenset(
    {
        "before",
        "after",
        "above",
        "below",
        "behind",
        "front",
        "left",
        "right",
        "no",
        "not",
        "never",
        "true",
        "false",
    }
)


def normalized_morph(token: Any) -> tuple[str, ...]:
    return tuple(
        sorted(
            feature
            for feature in token.morph
            if feature.split("=", 1)[0] in MORPH_FEATURES
        )
    )


def logical_trace(token: Any) -> dict[str, Any]:
    children = Counter(
        f"{child.dep_}:{child.pos_}"
        for child in token.children
        if child.pos_ not in {"PUNCT", "SPACE"}
    )
    return {
        "pos": token.pos_,
        "dependency": token.dep_,
        "head_pos": "ROOT" if token.head == token else token.head.pos_,
        "head_dependency": "ROOT" if token.head == token else token.head.dep_,
        "morphology": list(normalized_morph(token)),
        "children": [[name, count] for name, count in sorted(children.items())],
    }


def trace_id(trace: dict[str, Any]) -> str:
    canonical = json.dumps(trace, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:20]


def eligible_token(token: Any) -> bool:
    lemma = token.lemma_.casefold().strip()
    return (
        token.pos_ in TARGET_POS
        and token.is_alpha
        and bool(lemma)
        and lemma not in PROTECTED_LEMMAS
        and not token.ent_type_
    )


def selected_training_documents(
    parquet: Path,
    *,
    seed: int,
    validation_fraction: float,
    max_corpus_tokens: int,
    max_documents: int | None,
) -> Iterator[tuple[str, dict[str, Any]]]:
    selected_tokens = 0
    selected_documents = 0
    train_index = 0
    sources = _iter_documents(
        [parquet],
        line_documents=False,
        jsonl_text_field=None,
        parquet_text_field="text",
        parquet_token_field="metadata.token_count",
    )
    for source_index, source in enumerate(sources):
        corpus_tokens = int(source.corpus_tokens or 0)
        if selected_tokens + corpus_tokens > max_corpus_tokens:
            break
        selected_tokens += corpus_tokens
        selected_documents += 1
        if _split(source.text, seed, validation_fraction) != "train":
            continue
        text = source.text.decode("utf-8")
        yield (
            text,
            {
                "source_index": source_index,
                "train_index": train_index,
                "source_bytes": len(source.text),
                "corpus_tokens": corpus_tokens,
            },
        )
        train_index += 1
        if max_documents is not None and train_index >= max_documents:
            break


def build_histogram(
    documents: Iterator[tuple[str, dict[str, Any]]],
    *,
    model_name: str,
    batch_size: int,
    n_process: int,
    contexts_per_lemma_trace: int,
    minimum_trace_count: int,
    minimum_lemma_trace_count: int,
) -> dict[str, Any]:
    nlp = spacy.load(model_name, exclude=["ner"])
    nlp.max_length = max(nlp.max_length, 5_000_000)
    pos_counts: Counter[str] = Counter()
    lemma_counts: dict[str, Counter[str]] = defaultdict(Counter)
    surface_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    traces: dict[str, dict[str, Any]] = {}
    trace_counts: Counter[str] = Counter()
    trace_lemmas: dict[str, Counter[str]] = defaultdict(Counter)
    trace_surfaces: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    contexts: dict[tuple[str, str], list[str]] = defaultdict(list)
    document_count = 0
    source_bytes = 0
    corpus_tokens = 0
    token_count = 0

    for doc, context in nlp.pipe(
        documents,
        as_tuples=True,
        batch_size=batch_size,
        n_process=n_process,
    ):
        document_count += 1
        source_bytes += int(context["source_bytes"])
        corpus_tokens += int(context["corpus_tokens"])
        token_count += len(doc)
        for token in doc:
            if not eligible_token(token):
                continue
            pos = token.pos_
            lemma = token.lemma_.casefold()
            surface = token.text
            trace = logical_trace(token)
            identifier = trace_id(trace)
            traces.setdefault(identifier, trace)
            pos_counts[pos] += 1
            lemma_counts[pos][lemma] += 1
            surface_counts[(pos, lemma)][surface] += 1
            trace_counts[identifier] += 1
            trace_lemmas[identifier][lemma] += 1
            trace_surfaces[(identifier, lemma)][surface] += 1
            key = (identifier, lemma)
            if (
                trace_lemmas[identifier][lemma] >= minimum_lemma_trace_count
                and len(contexts[key]) < contexts_per_lemma_trace
            ):
                sentence = " ".join(token.sent.text.split())[:320]
                if sentence and sentence not in contexts[key]:
                    contexts[key].append(sentence)

    retained_trace_ids = {
        identifier
        for identifier, count in trace_counts.items()
        if count >= minimum_trace_count
        and sum(
            lemma_count >= minimum_lemma_trace_count
            for lemma_count in trace_lemmas[identifier].values()
        )
        >= 2
    }
    trace_payload = []
    for identifier in sorted(traces):
        if identifier not in retained_trace_ids:
            continue
        lemmas = []
        for lemma, count in sorted(
            trace_lemmas[identifier].items(), key=lambda item: (-item[1], item[0])
        ):
            if count < minimum_lemma_trace_count:
                continue
            lemmas.append(
                {
                    "lemma": lemma,
                    "count": count,
                    "surfaces": [
                        {"text": surface, "count": surface_count}
                        for surface, surface_count in sorted(
                            trace_surfaces[(identifier, lemma)].items(),
                            key=lambda item: (-item[1], item[0]),
                        )
                    ],
                    "contexts": contexts[(identifier, lemma)],
                }
            )
        trace_payload.append(
            {
                "trace_id": identifier,
                "count": trace_counts[identifier],
                "signature": traces[identifier],
                "lemmas": lemmas,
            }
        )

    histograms = {}
    for pos in sorted(TARGET_POS):
        histograms[pos] = {
            "occurrences": pos_counts[pos],
            "unique_lemmas": len(lemma_counts[pos]),
            "lemmas": [
                {
                    "lemma": lemma,
                    "count": count,
                    "surfaces": [
                        {"text": surface, "count": surface_count}
                        for surface, surface_count in sorted(
                            surface_counts[(pos, lemma)].items(),
                            key=lambda item: (-item[1], item[0]),
                        )
                    ],
                }
                for lemma, count in sorted(
                    lemma_counts[pos].items(), key=lambda item: (-item[1], item[0])
                )
            ],
        }
    return {
        "schema": SCHEMA,
        "target_pos": sorted(TARGET_POS),
        "statistics": {
            "train_documents": document_count,
            "source_bytes": source_bytes,
            "source_corpus_tokens": corpus_tokens,
            "spacy_tokens": token_count,
            "eligible_occurrences": sum(pos_counts.values()),
            "logical_trace_count": len(traces),
            "lemma_trace_pair_count": sum(
                len(lemmas) for lemmas in trace_lemmas.values()
            ),
            "retained_logical_trace_count": len(retained_trace_ids),
            "retained_lemma_trace_pair_count": sum(
                sum(
                    count >= minimum_lemma_trace_count
                    for count in trace_lemmas[identifier].values()
                )
                for identifier in retained_trace_ids
            ),
        },
        "histograms": histograms,
        "logical_traces": trace_payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="en_core_web_sm")
    parser.add_argument("--seed", type=int, default=260506546)
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--max-corpus-tokens", type=int, default=100_000_000)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-process", type=int, default=1)
    parser.add_argument("--contexts-per-lemma-trace", type=int, default=2)
    parser.add_argument("--minimum-trace-count", type=int, default=8)
    parser.add_argument("--minimum-lemma-trace-count", type=int, default=2)
    args = parser.parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        raise SystemExit("--validation-fraction must be in (0, 1)")
    if args.batch_size <= 0 or args.n_process <= 0:
        raise SystemExit("--batch-size and --n-process must be positive")
    if args.minimum_trace_count <= 0 or args.minimum_lemma_trace_count <= 0:
        raise SystemExit("trace count thresholds must be positive")
    payload = build_histogram(
        selected_training_documents(
            args.parquet,
            seed=args.seed,
            validation_fraction=args.validation_fraction,
            max_corpus_tokens=args.max_corpus_tokens,
            max_documents=args.max_documents,
        ),
        model_name=args.model,
        batch_size=args.batch_size,
        n_process=args.n_process,
        contexts_per_lemma_trace=args.contexts_per_lemma_trace,
        minimum_trace_count=args.minimum_trace_count,
        minimum_lemma_trace_count=args.minimum_lemma_trace_count,
    )
    payload["source"] = {
        "parquet": str(args.parquet.resolve()),
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "max_corpus_tokens": args.max_corpus_tokens,
        "max_documents": args.max_documents,
        "spacy_model": args.model,
        "minimum_trace_count": args.minimum_trace_count,
        "minimum_lemma_trace_count": args.minimum_lemma_trace_count,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["statistics"], indent=2))


if __name__ == "__main__":
    main()
