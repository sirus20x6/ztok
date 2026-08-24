#!/usr/bin/env python3
"""Materialize a deterministic text view of a document corpus for tokenizer training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from prepare_data import _iter_documents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--parquet-text-field", default="text")
    parser.add_argument("--parquet-token-field", default="metadata.token_count")
    parser.add_argument("--max-corpus-tokens", type=int, default=100_000_000)
    parser.add_argument("--max-source-bytes", type=int)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    order_hash = hashlib.sha256()
    materialized_hash = hashlib.sha256()
    document_count = 0
    corpus_tokens = 0
    source_bytes = 0
    with args.output.open("wb") as destination:
        documents = _iter_documents(
            args.input,
            line_documents=False,
            jsonl_text_field=None,
            parquet_text_field=args.parquet_text_field,
            parquet_token_field=args.parquet_token_field,
        )
        for source in documents:
            document = source.text
            next_tokens = corpus_tokens + (source.corpus_tokens or 0)
            next_bytes = source_bytes + len(document)
            if document_count and next_tokens > args.max_corpus_tokens:
                break
            if (
                args.max_source_bytes
                and document_count
                and next_bytes > args.max_source_bytes
            ):
                break
            digest = hashlib.sha256(document).digest()
            order_hash.update(digest)
            destination.write(document)
            destination.write(b"\n")
            materialized_hash.update(document)
            materialized_hash.update(b"\n")
            document_count += 1
            corpus_tokens = next_tokens
            source_bytes = next_bytes

    manifest = {
        "schema": "ztok.nanogpt_superposition.corpus.v1",
        "inputs": [str(path.resolve()) for path in args.input],
        "document_count": document_count,
        "corpus_token_count": corpus_tokens,
        "source_bytes": source_bytes,
        "materialized_bytes": args.output.stat().st_size,
        "document_order_hash": order_hash.hexdigest(),
        "materialized_sha256": materialized_hash.hexdigest(),
        "selection": {
            "parquet_text_field": args.parquet_text_field,
            "parquet_token_field": args.parquet_token_field,
            "max_corpus_tokens": args.max_corpus_tokens,
            "max_source_bytes": args.max_source_bytes,
        },
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
