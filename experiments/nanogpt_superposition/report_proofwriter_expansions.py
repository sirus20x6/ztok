#!/usr/bin/env python3
"""Render an auditable Markdown report for ProofWriter statement expansions."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_report(result_dir: Path, tokenizer: Path) -> str:
    summary = json.loads((result_dir / "summary.json").read_text())
    samples: dict[tuple[str, str], dict] = {}
    with (result_dir / "expansions.jsonl").open() as source:
        for line in source:
            expansion = json.loads(line)
            key = (expansion["statement_kind"], expansion["truth_class"])
            samples.setdefault(key, expansion)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    )
    lines = [
        "# ProofWriter statement-superposition expansion report",
        "",
        f"- Git commit: `{commit}`{' (dirty worktree)' if dirty else ''}",
        f"- Dataset: `{summary['dataset']}`",
        f"- Dataset revision: `{summary['revision']}`",
        f"- Split: `{summary['split']}`",
        f"- Tokenizer: `{tokenizer.resolve()}`",
        f"- Tokenizer SHA-256: `{file_sha256(tokenizer)}`",
        "",
        "## Coverage",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Training rows | {summary['row_count']:,} |",
        f"| Logical contexts | {summary['context_count']:,} |",
        f"| Expansion plans | {summary['expansion_count']:,} |",
        f"| Rows with at least one expansion | {summary['eligible_row_count']:,} |",
        f"| Eligible row fraction | {summary['eligible_row_fraction']:.2%} |",
        (
            "| Expansion-inventory logical-unit ratio | "
            f"{summary['expansion_inventory_compression_ratio']:.2%} |"
        ),
        "",
        (
            "The inventory ratio compares every explicit branch statement represented by "
            "an expansion with its one-slot fused form. It is not a claim that the "
            "ordinary training dataset itself is already shorter."
        ),
        "",
        "## Expansion distribution",
        "",
        "| Category | Plans |",
        "|---|---:|",
    ]
    for kind, count in summary["expanded_statement_kinds"].items():
        lines.append(f"| {kind} statements | {count:,} |")
    for kind, count in summary["expanded_slot_kinds"].items():
        lines.append(f"| {kind} slots | {count:,} |")
    lines.extend(
        [
            "",
            "## Safety constraints",
            "",
            "- Alternatives come from the same ProofWriter world.",
            "- Statement kind and truth class must agree.",
            "- Exactly one logical slot may differ.",
            "- Every alternative must be one token under the recorded ztok tokenizer.",
            "- Alternatives are proposition branches, not lexical synonyms.",
            "",
            "## Qualitative examples",
            "",
        ]
    )
    for (kind, truth), expansion in sorted(samples.items()):
        alternatives = " | ".join(
            alternative["surface"] for alternative in expansion["alternatives"]
        )
        statements = "; ".join(
            alternative["statements"][0] for alternative in expansion["alternatives"]
        )
        lines.extend(
            [
                f"### {kind}: {truth}",
                "",
                f"Superposed slot: `{alternatives}`",
                "",
                f"Branches: {statements}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.result_dir / "report.md"
    output.write_text(render_report(args.result_dir, args.tokenizer) + "\n")
    print(output)


if __name__ == "__main__":
    main()
