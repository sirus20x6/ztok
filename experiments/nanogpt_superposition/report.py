#!/usr/bin/env python3
"""Generate auditable JSON, Markdown, and required plots from run artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _metrics(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _load_run(path: Path) -> dict[str, Any]:
    summary = json.loads((path / "summary.json").read_text())
    rows = _metrics(path / "metrics.jsonl")
    config = summary["config"]
    training = config["training"]
    result = summary["results"]
    return {
        "path": str(path),
        "name": config["run_name"],
        "architecture": config["architecture"],
        "tokenizer": config.get("tokenizer_condition", "ordinary_bpe"),
        "representation": training["representation"],
        "group_size": training["group_size"],
        "fusion": training["fusion"],
        "target": training["target"],
        "budget_kind": training["budget_kind"],
        "lr_schedule_basis": training.get("lr_schedule_basis", "steps"),
        "parameters": summary["parameters"],
        "source_bytes": result["source_bytes"],
        "elapsed_seconds": result["elapsed_seconds"],
        "model_positions": result["model_positions"],
        "estimated_flops": result["estimated_flops"],
        "estimated_flops_per_ordinary_source_token": result.get(
            "estimated_flops_per_ordinary_source_token",
            result["estimated_flops"] / max(result["ordinary_token_equivalents"], 1),
        ),
        "validation_bits_per_byte": result["validation_bits_per_byte"],
        "ordinary_validation_loss": result["ordinary_validation_loss"],
        "peak_vram_bytes": max(
            (row.get("peak_vram_bytes", 0) for row in rows), default=0
        ),
        "device": summary["device"],
        "git_commit": summary["git_commit"],
        "git_dirty": summary["git_dirty"],
        "working_tree_sha256": summary.get("working_tree_sha256"),
        "parameter_dtype": summary.get("parameter_dtype"),
        "autocast_dtype": summary.get("autocast_dtype"),
        "dataset": summary["dataset"],
        "optimizer": {
            key: summary["config"]["training"].get(key)
            for key in (
                "learning_rate",
                "min_learning_rate",
                "warmup_steps",
                "lr_schedule_basis",
                "coarse_lr_multiplier",
                "ordinary_lr_multiplier",
                "coarse_superposed_source_gradient_multiplier",
                "mixed_superposed_source_gradient_multiplier",
                "weight_decay",
                "grad_clip",
            )
        },
        "rows": rows,
    }


def _recovery_cost(
    run: dict[str, Any], baseline_target: float | None
) -> dict[str, Any]:
    if run["representation"] == "ordinary":
        return {"status": "baseline", "seconds": 0.0, "source_bytes": 0}
    ordinary = [
        row
        for row in run["rows"]
        if row.get("phase") == "ordinary" and row.get("mode") == "ordinary"
    ]
    if not ordinary:
        return {"status": "no_ordinary_recovery"}
    start = ordinary[0]
    if baseline_target is None:
        return {"status": "missing_architecture_baseline"}
    for row in ordinary:
        value = row.get("validation_bits_per_byte")
        if value is not None and value <= baseline_target:
            return {
                "status": "recovered",
                "seconds": row["elapsed_seconds"] - start["elapsed_seconds"],
                "source_bytes": row["source_bytes"] - start["source_bytes"],
                "target_bits_per_byte": baseline_target,
            }
    return {
        "status": "phase_complete_target_not_reached",
        "seconds": ordinary[-1]["elapsed_seconds"] - start["elapsed_seconds"],
        "source_bytes": ordinary[-1]["source_bytes"] - start["source_bytes"],
        "target_bits_per_byte": baseline_target,
    }


def _branch_summary(paths: list[Path]) -> dict[str, Any]:
    cases = []
    for path in paths:
        payload = json.loads(path.read_text())
        cases.extend(
            {
                "source": str(path),
                "architecture": payload.get("architecture"),
                "rwkv_backend": payload.get("rwkv_backend"),
                **case,
            }
            for case in payload["cases"]
        )
    evaluated = [case for case in cases if case.get("status") == "ok"]
    true_positive = sum(
        case["expected_safe"] and case["cluster_decision"]["accepted"]
        for case in evaluated
    )
    false_positive = sum(
        not case["expected_safe"] and case["cluster_decision"]["accepted"]
        for case in evaluated
    )
    false_negative = sum(
        case["expected_safe"] and not case["cluster_decision"]["accepted"]
        for case in evaluated
    )
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else None
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else None
    )
    return {
        "cases": cases,
        "evaluated": len(evaluated),
        "safe_fusion_precision": precision,
        "safe_fusion_recall": recall,
        "unsafe_fusion_count": false_positive,
    }


def _plot_label(run: dict[str, Any]) -> str:
    lr = run["optimizer"].get("coarse_lr_multiplier") or 1.0
    if run["representation"] == "ordinary":
        return f"ordinary (LR {lr:g}x)"
    coarse_grad = (
        run["optimizer"].get("coarse_superposed_source_gradient_multiplier") or 1.0
    )
    mixed_grad = (
        run["optimizer"].get("mixed_superposed_source_gradient_multiplier") or 1.0
    )
    return (
        f"fixed-{run['group_size']} (LR {lr:g}x, "
        f"source grad {coarse_grad:.4g}/{mixed_grad:.4g})"
    )


def _plots(
    runs: list[dict[str, Any]], branch: dict[str, Any], output: Path
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    written = []

    def line_plot(filename: str, x_key: str, title: str, xlabel: str) -> None:
        figure, axis = plt.subplots(figsize=(7, 4.5))
        found = False
        for run in runs:
            rows = [row for row in run["rows"] if "validation_bits_per_byte" in row]
            if not rows:
                continue
            found = True
            axis.plot(
                [row[x_key] for row in rows],
                [row["validation_bits_per_byte"] for row in rows],
                marker="o",
                label=_plot_label(run),
            )
        if found:
            axis.set(title=title, xlabel=xlabel, ylabel="Validation bits/byte")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=7)
            figure.tight_layout()
            figure.savefig(output / filename, dpi=160)
            written.append(filename)
        plt.close(figure)

    line_plot(
        "bits_per_byte_vs_gpu_seconds.png",
        "elapsed_seconds",
        "Quality vs measured device time",
        "GPU/CPU-seconds",
    )
    line_plot(
        "bits_per_byte_vs_source_bytes.png",
        "source_bytes",
        "Quality vs source bytes",
        "Source bytes",
    )

    figure, axis = plt.subplots(figsize=(7, 4.5))
    for run in runs:
        if run["representation"] != "fixed":
            continue
        ordinary = [row for row in run["rows"] if row["mode"] == "ordinary"]
        if ordinary:
            start = ordinary[0]["source_bytes"]
            axis.plot(
                [row["source_bytes"] - start for row in ordinary],
                [row["loss"] for row in ordinary],
                label=_plot_label(run),
            )
    if axis.lines:
        axis.set(
            title="Ordinary-token recovery loss",
            xlabel="Recovery source bytes",
            ylabel="Ordinary CE loss",
        )
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(output / "recovery_loss.png", dpi=160)
        written.append("recovery_loss.png")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    labels = [_plot_label(run) for run in runs]
    throughput = [
        run["source_bytes"] / max(run["elapsed_seconds"], 1e-9) for run in runs
    ]
    axis.bar(range(len(runs)), throughput)
    axis.set_xticks(range(len(runs)), labels, rotation=35, ha="right", fontsize=7)
    axis.set(title="Measured throughput by condition", ylabel="Source bytes/second")
    figure.tight_layout()
    figure.savefig(output / "throughput_vs_group_size.png", dpi=160)
    written.append("throughput_vs_group_size.png")
    plt.close(figure)

    evaluated = [case for case in branch["cases"] if case.get("status") == "ok"]
    if evaluated:
        figure, axis = plt.subplots(figsize=(7, 4.5))
        for case in evaluated:
            immediate = case["metrics"]["horizons"][0]
            axis.scatter(
                case["metrics"]["candidate_embedding_diameter"],
                immediate["state_error"],
                label=case["class"],
                alpha=0.7,
            )
        axis.set(
            title="Branch error vs candidate diameter",
            xlabel="Maximum cosine distance",
            ylabel="Relative state error",
        )
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(output / "branch_error_by_diameter.png", dpi=160)
        written.append("branch_error_by_diameter.png")
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(7, 4.5))
        for case in evaluated:
            horizons = case["metrics"]["horizons"]
            axis.plot(
                [value["horizon"] for value in horizons],
                [value["state_error"] for value in horizons],
                alpha=0.45,
            )
        axis.set(
            title="Future branch-state divergence",
            xlabel="Future-token horizon",
            ylabel="Relative state error",
        )
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(output / "future_divergence.png", dpi=160)
        written.append("future_divergence.png")
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(5, 4))
        precision = branch["safe_fusion_precision"] or 0.0
        recall = branch["safe_fusion_recall"] or 0.0
        axis.bar(["precision", "recall"], [precision, recall])
        axis.set_ylim(0, 1)
        axis.set(title="Safe-fusion classification")
        figure.tight_layout()
        figure.savefig(output / "safe_fusion_precision_recall.png", dpi=160)
        written.append("safe_fusion_precision_recall.png")
        plt.close(figure)
    return written


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# NanoGPT superposition: Transformer vs RWKV",
        "",
        "| Run | Architecture | Tokenizer | Superposition | Group | Fusion | Fused-source grad P1/P2 | Params | Source bytes | GPU-hours | Positions | Val bits/byte | Recovery |",
        "|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for run in report["runs"]:
        recovery = run["recovery"]
        recovery_text = recovery["status"]
        if recovery["status"] == "recovered":
            recovery_text += f" ({recovery['seconds']:.1f}s)"
        gpu_hours = (
            f"{run['elapsed_seconds'] / 3600:.4f}"
            if str(run["device"]).startswith("cuda")
            else "N/A (CPU)"
        )
        if run["representation"] == "fixed":
            optimizer = run["optimizer"]
            coarse_grad = (
                optimizer.get("coarse_superposed_source_gradient_multiplier") or 1.0
            )
            mixed_grad = (
                optimizer.get("mixed_superposed_source_gradient_multiplier") or 1.0
            )
            source_gradient = f"{coarse_grad:.4g}/{mixed_grad:.4g}"
        else:
            source_gradient = "--"
        lines.append(
            f"| {run['name']} | {run['architecture']} | {run['tokenizer']} | {run['representation']} | {run['group_size']} | {run['fusion']} | {source_gradient} | {run['parameters']['total_with_auxiliary']:,} | {run['source_bytes']:,} | {gpu_hours} | {run['model_positions']:,} | {run['validation_bits_per_byte']:.5f} | {recovery_text} |"
        )
    lines.extend(
        [
            "",
            "## Branch-equivalence safety",
            "",
            f"- Evaluated cases: {report['branch_equivalence']['evaluated']}",
            f"- Safe-fusion precision: {report['branch_equivalence']['safe_fusion_precision']}",
            f"- Safe-fusion recall: {report['branch_equivalence']['safe_fusion_recall']}",
            f"- Unsafe fusions: {report['branch_equivalence']['unsafe_fusion_count']}",
            "",
            "Perplexity comparisons are valid only within one tokenizer. Cross-tokenizer conclusions use bits per byte.",
            "Equal-step runs are plumbing checks; compute claims require matched source-byte and GPU-second budgets.",
            "",
            "## Plots",
            "",
        ]
    )
    lines.extend(f"- `{name}`" for name in report["plots"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--branch-report", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    runs = [_load_run(path) for path in args.run_dir]
    baselines: dict[tuple[str, str, str, str], float] = {}
    for run in runs:
        if run["representation"] != "ordinary":
            continue
        key = (
            run["architecture"],
            run["tokenizer"],
            run["budget_kind"],
            run["lr_schedule_basis"],
        )
        baselines[key] = min(
            baselines.get(key, float("inf")), run["validation_bits_per_byte"]
        )
    for run in runs:
        baseline_key = (
            run["architecture"],
            run["tokenizer"],
            run["budget_kind"],
            run["lr_schedule_basis"],
        )
        run["recovery"] = _recovery_cost(run, baselines.get(baseline_key))
    branch = _branch_summary(args.branch_report)
    report = {
        "schema": "ztok.nanogpt_superposition.report.v1",
        "runs": runs,
        "branch_equivalence": branch,
    }
    # Avoid duplicating raw metric curves in the compact JSON summary.
    serializable = {
        **report,
        "runs": [
            {key: value for key, value in run.items() if key != "rows"} for run in runs
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots = _plots(runs, branch, args.output_dir)
    serializable["plots"] = plots
    (args.output_dir / "report.json").write_text(
        json.dumps(serializable, indent=2) + "\n"
    )
    (args.output_dir / "report.md").write_text(_markdown(serializable))
    print(_markdown(serializable))


if __name__ == "__main__":
    main()
