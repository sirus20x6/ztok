#!/usr/bin/env python3
"""Render the ztok-vs-the-field chart (docs/competitors.png).

Single-thread encode throughput, ztok vs each reference library on *that
library's own vocab*, over the same 9.06 MB corpus, same machine. Token
counts are exact for cl100k and Llama-2 BPE and differ by less than 0.1%
for GPT-2 and T5 Unigram. Every value is the median of three independent
warm-loop runs.

Measured 2026-07-10 on the reference box (AMD EPYC 7473X, 24c/48t),
ReleaseFast, warm loop (8 iters).

ztok side:   tiktoken/cl100k + HF/gpt2 via `ztok bench --corpus-file`;
             SentencePiece via `bench_cross` (real SP pipeline, id-exact).
competitor:  bench/bench_competitors.py --lib {tiktoken,hf,sentencepiece}

Regenerate with:  python3 docs/competitors_chart.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "benchmark_data.json").read_text())
rows = DATA["single_thread_competitors"]
labels = [r["label"] for r in rows]
ztok = [r["ztok_median_mb_s"] for r in rows]
comp = [r["reference_median_mb_s"] for r in rows]

x = np.arange(len(rows))
w = 0.38

fig, ax = plt.subplots(figsize=(8.4, 4.6))
b1 = ax.bar(x - w / 2, ztok, w, label="ztok", color="#2b6cb0", edgecolor="#1a365d", linewidth=1.1)
b2 = ax.bar(x + w / 2, comp, w, label="reference library", color="#cbd5e0", edgecolor="#1a365d", linewidth=1.1)

for b, v in zip(b1, ztok):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}", ha="center", va="bottom",
            fontsize=10.5, fontweight="bold", color="#1a365d")
for b, v in zip(b2, comp):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}", ha="center", va="bottom",
            fontsize=10.5, color="#4a5568")
for xi, (zt, ck) in enumerate(zip(ztok, comp)):
    ax.text(xi, max(zt, ck) + 1.6, f"{zt / ck:.1f}×", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="#2f855a")

ax.set_ylabel("single-thread throughput (MB/s)", fontsize=12)
ax.set_title("ztok vs the field — single-thread, each on its own vocab",
             fontsize=13, fontweight="bold", color="#1a365d", pad=12)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(0, max(ztok) * 1.28)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=10)
ax.legend(loc="upper right", fontsize=11, frameon=False)
ax.text(0.5, -0.26,
        "ReleaseFast, AMD EPYC 7473X, same 9.06 MB corpus & vocab per pair, 8 iters/run, median of 3 runs.\n"
        "Counts exact for cl100k/Llama-2 BPE; GPT-2/T5 differ <0.1%. cl100k scaling is shown in throughput.png.",
        transform=ax.transAxes, ha="center", va="top", fontsize=8.0,
        color="#718096", family="monospace")

plt.tight_layout()
output = HERE / "competitors.png"
fig.savefig(output, dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {output}")
