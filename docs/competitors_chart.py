#!/usr/bin/env python3
"""Render the ztok-vs-the-field chart (docs/competitors.png).

Single-thread encode throughput, ztok vs each reference library on *that
library's own vocab*, over the same 9.06 MB corpus, same machine. Each
pair is id-matched (ztok and the competitor emit the same token count on
the corpus, within chunk-seam noise), so every bar pair is a true
like-for-like comparison — same vocab, same algorithm, same bytes.

Measured 2026-05-22 on the reference box (AMD EPYC 7473X, 24c/48t),
ReleaseFast, warm loop (8 iters).

ztok side:   tiktoken/cl100k + HF/gpt2 via `ztok bench --corpus-file`;
             SentencePiece via `bench_cross` (real SP pipeline, id-exact).
competitor:  bench/bench_competitors.py --lib {tiktoken,hf,sentencepiece}

Regenerate with:  python3 docs/competitors_chart.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (label, ztok MB/s, competitor MB/s) — single-thread, id-matched.
rows = [
    ("tiktoken\n(cl100k)", 19.6, 10.4),
    ("HF tokenizers\n(gpt2)", 4.8, 1.5),
    ("SentencePiece\nBPE (llama2)", 3.0, 1.5),
    ("SentencePiece\nUnigram (t5)", 16.7, 10.3),
]
labels = [r[0] for r in rows]
ztok = [r[1] for r in rows]
comp = [r[2] for r in rows]

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
        "ReleaseFast, AMD EPYC 7473X, same 9 MB corpus & vocab per pair, warm loop (8 iters). "
        "Each pair is id-matched.\n"
        "Multithreaded gap is larger: vs tiktoken 3.8× and vs SentencePiece 3.3–5.6× at batch ×48 (see throughput.png).",
        transform=ax.transAxes, ha="center", va="top", fontsize=8.0,
        color="#718096", family="monospace")

plt.tight_layout()
fig.savefig("docs/competitors.png", dpi=150, bbox_inches="tight", facecolor="white")
print("wrote docs/competitors.png")
