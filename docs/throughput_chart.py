#!/usr/bin/env python3
"""Render the ztok-vs-tiktoken throughput chart (docs/throughput.png).

Apples-to-apples: same cl100k_base vocab, the *same* corpus bytes per
pair, same machine, warm encode loop (8 iters), ReleaseFast. Each value
is the median of three independent runs. Both tokenizers emit the same
id count on each corpus, so it's like-for-like.

Throughput is corpus-dependent, so each bar shows a RANGE across two
real corpora measured on the reference box (AMD EPYC 7473X, 24c/48t),
2026-07-10:
  * low  end = English + code + chat + multilingual mix (~9 MB) — the
    heavy non-ASCII content slows BPE for both tokenizers.
  * high end = English + code + chat (~10.6 MB) — ASCII-heavy, the
    common code/English LLM-training case.
The solid bar is the conservative mix median; the whisker reaches the
ASCII-heavy median. Raw runs and token counts live in
`docs/benchmark_data.json`.

Reproduce:
  # mix:   english+code+chat+multilingual, x3 ;  ascii: english+code+chat, x5
  ztok bench --include cl100k --corpus-file FILE --iters 8
  python3 bench/bench_competitors.py --lib tiktoken --corpus FILE --iters 8 --threads {0,8,48}

Regenerate with:  python3 docs/throughput_chart.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "benchmark_data.json").read_text())
scaling = DATA["cl100k_scaling"]

configs = ["single-thread", "batch ×8", "batch ×48\n(+pin-physical)"]
# low = multilingual mix (conservative), high = ASCII-heavy
ztok_lo = scaling["ztok"]["mix"]["median_mb_s"]
ztok_hi = scaling["ztok"]["ascii"]["median_mb_s"]
tik_lo = scaling["tiktoken"]["mix"]["median_mb_s"]
tik_hi = scaling["tiktoken"]["ascii"]["median_mb_s"]

x = np.arange(len(configs))
w = 0.38


def whisker(lo, hi):
    return np.array([[0, 0, 0], [hi[i] - lo[i] for i in range(3)]])


fig, ax = plt.subplots(figsize=(7.8, 4.8))
b1 = ax.bar(x - w / 2, ztok_lo, w, yerr=whisker(ztok_lo, ztok_hi), capsize=5,
            label="ztok", color="#2b6cb0", edgecolor="#1a365d", linewidth=1.1,
            error_kw=dict(ecolor="#1a365d", lw=1.4))
b2 = ax.bar(x + w / 2, tik_lo, w, yerr=whisker(tik_lo, tik_hi), capsize=5,
            label="tiktoken 0.12", color="#cbd5e0", edgecolor="#1a365d", linewidth=1.1,
            error_kw=dict(ecolor="#4a5568", lw=1.4))

for i, b in enumerate(b1):
    ax.text(b.get_x() + b.get_width() / 2, ztok_hi[i] + 8,
            f"{ztok_lo[i]:.0f}–{ztok_hi[i]:.0f}", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="#1a365d")
for i, b in enumerate(b2):
    label = (f"{tik_lo[i]:.0f}–{tik_hi[i]:.0f}"
             if round(tik_lo[i]) != round(tik_hi[i]) else f"{tik_lo[i]:.0f}")
    ax.text(b.get_x() + b.get_width() / 2, tik_hi[i] + 8,
            label, ha="center", va="bottom", fontsize=10, color="#4a5568")

# speedup range above each group
for i in range(3):
    smin = ztok_lo[i] / tik_hi[i]
    smax = ztok_hi[i] / tik_lo[i]
    ax.text(i, max(ztok_hi[i], tik_hi[i]) + 32, f"{smin:.1f}–{smax:.1f}× faster",
            ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#2f855a")

ax.set_ylabel("throughput (MB/s)", fontsize=12)
ax.set_title("ztok vs tiktoken — cl100k_base, identical corpus per pair",
             fontsize=13, fontweight="bold", color="#1a365d", pad=12)
ax.set_xticks(x)
ax.set_xticklabels(configs)
ax.set_ylim(0, max(ztok_hi) * 1.22)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=11)
ax.legend(loc="upper left", fontsize=11, frameon=False)
ax.text(0.5, -0.24,
        "ReleaseFast, AMD EPYC 7473X (24c/48t), 8 iters/run, median of 3 runs, same bytes & vocab per pair.\n"
        "Bar = English+code+chat+multilingual mix; whisker = ASCII-heavy median. Both emit equal id counts.",
        transform=ax.transAxes, ha="center", va="top", fontsize=8.0,
        color="#718096", family="monospace")

plt.tight_layout()
output = HERE / "throughput.png"
fig.savefig(output, dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {output}")
