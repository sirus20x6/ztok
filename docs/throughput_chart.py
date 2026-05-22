#!/usr/bin/env python3
"""Render the ztok throughput chart (docs/throughput.png).

Numbers measured 2026-05-21 on the reference box (AMD EPYC 7473X, 24
physical / 48 SMT cores), ReleaseFast, 10 MB cl100k-shaped corpus.
Reproduce: `ztok bench --corpus-bytes 10485760 --include cl100k`.
The batch x48 figure is the multithreaded peak and varies ~370-440 MB/s
run to run with scheduling; ~400 is representative.

Regenerate with:  python3 docs/throughput_chart.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

shapes = ["single-thread", "batch ×8", "batch ×48\n(+pin-physical)"]
mbps = [26, 133, 400]
colors = ["#90cdf4", "#4299e1", "#2b6cb0"]

fig, ax = plt.subplots(figsize=(7.2, 4.2))
bars = ax.bar(shapes, mbps, color=colors, width=0.62, edgecolor="#1a365d", linewidth=1.2)

for b, v in zip(bars, mbps):
    ax.text(b.get_x() + b.get_width() / 2, v + 6, f"{v} MB/s",
            ha="center", va="bottom", fontsize=12, fontweight="bold", color="#1a365d")

ax.set_ylabel("throughput (MB/s)", fontsize=12)
ax.set_title("ztok cl100k throughput — scales ~15× across 48 cores",
             fontsize=13, fontweight="bold", color="#1a365d", pad=12)
ax.set_ylim(0, max(mbps) * 1.18)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=11)
ax.margins(x=0.06)
ax.text(0.5, -0.22,
        "ReleaseFast, 10 MB corpus, AMD EPYC 7473X (24c/48t). "
        "batch ×48 varies ~370–440 MB/s run to run.\n"
        "Reproduce: ztok bench --corpus-bytes 10485760 --include cl100k",
        transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
        color="#718096", family="monospace")

plt.tight_layout()
fig.savefig("docs/throughput.png", dpi=150, bbox_inches="tight", facecolor="white")
print("wrote docs/throughput.png")
