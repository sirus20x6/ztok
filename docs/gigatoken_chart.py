#!/usr/bin/env python3
"""Render the exact-output GPT-2 comparison (docs/gigatoken.png).

The chart intentionally keeps ztok's persistent-process cold/warm measurements
separate from GigaToken's fresh-process samples. Both implementations encode the
same 500,000,000 bytes with the same GPT-2 vocabulary and emit identical IDs.

Regenerate with: python3 docs/gigatoken_chart.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "benchmark_data.json").read_text())["gpt2_gigatoken"]

giga = DATA["gigatoken"]
contiguous = DATA["ztok"]["contiguous"]
ragged = DATA["ztok"]["ordered_no_gather"]

labels = [
    "GigaToken\none-pass median",
    "ztok contiguous\ncold",
    "ztok contiguous\nwarm",
    "ztok no-gather\ncold",
    "ztok no-gather\nwarm",
]
values = [
    giga["median_mb_s"],
    contiguous["cold_mb_s"],
    contiguous["warm_mb_s"],
    ragged["cold_mb_s"],
    ragged["warm_mb_s"],
]
plot_values = np.asarray(values) / 1000.0
colors = ["#cbd5e0", "#90cdf4", "#2b6cb0", "#9ae6b4", "#2f855a"]

# Only GigaToken has a cross-process range. ztok's bars are explicit cold/warm
# summaries from its five-iteration run rather than an error range.
lower = np.zeros(len(values))
upper = np.zeros(len(values))
lower[0] = (giga["median_mb_s"] - giga["minimum_mb_s"]) / 1000.0
upper[0] = (giga["maximum_mb_s"] - giga["median_mb_s"]) / 1000.0

fig, ax = plt.subplots(figsize=(9.2, 4.9))
x = np.arange(len(labels))
bars = ax.bar(
    x,
    plot_values,
    width=0.68,
    color=colors,
    edgecolor="#1a365d",
    linewidth=1.1,
    yerr=np.vstack([lower, upper]),
    capsize=6,
    error_kw=dict(ecolor="#4a5568", lw=1.4),
)

for bar, value in zip(bars, plot_values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.07,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color="#1a365d",
    )

ax.axhline(giga["median_mb_s"] / 1000.0, color="#718096", linestyle="--", linewidth=1.0, alpha=0.7)
ax.set_ylabel("throughput (GB/s, decimal)", fontsize=12)
ax.set_title(
    "ztok vs GigaToken — GPT-2, 500 MB, 32 workers, exact ID match",
    fontsize=13,
    fontweight="bold",
    color="#1a365d",
    pad=12,
)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylim(0, max(giga["maximum_mb_s"] / 1000.0, *plot_values) * 1.16)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=10)
ax.text(
    0.5,
    -0.25,
    "AMD EPYC 7473X, ReleaseFast, same 500,000,000 bytes and GPT-2 vocab; 108,735,122 identical IDs.\n"
    "GigaToken whisker = min/max of 10 fresh-process samples; ztok cold/warm = persistent 5-iteration run.",
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=8.0,
    color="#718096",
    family="monospace",
)

plt.tight_layout()
output = HERE / "gigatoken.png"
fig.savefig(output, dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {output}")
