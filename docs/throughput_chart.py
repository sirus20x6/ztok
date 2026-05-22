#!/usr/bin/env python3
"""Render the ztok-vs-tiktoken throughput chart (docs/throughput.png).

Apples-to-apples: same cl100k_base vocab, the *same* 9.06 MB corpus file
(English + code + chat + multilingual mix), same machine, warm encode
loop (8 iters). Measured 2026-05-22 on the reference box (AMD EPYC
7473X, 24 physical / 48 SMT cores), ReleaseFast.

The two tokenizers produce the same id count on the corpus (~2.688 M
ids; tiktoken's batch path differs by a handful at chunk seams), so this
is a like-for-like tokenization, not a vocab/algorithm mismatch.

Reproduce:
  cat bench/corpora/english.txt bench/corpora/code.txt \
      bench/corpora/chat.txt bench/corpora/multilingual.txt > /tmp/mix.txt
  for i in 1 2 3; do cat /tmp/mix.txt >> /tmp/bench10.txt; done
  ztok bench --include cl100k --corpus-file /tmp/bench10.txt --iters 8
  python3 bench/bench_competitors.py --lib tiktoken \
      --corpus /tmp/bench10.txt --iters 8 --threads {0,8,48}

Multithreaded numbers are steady-state (median of 3 reps); ztok batch
×48 ranges ~274 (cold) → ~291 (warm), tiktoken ~75–79.

Regenerate with:  python3 docs/throughput_chart.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

configs = ["single-thread", "batch ×8", "batch ×48\n(+pin-physical)"]
ztok = [19.6, 104.7, 291.0]
tiktoken = [10.4, 48.8, 76.0]

x = np.arange(len(configs))
w = 0.38

fig, ax = plt.subplots(figsize=(7.6, 4.6))
b1 = ax.bar(x - w / 2, ztok, w, label="ztok", color="#2b6cb0", edgecolor="#1a365d", linewidth=1.1)
b2 = ax.bar(x + w / 2, tiktoken, w, label="tiktoken 0.12", color="#cbd5e0", edgecolor="#1a365d", linewidth=1.1)

for b, v in zip(b1, ztok):
    ax.text(b.get_x() + b.get_width() / 2, v + 4, f"{v:.0f}", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="#1a365d")
for b, v in zip(b2, tiktoken):
    ax.text(b.get_x() + b.get_width() / 2, v + 4, f"{v:.0f}", ha="center", va="bottom",
            fontsize=11, color="#4a5568")

# speedup callouts above each config group
for xi, (zt, tk) in enumerate(zip(ztok, tiktoken)):
    ax.text(xi, max(zt, tk) + 22, f"{zt / tk:.1f}× faster", ha="center", va="bottom",
            fontsize=10.5, fontweight="bold", color="#2f855a")

ax.set_ylabel("throughput (MB/s)", fontsize=12)
ax.set_title("ztok vs tiktoken — cl100k_base, identical 9 MB corpus",
             fontsize=13, fontweight="bold", color="#1a365d", pad=12)
ax.set_xticks(x)
ax.set_xticklabels(configs)
ax.set_ylim(0, max(ztok) * 1.30)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=11)
ax.legend(loc="upper left", fontsize=11, frameon=False)
ax.text(0.5, -0.24,
        "ReleaseFast, AMD EPYC 7473X (24c/48t), warm encode loop (8 iters), same corpus bytes & vocab.\n"
        "Both emit ~2.688 M ids. Reproduce: ztok bench --corpus-file FILE  +  bench/bench_competitors.py --lib tiktoken",
        transform=ax.transAxes, ha="center", va="top", fontsize=8.2,
        color="#718096", family="monospace")

plt.tight_layout()
fig.savefig("docs/throughput.png", dpi=150, bbox_inches="tight", facecolor="white")
print("wrote docs/throughput.png")
