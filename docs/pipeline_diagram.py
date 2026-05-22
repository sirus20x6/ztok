#!/usr/bin/env python3
"""Render the ztok pipeline diagram (docs/pipeline.png).

Regenerate with:  python3 docs/pipeline_diagram.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

STAGES = [
    ("Normalizer", ["identity", "NFC / NFD", "NFKC / NFKD", "byte_level", "capcode"]),
    ("PreTokenizer", ["identity", "cl100k", "HF byte_level"]),
    ("Model", ["byte_id", "BPE", "Unigram", "WordPiece", "TokenMonster"]),
    ("Decoder", ["concat", "WordPiece", "byte_level", "capcode"]),
]

BOX_W, BOX_H = 2.6, 1.0
GAP = 1.2
Y = 2.4
EDGE = "#2b6cb0"
FILL = "#ebf4ff"
ARROW = "#4a5568"

fig, ax = plt.subplots(figsize=(13, 4.2))
ax.set_xlim(0, len(STAGES) * BOX_W + (len(STAGES) + 1) * GAP)
ax.set_ylim(0, 4.2)
ax.axis("off")

def box_x(i):
    return GAP + i * (BOX_W + GAP)

# input label + arrow
ax.text(GAP * 0.45, Y + BOX_H / 2, "input", ha="center", va="center",
        fontsize=12, style="italic", color=ARROW)

for i, (name, variants) in enumerate(STAGES):
    x = box_x(i)
    ax.add_patch(FancyBboxPatch(
        (x, Y), BOX_W, BOX_H,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=2, edgecolor=EDGE, facecolor=FILL))
    ax.text(x + BOX_W / 2, Y + BOX_H / 2, name, ha="center", va="center",
            fontsize=14, fontweight="bold", color="#1a365d")
    # variants under the box
    ax.text(x + BOX_W / 2, Y - 0.25,
            "\n".join(variants), ha="center", va="top",
            fontsize=10.5, color="#4a5568", family="monospace")

# arrows: input -> box0 -> ... -> boxN -> ids
def arrow(x0, x1):
    ax.add_patch(FancyArrowPatch(
        (x0, Y + BOX_H / 2), (x1, Y + BOX_H / 2),
        arrowstyle="-|>", mutation_scale=20, linewidth=2, color=ARROW))

arrow(GAP * 0.7, box_x(0))
for i in range(len(STAGES) - 1):
    arrow(box_x(i) + BOX_W, box_x(i + 1))
last = box_x(len(STAGES) - 1) + BOX_W
arrow(last, last + GAP * 0.75)
ax.text(last + GAP * 0.9, Y + BOX_H / 2, "ids", ha="left", va="center",
        fontsize=12, style="italic", color=ARROW)

plt.tight_layout()
fig.savefig("docs/pipeline.png", dpi=150, bbox_inches="tight",
            facecolor="white")
print("wrote docs/pipeline.png")
