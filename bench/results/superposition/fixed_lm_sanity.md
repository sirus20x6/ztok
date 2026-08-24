# Fixed token-superposition language-model sanity check

- Commit: `72c52c2da8e2456dbf6e5b2e0cdee68b8cb0c320` (dirty)
- Dataset: 10000 sequences × 64 source tokens
- Device: `cuda`

| Condition | Reduction | Phase tok/s | Pre-recovery val loss | Final ordinary val loss | Δ vs baseline | Wall time |
|---|---:|---:|---:|---:|---:|---:|
| ordinary_baseline | 0.0% | 506652 | 5.2254 | 4.8598 | +0.0000 | 1.08s |
| fixed_2 | 50.0% | 666041 | 5.3169 | 4.9068 | +0.0470 | 0.89s |
| fixed_4 | 75.0% | 613920 | 5.4412 | 5.0147 | +0.1549 | 0.97s |

The tokenizer IDs used by the plan were asserted identical to ordinary byte-ID encoding. The training loss and embedding operations remain outside tokenizer core.
