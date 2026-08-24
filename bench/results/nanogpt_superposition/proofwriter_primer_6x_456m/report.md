# ProofWriter statement-superposition primer before RWKV 6× baseline

## Result

This condition is negative. A one-epoch statement-superposition primer followed
by the exact RWKV 6× FineWeb trajectory finished at **1.37175 validation
bits/byte**, versus **1.34177** for the unprimed baseline. Lower is better, so
the primer degraded final quality by **2.23%**.

| Metric | Unprimed 6× baseline | Primer then 6× baseline | Delta |
|---|---:|---:|---:|
| FineWeb source bytes | 455,725,318 | 455,725,318 | identical |
| Ordinary optimizer steps | 7,003 | 7,003 | identical |
| Ordinary model positions | 106,669,696 | 106,669,696 | identical |
| Validation loss | 3.89541 | 3.98245 | +2.23% |
| Validation bits/byte | 1.34177 | 1.37175 | +2.23% |
| Measured GPU-seconds, total | 714.73 | 715.33 | +0.08% |
| Estimated FLOPs, total | 2.48186e16 | 2.49443e16 | +0.51% |

The nearly equal measured total time is incidental run-to-run throughput
variation: the primer cost 31.38 seconds, while the later ordinary phase happened
to run 30.77 seconds faster than the historical baseline. Estimated FLOPs expose
the actual 0.51% compute addition.

## Primer

- Inventory: 121,146 ProofWriter one-slot expansions.
- Used: 111,685 (92.19%). The other 9,461 were rejected because their branch
  tokenizations did not have one exact aligned token position.
- Dose: one epoch, 449 optimizer steps, batch size 256.
- Representation: norm-preserving weighted mixture at the varying causal-input
  token; weighted set target at its prediction position; ordinary causal loss on
  the shared continuation.
- Cost: 5,400,712 explicit source bytes, 652,000 token equivalents, 540,315 model
  positions, 1.25714e14 estimated FLOPs, and 31.38 measured GPU-seconds.
- Model and AdamW state carried into FineWeb. FineWeb sampling RNG, ordinary step,
  LR schedule, and byte accounting restarted at zero.

## Controlled parity

Both arms use seed `260506546`, 35,124,096 parameters, BF16, RWKV-7, the same
32K BPE tokenizer (`cdf67b315ad2299aef10bd6a8b4894a0d982570bda2c4b06d000024dda2d7995`),
and git commit `72c52c2da8e2456dbf6e5b2e0cdee68b8cb0c320`. The first five ordinary
batches were checked directly: cumulative bytes, per-batch bytes, and learning
rates matched the historical baseline exactly.

The primer initially moved the model sharply away from held-out FineWeb. Its
FineWeb validation loss was 15.70 at handoff, then recovered, but remained behind
the baseline throughout the ordinary trajectory. This falsifies the tested
combination of a full primer epoch and the baseline 6× primer learning rate. It
does not falsify smaller primer doses, a lower primer-only LR, interleaving, or a
transition-equivalence objective without statement-domain causal loss.
