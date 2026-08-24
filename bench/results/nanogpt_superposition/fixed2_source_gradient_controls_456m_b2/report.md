# NanoGPT superposition: Transformer vs RWKV

| Run | Architecture | Tokenizer | Superposition | Group | Fusion | Fused-source grad P1/P2 | Params | Source bytes | GPU-hours | Positions | Val bits/byte | Recovery |
|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| ztok-superposition-lrablation-6x-456m-rwkv-ordinary_bpe-ordinary1-norm_preserving_mean-b2bytes | rwkv | ordinary_bpe | ordinary | 1 | norm_preserving_mean | -- | 35,124,096 | 455,725,318 | 0.1985 | 106,669,696 | 1.34177 | baseline |
| ztok-superposition-lrablation-6p5x-456m-rwkv-ordinary_bpe-ordinary1-norm_preserving_mean-b2bytes | rwkv | ordinary_bpe | ordinary | 1 | norm_preserving_mean | -- | 35,124,096 | 455,725,318 | 0.1819 | 106,669,696 | 1.34970 | baseline |
| ztok-superposition-fixed2-6p5-grad-sqrt2-456m-rwkv-ordinary_bpe-fixed2-norm_preserving_mean-b2bytes | rwkv | ordinary_bpe | fixed | 2 | norm_preserving_mean | 1.414/1 | 35,124,096 | 455,725,318 | 0.1538 | 73,253,704 | 1.40878 | phase_complete_target_not_reached |
| ztok-superposition-fixed2-6p5-grad-sqrt2-phase12-456m-rwkv-ordinary_bpe-fixed2-norm_preserving_mean-b2bytes | rwkv | ordinary_bpe | fixed | 2 | norm_preserving_mean | 1.414/1.414 | 35,124,096 | 455,725,318 | 0.1553 | 73,253,704 | 1.40881 | phase_complete_target_not_reached |

## Branch-equivalence safety

- Evaluated cases: 0
- Safe-fusion precision: None
- Safe-fusion recall: None
- Unsafe fusions: 0

Perplexity comparisons are valid only within one tokenizer. Cross-tokenizer conclusions use bits per byte.
Equal-step runs are plumbing checks; compute claims require matched source-byte and GPU-second budgets.

## Plots

- `bits_per_byte_vs_gpu_seconds.png`
- `bits_per_byte_vs_source_bytes.png`
- `recovery_loss.png`
- `throughput_vs_group_size.png`
