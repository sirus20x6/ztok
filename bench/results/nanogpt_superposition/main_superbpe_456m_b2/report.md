# NanoGPT superposition: Transformer vs RWKV

| Run | Architecture | Tokenizer | Superposition | Group | Fusion | Params | Source bytes | GPU-hours | Positions | Val bits/byte | Recovery cost |
|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|
| ztok-superposition-main-superbpe-456m-transformer-superbpe-ordinary1-norm_preserving_mean-b2bytes | transformer | superbpe | ordinary | 1 | norm_preserving_mean | 34,759,872 | 455,694,987 | 0.0874 | 107,627,520 | 1.37170 | baseline |
| ztok-superposition-main-superbpe-456m-transformer-superbpe-fixed2-norm_preserving_mean-b2bytes | transformer | superbpe | fixed | 2 | norm_preserving_mean | 34,759,872 | 455,694,987 | 0.0740 | 73,969,456 | 1.42860 | not_recovered |
| ztok-superposition-main-superbpe-456m-rwkv-superbpe-ordinary1-norm_preserving_mean-b2bytes | rwkv | superbpe | ordinary | 1 | norm_preserving_mean | 35,124,096 | 455,694,987 | 0.1889 | 107,627,520 | 1.53119 | baseline |
| ztok-superposition-main-superbpe-456m-rwkv-superbpe-fixed2-norm_preserving_mean-b2bytes | rwkv | superbpe | fixed | 2 | norm_preserving_mean | 35,124,096 | 455,694,987 | 0.1468 | 73,969,456 | 1.59139 | not_recovered |

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
