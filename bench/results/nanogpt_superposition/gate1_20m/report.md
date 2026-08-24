# NanoGPT superposition: Transformer vs RWKV

| Run | Architecture | Tokenizer | Superposition | Group | Fusion | Params | Source bytes | GPU-hours | Positions | Val bits/byte | Recovery cost |
|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|
| ztok-superposition-gate1-20m-transformer-ordinary_bpe-ordinary1-norm_preserving_mean-b2bytes | transformer | ordinary_bpe | ordinary | 1 | norm_preserving_mean | 34,759,872 | 20,037,156 | 0.0039 | 4,691,456 | 2.32700 | baseline |
| ztok-superposition-gate1-20m-transformer-ordinary_bpe-fixed2-norm_preserving_mean-b2bytes | transformer | ordinary_bpe | fixed | 2 | norm_preserving_mean | 34,759,872 | 20,037,156 | 0.0034 | 3,250,520 | 2.41667 | not_recovered |
| ztok-superposition-gate1-20m-rwkv-ordinary_bpe-ordinary1-norm_preserving_mean-b2bytes | rwkv | ordinary_bpe | ordinary | 1 | norm_preserving_mean | 35,124,096 | 20,037,156 | 0.0089 | 4,691,456 | 2.21021 | baseline |
| ztok-superposition-gate1-20m-rwkv-ordinary_bpe-fixed2-norm_preserving_mean-b2bytes | rwkv | ordinary_bpe | fixed | 2 | norm_preserving_mean | 35,124,096 | 20,037,156 | 0.0072 | 3,250,520 | 2.28756 | not_recovered |

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
