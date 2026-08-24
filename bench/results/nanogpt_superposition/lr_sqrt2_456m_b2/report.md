# NanoGPT superposition: Transformer vs RWKV

| Run | Architecture | Tokenizer | Superposition | Group | Fusion | Params | Source bytes | GPU-hours | Positions | Val bits/byte | Recovery cost |
|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|
| ztok-superposition-main-456m-transformer-ordinary_bpe-ordinary1-norm_preserving_mean-b2bytes | transformer | ordinary_bpe | ordinary | 1 | norm_preserving_mean | 34,759,872 | 455,725,318 | 0.0865 | 106,669,696 | 1.39236 | baseline |
| ztok-superposition-main-456m-transformer-ordinary_bpe-fixed2-norm_preserving_mean-b2bytes | transformer | ordinary_bpe | fixed | 2 | norm_preserving_mean | 34,759,872 | 455,725,318 | 0.0718 | 73,253,704 | 1.44130 | not_recovered |
| ztok-superposition-lrablation-sqrt2-456m-transformer-ordinary_bpe-fixed2-norm_preserving_mean-b2bytes | transformer | ordinary_bpe | fixed | 2 | norm_preserving_mean | 34,759,872 | 455,725,318 | 0.0721 | 73,253,704 | 1.43077 | not_recovered |
| ztok-superposition-main-456m-rwkv-ordinary_bpe-ordinary1-norm_preserving_mean-b2bytes | rwkv | ordinary_bpe | ordinary | 1 | norm_preserving_mean | 35,124,096 | 455,725,318 | 0.1835 | 106,669,696 | 1.54885 | baseline |
| ztok-superposition-main-456m-rwkv-ordinary_bpe-fixed2-norm_preserving_mean-b2bytes | rwkv | ordinary_bpe | fixed | 2 | norm_preserving_mean | 35,124,096 | 455,725,318 | 0.1498 | 73,253,704 | 1.60745 | not_recovered |
| ztok-superposition-lrablation-sqrt2-456m-rwkv-ordinary_bpe-fixed2-norm_preserving_mean-b2bytes | rwkv | ordinary_bpe | fixed | 2 | norm_preserving_mean | 35,124,096 | 455,725,318 | 0.1514 | 73,253,704 | 1.52543 | recovered (164.8s) |

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
