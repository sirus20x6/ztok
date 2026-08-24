# NanoGPT superposition: Transformer vs RWKV

| Run | Architecture | Tokenizer | Superposition | Group | Fusion | Params | Source bytes | GPU-hours | Positions | Val bits/byte | Recovery cost |
|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|
| ztok-superposition-main-456m-rwkv-ordinary_bpe-ordinary1-norm_preserving_mean-b2bytes | rwkv | ordinary_bpe | ordinary | 1 | norm_preserving_mean | 35,124,096 | 455,725,318 | 0.1835 | 106,669,696 | 1.54885 | baseline |
| ztok-superposition-lrablation-sqrt2-456m-rwkv-ordinary_bpe-ordinary1-norm_preserving_mean-b2bytes | rwkv | ordinary_bpe | ordinary | 1 | norm_preserving_mean | 35,124,096 | 455,725,318 | 0.1972 | 106,669,696 | 1.46458 | baseline |
| ztok-superposition-lrablation-2x-456m-rwkv-ordinary_bpe-ordinary1-norm_preserving_mean-b2bytes | rwkv | ordinary_bpe | ordinary | 1 | norm_preserving_mean | 35,124,096 | 455,725,318 | 0.1943 | 106,669,696 | 1.40883 | baseline |
| ztok-superposition-lrablation-3x-456m-rwkv-ordinary_bpe-ordinary1-norm_preserving_mean-b2bytes | rwkv | ordinary_bpe | ordinary | 1 | norm_preserving_mean | 35,124,096 | 455,725,318 | 0.1993 | 106,669,696 | 1.36026 | baseline |
| ztok-superposition-lrablation-6x-456m-rwkv-ordinary_bpe-ordinary1-norm_preserving_mean-b2bytes | rwkv | ordinary_bpe | ordinary | 1 | norm_preserving_mean | 35,124,096 | 455,725,318 | 0.1985 | 106,669,696 | 1.34177 | baseline |
| ztok-superposition-lrablation-6p5x-456m-rwkv-ordinary_bpe-ordinary1-norm_preserving_mean-b2bytes | rwkv | ordinary_bpe | ordinary | 1 | norm_preserving_mean | 35,124,096 | 455,725,318 | 0.1819 | 106,669,696 | 1.34970 | baseline |
| ztok-superposition-lrablation-7x-456m-rwkv-ordinary_bpe-ordinary1-norm_preserving_mean-b2bytes | rwkv | ordinary_bpe | ordinary | 1 | norm_preserving_mean | 35,124,096 | 455,725,318 | 0.1812 | 106,669,696 | 1.34306 | baseline |
| ztok-superposition-lrablation-sqrt2-456m-rwkv-ordinary_bpe-fixed2-norm_preserving_mean-b2bytes | rwkv | ordinary_bpe | fixed | 2 | norm_preserving_mean | 35,124,096 | 455,725,318 | 0.1514 | 73,253,704 | 1.52543 | not_recovered |

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
