# NanoGPT superposition: Transformer vs RWKV

| Run | Architecture | Tokenizer | Superposition | Group | Fusion | Params | Source bytes | GPU-hours | Positions | Val bits/byte | Recovery cost |
|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|
| ztok-superposition-main-b3-transformer-312s-transformer-ordinary_bpe-ordinary1-norm_preserving_mean-b3seconds | transformer | ordinary_bpe | ordinary | 1 | norm_preserving_mean | 34,759,872 | 458,662,134 | 0.0865 | 107,355,136 | 1.42440 | baseline |
| ztok-superposition-main-b3-transformer-312s-transformer-ordinary_bpe-fixed2-norm_preserving_mean-b3seconds | transformer | ordinary_bpe | fixed | 2 | norm_preserving_mean | 34,759,872 | 558,583,709 | 0.0865 | 86,570,408 | 1.45470 | not_recovered |
| ztok-superposition-main-b3-rwkv-661s-rwkv-ordinary_bpe-ordinary1-norm_preserving_mean-b3seconds | rwkv | ordinary_bpe | ordinary | 1 | norm_preserving_mean | 35,124,096 | 460,614,836 | 0.1835 | 107,812,096 | 1.60086 | baseline |
| ztok-superposition-main-b3-rwkv-661s-rwkv-ordinary_bpe-fixed2-norm_preserving_mean-b3seconds | rwkv | ordinary_bpe | fixed | 2 | norm_preserving_mean | 35,124,096 | 571,852,553 | 0.1835 | 88,046,200 | 1.64760 | not_recovered |

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
