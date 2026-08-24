# RWKV xIELU GEMM-epilogue fusion

## Outcome

The fused `key projection -> xIELU` Triton epilogue is correct and remains an
explicit `xielu_fused` backend, but it is rejected as the default. At an
identical 455,725,318-byte budget it was 0.30% slower and its final validation
bits/byte was 0.23% higher than the qualified standalone CUDA xIELU condition.

| Activation path | GPU seconds | Validation BPB | Validation loss | Peak VRAM |
|---|---:|---:|---:|---:|
| ReLU² baseline | 714.727 | 1.341768 | 3.895408 | 13,461,296,128 |
| Standalone CUDA xIELU | **669.089** | **1.330651** | **3.863135** | **12,642,454,528** |
| Fused GEMM epilogue xIELU | 671.073 | 1.333671 | 3.871902 | 12,642,454,528 |

Relative to ReLU², the fused run was still 6.11% faster and 0.60% better in
validation BPB. The standalone CUDA xIELU result remains stronger: 6.39% faster
and 0.83% better than ReLU².

## What was fused

The training-shape forward path computes the BF16 ChannelMix key projection and
xIELU in one Triton kernel. The FP32 accumulator is rounded to BF16 before xIELU,
matching the numerical boundary of `torch.nn.functional.linear` followed by the
standalone kernel. Both the pre-activation and activated tensors are stored
because the backward needs them for the key and value matrix gradients.

The backward retains Nathan Ranchin's fused xIELU derivative/scalar-reduction
kernel, followed by the three required GEMMs. Unsupported devices, dtypes, or
contraction dimensions fall back to the standalone CUDA operation. Ordinary
tokenization and the ReLU²/xIELU backends are unchanged.

## Verification

- Same seed, data order, model, LR schedule, positions, source bytes, and
  estimated FLOPs.
- 35,124,116 parameters in both xIELU runs.
- CUDA tests cover the fused pre/post tensors and gradients for the input, both
  projection matrices, and both learned curvature scalars.
- Full experiment suite: 65 passed.
- At the actual training shape, the initial losses were identical; legal BF16
  GEMM reduction-order differences caused a small later trajectory divergence.

Across 6,939 paired non-evaluation steps after the first 50, fused and standalone
had virtually identical medians (95.524 ms versus 95.521 ms). Fused was faster
on 50.06% of steps, indistinguishable from chance. Its mean was 0.251 ms slower,
which agrees with the complete-run result.

## Decision

Keep `xielu_fused` for future kernel work and shape-specific profiling, but use
`xielu_cuda` for experiments. The extra launch removed by epilogue fusion is too
small to overcome the highly optimized cuBLAS key projection on this shape.
Larger performance work should target the RWKV scan, projection GEMMs, optimizer,
or a deeper fusion that removes an intermediate rather than only one launch.
