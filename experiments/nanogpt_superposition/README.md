# Dataset semantic superposition experiments

The primary experiment in this directory is the dataset-level mapping,
segmentation, and fusion design documented in
[`docs/DATASET_SEMANTIC_SUPERPOSITION.md`](../../docs/DATASET_SEMANTIC_SUPERPOSITION.md).
It aligns spans from different examples only when their meanings and their
grammatical/semantic relations agree, then builds a shorter fused superbatch
while retaining unmatched information.

This directory is a falsification-oriented language-model experiment built on
ztok's optional superposition plans. It does not modify tokenizer semantics and
does not place PyTorch or model code in the tokenizer core.

The fixed contiguous group-2/group-4 implementation is retained only as a
separate control. It is not the dataset semantic-superposition algorithm and its
results do not establish whether semantic mapping and fusion works.

The primary question is: can structurally equivalent information from separate
dataset examples be represented once, without losing their unique spans or
relation bindings, and improve validation bits/byte per measured GPU-second?

## Existing moe-mla infrastructure

The serious RWKV arm uses `/thearray/git/moe-mla` rather than an invented block:

- `models/rwkv_lab_adapter.py` wraps `rwkv_lab.rwkv_pretrain.RWKV7Small`, including
  its RWKV-7 blocks and exact recurrent-state path.
- `models/rwkv.py` is a small CPU reference oracle for tests and plumbing only.
- Every trainer emits the existing trainboard `train.jsonl` event contract and
  `step_NNNNNN/config.json` plus `ckpt.pt` sidecars.
- Put serious run outputs below `/thearray/git/moe-mla/runs`; trainboard discovers
  and ingests them without a database migration. All superposition-specific
  fields remain queryable through its `extra_json` storage.

Run the dashboard with:

```bash
go -C /thearray/git/moe-mla/dashboard run ./cmd/trainboard
# http://127.0.0.1:9124
```

The proposed 32K/448-wide/10-layer pair has closely matched size before auxiliary
fusion heads:

| Architecture | Total parameters | Embedding | Non-embedding |
|---|---:|---:|---:|
| Transformer | 34,759,872 | 14,680,064 | 20,079,808 |
| RWKV-7 | 35,124,096 | 14,680,064 | 20,444,032 |

Learned fusion and ordered heads are reported separately and included in totals.

## Layout

- `models/`: Transformer, RWKV-7 adapter, and CPU recurrence oracle.
- `superposition/`: fixed fusion/objectives, recovery schedules, sparse soft
  tokens, and complete-link safety clustering.
- `prepare_data.py`: deterministic document splits, aligned ztok offsets, and
  exact source-byte accounting.
- `train.py`: equal-step, equal-source-byte, equal-FLOP, or equal-device-time
  training with checkpoint/resume.
- `branch_equivalence.py`: fused state versus explicit branches at horizons
  0/1/4/16/64.
- `evaluate.py`: ordinary, guarded assisted, and checkpointed latent inference.
- `report.py`: JSON/Markdown summaries and the required plots.
- `make_matrix.py`: materializes paired screening or full-matrix configs.

## Environment

Build ztok and make both local packages importable:

```bash
zig build -Doptimize=ReleaseFast
python -m pip install -e bindings/python
python -m pip install -e /thearray/git/moe-mla
python -m pip install -r experiments/nanogpt_superposition/requirements.txt
```

For serious RWKV runs, use moe-mla's qualified CUDA environment. The RWKV-Lab
parallel scan is a CUDA/Triton path; CPU smoke tests deliberately select
`rwkv_backend: "reference"`.

### xIELU ChannelMix backends

RWKV-Lab experiments may set `rwkv_channel_activation` to:

- `relu_squared`: the unchanged RWKV ChannelMix baseline;
- `xielu`: the auditable PyTorch reference;
- `xielu_cuda`: Nathan Ranchin's standalone BF16 CUDA kernel, pinned for the
  reported run at commit `185b5126b4d3d6d8b8795f9476b6c61d6e952c3f`;
- `xielu_fused`: a Triton `key projection -> xIELU` GEMM epilogue with the same
  standalone CUDA backward and a fail-closed fallback for unsupported shapes.

The fused forward rounds the GEMM result to BF16 before evaluating xIELU in
FP32, matching the standalone kernel's numerical boundary. It saves both that
pre-activation and the BF16 activated tensor because they are required by the
two weight gradients. Tests compare outputs, inputs, both matrices, and both
per-layer curvature scalars. Fusion remains an explicit experiment condition;
it never changes the ordinary RWKV or tokenizer paths.

## Prepare identical document splits

Train ordinary BPE and SuperBPE vocabularies of the same size from the same raw
training documents. SuperBPE is a different tokenizer condition, not a hidden
change to the model objective.

```bash
zig-out/bin/ztok train --kind bpe --input corpus.txt --vocab-size 32768 \
  --output ordinary.tiktoken --cl100k
zig-out/bin/ztok train --kind superbpe --input corpus.txt --vocab-size 32768 \
  --superword-phase-vocab 29491 --output superbpe.tiktoken --cl100k

PYTHONPATH=bindings/python python experiments/nanogpt_superposition/prepare_data.py \
  --input corpus.txt --line-documents --tokenizer ordinary.tiktoken \
  --tokenizer-kind bpe --output experiments/nanogpt_superposition/data/main-bpe
PYTHONPATH=bindings/python python experiments/nanogpt_superposition/prepare_data.py \
  --input corpus.txt --line-documents --tokenizer superbpe.tiktoken \
  --tokenizer-kind superbpe --output experiments/nanogpt_superposition/data/main-superbpe
```

Both metadata files record source bytes, ordinary tokens, bytes/token, document
ordering, language mix, tokenizer hash, and train/validation split hashes. Check
that their source and split hashes match before comparing tokenizers.
`make_matrix.py` enforces those hashes and converts its `--context-bytes` target
to a tokenizer-specific token length (rounded to a group-compatible multiple),
so ordinary BPE and SuperBPE do not receive silently different byte contexts.

Tiny Shakespeare or `bench/corpora/` is for plumbing only. A conclusion requires
a fixed, legally usable primarily-English 100M–500M-token shard.

## Statement-expansion primer

`superposition/statement_primer.py` consumes the conservative one-slot
ProofWriter inventory produced by `build_proofwriter_statement_expansions.py`.
It retains only branch sets whose tokenizations are equal length and differ at
exactly one causal-input position. At that position it inserts a
norm-preserving weighted embedding mixture; the preceding output predicts the
complete alternative set, while all shared continuation positions retain the
ordinary causal objective.

Set `training.primer_path`, `primer_epochs`, and `primer_batch_size` to run this
phase before the configured ordinary corpus. Primer steps, explicit source
bytes, positions, FLOPs, and GPU-seconds are logged separately. The model and
optimizer state carry across the phase boundary, but the ordinary sampling RNG,
ordinary step counter, LR schedule, and ordinary data budget begin at zero. This
makes the post-primer ordinary stream directly comparable to an unprimed run
while ensuring the primer's extra compute remains visible.

## Legacy fixed-contiguous control

The following matrix and training objective describe the earlier contiguous
control. Use it only as a baseline against the dataset-mapped experiment.

### Build and launch the control matrix

```bash
python experiments/nanogpt_superposition/make_matrix.py \
  --ordinary-data experiments/nanogpt_superposition/data/main-bpe \
  --superbpe-data experiments/nanogpt_superposition/data/main-superbpe \
  --output experiments/nanogpt_superposition/configs/generated

PYTHONPATH=bindings/python python experiments/nanogpt_superposition/train.py \
  --config experiments/nanogpt_superposition/configs/generated/<run>.json
```

The default generated matrix is the requested first screen for both
architectures: ordinary BPE baseline, groups 2 and 4, SuperBPE baseline, and
SuperBPE plus group 2. `--full-matrix` expands mean versus norm-preserving fusion.
The generated output directories point at moe-mla's dashboard runs directory.

Generate the three budget views separately with `--budget-kind steps`,
`--budget-kind source_bytes`, and either `--budget-kind flops` or
`--budget-kind gpu_seconds`; `--budget-value` sets the latter three horizons.

The default recovery schedule is 50/25/25 by source-byte budget. Configs may use
75/15/10 or 30/30/40, and may choose `steps`, `source_bytes`, `flops`, or
`gpu_seconds`. Equal steps are a debug view only; conclusions use source bytes and
GPU-seconds.

For learning-rate ablations, `training.coarse_lr_multiplier` scales the base
schedule during the coarse phase and linearly returns to `1.0` through the mixed
phase for either ordinary or fixed representations. For example,
`make_matrix.py --coarse-lr-multiplier 1.4142135623730951` creates the
square-root-of-two condition without changing ordinary runs.

`coarse_superposed_source_gradient_multiplier` scales only the gradient entering
the fused source-token activation during coarse fixed-superposition training; it
does not scale downstream model or output-head gradients.
`mixed_superposed_source_gradient_multiplier` independently controls that same
source-only gradient path on superposed batches during the mixed phase. Neither
setting changes the optimizer LR or any downstream parameter gradient.

To reuse an equal-byte/step trajectory for an equal-compute recovery point,
extend the fixed checkpoint to the baseline's cumulative device time:

```bash
python experiments/nanogpt_superposition/train.py \
  --config configs/fixed2.json \
  --resume RUN/checkpoint.pt \
  --extend-gpu-seconds-to 311.52
```

This appends to the same metrics and preserves model, optimizer, sampling, and
RNG state. Extension is rejected for budget-relative LR schedules because
changing their denominator mid-run would silently redefine the schedule; declare
the combined budget up front in that case.

### Fixed objective

For source groups `G_t`, the model sees one mean, norm-preserving mean, or learned
ordered fusion per group. The bag condition minimizes the average negative log
probability of every member of `G_(t+1)` and is invariant to group order. The
ordered condition uses `group_size` auxiliary vocabulary heads. The trainer first
audits its vectorized grouping against an actual `ztok.superposition.v1` plan.

Ordinary validation is always performed with discrete tokens. Cross-tokenizer
comparisons use bits/byte; perplexity is retained only for within-tokenizer views.

## Soft-token and branch gate

Soft-token exposure is disabled by default. Once a fixed condition passes the 15%
compute gate, set `soft_token_rate` to at most `0.20`. The trainer applies the
0/5/10/20%-style curriculum, keeps the true token dominant, and adds learned type,
entropy, cluster-mass, and dispersion embeddings.

Before recurrent generation, run:

```bash
python experiments/nanogpt_superposition/branch_equivalence.py \
  --checkpoint RUN/checkpoint.pt \
  --cases experiments/nanogpt_superposition/data/semantic_cases.json \
  --output RUN/branch_equivalence.json --device cuda
```

Candidate fusion is sparse and complete-link constrained. Protected contradiction,
binding, and exact-sensitive pairs fall back to a discrete token. Assisted/latent
evaluation refuses checkpoints with no soft-token training exposure.

## Reports and tests

```bash
python -m pytest experiments/nanogpt_superposition/tests -q
python experiments/nanogpt_superposition/report.py \
  --run-dir RUN_A --run-dir RUN_B \
  --branch-report RUN_A/branch_equivalence.json \
  --output-dir REPORT
```

Reports include exact commits/configs, parameters, source bytes, model positions,
estimated FLOPs, measured time/VRAM, bits/byte, recovery cost, branch KL/state
error, safe-fusion precision/recall, and all required plots. A shorter sequence by
itself is never treated as a positive result.
