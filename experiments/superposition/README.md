# Superposition experiments

These scripts keep training and contextual-embedding dependencies outside the
tokenizer library. Run them from the repository root after building
`libztok`:

```bash
zig build -Doptimize=ReleaseFast
export PYTHONPATH="$PWD/bindings/python"
export ZTOK_LIB_PATH="$PWD/zig-out/lib/libztok.so"
```

## 1. Fixed-token sanity check

The default run creates 10,000 deterministic sequences, trains an ordinary
tiny decoder and fixed-group 2/4 variants, then measures ordinary-token
recovery:

```bash
python experiments/superposition/fixed_lm_sanity.py \
  --json-output bench/results/superposition/fixed_lm_sanity.json \
  --markdown-output bench/results/superposition/fixed_lm_sanity.md
```

This tests the plan/embedding/loss plumbing. It is not a reproduction of the
paper's training scale.

## 2. Synthetic semantic safety benchmark

```bash
python experiments/superposition/synthetic_semantic_benchmark.py \
  --json-output bench/results/superposition/semantic_synthetic.json \
  --markdown-output bench/results/superposition/semantic_synthetic.md
```

The benchmark fails unless merge precision is 100%, protected contradictions
remain separate, and all required unique details survive.

## 3. Frozen-encoder offline benchmark

First export one `ztok.semantic_spans.v1` document per image. Contextual span
vectors must identify the exact encoder model, layer, and pooling policy. For
Qwen, test at least a middle layer, a late layer, and an average of selected
layers as separate inputs.

```bash
python experiments/superposition/ccss_offline_benchmark.py \
  --input semantic_spans.jsonl \
  --config ccss_config.json \
  --tokenizer-identifier tokenizer.json \
  --vocabulary-identifier sha256:... \
  --embedding-model Qwen/... \
  --embedding-layer middle-18 \
  --span-extraction-policy noun-adjective-chunks-v1 \
  --run-ablations \
  --json-output bench/results/superposition/ccss_offline.json \
  --markdown-output bench/results/superposition/ccss_offline.md
```

The report leaves reconstruction, learned-resampler, and annotated
false-merge gates explicitly unknown when the input lacks those measurements.
Compression alone never produces a go decision.

Do not start generator integration until the offline report has annotations
for protected false merges and unique-detail retention, a caption
reconstruction comparison against mean pooling, and a learned-resampler
comparison. The tiny conditioning-adapter experiment belongs in the image
training project after those gates pass; it is deliberately not coupled to
the tokenizer core.
