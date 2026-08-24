# i1 CCSS offline benchmark readiness

- Git commit: `72c52c2da8e2456dbf6e5b2e0cdee68b8cb0c320` (dirty working tree)
- Source: `/thearray/git/datasets/i1/qwen3.6-i1-modular.jsonl`
- Caption model recorded by the source: `Qwen3.6-35B-A3B-heretic`
- Selected: 1,000 image groups × 12 caption-stage phrases
- Sampling: deterministic round-robin across 12 observed media,
  subject-count, OCR, and scene-complexity strata
- Contextual embedding status: **not run**
- Generator integration decision: **no-go**

The i1 source contains structured captions and role/entity provenance, but no
contextual token hidden states. No local Qwen encoder checkpoint or
precomputed hidden-state/span matrix was available, and the active generation
service does not expose hidden states. Running clustering on fabricated,
lexical, or unrelated embeddings would make the semantic results meaningless.

The selected groups were exported successfully to
`/tmp/i1-ccss-caption-groups-stratified.jsonl`. The reproducible next step is:

```bash
python examples/python/qwen_span_adapter.py \
  --input /tmp/i1-ccss-caption-groups-stratified.jsonl \
  --output-prefix /path/to/i1-qwen-spans \
  --tokenizer /path/to/the/exact/qwen/tokenizer.json \
  --model /path/to/the/exact/qwen/checkpoint \
  --layers MIDDLE,LATE \
  --average-layers MIDDLE,SELECTED,LATE
```

Then run `experiments/superposition/ccss_offline_benchmark.py` separately for
each layer/pooling export. Until those vectors and the reconstruction/resampler
comparisons exist, compression, detail-retention, false-merge, retrieval, and
generator gates remain unknown rather than being inferred.
