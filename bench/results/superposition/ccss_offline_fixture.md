# CCSS offline benchmark

- Git commit: `72c52c2da8e2456dbf6e5b2e0cdee68b8cb0c320` (dirty)
- Tokenizer/vocabulary: `fixture-offsets` / `fixture-token-ids`
- Embedding model/layer: `hand-authored-vectors` / `n/a`
- Span extraction: `fixture-phrases-v1`
- Images/captions: 1 / 3
- Source tokens / spans / output units: 18 / 6 / 3
- Consensus / unique / relation units: 2 / 0 / 1
- Compression: **6.000x** (0.16667 output units/source token)
- Runtime: 0.002s; peak RSS: 166512 KiB

## Cluster examples

- `fixture-lighting` / `lighting_intensity`: c0-bright, c1-brilliant, c2-intense (min cosine 0.9992, support 100.0%)
- `fixture-lighting` / `color`: c0-white, c1-pale (min cosine 0.9998, support 66.7%)

## Go/no-go gates

- At least 2x compression: `True`
- Protected false-merge rate below 1%: `None`
- Unique-detail retention at least 95%: `None`
- Beats mean pooling on reconstruction/detail: `None`
- No significant learned-resampler degradation: `None`

Unknown gates require expected-pair/detail annotations and downstream decoder/resampler metrics; they are never inferred from compression alone.
