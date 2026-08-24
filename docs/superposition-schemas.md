# Experimental superposition schemas

All ranges are half-open. Schema names, field meanings, enum strings, and
fusion formulas in this document are fixed for version 1.

## `ztok.superposition.v1`

A fixed plan contains the untouched ordinary token count and an ordered
`groups` array. Every source records its ordinary `token_index`, `token_id`,
original-input byte range, and normalized weight. A group records its covered
source-position and byte ranges, center position, fusion operation, and reason:

- `fixed_window`
- `partial_window`
- `preserved_special`
- `preserved_boundary`
- `uncovered_tail`

No group is a vocabulary item and no output index may be passed to a tokenizer
decoder as a token ID.

For source embeddings `e_i` and serialized weights `w_i`:

```text
mean:
  m = sum(e_i) / n

weighted_mean:
  m = sum(w_i * e_i) / sum(w_i)

norm_preserving_mean:
  u = sum(w_i * e_i) / sum(w_i)
  target_norm = sum(w_i * ||e_i||) / sum(w_i)
  m = normalize(u) * target_norm
```

The core serializes these operations; it never reads or calculates model
embeddings.

## `ztok.semantic_spans.v1`

This exchange input contains an image ID and caption records. Caption tokens
carry ordinary token IDs and original-input offsets. Semantic spans carry:

- stable `span_id` and caption provenance;
- token and byte ranges;
- phrase text;
- optional role and entity labels;
- confidence and contextual embedding;
- optional caption quality, section reliability, grounding vector/confidence;
- optional relation endpoints and explicit contradiction IDs/scores.

Embeddings for all spans in one document must have the same width. If
grounding vectors are used, they must be present for every span and have that
same width. For large arrays a future schema may specify a binary sidecar;
version 1 is intentionally JSON-first and inspectable.

## `ztok.ccss.v1`

The output is an ordered list of `consensus`, `unique`, `relation`,
`uncertainty`, and `separator` units. Consensus units record member span IDs
and caption IDs, normalized fusion weights, support, mean/minimum pair cosine,
dispersion, fusion operation, and grouping reason. Unique/relation units retain
their original caption, token, byte, confidence, and text provenance.

Default clustering is conservative:

1. only cross-caption pairs are candidates;
2. known roles and entities must match;
3. explicit contradictions are blocked;
4. protected roles use stricter thresholds or are never fused;
5. every cross-pair in a merged cluster must pass its threshold;
6. a cluster contains at most one span from each caption;
7. cluster size is capped.

Verbose diagnostics include rejected pairs and rejection reasons. Normal
output omits this potentially quadratic array.

Ordering uses semantic buckets first and first source occurrence as the stable
fallback. Similarity never determines output order by itself.
