# Token Superposition and Cross-Caption Semantic Superposition

## Purpose

Implement two experimental capabilities in `ztok` without changing ordinary tokenization behavior or breaking bit-identical compatibility:

1. **Baseline token superposition support** inspired by *Efficient Pre-Training with Token Superposition* (`arXiv:2605.06546`).
2. **Cross-Caption Semantic Superposition (CCSS)** for merging semantically aligned concepts across multiple captions of the same image while preserving unique concepts and relation structure.

This document is an implementation brief for Codex. Work incrementally. The baseline tokenizer must remain unchanged unless a caller explicitly enables an experimental superposition API or CLI command.

---

# 1. Design constraints

## 1.1 Preserve normal tokenizer behavior

The following must remain bit-identical when superposition is disabled:

- `Pipeline.encode`
- `Pipeline.encodeWithOffsets`
- batch encoding
- streaming encoding
- all C ABI functions
- all language bindings
- all existing tokenizer file readers and writers

Do not introduce synthetic superposition IDs into the base vocabulary. A superposed unit is a **training-time descriptor over one or more existing token IDs**, not a replacement tokenizer vocabulary entry.

## 1.2 Keep embeddings outside the tokenizer core

`ztok` owns tokenization, offsets, grouping metadata, and export formats. It does not own the model embedding matrix.

Therefore, `ztok` should emit:

- source token IDs;
- source byte spans;
- group membership;
- fusion weights;
- target-position metadata;
- optional semantic-role and caption-origin metadata;
- an embedding-operation plan that downstream Python/C++/Zig training code can execute.

Do not add a dependency on PyTorch, CUDA, Qwen, or a particular model architecture to the tokenizer core.

## 1.3 Deterministic and inspectable

Given the same input, configuration, tokenizer, embeddings, and seed, outputs must be deterministic.

Every experimental result must be explainable through a JSON representation showing:

- which source tokens or spans were grouped;
- why they were grouped;
- their similarity scores;
- their fusion weights;
- which concepts remained separate;
- which relation tokens connect the retained concepts.

## 1.4 Experimental APIs must be versioned

Use an explicit schema version in serialized outputs:

```json
{
  "schema": "ztok.superposition.v1"
}
```

Do not silently change the meaning of fields later.

---

# 2. Phase A: baseline token superposition

Implement the paper-like primitive first, before semantic matching.

## 2.1 Definition

Given a token sequence:

```text
[t0, t1, t2, t3, t4, t5, ...]
```

and a group size of four, produce descriptors:

```text
S0 = superpose(t0, t1, t2, t3)
S1 = superpose(t4, t5, t6, t7)
...
```

The tokenizer output becomes a shorter sequence of **superposition groups**, where each group references the original tokens. The downstream model may construct each superposed embedding using a weighted mean or another configured fusion rule.

This phase does not implement the paper's model loss. It supplies the tokenizer-side grouping and embedding plan needed to test that loss in a training project.

## 2.2 Core data types

Add a module such as:

```text
src/superposition.zig
```

Suggested types:

```zig
pub const SuperpositionSchemaVersion = 1;

pub const FusionKind = enum {
    mean,
    weighted_mean,
    norm_preserving_mean,
};

pub const SourceToken = struct {
    token_index: u32,
    token_id: TokenId,
    byte_start: u32,
    byte_end: u32,
    weight: f32,
};

pub const SuperpositionGroup = struct {
    output_index: u32,
    sources: []SourceToken,
    fusion: FusionKind,
    position_start: u32,
    position_end: u32,
};

pub const SuperpositionPlan = struct {
    schema_version: u32,
    original_token_count: u32,
    output_token_count: u32,
    groups: []SuperpositionGroup,
};
```

Use repository conventions for allocation and ownership. Do not add per-token heap allocation to existing hot paths. Experimental APIs may allocate through a caller-provided allocator.

## 2.3 Configuration

Suggested configuration:

```zig
pub const FixedSuperpositionConfig = struct {
    group_size: u16 = 4,
    stride: ?u16 = null,
    preserve_special_tokens: bool = true,
    preserve_boundary_tokens: bool = true,
    allow_partial_final_group: bool = true,
    fusion: FusionKind = .norm_preserving_mean,
};
```

Default stride should equal group size.

Special tokens must never be averaged with ordinary lexical tokens unless explicitly requested.

When `preserve_boundary_tokens` is enabled, use token offsets and overlay/boundary metadata to avoid crossing hard boundaries such as:

- caption separators;
- document separators;
- user-defined protected spans;
- multimodal sentinel tokens;
- start/end/control tokens.

## 2.4 API shape

Add an API that accepts an already encoded sequence plus offsets:

```zig
pub fn buildFixedPlan(
    allocator: std.mem.Allocator,
    token_ids: []const TokenId,
    offsets: []const TokenOffset,
    config: FixedSuperpositionConfig,
) !SuperpositionPlan
```

Also add a convenience pipeline API:

```zig
pub fn encodeWithSuperpositionPlan(
    self: *const Pipeline,
    allocator: std.mem.Allocator,
    text: []const u8,
    config: FixedSuperpositionConfig,
) !EncodedSuperposition
```

where `EncodedSuperposition` contains the original encoded tokens, offsets, and plan.

Do not make `Pipeline.encode` return synthetic IDs.

## 2.5 Embedding plan semantics

For a group with source embeddings `e_i` and weights `w_i`, support these downstream formulas.

### Mean

```text
m = sum(e_i) / n
```

### Weighted mean

```text
m = sum(w_i * e_i) / sum(w_i)
```

### Norm-preserving mean

```text
u = sum(w_i * e_i) / sum(w_i)
mean_norm = sum(w_i * ||e_i||) / sum(w_i)
m = normalize(u) * mean_norm
```

`ztok` should serialize the operation and weights, not calculate model embeddings inside the core library.

## 2.6 Position semantics

Each output group must retain the covered source-position range. Provide at least:

- first source token index;
- last source token index;
- first byte offset;
- last byte offset;
- optional normalized center position.

Downstream models may choose one of:

- mean source position;
- first position;
- learned group position;
- position range encoding.

Do not force one positional policy into the tokenizer.

## 2.7 CLI

Add an experimental CLI command or subcommand, preferably:

```bash
ztok superpose fixed \
  --model tokenizer.json \
  --group-size 4 \
  --fusion norm-preserving-mean \
  --text "A bright white light fills the room."
```

Human-readable output should show each group and source token.

JSON mode should produce the complete versioned plan:

```bash
ztok superpose fixed ... --json
```

## 2.8 Baseline tests

Add tests for:

1. exact grouping with divisible token counts;
2. final partial group;
3. protected special tokens;
4. no crossing caption separators;
5. offsets preserved exactly;
6. deterministic JSON serialization;
7. normal encode output unchanged;
8. empty input;
9. one-token input;
10. very large input;
11. Unicode and multi-byte spans;
12. all supported tokenizer families.

---

# 3. Phase B: cross-caption semantic superposition

Implement CCSS only after Phase A is correct and tested.

## 3.1 Goal

Given 5–12 captions describing one image, create a compact semantic conditioning plan that:

- fuses repeated or near-equivalent concepts across captions;
- preserves unique details;
- avoids merging related but distinct attributes;
- retains relation and syntax-bearing spans needed to connect concepts;
- records support count, agreement, and uncertainty;
- can be converted downstream into fused conditioning embeddings.

Example input:

```text
A bright white light illuminates the room.
A brilliant luminous glow fills the chamber.
An intense pale light casts illumination across the interior.
```

Desired conceptual result:

```text
ENTITY(room/interior)
ENTITY(light/glow)
ATTRIBUTE_INTENSITY{bright, brilliant, intense}
ATTRIBUTE_COLOR{white, pale}
ATTRIBUTE_EMISSION{luminous}
RELATION(light -> illuminates -> room)
```

Do not collapse all related words into one cluster. `bright`, `white`, and `luminous` are associated, but they occupy different semantic roles.

## 3.2 Separation of responsibilities

The `ztok` core should provide:

- caption tokenization;
- byte offsets;
- caption and section provenance;
- span construction support;
- candidate matching utilities over caller-supplied vectors and labels;
- constrained clustering;
- fusion-plan export;
- diagnostics.

An external semantic adapter should provide:

- contextual token/span embeddings from Qwen or another encoder;
- optional semantic-role labels;
- optional subject/entity IDs;
- optional image-grounding maps or grounding similarity;
- optional contradiction scores.

Do not compile Qwen inference into `ztok`.

## 3.3 Input schema

Define a versioned exchange format such as:

```json
{
  "schema": "ztok.semantic_spans.v1",
  "image_id": "example-123",
  "captions": [
    {
      "caption_id": 0,
      "section": "lighting",
      "text": "A bright white light illuminates the room.",
      "tokens": [
        {
          "token_index": 0,
          "token_id": 123,
          "byte_start": 0,
          "byte_end": 1
        }
      ],
      "spans": [
        {
          "span_id": "c0-s0",
          "token_start": 1,
          "token_end": 2,
          "text": "bright",
          "role": "lighting_intensity",
          "entity_id": "light-1",
          "embedding": [0.01, 0.02],
          "grounding": null,
          "confidence": 0.94
        }
      ]
    }
  ]
}
```

For large embedding arrays, support a binary sidecar format later. Start with JSON for testability.

## 3.4 Work on spans, not raw subword pieces

Semantic matching must operate on word or phrase spans rather than isolated BPE/Unigram pieces.

Examples that must remain whole when possible:

- `luminous` even if split into multiple tokenizer tokens;
- `soft directional light`;
- `looking over her shoulder`;
- `highly detailed`;
- `in the upper-left corner`.

Provide utilities to map span byte ranges back to token ranges through existing offset support.

Suggested type:

```zig
pub const SemanticSpan = struct {
    span_id: []const u8,
    caption_id: u32,
    section_id: ?u32,
    token_start: u32,
    token_end: u32,
    byte_start: u32,
    byte_end: u32,
    role_id: ?u32,
    entity_id: ?u32,
    confidence: f32,
    embedding_index: u32,
};
```

Embeddings should be stored separately in a contiguous matrix.

## 3.5 Candidate similarity

For two spans `i` and `j`, compute a configurable score:

```text
score(i,j) =
    semantic_weight * cosine(embedding_i, embedding_j)
  + role_weight     * role_compatibility(i,j)
  + entity_weight   * entity_compatibility(i,j)
  + grounding_weight* grounding_similarity(i,j)
  - contradiction_weight * contradiction(i,j)
```

Do not merge based on cosine similarity alone.

Required hard constraints for the first implementation:

- spans must come from different captions;
- at most one span from any caption may enter a cluster;
- roles must match exactly when both roles are known;
- entity IDs must match when both are known;
- explicit contradiction blocks a merge;
- all pairwise similarities inside a cluster must pass the threshold;
- cluster size must not exceed a configured maximum.

The all-pairs condition prevents semantic chaining such as:

```text
bright ~ luminous
luminous ~ white
therefore bright ~ white
```

when `bright` and `white` should remain separate.

## 3.6 Role-specific thresholds

Support a global default plus role overrides:

```json
{
  "default_cosine_threshold": 0.92,
  "role_thresholds": {
    "lighting_intensity": 0.88,
    "color": 0.94,
    "spatial_relation": 0.97,
    "count": 0.995
  }
}
```

Counts, negations, directions, and mutually exclusive states should use very strict matching or no fusion at all.

Protected roles should include at least:

- count;
- negation;
- left/right;
- above/below;
- front/behind;
- standing/sitting/lying;
- day/night;
- clothed/unclothed;
- object identity where ambiguity is dangerous.

## 3.7 Matching and clustering algorithm

Implement a clear reference algorithm before optimizing.

Recommended initial algorithm:

1. Generate candidate pairs across different captions.
2. Remove pairs that violate hard constraints.
3. Calculate pair score and role-specific threshold.
4. Sort candidate pairs by descending score.
5. Greedily build clusters while enforcing:
   - one span per caption;
   - complete-link threshold across all current members;
   - role and entity compatibility;
   - maximum cluster size.
6. Leave unmatched spans as individual semantic units.
7. Preserve connector/relation spans even when they have no cross-caption match.

Later alternatives may include bipartite matching, constrained agglomerative clustering, or optimal transport, but do not begin there.

## 3.8 Consensus group output

Suggested output type:

```zig
pub const ConsensusGroup = struct {
    group_id: u32,
    role_id: ?u32,
    entity_id: ?u32,
    member_span_indices: []u32,
    member_caption_ids: []u32,
    support_count: u16,
    support_fraction: f32,
    mean_similarity: f32,
    minimum_pair_similarity: f32,
    dispersion: f32,
    fusion: FusionKind,
    weights: []f32,
};
```

Fusion weights may combine:

- source confidence;
- caption quality;
- centrality within the cluster;
- grounding confidence;
- section reliability.

Normalize weights to sum to one.

## 3.9 Unique and connector spans

Not every unmatched span should be discarded.

Retain unmatched spans when they are:

- an entity;
- an attribute absent from other captions;
- an action;
- a relation;
- a quantifier;
- a negation;
- a spatial connector;
- an uncertainty marker;
- visible text/OCR content;
- a rare detail with sufficient confidence.

Add a classification field:

```zig
pub const UnitKind = enum {
    consensus,
    unique,
    relation,
    uncertainty,
    separator,
};
```

The final output should be an ordered semantic-unit sequence or graph-like plan. It does not need to be valid natural language.

## 3.10 Ordering semantic units

Use deterministic ordering:

1. global scene/entity units;
2. main subjects in spatial or source order;
3. attributes attached to each entity;
4. actions and relations;
5. environment;
6. lighting and appearance;
7. style/medium;
8. visible text;
9. uncertainty and alternatives.

When role information is unavailable, use first occurrence across captions as the stable fallback.

Do not order fused units solely by vector similarity.

## 3.11 Contradictions and alternatives

Do not average contradictory observations.

Examples:

- `white light` versus `warm yellow light`;
- `standing` versus `seated`;
- `one person` versus `two people`;
- `left` versus `right`.

Represent conflicting groups separately and connect them with an uncertainty/alternative unit:

```json
{
  "kind": "uncertainty",
  "alternatives": [12, 18],
  "reason": "role-compatible spans form distinct semantic modes"
}
```

Start with conservative contradiction handling. False non-merges are safer than false merges.

## 3.12 Output schema

Add a JSON serializer with enough detail for audit and downstream training:

```json
{
  "schema": "ztok.ccss.v1",
  "image_id": "example-123",
  "caption_count": 8,
  "units": [
    {
      "unit_id": 0,
      "kind": "consensus",
      "role": "lighting_intensity",
      "members": ["c0-s2", "c1-s4", "c3-s1"],
      "weights": [0.34, 0.33, 0.33],
      "support_fraction": 0.375,
      "fusion": "norm_preserving_mean"
    }
  ],
  "relations": [],
  "diagnostics": {
    "original_token_count": 241,
    "semantic_span_count": 73,
    "output_unit_count": 39,
    "compression_ratio": 0.1618,
    "rejected_pairs": []
  }
}
```

Include rejected-pair diagnostics in verbose mode only to avoid huge normal outputs.

## 3.13 CLI

Add a command such as:

```bash
ztok superpose semantic \
  --input semantic_spans.json \
  --config ccss_config.json \
  --output ccss_plan.json
```

Also support:

```bash
ztok superpose inspect --plan ccss_plan.json
```

The inspection view should print:

- cluster members;
- role;
- support;
- minimum/mean similarity;
- reasons candidate merges were rejected;
- unique spans retained;
- contradictions detected.

---

# 4. Language bindings

Expose the plan-building APIs through the stable C ABI, then add a Python wrapper first.

Python is required for the initial experiment because Qwen embeddings and image-model training are likely to run in PyTorch.

Suggested Python API:

```python
from ztok.superposition import (
    FixedSuperpositionConfig,
    SemanticSpan,
    SemanticSuperpositionConfig,
    build_fixed_plan,
    build_semantic_plan,
)

plan = build_fixed_plan(
    pipeline,
    text,
    FixedSuperpositionConfig(group_size=4),
)

semantic_plan = build_semantic_plan(
    spans=spans,
    embeddings=embedding_matrix,
    config=config,
)
```

The Python wrapper should return typed dataclasses or dictionaries plus an option to retrieve zero-copy/contiguous numeric arrays where practical.

Do not block Phase A on all eight bindings. Complete Zig, C ABI, CLI, and Python first, then expand bindings after the schema stabilizes.

---

# 5. Downstream PyTorch reference adapter

Add a small example outside the tokenizer core, such as:

```text
examples/python/superposition_embeddings.py
```

It should demonstrate applying a plan to an embedding tensor.

Pseudocode:

```python
def apply_superposition_plan(token_embeddings, plan):
    outputs = []
    for group in plan.groups:
        idx = torch.tensor([s.token_index for s in group.sources])
        weights = torch.tensor([s.weight for s in group.sources])
        x = token_embeddings.index_select(0, idx)

        if group.fusion == "mean":
            fused = x.mean(dim=0)
        elif group.fusion == "weighted_mean":
            fused = (x * weights[:, None]).sum(0) / weights.sum()
        elif group.fusion == "norm_preserving_mean":
            raw = (x * weights[:, None]).sum(0) / weights.sum()
            target_norm = (x.norm(dim=-1) * weights).sum() / weights.sum()
            fused = torch.nn.functional.normalize(raw, dim=-1) * target_norm
        else:
            raise ValueError(group.fusion)

        outputs.append(fused)

    return torch.stack(outputs)
```

For CCSS, source indices refer to contextual span embeddings rather than the tokenizer embedding table.

This example must explicitly state that contextual span vectors should come from a chosen Qwen layer and pooling policy.

---

# 6. Minimal small-scale experiments

Implement tooling for the following tests before integrating CCSS into the full MageFlow run.

## 6.1 Test 1: fixed token superposition sanity check

### Dataset

Use a small text corpus of approximately 10,000–100,000 sequences.

### Model

Use a tiny decoder-only transformer or an existing small language-model training harness.

### Conditions

- baseline ordinary tokens;
- fixed groups of 2;
- fixed groups of 4;
- recovery phase with ordinary tokens.

### Measurements

- training tokens/second;
- sequence-length reduction;
- loss during superposition phase;
- ordinary-token validation loss after recovery;
- whether recovery reaches baseline quality at lower wall-clock cost.

This test validates the plumbing and paper-like primitive. It does not validate CCSS.

## 6.2 Test 2: synthetic semantic-clustering unit set

Create hand-authored caption groups containing:

- true synonyms;
- related but distinct attributes;
- antonyms;
- different entities sharing an adjective;
- multi-token phrases;
- counts and directions;
- ambiguous cases.

Required cases include:

```text
bright / brilliant / intense
white / pale
luminous / glowing
standing / seated
left / right
one person / two people
bright room / bright dress
red dress / crimson gown
looking over her shoulder / glancing backward
```

Manually define expected clusters and non-clusters.

Metrics:

- pair precision;
- pair recall;
- cluster purity;
- contradiction false-merge rate;
- retained unique-detail rate.

Set the primary safety requirement to **very high merge precision**, even at lower recall.

## 6.3 Test 3: frozen-encoder caption compression benchmark

### Dataset

Select 1,000–10,000 i1 images with 5–12 captions each.

Stratify by:

- photo versus animation;
- one versus multiple subjects;
- OCR/text-heavy images;
- simple versus complex scenes;
- common versus rare styles;
- caption disagreement level.

### Precomputation

For every caption:

1. tokenize with `ztok`;
2. run the frozen Qwen text encoder;
3. export contextual token hidden states from several candidate layers;
4. pool token ranges into semantic spans;
5. attach caption-stage/section labels;
6. optionally attach entity and role labels from the existing caption pipeline.

Compare Qwen layers rather than assuming the final layer is best. Test at least one middle and one late layer.

### Conditioning representations

Compare:

A. one random caption;

B. all captions concatenated;

C. mean-pooled caption vectors;

D. learned fixed-slot resampler;

E. CCSS fused units plus retained unique/relation units.

### Offline metrics before image-model training

- compression ratio;
- unique-detail retention;
- reconstruction of all source captions using a small training-only decoder;
- nearest-neighbor semantic coherence;
- false contradiction merges;
- support-count calibration;
- image-text retrieval score if image embeddings are available.

CCSS should not proceed to generator training unless it beats simple mean pooling and is competitive with the learned resampler on detail retention.

## 6.4 Test 4: tiny image-model conditioning experiment

Do not begin with the full 12-block MageFlow model.

Use one of:

- a very small DiT/flow transformer;
- a reduced-depth MageFlow configuration;
- frozen MageFlow with only a small conditioning adapter trained;
- a LoRA on the text-conditioning projections.

Recommended first test:

- freeze the image model;
- train only a small adapter that maps each conditioning representation into the existing text-conditioning width;
- use 10,000–100,000 images;
- use identical images, noise, timesteps, and optimizer settings across conditions.

Conditions:

1. random single caption;
2. all captions concatenated;
3. learned resampler;
4. CCSS.

Measurements:

- flow validation loss;
- validation loss on rare attributes;
- caption-detail recall in generated samples;
- prompt adherence under ordinary single prompts;
- conditioning token count;
- training throughput;
- memory use.

The key question is whether CCSS gives lower validation loss or better detail recall per conditioning token than concatenation and learned resampling.

## 6.5 Ablation matrix

At minimum test:

### Similarity threshold

```text
0.85
0.88
0.90
0.92
0.94
0.96
```

### Fusion

```text
mean
weighted mean
norm-preserving mean
learned set fusion downstream
```

### Constraints

```text
cosine only
cosine + same role
cosine + role + entity
cosine + role + entity + grounding
```

### Caption support

```text
minimum support 2
minimum support 3
minimum support 50% of captions
```

### Contextual layer

```text
middle Qwen layer
late Qwen layer
average of selected layers
```

### Span granularity

```text
word spans
noun/adjective chunks
caption-stage-specific phrases
```

---

# 7. Acceptance criteria

## Phase A is complete when

1. normal tokenization remains bit-identical;
2. fixed grouping plans are deterministic;
3. offsets and protected boundaries are correct;
4. JSON plan serialization is stable and versioned;
5. Zig, CLI, C ABI, and Python tests pass;
6. the PyTorch example reconstructs expected fused embeddings;
7. a tiny language-model test can consume the plan.

## Phase B is complete when

1. contextual spans from multiple captions can be loaded;
2. candidate similarities are computed correctly;
3. role/entity/contradiction constraints are enforced;
4. complete-link clustering prevents chaining errors;
5. unique and relation spans are retained;
6. contradictions remain separate;
7. fusion plans serialize deterministically;
8. synthetic clustering tests meet the precision target;
9. the i1 offline benchmark produces an auditable report;
10. no model-specific dependency enters the tokenizer core.

## Initial go/no-go criteria for generator integration

Proceed only if CCSS demonstrates all of the following on the small benchmark:

- false-merge rate below 1% on protected roles;
- at least 95% retention of high-confidence unique details;
- at least 2x conditioning-token compression versus concatenating all captions;
- better caption reconstruction/detail retention than mean pooling;
- no significant degradation versus a learned resampler;
- stable single-caption inference after adapter training.

These values are starting targets and may be revised after inspecting real data.

---

# 8. Logging and report output

Add a benchmark script that writes a Markdown and JSON report containing:

- exact git commit;
- tokenizer and vocabulary identifiers;
- embedding model and layer;
- span extraction policy;
- threshold configuration;
- number of images and captions;
- source token count;
- semantic span count;
- consensus-group count;
- unique/relation unit count;
- compression ratio;
- cluster-size distribution;
- support distribution;
- false-merge examples;
- missed-merge examples;
- contradiction cases;
- runtime and memory use.

Always include qualitative nearest-neighbor and cluster examples. Aggregate metrics alone will hide semantically destructive merges.

---

# 9. Implementation order

Follow this order strictly:

1. Add versioned superposition data structures.
2. Implement fixed contiguous grouping.
3. Add JSON serialization and CLI inspection.
4. Add C ABI and Python wrapper.
5. Add PyTorch embedding-plan example.
6. Run baseline token-superposition sanity test.
7. Add semantic-span exchange schema.
8. Add cosine similarity and role/entity constraints.
9. Add conservative complete-link clustering.
10. Add consensus/unique/relation/uncertainty units.
11. Build synthetic semantic test set.
12. Benchmark real i1 caption groups offline.
13. Only then train a tiny conditioning adapter or reduced model.
14. Do not integrate into the full MageFlow training run until the small-scale test passes.

---

# 10. Non-goals for the first implementation

Do not implement these initially:

- learned clustering inside Zig;
- Qwen inference inside `ztok`;
- image-grounding model inference inside `ztok`;
- synthetic vocabulary IDs for fused concepts;
- full scene-graph parsing;
- end-to-end differentiable hard clustering;
- approximate nearest-neighbor indexing;
- distributed clustering;
- direct integration with the production MageFlow trainer;
- replacement of ordinary tokenizer output.

The first objective is a correct, conservative, auditable representation and a small experiment that can falsify the idea cheaply.

---

# 11. Final architectural principle

Treat superposition as an **overlay over ordinary tokens and contextual spans**:

```text
ordinary tokenization
        +
offsets and provenance
        +
optional semantic grouping plan
        =
training-time superposition representation
```

The base tokenizer remains a deterministic text-to-ID system. Superposition is an optional higher-level plan describing how a model may combine already valid token or span representations.

For CCSS, optimize first for semantic correctness and detail preservation, then for compression. A missed merge costs a few extra conditioning tokens; a false merge can teach the image model that distinct concepts are interchangeable.