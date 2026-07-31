# NanoGPT Superposition Experiment: Transformer vs RWKV

## Purpose

Build a small, falsifiable experiment that tests whether token superposition improves language-model training and whether the effect differs between a Transformer and RWKV.

This experiment should be implemented only after the tokenizer-side plan machinery in `docs/TOKEN_SUPERPOSITION_CODEX.md` is usable.

The experiment has four progressively harder goals:

1. Reproduce fixed contiguous token superposition from `arXiv:2605.06546` at NanoGPT scale.
2. Test whether a model can consume sparse weighted token mixtures directly.
3. Test whether semantic clustering of likely tokens is safer than unconstrained soft-token recurrence.
4. Compare matched Transformer and RWKV students under the same data, tokenizer, parameter budget, effective context in bytes, and training FLOPs.

Do not begin with multilingual semantic graphs, anti-superpositions, or the full CCSS pipeline. Establish the simplest working superposition primitive first.

---

# 1. Central hypotheses

## H1: fixed token superposition can accelerate early training

During an initial coarse phase, several consecutive source tokens are represented by one superposed embedding and the model predicts a small ordered or unordered target group. A later recovery phase returns to ordinary next-token training.

Expected result:

- fewer sequence positions processed per source byte;
- better early progress per unit of wall-clock or FLOPs;
- recovery to ordinary-token generation without permanent quality loss.

## H2: semantically tight soft tokens can preserve continuation behavior

Given a next-token distribution, a sparse cluster of semantically and functionally similar candidates can be fused into one norm-preserving embedding and fed back to the model.

Expected result:

- continuation loss remains close to explicit branching for tight lexical alternatives;
- unrestricted averaging of top-k candidates performs worse;
- discrete fallback remains necessary for multimodal or contradictory distributions.

## H3: RWKV and Transformer may respond differently

A Transformer processes all positions through global attention. RWKV accumulates sequential state. A soft or superposed token may therefore affect later computation differently.

Possible outcomes:

- Transformer tolerates soft embeddings better because later attention can recontextualize them.
- RWKV benefits more because a compressed sequence reduces recurrent state updates.
- RWKV is more sensitive because one off-manifold soft token contaminates persistent state.
- both work, but require different normalization and recovery schedules.

The experiment must be designed to distinguish these outcomes.

---

# 2. Repository layout

Do not place the experimental model trainer inside the tokenizer hot path.

Add a self-contained experiment directory:

```text
experiments/nanogpt_superposition/
  README.md
  requirements.txt
  configs/
  data/
  models/
    transformer.py
    rwkv.py
    common.py
  superposition/
    fixed_groups.py
    soft_tokens.py
    semantic_clusters.py
    recovery.py
  train.py
  evaluate.py
  branch_equivalence.py
  prepare_data.py
  report.py
  tests/
```

The experiment may be Python/PyTorch. `ztok` should be consumed through its Python binding and plan APIs.

Do not modify existing ztok tokenizer semantics merely to simplify the trainer.

---

# 3. Dataset

Use two datasets so the result is not an artifact of one corpus.

## 3.1 Fast debugging corpus

Use Tiny Shakespeare or another tiny public corpus for plumbing only.

Purpose:

- verify training runs;
- test save/resume;
- test all loss functions;
- test recurrence and recovery behavior;
- debug the Transformer/RWKV parity harness.

Do not use this corpus for the final conclusion.

## 3.2 Main small-scale corpus

Use a compact but diverse corpus of approximately 100M to 500M raw tokens before superposition.

Preferred choices:

- FineWeb-Edu subset;
- SlimPajama subset;
- a fixed shard of FineWeb-2;
- another legally usable corpus already available locally.

The exact sample order must be deterministic and shared between all model types.

Record:

- source bytes;
- ordinary token count;
- bytes per token;
- document boundaries;
- language distribution if multilingual text is included;
- train/validation split hashes.

The first serious run should be primarily English. Add multilingual tests only after the baseline works.

---

# 4. Tokenizer conditions

Use one fixed ztok tokenizer vocabulary for all primary comparisons.

Recommended initial vocabulary sizes:

```text
16K or 32K for the smallest tests
50K for the main experiment
```

Run at least these tokenizer conditions:

## T0: ordinary BPE

Standard ztok tokenization.

## T1: SuperBPE surface tokenizer

Use ztok's existing SuperBPE support from `arXiv:2503.13423`.

This tests exact, lossless surface-level sequence compression without semantic mixtures.

Keep vocabulary size matched to T0.

## T2: ordinary BPE plus training-time fixed superposition

Tokenize normally, then apply the versioned superposition plan from `docs/TOKEN_SUPERPOSITION_CODEX.md`.

Test group sizes:

```text
2
4
8
```

Group size 16 may be added only if recovery works reliably.

## T3: SuperBPE plus training-time fixed superposition

This tests whether exact surface superwords and coarse training-time superposition are complementary.

Do not compare these conditions only by token count. Match training by source bytes and FLOPs.

---

# 5. Model matching

Implement a small Transformer and a small RWKV with comparable parameter count and training FLOPs.

Suggested first target:

```text
30M to 60M parameters
```

Then repeat the best configurations at:

```text
100M to 150M parameters
```

## 5.1 Transformer

Use a conventional NanoGPT-style decoder-only Transformer:

- pre-norm;
- RMSNorm preferred;
- SwiGLU or a matched MLP variant;
- RoPE or another fixed positional policy;
- causal attention;
- tied or untied embeddings recorded explicitly.

Keep it simple. Do not add MoE, looping, TREAD, or speculative heads.

## 5.2 RWKV

Use the same generation vocabulary, embedding width where practical, context in source bytes, and output head.

The RWKV implementation should expose:

- token-shift/time-mix state;
- channel-mix state;
- state reset at document boundaries;
- recurrent inference;
- parallel training mode if available.

Use an established RWKV block rather than inventing a new variant for this experiment.

## 5.3 Matching criteria

For each Transformer/RWKV pair, report:

- total parameters;
- embedding parameters;
- non-embedding parameters;
- trainable FLOPs per ordinary source token;
- measured forward/backward time;
- peak VRAM;
- effective context in bytes;
- optimizer and schedule;
- tokens or bytes seen.

Do not claim an architecture win from unmatched model widths or training budgets.

---

# 6. Baseline fixed-superposition objective

The paper's exact objective should be implemented as one condition, but also include an ordered visualizable target for easier debugging.

Given a group of source tokens:

```text
[x_t, x_{t+1}, ..., x_{t+s-1}]
```

construct one superposed input embedding:

```text
m_t = fuse(E[x_t], ..., E[x_{t+s-1}])
```

The model processes the shorter sequence of `m_t` values.

## 6.1 Input fusion variants

Test:

### Mean

```text
m = mean(E[x_i])
```

### Norm-preserving mean

```text
u = mean(E[x_i])
mean_norm = mean(||E[x_i]||)
m = normalize(u) * mean_norm
```

### Learned fixed-size fusion

A small permutation-aware or ordered fusion projection receives the grouped embeddings.

For example:

```text
concat(E[x_0] + P[0], ..., E[x_{s-1}] + P[s-1])
    -> small MLP or attention pooler
    -> one model-width vector
```

The learned fusion module must be included in the parameter and FLOP report.

Start with mean and norm-preserving mean.

## 6.2 Target variants

### Bag target

Predict all tokens in the next group using a multi-hot or multi-label objective.

This is closest to the paper.

### Ordered multi-head target

Use `s` output heads or one reshaped output projection to predict each ordered token position:

```text
head_0 -> token x_{t+s}
head_1 -> token x_{t+s+1}
...
```

This preserves order and may be easier to recover from.

### Autoregressive micro-decoder target

Use the coarse model state to seed a tiny local decoder that reconstructs the next group in order.

Do not begin with this variant. Add it only after the bag and ordered-head versions are stable.

## 6.3 Recovery phase

Every fixed-superposition run must include ordinary-token recovery.

Suggested schedule:

```text
Phase A: 50% of training FLOPs in superposition mode
Phase B: 25% mixed, with 50% superposed and 50% ordinary batches
Phase C: 25% ordinary next-token training only
```

Also test:

```text
75/15/10
30/30/40
```

Use source-byte or FLOP fractions, not optimizer-step fractions, because superposition changes work per step.

At the transition to recovery:

- preserve model weights;
- retain or disable auxiliary group heads as configured;
- do not reset optimizer unless separately ablated;
- record the immediate loss spike;
- record time and bytes required to recover baseline validation loss.

---

# 7. Soft-token recurrence experiment

This is separate from fixed contiguous superposition.

At a selected generation position, obtain logits:

```text
z = model_output(h_t)
p = softmax(z)
```

Select a sparse candidate set using:

- top-p mass;
- a hard maximum `k`;
- a logit-margin cutoff.

Recommended initial values:

```text
top_p = 0.90
max_k = 8
logit_margin = 5.0
```

## 7.1 Control conditions

Compare:

### S0: ordinary discrete feedback

Sample or argmax one token and feed its embedding.

### S1: unconstrained top-k weighted mean

Fuse all selected candidate embeddings.

This is expected to fail often and serves as a control.

### S2: similarity-clustered soft token

Cluster candidates by embedding distance and fuse only the highest-mass tight cluster.

### S3: transition-clustered soft token

For each candidate, evaluate a cheap one-step transition probe and require both embedding and transition similarity.

### S4: explicit small beam

Carry the top few discrete branches separately. This is expensive but serves as the reference approximation to the true mixture.

## 7.2 Cluster safety rule

A cluster may be fused only if:

- total cluster probability exceeds `min_cluster_mass`;
- maximum pairwise cosine distance is below threshold;
- no protected contradiction rule fires;
- optional transition dispersion is below threshold.

Start with:

```text
min_cluster_mass = 0.60
max_pair_cosine_distance = 0.08
```

These are placeholders and must be calibrated.

## 7.3 Feedback representation

A soft token should include more than the fused embedding.

Provide:

- norm-preserving fused embedding;
- a learned `<SUPERPOSED>` type embedding;
- entropy bucket embedding;
- cluster-mass bucket embedding;
- dispersion bucket embedding.

Conceptually:

```text
soft_input = fused_embedding
           + type_embedding
           + entropy_embedding
           + mass_embedding
           + dispersion_embedding
```

Do not expect a normally trained model to understand soft-token inputs without exposure.

## 7.4 Training exposure

Use scheduled soft-token corruption during ordinary teacher forcing.

At selected positions:

1. include the ground-truth token;
2. retrieve semantically nearby alternatives;
3. construct a sparse mixture dominated by the ground-truth token;
4. replace the ordinary input embedding with the soft token;
5. keep the continuation target unchanged.

Curriculum example:

```text
0-20% training: 0% soft-token positions
20-50%: 5%
50-80%: 10%
80-100%: 20%
```

Mixture example:

```text
true token: 0.70
near synonym 1: 0.20
near synonym 2: 0.10
```

Do not introduce model-generated soft-token recurrence until teacher-forced soft-token exposure is stable.

---

# 8. Semantic clustering spaces

Test several spaces. Do not assume raw input embeddings are sufficient.

## E0: input embedding table

Cheap baseline.

## E1: output embedding or unembedding vectors

Useful when embeddings are untied.

## E2: teacher contextual token representations

Use a frozen stronger model to precompute contextual representations for a controlled vocabulary or corpus sample.

## E3: learned semantic projection

Train a small projection so that candidates with equivalent future behavior are close.

## E4: one-step transition representation

For candidate token `i`:

```text
r_i = probe_transition(h_t, E[i])
```

Cluster by `r_i` or by a combination of `E[i]` and `r_i`.

The primary metric is not lexical synonym accuracy. It is continuation equivalence.

---

# 9. Branch-equivalence test

Implement `branch_equivalence.py` before long soft-token training.

For a context with candidate tokens `i` and probabilities `p_i`, compare:

```text
A = model_next_state(fuse(E[i], p_i))
B = sum_i p_i * model_next_state(E[i])
```

Because the model is nonlinear, these will differ.

Measure:

```text
state_error = ||A - B|| / ||B||
logit_KL = KL(softmax(logits_B) || softmax(logits_A))
continuation_loss_delta
```

Run this across:

- semantically tight synonym clusters;
- morphology variants;
- punctuation alternatives;
- antonyms;
- categorical alternatives;
- random top-k groups.

Expected result:

- tight clusters have lower state error;
- antonym/categorical groups have high error;
- transition-space clustering predicts safe fusion better than raw embedding distance.

Report results separately for Transformer and RWKV.

For RWKV, measure both:

- immediate hidden/output difference;
- recurrent state divergence over the next 1, 4, 16, and 64 tokens.

For the Transformer, measure divergence over the same future token horizon with the soft token retained in the KV history.

---

# 10. Optional positive/anti representation

Do not include this in the first training sweep. Add it only after positive-only soft tokens work.

Represent:

```text
positive = {bright: 0.55, glowing: 0.35, luminous: 0.10}
negative = {dark: 0.80, unlit: 0.20}
```

as separate channels:

```text
input = W_pos(fuse(positive))
      + W_neg(fuse(negative))
      + polarity_metadata
      + role_metadata
```

Do not subtract raw embeddings as the primary method.

Initial tests should use curated scalar or categorical oppositions:

- bright vs dark;
- open vs closed;
- increase vs decrease;
- before vs after;
- left vs right;
- present vs absent.

Evaluate antonym-flip rate and continuation stability.

---

# 11. Transformer vs RWKV considerations

## 11.1 Transformer

A superposed token becomes one position in the causal sequence and remains in the KV cache.

Potential strengths:

- later layers and later tokens can re-attend to it;
- uncertainty metadata remains addressable;
- local errors may be diluted by broader context.

Potential weaknesses:

- the soft token is an off-manifold KV entry;
- every later token may repeatedly attend to its ambiguity;
- position compression changes attention geometry.

Required logs:

- attention paid to superposed positions;
- layerwise representation norm;
- entropy change after superposed positions;
- persistence of branch-equivalence error.

## 11.2 RWKV

A superposed token updates the recurrent state once instead of multiple discrete token updates.

Potential strengths:

- fewer state transitions;
- direct sequence-length speedup;
- recurrent state may naturally compress semantic alternatives.

Potential weaknesses:

- one mixed update may irreversibly contaminate state;
- averaging several source tokens is not equivalent to processing them sequentially;
- fixed contiguous token superposition removes order-sensitive state updates.

Required logs:

- time-mix state norms;
- channel-mix state norms;
- state divergence from ordinary processing;
- recovery after 1, 4, 16, and 64 following tokens;
- whether learned fusion outperforms mean fusion more strongly than in Transformer.

## 11.3 Fairness rule

For fixed contiguous superposition, RWKV receives fewer recurrent steps and Transformer receives fewer attention positions. Report both theoretical and measured savings.

Do not equate a 4x shorter sequence with a 4x wall-clock speedup.

---

# 12. Training matrix

Start with this minimal matrix.

## Architecture

```text
Transformer
RWKV
```

## Tokenizer

```text
ordinary BPE
SuperBPE
```

## Training representation

```text
ordinary
fixed superposition group=2
fixed superposition group=4
```

## Fusion

```text
mean
norm-preserving mean
```

This is 24 configurations before seeds. Use one seed for screening, then repeat promising configurations with at least three seeds.

After screening, add:

```text
ordered multi-head target
bag target
learned fusion
soft-token exposure
semantic-clustered recurrence
```

Do not launch the complete Cartesian product initially.

---

# 13. Budget matching

Report results under three budgets.

## B1: equal optimizer steps

Useful for debugging only.

## B2: equal source bytes seen

Measures data efficiency.

## B3: equal measured training FLOPs or GPU-seconds

Measures compute efficiency.

The main conclusion should rely on B2 and B3, not B1.

For all conditions, record:

- source bytes processed;
- ordinary token equivalents;
- model positions processed;
- measured GPU-seconds;
- estimated FLOPs;
- validation loss in bits per byte;
- ordinary-token perplexity after recovery.

Bits per byte is essential because tokenizations differ.

---

# 14. Evaluation metrics

## 14.1 Core language metrics

- validation bits per byte;
- validation loss under ordinary tokens;
- perplexity only within the same tokenizer;
- compression ratio;
- bytes per internal model position;
- recovery loss curve;
- wall-clock to target bits per byte.

## 14.2 Generation metrics

Use fixed prompts and seeds.

Measure:

- grammaticality;
- repetition;
- coherence;
- exact string copying;
- punctuation stability;
- code-like text behavior;
- ability to return to ordinary generation after soft-token recurrence.

## 14.3 Superposition metrics

- branch-equivalence state error;
- next-logit KL;
- safe-fusion precision;
- unsafe-fusion rate;
- average cluster size;
- average probability mass fused;
- number of consecutive soft-token steps before divergence;
- discrete fallback rate;
- semantic preservation after recovery.

## 14.4 Architecture comparison

For Transformer and RWKV, compare:

- improvement over their own baseline;
- speedup at matched quality;
- quality at matched GPU-seconds;
- sensitivity to group size;
- sensitivity to soft-token rate;
- persistence of state error;
- recovery cost.

Do not compare raw perplexity across different tokenizers without bytes-per-token normalization.

---

# 15. Small semantic test set

Build a curated set of contexts whose next-token candidates fall into known classes.

## Tight lexical alternatives

```text
bright / brilliant / radiant
quick / fast / rapid
large / big / sizable
said / stated / remarked
```

## Morphological alternatives

```text
jump / jumps / jumped
run / running / ran
```

## Related but distinct

```text
bright / white / luminous
dog / animal / wolf
room / house / interior
```

## Antonyms or contradictions

```text
bright / dark
left / right
increase / decrease
open / closed
standing / seated
```

## Binding-sensitive contexts

```text
red cube beside blue sphere
blue cube beside red sphere
```

## Exact-sensitive tokens

```text
< vs <=
+ vs -
true vs false
identifier variants
numbers differing by one digit
```

This test set should be used for safe-fusion precision and branch-equivalence analysis.

---

# 16. Recovery and mixed-granularity generation

A successful model should not require superposed tokens at every position.

Support three inference modes:

## Ordinary mode

Always emit discrete tokens.

## Assisted mode

Use semantic fusion only when the safety criteria pass; otherwise emit a discrete token.

## Latent mode

Allow multiple consecutive soft tokens, then force periodic discrete checkpoints.

Test checkpoint intervals:

```text
1
2
4
8
16
```

A discrete checkpoint samples or selects a normal token and resets the soft-token chain.

This may prevent unbounded drift while retaining some compression.

---

# 17. Suggested first experiment

Use:

```text
Model size: approximately 40M parameters
Dataset: approximately 100M ordinary tokens
Context: matched to approximately 2K ordinary tokens or equal source bytes
Tokenizer: 32K BPE
Architectures: Transformer and RWKV
Seeds: one screening seed
```

Run:

```text
A. Ordinary baseline
B. Group-2 norm-preserving superposition + recovery
C. Group-4 norm-preserving superposition + recovery
D. SuperBPE baseline
E. SuperBPE + group-2 superposition + recovery
```

Primary success criterion:

```text
lower wall-clock or FLOPs to the same validation bits per byte after ordinary-token recovery
```

Secondary criterion:

```text
no meaningful degradation in ordinary discrete generation
```

Only after one of B/C/E succeeds should soft-token recurrence training begin.

---

# 18. Go/no-go gates

## Gate 1: fixed superposition plumbing

Proceed when:

- sequence shortening is real;
- targets align correctly;
- Transformer and RWKV both train without NaNs;
- recovery returns to coherent ordinary generation.

## Gate 2: compute benefit

Proceed when at least one architecture reaches a target validation bits-per-byte value with at least 15% lower measured GPU-seconds than its ordinary baseline.

A smaller gain may still be interesting, but does not justify immediate scaling.

## Gate 3: soft-token exposure

Proceed when teacher-forced soft tokens cause less than a small predefined validation-loss increase and do not destabilize recurrent state.

Suggested initial threshold:

```text
less than 2% validation bits-per-byte degradation at 10% soft-token positions
```

## Gate 4: semantic recurrence

Proceed when clustered soft tokens outperform unconstrained top-k averaging and approximate explicit branching substantially better.

## Gate 5: architecture conclusion

Do not declare Transformer or RWKV superior unless the result repeats across at least three seeds and one larger model size.

---

# 19. Required reports

Every run should write JSONL metrics and a Markdown summary.

Required summary table:

| Run | Architecture | Tokenizer | Superposition | Group | Fusion | Params | Source bytes | GPU-hours | Positions | Val bits/byte | Recovery cost |
|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|

Required plots:

- validation bits/byte vs GPU-seconds;
- validation bits/byte vs source bytes;
- recovery loss after switching to ordinary tokens;
- throughput vs group size;
- branch-equivalence error by candidate-cluster diameter;
- future divergence horizon for Transformer vs RWKV;
- safe-fusion precision/recall.

Include exact config files and git commit hashes.

---

# 20. Implementation cautions

1. **Do not use token ID distance as semantic distance.** IDs are addresses unless a compositional embedding scheme is explicitly implemented.
2. **Do not average the entire softmax.** Use sparse, tight clusters.
3. **Do not fuse antonyms or categorical alternatives.** Use discrete fallback or a later signed representation.
4. **Do not hide tokenizer differences behind perplexity.** Use bits per byte.
5. **Do not compare equal steps as the primary budget.** Use bytes and GPU-seconds.
6. **Do not assume RWKV state updates commute.** Processing four tokens sequentially is not equivalent to one averaged token.
7. **Do not omit recovery.** The original token-superposition method requires ordinary-token recovery for normal generation.
8. **Do not over-optimize compression.** The SuperBPE paper shows the best compression point is not necessarily the best downstream point.
9. **Do not start with a huge teacher model.** Validate the mechanics before precomputing expensive contextual embeddings.
10. **Keep all plans auditable.** Record source token IDs, weights, positions, cluster decisions, and fallback reasons.

---

# 21. Relationship to future CCSS and semantic-code work

This NanoGPT experiment is the prerequisite for later work.

If fixed and soft superposition succeed, the next stages are:

1. Replace raw embedding distance with teacher contextual span distance.
2. Use SuperBPE tokens as surface-span proposals.
3. Align multilingual paraphrases and translations.
4. Introduce role and relation tokens.
5. Train product-quantized semantic codes.
6. Distill a student on shorter semantic sequences.
7. Test positive and anti-superpositions.
8. Apply the same representation to image-caption conditioning.

Do not combine these stages until the base experiment shows that the model can consume and recover from superposed representations.

---

# 22. Final experimental question

The experiment should answer:

> Can a model learn more language per unit of compute by temporarily or selectively operating on denser token representations, and does a recurrent RWKV backbone exploit that density differently from a Transformer?

A positive result requires more than shorter sequences. It must show equal or better ordinary-language quality after recovery at lower measured compute, with safe soft-token behavior that can be predicted from semantic or transition-space similarity.
