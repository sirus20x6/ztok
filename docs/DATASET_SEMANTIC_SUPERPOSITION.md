# Dataset Mapping, Segmentation, and Semantic Fusion

This experiment implements a dataset-level semantic-superposition design. It is
independent of fixed contiguous token grouping: the algorithm first discovers
redundant structures across different examples, then fuses only corresponding
spans whose meanings, bindings, and grammatical/semantic relationships agree.

The unit of optimization is a mapped **superbatch**, not an arbitrary adjacent
token pair and not a set of likely next tokens.

```text
dataset examples
  -> retrieve structurally compatible examples
  -> word/phrase segmentation with exact tokenizer ranges
  -> contextual span embeddings
  -> grammatical and semantic relation graphs
  -> constrained cross-example mapping
  -> fuse mapped spans and retain unmatched spans
  -> auditable, causally ordered superbatch plan
```

The input schema is `ztok.dataset_semantic_spans.v1`; the output schema is
`ztok.dataset_superposition.v1`. The mapper never treats token IDs or randomly
initialized student embeddings as semantic coordinates.

## Conservative reference constraints

A candidate pair must come from different examples, exceed the contextual
cosine threshold, match grammatical roles plus known semantic values, match
known entity and binding types, and have the same incident relation signature.
Protected syntax and exact-sensitive spans can fuse only when their complete
source-token sequences are identical; semantic similarity alone can never fuse
them. Named-entity categories are also exact-token-only until an external entity
linker supplies stable identity IDs; matching `DATE` or `PERSON` labels are not
enough to fuse two values. Clusters use complete-link compatibility and contain
at most one span from each example.

The first causal-superbatch planner only packs pairs whose aligned cluster order
is monotonic in both examples. Unmatched spans remain unique units. This favors
precision and preserves causal order while the representation is validated.

## Mapping and fusion semantics

For examples `A` and `B`, segmentation produces span sequences plus relation
edges. A mapping `M = {(a_i, b_j)}` is accepted only if every aligned pair passes
the semantic and structural constraints and the accepted pairs preserve causal
order. The emitted sequence contains one fused unit for each pair in `M` and one
ordinary unit for every unmatched span in either example.

Thus two paraphrases may share an entity, action, and modifier while still
retaining a detail that occurs in only one source. A repeated surface word with
a different entity binding or dependency role remains separate. Compression is
a consequence of valid mappings; it is not the criterion used to create them.

## Example

```bash
python experiments/nanogpt_superposition/map_dataset.py \
  --input experiments/nanogpt_superposition/data/dataset_mapping_synthetic.json \
  --output /tmp/dataset_superposition_plan.json \
  --inspect
```

Every aligned unit records source example IDs, span IDs, indices, pair-specific
fusion weights, similarities, and its structural match reason. Contextual
vectors should come from a frozen teacher layer and documented pooling policy.

`contextualize_dataset.py` supplies the initial frozen-teacher adapter. It maps
auditable character spans through the teacher tokenizer, pools all overlapping
teacher pieces, and records the exact model revision and hidden-state index.

`retrieve_semantic_structural_pairs.py` retrieves nearest examples in a frozen
sentence-semantic space and then applies an exact structural-compatibility gate.
It labels exact duplicates separately so they cannot be presented as evidence
that non-identical semantic mapping works. Only the selected examples proceed to
the more expensive span-level contextualization and complete-link mapper.
Reports distinguish packed-pair compression, selected-candidate compression
with ordinary fallback, and full-source-dataset compression including examples
that retrieval did not select.

## ProofWriter logical-trace ceiling

`benchmark_proofwriter_superposition.py` builds proof-level primer groups from
the explicit facts, rules, questions, answers, and used proof chains in
ProofWriter. Its `role_safe` mode keeps predicate identities exact and only
alpha-renames entity/variable roles. Its separate
`logical_isomorphism_ceiling` consistently alpha-renames predicates too; that
mode measures a symbolic upper bound and must not be described as lexical
semantic equivalence. Unknown examples are excluded from proof priming because
they provide no positive derivation trace, but remain ordinary training rows.

`build_proofwriter_statement_expansions.py` implements the more direct primer
construction. Within one ProofWriter world, facts, rules, or questions with the
same truth class and identical structure except for one logical slot become a
versioned superposition expansion. Only one slot is expanded at a time, avoiding
false Cartesian products between independently fused subjects and predicates.
The original ordinary row remains the grounding example immediately following
the generated primer.

These alternatives have set-valued semantics. For example,
`Gary is [blue|red|rough]` represents several valid propositions; it does not
claim that the predicates are synonyms. A training adapter must therefore match
the fused transition to the weighted explicit-branch transitions and/or
reconstruct the alternative set. Applying ordinary next-token loss as if the
alternatives were interchangeable would implement a different, unsafe
objective.

The mapper only emits a plan. A trainer adapter must still define how aligned
span embeddings, per-example targets, unique units, and relations are presented
to the student. GPU training must wait until synthetic precision and real-corpus
alignment coverage are measured.
