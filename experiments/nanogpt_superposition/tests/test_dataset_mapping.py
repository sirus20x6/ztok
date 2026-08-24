import torch
from superposition.dataset_mapping import (
    DatasetMappingConfig,
    RelationEdge,
    SpanNode,
    build_dataset_superposition_plan,
)


def _node(
    example: int,
    span_id: str,
    position: int,
    embedding: int,
    grammar: str,
    semantics: str,
    *,
    entity: str | None = None,
    binding: str | None = None,
    value: str | None = None,
    protected: bool = False,
    token_ids: tuple[int, ...] | None = None,
) -> SpanNode:
    return SpanNode(
        example_id=example,
        span_id=span_id,
        token_start=position,
        token_end=position + 1,
        embedding_index=embedding,
        grammatical_role=grammar,
        semantic_role=semantics,
        semantic_value=value,
        entity_type=entity,
        binding_type=binding,
        source_token_ids=token_ids,
        protected=protected,
    )


def test_maps_semantically_and_structurally_equivalent_examples() -> None:
    spans = [
        _node(0, "bright", 0, 0, "amod", "intensity", binding="animal"),
        _node(0, "dog", 1, 1, "nsubj", "entity", entity="animal"),
        _node(0, "runs", 2, 2, "root", "action"),
        _node(1, "radiant", 0, 3, "amod", "intensity", binding="animal"),
        _node(1, "canine", 1, 4, "nsubj", "entity", entity="animal"),
        _node(1, "sprints", 2, 5, "root", "action"),
    ]
    edges = [
        RelationEdge(0, "bright", "modifies", "dog"),
        RelationEdge(0, "dog", "agent_of", "runs"),
        RelationEdge(1, "radiant", "modifies", "canine"),
        RelationEdge(1, "canine", "agent_of", "sprints"),
    ]
    embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.99, 0.01, 0.0],
            [0.01, 0.99, 0.0],
            [0.0, 0.01, 0.99],
        ]
    )
    plan = build_dataset_superposition_plan(spans, edges, embeddings)

    assert plan["schema"] == "ztok.dataset_superposition.v1"
    assert len(plan["clusters"]) == 3
    assert len(plan["superbatches"]) == 1
    superbatch = plan["superbatches"][0]
    assert superbatch["example_ids"] == [0, 1]
    assert superbatch["aligned_cluster_ids"] == [0, 1, 2]
    assert superbatch["unique_span_ids"] == {"0": [], "1": []}
    assert superbatch["compression_ratio"] == 0.5
    assert superbatch["alignment_coverage"] == 1.0
    assert superbatch["total_alignment_coverage"] == 1.0
    assert plan["execution"] == {
        "packed_example_ids": [0, 1],
        "ordinary_example_ids": [],
        "ordinary_span_count": 0,
        "retained_output_unit_count": 3,
        "dataset_compression_ratio": 0.5,
    }
    assert [
        member["weight"] for member in superbatch["aligned_units"][0]["members"]
    ] == [0.5, 0.5]


def test_blocks_same_word_when_entity_binding_differs() -> None:
    spans = [
        _node(0, "bright-room", 0, 0, "amod", "intensity", binding="environment"),
        _node(1, "bright-dress", 0, 1, "amod", "intensity", binding="clothing"),
    ]
    plan = build_dataset_superposition_plan(
        spans,
        [],
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        DatasetMappingConfig(verbose_diagnostics=True),
    )

    assert plan["clusters"] == []
    assert plan["diagnostics"]["rejected_pairs"][0]["reason"] == "binding_type_mismatch"


def test_blocks_contradictory_semantic_values_even_with_identical_vectors() -> None:
    spans = [
        _node(0, "bright", 0, 0, "amod", "intensity", value="high"),
        _node(1, "dark", 0, 1, "amod", "intensity", value="low"),
    ]
    plan = build_dataset_superposition_plan(
        spans,
        [],
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        DatasetMappingConfig(verbose_diagnostics=True),
    )

    assert plan["clusters"] == []
    assert (
        plan["diagnostics"]["rejected_pairs"][0]["reason"] == "semantic_value_mismatch"
    )


def test_named_entity_types_require_exact_source_tokens_without_entity_linking() -> (
    None
):
    spans = [
        _node(0, "may", 0, 0, "pobj", "entity", entity="DATE", token_ids=(10,)),
        _node(1, "april", 0, 1, "pobj", "entity", entity="DATE", token_ids=(11,)),
    ]
    plan = build_dataset_superposition_plan(
        spans,
        [],
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        DatasetMappingConfig(verbose_diagnostics=True, minimum_aligned_spans=1),
    )

    assert plan["clusters"] == []
    assert (
        plan["diagnostics"]["rejected_pairs"][0]["reason"]
        == "protected_entity_identity"
    )


def test_complete_link_blocks_semantic_chaining() -> None:
    spans = [
        _node(0, "a", 0, 0, "amod", "intensity"),
        _node(1, "b", 0, 1, "amod", "intensity"),
        _node(2, "c", 0, 2, "amod", "intensity"),
    ]
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.94, 0.341],
            [0.766, 0.643],
        ]
    )
    plan = build_dataset_superposition_plan(
        spans,
        [],
        embeddings,
        DatasetMappingConfig(cosine_threshold=0.9, minimum_aligned_spans=1),
    )

    assert max(len(cluster["member_span_ids"]) for cluster in plan["clusters"]) == 2


def test_non_monotonic_alignment_is_not_packed_as_causal_superbatch() -> None:
    spans = [
        _node(0, "red-0", 0, 0, "amod", "color"),
        _node(0, "blue-0", 1, 1, "amod", "color"),
        _node(1, "blue-1", 0, 2, "amod", "color"),
        _node(1, "red-1", 1, 3, "amod", "color"),
    ]
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    plan = build_dataset_superposition_plan(
        spans,
        [],
        embeddings,
        DatasetMappingConfig(minimum_aligned_spans=2),
    )

    assert len(plan["clusters"]) == 2
    assert plan["superbatches"] == []


def test_pair_specific_unmatched_cluster_member_is_retained_as_unique() -> None:
    spans = [
        _node(0, "a-0", 0, 0, "nsubj", "entity"),
        _node(0, "b-0", 1, 1, "root", "action"),
        _node(1, "a-1", 0, 2, "nsubj", "entity"),
        _node(2, "b-2", 0, 3, "root", "action"),
    ]
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    plan = build_dataset_superposition_plan(
        spans,
        [],
        embeddings,
        DatasetMappingConfig(
            minimum_aligned_spans=1,
            minimum_alignment_coverage=0.5,
        ),
    )

    assert plan["superbatches"][0]["example_ids"] == [0, 1]
    assert plan["superbatches"][0]["unique_span_ids"]["0"] == ["b-0"]


def test_retrieval_pairs_bound_candidate_generation() -> None:
    spans = [
        _node(0, "x-0", 0, 0, "nsubj", "entity"),
        _node(1, "x-1", 0, 1, "nsubj", "entity"),
        _node(2, "x-2", 0, 2, "nsubj", "entity"),
    ]
    plan = build_dataset_superposition_plan(
        spans,
        [],
        torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        DatasetMappingConfig(minimum_aligned_spans=1),
        allowed_example_pairs={(0, 1)},
    )

    assert plan["diagnostics"]["candidate_pair_count"] == 1
    assert plan["clusters"][0]["member_example_ids"] == [0, 1]
    assert plan["execution"]["ordinary_example_ids"] == [2]
    assert plan["execution"]["ordinary_span_count"] == 1
    assert plan["execution"]["dataset_compression_ratio"] == 2 / 3


def test_protected_spans_fuse_only_when_source_token_ids_are_exact() -> None:
    spans = [
        _node(0, "the-0", 0, 0, "det", "syntax", protected=True, token_ids=(7,)),
        _node(1, "the-1", 0, 1, "det", "syntax", protected=True, token_ids=(7,)),
        _node(2, "a-2", 0, 2, "det", "syntax", protected=True, token_ids=(8,)),
    ]
    plan = build_dataset_superposition_plan(
        spans,
        [],
        torch.zeros(3, 2),
        DatasetMappingConfig(
            minimum_aligned_spans=1,
            require_semantic_role=False,
        ),
    )

    assert len(plan["clusters"]) == 1
    assert plan["clusters"][0]["member_span_ids"] == ["the-0", "the-1"]
    assert plan["clusters"][0]["mean_pair_similarity"] == 1.0


def test_protected_alignment_does_not_inflate_content_coverage() -> None:
    spans = [
        _node(0, "dog-0", 0, 0, "nsubj", "entity"),
        _node(0, "the-0", 1, 1, "det", "syntax", protected=True, token_ids=(7,)),
        _node(1, "dog-1", 0, 2, "nsubj", "entity"),
        _node(1, "the-1", 1, 3, "det", "syntax", protected=True, token_ids=(7,)),
    ]
    plan = build_dataset_superposition_plan(
        spans,
        [],
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
        DatasetMappingConfig(minimum_aligned_spans=1),
    )

    superbatch = plan["superbatches"][0]
    assert superbatch["alignment_coverage"] == 1.0
    assert superbatch["total_alignment_coverage"] == 1.0
