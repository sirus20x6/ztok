"""Dataset-level semantic/grammatical mapping for auditable superbatches."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any

import torch
import torch.nn.functional as F

SCHEMA = "ztok.dataset_superposition.v1"
PROTECTED_ENTITY_TYPES = frozenset(
    {
        "CARDINAL",
        "DATE",
        "EVENT",
        "FAC",
        "GPE",
        "LANGUAGE",
        "LAW",
        "LOC",
        "MONEY",
        "NORP",
        "ORDINAL",
        "ORG",
        "PERCENT",
        "PERSON",
        "PRODUCT",
        "QUANTITY",
        "TIME",
        "WORK_OF_ART",
    }
)


@dataclass(frozen=True)
class SpanNode:
    example_id: int
    span_id: str
    token_start: int
    token_end: int
    embedding_index: int
    grammatical_role: str | None = None
    semantic_role: str | None = None
    semantic_value: str | None = None
    entity_type: str | None = None
    binding_type: str | None = None
    source_token_ids: tuple[int, ...] | None = None
    confidence: float = 1.0
    protected: bool = False


@dataclass(frozen=True)
class RelationEdge:
    example_id: int
    source_span_id: str
    relation: str
    target_span_id: str


@dataclass(frozen=True)
class DatasetMappingConfig:
    cosine_threshold: float = 0.92
    maximum_cluster_size: int = 8
    minimum_aligned_spans: int = 2
    minimum_alignment_coverage: float = 0.75
    require_grammatical_role: bool = True
    require_semantic_role: bool = True
    require_relation_signature: bool = True
    verbose_diagnostics: bool = False


@dataclass(frozen=True)
class PairDecision:
    left_span_index: int
    right_span_index: int
    cosine_similarity: float
    accepted: bool
    reason: str


def _known_equal(left: str | None, right: str | None, *, required: bool) -> bool:
    if required and (left is None or right is None):
        return False
    return left is None or right is None or left == right


def _validate(
    spans: list[SpanNode], edges: list[RelationEdge], embeddings: torch.Tensor
) -> dict[tuple[int, str], int]:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a [span, dimension] matrix")
    if not spans:
        raise ValueError("at least one span is required")
    identities: dict[tuple[int, str], int] = {}
    by_example: dict[int, list[SpanNode]] = {}
    for index, span in enumerate(spans):
        identity = (span.example_id, span.span_id)
        if identity in identities:
            raise ValueError(f"duplicate span identity: {identity}")
        if span.token_end <= span.token_start:
            raise ValueError(f"empty or reversed token span: {identity}")
        if not 0 <= span.embedding_index < embeddings.shape[0]:
            raise ValueError(f"embedding index out of range: {identity}")
        if not 0.0 <= span.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1]: {identity}")
        identities[identity] = index
        by_example.setdefault(span.example_id, []).append(span)
    for example_spans in by_example.values():
        ordered = sorted(
            example_spans, key=lambda span: (span.token_start, span.token_end)
        )
        for left, right in pairwise(ordered):
            if left.token_end > right.token_start:
                raise ValueError(
                    f"overlapping spans in example {left.example_id}: "
                    f"{left.span_id}, {right.span_id}"
                )
    for edge in edges:
        source = (edge.example_id, edge.source_span_id)
        target = (edge.example_id, edge.target_span_id)
        if source not in identities or target not in identities:
            raise ValueError(f"relation references unknown span: {edge}")
    return identities


def _relation_signatures(
    spans: list[SpanNode],
    edges: list[RelationEdge],
    identities: dict[tuple[int, str], int],
) -> list[tuple[tuple[str, ...], ...]]:
    signatures: list[list[tuple[str, ...]]] = [[] for _ in spans]
    for edge in edges:
        source_index = identities[(edge.example_id, edge.source_span_id)]
        target_index = identities[(edge.example_id, edge.target_span_id)]
        source = spans[source_index]
        target = spans[target_index]
        signatures[source_index].append(
            (
                "out",
                edge.relation,
                target.grammatical_role or "",
                target.semantic_role or "",
                target.semantic_value or "",
                target.entity_type or "",
                target.binding_type or "",
            )
        )
        signatures[target_index].append(
            (
                "in",
                edge.relation,
                source.grammatical_role or "",
                source.semantic_role or "",
                source.semantic_value or "",
                source.entity_type or "",
                source.binding_type or "",
            )
        )
    return [tuple(sorted(signature)) for signature in signatures]


def _pair_decision(
    left_index: int,
    right_index: int,
    spans: list[SpanNode],
    normalized_embeddings: torch.Tensor,
    signatures: list[tuple[tuple[str, ...], ...]],
    config: DatasetMappingConfig,
    allowed_example_pairs: set[tuple[int, int]] | None,
) -> PairDecision:
    left = spans[left_index]
    right = spans[right_index]
    protected_exact = (
        left.protected
        and right.protected
        and left.source_token_ids is not None
        and left.source_token_ids == right.source_token_ids
    )
    exact_source = (
        left.source_token_ids is not None
        and right.source_token_ids is not None
        and left.source_token_ids == right.source_token_ids
    )
    identity_sensitive_entity = (
        left.entity_type in PROTECTED_ENTITY_TYPES
        or right.entity_type in PROTECTED_ENTITY_TYPES
    )
    similarity = (
        1.0
        if protected_exact
        else float(
            torch.dot(
                normalized_embeddings[left.embedding_index],
                normalized_embeddings[right.embedding_index],
            )
        )
    )
    reason = "compatible"
    if left.example_id == right.example_id:
        reason = "same_example"
    elif (
        allowed_example_pairs is not None
        and (
            min(left.example_id, right.example_id),
            max(left.example_id, right.example_id),
        )
        not in allowed_example_pairs
    ):
        reason = "retrieval_pair_blocked"
    elif (left.protected or right.protected) and not protected_exact:
        reason = "protected_span"
    elif identity_sensitive_entity and not exact_source:
        reason = "protected_entity_identity"
    elif not _known_equal(
        left.grammatical_role,
        right.grammatical_role,
        required=config.require_grammatical_role,
    ):
        reason = "grammatical_role_mismatch"
    elif not _known_equal(
        left.semantic_role,
        right.semantic_role,
        required=config.require_semantic_role,
    ):
        reason = "semantic_role_mismatch"
    elif not _known_equal(left.semantic_value, right.semantic_value, required=False):
        reason = "semantic_value_mismatch"
    elif not _known_equal(left.entity_type, right.entity_type, required=False):
        reason = "entity_type_mismatch"
    elif not _known_equal(left.binding_type, right.binding_type, required=False):
        reason = "binding_type_mismatch"
    elif (
        config.require_relation_signature
        and signatures[left_index] != signatures[right_index]
    ):
        reason = "relation_signature_mismatch"
    elif similarity < config.cosine_threshold:
        reason = "cosine_below_threshold"
    return PairDecision(
        left_span_index=left_index,
        right_span_index=right_index,
        cosine_similarity=similarity,
        accepted=reason == "compatible",
        reason=reason,
    )


def _cluster_spans(
    spans: list[SpanNode],
    decisions: list[PairDecision],
    config: DatasetMappingConfig,
) -> list[list[int]]:
    accepted = {
        (decision.left_span_index, decision.right_span_index): decision.accepted
        for decision in decisions
    }
    clusters = [{index} for index in range(len(spans))]

    def pair_passes(left: int, right: int) -> bool:
        key = (min(left, right), max(left, right))
        return accepted.get(key, False)

    for decision in sorted(
        (decision for decision in decisions if decision.accepted),
        key=lambda item: (
            -item.cosine_similarity,
            item.left_span_index,
            item.right_span_index,
        ),
    ):
        left_cluster = next(
            cluster for cluster in clusters if decision.left_span_index in cluster
        )
        right_cluster = next(
            cluster for cluster in clusters if decision.right_span_index in cluster
        )
        if left_cluster is right_cluster:
            continue
        proposed = left_cluster | right_cluster
        if len(proposed) > config.maximum_cluster_size:
            continue
        example_ids = [spans[index].example_id for index in proposed]
        if len(example_ids) != len(set(example_ids)):
            continue
        if not all(
            pair_passes(left, right) for left in left_cluster for right in right_cluster
        ):
            continue
        left_cluster.update(right_cluster)
        clusters.remove(right_cluster)
    return [sorted(cluster) for cluster in clusters if len(cluster) > 1]


def _normalized_weights(indices: list[int], spans: list[SpanNode]) -> list[float]:
    values = [spans[index].confidence for index in indices]
    total = sum(values)
    if total <= 0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in values]


def _build_superbatches(
    spans: list[SpanNode],
    clusters: list[list[int]],
    config: DatasetMappingConfig,
) -> list[dict[str, Any]]:
    span_to_cluster = {
        span_index: cluster_index
        for cluster_index, cluster in enumerate(clusters)
        for span_index in cluster
    }
    by_example: dict[int, list[int]] = {}
    for index, span in enumerate(spans):
        by_example.setdefault(span.example_id, []).append(index)
    for indices in by_example.values():
        indices.sort(
            key=lambda index: (spans[index].token_start, spans[index].token_end)
        )

    candidates: list[
        tuple[float, int, int, list[int], list[int], list[int], int, int]
    ] = []
    example_ids = sorted(by_example)
    for offset, left_example in enumerate(example_ids):
        for right_example in example_ids[offset + 1 :]:
            left_indices = by_example[left_example]
            right_indices = by_example[right_example]
            left_order = [
                span_to_cluster[index]
                for index in left_indices
                if index in span_to_cluster
                and any(
                    spans[member].example_id == right_example
                    for member in clusters[span_to_cluster[index]]
                )
            ]
            right_order = [
                span_to_cluster[index]
                for index in right_indices
                if index in span_to_cluster
                and any(
                    spans[member].example_id == left_example
                    for member in clusters[span_to_cluster[index]]
                )
            ]
            if left_order != right_order:
                continue
            aligned = left_order
            aligned_content = [
                cluster_id
                for cluster_id in aligned
                if any(
                    spans[member].example_id in {left_example, right_example}
                    and not spans[member].protected
                    for member in clusters[cluster_id]
                )
            ]
            left_eligible = sum(not spans[index].protected for index in left_indices)
            right_eligible = sum(not spans[index].protected for index in right_indices)
            eligible_denominator = max(left_eligible, right_eligible)
            coverage = (
                len(aligned_content) / eligible_denominator
                if eligible_denominator
                else 0.0
            )
            if (
                len(aligned) < config.minimum_aligned_spans
                or coverage < config.minimum_alignment_coverage
            ):
                continue
            aligned_set = set(aligned)
            left_unique = [
                index
                for index in left_indices
                if span_to_cluster.get(index) not in aligned_set
            ]
            right_unique = [
                index
                for index in right_indices
                if span_to_cluster.get(index) not in aligned_set
            ]
            candidates.append(
                (
                    coverage,
                    left_example,
                    right_example,
                    aligned,
                    left_unique,
                    right_unique,
                    left_eligible,
                    right_eligible,
                )
            )

    used: set[int] = set()
    superbatches = []
    for (
        coverage,
        left,
        right,
        aligned,
        left_unique,
        right_unique,
        left_eligible,
        right_eligible,
    ) in sorted(
        candidates, key=lambda item: (-item[0], -len(item[3]), item[1], item[2])
    ):
        if left in used or right in used:
            continue
        used.update((left, right))
        original_count = len(by_example[left]) + len(by_example[right])
        output_count = len(aligned) + len(left_unique) + len(right_unique)
        total_alignment_coverage = len(aligned) / max(
            len(by_example[left]), len(by_example[right])
        )
        aligned_units = []
        for cluster_id in aligned:
            members = [
                index
                for index in clusters[cluster_id]
                if spans[index].example_id in {left, right}
            ]
            aligned_units.append(
                {
                    "cluster_id": cluster_id,
                    "members": [
                        {
                            "example_id": spans[index].example_id,
                            "span_id": spans[index].span_id,
                            "span_index": index,
                            "weight": weight,
                        }
                        for index, weight in zip(
                            members, _normalized_weights(members, spans), strict=True
                        )
                    ],
                }
            )
        superbatches.append(
            {
                "superbatch_id": len(superbatches),
                "example_ids": [left, right],
                "aligned_cluster_ids": aligned,
                "aligned_units": aligned_units,
                "unique_span_ids": {
                    str(left): [spans[index].span_id for index in left_unique],
                    str(right): [spans[index].span_id for index in right_unique],
                },
                "alignment_coverage": coverage,
                "total_alignment_coverage": total_alignment_coverage,
                "eligible_content_span_count": {
                    str(left): left_eligible,
                    str(right): right_eligible,
                },
                "original_span_count": original_count,
                "output_unit_count": output_count,
                "compression_ratio": output_count / original_count,
            }
        )
    return superbatches


def build_dataset_superposition_plan(
    spans: list[SpanNode],
    edges: list[RelationEdge],
    embeddings: torch.Tensor,
    config: DatasetMappingConfig | None = None,
    *,
    allowed_example_pairs: set[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Build a deterministic mapping plan without calculating model embeddings."""
    config = config or DatasetMappingConfig()
    if not 0.0 <= config.cosine_threshold <= 1.0:
        raise ValueError("cosine_threshold must be in [0, 1]")
    if not 0.0 <= config.minimum_alignment_coverage <= 1.0:
        raise ValueError("minimum_alignment_coverage must be in [0, 1]")
    identities = _validate(spans, edges, embeddings)
    signatures = _relation_signatures(spans, edges, identities)
    normalized = F.normalize(embeddings.float(), dim=-1)
    if allowed_example_pairs is None:
        candidate_indices = (
            (left, right)
            for left in range(len(spans))
            for right in range(left + 1, len(spans))
        )
    else:
        by_example: dict[int, list[int]] = {}
        for index, span in enumerate(spans):
            by_example.setdefault(span.example_id, []).append(index)
        candidate_indices = (
            (left, right)
            for left_example, right_example in sorted(allowed_example_pairs)
            for left in by_example.get(left_example, [])
            for right in by_example.get(right_example, [])
        )
    decisions = [
        _pair_decision(
            min(left, right),
            max(left, right),
            spans,
            normalized,
            signatures,
            config,
            allowed_example_pairs,
        )
        for left, right in candidate_indices
    ]
    clusters = _cluster_spans(spans, decisions, config)
    decisions_by_pair = {
        (decision.left_span_index, decision.right_span_index): decision
        for decision in decisions
    }
    cluster_payload = []
    for cluster_id, indices in enumerate(clusters):
        similarities = [
            decisions_by_pair[(min(left, right), max(left, right))].cosine_similarity
            for offset, left in enumerate(indices)
            for right in indices[offset + 1 :]
        ]
        cluster_payload.append(
            {
                "cluster_id": cluster_id,
                "member_span_ids": [spans[index].span_id for index in indices],
                "member_example_ids": [spans[index].example_id for index in indices],
                "member_span_indices": indices,
                "members": [
                    {
                        "example_id": spans[index].example_id,
                        "span_id": spans[index].span_id,
                        "span_index": index,
                    }
                    for index in indices
                ],
                "weights": _normalized_weights(indices, spans),
                "minimum_pair_similarity": min(similarities),
                "mean_pair_similarity": sum(similarities) / len(similarities),
                "fusion": "norm_preserving_mean",
                "reason": "semantic_similarity_and_relation_signature",
            }
        )
    clustered = {index for cluster in clusters for index in cluster}
    superbatches = _build_superbatches(spans, clusters, config)
    packed_example_ids = {
        example_id
        for superbatch in superbatches
        for example_id in superbatch["example_ids"]
    }
    all_example_ids = {span.example_id for span in spans}
    ordinary_example_ids = sorted(all_example_ids - packed_example_ids)
    ordinary_span_count = sum(span.example_id in ordinary_example_ids for span in spans)
    retained_output_unit_count = ordinary_span_count + sum(
        superbatch["output_unit_count"] for superbatch in superbatches
    )
    plan: dict[str, Any] = {
        "schema": SCHEMA,
        "example_count": len(all_example_ids),
        "span_count": len(spans),
        "clusters": cluster_payload,
        "superbatches": superbatches,
        "execution": {
            "packed_example_ids": sorted(packed_example_ids),
            "ordinary_example_ids": ordinary_example_ids,
            "ordinary_span_count": ordinary_span_count,
            "retained_output_unit_count": retained_output_unit_count,
            "dataset_compression_ratio": retained_output_unit_count / len(spans),
        },
        "unique_spans": [
            {
                "example_id": spans[index].example_id,
                "span_id": spans[index].span_id,
                "span_index": index,
            }
            for index in range(len(spans))
            if index not in clustered
        ],
        "diagnostics": {
            "retrieval_pair_count": (
                len(allowed_example_pairs)
                if allowed_example_pairs is not None
                else None
            ),
            "candidate_pair_count": len(decisions),
            "accepted_pair_count": sum(decision.accepted for decision in decisions),
            "rejected_pairs": (
                [asdict(decision) for decision in decisions if not decision.accepted]
                if config.verbose_diagnostics
                else []
            ),
        },
    }
    return plan
