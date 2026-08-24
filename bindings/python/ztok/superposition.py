"""Experimental, opt-in superposition plans.

These APIs build descriptors over ordinary token IDs. They never introduce
synthetic vocabulary entries or change :class:`ztok.Pipeline` encoding.
"""

from __future__ import annotations

import ctypes
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, Union

from . import _get_lib, _raise_for_status
from ._ffi import (
    SUPERPOSITION_FUSION_MEAN,
    SUPERPOSITION_FUSION_NORM_PRESERVING_MEAN,
    SUPERPOSITION_FUSION_WEIGHTED_MEAN,
    ZTOK_ERR_BUFFER_TOO_SMALL,
    ZtokSuperpositionFixedConfig,
)

if TYPE_CHECKING:
    from . import Pipeline


SCHEMA = "ztok.superposition.v1"
SCHEMA_VERSION = 1


class FusionKind(str, Enum):
    MEAN = "mean"
    WEIGHTED_MEAN = "weighted_mean"
    NORM_PRESERVING_MEAN = "norm_preserving_mean"


_FUSION_TO_C = {
    FusionKind.MEAN: SUPERPOSITION_FUSION_MEAN,
    FusionKind.WEIGHTED_MEAN: SUPERPOSITION_FUSION_WEIGHTED_MEAN,
    FusionKind.NORM_PRESERVING_MEAN:
        SUPERPOSITION_FUSION_NORM_PRESERVING_MEAN,
}

_GROUP_KIND_NAMES = {
    0: "fixed_window",
    1: "partial_window",
    2: "preserved_special",
    3: "preserved_boundary",
    4: "uncovered_tail",
}


@dataclass(frozen=True)
class FixedSuperpositionConfig:
    group_size: int = 4
    stride: Optional[int] = None
    preserve_special_tokens: bool = True
    preserve_boundary_tokens: bool = True
    allow_partial_final_group: bool = True
    fusion: FusionKind = FusionKind.NORM_PRESERVING_MEAN

    def _as_c(self) -> ZtokSuperpositionFixedConfig:
        fusion = (
            self.fusion
            if isinstance(self.fusion, FusionKind)
            else FusionKind(self.fusion)
        )
        if not 1 <= self.group_size <= 0xFFFF:
            raise ValueError("group_size must be in [1, 65535]")
        if self.stride is not None and not 1 <= self.stride <= self.group_size:
            raise ValueError("stride must be in [1, group_size]")
        return ZtokSuperpositionFixedConfig(
            group_size=self.group_size,
            stride=self.stride or 0,
            fusion=_FUSION_TO_C[fusion],
            preserve_special_tokens=int(self.preserve_special_tokens),
            preserve_boundary_tokens=int(self.preserve_boundary_tokens),
            allow_partial_final_group=int(self.allow_partial_final_group),
        )


@dataclass(frozen=True)
class SourceToken:
    token_index: int
    token_id: int
    byte_start: int
    byte_end: int
    weight: float


@dataclass(frozen=True)
class SuperpositionGroup:
    output_index: int
    sources: tuple[SourceToken, ...]
    fusion: FusionKind
    kind: str
    position_start: int
    position_end: int
    byte_start: int
    byte_end: int
    center_position: float
    normalized_center: float


@dataclass(frozen=True)
class SuperpositionPlan:
    original_ids: tuple[int, ...]
    original_offsets: tuple[tuple[int, int], ...]
    groups: tuple[SuperpositionGroup, ...]
    schema: str = SCHEMA
    schema_version: int = SCHEMA_VERSION
    _json: str = field(default="", repr=False, compare=False)

    @property
    def original_token_count(self) -> int:
        return len(self.original_ids)

    @property
    def output_token_count(self) -> int:
        return len(self.groups)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical schema object emitted by the Zig core."""

        return json.loads(self._json)

    def to_json(self) -> str:
        """Return deterministic ``ztok.superposition.v1`` JSON."""

        return self._json


def _materialize_plan(lib: ctypes.CDLL, handle: int) -> SuperpositionPlan:
    n_tokens = int(lib.ztok_superposition_plan_original_token_count(handle))
    n_groups = int(lib.ztok_superposition_plan_output_token_count(handle))
    n_sources = int(lib.ztok_superposition_plan_source_count(handle))

    ids_ptr = lib.ztok_superposition_plan_original_ids(handle)
    ids = tuple(int(ids_ptr[i]) for i in range(n_tokens)) if n_tokens else ()

    offsets: list[tuple[int, int]] = []
    for index in range(n_tokens):
        start = ctypes.c_uint32()
        end = ctypes.c_uint32()
        rc = lib.ztok_superposition_plan_original_offset(
            handle, index, ctypes.byref(start), ctypes.byref(end)
        )
        _raise_for_status(rc, "ztok_superposition_plan_original_offset")
        offsets.append((start.value, end.value))

    sources_ptr = lib.ztok_superposition_plan_sources(handle)
    sources = [
        SourceToken(
            token_index=int(sources_ptr[i].token_index),
            token_id=int(sources_ptr[i].token_id),
            byte_start=int(sources_ptr[i].byte_start),
            byte_end=int(sources_ptr[i].byte_end),
            weight=float(sources_ptr[i].weight),
        )
        for i in range(n_sources)
    ]

    groups_ptr = lib.ztok_superposition_plan_groups(handle)
    fusion_by_code = {
        SUPERPOSITION_FUSION_MEAN: FusionKind.MEAN,
        SUPERPOSITION_FUSION_WEIGHTED_MEAN: FusionKind.WEIGHTED_MEAN,
        SUPERPOSITION_FUSION_NORM_PRESERVING_MEAN:
            FusionKind.NORM_PRESERVING_MEAN,
    }
    groups: list[SuperpositionGroup] = []
    for i in range(n_groups):
        group = groups_ptr[i]
        source_start = int(group.source_start)
        source_end = source_start + int(group.source_count)
        groups.append(
            SuperpositionGroup(
                output_index=int(group.output_index),
                sources=tuple(sources[source_start:source_end]),
                fusion=fusion_by_code[int(group.fusion)],
                kind=_GROUP_KIND_NAMES[int(group.kind)],
                position_start=int(group.position_start),
                position_end=int(group.position_end),
                byte_start=int(group.byte_start),
                byte_end=int(group.byte_end),
                center_position=float(group.center_position),
                normalized_center=float(group.normalized_center),
            )
        )

    json_len = ctypes.c_size_t()
    rc = lib.ztok_superposition_plan_json(
        handle, None, 0, ctypes.byref(json_len)
    )
    if rc != ZTOK_ERR_BUFFER_TOO_SMALL:
        _raise_for_status(rc, "ztok_superposition_plan_json(size)")
    json_buffer = (ctypes.c_char * json_len.value)()
    rc = lib.ztok_superposition_plan_json(
        handle, json_buffer, json_len.value, ctypes.byref(json_len)
    )
    _raise_for_status(rc, "ztok_superposition_plan_json")

    return SuperpositionPlan(
        original_ids=ids,
        original_offsets=tuple(offsets),
        groups=tuple(groups),
        _json=bytes(json_buffer).decode("utf-8"),
    )


@dataclass(frozen=True)
class SemanticSpan:
    span_id: str
    caption_id: int
    token_start: int
    token_end: int
    byte_start: int
    byte_end: int
    text: str = ""
    role: Optional[str] = None
    entity_id: Optional[str] = None
    confidence: float = 1.0
    caption_quality: float = 1.0
    grounding: Optional[tuple[float, ...]] = None
    grounding_confidence: float = 1.0
    section_reliability: float = 1.0
    kind: str = "other"
    order_class: Optional[str] = None
    relation_from_entity_id: Optional[str] = None
    relation_to_entity_id: Optional[str] = None
    contradicts: tuple[str, ...] = ()
    section: Optional[str] = None


@dataclass(frozen=True)
class SemanticSuperpositionConfig:
    default_cosine_threshold: float = 0.92
    role_thresholds: Mapping[str, float] = field(default_factory=dict)
    protected_roles: Optional[tuple[str, ...]] = None
    never_fuse_roles: Optional[tuple[str, ...]] = None
    protected_cosine_threshold: float = 0.995
    semantic_weight: float = 1.0
    role_weight: float = 0.0
    entity_weight: float = 0.0
    grounding_weight: float = 0.0
    contradiction_weight: float = 1.0
    contradiction_block_threshold: float = 0.5
    minimum_score: float = 0.92
    maximum_cluster_size: int = 12
    minimum_support_count: int = 2
    minimum_support_fraction: float = 0.0
    require_role_match: bool = True
    require_entity_match: bool = True
    block_contradictions: bool = True
    fusion: FusionKind = FusionKind.NORM_PRESERVING_MEAN
    verbose_diagnostics: bool = False

    def to_dict(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "default_cosine_threshold": self.default_cosine_threshold,
            "role_thresholds": dict(self.role_thresholds),
            "protected_cosine_threshold": self.protected_cosine_threshold,
            "semantic_weight": self.semantic_weight,
            "role_weight": self.role_weight,
            "entity_weight": self.entity_weight,
            "grounding_weight": self.grounding_weight,
            "contradiction_weight": self.contradiction_weight,
            "contradiction_block_threshold": self.contradiction_block_threshold,
            "minimum_score": self.minimum_score,
            "maximum_cluster_size": self.maximum_cluster_size,
            "minimum_support_count": self.minimum_support_count,
            "minimum_support_fraction": self.minimum_support_fraction,
            "require_role_match": self.require_role_match,
            "require_entity_match": self.require_entity_match,
            "block_contradictions": self.block_contradictions,
            "fusion": getattr(self.fusion, "value", self.fusion),
            "verbose_diagnostics": self.verbose_diagnostics,
        }
        if self.protected_roles is not None:
            document["protected_roles"] = list(self.protected_roles)
        if self.never_fuse_roles is not None:
            document["never_fuse_roles"] = list(self.never_fuse_roles)
        return document


def _embedding_rows(embeddings: Any) -> list[list[float]]:
    value = embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings
    rows = [[float(component) for component in row] for row in value]
    if rows:
        dimensions = len(rows[0])
        if dimensions == 0 or any(len(row) != dimensions for row in rows):
            raise ValueError("embeddings must be a rectangular non-empty-width matrix")
    return rows


def build_semantic_plan(
    spans: Sequence[SemanticSpan],
    embeddings: Any,
    config: SemanticSuperpositionConfig = SemanticSuperpositionConfig(),
    *,
    image_id: str = "",
    original_token_count: Optional[int] = None,
) -> dict[str, Any]:
    """Build a conservative CCSS plan from contextual span embeddings.

    Rows in ``embeddings`` align one-to-one with ``spans``. Those vectors
    should come from a documented contextual encoder layer and span-pooling
    policy; ztok does not run the encoder.
    """

    rows = _embedding_rows(embeddings)
    if len(rows) != len(spans):
        raise ValueError("embeddings row count must equal len(spans)")
    grounding_presence = {span.grounding is not None for span in spans}
    if len(grounding_presence) > 1:
        raise ValueError("grounding vectors must be present for every span or none")
    grounding_widths = {
        len(span.grounding) for span in spans if span.grounding is not None
    }
    if 0 in grounding_widths or len(grounding_widths) > 1:
        raise ValueError("grounding vectors must have one shared non-zero width")
    caption_ids = sorted({span.caption_id for span in spans})
    if caption_ids and caption_ids != list(range(max(caption_ids) + 1)):
        raise ValueError("caption_id values must be contiguous starting at zero")

    by_caption: dict[int, list[tuple[SemanticSpan, list[float]]]] = {}
    for span, embedding in zip(spans, rows):
        by_caption.setdefault(span.caption_id, []).append((span, embedding))
    captions = []
    for caption_id in range(max(caption_ids) + 1 if caption_ids else 0):
        caption_spans = by_caption.get(caption_id, [])
        if caption_spans:
            qualities = {span.caption_quality for span, _ in caption_spans}
            if len(qualities) != 1:
                raise ValueError(
                    f"all spans in caption {caption_id} must share caption_quality"
                )
        section = next(
            (span.section for span, _ in caption_spans if span.section is not None),
            None,
        )
        serialized_spans = []
        for span, embedding in caption_spans:
            item = {
                "span_id": span.span_id,
                "token_start": span.token_start,
                "token_end": span.token_end,
                "byte_start": span.byte_start,
                "byte_end": span.byte_end,
                "text": span.text,
                "role": span.role,
                "entity_id": span.entity_id,
                "confidence": span.confidence,
                "grounding": (
                    list(span.grounding) if span.grounding is not None else None
                ),
                "grounding_confidence": span.grounding_confidence,
                "section_reliability": span.section_reliability,
                "embedding": embedding,
                "kind": span.kind,
                "order_class": span.order_class,
                "relation_from_entity_id": span.relation_from_entity_id,
                "relation_to_entity_id": span.relation_to_entity_id,
                "contradicts": list(span.contradicts),
            }
            serialized_spans.append(
                {key: value for key, value in item.items() if value is not None}
            )
        caption = {
            "caption_id": caption_id,
            "quality": (
                caption_spans[0][0].caption_quality if caption_spans else 1.0
            ),
            "text": "",
            "tokens": [],
            "spans": serialized_spans,
        }
        if section is not None:
            caption["section"] = section
        captions.append(caption)

    if original_token_count is None:
        token_ends: dict[int, int] = {}
        for span in spans:
            token_ends[span.caption_id] = max(
                token_ends.get(span.caption_id, 0), span.token_end
            )
        original_token_count = sum(token_ends.values())

    semantic_document = {
        "schema": "ztok.semantic_spans.v1",
        "image_id": image_id,
        "original_token_count": original_token_count,
        "captions": captions,
    }
    return build_semantic_plan_from_exchange(semantic_document, config.to_dict())


def build_semantic_plan_from_exchange(
    semantic_document: Mapping[str, Any],
    config: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build CCSS directly from a ``ztok.semantic_spans.v1`` document."""

    semantic_json = json.dumps(
        semantic_document, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    config_json = (
        json.dumps(config, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if config is not None
        else b""
    )
    lib = _get_lib()
    output_length = ctypes.c_size_t()
    rc = lib.ztok_ccss_build_json(
        semantic_json,
        len(semantic_json),
        config_json if config_json else None,
        len(config_json),
        None,
        0,
        ctypes.byref(output_length),
    )
    if rc != ZTOK_ERR_BUFFER_TOO_SMALL:
        _raise_for_status(rc, "ztok_ccss_build_json(size)")
    output = (ctypes.c_char * output_length.value)()
    rc = lib.ztok_ccss_build_json(
        semantic_json,
        len(semantic_json),
        config_json if config_json else None,
        len(config_json),
        output,
        output_length.value,
        ctypes.byref(output_length),
    )
    _raise_for_status(rc, "ztok_ccss_build_json")
    return json.loads(bytes(output).decode("utf-8"))


def build_fixed_plan(
    pipeline: "Pipeline",
    text: Union[str, bytes],
    config: FixedSuperpositionConfig = FixedSuperpositionConfig(),
) -> SuperpositionPlan:
    """Encode ``text`` normally, then build a fixed superposition plan."""

    lib = _get_lib()
    data = text.encode("utf-8") if isinstance(text, str) else bytes(text)
    c_config = config._as_c()
    status = ctypes.c_int()
    handle = lib.ztok_pipeline_encode_superposition(
        pipeline._raw(),
        data,
        len(data),
        ctypes.byref(c_config),
        ctypes.byref(status),
    )
    _raise_for_status(status.value, "ztok_pipeline_encode_superposition")
    if not handle:
        raise RuntimeError("ztok_pipeline_encode_superposition returned NULL")
    try:
        return _materialize_plan(lib, handle)
    finally:
        lib.ztok_superposition_plan_free(handle)


__all__ = [
    "FixedSuperpositionConfig",
    "FusionKind",
    "SCHEMA",
    "SCHEMA_VERSION",
    "SemanticSpan",
    "SemanticSuperpositionConfig",
    "SourceToken",
    "SuperpositionGroup",
    "SuperpositionPlan",
    "build_fixed_plan",
    "build_semantic_plan",
    "build_semantic_plan_from_exchange",
]
