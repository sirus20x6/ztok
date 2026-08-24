from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import ztok
from ztok.superposition import (
    FixedSuperpositionConfig,
    FusionKind,
    SemanticSpan,
    SemanticSuperpositionConfig,
    build_fixed_plan,
    build_semantic_plan,
)


def test_fixed_plan_preserves_ordinary_encoding_and_offsets() -> None:
    with ztok.Pipeline.byte_id() as pipeline:
        text = "A φω"
        ordinary = pipeline.encode(text)
        plan = build_fixed_plan(
            pipeline,
            text,
            FixedSuperpositionConfig(group_size=2, fusion=FusionKind.MEAN),
        )

    assert list(plan.original_ids) == ordinary
    assert plan.original_offsets[0] == (0, 1)
    assert plan.original_offsets[-1][1] == len(text.encode("utf-8"))
    assert plan.output_token_count == (len(ordinary) + 1) // 2
    assert all(group.fusion is FusionKind.MEAN for group in plan.groups)
    assert sum(len(group.sources) for group in plan.groups) == len(ordinary)


def test_fixed_plan_json_is_versioned_and_deterministic() -> None:
    with ztok.Pipeline.byte_id() as pipeline:
        config = FixedSuperpositionConfig(group_size=4)
        first = build_fixed_plan(pipeline, "abcdefgh", config)
        second = build_fixed_plan(pipeline, "abcdefgh", config)

    assert first.to_json() == second.to_json()
    document = json.loads(first.to_json())
    assert document["schema"] == "ztok.superposition.v1"
    assert document["schema_version"] == 1
    assert document["original_token_count"] == 8
    assert document["output_token_count"] == 2


def test_fixed_plan_empty_and_partial_tail() -> None:
    with ztok.Pipeline.byte_id() as pipeline:
        empty = build_fixed_plan(pipeline, b"")
        tail = build_fixed_plan(
            pipeline,
            b"abcde",
            FixedSuperpositionConfig(
                group_size=4,
                allow_partial_final_group=False,
            ),
        )

    assert empty.original_ids == ()
    assert empty.groups == ()
    assert tail.groups[-1].kind == "uncovered_tail"
    assert tail.groups[-1].sources[0].token_id == ord("e")


def test_semantic_plan_fuses_synonyms_and_keeps_role_mismatch() -> None:
    spans = [
        SemanticSpan(
            "bright",
            caption_id=0,
            token_start=0,
            token_end=1,
            byte_start=0,
            byte_end=6,
            role="lighting_intensity",
            entity_id="light",
            kind="attribute",
        ),
        SemanticSpan(
            "brilliant",
            caption_id=1,
            token_start=0,
            token_end=1,
            byte_start=0,
            byte_end=9,
            role="lighting_intensity",
            entity_id="light",
            kind="attribute",
        ),
        SemanticSpan(
            "white",
            caption_id=2,
            token_start=0,
            token_end=1,
            byte_start=0,
            byte_end=5,
            role="color",
            entity_id="light",
            kind="attribute",
        ),
    ]
    plan = build_semantic_plan(
        spans,
        [[1.0, 0.0], [0.999, 0.02], [1.0, 0.0]],
        SemanticSuperpositionConfig(verbose_diagnostics=True),
        image_id="python",
    )

    consensus = [unit for unit in plan["units"] if unit["kind"] == "consensus"]
    assert len(consensus) == 1
    assert consensus[0]["members"] == ["bright", "brilliant"]
    assert any(
        unit.get("span_id") == "white" and unit["kind"] == "unique"
        for unit in plan["units"]
    )
    assert any(
        pair["reason"] == "role_mismatch"
        for pair in plan["diagnostics"]["rejected_pairs"]
    )


def test_semantic_plan_keeps_explicit_contradictions_as_alternatives() -> None:
    spans = [
        SemanticSpan(
            "standing",
            caption_id=0,
            token_start=0,
            token_end=1,
            byte_start=0,
            byte_end=8,
            role="posture",
            entity_id="person",
            kind="action",
            contradicts=("seated",),
        ),
        SemanticSpan(
            "seated",
            caption_id=1,
            token_start=0,
            token_end=1,
            byte_start=0,
            byte_end=6,
            role="posture",
            entity_id="person",
            kind="action",
        ),
    ]
    plan = build_semantic_plan(spans, [[1.0, 0.0], [1.0, 0.0]])
    alternatives = [
        unit for unit in plan["units"]
        if unit["kind"] == "uncertainty" and "alternatives" in unit
    ]
    assert len(alternatives) == 1
    assert alternatives[0]["reason"] == "explicit contradiction"


def test_pytorch_reference_adapter_reconstructs_fusion_formulas() -> None:
    torch = pytest.importorskip("torch")
    examples = Path(__file__).resolve().parents[3] / "examples" / "python"
    sys.path.insert(0, str(examples))
    try:
        from superposition_embeddings import apply_superposition_plan
    finally:
        sys.path.pop(0)

    embeddings = torch.tensor(
        [[3.0, 0.0], [0.0, 4.0], [1.0, 0.0], [1.0, 0.0]]
    )
    with ztok.Pipeline.byte_id() as pipeline:
        mean_plan = build_fixed_plan(
            pipeline,
            b"abcd",
            FixedSuperpositionConfig(group_size=2, fusion=FusionKind.MEAN),
        )
        norm_plan = build_fixed_plan(
            pipeline,
            b"abcd",
            FixedSuperpositionConfig(
                group_size=2,
                fusion=FusionKind.NORM_PRESERVING_MEAN,
            ),
        )

    mean = apply_superposition_plan(embeddings, mean_plan)
    norm = apply_superposition_plan(embeddings, norm_plan)
    assert torch.allclose(mean[0], torch.tensor([1.5, 2.0]))
    assert torch.allclose(norm[0], torch.tensor([2.1, 2.8]), atol=1e-6)
