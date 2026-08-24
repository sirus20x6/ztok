#!/usr/bin/env python3
"""Extract deterministic 5–12 caption-stage phrases from i1 modular JSONL.

The output is an embedding-free precursor consumed by
``examples/python/qwen_span_adapter.py``. Each selected stage phrase is one
caption and one whole-phrase semantic span, preserving section, role, entity,
and relation provenance available in the existing caption pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def add(
    candidates: list[dict[str, Any]],
    text: Any,
    *,
    section: str,
    role: str,
    kind: str,
    entity_id: str | None = None,
    relation_from: str | None = None,
    relation_to: str | None = None,
    confidence: float = 1.0,
) -> None:
    if not isinstance(text, str) or not text.strip():
        return
    normalized = " ".join(text.split())
    candidates.append(
        {
            "text": normalized,
            "section": section,
            "role": role,
            "kind": kind,
            "entity_id": entity_id,
            "relation_from_entity_id": relation_from,
            "relation_to_entity_id": relation_to,
            "confidence": confidence,
        }
    )


def strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                yield item


def extract(record: dict[str, Any], maximum: int) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    scene = record.get("scene") or {}
    add(candidates, scene.get("setting"), section="scene", role="global_scene", kind="entity")
    for field in ("foreground", "midground", "background"):
        for text in strings(scene.get(field)):
            add(candidates, text, section="scene", role=f"scene_{field}", kind="detail")

    entities = record.get("entities") or {}
    for entity in (entities.get("primary_entities") or []):
        if isinstance(entity, str):
            add(
                candidates,
                entity,
                section="entities",
                role="object_identity",
                kind="entity",
            )
            continue
        if not isinstance(entity, dict):
            continue
        identifier = f"entity-{entity.get('entity_id', len(candidates))}"
        add(
            candidates,
            entity.get("identity_or_species") or entity.get("type"),
            section="entities",
            role="object_identity",
            kind="entity",
            entity_id=identifier,
        )
        add(
            candidates,
            entity.get("appearance"),
            section="entities",
            role="appearance",
            kind="attribute",
            entity_id=identifier,
        )
        add(
            candidates,
            entity.get("pose_or_orientation"),
            section="entities",
            role="posture",
            kind="action",
            entity_id=identifier,
        )
        add(
            candidates,
            entity.get("action"),
            section="entities",
            role="action",
            kind="action",
            entity_id=identifier,
        )
    for relation in strings(entities.get("relationships")):
        add(
            candidates,
            relation,
            section="entities",
            role="entity_relation",
            kind="relation",
        )

    lighting = record.get("lighting") or {}
    lighting_roles = (
        ("light_sources", "light_source"),
        ("direction", "lighting_direction"),
        ("quality", "lighting_quality"),
        ("intensity_and_exposure", "lighting_intensity"),
        ("shadow_character", "shadow_character"),
        ("color_temperature", "color_temperature"),
        ("palette", "color_palette"),
        ("medium_or_style", "style_medium"),
    )
    for field, role in lighting_roles:
        for text in strings(lighting.get(field)):
            add(candidates, text, section="lighting", role=role, kind="attribute")

    camera = record.get("camera") or {}
    for field, role in (
        ("framing", "spatial_relation"),
        ("focus", "camera_focus"),
        ("perspective", "camera_perspective"),
    ):
        add(candidates, camera.get(field), section="camera", role=role, kind="relation")

    text_stage = record.get("text") or {}
    for region in text_stage.get("text_regions") or []:
        if isinstance(region, dict):
            add(
                candidates,
                region.get("text"),
                section="visible_text",
                role="visible_text",
                kind="visible_text",
            )

    # Stable first-occurrence de-duplication. Long final captions are not
    # injected as single spans; stage phrases retain useful granularity.
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str | None]] = set()
    for candidate in candidates:
        key = (
            candidate["text"].casefold(),
            candidate["role"],
            candidate["entity_id"],
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(candidate)
        if len(selected) == maximum:
            break
    if len(selected) < 5:
        return None

    style_text = " ".join(
        strings((record.get("lighting") or {}).get("medium_or_style"))
    ).casefold()
    animation_markers = ("anime", "animation", "illustration", "digital art", "render")
    media = "animation_or_render" if any(
        marker in style_text for marker in animation_markers
    ) else "photo_or_unknown"
    primary_entities = (record.get("entities") or {}).get("primary_entities") or []
    subject_count = "multiple_subjects" if len(primary_entities) > 1 else "one_subject"
    text_regions = (record.get("text") or {}).get("text_regions") or []
    physical_text = (record.get("qc") or {}).get("physical_text") or []
    text_density = "ocr_text_heavy" if text_regions or physical_text else "no_visible_text"
    complexity = "complex_scene" if len(candidates) >= 20 else "simple_scene"

    image_id = (
        record.get("dataset_key")
        or record.get("filename")
        or record.get("relative_path")
    )
    captions = []
    for caption_id, candidate in enumerate(selected):
        text = candidate.pop("text")
        span = {
            "span_id": f"c{caption_id}-s0",
            "text": text,
            "byte_start": 0,
            "byte_end": len(text.encode("utf-8")),
            **{key: value for key, value in candidate.items() if key != "section" and value is not None},
        }
        captions.append(
            {
                "caption_id": caption_id,
                "section": candidate["section"],
                "text": text,
                "spans": [span],
            }
        )
    return {
        "image_id": str(image_id),
        "source_image": record.get("relative_path"),
        "source_model": record.get("model"),
        "dataset_group": record.get("dataset_group"),
        "strata": {
            "media": media,
            "subject_count": subject_count,
            "text_density": text_density,
            "complexity": complexity,
            "caption_disagreement": "not_available",
        },
        "captions": captions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10_000)
    parser.add_argument("--captions-per-image", type=int, default=12)
    parser.add_argument(
        "--stratify",
        action="store_true",
        help="round-robin across media/subject/OCR/complexity strata",
    )
    args = parser.parse_args()
    if not 5 <= args.captions_per_image <= 12:
        raise SystemExit("--captions-per-image must be in [5, 12]")
    skipped = 0
    groups: list[dict[str, Any]] = []
    with args.input.open() as source:
        for line in source:
            if not line.strip():
                continue
            group = extract(json.loads(line), args.captions_per_image)
            if group is None:
                skipped += 1
                continue
            groups.append(group)
            if not args.stratify and args.limit and len(groups) >= args.limit:
                break
    if args.stratify:
        buckets: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        for group in groups:
            strata = group["strata"]
            key = (
                strata["media"],
                strata["subject_count"],
                strata["text_density"],
                strata["complexity"],
            )
            buckets.setdefault(key, []).append(group)
        selected = []
        positions = {key: 0 for key in buckets}
        while (not args.limit or len(selected) < args.limit):
            progressed = False
            for key in sorted(buckets):
                position = positions[key]
                if position >= len(buckets[key]):
                    continue
                selected.append(buckets[key][position])
                positions[key] += 1
                progressed = True
                if args.limit and len(selected) >= args.limit:
                    break
            if not progressed:
                break
        groups = selected
    elif args.limit:
        groups = groups[: args.limit]
    with args.output.open("w") as destination:
        for group in groups:
            destination.write(json.dumps(group, ensure_ascii=False) + "\n")
    print(json.dumps({
        "written": len(groups),
        "skipped": skipped,
        "stratified": args.stratify,
        "output": str(args.output),
    }))


if __name__ == "__main__":
    main()
