from contextualize_dataset import (
    find_surface,
    overlapping_token_indices,
    teacher_embedding_eligible,
)


def test_find_surface_uses_explicit_range_or_auditable_surface() -> None:
    text = "A radiant canine sprints."
    assert find_surface(text, {"span_id": "x", "char_start": 2, "char_end": 9}) == (
        2,
        9,
    )
    assert find_surface(text, {"span_id": "e1-radiant"}) == (2, 9)
    assert find_surface(text, {"span_id": "x", "text": "canine"}) == (10, 16)


def test_teacher_offset_overlap_keeps_multi_piece_span_whole() -> None:
    offsets = [(0, 0), (0, 1), (2, 5), (5, 9), (10, 16)]
    assert overlapping_token_indices(offsets, 2, 9) == [2, 3]


def test_partial_utf8_residual_is_not_sent_to_semantic_teacher() -> None:
    assert not teacher_embedding_eligible(
        {"protected": True, "teacher_embedding_eligible": False}
    )
    assert teacher_embedding_eligible({"protected": False})
