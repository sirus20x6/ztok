from annotate_corpus_sample import (
    assign_ztok_tokens,
    byte_range,
    semantic_role,
    token_range,
)


def test_annotation_helpers_preserve_utf8_and_ztok_ranges() -> None:
    text = "A café glows"
    assert byte_range(text, 2, 6) == (2, 7)
    assert token_range([0, 2, 7], [1, 7, 12], 2, 7) == (1, 2)


def test_automatic_semantic_roles_are_conservative() -> None:
    assert semantic_role("NOUN", "nsubj") == "entity"
    assert semantic_role("VERB", "ROOT") == "action"
    assert semantic_role("NUM", "nummod") == "count"
    assert semantic_role("CCONJ", "cc") == "syntax"


def test_each_ztok_token_is_assigned_to_the_largest_byte_overlap() -> None:
    assignments, unassigned = assign_ztok_tokens(
        [0, 1, 6],
        [1, 6, 7],
        [(1, 2), (2, 6)],
    )
    assert assignments == [[], [1]]
    assert unassigned == [0, 2]
