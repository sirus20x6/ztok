from build_proofwriter_statement_expansions import (
    build_context_expansions,
    statement_records,
)


class FakePipeline:
    def encode(self, text: str) -> list[int]:
        return [sum(text.encode()) % 997]


def test_statements_expand_one_valid_logical_slot_at_a_time() -> None:
    records = statement_records(
        ["Gary is furry.", "Gary is green.", "Bob is furry."],
        [],
        {
            ("Gary is furry.", "True"): [0],
            ("Gary is green.", "True"): [1],
            ("Gary is rough.", "Unknown"): [2],
        },
        (0, 1, 2),
    )
    expansions = build_context_expansions(
        "context", records, FakePipeline(), maximum_alternatives=8
    )

    fact_predicates = [
        expansion
        for expansion in expansions
        if expansion["statement_kind"] == "fact"
        and expansion["expanded_slot_kind"] == "predicate"
    ]
    assert len(fact_predicates) == 1
    assert [value["surface"] for value in fact_predicates[0]["alternatives"]] == [
        "furry",
        "green",
    ]
    assert fact_predicates[0]["branch_policy"] == "one_logical_slot_only"


def test_question_expansion_never_mixes_truth_classes() -> None:
    records = statement_records(
        ["Gary is furry."],
        [],
        {
            ("Gary is furry.", "True"): [0],
            ("Gary is green.", "True"): [1],
            ("Gary is rough.", "Unknown"): [2],
        },
        (0, 1, 2),
    )
    expansions = build_context_expansions(
        "context", records, FakePipeline(), maximum_alternatives=8
    )

    questions = [
        expansion
        for expansion in expansions
        if expansion["statement_kind"] == "question"
    ]
    assert len(questions) == 1
    assert questions[0]["truth_class"] == "True"
    assert {value["surface"] for value in questions[0]["alternatives"]} == {
        "furry",
        "green",
    }
