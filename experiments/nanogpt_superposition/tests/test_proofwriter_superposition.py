from benchmark_proofwriter_superposition import (
    canonicalize_trace,
    parse_fact,
    proof_traces,
)


def test_proofwriter_fact_parser_handles_unary_and_binary_atoms() -> None:
    assert parse_fact("Gary is not smart.") == ("Gary", "smart", None)
    assert parse_fact("The squirrel eats the lion.") == (
        "The squirrel",
        "eats",
        "the lion",
    )


def test_role_safe_trace_renames_entities_but_keeps_predicate_identity() -> None:
    left = canonicalize_trace(
        ["Gary is furry."], [], "Gary is furry.", "True", rename_predicates=False
    )
    right = canonicalize_trace(
        ["Bob is furry."], [], "Bob is furry.", "True", rename_predicates=False
    )
    different = canonicalize_trace(
        ["Bob is green."], [], "Bob is green.", "True", rename_predicates=False
    )

    assert [unit.signature for unit in left] == [unit.signature for unit in right]
    assert [unit.signature for unit in left] != [unit.signature for unit in different]


def test_isomorphism_ceiling_consistently_renames_predicates() -> None:
    furry = canonicalize_trace(
        ["Gary is furry."],
        ["If Gary is furry then Gary is green."],
        "Gary is green.",
        "True",
        rename_predicates=True,
    )
    rough = canonicalize_trace(
        ["Bob is rough."],
        ["If Bob is rough then Bob is white."],
        "Bob is white.",
        "True",
        rename_predicates=True,
    )

    assert [unit.signature for unit in furry] == [unit.signature for unit in rough]


def test_role_safe_mode_does_not_merge_distinct_variable_classes() -> None:
    someone = canonicalize_trace(
        [],
        ["If someone is green then they are nice."],
        "Bob is nice.",
        "True",
        rename_predicates=False,
    )
    something = canonicalize_trace(
        [],
        ["If something is green then it is nice."],
        "Bob is nice.",
        "True",
        rename_predicates=False,
    )

    assert [unit.signature for unit in someone] != [
        unit.signature for unit in something
    ]


def test_unknown_rows_do_not_claim_a_known_proof_trace() -> None:
    row = {
        "answer": "Unknown",
        "used_facts": [],
        "used_rules": [],
        "question": "Bob is white.",
        "depth": 1,
    }
    assert proof_traces(row, 0) == []
