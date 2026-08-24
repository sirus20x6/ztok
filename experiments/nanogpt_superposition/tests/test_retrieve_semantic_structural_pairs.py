import torch
from retrieve_semantic_structural_pairs import retrieve_pairs


def _example(example_id: int, text: str, role: str = "entity") -> dict:
    return {
        "example_id": example_id,
        "text": text,
        "spans": [
            {
                "span_id": f"e{example_id}-s0",
                "grammatical_role": "nsubj",
                "semantic_role": role,
                "protected": False,
            },
            {
                "span_id": f"e{example_id}-s1",
                "grammatical_role": "root",
                "semantic_role": "action",
                "protected": False,
            },
            {
                "span_id": f"e{example_id}-s2",
                "grammatical_role": "dobj",
                "semantic_role": "entity",
                "protected": False,
            },
        ],
        "relations": [],
    }


def test_semantic_retrieval_precedes_structural_gate() -> None:
    examples = [
        _example(10, "A canine chases a ball."),
        _example(20, "A dog pursues the toy."),
        _example(30, "Markets closed lower today."),
    ]
    selected, counts = retrieve_pairs(
        examples,
        torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]),
        semantic_top_k=2,
        minimum_semantic_score=0.9,
        minimum_structural_score=0.9,
        minimum_content_spans=3,
        max_pairs=2,
        similarity_device="cpu",
    )

    assert [
        (pair["left_example_id"], pair["right_example_id"]) for pair in selected
    ] == [(10, 20)]
    assert selected[0]["exact_duplicate"] is False
    assert counts["selected_exact_duplicate_count"] == 0


def test_retrieval_labels_exact_duplicates_separately() -> None:
    examples = [_example(0, "same"), _example(1, "same")]
    selected, counts = retrieve_pairs(
        examples,
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        semantic_top_k=1,
        minimum_semantic_score=0.9,
        minimum_structural_score=0.9,
        minimum_content_spans=3,
        max_pairs=1,
        similarity_device="cpu",
    )

    assert selected[0]["exact_duplicate"] is True
    assert counts["selected_exact_duplicate_count"] == 1
