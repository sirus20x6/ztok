from collections import Counter

from retrieve_structural_pairs import structural_features, weighted_jaccard


def test_weighted_jaccard_respects_repeated_structure() -> None:
    assert weighted_jaccard(Counter(a=2, b=1), Counter(a=1, b=1)) == 2 / 3
    assert weighted_jaccard(Counter(), Counter()) == 0.0


def test_structural_features_exclude_protected_nodes() -> None:
    example = {
        "spans": [
            {
                "span_id": "entity",
                "grammatical_role": "nsubj",
                "semantic_role": "entity",
                "protected": False,
            },
            {
                "span_id": "det",
                "grammatical_role": "det",
                "semantic_role": "syntax",
                "protected": True,
            },
        ],
        "relations": [
            {
                "source_span_id": "det",
                "relation": "det",
                "target_span_id": "entity",
            }
        ],
    }
    features = structural_features(example)
    assert sum(features.values()) == 1
    assert next(iter(features)).startswith("node:nsubj:entity")
