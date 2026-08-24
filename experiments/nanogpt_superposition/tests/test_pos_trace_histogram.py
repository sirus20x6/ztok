import spacy
from build_pos_trace_histogram import (
    eligible_token,
    logical_trace,
    normalized_morph,
    trace_id,
)


def test_logical_trace_keeps_role_relations_and_morphology() -> None:
    nlp = spacy.load("en_core_web_sm", exclude=["ner"])
    doc = nlp("Dogs quickly chase balls.")
    chase = next(token for token in doc if token.lemma_ == "chase")
    trace = logical_trace(chase)

    assert trace["pos"] == "VERB"
    assert "VerbForm=Fin" in normalized_morph(chase)
    assert ["advmod:ADV", 1] in trace["children"]
    assert ["nsubj:NOUN", 1] in trace["children"]
    assert trace_id(trace) == trace_id(dict(reversed(list(trace.items()))))


def test_histogram_excludes_protected_and_named_values() -> None:
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("Alice never runs left.")
    by_text = {token.text: token for token in doc}

    assert not eligible_token(by_text["Alice"])
    assert not eligible_token(by_text["never"])
    assert eligible_token(by_text["runs"])
    assert not eligible_token(by_text["left"])
