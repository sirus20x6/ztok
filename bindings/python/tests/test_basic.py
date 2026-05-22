"""Basic encode/decode + lifecycle tests."""

from __future__ import annotations

import pytest

import ztok


def test_version_is_nonempty_string() -> None:
    v = ztok.version()
    assert isinstance(v, str)
    assert v.count(".") >= 1
    assert all(part.isdigit() for part in v.split(".")[:2])


def test_byte_id_encode_decode_roundtrip() -> None:
    with ztok.Pipeline.byte_id() as pipe:
        ids = pipe.encode("hi")
        assert ids == [0x68, 0x69]
        assert pipe.decode(ids) == "hi"


def test_bpe_encode_decode_roundtrip(bpe_pipeline: ztok.Pipeline) -> None:
    text = "hello world"
    ids = bpe_pipeline.encode(text)
    assert len(ids) > 0
    assert bpe_pipeline.decode(ids) == text


def test_decoded_text_equals_input_on_100_lines(bpe_pipeline: ztok.Pipeline) -> None:
    # 100 lines: alternating short/medium English snippets from the vocab's
    # merge coverage so we don't blow byte-fallback budget on novel UTF-8.
    snippets = [
        "hello world",
        "the quick brown fox",
        "foo bar baz",
        "hello there hello world",
        "the the the",
        "  the  ",
        "foo",
        "hello",
        " world",
        "bar baz",
    ]
    lines = [snippets[i % len(snippets)] for i in range(100)]
    for line in lines:
        ids = bpe_pipeline.encode(line)
        decoded = bpe_pipeline.decode(ids)
        assert decoded == line, f"round-trip failed for {line!r}: got {decoded!r}"


def test_empty_input_returns_empty_ids() -> None:
    with ztok.Pipeline.byte_id() as pipe:
        assert pipe.encode("") == []
        assert pipe.decode([]) == ""


def test_context_manager_closes_pipeline() -> None:
    with ztok.Pipeline.byte_id() as pipe:
        assert pipe.encode("x") == [ord("x")]
    # After exit the handle must be released. Calling encode raises.
    with pytest.raises(ztok.ZtokError):
        pipe.encode("y")


def test_explicit_close_is_idempotent() -> None:
    pipe = ztok.Pipeline.byte_id()
    pipe.close()
    pipe.close()  # second close must be a no-op
    with pytest.raises(ztok.ZtokError):
        pipe.encode("z")


def test_invalid_input_raises_typed_error() -> None:
    # Passing an unknown model kind through the raw config trips
    # ZTOK_ERR_INVALID_INPUT -> ZtokInvalidInputError.
    with pytest.raises(ztok.ZtokInvalidInputError):
        ztok.Pipeline.byte_id(normalizer=999)  # type: ignore[arg-type]


def test_decode_bytes_returns_raw_bytes() -> None:
    with ztok.Pipeline.byte_id() as pipe:
        ids = pipe.encode("ab")
        assert pipe.decode_bytes(ids) == b"ab"
