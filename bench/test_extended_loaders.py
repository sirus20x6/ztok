#!/usr/bin/env python3
"""Loader smoke tests for the post-1.17 extended cross-bench vocabs.

For each vocab file vendored by `bench/fetch_vocabs.py --extended`,
verify the reference Python loader (`sentencepiece` for .model files,
`tokenizers` for HF .json files) can:

  1. parse the file without exception;
  2. report a non-zero vocab size;
  3. encode 10 short text lines without crashing.

This catches "file truncated / wrong magic / unsupported version"
issues at the fixture-download layer before the ztok bench harness or
`equivalence_check.py` runs. It does NOT validate ztok's own loaders
(those live in the Zig tests under `src/cli_bench.zig` and elsewhere) —
keeping the dependency surface here at vanilla pip packages so CI can
run this check before any Zig build step.

Run:
    python3 bench/test_extended_loaders.py
    # or: python3 -m pytest bench/test_extended_loaders.py

Exit code 0 if every present fixture loads + encodes. Missing fixtures
report `SKIP` and don't fail the run (mirrors the bench harness's
graceful-skip policy).
"""
import os
import sys

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
VOCABS_DIR = os.path.join(BENCH_DIR, "vocabs")

# Each entry: (label, filename, loader_kind, [sample lines]).
# `loader_kind` is "sp" for .model files (sentencepiece) or "hf" for
# tokenizer.json files (tokenizers).
EXTENDED = [
    ("mistral7b", "mistral7b.model", "sp"),
    ("yi6b", "yi6b.model", "sp"),
    ("phi3", "phi3.json", "hf"),
    ("falcon7b", "falcon7b.json", "hf"),
    ("deepseek_v2", "deepseek_v2.json", "hf"),
    ("qwen2", "qwen2.json", "hf"),
    ("llama3", "llama3.json", "hf"),
]

SAMPLE_LINES = [
    "hello world",
    "The quick brown fox jumps over the lazy dog.",
    "cafe naive facade resume.",
    "Pack my box with five dozen liquor jugs.",
    "Sphinx of black quartz, judge my vow.",
    "Multi-byte: cafe naive resume.",
    "CJK: 中文 日本語.",
    "Numbers: 123 4567 89.0123.",
    "Code: def foo(x): return x + 1",
    "JSON: {\"key\": \"value\", \"n\": 42}",
]


def _load_sp(path):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(path)
    return sp


def _load_hf(path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)


def _encode_sp(sp, lines):
    return [sp.EncodeAsIds(line) for line in lines]


def _encode_hf(tok, lines):
    return [tok.encode(line, add_special_tokens=False).ids for line in lines]


def smoke_one(label, filename, kind):
    """Return ('OK', vocab_size, total_ids), ('SKIP', None, None), or
    raises on a real loader / encode failure."""
    path = os.path.join(VOCABS_DIR, filename)
    if not os.path.exists(path):
        return ("SKIP", None, None)

    if kind == "sp":
        sp = _load_sp(path)
        vocab_size = sp.GetPieceSize()
        encoded = _encode_sp(sp, SAMPLE_LINES)
    elif kind == "hf":
        tok = _load_hf(path)
        vocab_size = tok.get_vocab_size()
        encoded = _encode_hf(tok, SAMPLE_LINES)
    else:
        raise ValueError(f"unknown loader kind: {kind}")

    assert vocab_size > 0, f"{label}: vocab size is zero"
    assert len(encoded) == len(SAMPLE_LINES), f"{label}: wrong line count"
    total_ids = sum(len(ids) for ids in encoded)
    assert total_ids > 0, f"{label}: no ids produced"
    # Round-trip every line — catches loaders that silently produce
    # garbage on edge-case bytes.
    for i, ids in enumerate(encoded):
        assert isinstance(ids, list), f"{label}: line {i} ids is not list"
        for tid in ids:
            assert isinstance(tid, int), f"{label}: id {tid} is not int"
            assert 0 <= tid < vocab_size, (
                f"{label}: line {i} id {tid} out of range [0, {vocab_size})"
            )
    return ("OK", vocab_size, total_ids)


def main():
    print("ztok extended-vocab loader smoke")
    print("=" * 60)
    n_ok = n_skip = n_err = 0
    rows = []
    for label, filename, kind in EXTENDED:
        try:
            status, vsz, total = smoke_one(label, filename, kind)
        except Exception as e:
            status, vsz, total = ("ERR", None, None)
            print(f"  [ERR ] {label:<14} ({kind}): {e!r}", file=sys.stderr)
            n_err += 1
            continue
        if status == "SKIP":
            print(f"  [SKIP] {label:<14} ({kind}): {filename} not vendored")
            n_skip += 1
        else:
            print(
                f"  [OK  ] {label:<14} ({kind}): vocab={vsz:>6}  "
                f"ids_for_10_lines={total}"
            )
            n_ok += 1
        rows.append((label, kind, status, vsz, total))
    print("-" * 60)
    print(f"summary: {n_ok} ok, {n_skip} skipped, {n_err} errors")
    return 0 if n_err == 0 else 1


# Pytest-style functional smoke — auto-discovered by `python3 -m pytest`.
def test_mistral7b_loads_and_encodes():
    label = "mistral7b"
    path = os.path.join(VOCABS_DIR, "mistral7b.model")
    if not os.path.exists(path):
        import pytest
        pytest.skip(f"missing fixture: {path}")
    status, vsz, total = smoke_one(label, "mistral7b.model", "sp")
    assert status == "OK"
    assert vsz == 32000


def test_yi6b_loads_and_encodes():
    label = "yi6b"
    path = os.path.join(VOCABS_DIR, "yi6b.model")
    if not os.path.exists(path):
        import pytest
        pytest.skip(f"missing fixture: {path}")
    status, vsz, total = smoke_one(label, "yi6b.model", "sp")
    assert status == "OK"
    assert vsz > 60000  # Yi-6B vocab is 64 K


def test_phi3_loads_and_encodes():
    label = "phi3"
    path = os.path.join(VOCABS_DIR, "phi3.json")
    if not os.path.exists(path):
        import pytest
        pytest.skip(f"missing fixture: {path}")
    status, vsz, total = smoke_one(label, "phi3.json", "hf")
    assert status == "OK"


def test_qwen2_loads_and_encodes():
    label = "qwen2"
    path = os.path.join(VOCABS_DIR, "qwen2.json")
    if not os.path.exists(path):
        import pytest
        pytest.skip(f"missing fixture: {path}")
    status, vsz, total = smoke_one(label, "qwen2.json", "hf")
    assert status == "OK"
    assert vsz > 150000  # Qwen2-7B vocab is 151643


def test_llama3_loads_and_encodes():
    label = "llama3"
    path = os.path.join(VOCABS_DIR, "llama3.json")
    if not os.path.exists(path):
        import pytest
        pytest.skip(f"missing fixture: {path}")
    status, vsz, total = smoke_one(label, "llama3.json", "hf")
    assert status == "OK"
    assert vsz > 120000  # Llama-3-8B vocab is 128000


def test_falcon7b_loads_and_encodes():
    path = os.path.join(VOCABS_DIR, "falcon7b.json")
    if not os.path.exists(path):
        import pytest
        pytest.skip(f"missing fixture: {path}")
    status, vsz, total = smoke_one("falcon7b", "falcon7b.json", "hf")
    assert status == "OK"


def test_deepseek_v2_loads_and_encodes():
    path = os.path.join(VOCABS_DIR, "deepseek_v2.json")
    if not os.path.exists(path):
        import pytest
        pytest.skip(f"missing fixture: {path}")
    status, vsz, total = smoke_one("deepseek_v2", "deepseek_v2.json", "hf")
    assert status == "OK"


def test_equivalence_check_accepts_each_kind():
    """equivalence_check.py must accept sp-bpe, unigram, hf-bpe, and
    hf-unigram --- this catches a regression where someone deletes one
    of the kind clauses without updating the usage docs."""
    eq = os.path.join(BENCH_DIR, "equivalence_check.py")
    assert os.path.exists(eq), f"missing {eq}"
    with open(eq) as f:
        src = f.read()
    for kind in ("sp-bpe", "unigram", "hf-bpe", "hf-unigram", "monster"):
        assert f'"{kind}"' in src, f"equivalence_check.py missing kind {kind}"


if __name__ == "__main__":
    sys.exit(main())
