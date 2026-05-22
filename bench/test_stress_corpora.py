#!/usr/bin/env python3
"""Tests for the 1.21 10K-line stress equivalence harness.

Three tests, mirroring the spec from agent D:

  1. equivalence_check.py --corpus english.txt --lines 100 runs cleanly.
  2. The 5 new corpus files exist and are non-empty.
  3. A specific Unicode-stress 4-byte emoji + combining mark round-trips
     bit-identical between tiktoken cl100k and ztok.

Run:
    python3 bench/test_stress_corpora.py
    python3 -m pytest bench/test_stress_corpora.py
"""
import json
import os
import shutil
import subprocess
import sys

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BENCH_DIR, ".."))
CORPORA_DIR = os.path.join(BENCH_DIR, "corpora")
VOCABS_DIR = os.path.join(BENCH_DIR, "vocabs")
BENCH_CROSS = os.path.join(REPO_ROOT, "zig-out", "bin", "bench_cross")

CORPUS_FILES = (
    "english.txt",
    "code.txt",
    "multilingual.txt",
    "chat.txt",
    "unicode_stress.txt",
)


def test_corpora_files_exist_and_nonempty():
    """All 5 vendored corpus files must be present and non-empty after
    `bench/corpora/build_corpora.py` is run."""
    for name in CORPUS_FILES:
        path = os.path.join(CORPORA_DIR, name)
        assert os.path.exists(path), f"missing corpus: {path}"
        sz = os.path.getsize(path)
        assert sz > 0, f"empty corpus: {path} ({sz} bytes)"
        # Sanity: line count should be >= 1000 for unicode_stress, >= 10K for others.
        expected_min_lines = 1000 if name == "unicode_stress.txt" else 10000
        with open(path, "rb") as f:
            n_lines = sum(1 for _ in f)
        assert n_lines >= expected_min_lines, (
            f"{path}: only {n_lines} lines (expected >= {expected_min_lines})"
        )


def test_equivalence_check_runs_cleanly_on_english_100():
    """The harness should accept --corpus / --lines flags and complete
    cleanly with a known-good fixture (LLaMA-2 SP-BPE) at 100 lines of
    English prose. Exit code must be 0 and the human summary line must
    include the match-rate."""
    if not os.path.exists(BENCH_CROSS):
        import pytest
        pytest.skip(f"missing {BENCH_CROSS} (run `zig build`)")
    if not os.path.exists(os.path.join(VOCABS_DIR, "llama2.model")):
        import pytest
        pytest.skip("missing bench/vocabs/llama2.model")
    if not os.path.exists(os.path.join(CORPORA_DIR, "english.txt")):
        import pytest
        pytest.skip("missing bench/corpora/english.txt")
    try:
        import sentencepiece  # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("requires `pip install sentencepiece`")
    cmd = [
        sys.executable, os.path.join(BENCH_DIR, "equivalence_check.py"),
        "sp-bpe", "bench/vocabs/llama2",
        "--corpus", "bench/corpora/english.txt", "--lines", "100",
        "--json",
    ]
    p = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, timeout=60)
    assert p.returncode == 0, (
        f"equivalence_check.py exited {p.returncode}\n"
        f"  stdout: {p.stdout!r}\n  stderr: {p.stderr!r}"
    )
    # Stdout should be a single JSON record.
    rec = json.loads(p.stdout.strip().splitlines()[-1])
    assert rec["kind"] == "sp-bpe"
    assert rec["lines_compared"] == 100
    assert 0.0 <= rec["match_rate"] <= 1.0
    # Stderr should contain the human summary line.
    assert b"lines match" in p.stderr


def test_unicode_stress_4byte_emoji_roundtrips_on_cl100k():
    """A specific 4-byte astral emoji (U+1F600 GRINNING FACE) followed
    by a combining-acute mark should round-trip bit-identical between
    tiktoken cl100k and ztok's cl100k encoder. Catches regressions in
    UTF-8 handling for the astral plane + Mn category combiners."""
    try:
        import tiktoken
    except ImportError:
        import pytest
        pytest.skip("requires `pip install tiktoken`")
    if not shutil.which("zig"):
        import pytest
        pytest.skip("requires zig on PATH")
    ztok_cli = os.path.join(REPO_ROOT, "zig-out", "bin", "ztok")
    if not os.path.exists(ztok_cli):
        import pytest
        pytest.skip(f"missing {ztok_cli} (run `zig build`)")

    # Find a cl100k_base.tiktoken fixture. tiktoken caches it under
    # ~/.cache/tiktoken or /tmp/data-gym-cache, depending on env.
    enc = tiktoken.get_encoding("cl100k_base")
    cl100k_path = None
    candidates = [
        os.path.expanduser("~/.cache/tiktoken/9b5ad71b2ce5302211f9c61530b329a4922fc6a4"),
        "/tmp/cl100k_base.tiktoken",
        "/tmp/data-gym-cache/9b5ad71b2ce5302211f9c61530b329a4922fc6a4",
    ]
    for c in candidates:
        if os.path.exists(c):
            cl100k_path = c
            break
    if cl100k_path is None:
        import pytest
        pytest.skip("no cl100k_base.tiktoken cache found")

    # The adversarial input: a 4-byte emoji (U+1F600), then a combining
    # acute (U+0301) attached to "e", then a ZWJ family, then NBSP.
    # Every byte here exercises a different UTF-8 path:
    #   - U+1F600 -> 4-byte sequence (F0 9F 98 80)
    #   - e + U+0301 -> 1-byte base + 2-byte combiner
    #   - U+200D ZWJ -> 3-byte format char
    #   - U+00A0 NBSP -> 2-byte whitespace
    text = "\U0001F600é‍ test"
    ref_ids = enc.encode(text)

    # ztok cl100k encode via CLI.
    p = subprocess.run(
        [ztok_cli, "encode", "--model", cl100k_path, "--cl100k", text],
        capture_output=True, timeout=15,
    )
    assert p.returncode == 0, (
        f"ztok encode exited {p.returncode}: {p.stderr!r}"
    )
    # CLI emits one id per line.
    ztok_ids = [int(x) for x in p.stdout.decode().split() if x]
    assert ztok_ids == ref_ids, (
        f"cl100k 4-byte emoji + combining mark divergence:\n"
        f"  text   : {text!r} ({text.encode('utf-8').hex()})\n"
        f"  tiktoken: {ref_ids}\n"
        f"  ztok    : {ztok_ids}"
    )


if __name__ == "__main__":
    # CLI runner: each test individually so we report status per-test.
    import traceback
    tests = [
        test_corpora_files_exist_and_nonempty,
        test_equivalence_check_runs_cleanly_on_english_100,
        test_unicode_stress_4byte_emoji_roundtrips_on_cl100k,
    ]
    n_ok = n_skip = n_fail = 0
    for fn in tests:
        try:
            fn()
            print(f"  [OK  ] {fn.__name__}")
            n_ok += 1
        except SystemExit:
            raise
        except BaseException as e:
            if e.__class__.__name__ == "Skipped":
                # pytest.skip raises an exception named Skipped in pytest
                # context; mimic the same when run standalone.
                print(f"  [SKIP] {fn.__name__}: {e}")
                n_skip += 1
            else:
                print(f"  [FAIL] {fn.__name__}: {e}")
                traceback.print_exc()
                n_fail += 1
    print(f"summary: {n_ok} ok, {n_skip} skip, {n_fail} fail")
    sys.exit(0 if n_fail == 0 else 1)
