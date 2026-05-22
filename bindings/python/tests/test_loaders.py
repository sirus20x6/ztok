"""Format-detection tests for each of the four supported loaders.

The cl100k/HF/SP/ztm format detection mirrors src/auto_detect.zig — we
synthesize a minimal file per format and check that
``Pipeline.from_path`` routes through the right C constructor.

The SentencePiece and Monster file builders rely on the ztok CLI to
produce real on-disk artifacts. If the CLI isn't available, those tests
are skipped (the format-detection logic itself is still exercised by the
auto-detect unit tests in test_basic.py / this file).
"""

from __future__ import annotations

import base64
import os
import shutil
import subprocess
from pathlib import Path

import pytest

import ztok
from ztok import _detect_format, _ffi
from ztok._ffi import (
    FORMAT_HF_JSON,
    FORMAT_SP_MODEL,
    FORMAT_TIKTOKEN,
    FORMAT_UNKNOWN,
    FORMAT_ZTM,
)


# --- format detection sanity checks --------------------------------------
#
# Post-1.18 agent C exposes `ztok_auto_detect` via the C ABI. The
# legacy `_detect_format` wrapper (string return) now routes through it;
# we also exercise the raw int-code path so signature drift surfaces
# immediately if the C enum ever changes.


def test_detect_tiktoken(tiktoken_path: Path) -> None:
    assert _detect_format(tiktoken_path) == "tiktoken"
    # Raw int-code path:
    lib = ztok._get_lib()
    import os as _os
    assert lib.ztok_auto_detect(_os.fsencode(str(tiktoken_path))) == FORMAT_TIKTOKEN


def test_detect_hf_json(tmp_path: Path) -> None:
    # Minimal HF tokenizer.json that hf_json.zig will parse: a BPE model
    # with an empty merges + vocab dict is enough for the loader smoke
    # test, but we don't actually load it here — just check detection.
    p = tmp_path / "tokenizer.json"
    p.write_text('{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]}}')
    assert _detect_format(p) == "hf_json"
    lib = ztok._get_lib()
    import os as _os
    assert lib.ztok_auto_detect(_os.fsencode(str(p))) == FORMAT_HF_JSON


def test_detect_sentencepiece(tmp_path: Path) -> None:
    # Real SP files start with the wire-format tag 0x0A followed by a
    # length byte. We use a sample from refs/sentencepiece if present.
    sp = Path("/thearray/git/ztok/bench/vocabs/llama2.model")
    if not sp.exists():
        pytest.skip("llama2.model fixture missing")
    assert _detect_format(sp) == "sentencepiece"
    lib = ztok._get_lib()
    import os as _os
    assert lib.ztok_auto_detect(_os.fsencode(str(sp))) == FORMAT_SP_MODEL


def test_detect_ztm(tmp_path: Path) -> None:
    p = tmp_path / "v.ztm"
    p.write_bytes(b"ZTM\x01" + b"\x00" * 60)
    assert _detect_format(p) == "ztm"
    lib = ztok._get_lib()
    import os as _os
    assert lib.ztok_auto_detect(_os.fsencode(str(p))) == FORMAT_ZTM


def test_detect_unknown_for_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "no_such_file.bin"
    assert _detect_format(missing) == "unknown"
    lib = ztok._get_lib()
    import os as _os
    assert lib.ztok_auto_detect(_os.fsencode(str(missing))) == FORMAT_UNKNOWN


# --- end-to-end loader dispatch ------------------------------------------


def test_from_path_loads_tiktoken(tiktoken_path: Path) -> None:
    with ztok.Pipeline.from_path(str(tiktoken_path)) as pipe:
        ids = pipe.encode("hello world")
        assert pipe.decode(ids) == "hello world"


def test_from_path_loads_sentencepiece() -> None:
    sp = Path("/thearray/git/ztok/bench/vocabs/llama2.model")
    if not sp.exists():
        pytest.skip("llama2.model fixture missing")
    # LLaMA-2 vocab uses id 0 as <unk>.
    with ztok.Pipeline.from_path(str(sp), unk_id=0) as pipe:
        ids = pipe.encode("hello world")
        assert len(ids) > 0


def test_from_path_loads_ztm() -> None:
    zm = Path("/thearray/git/ztok/bench/vocabs/tm_englishcode_32k.ztm")
    if not zm.exists():
        pytest.skip("tm_englishcode_32k.ztm fixture missing")
    with ztok.Pipeline.from_path(str(zm)) as pipe:
        ids = pipe.encode("hello world")
        assert len(ids) > 0


def test_from_path_unknown_format_raises(tmp_path: Path) -> None:
    p = tmp_path / "mystery.bin"
    p.write_bytes(b"\xff\xfe\xfd\xfc not a tokenizer")
    with pytest.raises(ztok.ZtokInvalidInputError):
        ztok.Pipeline.from_path(str(p))


def test_explicit_from_tiktoken_no_cl100k(tiktoken_path: Path) -> None:
    # When cl100k=False, the pre-tokenizer is identity, so "hello world"
    # is a single span (not split on whitespace) and may merge differently.
    with ztok.Pipeline.from_tiktoken(str(tiktoken_path), cl100k=False) as pipe:
        ids = pipe.encode("hello world")
        # Either way, the round-trip must hold.
        assert pipe.decode(ids) == "hello world"
