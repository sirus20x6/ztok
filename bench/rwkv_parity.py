#!/usr/bin/env python3
"""RWKV "World" tokenizer bit-parity gate.

Cross-checks ztok's greedy longest-match byte tokenizer (the
`--format rwkv` model family) against an INDEPENDENT clean-room
reference encoder over the canonical `rwkv_vocab_v20230424.txt`.

The reference here is a from-scratch longest-match implementation (a
true second implementation, not a copy of BlinkDL's reference code), so
agreement is meaningful evidence of correctness rather than a shared
bug. The RWKV World scheme is unambiguous: at each byte position pick
the longest vocab entry that is a prefix of the remaining bytes; the
vocab contains all 256 single bytes, so a match always exists (no UNK).

Usage:
  python3 bench/rwkv_parity.py                 # sample texts + corpora
  python3 bench/rwkv_parity.py --emit-golden   # print Zig-test goldens
  python3 bench/rwkv_parity.py --fetch          # download vocab if absent

Exits nonzero on any mismatch (suitable as a CI gate). Skips cleanly
(exit 0) when neither the vocab fixture nor a fetch is available.
"""

import ast
import os
import subprocess
import sys
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOCAB = os.path.join(REPO, "bench", "vocabs", "rwkv_vocab_v20230424.txt")
ZTOK = os.path.join(REPO, "zig-out", "bin", "ztok")
VOCAB_URL = (
    "https://raw.githubusercontent.com/BlinkDL/ChatRWKV/"
    "main/tokenizer/rwkv_vocab_v20230424.txt"
)

# A spread of scripts and edge cases for the small in-line gate. The
# corpora under bench/corpora/ provide the broad sweep.
SAMPLES = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "  leading spaces and 12345 numbers 007",
    "def foo(x):\n    return x + 1\n",
    "起業家イーロン・マスク氏が創業した宇宙開発企業",
    "日本語のテキスト、句読点。",
    "emoji 😀🚀✨ test",
    "Café naïve résumé — em dash",
    "مرحبا بالعالم",
    "नमस्ते दुनिया",
    "tab\tand\nnewline\r\nend",
    "0 1 2 10 99 100 ' 0' ' 99'",
    "mixed CASE Words With Punctuation!?;:",
    "x",
]


def parse_vocab(path):
    """Parse `<id> <python-repr> <byte-len>` lines into {bytes: id}."""
    tok2id = {}
    maxlen = 1
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sp = line.index(" ")
            rsp = line.rindex(" ")
            idx = int(line[:sp])
            x = ast.literal_eval(line[sp:rsp])
            b = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(b, (bytes, bytearray))
            assert len(b) == int(line[rsp:]), f"len mismatch on id {idx}"
            tok2id[bytes(b)] = idx
            maxlen = max(maxlen, len(b))
    return tok2id, maxlen


def ref_encode(src_bytes, tok2id, maxlen):
    """Independent greedy longest-match reference."""
    out = []
    i, n = 0, len(src_bytes)
    while i < n:
        hi = min(maxlen, n - i)
        for L in range(hi, 0, -1):
            piece = src_bytes[i : i + L]
            tid = tok2id.get(piece)
            if tid is not None:
                out.append(tid)
                i += L
                break
        else:  # pragma: no cover - vocab always has single bytes
            raise AssertionError(f"no token match at byte {i}")
    return out


def ztok_encode(text):
    p = subprocess.run(
        [ZTOK, "encode", "--format", "rwkv", "--vocab", VOCAB],
        input=text.encode("utf-8"),
        capture_output=True,
    )
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode(errors="replace"))
    return [int(t) for t in p.stdout.split()]


def main():
    args = set(sys.argv[1:])

    if not os.path.exists(VOCAB):
        if "--fetch" in args:
            print(f"fetching {VOCAB_URL}", file=sys.stderr)
            urllib.request.urlretrieve(VOCAB_URL, VOCAB)
        else:
            print("rwkv vocab fixture absent; skipping (pass --fetch to download)")
            return 0

    tok2id, maxlen = parse_vocab(VOCAB)
    print(f"loaded {len(tok2id)} tokens, max len {maxlen}")

    if "--emit-golden" in args:
        # Emit Zig-test golden lines for the in-tree parity gate.
        for s in SAMPLES:
            ids = ref_encode(s.encode("utf-8"), tok2id, maxlen)
            print(f"// {s!r}")
            print(".{ .input = " + zig_str(s) + ", .want = &.{ "
                  + ", ".join(map(str, ids)) + " } },")
        return 0

    if not os.path.exists(ZTOK):
        print(f"ztok binary not built at {ZTOK}; run `zig build`", file=sys.stderr)
        return 1

    inputs = list(SAMPLES)
    corp_dir = os.path.join(REPO, "bench", "corpora")
    for name in ("english.txt", "code.txt", "multilingual.txt",
                 "chat.txt", "unicode_stress.txt"):
        p = os.path.join(corp_dir, name)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                # cap each corpus to keep the subprocess sweep quick
                inputs.append(f.read(200_000))

    ok = fail = 0
    for text in inputs:
        ref = ref_encode(text.encode("utf-8"), tok2id, maxlen)
        got = ztok_encode(text)
        if got == ref:
            ok += 1
        else:
            fail += 1
            # report the first divergence
            j = next((k for k in range(min(len(ref), len(got)))
                      if ref[k] != got[k]), min(len(ref), len(got)))
            print(f"MISMATCH (len ref={len(ref)} got={len(got)}) "
                  f"first diff at token {j}: ref={ref[j-1:j+2]} got={got[j-1:j+2]}")
            print(f"  input prefix: {text[:80]!r}")

    print(f"\n{ok}/{ok + fail} inputs match the reference")
    return 0 if fail == 0 else 1


def zig_str(s):
    """Render a Python str as a Zig double-quoted string literal."""
    out = ['"']
    for ch in s:
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\r":
            out.append("\\r")
        else:
            out.append(ch)  # UTF-8 verbatim; Zig source is UTF-8
    out.append('"')
    return "".join(out)


if __name__ == "__main__":
    sys.exit(main())
