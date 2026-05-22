#!/usr/bin/env python3
"""Encoding-equivalence checker for the cross-tokenizer benchmark.

Runs ztok's bench_cross AND the chosen reference tool on the same N
lines of a corpus, collecting the per-line id streams via the
`--dump-sample` option each tool exposes. Reports how many lines agree
exactly.

This is the only correctness gate we have for the ztok TokenMonster /
SentencePiece loaders — throughput numbers are meaningless if the ids
diverge wildly.

Usage:
  # Back-compat positional form (uses defaults from env: ZTOK_BENCH_CORPUS,
  # ZTOK_BENCH_LINES — default /tmp/corpus.txt, 100 lines).
  bench/equivalence_check.py monster bench/vocabs/tm_englishcode_32k
  bench/equivalence_check.py sp-bpe  bench/vocabs/llama2

  # 1.21 stress gate flags:
  bench/equivalence_check.py sp-bpe bench/vocabs/llama2 \\
      --corpus bench/corpora/english.txt --lines 10000
  bench/equivalence_check.py hf-bpe bench/vocabs/gpt2_hf \\
      --corpus bench/corpora/unicode_stress.txt --lines 1000 \\
      --first-diff-only --json

Where each kind expects a vocab pair: ztok loads `<basename>.ztm` (TM)
or `<basename>.model` (SP); the reference tool loads the corresponding
native vocab.
"""
import argparse, json, os, subprocess, sys


CORPUS = os.environ.get(
    "ZTOK_BENCH_CORPUS",
    "/tmp/corpus.txt",
)
LINES = int(os.environ.get("ZTOK_BENCH_LINES", "100"))
ZTOK_BENCH = os.environ.get(
    "ZTOK_BENCH_CROSS",
    "zig-out/bin/bench_cross",
)


def parse_dump(stderr: bytes) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for line in stderr.decode("utf-8", errors="replace").splitlines():
        if not line.startswith("LINE "):
            continue
        parts = line.split()
        idx = int(parts[1])
        ids = [int(x) for x in parts[2:]]
        out[idx] = ids
    return out


def parse_norm_dump(stderr: bytes) -> dict[int, bytes]:
    """Parse `NORM <idx> <hex>` lines from bench_cross --dump-normalized."""
    out: dict[int, bytes] = {}
    for line in stderr.decode("utf-8", errors="replace").splitlines():
        if not line.startswith("NORM "):
            continue
        parts = line.split(maxsplit=2)
        if len(parts) < 3:
            continue
        idx = int(parts[1])
        try:
            out[idx] = bytes.fromhex(parts[2])
        except ValueError:
            continue
    return out


def run(cmd, env=None) -> tuple[bytes, bytes]:
    p = subprocess.run(cmd, capture_output=True, env=env)
    return p.stdout, p.stderr


def _tekken_reference(
    vocab_path: str,
    corpus_path: str,
    lines_n: int,
) -> tuple[str, dict[int, list[int]] | None]:
    """Encode the first `lines_n` lines of `corpus_path` through a
    reference Tekken implementation. Returns (ref_name, ids_by_line).

    Strategy:
      1) Prefer `mistral_common`'s `Tekkenizer.from_file` — that's the
         official Mistral implementation and is what ztok aims to match
         id-for-id on the Tekken family.
      2) Fall back to a hand-rolled tiktoken `Encoding` constructed from
         the same `tekken.json` (base64 vocab + `pattern` from `config`).
         Mistral's encoder is itself a `tiktoken.Encoding` under the hood
         once you peel away the JSON; the only subtlety is the special-id
         shift (Mistral packs specials at the BOTTOM of the id space and
         shifts the regular vocab up by `num_special_tokens`) which we
         apply here too so the comparison stays apples-to-apples with the
         ztok loader.

    Returns (ref_name, None) when neither backend is installed.
    """
    # Path 1: official mistral_common.
    try:
        from mistral_common.tokens.tokenizers.tekken import Tekkenizer  # type: ignore[import-not-found]
    except ImportError:
        Tekkenizer = None  # type: ignore[assignment]

    if Tekkenizer is not None:
        tok = Tekkenizer.from_file(vocab_path)
        out: dict[int, list[int]] = {}
        with open(corpus_path, "rb") as f:
            for i, raw in enumerate(f):
                if i >= lines_n:
                    break
                line = raw.rstrip(b"\n").decode("utf-8", errors="replace")
                # Tekkenizer.encode signature: encode(text, bos, eos).
                # We pass bos=False, eos=False so the id stream is
                # comparable to ztok's bare model encode.
                ids = tok.encode(line, bos=False, eos=False)
                out[i] = list(ids)
        return ("mistral_common", out)

    # Path 2: tiktoken fallback.
    try:
        import tiktoken  # type: ignore[import-not-found]
    except ImportError:
        return ("tekken-ref-missing", None)

    import base64 as _b64
    with open(vocab_path, "rb") as f:
        spec = json.load(f)
    cfg = spec.get("config", {})
    pattern = cfg.get("pattern")
    num_special = int(cfg.get("default_num_special_tokens", 0))
    if not pattern:
        return ("tiktoken", None)

    # Build the {bytes: rank} map. Tekken stores `token_bytes` as a
    # base64 string. Ranks in the file are 0..N_voc-1; we shift them up
    # by num_special so the id space matches Mistral's runtime layout
    # (specials live at ids [0, num_special), vocab at [num_special, ...)).
    mergeable: dict[bytes, int] = {}
    for entry in spec.get("vocab", []):
        rank = int(entry["rank"])
        tb = _b64.b64decode(entry["token_bytes"])
        mergeable[tb] = rank + num_special

    # Specials map: name -> id (already in the bottom of the id space).
    specials_map: dict[str, int] = {}
    for entry in spec.get("special_tokens", []) or []:
        specials_map[entry["token_str"]] = int(entry["rank"])

    enc = tiktoken.Encoding(
        name="ztok-tekken-shim",
        pat_str=pattern,
        mergeable_ranks=mergeable,
        special_tokens=specials_map,
    )
    out: dict[int, list[int]] = {}
    with open(corpus_path, "rb") as f:
        for i, raw in enumerate(f):
            if i >= lines_n:
                break
            line = raw.rstrip(b"\n").decode("utf-8", errors="replace")
            # disallowed_special=() so literal special-token strings in
            # the corpus don't raise — we let them resolve through the
            # specials_map if they match, otherwise they're plain text.
            ids = enc.encode(line, allowed_special=set(specials_map.keys()), disallowed_special=())
            out[i] = list(ids)
    return ("tiktoken", out)


def cmp_streams(
    name: str,
    a: dict,
    b: dict,
    total: int,
    *,
    first_diff_only: bool = False,
    max_diff_print: int = 3,
) -> dict:
    """Compare two dict-of-ids streams. Returns a stats dict:
        {matches, diffs, total, first_diff_idx, first_diff_a, first_diff_b}
    """
    matches = 0
    diff_count = 0
    first_diff_idx: int | None = None
    first_diff_a: list | None = None
    first_diff_b: list | None = None
    for i in range(total):
        if a.get(i) == b.get(i):
            matches += 1
        else:
            diff_count += 1
            if first_diff_idx is None:
                first_diff_idx = i
                first_diff_a = list(a.get(i, []))
                first_diff_b = list(b.get(i, []))
                if first_diff_only:
                    # Still need to count the rest of the matches/diffs
                    # so the stats are accurate, but skip the per-line
                    # print noise.
                    pass
            if diff_count <= max_diff_print:
                print(f"  diff line {i}:", file=sys.stderr)
                print(f"    ztok  ({len(a.get(i, []))} ids): {a.get(i)}", file=sys.stderr)
                print(f"    {name} ({len(b.get(i, []))} ids): {b.get(i)}", file=sys.stderr)
    return {
        "matches": matches,
        "diffs": total - matches,
        "total": total,
        "first_diff_idx": first_diff_idx,
        "first_diff_a": first_diff_a,
        "first_diff_b": first_diff_b,
    }


def emit_summary(
    kind: str,
    base: str,
    ref_name: str,
    corpus: str,
    lines_requested: int,
    stats: dict,
    *,
    as_json: bool,
) -> None:
    """Single source of truth for summary output. Plain text on stderr by
    default (back-compat with the existing harness output); machine-
    readable JSON on stdout when --json is passed."""
    pct = 100.0 * stats["matches"] / stats["total"] if stats["total"] else 0.0
    if as_json:
        rec = {
            "kind": kind,
            "vocab": base,
            "reference": ref_name,
            "corpus": corpus,
            "lines_requested": lines_requested,
            "lines_compared": stats["total"],
            "matches": stats["matches"],
            "diffs": stats["diffs"],
            "match_rate": pct / 100.0,
            "first_diff_idx": stats["first_diff_idx"],
            "first_diff_ztok": stats["first_diff_a"],
            "first_diff_ref": stats["first_diff_b"],
        }
        print(json.dumps(rec))
    print(
        f"[{kind}] ztok vs {ref_name}: {stats['matches']}/{stats['total']} "
        f"lines match ({pct:.1f}%) over {lines_requested}-line sample of {corpus}",
        file=sys.stderr,
    )


def main():
    # Back-compat: when the first positional arg looks like a kind, fall
    # through to argparse with the remaining args. Argparse handles both
    # `kind base ...` (positional) and the new --corpus/--lines/--first-
    # diff-only flags.
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("kind",
                    choices=["monster", "sp-bpe", "unigram", "hf-unigram",
                             "hf-bpe", "hf-wordpiece", "tekken"])
    ap.add_argument("base",
                    help="vocab basename (ztok reads <base>.ztm/.model/.json)")
    ap.add_argument("--corpus", default=CORPUS,
                    help=f"corpus path (default: {CORPUS} or $ZTOK_BENCH_CORPUS)")
    ap.add_argument("--lines", type=int, default=LINES,
                    help=f"number of lines to compare (default: {LINES} or $ZTOK_BENCH_LINES)")
    ap.add_argument("--first-diff-only", action="store_true",
                    help="suppress per-diff noise; emit only the first divergence sample")
    ap.add_argument("--json", action="store_true",
                    help="emit a single JSON summary record on stdout in addition to the human line on stderr")
    args = ap.parse_args()

    kind = args.kind
    base = args.base
    corpus = args.corpus
    lines_n = args.lines
    first_diff_only = args.first_diff_only
    as_json = args.json
    max_diff_print = 1 if first_diff_only else 3

    # Resolve vocab paths.
    if kind == "monster":
        ztok_vocab = base + ".ztm"
        ref_vocab = base + ".vocab"
        if not os.path.exists(ref_vocab):
            # Allow caller to point at a TM cache name.
            ref_vocab = base
    elif kind in ("sp-bpe", "unigram"):
        # Both SP-BPE and Unigram ride on the same .model proto format.
        ztok_vocab = base + ".model"
        ref_vocab = base + ".model"
    elif kind == "hf-unigram":
        # HF tokenizer.json (Unigram + byte_fallback). The reference is
        # the `tokenizers` Python library's model.tokenize(text), which
        # we compare against ztok's hf_bridge.unigramFromHF result on
        # PRE-NORMALIZED text (HF normalizers for real-world Unigram
        # exports use Regex Replace that ztok doesn't apply).
        ztok_vocab = base + ".json"
        ref_vocab = base + ".json"
    elif kind == "hf-bpe":
        # HF tokenizer.json (BPE + ByteLevel pre-tok). The reference is
        # the `tokenizers` Python library's Tokenizer.encode(text), which
        # exercises the full HF pipeline (Normalizer + PreTokenizer +
        # Model + PostProcessor). ztok's hf_byte_level pre-tokenizer +
        # bpeFromHF + byte_level decoder reproduce the same chain. We
        # compare end-to-end ids on the raw input (no manual
        # normalization on either side — GPT-2 has no normalizer).
        ztok_vocab = base + ".json"
        ref_vocab = base + ".json"
    elif kind == "hf-wordpiece":
        # HF tokenizer.json (WordPiece + BertNormalizer). Compares the
        # NORMALIZED bytes from ztok's `normalizerFromHF` chain against
        # the upstream `tokenizers` library's BertNormalizer on the same
        # input. The WordPiece encoder + Bert pretokenizer aren't
        # exercised here — that's a separate end-to-end check once
        # ztok grows a Bert pretok. This is the "is the BertNormalizer
        # arm byte-correct?" gate for bert-base-uncased.
        ztok_vocab = base + ".json"
        ref_vocab = base + ".json"
    elif kind == "tekken":
        # Mistral Tekken (`tekken.json`). Both sides take the same file;
        # accept either a bare basename (we append `.json`) or a full
        # `.json` path so the call sites in the smoke/stress scripts can
        # pass `bench/vocabs/mistral_nemo_tekken.json` directly.
        if base.endswith(".json"):
            ztok_vocab = base
            ref_vocab = base
        else:
            ztok_vocab = base + ".json"
            ref_vocab = base + ".json"
    else:
        print("unknown kind:", kind, file=sys.stderr)
        return 2

    for p in (ztok_vocab, ref_vocab):
        if not os.path.exists(p):
            print(f"missing vocab: {p}", file=sys.stderr)
            return 1
    if not os.path.exists(corpus):
        print(f"missing corpus: {corpus}", file=sys.stderr)
        return 1

    # tekken path: encode the same sample lines on both sides and
    # compare the id streams. Reference encoder selection:
    #   1) `mistral_common.tokens.tokenizers.tekken.Tekkenizer.from_file`
    #      is the canonical implementation Mistral ships, so we prefer it.
    #   2) Fall back to `tiktoken` with the raw vocab + the `pattern`
    #      field lifted out of the tekken.json `config` block. The BPE
    #      layer is bit-identical to tiktoken (same merge-by-rank loop,
    #      same byte_fallback), so given the right regex this fallback
    #      also reproduces the official encoder. Ids are returned in
    #      Mistral's id space (offset by `num_special_tokens`) to match
    #      what the ztok loader exposes.
    if kind == "tekken":
        ref_name, ref_lines = _tekken_reference(ref_vocab, corpus, lines_n)
        if ref_lines is None:
            print(
                "FAIL: neither mistral_common nor tiktoken is installed; "
                "`pip install mistral-common` (preferred) or `pip install tiktoken`",
                file=sys.stderr,
            )
            return 1
        _, z_err = run([
            ZTOK_BENCH,
            "--kind", "tekken", "--model", ztok_vocab,
            "--corpus", corpus, "--iters", "1",
            "--dump-sample", "--sample-lines", str(lines_n),
        ])
        ztok_ids = parse_dump(z_err)
        if not ztok_ids:
            print("FAIL: ztok dump empty", file=sys.stderr)
            return 1
        total = min(max(ztok_ids), max(ref_lines)) + 1
        stats = cmp_streams(ref_name, ztok_ids, ref_lines, total,
                            first_diff_only=first_diff_only,
                            max_diff_print=max_diff_print)
        emit_summary(kind, base, ref_name, corpus, lines_n, stats,
                     as_json=as_json)
        return 0

    # hf-wordpiece path: dump NORMALIZED bytes (not ids) on the ztok
    # side and compare against the HF tokenizers library's normalizer
    # output (BertNormalizer for bert-base-uncased). The downstream
    # WordPiece encoder + Bert pretokenizer aren't compared end-to-end
    # here — this verifies the BertNormalizer arm of the new HF
    # normalizer chain is byte-correct.
    if kind == "hf-wordpiece":
        try:
            from tokenizers import Tokenizer
        except ImportError:
            print(
                "FAIL: hf-wordpiece needs `pip install tokenizers`",
                file=sys.stderr,
            )
            return 1
        _, z_err = run([
            ZTOK_BENCH,
            "--kind", kind, "--model", ztok_vocab,
            "--corpus", corpus, "--iters", "1",
            "--dump-normalized", "--sample-lines", str(lines_n),
        ])
        ztok_norm = parse_norm_dump(z_err)
        if not ztok_norm:
            print("FAIL: ztok normalized-bytes dump empty", file=sys.stderr)
            return 1
        tok = Tokenizer.from_file(ref_vocab)
        ref_norm: dict[int, bytes] = {}
        with open(corpus, "rb") as f:
            for i, raw in enumerate(f):
                if i >= lines_n:
                    break
                line = raw.rstrip(b"\n").decode("utf-8", errors="replace")
                # tokenizers exposes `normalizer.normalize_str` which
                # runs the full HF normalizer chain. For bert-base-
                # uncased that's BertNormalizer (no Sequence wrap).
                if tok.normalizer is not None:
                    out = tok.normalizer.normalize_str(line)
                else:
                    out = line
                ref_norm[i] = out.encode("utf-8")
        total = min(max(ztok_norm) if ztok_norm else 0,
                    max(ref_norm) if ref_norm else 0) + 1
        matches = 0
        diffs = 0
        first_diff_idx: int | None = None
        first_diff_a: bytes | None = None
        first_diff_b: bytes | None = None
        for i in range(total):
            a = ztok_norm.get(i, b"")
            b = ref_norm.get(i, b"")
            if a == b:
                matches += 1
            else:
                diffs += 1
                if first_diff_idx is None:
                    first_diff_idx = i
                    first_diff_a = a
                    first_diff_b = b
                if diffs <= max_diff_print:
                    print(f"  diff line {i}:", file=sys.stderr)
                    print(f"    ztok ({len(a)}B): {a!r}", file=sys.stderr)
                    print(f"    hf   ({len(b)}B): {b!r}", file=sys.stderr)
        stats = {
            "matches": matches,
            "diffs": diffs,
            "total": total,
            "first_diff_idx": first_diff_idx,
            "first_diff_a": first_diff_a.hex() if first_diff_a else None,
            "first_diff_b": first_diff_b.hex() if first_diff_b else None,
        }
        emit_summary(kind, base, "hf-tokenizers (normalizer-only)",
                     corpus, lines_n, stats, as_json=as_json)
        return 0

    # Run ztok dump.
    _, z_err = run([
        ZTOK_BENCH,
        "--kind", kind, "--model", ztok_vocab,
        "--corpus", corpus, "--iters", "1",
        "--dump-sample", "--sample-lines", str(lines_n),
    ])
    ztok_ids = parse_dump(z_err)

    # Run reference dump.
    if kind == "monster":
        _, r_err = run([
            sys.executable, "bench/bench_competitors.py",
            "--corpus", corpus, "--iters", "1",
            "--lib", "tokenmonster",
            "--vocab", ref_vocab,
            "--dump-sample", str(lines_n),
        ])
        ref_name = "tm-go"
    elif kind in ("sp-bpe", "unigram"):
        # `bench_competitors.py --lib sentencepiece` defers to SP's Python
        # binding, which honors whichever model_type the .model encodes
        # (BPE or Unigram). So one command handles both kinds.
        _, r_err = run([
            sys.executable, "bench/bench_competitors.py",
            "--corpus", corpus, "--iters", "1",
            "--lib", "sentencepiece",
            "--vocab", ref_vocab,
            "--dump-sample", str(lines_n),
        ])
        ref_name = "sp-python"
    elif kind == "hf-bpe":
        # HF reference: tokenizers library's full-pipeline encoder. GPT-2
        # has no normalizer, so the only state on the ref side is the
        # ByteLevel pre-tokenizer + BPE model + ByteLevel post-processor
        # (which is identity on ids). Compare ztok's full-chain output
        # line-by-line.
        try:
            from tokenizers import Tokenizer
        except ImportError:
            print(
                "FAIL: hf-bpe needs `pip install tokenizers`",
                file=sys.stderr,
            )
            return 1
        tok = Tokenizer.from_file(ref_vocab)
        ref_lines: dict[int, list[int]] = {}
        with open(corpus, "rb") as f:
            for i, raw in enumerate(f):
                if i >= lines_n:
                    break
                line = raw.rstrip(b"\n").decode("utf-8", errors="replace")
                ref_lines[i] = tok.encode(line, add_special_tokens=False).ids
        ref_name = "hf-tokenizers"
        ref_ids = ref_lines
        if not ztok_ids:
            print("FAIL: ztok dump empty", file=sys.stderr)
            return 1
        total = min(max(ztok_ids), max(ref_ids)) + 1
        stats = cmp_streams(ref_name, ztok_ids, ref_ids, total,
                            first_diff_only=first_diff_only,
                            max_diff_print=max_diff_print)
        emit_summary(kind, base, ref_name, corpus, lines_n, stats,
                     as_json=as_json)
        return 0
    elif kind == "hf-unigram":
        # HF reference: tokenizers library's model-level encoder. The
        # ztok side runs with the identity normalizer; we pre-normalize
        # here too (escape spaces to U+2581, add leading dummy prefix)
        # so the comparison is apples-to-apples at the model level.
        try:
            from tokenizers import Tokenizer
        except ImportError:
            print(
                "FAIL: hf-unigram needs `pip install tokenizers`",
                file=sys.stderr,
            )
            return 1
        tok = Tokenizer.from_file(ref_vocab)
        ref_lines: dict[int, list[int]] = {}
        with open(corpus, "rb") as f:
            for i, raw in enumerate(f):
                if i >= lines_n:
                    break
                # Strip the trailing newline so the ztok side (which
                # splits on \n) gets the same input.
                line = raw.rstrip(b"\n").decode("utf-8", errors="replace")
                # Mirror SP-style pre-normalization minimally: spaces
                # become U+2581, and a leading U+2581 is prepended (the
                # llm-jp normalizer does both via Regex Replace).
                norm = "▁" + line.replace(" ", "▁") if line else ""
                ids = [t.id for t in tok.model.tokenize(norm)]
                ref_lines[i] = ids
        # The ztok side must encode the SAME normalized text. We invoke
        # bench_cross with a CORPUS that has been pre-normalized into a
        # temp file (so the existing --dump-sample path Just Works).
        import tempfile
        tf = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        try:
            with open(corpus, "rb") as f:
                for i, raw in enumerate(f):
                    if i >= lines_n:
                        break
                    line = raw.rstrip(b"\n").decode("utf-8", errors="replace")
                    norm = "▁" + line.replace(" ", "▁") if line else ""
                    tf.write(norm + "\n")
            tf.flush()
            tf.close()
            _, z_err2 = run([
                ZTOK_BENCH,
                "--kind", "hf-unigram", "--model", ztok_vocab,
                "--corpus", tf.name, "--iters", "1",
                "--dump-sample", "--sample-lines", str(lines_n),
            ])
            ztok_ids = parse_dump(z_err2)
        finally:
            os.unlink(tf.name)
        # Re-purpose the existing comparison below.
        ref_name = "hf-tokenizers"
        ref_ids = ref_lines
        # Short-circuit: we don't go through bench_competitors.py for
        # hf-unigram, so jump straight to the compare step.
        if not ztok_ids:
            print("FAIL: ztok dump empty", file=sys.stderr)
            return 1
        total = min(max(ztok_ids), max(ref_ids)) + 1
        stats = cmp_streams(ref_name, ztok_ids, ref_ids, total,
                            first_diff_only=first_diff_only,
                            max_diff_print=max_diff_print)
        emit_summary(kind, base, ref_name, corpus, lines_n, stats,
                     as_json=as_json)
        return 0
    ref_ids = parse_dump(r_err)

    if not ztok_ids:
        print("FAIL: ztok dump empty", file=sys.stderr)
        return 1
    if not ref_ids:
        print("FAIL: reference dump empty", file=sys.stderr)
        return 1

    total = min(max(ztok_ids), max(ref_ids)) + 1
    stats = cmp_streams(ref_name, ztok_ids, ref_ids, total,
                        first_diff_only=first_diff_only,
                        max_diff_print=max_diff_print)
    emit_summary(kind, base, ref_name, corpus, lines_n, stats, as_json=as_json)
    # Always exit 0 — we report the match rate; the caller decides
    # whether it's acceptable (cross-vocab encoders are expected to
    # diverge on capcode / normalization quirks).
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
