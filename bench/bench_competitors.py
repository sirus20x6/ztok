#!/usr/bin/env python3
"""Benchmark reference tokenizers (tiktoken / HF / TokenMonster /
SentencePiece) on a shared corpus.

The --lib choice picks which native tool to drive; ztok is benchmarked
separately via `zig build bench-cross -- ...` against the same corpus
bytes, and the throughput numbers go side-by-side in
`bench/RESULTS.md`.
"""
import argparse, time, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument(
        "--lib",
        choices=["tiktoken", "hf", "tokenmonster", "sentencepiece"],
        default="tiktoken",
    )
    ap.add_argument(
        "--threads",
        type=int,
        default=0,
        help=(
            "tiktoken: encode_ordinary_batch num_threads; "
            "hf: ignored; "
            "tokenmonster: ignored (no batch API); "
            "sentencepiece: ignored (no batch API in SP Python)"
        ),
    )
    ap.add_argument(
        "--vocab",
        default=None,
        help=(
            "tokenmonster: vocab name or .vocab path (default englishcode-32000-clean-nocapcode-v1); "
            "sentencepiece: .model path (required)"
        ),
    )
    ap.add_argument(
        "--dump-sample",
        type=int,
        default=0,
        help="dump the first N corpus lines + ids to stderr (for equivalence check)",
    )
    args = ap.parse_args()

    with open(args.corpus, "rb") as f:
        raw = f.read()
    data = raw.decode("utf-8")
    bytes_total_per_iter = len(raw)

    if args.lib == "tiktoken":
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        print(f"corpus:  {bytes_total_per_iter} bytes")
        print(f"vocab:   {enc.n_vocab} tokens (cl100k_base)")
        if args.threads > 0:
            n = args.threads
            chunk = len(data) // n
            chunks = [data[i*chunk:(i+1)*chunk] if i < n-1 else data[i*chunk:] for i in range(n)]
            t0 = time.perf_counter()
            total = 0
            for _ in range(args.iters):
                results = enc.encode_ordinary_batch(chunks, num_threads=n)
                total += sum(len(r) for r in results)
            elapsed = time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            total = 0
            for _ in range(args.iters):
                ids = enc.encode_ordinary(data)
                total += len(ids)
            elapsed = time.perf_counter() - t0
            if args.dump_sample > 0:
                _dump_sample_text(data, args.dump_sample, lambda s: enc.encode_ordinary(s))
        mode = f"batch, chunks={args.threads}" if args.threads > 0 else "single-thread"

    elif args.lib == "hf":
        from tokenizers import Tokenizer
        print("hf path: loading gpt2 tokenizer (different vocab from cl100k)", file=sys.stderr)
        tok = Tokenizer.from_pretrained("gpt2")
        print(f"corpus:  {bytes_total_per_iter} bytes")
        print(f"vocab:   {tok.get_vocab_size()} tokens (gpt2, NOT cl100k)")
        t0 = time.perf_counter()
        total = 0
        for _ in range(args.iters):
            enc = tok.encode(data, add_special_tokens=False)
            total += len(enc.ids)
        elapsed = time.perf_counter() - t0
        mode = "single-thread (HF tokenizers)"

    elif args.lib == "tokenmonster":
        # TokenMonster's Python API talks to a Go subprocess via a pipe
        # (tokenmonsterserver). The Go binary is the canonical reference
        # tokenizer — same code as the `go/tokenmonster.go` package.
        # Vocab is auto-downloaded by name or loaded from a local path.
        import tokenmonster
        vocab_name = args.vocab or "englishcode-32000-clean-nocapcode-v1"
        vocab = tokenmonster.load(vocab_name)
        print(f"corpus:  {bytes_total_per_iter} bytes")
        print(
            f"vocab:   {vocab.vocab_size} tokens "
            f"({vocab_name}; capcode={vocab.capcode()}; norm={vocab.normalization()})"
        )
        # TM Python only exposes single-text and list-of-text tokenize().
        # `threads` parameter would be size of list — emulate batch by
        # splitting the corpus into N pieces.
        if args.threads > 0:
            n = args.threads
            chunk = len(data) // n
            chunks_bytes = [
                data[i*chunk:(i+1)*chunk].encode("utf-8")
                if i < n-1 else data[i*chunk:].encode("utf-8")
                for i in range(n)
            ]
            t0 = time.perf_counter()
            total = 0
            for _ in range(args.iters):
                results = vocab.tokenize(chunks_bytes)
                total += sum(len(r) for r in results)
            elapsed = time.perf_counter() - t0
            mode = f"batch, chunks={n}"
        else:
            payload = data.encode("utf-8")
            t0 = time.perf_counter()
            total = 0
            for _ in range(args.iters):
                ids = vocab.tokenize(payload)
                total += len(ids)
            elapsed = time.perf_counter() - t0
            mode = "single-thread (tokenmonster-go subprocess)"
            if args.dump_sample > 0:
                def _tm_enc(b):
                    r = vocab.tokenize(b)
                    return list(r) if r is not None else []
                _dump_sample_bytes(raw, args.dump_sample, _tm_enc)
        tokenmonster.disconnect()

    elif args.lib == "sentencepiece":
        if not args.vocab:
            sys.exit("--vocab REQUIRED for sentencepiece (.model path)")
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=args.vocab)
        print(f"corpus:  {bytes_total_per_iter} bytes")
        print(f"vocab:   {sp.vocab_size()} tokens ({args.vocab})")
        # SP's Python wrapper supports `num_threads` only for batch
        # encode (list of strings); single-string encode is single-thread.
        if args.threads > 0:
            n = args.threads
            chunk = len(data) // n
            pieces = [
                data[i*chunk:(i+1)*chunk] if i < n-1 else data[i*chunk:]
                for i in range(n)
            ]
            t0 = time.perf_counter()
            total = 0
            for _ in range(args.iters):
                results = sp.encode(pieces, num_threads=n)
                total += sum(len(r) for r in results)
            elapsed = time.perf_counter() - t0
            mode = f"batch, chunks={n}"
        else:
            t0 = time.perf_counter()
            total = 0
            for _ in range(args.iters):
                ids = sp.encode(data)
                total += len(ids)
            elapsed = time.perf_counter() - t0
            mode = "single-thread (sentencepiece)"
            if args.dump_sample > 0:
                _dump_sample_text(
                    data, args.dump_sample,
                    lambda s: sp.encode(s),
                )

    bytes_total = bytes_total_per_iter * args.iters
    mb_per_s = bytes_total / elapsed / 1e6
    tok_per_s = total / elapsed
    bpt = bytes_total_per_iter / max(total // args.iters, 1)
    print(f"mode:    {mode}, iters={args.iters}")
    print(f"ids/run: {total // args.iters}")
    print(f"time:    {elapsed*1000:.2f} ms")
    print(f"MB/s:    {mb_per_s:.1f}")
    print(f"tok/s:   {tok_per_s:.0f}")
    print(f"bytes/tok: {bpt:.2f}")


def _dump_sample_text(data: str, n_lines: int, encode):
    """Encode the first n_lines lines and emit `LINE i id id ...` to stderr.

    Split on `\\n` only — NOT `splitlines()`, which also splits on
    U+2028/U+2029/U+0085. ztok's harness splits on `\\n` only; the
    equivalence_check matches outputs line-by-line, so using
    splitlines() here misaligned every line after the first
    U+2028/U+2029 in the corpus (visible on unicode_stress.txt as
    a ~4.8% match rate for SP fixtures even when ztok and SP-python
    actually agreed per-line — they just disagreed on what a "line"
    is).
    """
    for i, line in enumerate(data.split("\n")[:n_lines]):
        ids = encode(line)
        sys.stderr.write(
            "LINE " + str(i) + " " + " ".join(str(x) for x in ids) + "\n"
        )


def _dump_sample_bytes(raw: bytes, n_lines: int, encode):
    """Same but for tools that want raw bytes (TokenMonster).

    TM's tokenize() returns None on empty input; emit an empty id list
    so line indices stay aligned with the other tool's dump.
    """
    lines = raw.split(b"\n")[:n_lines]
    for i, line in enumerate(lines):
        ids = encode(line) or []
        sys.stderr.write(
            "LINE " + str(i) + " " + " ".join(str(x) for x in ids) + "\n"
        )


if __name__ == "__main__":
    main()
