#!/usr/bin/env python3
"""Fetch the cross-tokenizer benchmark vocabs into bench/vocabs/.

Both files are small enough to vendor (~900 KB each) and are committed
to the repo. This script re-downloads them on a clean checkout where
the binaries weren't checked in (CI cache miss, etc.). It also converts
the TokenMonster .vocab into ztok's .ztm format on the fly.

Vocab sources:
  - TokenMonster: englishcode-32000-clean-nocapcode-v1 from the
    `tokenmonster` Python pip pkg (auto-downloads on first `load()`).
  - SentencePiece: LLaMA-2's tokenizer.model from a public HF mirror
    that doesn't require a gated download. We pick this over T5 because
    LLaMA-2 doesn't apply NFKC pre-tokenization, so the ztok vs SP
    apples-to-apples is closer to honest (SP still does dummy_prefix
    insertion that we skip — see bench/RESULTS.md "Cross-tokenizer
    benchmarks").

Extended set (post-1.17 agent E, `--extended` flag):
  Six widely-deployed real-world tokenizers covering both SP-BPE
  (.model) and HF JSON paths beyond the 1.17 baseline of LLaMA-2 /
  T5 / Gemma / llm-jp / GPT-2.

    - Mistral-7B  (SP-BPE .model, ~493 KB)
    - Yi-6B       (SP-BPE .model, ~1.0 MB)
    - Phi-3-mini  (HF tokenizer.json, ~1.9 MB)
    - Falcon-7B   (HF BPE byte_level tokenizer.json, ~2.7 MB)
    - DeepSeek-V2 (HF BPE tokenizer.json, ~4.6 MB)
    - Qwen2-7B    (HF BPE byte_level tokenizer.json, ~7.0 MB)
    - Llama-3-8B  (HF byte_level BPE tokenizer.json, ~9.1 MB, optional)

  Total ~25 MB for the six required, well inside the 50 MB bench/vocabs
  budget. Each download is non-fatal; equivalence_check.py /
  cli_bench.zig skip gracefully when a fixture is absent.
"""
import os, subprocess, sys, urllib.request

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
VOCABS_DIR = os.path.join(BENCH_DIR, "vocabs")

LLAMA2_URL = "https://huggingface.co/hf-internal-testing/llama-tokenizer/resolve/main/tokenizer.model"
LLAMA2_PATH = os.path.join(VOCABS_DIR, "llama2.model")

# Optional second SP vocab for the Unigram path. T5 uses NFKC + ▁
# replacement that ztok doesn't apply, so token streams diverge widely
# from SP — included anyway for the Unigram throughput half of the bench.
T5_URL = "https://huggingface.co/google-t5/t5-small/resolve/main/spiece.model"
T5_PATH = os.path.join(VOCABS_DIR, "t5_unigram.model")

# Optional third SP vocab for the casefold (_cf normalizer) + byte_fallback
# code paths. Gemma's SP tokenizer.model uses `nmt_nfkc_cf` + a 256-entry
# `<0xNN>` byte bank, exercising the post-1.14 agent B changes
# (SpNormalizer.casefold + Unigram.byte_fallback). Not required for the
# baseline bench — equivalence_check.py will skip if absent.
# google/gemma-2b is gated (401 without auth); use the unsloth public
# mirror — same tokenizer.model bytes, no auth required.
GEMMA_URL = "https://huggingface.co/unsloth/gemma-2b/resolve/main/tokenizer.model"
GEMMA_PATH = os.path.join(VOCABS_DIR, "gemma.model")

# Optional fourth vocab for the HF-Unigram byte_fallback verification path.
# llm-jp's tokenizers are real-world `type=Unigram` HF tokenizer.json
# exports with `byte_fallback=true` and the full 256-piece `<0xNN>` bank.
# Used by post-1.15 agent B to exercise the HF byte-fallback table on a
# non-synthetic vocab.
LLMJP_HF_URL = "https://huggingface.co/llm-jp/llm-jp-3-1.8b/resolve/main/tokenizer.json"
LLMJP_HF_PATH = os.path.join(VOCABS_DIR, "llmjp3_hf.json")

# Optional fifth vocab for the HF-BPE cross-tokenizer parity check.
# GPT-2's tokenizer.json is the canonical real-world `type=BPE` HF export
# (ByteLevel pre-tokenizer + decoder, no normalizer, ~50K vocab, 1.4 MB
# on disk). Used by post-1.16 agent D to verify the hf_json+hf_bridge
# loader's BPE path against the upstream `tokenizers` library on the
# same input.
GPT2_HF_URL = "https://huggingface.co/openai-community/gpt2/resolve/main/tokenizer.json"
GPT2_HF_PATH = os.path.join(VOCABS_DIR, "gpt2_hf.json")

# Optional sixth vocab for the HF WordPiece + BertNormalizer parity check
# (post-1.17 agent D). bert-base-uncased is the canonical real-world
# `type=WordPiece` HF export with a non-trivial normalizer
# (BertNormalizer { lowercase: true, clean_text: true,
# handle_chinese_chars: true, strip_accents: null }). ~250 KB on disk.
# Exercises BertNormalizer + WordPiece in the cross-bench.
BERT_BASE_URL = "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/tokenizer.json"
BERT_BASE_PATH = os.path.join(VOCABS_DIR, "bert_base_uncased.json")

TM_VOCAB_NAME = "englishcode-32000-clean-nocapcode-v1"
TM_LOCAL_NAME = "tm_englishcode_32k.vocab"
TM_PATH = os.path.join(VOCABS_DIR, TM_LOCAL_NAME)
ZTM_PATH = os.path.join(VOCABS_DIR, "tm_englishcode_32k.ztm")

# ----------------------------------------------------------------------
# Extended cross-bench fixtures (post-1.17 agent E). Downloaded only
# when `fetch_vocabs.py --extended` is passed so default CI runs stay
# at the original ~10 MB total. Each entry pairs an HF URL with the
# local basename. Mirrors are chosen for un-gated public access — if
# a model becomes gated, swap the URL for an `unsloth/` or other open
# mirror (same tokenizer bytes; smaller weight files don't matter).
# ----------------------------------------------------------------------

EXTENDED_VOCABS = [
    # (label, url, local_basename, kind)
    (
        "mistral7b",
        "https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/tokenizer.model",
        "mistral7b.model",
        "sp-bpe",
    ),
    (
        "yi6b",
        "https://huggingface.co/01-ai/Yi-6B/resolve/main/tokenizer.model",
        "yi6b.model",
        "sp-bpe",
    ),
    (
        "phi3",
        "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/resolve/main/tokenizer.json",
        "phi3.json",
        "hf-bpe",
    ),
    (
        "falcon7b",
        "https://huggingface.co/tiiuae/falcon-7b/resolve/main/tokenizer.json",
        "falcon7b.json",
        "hf-bpe",
    ),
    (
        "deepseek_v2",
        "https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/resolve/main/tokenizer.json",
        "deepseek_v2.json",
        "hf-bpe",
    ),
    (
        "qwen2",
        "https://huggingface.co/Qwen/Qwen2-7B/resolve/main/tokenizer.json",
        "qwen2.json",
        "hf-bpe",
    ),
    # Optional larger fixture — Llama-3 weighs in at ~9 MB on disk. Kept
    # in the extended set since the byte_level BPE path is heavily used
    # in the wild and the 50 MB total budget still has plenty of room.
    (
        "llama3",
        "https://huggingface.co/unsloth/llama-3-8b/resolve/main/tokenizer.json",
        "llama3.json",
        "hf-bpe",
    ),
    # ------------------------------------------------------------------
    # 1.22+ additions: cover the Mistral Tekken family and the broader
    # set of post-Llama-3 / post-Phi-3 / post-Qwen-2 tokenizers that
    # appear in real-world deployments. These are *not* vendored in the
    # repo (they push the bench/vocabs/ budget past 50 MB) — `--extended`
    # downloads them on demand. See bench/vocabs/AVAILABILITY.md for the
    # full per-model availability matrix (auth status, sizes, mirrors).
    # ------------------------------------------------------------------
    (
        # Mistral Nemo Tekken — the canonical real-world Tekkenizer
        # fixture for the post-1.22 Tekken loader (`src/tekken.zig`).
        # v3 pattern, 130 000 BPE ranks, 1 000 special slots.
        "mistral_nemo_tekken",
        "https://huggingface.co/mistralai/Mistral-Nemo-Base-2407/resolve/main/tekken.json",
        "mistral_nemo_tekken.json",
        "tekken",
    ),
    (
        # Mistral-Small-Instruct-2409 — SentencePiece v3 (.model.v3).
        # ~587 KB. Same wire format as Mistral-7B's tokenizer.model so
        # the existing `sp-bpe` loader covers it; useful as a more
        # recent post-2024 SP fixture.
        "mistral_small_v3",
        "https://huggingface.co/mistralai/Mistral-Small-Instruct-2409/resolve/main/tokenizer.model.v3",
        "mistral_small_v3.model",
        "sp-bpe",
    ),
    (
        # Codestral-22B v0.1 — SentencePiece v3 (.model.v3), 587 KB.
        # Code-tuned variant of the Mistral SP family.
        "codestral_v3",
        "https://huggingface.co/mistralai/Codestral-22B-v0.1/resolve/main/tokenizer.model.v3",
        "codestral_v3.model",
        "sp-bpe",
    ),
    (
        # Phi-3.5-mini — HF BPE, ~1.8 MB. Tokenizer largely shared with
        # Phi-3-mini but rebuilt by the Phi team; included to cover any
        # drift from the existing phi3.json fixture.
        "phi35_mini",
        "https://huggingface.co/microsoft/Phi-3.5-mini-instruct/resolve/main/tokenizer.json",
        "phi35_mini.json",
        "hf-bpe",
    ),
    (
        # Phi-4 — HF BPE, ~4.3 MB. Tokenizer was re-derived for Phi-4
        # (vocab grew vs Phi-3); useful baseline for the post-Phi-3
        # tokenizer family.
        "phi4",
        "https://huggingface.co/microsoft/phi-4/resolve/main/tokenizer.json",
        "phi4.json",
        "hf-bpe",
    ),
    (
        # Qwen-2.5-7B — HF BPE, ~7 MB. Direct successor to Qwen2 with
        # vocab tweaks; mostly an incremental refresh.
        "qwen25",
        "https://huggingface.co/Qwen/Qwen2.5-7B/resolve/main/tokenizer.json",
        "qwen25.json",
        "hf-bpe",
    ),
    (
        # Qwen-3-8B — HF BPE, ~11.4 MB. Adds many newer multilingual /
        # code tokens vs Qwen-2.5; among the largest open BPE vocabs.
        "qwen3",
        "https://huggingface.co/Qwen/Qwen3-8B/resolve/main/tokenizer.json",
        "qwen3.json",
        "hf-bpe",
    ),
    (
        # DeepSeek-V3 — HF BPE, ~7.8 MB. Bigger vocab than V2 / V2-Lite;
        # the V3 family is the current DeepSeek baseline.
        "deepseek_v3",
        "https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/main/tokenizer.json",
        "deepseek_v3.json",
        "hf-bpe",
    ),
]


def fetch_extended(dry_run: bool = False) -> None:
    """Download the extended cross-bench fixtures into bench/vocabs/.

    Each download is wrapped in try/except so a transient HF outage on
    any single mirror doesn't abort the rest. Files larger than 32 MB
    are skipped with a printed note. The 32 MB ceiling is sized to
    accommodate the largest currently-listed fixture (Mistral Nemo
    Tekken at ~14.8 MB) plus headroom for Pixtral (~19 MB) and GPT-OSS
    (~28 MB) if they get added — see AVAILABILITY.md.

    When `dry_run` is True, prints the URL / destination / kind for each
    entry without touching the network or disk. Useful for verifying
    that new fetch entries are wired up correctly (matches the task
    verification step in agent E follow-ups).
    """
    MAX_BYTES = 32 * 1024 * 1024  # per-file cap
    if dry_run:
        print(f"DRY RUN — {len(EXTENDED_VOCABS)} extended fixtures registered:")
        for label, url, basename, kind in EXTENDED_VOCABS:
            dest = os.path.join(VOCABS_DIR, basename)
            present = "present" if os.path.exists(dest) else "absent"
            print(f"  [{kind:>11s}] {label:<24s} -> {basename}  ({present})")
            print(f"               {url}")
        return
    for label, url, basename, _kind in EXTENDED_VOCABS:
        dest = os.path.join(VOCABS_DIR, basename)
        if os.path.exists(dest):
            print(f"already have {dest} ({os.path.getsize(dest)} bytes)")
            continue
        try:
            print(f"downloading [{label}] {url} -> {dest}")
            urllib.request.urlretrieve(url, dest)
        except Exception as e:
            print(f"  warning: {label} fetch failed ({e!r}); skipping", file=sys.stderr)
            continue
        sz = os.path.getsize(dest)
        if sz > MAX_BYTES:
            print(
                f"  warning: {label} is {sz} bytes (> {MAX_BYTES}); removing",
                file=sys.stderr,
            )
            try:
                os.unlink(dest)
            except OSError:
                pass
            continue
        print(f"  {sz} bytes")


def download(url: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"already have {dest} ({os.path.getsize(dest)} bytes)")
        return
    print(f"downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    print(f"  {os.path.getsize(dest)} bytes")


def fetch_tm_vocab() -> None:
    if os.path.exists(TM_PATH):
        print(f"already have {TM_PATH}")
    else:
        try:
            import tokenmonster
        except ImportError:
            print(
                "ERROR: `pip install tokenmonster` first "
                "(needed to download the TM vocab + ship the Go server binary)",
                file=sys.stderr,
            )
            sys.exit(1)
        # Triggers download of both the .vocab file and the
        # tokenmonsterserver binary into ~/_tokenmonster.
        vocab = tokenmonster.load(TM_VOCAB_NAME)
        local = vocab.fname
        import shutil
        shutil.copyfile(local, TM_PATH)
        print(f"  copied {local} -> {TM_PATH} ({os.path.getsize(TM_PATH)} bytes)")
        tokenmonster.disconnect()


def convert_tm_to_ztm() -> None:
    if os.path.exists(ZTM_PATH) and os.path.getmtime(ZTM_PATH) > os.path.getmtime(TM_PATH):
        print(f"already have {ZTM_PATH} (newer than .vocab)")
        return
    print(f"converting {TM_PATH} -> {ZTM_PATH}")
    cmd = [sys.executable, os.path.join(BENCH_DIR, "convert_tm_to_ztm.py"), TM_PATH, ZTM_PATH]
    subprocess.check_call(cmd)


def main() -> int:
    # `--extended` downloads the post-1.17 cross-bench fixtures in
    # addition to the 1.17 baseline. Kept as an opt-in flag so default
    # CI runs stay at the original ~10 MB total.
    extended = "--extended" in sys.argv[1:]
    only_extended = "--only-extended" in sys.argv[1:]
    dry_run = "--dry-run" in sys.argv[1:]
    os.makedirs(VOCABS_DIR, exist_ok=True)
    if only_extended:
        fetch_extended(dry_run=dry_run)
        if not dry_run:
            print("ok — extended bench vocabs present")
        return 0
    download(LLAMA2_URL, LLAMA2_PATH)
    download(T5_URL, T5_PATH)
    # Gemma's tokenizer.model is on a gated repo for some mirrors —
    # don't hard-fail the fetch step if 401/403 comes back. The
    # casefold/byte_fallback tests skip when the file's absent.
    try:
        download(GEMMA_URL, GEMMA_PATH)
    except Exception as e:
        print(f"  warning: gemma fetch failed ({e!r}); skipping", file=sys.stderr)
    # The llm-jp HF tokenizer.json is ~6 MB; non-fatal if it fails so
    # CI envs without the byte_fallback HF check skip silently.
    try:
        download(LLMJP_HF_URL, LLMJP_HF_PATH)
    except Exception as e:
        print(f"  warning: llm-jp HF fetch failed ({e!r}); skipping", file=sys.stderr)
    # GPT-2 tokenizer.json is ~1.4 MB; non-fatal if HF is unreachable so
    # the HF-BPE equivalence check skips gracefully.
    try:
        download(GPT2_HF_URL, GPT2_HF_PATH)
    except Exception as e:
        print(f"  warning: gpt2 HF fetch failed ({e!r}); skipping", file=sys.stderr)
    # bert-base-uncased tokenizer.json is ~250 KB; gates the HF
    # WordPiece + BertNormalizer cross-bench. Non-fatal on failure.
    try:
        download(BERT_BASE_URL, BERT_BASE_PATH)
    except Exception as e:
        print(f"  warning: bert-base-uncased fetch failed ({e!r}); skipping", file=sys.stderr)
    fetch_tm_vocab()
    convert_tm_to_ztm()
    if extended:
        fetch_extended(dry_run=dry_run)
    print("ok — all bench vocabs present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
