#!/usr/bin/env bash
# Smoke test for the extended (post-1.17 agent E) cross-bench fixtures.
#
# For each vocab file vendored by `bench/fetch_vocabs.py --extended`,
# runs `bench/equivalence_check.py <kind> bench/vocabs/<basename>` and
# prints the resulting match rate. Exits 0 unless a required step is
# missing — divergence is expected on some fixtures (Phi-3 uses an
# SP-style normalizer chain that ztok's identity path can't reproduce;
# Llama-3 / Qwen2 use a Llama-3-style Split regex variant that ztok's
# `hf_byte_level` pretok doesn't apply). The exact match-rate numbers
# are reported by `equivalence_check.py` and tracked in
# `bench/RESULTS.md` under "1.18 extended cross-tokenizer equivalence".
#
# Tekken fixtures (Mistral Nemo et al, post-1.22) are not yet wired
# into equivalence_check.py — `bench_cross` doesn't have a `tekken`
# --kind. Until that's added, the Tekken loader is exercised via a
# parse-only check using `zig-out/bin/ztok info` (which calls
# `loadPipelineAutoDetect` -> `tekken.loadTekkenFile`). That at least
# proves the file is real, parses, and the v3 invariants hold.
#
# Usage:
#   bench/equivalence_smoke_extended.sh
#
# Requires:
#   - `zig build` already run (so `zig-out/bin/bench_cross` exists)
#   - `pip install sentencepiece tokenizers`
#   - $ZTOK_BENCH_CORPUS (default /tmp/corpus.txt) populated

set -u
cd "$(dirname "$0")/.."

CORPUS="${ZTOK_BENCH_CORPUS:-/tmp/corpus.txt}"
LINES="${ZTOK_BENCH_LINES:-100}"

if [ ! -f "$CORPUS" ]; then
  echo "missing corpus: $CORPUS" >&2
  exit 1
fi
if [ ! -x zig-out/bin/bench_cross ]; then
  echo "missing zig-out/bin/bench_cross (run \`zig build\`)" >&2
  exit 1
fi

ok=0
total=0
run_one() {
  local kind="$1" base="$2" disp="$3"
  total=$((total + 1))
  if [ ! -f "bench/vocabs/${base}.model" ] && [ ! -f "bench/vocabs/${base}.json" ]; then
    echo "[SKIP] ${disp} — fixture bench/vocabs/${base}.{model,json} not present"
    return
  fi
  echo "[RUN ] ${disp} (kind=${kind})"
  if ZTOK_BENCH_LINES="$LINES" python3 bench/equivalence_check.py "$kind" "bench/vocabs/${base}" 2>&1 | tail -1; then
    ok=$((ok + 1))
  fi
}

# Tekken fixtures land as a single `tekken.json` per model; the file is
# the basename + ".json". Mirrors `run_one` but gates on the .json only.
run_tekken() {
  local base="$1" disp="$2"
  total=$((total + 1))
  if [ ! -f "bench/vocabs/${base}.json" ]; then
    echo "[SKIP] ${disp} — fixture bench/vocabs/${base}.json not present (run \`python3 bench/fetch_vocabs.py --only-extended\`)"
    return
  fi
  echo "[RUN ] ${disp} (kind=tekken)"
  if ZTOK_BENCH_LINES="$LINES" python3 bench/equivalence_check.py tekken "bench/vocabs/${base}" 2>&1 | tail -1; then
    ok=$((ok + 1))
  fi
}

# Parse-only smoke check for fixtures the equivalence harness doesn't
# yet cover (Tekken family). Asserts ztok's auto-detect loader can
# read the file without erroring — that's the minimum gate for the
# fixture being "added" to the bench.
run_tekken_parse() {
  local fname="$1" disp="$2"
  total=$((total + 1))
  if [ ! -f "bench/vocabs/${fname}" ]; then
    echo "[SKIP] ${disp} — fixture bench/vocabs/${fname} not present (run \`python3 bench/fetch_vocabs.py --extended\`)"
    return
  fi
  if [ ! -x zig-out/bin/ztok ]; then
    echo "[SKIP] ${disp} — zig-out/bin/ztok not built"
    return
  fi
  echo "[RUN ] ${disp} (parse-only)"
  if zig-out/bin/ztok info "bench/vocabs/${fname}" >/dev/null 2>&1; then
    echo "  parsed ok"
    ok=$((ok + 1))
  else
    echo "  parse FAILED"
  fi
}

# 1.17 baseline extended set
run_one sp-bpe  mistral7b   "Mistral-7B"
run_one sp-bpe  yi6b        "Yi-6B"
run_one hf-bpe  phi3        "Phi-3-mini"
run_one hf-bpe  falcon7b    "Falcon-7B"
run_one hf-bpe  deepseek_v2 "DeepSeek-V2-Lite"
run_one hf-bpe  qwen2       "Qwen2-7B"
run_one hf-bpe  llama3      "Llama-3-8B"

# 1.22+ additions — newer Mistral, Phi, Qwen, DeepSeek fixtures.
# These use the existing sp-bpe / hf-bpe kinds; only the basenames are new.
run_one sp-bpe  mistral_small_v3 "Mistral-Small-Instruct-2409 (v3 SP)"
run_one sp-bpe  codestral_v3     "Codestral-22B v0.1 (v3 SP)"
run_one hf-bpe  phi35_mini       "Phi-3.5-mini"
run_one hf-bpe  phi4             "Phi-4"
run_one hf-bpe  qwen25           "Qwen-2.5-7B"
run_one hf-bpe  qwen3            "Qwen-3-8B"
run_one hf-bpe  deepseek_v3      "DeepSeek-V3"

# 1.22+ Tekken — `bench_cross --kind tekken` now wired; equivalence
# checked against mistral_common (preferred) or tiktoken (fallback). Any
# future Tekken-family fixtures (Pixtral / Ministral / Mistral-Small 3.1
# Tekken builds) get one `run_tekken <basename> <display>` line each as
# they're added to `fetch_vocabs.py`'s EXTENDED_VOCABS list.
run_tekken mistral_nemo_tekken "Mistral-Nemo Tekken"

# Parse-only smoke for the still-unwired Tekken family (e.g. Pixtral
# multimodal builds), kept for completeness. No fixtures here today —
# move entries up into `run_tekken` when `fetch_vocabs.py` starts
# downloading them and they fit the equivalence harness.

echo "---"
echo "extended cross-bench: ${ok}/${total} fixtures executed"
