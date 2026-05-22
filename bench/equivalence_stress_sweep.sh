#!/usr/bin/env bash
# Run the 13 × 5 stress equivalence sweep for the 1.21 gate.
#
# Each of the 13 SP+HF fixtures × 4 prose corpora (english/code/
# multilingual/chat) at 10K lines + 1 Unicode-stress corpus at 1K lines
# is shelled through `bench/equivalence_check.py --json`. Records are
# emitted one-per-line on stdout (NDJSON) for downstream parsing; a
# human-readable progress line goes to stderr.
#
# Requires:
#   - `zig build` already run (so `zig-out/bin/bench_cross` exists)
#   - `pip install sentencepiece tokenizers`
#   - bench/corpora/*.txt populated (run `bench/corpora/build_corpora.py`)
#
# Usage:
#   bench/equivalence_stress_sweep.sh                   # full 65-cell sweep
#   bench/equivalence_stress_sweep.sh --first-diff-only # fast triage
#   bench/equivalence_stress_sweep.sh --only english.txt
#
# Output: NDJSON on stdout. Each record has at minimum:
#   {kind, vocab, reference, corpus, lines_requested, lines_compared,
#    matches, diffs, match_rate, first_diff_idx, ...}

set -u
cd "$(dirname "$0")/.."

extra_args=()
only_corpus=""
while [ $# -gt 0 ]; do
  case "$1" in
    --first-diff-only)
      extra_args+=(--first-diff-only)
      shift ;;
    --only)
      only_corpus="$2"
      shift 2 ;;
    *)
      echo "unknown flag: $1" >&2
      exit 2 ;;
  esac
done

if [ ! -x zig-out/bin/bench_cross ]; then
  echo "missing zig-out/bin/bench_cross (run \`zig build\`)" >&2
  exit 1
fi
for c in english.txt code.txt multilingual.txt chat.txt unicode_stress.txt; do
  if [ ! -f "bench/corpora/$c" ]; then
    echo "missing bench/corpora/$c (run \`python3 bench/corpora/build_corpora.py\`)" >&2
    exit 1
  fi
done

# (kind, basename, display_name)
FIXTURES=(
  "sp-bpe  llama2       LLaMA-2"
  "unigram t5_unigram   T5"
  "sp-bpe  gemma        Gemma"
  "hf-bpe  gpt2_hf      GPT-2"
  "sp-bpe  mistral7b    Mistral-7B"
  "sp-bpe  yi6b         Yi-6B"
  "hf-bpe  deepseek_v2  DeepSeek-V2-Lite"
  "hf-unigram llmjp3_hf llm-jp-3"
  "hf-wordpiece bert_base_uncased bert-base-uncased"
  "hf-bpe  falcon7b     Falcon-7B"
  "hf-bpe  qwen2        Qwen2-7B"
  "hf-bpe  llama3       Llama-3-8B"
  "hf-bpe  phi3         Phi-3-mini"
  # 1.22+ extended fixtures — the SP/HF entries ride on the existing
  # kinds; the lone Tekken family member uses the new `tekken` kind
  # added in the post-1.23 wiring pass. Each row is gated by file
  # presence in `equivalence_check.py` itself, so missing fetches just
  # skip rather than fail the sweep.
  "sp-bpe  mistral_small_v3    Mistral-Small-Instruct-2409"
  "sp-bpe  codestral_v3        Codestral-22B-v0.1"
  "hf-bpe  phi35_mini          Phi-3.5-mini"
  "hf-bpe  phi4                Phi-4"
  "hf-bpe  qwen25              Qwen-2.5-7B"
  "hf-bpe  qwen3               Qwen-3-8B"
  "hf-bpe  deepseek_v3         DeepSeek-V3"
  "tekken  mistral_nemo_tekken Mistral-Nemo-Tekken"
)

# (corpus_file, line_count)
CORPORA=(
  "english.txt        10000"
  "code.txt           10000"
  "multilingual.txt   10000"
  "chat.txt           10000"
  "unicode_stress.txt 1000"
)

run_one() {
  local kind="$1" base="$2" disp="$3" corpus_file="$4" n="$5"
  echo "[run] ${disp} × ${corpus_file} (${n} lines)" >&2
  python3 bench/equivalence_check.py "$kind" "bench/vocabs/${base}" \
    --corpus "bench/corpora/${corpus_file}" --lines "$n" --json \
    "${extra_args[@]}" 2> >(tail -1 >&2)
}

for fx in "${FIXTURES[@]}"; do
  set -- $fx
  kind="$1"; base="$2"; shift 2
  disp="$*"
  for co in "${CORPORA[@]}"; do
    set -- $co
    corpus_file="$1"; n="$2"
    if [ -n "$only_corpus" ] && [ "$corpus_file" != "$only_corpus" ]; then
      continue
    fi
    run_one "$kind" "$base" "$disp" "$corpus_file" "$n"
  done
done
