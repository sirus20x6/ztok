#!/usr/bin/env bash
# TokenMonster (TM-Go) parity gate.
#
# Enforces a per-vocab × per-corpus floor on ztok's Monster encoder vs
# the TokenMonster-Go reference. This is the CI guard that "locks in"
# the parity work: each of the four cells below must stay at or above
# TM_PARITY_FLOOR (%), or the gate exits non-zero.
#
# History (100-line sample, ztok vs TM-Go):
#   * Pre-fix ceiling:  80/85 nocapcode, 56/77 full-capcode
#       (nocap×eng 80, nocap×code 85, fullcap×eng 56, fullcap×code 77)
#   * Post-fix (this branch): nocap×eng 92, nocap×code 93,
#       fullcap×eng 96, fullcap×code 96 — see bench/RESULTS.md
#       "TM Monster equivalence".
#
# The floor is deliberately set BELOW the observed numbers (a safe 90%
# margin) so CI can't silently regress below 90 but also won't flake on
# the exact per-corpus value, which shifts a point or two with the
# corpus sample. Raise the floor only when the encoder has durably
# moved the whole matrix above the new value.
#
# This is a correctness comparison at 100 lines — NOT the 10K stress
# sweep and NOT a throughput benchmark. It is cheap (a few seconds per
# cell) and safe to run on a shared box.
#
# Requires:
#   - `zig build -Doptimize=ReleaseFast` already run (zig-out/bin/bench_cross)
#   - `pip install tokenmonster`
#   - bench/corpora/{english,code}.txt and bench/vocabs/*.ztm present
#
# Usage:
#   bench/tm_parity_gate.sh                 # gate at the default floor
#   TM_PARITY_FLOOR=92 bench/tm_parity_gate.sh
#   TM_PARITY_LINES=100 bench/tm_parity_gate.sh

set -u
cd "$(dirname "$0")/.."

FLOOR="${TM_PARITY_FLOOR:-90}"
LINES="${TM_PARITY_LINES:-100}"

if [ ! -x zig-out/bin/bench_cross ]; then
  echo "missing zig-out/bin/bench_cross (run \`zig build -Doptimize=ReleaseFast\`)" >&2
  exit 1
fi

# (vocab_basename, corpus_basename, display)
CELLS=(
  "tm_englishcode_32k         english  nocapcode×english"
  "tm_englishcode_32k         code     nocapcode×code"
  "tm_englishcode_capcode_32k english  fullcapcode×english"
  "tm_englishcode_capcode_32k code     fullcapcode×code"
)

fail=0
echo "TM parity gate — floor=${FLOOR}% @ ${LINES} lines"
echo "---"
for cell in "${CELLS[@]}"; do
  set -- $cell
  vocab="$1"; corpus="$2"; disp="$3"
  ztm="bench/vocabs/${vocab}.ztm"
  corp="bench/corpora/${corpus}.txt"
  if [ ! -f "$ztm" ]; then
    echo "[FAIL] ${disp} — missing fixture ${ztm}" >&2
    fail=1
    continue
  fi
  if [ ! -f "$corp" ]; then
    echo "[FAIL] ${disp} — missing corpus ${corp}" >&2
    fail=1
    continue
  fi
  # --json prints one NDJSON record to stdout with a `match_rate` field
  # in [0,1]; the human summary goes to stderr (discarded here).
  json="$(python3 bench/equivalence_check.py monster "bench/vocabs/${vocab}" \
            --corpus "$corp" --lines "$LINES" --json 2>/dev/null | tail -1)"
  pct="$(printf '%s' "$json" | python3 -c \
        'import sys,json; r=json.loads(sys.stdin.read() or "{}"); print(round(100.0*r.get("match_rate",0.0),1))' \
        2>/dev/null)"
  if [ -z "$pct" ]; then
    echo "[FAIL] ${disp} — no match_rate (check tokenmonster install / bench_cross)" >&2
    fail=1
    continue
  fi
  # Floor comparison in awk (no bc dependency).
  if awk -v p="$pct" -v f="$FLOOR" 'BEGIN{exit !(p+0 >= f+0)}'; then
    echo "[PASS] ${disp}: ${pct}% (>= ${FLOOR}%)"
  else
    echo "[FAIL] ${disp}: ${pct}% (< ${FLOOR}%)" >&2
    fail=1
  fi
done

echo "---"
if [ "$fail" -ne 0 ]; then
  echo "TM parity gate FAILED (one or more cells below ${FLOOR}%)" >&2
  exit 1
fi
echo "TM parity gate PASSED (all cells >= ${FLOOR}%)"
