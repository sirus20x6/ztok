#!/usr/bin/env bash
# Minimal smoke test: serve `examples/wasm/` over HTTP, fetch each file
# the demo + bench depend on, and verify they come back non-empty + the
# HTML references the wasm module. Doesn't run wasm — for that, see
# `node_smoke.mjs` next to this script.
#
# Usage:    ./examples/wasm/smoke.sh
# Exits 0 on success, 1 on any failure.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PORT=${PORT:-8765}

pushd "$HERE" >/dev/null

# Make sure the wasm has actually been built before we try to serve it.
WASM="$HERE/../../zig-out/bin/ztok_browser.wasm"
if [ ! -f "$WASM" ]; then
  echo "missing $WASM — run 'zig build ztok-wasm-browser' first" >&2
  exit 1
fi
ln -sf "$WASM" ztok_browser.wasm

python3 -m http.server "$PORT" >/tmp/ztok_wasm_http.log 2>&1 &
PID=$!
trap "kill $PID 2>/dev/null; rm -f ztok_browser.wasm" EXIT

# Wait briefly for the server to come up — poll instead of fixed sleep.
for i in $(seq 1 20); do
  if curl -sf "http://localhost:$PORT/index.html" >/dev/null 2>&1; then break; fi
  sleep 0.1
done

fail=0
check() {
  local path="$1" min_bytes="$2" must_contain="$3"
  local sz=$(curl -sf "http://localhost:$PORT/$path" | wc -c)
  if [ "$sz" -lt "$min_bytes" ]; then
    echo "FAIL $path: $sz bytes < $min_bytes" >&2
    fail=1; return
  fi
  if [ -n "$must_contain" ]; then
    if ! curl -sf "http://localhost:$PORT/$path" | grep -q "$must_contain"; then
      echo "FAIL $path: missing '$must_contain'" >&2
      fail=1; return
    fi
  fi
  echo "OK $path ($sz bytes)"
}

check index.html       2000 "ztok_browser.wasm"
check bench.html       2000 "tiktoken-js"
check corpus-small.txt 100000 ""
check ztok_browser.wasm 100000 ""

exit $fail
