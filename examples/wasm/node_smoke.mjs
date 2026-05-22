// Node smoke test for the ztok browser wasm. Cross-platform: runs the
// exact JS loader logic we use in the browser, just with `fs.readFileSync`
// in place of `fetch` and `URL.createObjectURL`. Doesn't need a vocab
// file — we synthesize a small tiktoken vocab inline so this test runs
// completely standalone.
//
// Usage:
//     node examples/wasm/node_smoke.mjs
//
// Exits non-zero on any assertion failure.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const WASM_PATH = path.join(__dirname, "..", "..", "zig-out", "bin", "ztok_browser.wasm");

function buildMinimalTiktoken() {
  // 256 single-byte tokens + a few common merges. Same shape as the
  // test fixture in src/c_api.zig.
  const lines = [];
  for (let b = 0; b < 256; b++) {
    const buf = Buffer.from([b]);
    lines.push(`${buf.toString("base64")} ${b}`);
  }
  const extras = ["he", "hel", "hell", "hello", " w", " wo", " wor", " worl", " world"];
  let rank = 256;
  for (const piece of extras) {
    const buf = Buffer.from(piece, "utf8");
    lines.push(`${buf.toString("base64")} ${rank++}`);
  }
  return new TextEncoder().encode(lines.join("\n") + "\n");
}

async function main() {
  const wasmBytes = readFileSync(WASM_PATH);
  const { instance } = await WebAssembly.instantiate(wasmBytes, {});
  const exp = instance.exports;
  const mem = () => new Uint8Array(exp.memory.buffer);

  // Version check.
  const verPtr = exp.ztok_version_ptr();
  const verLen = exp.ztok_version_len();
  const ver = new TextDecoder().decode(mem().subarray(verPtr, verPtr + verLen));
  console.log(`ztok version: ${ver}`);
  if (!ver.startsWith("1.16")) throw new Error(`unexpected version: ${ver}`);

  // Load vocab.
  const vocab = buildMinimalTiktoken();
  const vocabPtr = exp.ztok_malloc(vocab.length);
  if (vocabPtr === 0) throw new Error("ztok_malloc(vocab) returned 0");
  mem().set(vocab, vocabPtr);

  // Status output slot. Two i32s would be 8 bytes; we use one.
  const statusPtr = exp.ztok_malloc(4);
  const pipePtr = exp.ztok_pipeline_new_bpe_from_tiktoken_bytes(vocabPtr, vocab.length, statusPtr);
  const status = new Int32Array(exp.memory.buffer, statusPtr, 1)[0];
  if (status !== 0 || pipePtr === 0) {
    throw new Error(`pipeline_new failed: status=${status}, ptr=${pipePtr}`);
  }
  exp.ztok_free(statusPtr);
  exp.ztok_free(vocabPtr);

  // Encode "hello world" — should produce exactly 2 ids per the
  // c_api test fixture (cl100k splits, then BPE merges to one token
  // each).
  const input = new TextEncoder().encode("hello world");
  const inputPtr = exp.ztok_malloc(input.length);
  mem().set(input, inputPtr);

  // Two u32 out-slots, contiguous (allocate 8 bytes).
  const outBlk = exp.ztok_malloc(8);
  const outIdsPtrSlot = outBlk;
  const outLenSlot = outBlk + 4;

  const rc = exp.ztok_encode(pipePtr, inputPtr, input.length, outIdsPtrSlot, outLenSlot);
  if (rc !== 0) throw new Error(`encode failed: rc=${rc}`);

  const outIdsPtr = new Uint32Array(exp.memory.buffer, outIdsPtrSlot, 1)[0];
  const outLen = new Uint32Array(exp.memory.buffer, outLenSlot, 1)[0];
  const ids = Array.from(new Uint32Array(exp.memory.buffer, outIdsPtr, outLen));
  console.log(`encode("hello world") -> [${ids.join(", ")}]`);
  if (ids.length !== 2) throw new Error(`expected 2 ids, got ${ids.length}`);

  // Decode round-trip.
  const idsPtr = exp.ztok_malloc(outLen * 4);
  new Uint32Array(exp.memory.buffer, idsPtr, outLen).set(ids);
  const decBlk = exp.ztok_malloc(8);
  const decBytesSlot = decBlk;
  const decLenSlot = decBlk + 4;
  const drc = exp.ztok_decode(pipePtr, idsPtr, outLen, decBytesSlot, decLenSlot);
  if (drc !== 0) throw new Error(`decode failed: rc=${drc}`);
  const decBytesPtr = new Uint32Array(exp.memory.buffer, decBytesSlot, 1)[0];
  const decLen = new Uint32Array(exp.memory.buffer, decLenSlot, 1)[0];
  const decoded = new TextDecoder().decode(new Uint8Array(exp.memory.buffer, decBytesPtr, decLen));
  console.log(`decode -> ${JSON.stringify(decoded)}`);
  if (decoded !== "hello world") throw new Error(`round-trip mismatch: ${decoded}`);

  // Cleanup.
  exp.ztok_free(outIdsPtr);
  exp.ztok_free(decBytesPtr);
  exp.ztok_free(outBlk);
  exp.ztok_free(decBlk);
  exp.ztok_free(idsPtr);
  exp.ztok_free(inputPtr);
  exp.ztok_pipeline_free(pipePtr);

  console.log("OK");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
