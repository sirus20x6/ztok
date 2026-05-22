// Node-side counterpart to bench.html — runs the ztok-wasm encoder on
// the vendored corpus-small.txt and reports MB/s, so we have a hard
// throughput number without needing a browser tab.
//
// (Doesn't fetch tiktoken-js — that lives in bench.html where the
// CDN is available; here we just gauge ztok's freestanding wasm
// throughput single-threaded under v8.)
//
// Usage: node examples/wasm/node_bench.mjs [path-to-cl100k.tiktoken]
//   - If no vocab path is given, synthesizes a tiny vocab (slower
//     pre-tokenizer dispatch, no real BPE merges to compare with).

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const WASM_PATH = path.join(__dirname, "..", "..", "zig-out", "bin", "ztok_browser.wasm");
const CORPUS = path.join(__dirname, "corpus-small.txt");

function synthVocab() {
  const lines = [];
  for (let b = 0; b < 256; b++) {
    lines.push(`${Buffer.from([b]).toString("base64")} ${b}`);
  }
  const extras = ["he","hel","hell","hello","th","the"," the","ing"," and"," of"," in"];
  let r = 256;
  for (const e of extras) lines.push(`${Buffer.from(e,"utf8").toString("base64")} ${r++}`);
  return new TextEncoder().encode(lines.join("\n")+"\n");
}

async function main() {
  const wasmBytes = readFileSync(WASM_PATH);
  const { instance } = await WebAssembly.instantiate(wasmBytes, {});
  const exp = instance.exports;
  const memU8 = () => new Uint8Array(exp.memory.buffer);
  const memI32 = () => new Int32Array(exp.memory.buffer);
  const memU32 = () => new Uint32Array(exp.memory.buffer);

  const vocabPath = process.argv[2];
  const vocab = vocabPath ? readFileSync(vocabPath) : synthVocab();
  console.log(`vocab: ${vocab.length} bytes${vocabPath ? " from " + vocabPath : " (synthetic)"}`);

  const vp = exp.ztok_malloc(vocab.length);
  memU8().set(vocab, vp);
  const sp = exp.ztok_malloc(4);
  const t0 = performance.now();
  const pipe = exp.ztok_pipeline_new_bpe_from_tiktoken_bytes(vp, vocab.length, sp);
  const t1 = performance.now();
  const st = memI32()[sp >>> 2];
  if (st !== 0 || pipe === 0) throw new Error(`load failed: status=${st}`);
  console.log(`load: ${(t1-t0).toFixed(1)} ms`);
  exp.ztok_free(vp); exp.ztok_free(sp);

  const corpus = readFileSync(CORPUS);
  console.log(`corpus: ${corpus.length} bytes`);

  // Encode once for ids count, then 100 iters for timing.
  const ip = exp.ztok_malloc(corpus.length);
  memU8().set(corpus, ip);
  const out = exp.ztok_malloc(8);

  let firstIds = 0;
  for (let i = 0; i < 3; i++) { // warmup
    const rc = exp.ztok_encode(pipe, ip, corpus.length, out, out + 4);
    if (rc !== 0) throw new Error(`encode rc=${rc}`);
    const idsPtr = memU32()[out >>> 2];
    firstIds = memU32()[(out+4) >>> 2];
    if (idsPtr !== 0) exp.ztok_free(idsPtr);
  }
  const iters = 100;
  const e0 = performance.now();
  for (let i = 0; i < iters; i++) {
    const rc = exp.ztok_encode(pipe, ip, corpus.length, out, out + 4);
    if (rc !== 0) throw new Error(`encode rc=${rc}`);
    const idsPtr = memU32()[out >>> 2];
    if (idsPtr !== 0) exp.ztok_free(idsPtr);
  }
  const ms = performance.now() - e0;
  const totalBytes = corpus.length * iters;
  const mbps = (totalBytes / 1e6) / (ms / 1000);
  console.log(`encode ×${iters}: ${ms.toFixed(0)} ms → ${mbps.toFixed(2)} MB/s, ${firstIds} ids/encode (${(corpus.length/firstIds).toFixed(2)} bytes/token)`);

  exp.ztok_free(out); exp.ztok_free(ip); exp.ztok_pipeline_free(pipe);
}

main().catch(e => { console.error(e); process.exit(1); });
