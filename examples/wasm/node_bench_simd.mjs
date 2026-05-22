// Node-side scalar-vs-SIMD wasm bench harness.
//
// Loads BOTH `ztok_browser.wasm` (compiled with `+simd128`) and
// `ztok_browser_scalar.wasm` (compiled without), runs the same
// encode workload through each, and prints throughput in MB/s so we
// can quote the SIMD lift in CHANGELOG / RESULTS.md without needing
// a browser tab.
//
// Usage:
//   node examples/wasm/node_bench_simd.mjs [path-to-cl100k.tiktoken] [iters]
//
// Without a vocab path, synthesizes a tiny 268-entry vocab — exercises
// the pre-tokenizer + scanMin loop but no real BPE merges, so numbers
// are only directly comparable against the same harness invocation.

import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SIMD_PATH = path.join(__dirname, "..", "..", "zig-out", "bin", "ztok_browser.wasm");
const SCALAR_PATH = path.join(__dirname, "..", "..", "zig-out", "bin", "ztok_browser_scalar.wasm");
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

// A richer synthetic vocab built by mining all 2..6 char ascii bigrams /
// trigrams / etc. from the corpus and keeping the most frequent.
// Produces ~5K merges, so the BPE scanMin loop runs many merges per
// chunk — that's where SIMD lift shows up. Mirrors what cl100k would
// do without requiring the actual vocab file.
function richSynthVocab(corpus, maxExtras = 5000) {
  const counts = new Map();
  const text = Buffer.from(corpus).toString("binary"); // byte-identity
  for (let len = 2; len <= 6; len++) {
    for (let i = 0; i + len <= text.length; i++) {
      // Only ASCII / common punctuation; skip pieces touching whitespace
      // weirdly to keep it tiktoken-shaped.
      const s = text.slice(i, i + len);
      counts.set(s, (counts.get(s) || 0) + 1);
    }
  }
  const ranked = [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, maxExtras);

  const lines = [];
  for (let b = 0; b < 256; b++) lines.push(`${Buffer.from([b]).toString("base64")} ${b}`);
  let r = 256;
  for (const [s] of ranked) {
    lines.push(`${Buffer.from(s, "binary").toString("base64")} ${r++}`);
  }
  return new TextEncoder().encode(lines.join("\n") + "\n");
}

// WebAssembly.validate feature-test for simd128.
// Module: imports nothing; exports one fn whose return type is v128
// and whose body returns `v128.const 0...0`. Runtimes without
// simd128 reject the v128 type and / or the 0xFD 0x0C opcode.
//
// Layout: type (() -> v128) + 1 func + code section with body
//   locals_count(1) + v128.const(2) + immediate(16) + end(1) = 20 B
const SIMD_PROBE = new Uint8Array([
  0x00,0x61,0x73,0x6d, 0x01,0x00,0x00,0x00,
  0x01,0x05,0x01,0x60,0x00,0x01,0x7b,  // type: () -> v128 (0x7b)
  0x03,0x02,0x01,0x00,                  // func 0 uses type 0
  0x0a,0x16,0x01,0x14,0x00,             // code: sec_size=22, 1 func, body=20, locals=0
  0xfd,0x0c,                            // v128.const = 0xFD 0x0C
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,      // 16-byte immediate
  0x0b,                                 // end
]);

async function makeRunner(wasmPath, label) {
  if (!existsSync(wasmPath)) {
    console.log(`SKIP ${label}: ${wasmPath} not found`);
    return null;
  }
  const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {});
  const exp = instance.exports;
  const memU8 = () => new Uint8Array(exp.memory.buffer);
  const memI32 = () => new Int32Array(exp.memory.buffer);
  const memU32 = () => new Uint32Array(exp.memory.buffer);
  return { label, wasmPath, exp, memU8, memI32, memU32 };
}

function runBench(r, vocab, corpus, iters) {
  const { exp, memU8, memI32, memU32 } = r;

  const vp = exp.ztok_malloc(vocab.length);
  memU8().set(vocab, vp);
  const sp = exp.ztok_malloc(4);
  const pipe = exp.ztok_pipeline_new_bpe_from_tiktoken_bytes(vp, vocab.length, sp);
  const st = memI32()[sp >>> 2];
  if (st !== 0 || pipe === 0) throw new Error(`${r.label} load: status=${st}`);
  exp.ztok_free(vp); exp.ztok_free(sp);

  const ip = exp.ztok_malloc(corpus.length);
  memU8().set(corpus, ip);
  const out = exp.ztok_malloc(8);

  // Warmup.
  let firstIds = 0;
  for (let i = 0; i < 3; i++) {
    const rc = exp.ztok_encode(pipe, ip, corpus.length, out, out + 4);
    if (rc !== 0) throw new Error(`${r.label} encode rc=${rc}`);
    const idsPtr = memU32()[out >>> 2];
    firstIds = memU32()[(out+4) >>> 2];
    if (idsPtr !== 0) exp.ztok_free(idsPtr);
  }

  const t0 = performance.now();
  for (let i = 0; i < iters; i++) {
    const rc = exp.ztok_encode(pipe, ip, corpus.length, out, out + 4);
    if (rc !== 0) throw new Error(`${r.label} encode rc=${rc}`);
    const idsPtr = memU32()[out >>> 2];
    if (idsPtr !== 0) exp.ztok_free(idsPtr);
  }
  const ms = performance.now() - t0;
  const totalBytes = corpus.length * iters;
  const mbps = (totalBytes / 1e6) / (ms / 1000);

  exp.ztok_free(out); exp.ztok_free(ip); exp.ztok_pipeline_free(pipe);
  return { ms, mbps, ids: firstIds };
}

async function main() {
  const argv = process.argv.slice(2);
  const vocabPath = argv.find((a) => !/^\d+$/.test(a));
  const iters = parseInt(argv.find((a) => /^\d+$/.test(a)) || "20", 10);

  // SIMD feature probe — sanity check that the host runtime honors
  // simd128. Modern Node (>= 16.4) does; very old runtimes may not.
  const hostHasSimd = WebAssembly.validate(SIMD_PROBE);
  console.log(`host runtime supports wasm simd128: ${hostHasSimd}`);
  if (!hostHasSimd) {
    console.error("This Node runtime can't run the SIMD wasm. Upgrade to Node >= 16.4.");
    process.exit(1);
  }

  const corpus = readFileSync(CORPUS);
  const vocab = vocabPath ? readFileSync(vocabPath) : richSynthVocab(corpus, 5000);
  // Optional: a heavy corpus of long unbroken runs to exercise the
  // BPE inner-loop scanMin path (cl100k splits make most chunks
  // 2-20 bytes, so the SIMD vector body never fills — but base64 /
  // identifier-heavy data routinely hits 32+).
  let benchCorpus = corpus;
  if (process.env.HEAVY === "1") {
    const heavy = Buffer.from(corpus).toString("base64");
    benchCorpus = new TextEncoder().encode(heavy.repeat(2));
  } else if (process.env.HEAVY === "2") {
    // Super heavy: long unbroken runs designed to push live counts
    // past the V=16 threshold so the scanMin vector body actually
    // fires. Each segment is ~200 chars of one byte plus a varied
    // 16-byte alphabet that admits long merge chains.
    const seg = "x".repeat(200) + "y".repeat(200) + "abcdefghijklmnop".repeat(200);
    benchCorpus = new TextEncoder().encode(seg.repeat(20));
  }
  console.log(`vocab: ${vocab.length} bytes${vocabPath ? " from " + vocabPath : " (synthetic ~5K-merge)"}`);
  console.log(`corpus: ${benchCorpus.length} bytes${process.env.HEAVY==="1" ? " (HEAVY base64)" : ""}, iters: ${iters}`);

  const simd = await makeRunner(SIMD_PATH, "SIMD128");
  const scalar = await makeRunner(SCALAR_PATH, "scalar ");

  let rSimd = null, rScalar = null;
  if (simd) { rSimd = runBench(simd, vocab, benchCorpus, iters);
    console.log(`SIMD128  : ${rSimd.ms.toFixed(0)} ms  ${rSimd.mbps.toFixed(2)} MB/s  ${rSimd.ids} ids/encode`); }
  if (scalar) { rScalar = runBench(scalar, vocab, benchCorpus, iters);
    console.log(`scalar   : ${rScalar.ms.toFixed(0)} ms  ${rScalar.mbps.toFixed(2)} MB/s  ${rScalar.ids} ids/encode`); }

  if (rSimd && rScalar) {
    const lift = rSimd.mbps / rScalar.mbps;
    console.log(`\nSIMD128 lift over scalar wasm: ${lift.toFixed(2)}x`);
    if (rSimd.ids !== rScalar.ids) {
      console.error(`WARN: id counts differ — encode result not identical between builds!`);
      process.exit(2);
    }
  }
}

main().catch(e => { console.error(e); process.exit(1); });
