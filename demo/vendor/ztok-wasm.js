// ztok-wasm.js — ES-module loader for the ztok browser WASM build.
//
// The Zig build emits a freestanding wasm32 module with no imports. We
// instantiate it once per call to `loadZtok({wasmUrl})`, then expose an
// ergonomic Pipeline class that owns its native handle and a vocab byte
// buffer. JS never touches raw allocator pointers — every malloc is paired
// with a free in a try/finally, and Pipeline.free() is idempotent.
//
// The browser binding currently hard-wires the cl100k pre-tokenizer, so
// only cl100k-style `.tiktoken` files actually produce meaningful ids.
// We surface that constraint in the UI; this module deliberately doesn't
// pretend to support other vocab formats.

const SIMD_PROBE = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
  0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b,
  0x03, 0x02, 0x01, 0x00,
  0x0a, 0x16, 0x01, 0x14, 0x00,
  0xfd, 0x0c,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0x0b,
]);

export const hasSimd128 = () => WebAssembly.validate(SIMD_PROBE);

const ZTOK_OK = 0;

export class ZtokError extends Error {
  constructor(message, code) {
    super(message);
    this.name = "ZtokError";
    this.code = code;
  }
}

export class Pipeline {
  constructor(exports, handle) {
    this._exp = exports;
    this._handle = handle;
  }

  get version() {
    const exp = this._exp;
    const ptr = exp.ztok_version_ptr();
    const len = exp.ztok_version_len();
    return new TextDecoder().decode(new Uint8Array(exp.memory.buffer, ptr, len));
  }

  free() {
    if (this._handle !== 0) {
      this._exp.ztok_pipeline_free(this._handle);
      this._handle = 0;
    }
  }

  encode(text) {
    if (this._handle === 0) throw new ZtokError("pipeline freed", -1);
    const exp = this._exp;
    const input = new TextEncoder().encode(text);
    return this._withBuf(input, (ip) => {
      const out = exp.ztok_malloc(8);
      if (out === 0) throw new ZtokError("ztok_malloc returned 0", -1);
      try {
        const rc = exp.ztok_encode(this._handle, ip, input.length, out, out + 4);
        if (rc !== ZTOK_OK) throw new ZtokError(`encode rc=${rc}`, rc);
        const u32 = new Uint32Array(exp.memory.buffer);
        const idsPtr = u32[out >>> 2];
        const idsLen = u32[(out + 4) >>> 2];
        if (idsLen === 0) return new Uint32Array(0);
        const ids = new Uint32Array(idsLen);
        ids.set(new Uint32Array(exp.memory.buffer, idsPtr, idsLen));
        if (idsPtr !== 0) exp.ztok_free(idsPtr);
        return ids;
      } finally {
        exp.ztok_free(out);
      }
    });
  }

  decode(ids) {
    if (this._handle === 0) throw new ZtokError("pipeline freed", -1);
    const exp = this._exp;
    const buf = ids instanceof Uint32Array ? ids : new Uint32Array(ids);
    const view = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
    return this._withBuf(view, (ip) => {
      const out = exp.ztok_malloc(8);
      if (out === 0) throw new ZtokError("ztok_malloc returned 0", -1);
      try {
        const rc = exp.ztok_decode(this._handle, ip, buf.length, out, out + 4);
        if (rc !== ZTOK_OK) throw new ZtokError(`decode rc=${rc}`, rc);
        const u32 = new Uint32Array(exp.memory.buffer);
        const bptr = u32[out >>> 2];
        const blen = u32[(out + 4) >>> 2];
        if (blen === 0) return "";
        const bytes = new Uint8Array(blen);
        bytes.set(new Uint8Array(exp.memory.buffer, bptr, blen));
        if (bptr !== 0) exp.ztok_free(bptr);
        return new TextDecoder().decode(bytes);
      } finally {
        exp.ztok_free(out);
      }
    });
  }

  // Decode a *single* token id to its raw byte string. Useful for the
  // highlighter overlay where we want each token's literal text without
  // re-running the BPE merge pass.
  decodeOne(id) {
    return this.decode(new Uint32Array([id]));
  }

  _withBuf(bytes, fn) {
    const exp = this._exp;
    const ptr = exp.ztok_malloc(bytes.length);
    if (ptr === 0) throw new ZtokError("ztok_malloc returned 0", -1);
    new Uint8Array(exp.memory.buffer).set(bytes, ptr);
    try {
      return fn(ptr);
    } finally {
      exp.ztok_free(ptr);
    }
  }
}

// Single WASM module instance is reused across pipelines — each
// pipeline owns its own allocator slot inside the shared linear memory,
// and the binding's `gpa` is process-global within the wasm.
let _moduleExports = null;
let _modulePromise = null;

async function loadModule(wasmUrl) {
  if (_moduleExports) return _moduleExports;
  if (_modulePromise) return _modulePromise;
  _modulePromise = (async () => {
    const resp = await fetch(wasmUrl);
    if (!resp.ok) throw new ZtokError(`fetch ${wasmUrl} failed: ${resp.status}`, -1);
    const { instance } = await WebAssembly.instantiateStreaming(resp, {});
    _moduleExports = instance.exports;
    return _moduleExports;
  })();
  return _modulePromise;
}

export async function loadZtok({ wasmUrl }) {
  if (!hasSimd128() && !wasmUrl.includes("scalar")) {
    throw new ZtokError(
      "browser lacks WebAssembly SIMD128 — use the scalar WASM build (ztok-wasm-scalar.wasm)",
      -1,
    );
  }
  const exp = await loadModule(wasmUrl);
  return {
    version: (() => {
      const ptr = exp.ztok_version_ptr();
      const len = exp.ztok_version_len();
      return new TextDecoder().decode(new Uint8Array(exp.memory.buffer, ptr, len));
    })(),
    /**
     * Build a Pipeline from a cl100k-style tiktoken vocab byte buffer.
     * The byte buffer is consumed by the binding (copied into wasm
     * memory), so the caller may free its ArrayBuffer afterwards.
     */
    pipelineFromTiktokenBytes(bytes) {
      const ptr = exp.ztok_malloc(bytes.length);
      if (ptr === 0) throw new ZtokError("ztok_malloc returned 0", -1);
      new Uint8Array(exp.memory.buffer).set(bytes, ptr);
      const statusPtr = exp.ztok_malloc(4);
      if (statusPtr === 0) {
        exp.ztok_free(ptr);
        throw new ZtokError("ztok_malloc returned 0", -1);
      }
      try {
        const handle = exp.ztok_pipeline_new_bpe_from_tiktoken_bytes(
          ptr,
          bytes.length,
          statusPtr,
        );
        const st = new Int32Array(exp.memory.buffer, statusPtr, 1)[0];
        if (st !== ZTOK_OK || handle === 0) {
          throw new ZtokError(`pipeline_new status=${st}`, st);
        }
        return new Pipeline(exp, handle);
      } finally {
        exp.ztok_free(statusPtr);
        exp.ztok_free(ptr);
      }
    },
  };
}
