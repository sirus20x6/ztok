'use strict';

// ztok — Node.js bindings for libztok.
//
// Thin koffi wrapper around the C ABI. Mirrors the Python binding's
// design: zero non-stdlib deps beyond koffi, opaque handles managed via
// FinalizationRegistry so dropped objects still release their C resources
// on GC, but explicit close() is strongly preferred for determinism.
//
// All id buffers returned by `ztok_encode_batch_pooled` / `ztok_stream_*`
// carry a length-prefix header (see src/c_api.zig::allocIdBuf). The ONLY
// safe free is `ztok_ids_free`. We never copy id arrays unnecessarily —
// we decode straight into a Uint32Array, then free the C buffer.

const fs = require('fs');
const path = require('path');

const ffi = require('./ffi');
const { ZtokLibraryNotFoundError } = require('./lib');

// --- exceptions ---

class ZtokError extends Error {
    constructor(message) {
        super(message);
        this.name = 'ZtokError';
    }
}

class ZtokInvalidInputError extends ZtokError {
    constructor(message) {
        super(message);
        this.name = 'ZtokInvalidInputError';
    }
}

class ZtokOutOfMemoryError extends ZtokError {
    constructor(message) {
        super(message);
        this.name = 'ZtokOutOfMemoryError';
    }
}

class ZtokBufferTooSmallError extends ZtokError {
    constructor(message) {
        super(message);
        this.name = 'ZtokBufferTooSmallError';
    }
}

class ZtokInternalError extends ZtokError {
    constructor(message) {
        super(message);
        this.name = 'ZtokInternalError';
    }
}

const ERROR_MAP = {
    [ffi.ZTOK_ERR_OUT_OF_MEMORY]: ZtokOutOfMemoryError,
    [ffi.ZTOK_ERR_INVALID_INPUT]: ZtokInvalidInputError,
    [ffi.ZTOK_ERR_BUFFER_TOO_SMALL]: ZtokBufferTooSmallError,
    [ffi.ZTOK_ERR_INTERNAL]: ZtokInternalError,
};

function raiseForStatus(status, ctx) {
    if (status === ffi.ZTOK_OK) return;
    const Cls = ERROR_MAP[status] || ZtokInternalError;
    throw new Cls(`${ctx}: ztok status ${status}`);
}

// --- version ---

function version() {
    const { ztok_version } = ffi.getLib();
    const raw = ztok_version();
    if (!raw) throw new ZtokInternalError('ztok_version returned NULL');
    return raw;
}

// --- FinalizationRegistry for handle cleanup ---
//
// One global registry per handle kind. The held-value is the function
// that frees the C handle when the JS wrapper is GC'd without close().

const pipelineRegistry = new FinalizationRegistry((handle) => {
    if (!handle) return;
    try {
        ffi.getLib().ztok_pipeline_free(handle);
    } catch (_) {
        // best-effort during GC; library may already be torn down
    }
});

const poolRegistry = new FinalizationRegistry((handle) => {
    if (!handle) return;
    try {
        ffi.getLib().ztok_batch_pool_free(handle);
    } catch (_) { /* ignore */ }
});

const streamRegistry = new FinalizationRegistry((handle) => {
    if (!handle) return;
    try {
        ffi.getLib().ztok_stream_free(handle);
    } catch (_) { /* ignore */ }
});

// --- BatchPool ---

class BatchPool {
    /**
     * @param {{ workers?: number }} [opts]
     */
    constructor(opts = {}) {
        const workers = opts.workers ?? 0;
        if (!Number.isInteger(workers) || workers < 0) {
            throw new TypeError('workers must be a non-negative integer (0 = auto)');
        }
        const lib = ffi.getLib();
        const status = [0];
        const handle = lib.ztok_batch_pool_new(workers, status);
        raiseForStatus(status[0], 'ztok_batch_pool_new');
        if (!handle) throw new ZtokInternalError('ztok_batch_pool_new returned NULL');

        this._handle = handle;
        this._closed = false;
        poolRegistry.register(this, handle, this);
    }

    /** Actual worker count (resolves workers=0 to detected CPU count). */
    get workers() {
        this._check();
        return Number(ffi.getLib().ztok_batch_pool_worker_count(this._handle));
    }

    /** Release the pool's worker threads + arenas. Idempotent. */
    close() {
        if (this._closed) return;
        this._closed = true;
        poolRegistry.unregister(this);
        try {
            ffi.getLib().ztok_batch_pool_free(this._handle);
        } finally {
            this._handle = null;
        }
    }

    _check() {
        if (this._closed) throw new ZtokError('BatchPool is closed');
    }

    _raw() {
        this._check();
        return this._handle;
    }
}

// --- Pipeline ---

function makeConfig({
    normalizer = ffi.NORMALIZER_IDENTITY,
    pre_tokenizer = ffi.PRETOK_IDENTITY,
    model = ffi.MODEL_BYTE_ID,
    decoder = ffi.DECODER_CONCAT,
} = {}) {
    return { normalizer, pre_tokenizer, model, decoder };
}

class Pipeline {
    constructor(handle) {
        if (!handle) throw new ZtokInternalError('Pipeline constructed with NULL handle');
        this._handle = handle;
        this._closed = false;
        pipelineRegistry.register(this, handle, this);
    }

    // --- constructors ---

    /** Baseline byte_id pipeline (each input byte maps to its own id). */
    static byteId(opts = {}) {
        const lib = ffi.getLib();
        const cfg = makeConfig({
            normalizer: opts.normalizer ?? ffi.NORMALIZER_IDENTITY,
            pre_tokenizer: opts.preTokenizer ?? ffi.PRETOK_IDENTITY,
            model: ffi.MODEL_BYTE_ID,
            decoder: opts.decoder ?? ffi.DECODER_CONCAT,
        });
        const status = [0];
        const handle = lib.ztok_pipeline_new(cfg, status);
        raiseForStatus(status[0], 'ztok_pipeline_new');
        return new Pipeline(handle);
    }

    /** Load a .tiktoken vocab into a byte-level BPE pipeline. */
    static fromTiktoken(filePath, opts = {}) {
        const cl100k = opts.cl100k !== false; // default true
        return Pipeline._fromFileWithCfg(
            'ztok_pipeline_new_bpe_from_tiktoken',
            filePath,
            {
                normalizer: opts.normalizer ?? ffi.NORMALIZER_IDENTITY,
                pre_tokenizer: cl100k ? ffi.PRETOK_CL100K : ffi.PRETOK_IDENTITY,
                model: ffi.MODEL_BYTE_ID,
                decoder: opts.decoder ?? ffi.DECODER_CONCAT,
            }
        );
    }

    /** Load a HuggingFace tokenizer.json BPE model. */
    static fromHfJson(filePath, opts = {}) {
        const cl100k = opts.cl100k === true; // default false
        return Pipeline._fromFileWithCfg(
            'ztok_pipeline_new_bpe_from_hf_json',
            filePath,
            {
                normalizer: opts.normalizer ?? ffi.NORMALIZER_IDENTITY,
                pre_tokenizer: cl100k ? ffi.PRETOK_CL100K : ffi.PRETOK_IDENTITY,
                model: ffi.MODEL_BYTE_ID,
                decoder: opts.decoder ?? ffi.DECODER_CONCAT,
            }
        );
    }

    /** Load a HuggingFace WordPiece model from tokenizer.json. */
    static fromWordPiece(filePath, opts = {}) {
        if (!Number.isInteger(opts.unkId)) {
            throw new TypeError('fromWordPiece: opts.unkId (uint32) is required');
        }
        const lib = ffi.getLib();
        const cfg = makeConfig({
            normalizer: opts.normalizer ?? ffi.NORMALIZER_IDENTITY,
            pre_tokenizer: opts.preTokenizer ?? ffi.PRETOK_IDENTITY,
            model: ffi.MODEL_BYTE_ID,
            decoder: opts.decoder ?? ffi.DECODER_WORDPIECE,
        });
        const status = [0];
        const handle = lib.ztok_pipeline_new_wordpiece_from_hf_json(
            filePath, opts.unkId, cfg, status
        );
        raiseForStatus(status[0], 'ztok_pipeline_new_wordpiece_from_hf_json');
        return new Pipeline(handle);
    }

    /** Load a SentencePiece .model (Unigram) file. */
    static fromSentencePiece(filePath, opts = {}) {
        const unkId = opts.unkId ?? 0;
        const lib = ffi.getLib();
        const cfg = makeConfig({
            normalizer: opts.normalizer ?? ffi.NORMALIZER_IDENTITY,
            pre_tokenizer: opts.preTokenizer ?? ffi.PRETOK_IDENTITY,
            model: ffi.MODEL_BYTE_ID,
            decoder: opts.decoder ?? ffi.DECODER_CONCAT,
        });
        const status = [0];
        const handle = lib.ztok_pipeline_new_unigram_from_sp_model(
            filePath, unkId, cfg, status
        );
        raiseForStatus(status[0], 'ztok_pipeline_new_unigram_from_sp_model');
        return new Pipeline(handle);
    }

    /** Load a ztok TokenMonster .ztm vocab file. */
    static fromMonster(filePath, opts = {}) {
        return Pipeline._fromFileWithCfg(
            'ztok_pipeline_new_monster_from_file',
            filePath,
            {
                normalizer: opts.normalizer ?? ffi.NORMALIZER_IDENTITY,
                pre_tokenizer: opts.preTokenizer ?? ffi.PRETOK_IDENTITY,
                model: ffi.MODEL_BYTE_ID,
                decoder: opts.decoder ?? ffi.DECODER_CONCAT,
            }
        );
    }

    /**
     * Load an RWKV "World" vocab (rwkv_vocab_v20230424.txt).
     *
     * The model is a greedy longest-match byte trie that runs over the
     * whole input as a single span — there is no pre-tokenizer, so the
     * pre-tokenizer slot is fixed to identity.
     */
    static fromRWKV(filePath, opts = {}) {
        return Pipeline._fromFileWithCfg(
            'ztok_pipeline_new_rwkv_from_file',
            filePath,
            {
                normalizer: opts.normalizer ?? ffi.NORMALIZER_IDENTITY,
                pre_tokenizer: ffi.PRETOK_IDENTITY,
                model: ffi.MODEL_BYTE_ID,
                decoder: opts.decoder ?? ffi.DECODER_CONCAT,
            }
        );
    }

    /**
     * Auto-detect the file format and dispatch to the right loader.
     * - .tiktoken          -> BPE + cl100k pre-tokenizer
     * - tokenizer.json     -> BPE (HF JSON)
     * - .model             -> SentencePiece Unigram (unkId defaults to 0)
     * - .ztm               -> TokenMonster
     */
    static fromPath(filePath, opts = {}) {
        const fmt = ffi.detectFormat(filePath);
        switch (fmt) {
            case 'tiktoken':
                return Pipeline.fromTiktoken(filePath, opts);
            case 'hf_json':
                return Pipeline.fromHfJson(filePath, opts);
            case 'sentencepiece':
                return Pipeline.fromSentencePiece(filePath, opts);
            case 'ztm':
                return Pipeline.fromMonster(filePath, opts);
            case 'rwkv':
                return Pipeline.fromRWKV(filePath, opts);
            default:
                throw new ZtokInvalidInputError(
                    `could not auto-detect tokenizer format for ${filePath}; ` +
                    'use a specific from* constructor instead'
                );
        }
    }

    static _fromFileWithCfg(fnName, filePath, cfg) {
        const lib = ffi.getLib();
        const status = [0];
        const handle = lib[fnName](filePath, cfg, status);
        raiseForStatus(status[0], fnName);
        return new Pipeline(handle);
    }

    // --- lifecycle ---

    /** Free the underlying pipeline. Idempotent. */
    close() {
        if (this._closed) return;
        this._closed = true;
        pipelineRegistry.unregister(this);
        try {
            ffi.getLib().ztok_pipeline_free(this._handle);
        } finally {
            this._handle = null;
        }
    }

    _check() {
        if (this._closed) throw new ZtokError('Pipeline is closed');
    }

    _raw() {
        this._check();
        return this._handle;
    }

    // --- encode / decode ---

    /**
     * Encode a string (UTF-8) or Buffer into a Uint32Array of token ids.
     * @param {string | Buffer | Uint8Array} text
     * @returns {Uint32Array}
     */
    encode(text) {
        this._check();
        const lib = ffi.getLib();
        const handle = this._handle;
        const data = typeof text === 'string'
            ? Buffer.from(text, 'utf-8')
            : Buffer.isBuffer(text) ? text : Buffer.from(text);
        if (data.length === 0) return new Uint32Array(0);

        // The C ABI's per-span maxTokensFor bound is conservative, so a
        // buffer sized to the encoded length can still trip
        // BUFFER_TOO_SMALL mid-stream. Start generous, grow on demand.
        let cap = Math.max(data.length + 16, 64);
        for (let attempt = 0; attempt < 8; attempt++) {
            const buf = new Uint32Array(cap);
            const outLen = [0];
            const rc = lib.ztok_encode(handle, data, data.length, buf, cap, outLen);
            if (rc === ffi.ZTOK_OK) {
                return buf.slice(0, Number(outLen[0]));
            }
            if (rc === ffi.ZTOK_ERR_BUFFER_TOO_SMALL) {
                cap = Math.max(cap * 2, Number(outLen[0]) + 16);
                continue;
            }
            raiseForStatus(rc, 'ztok_encode');
        }
        throw new ZtokInternalError(
            'ztok_encode kept reporting BUFFER_TOO_SMALL after 8 grow attempts'
        );
    }

    /**
     * Encode `text` and return ids plus per-token overlay channels.
     *
     * `channels` is an array of overlay-kind codes (the `OVERLAY_*`
     * constants). Requesting overlays never changes tokenization — the id
     * stream is identical to {@link encode}.
     *
     * Returns `{ ids, overlays }` where `ids` is a Uint32Array and
     * `overlays` is a Map<number, Uint32Array> keyed by overlay kind; each
     * channel array has exactly `ids.length` entries. Cheap channels
     * (BYTE_START/BYTE_END/BOUNDARY/PROVENANCE) carry encoder-derived
     * values; domain channels (OPCODE/OPERAND/SYMBOL_REF/HUNK) come back
     * zero-filled until a domain plugin populates them.
     *
     * Mirrors the C ABI sizing protocol: a first call with out_ids = NULL
     * queries the token count, then buffers are allocated and a second
     * call fills them.
     *
     * @param {string | Buffer | Uint8Array} text
     * @param {number[]} channels
     * @returns {{ ids: Uint32Array, overlays: Map<number, Uint32Array> }}
     */
    encodeWithOverlays(text, channels) {
        this._check();
        const lib = ffi.getLib();
        const koffi = lib.koffi;
        const handle = this._handle;

        const kinds = Array.from(channels ?? []);
        if (new Set(kinds).size !== kinds.length) {
            throw new ZtokInvalidInputError(
                'encodeWithOverlays: duplicate overlay kinds requested'
            );
        }

        const data = typeof text === 'string'
            ? Buffer.from(text, 'utf-8')
            : Buffer.isBuffer(text) ? text : Buffer.from(text);

        const emptyOverlays = () => {
            const m = new Map();
            for (const k of kinds) m.set(k, new Uint32Array(0));
            return m;
        };

        if (data.length === 0) {
            return { ids: new Uint32Array(0), overlays: emptyOverlays() };
        }

        const n = kinds.length;

        // Sizing pass: out_ids = NULL, channels carry NULL out buffers.
        const sizeChannels = kinds.map((kind) => ({ kind, out: null, out_cap: 0 }));
        const outLen = [0];
        let rc = lib.ztok_encode_with_overlays(
            handle, data, data.length, null, 0,
            n ? sizeChannels : null, n, outLen
        );
        if (rc !== ffi.ZTOK_OK && rc !== ffi.ZTOK_ERR_BUFFER_TOO_SMALL) {
            raiseForStatus(rc, 'ztok_encode_with_overlays (sizing)');
        }
        const count = Number(outLen[0]);
        if (count === 0) {
            return { ids: new Uint32Array(0), overlays: emptyOverlays() };
        }

        // Fill pass: allocate an ids buffer plus one uint32 buffer per
        // requested channel, each sized to the exact token count.
        const idBuf = koffi.alloc('uint32_t', count);
        const chanBufs = kinds.map(() => koffi.alloc('uint32_t', count));
        const fillChannels = kinds.map((kind, i) => ({
            kind, out: chanBufs[i], out_cap: count,
        }));
        const outLen2 = [0];
        rc = lib.ztok_encode_with_overlays(
            handle, data, data.length, idBuf, count,
            n ? fillChannels : null, n, outLen2
        );
        raiseForStatus(rc, 'ztok_encode_with_overlays');

        const len = Number(outLen2[0]);
        const ids = Uint32Array.from(koffi.decode(idBuf, koffi.array('uint32_t', len)));
        const overlays = new Map();
        for (let i = 0; i < n; i++) {
            const vals = koffi.decode(chanBufs[i], koffi.array('uint32_t', len));
            overlays.set(kinds[i], Uint32Array.from(vals));
        }
        return { ids, overlays };
    }

    /**
     * Decode a Uint32Array (or array of ints) into a UTF-8 string.
     * @param {Uint32Array | number[]} ids
     * @returns {string}
     */
    decode(ids) {
        return this.decodeBytes(ids).toString('utf-8');
    }

    /**
     * Decode token ids into raw bytes (no UTF-8 round-tripping).
     * @param {Uint32Array | number[]} ids
     * @returns {Buffer}
     */
    decodeBytes(ids) {
        this._check();
        const lib = ffi.getLib();
        const handle = this._handle;
        const idArr = ids instanceof Uint32Array ? ids : Uint32Array.from(ids);
        if (idArr.length === 0) return Buffer.alloc(0);

        // Sizing pass: null out arg is not safe in koffi, so we send a
        // 1-byte buffer and accept the BUFFER_TOO_SMALL signal.
        const outLen = [0];
        const probe = Buffer.alloc(1);
        const rc1 = lib.ztok_decode(handle, idArr, idArr.length, probe, 0, outLen);
        if (rc1 !== ffi.ZTOK_OK && rc1 !== ffi.ZTOK_ERR_BUFFER_TOO_SMALL) {
            raiseForStatus(rc1, 'ztok_decode (sizing)');
        }
        const nbytes = Number(outLen[0]);
        if (nbytes === 0) return Buffer.alloc(0);

        const outBuf = Buffer.alloc(nbytes);
        const rc2 = lib.ztok_decode(handle, idArr, idArr.length, outBuf, nbytes, outLen);
        raiseForStatus(rc2, 'ztok_decode');
        return outBuf.subarray(0, Number(outLen[0]));
    }

    // --- batch ---

    /**
     * Encode many strings in parallel via a persistent BatchPool.
     * Each per-input id array is materialized into a Uint32Array, then
     * the C-owned buffer is freed via ztok_ids_free (the only safe path).
     *
     * @param {BatchPool} pool
     * @param {(string | Buffer | Uint8Array)[]} inputs
     * @returns {Uint32Array[]}
     */
    encodeBatch(pool, inputs) {
        this._check();
        if (!(pool instanceof BatchPool)) {
            throw new TypeError('encodeBatch: first arg must be a BatchPool');
        }
        const lib = ffi.getLib();
        const handle = this._handle;
        const poolHandle = pool._raw();

        const n = inputs.length;
        if (n === 0) return [];

        // Convert each input to a NUL-terminated Buffer so koffi sees a
        // valid C string. We keep the Buffers alive in the array until
        // the call returns.
        const bufs = inputs.map((s) => {
            if (typeof s === 'string') {
                const b = Buffer.alloc(Buffer.byteLength(s, 'utf-8') + 1);
                b.write(s, 0, 'utf-8');
                return b;
            }
            if (Buffer.isBuffer(s)) {
                // Ensure NUL termination
                const b = Buffer.alloc(s.length + 1);
                s.copy(b);
                return b;
            }
            const src = Buffer.from(s);
            const b = Buffer.alloc(src.length + 1);
            src.copy(b);
            return b;
        });
        const lens = inputs.map((s) => {
            if (typeof s === 'string') return Buffer.byteLength(s, 'utf-8');
            if (Buffer.isBuffer(s) || s instanceof Uint8Array) return s.length;
            return Buffer.byteLength(String(s), 'utf-8');
        });

        const outIds = new Array(n).fill(null);
        const outLens = new Array(n).fill(0);

        const rc = lib.ztok_encode_batch_pooled(
            handle, poolHandle, bufs, lens, n, outIds, outLens
        );

        try {
            raiseForStatus(rc, 'ztok_encode_batch_pooled');
            const results = new Array(n);
            for (let i = 0; i < n; i++) {
                const length = Number(outLens[i]);
                const ptr = outIds[i];
                if (length === 0 || !ptr) {
                    results[i] = new Uint32Array(0);
                    continue;
                }
                // koffi.decode copies the buffer into a JS-owned typed
                // array, so we can safely free the C side below.
                const decoded = ffi.getLib().koffi.decode(
                    ptr, ffi.getLib().koffi.array('uint32_t', length)
                );
                // decoded is a plain Array<number> from koffi; copy into
                // a Uint32Array for the public typed-array contract.
                results[i] = Uint32Array.from(decoded);
            }
            return results;
        } finally {
            // ONLY safe free path — the buffers carry a length-prefix
            // header (see src/c_api.zig::allocIdBuf).
            for (let i = 0; i < n; i++) {
                if (outIds[i]) {
                    try { lib.ztok_ids_free(outIds[i]); } catch (_) { /* ignore */ }
                }
            }
        }
    }

    // --- streaming ---

    /**
     * Stream-encode text by chopping it into chunkSize-byte feeds. Each
     * yield is a Uint32Array of ids newly emitted by that feed. A final
     * flush via ztok_stream_finish drains any remaining carry. Empty
     * yields are skipped so callers only see non-empty chunks.
     *
     * Internally wraps ztok_stream_{new,feed,finish,free}. The encoder
     * defers a trailing partial UTF-8 codepoint / pre-tokenizer span up
     * to a soft cap of 1 MiB (see src/stream.zig); past that it
     * force-cuts at the nearest codepoint boundary.
     *
     * @param {string | Buffer | Uint8Array} text
     * @param {{ chunkSize?: number }} [opts]
     * @returns {Generator<Uint32Array>}
     */
    *encodeStream(text, opts = {}) {
        this._check();
        const chunkSize = opts.chunkSize ?? 64 * 1024;
        if (!Number.isInteger(chunkSize) || chunkSize <= 0) {
            throw new RangeError('chunkSize must be a positive integer');
        }
        const lib = ffi.getLib();
        const handle = this._handle;
        const data = typeof text === 'string'
            ? Buffer.from(text, 'utf-8')
            : Buffer.isBuffer(text) ? text : Buffer.from(text);

        const status = [0];
        const streamHandle = lib.ztok_stream_new(handle, status);
        raiseForStatus(status[0], 'ztok_stream_new');
        if (!streamHandle) throw new ZtokInternalError('ztok_stream_new returned NULL');

        try {
            for (let i = 0; i < data.length; i += chunkSize) {
                const chunk = data.subarray(i, Math.min(i + chunkSize, data.length));
                const outIds = [null];
                const outN = [0];
                const rc = lib.ztok_stream_feed(
                    streamHandle, chunk, chunk.length, outIds, outN
                );
                raiseForStatus(rc, 'ztok_stream_feed');
                const ids = materializeAndFreeIds(lib, outIds[0], Number(outN[0]));
                if (ids.length > 0) yield ids;
            }

            // Final flush.
            const outIds = [null];
            const outN = [0];
            const rc = lib.ztok_stream_finish(streamHandle, outIds, outN);
            raiseForStatus(rc, 'ztok_stream_finish');
            const ids = materializeAndFreeIds(lib, outIds[0], Number(outN[0]));
            if (ids.length > 0) yield ids;
        } finally {
            try { lib.ztok_stream_free(streamHandle); } catch (_) { /* ignore */ }
        }
    }

    // --- chunking ---

    /**
     * Split `text` into overlapping token windows (late chunking). Each
     * window holds at most `maxTokens` ids with `overlap` ids shared
     * between neighbors (stride = maxTokens - overlap). `boundary` is one
     * of the CHUNK_BOUNDARY_* constants. Returns [] for empty input.
     *
     * Mirrors the C ABI sizing protocol: a first call with out_chunks =
     * NULL queries the chunk count, then a record array is allocated and a
     * second call fills it. The C-owned per-record id buffers are copied
     * into Uint32Arrays and freed via ztok_chunks_free before returning.
     *
     * @param {string | Buffer | Uint8Array} text
     * @param {number} maxTokens
     * @param {{ overlap?: number, boundary?: number }} [opts]
     * @returns {{ ids: Uint32Array, byteStart: number, byteEnd: number, tokenStart: number, tokenEnd: number }[]}
     */
    chunk(text, maxTokens, opts = {}) {
        this._check();
        const overlap = opts.overlap ?? 0;
        const boundary = opts.boundary ?? ffi.CHUNK_BOUNDARY_TOKEN;
        if (!Number.isInteger(maxTokens) || maxTokens <= 0
            || !Number.isInteger(overlap) || overlap < 0 || overlap >= maxTokens) {
            throw new ZtokInvalidInputError(
                'chunk: maxTokens must be > 0 and 0 <= overlap < maxTokens'
            );
        }

        const lib = ffi.getLib();
        const koffi = lib.koffi;
        const handle = this._handle;
        const data = typeof text === 'string'
            ? Buffer.from(text, 'utf-8')
            : Buffer.isBuffer(text) ? text : Buffer.from(text);
        if (data.length === 0) return [];

        // Sizing pass: out_chunks = NULL -> *out_len = chunk count.
        const outLen = [0];
        let rc = lib.ztok_chunk(
            handle, data, data.length,
            maxTokens, overlap, boundary,
            null, 0, outLen
        );
        if (rc !== ffi.ZTOK_OK && rc !== ffi.ZTOK_ERR_BUFFER_TOO_SMALL) {
            raiseForStatus(rc, 'ztok_chunk (sizing)');
        }
        const count = Number(outLen[0]);
        if (count === 0) return [];

        // Fill pass: allocate a record array sized to the chunk count.
        const recs = koffi.alloc('ztok_chunk_rec', count);
        const outLen2 = [0];
        rc = lib.ztok_chunk(
            handle, data, data.length,
            maxTokens, overlap, boundary,
            recs, count, outLen2
        );
        const got = Number(outLen2[0]);
        try {
            raiseForStatus(rc, 'ztok_chunk');
            const decoded = koffi.decode(recs, koffi.array('ztok_chunk_rec', got));
            const out = new Array(got);
            for (let i = 0; i < got; i++) {
                const r = decoded[i];
                const idsLen = Number(r.ids_len);
                let ids;
                if (r.ids && idsLen > 0) {
                    const vals = koffi.decode(r.ids, koffi.array('uint32_t', idsLen));
                    ids = Uint32Array.from(vals);
                } else {
                    ids = new Uint32Array(0);
                }
                out[i] = {
                    ids,
                    byteStart: r.byte_start,
                    byteEnd: r.byte_end,
                    tokenStart: r.token_start,
                    tokenEnd: r.token_end,
                };
            }
            return out;
        } finally {
            // Release each record's ztok-allocated id buffer. The recs
            // array itself is JS-owned (koffi.alloc).
            try { lib.ztok_chunks_free(recs, got); } catch (_) { /* ignore */ }
        }
    }
}

function materializeAndFreeIds(lib, ptr, n) {
    if (!ptr || n <= 0) return new Uint32Array(0);
    const decoded = lib.koffi.decode(ptr, lib.koffi.array('uint32_t', n));
    const out = Uint32Array.from(decoded);
    try { lib.ztok_ids_free(ptr); } catch (_) { /* ignore */ }
    return out;
}

// Decode `n` u64 hashes at `ptr` into a BigUint64Array. uint64 values
// routinely exceed 2^53, so we read raw little-endian bytes rather than
// koffi's number-decode (which would lose precision on big hashes).
function decodeU64s(koffi, ptr, n) {
    if (!ptr || n <= 0) return new BigUint64Array(0);
    const bytes = koffi.decode(ptr, koffi.array('uint8_t', n * 8, 'Array'));
    const buf = Buffer.from(bytes);
    // Copy into a fresh BigUint64Array (Buffer.buffer may be a larger
    // pooled ArrayBuffer; constructing over the exact slice is safest).
    const out = new BigUint64Array(n);
    for (let i = 0; i < n; i++) out[i] = buf.readBigUInt64LE(i * 8);
    return out;
}

// --- Engram n-gram hashing ---

/**
 * Hash every length-`n` window of `ids` under `heads` hash functions,
 * returning the row-major [position][head] uint64 hashes as a
 * BigUint64Array (positions = ids.length - n + 1, or 0 if the stream is
 * shorter than one window). Mask each hash to your table width
 * (hash & ((1n << bits) - 1n)). Deterministic: identical ids always
 * yield identical hashes. Operates on raw ids — no Pipeline needed.
 *
 * @param {Uint32Array | number[]} ids
 * @param {number} n
 * @param {number} heads
 * @returns {BigUint64Array}
 */
function hashNgrams(ids, n, heads) {
    if (!Number.isInteger(n) || !Number.isInteger(heads) || n <= 0 || heads <= 0) {
        return new BigUint64Array(0);
    }
    const idArr = ids instanceof Uint32Array ? ids : Uint32Array.from(ids);
    const nIds = idArr.length;
    if (nIds < n) return new BigUint64Array(0);
    const positions = nIds - n + 1;
    let want = positions * heads;
    if (want === 0) return new BigUint64Array(0);

    const lib = ffi.getLib();
    const koffi = lib.koffi;
    let cap = want;
    for (let attempt = 0; attempt < 2; attempt++) {
        const out = koffi.alloc('uint64_t', cap);
        const outLen = [0];
        const rc = lib.ztok_ngram_hash(idArr, nIds, n, heads, out, cap, outLen);
        if (rc === ffi.ZTOK_OK) {
            return decodeU64s(koffi, out, Number(outLen[0]));
        }
        if (rc === ffi.ZTOK_ERR_BUFFER_TOO_SMALL) {
            cap = Number(outLen[0]);
            if (cap === 0) return new BigUint64Array(0);
            continue;
        }
        raiseForStatus(rc, 'ztok_ngram_hash');
    }
    throw new ZtokInternalError('ztok_ngram_hash reported BUFFER_TOO_SMALL twice');
}

/**
 * Hash many id streams in parallel across `pool`. results[i] holds the
 * row-major hashes for streams[i] (empty for a stream shorter than one
 * window). Equivalent to calling {@link hashNgrams} on each stream,
 * fanned out across the pool's workers. Each per-doc C buffer is
 * materialized into a BigUint64Array and freed via ztok_u64s_free (the
 * only safe path — buffers carry a length-prefix header).
 *
 * @param {BatchPool} pool
 * @param {(Uint32Array | number[])[]} streams
 * @param {number} n
 * @param {number} heads
 * @returns {BigUint64Array[]}
 */
function hashNgramsBatch(pool, streams, n, heads) {
    if (!(pool instanceof BatchPool)) {
        throw new TypeError('hashNgramsBatch: first arg must be a BatchPool');
    }
    const poolHandle = pool._raw();
    const nDocs = streams.length;
    if (nDocs === 0) return [];

    const lib = ffi.getLib();
    const koffi = lib.koffi;

    // Materialize each id stream into its own Uint32Array buffer. koffi
    // needs an array of pointers for `const uint32_t**`; we keep the
    // typed arrays alive in `bufs` for the duration of the call.
    const bufs = new Array(nDocs);
    const idArrays = new Array(nDocs);
    const idLens = new Array(nDocs);
    for (let i = 0; i < nDocs; i++) {
        const s = streams[i];
        const arr = s instanceof Uint32Array ? s : Uint32Array.from(s);
        idLens[i] = arr.length;
        if (arr.length === 0) {
            bufs[i] = null;
            idArrays[i] = null;
            continue;
        }
        bufs[i] = arr;
        idArrays[i] = arr;
    }

    const outHashes = new Array(nDocs).fill(null);
    const outLens = new Array(nDocs).fill(0);

    const rc = lib.ztok_ngram_hash_batch(
        poolHandle, idArrays, idLens, nDocs, n, heads, outHashes, outLens
    );

    try {
        // Materialize every (possibly partial) buffer before raising, so an
        // error mid-batch can't leak the buffers already allocated.
        const results = new Array(nDocs);
        for (let i = 0; i < nDocs; i++) {
            results[i] = decodeU64s(koffi, outHashes[i], Number(outLens[i]));
        }
        raiseForStatus(rc, 'ztok_ngram_hash_batch');
        return results;
    } finally {
        for (let i = 0; i < nDocs; i++) {
            if (outHashes[i]) {
                try { lib.ztok_u64s_free(outHashes[i]); } catch (_) { /* ignore */ }
            }
        }
    }
}

module.exports = {
    Pipeline,
    BatchPool,
    version,
    hashNgrams,
    hashNgramsBatch,

    // Exceptions
    ZtokError,
    ZtokInvalidInputError,
    ZtokOutOfMemoryError,
    ZtokBufferTooSmallError,
    ZtokInternalError,
    ZtokLibraryNotFoundError,

    // Enum constants (re-exported so callers can pass non-default kinds)
    NORMALIZER_IDENTITY: ffi.NORMALIZER_IDENTITY,
    NORMALIZER_NFC: ffi.NORMALIZER_NFC,
    NORMALIZER_NFD: ffi.NORMALIZER_NFD,
    NORMALIZER_NFKC: ffi.NORMALIZER_NFKC,
    NORMALIZER_NFKD: ffi.NORMALIZER_NFKD,
    NORMALIZER_BYTE_LEVEL: ffi.NORMALIZER_BYTE_LEVEL,
    PRETOK_IDENTITY: ffi.PRETOK_IDENTITY,
    PRETOK_CL100K: ffi.PRETOK_CL100K,
    DECODER_CONCAT: ffi.DECODER_CONCAT,
    DECODER_WORDPIECE: ffi.DECODER_WORDPIECE,
    DECODER_BYTE_LEVEL: ffi.DECODER_BYTE_LEVEL,

    // Overlay channel kinds (mirror ztok_overlay_kind in include/ztok.h)
    OVERLAY_BYTE_START: ffi.OVERLAY_BYTE_START,
    OVERLAY_BYTE_END: ffi.OVERLAY_BYTE_END,
    OVERLAY_BOUNDARY: ffi.OVERLAY_BOUNDARY,
    OVERLAY_OPCODE: ffi.OVERLAY_OPCODE,
    OVERLAY_OPERAND: ffi.OVERLAY_OPERAND,
    OVERLAY_SYMBOL_REF: ffi.OVERLAY_SYMBOL_REF,
    OVERLAY_HUNK: ffi.OVERLAY_HUNK,
    OVERLAY_PROVENANCE: ffi.OVERLAY_PROVENANCE,
    OVERLAY_USER_BASE: ffi.OVERLAY_USER_BASE,

    // Chunk boundary modes (mirror ztok_chunk_boundary in include/ztok.h)
    CHUNK_BOUNDARY_TOKEN: ffi.CHUNK_BOUNDARY_TOKEN,
    CHUNK_BOUNDARY_CODEPOINT: ffi.CHUNK_BOUNDARY_CODEPOINT,
    CHUNK_BOUNDARY_WORD: ffi.CHUNK_BOUNDARY_WORD,
    CHUNK_BOUNDARY_WORD_DICT: ffi.CHUNK_BOUNDARY_WORD_DICT,
    CHUNK_BOUNDARY_SENTENCE: ffi.CHUNK_BOUNDARY_SENTENCE,
    CHUNK_BOUNDARY_PARAGRAPH: ffi.CHUNK_BOUNDARY_PARAGRAPH,

    // Format detection (string form). Mirrors Python's _detect_format.
    detectFormat: ffi.detectFormat,
};
