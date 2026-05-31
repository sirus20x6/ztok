/*
 * Thin AutoCloseable wrapper around a libztok ztok_pipeline*.
 *
 * Lifetime: explicit close() (or try-with-resources) is strongly
 * preferred. A Cleaner-registered fallback frees the native handle if
 * the wrapper is GC'd without close(), mirroring the FinalizationRegistry
 * pattern used by the Node binding.
 *
 * Thread safety: a Pipeline is safe to share across threads for encode /
 * decode / encodeBatch / fingerprint. Streaming encode is per-call;
 * StreamEncoder is NOT thread-safe.
 *
 * Memory: each encode/decode uses an Arena.ofConfined() that's freed
 * on close — no native bytes outlive a single call.
 */
package com.anthropic.ztok;

import com.anthropic.ztok.internal.Native;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.ref.Cleaner;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.EnumMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

public final class Pipeline implements AutoCloseable {
    private static final Cleaner CLEANER = Cleaner.create();

    private static final class State implements Runnable {
        volatile MemorySegment handle;

        State(MemorySegment handle) { this.handle = handle; }

        @Override
        public void run() {
            MemorySegment h = handle;
            if (h != null && h.address() != 0) {
                handle = null;
                try { Native.ZTOK_PIPELINE_FREE.invoke(h); }
                catch (Throwable ignored) { /* best-effort */ }
            }
        }
    }

    private final State state;
    private final Cleaner.Cleanable cleanable;
    private volatile boolean closed;

    private Pipeline(MemorySegment handle) {
        if (handle == null || handle.address() == 0) {
            throw new ZtokException.Internal(
                "Pipeline constructed with NULL handle", Native.ZTOK_ERR_INTERNAL);
        }
        this.state = new State(handle);
        this.cleanable = CLEANER.register(this, state);
    }

    // --- constructors -----------------------------------------------------

    /** byte_id baseline (each input byte maps to its own id). */
    public static Pipeline byteId() {
        return byteId(Native.NORMALIZER_IDENTITY, Native.PRETOK_IDENTITY,
                      Native.DECODER_CONCAT);
    }

    public static Pipeline byteId(int normalizer, int preTokenizer, int decoder) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment cfg = allocConfig(arena, normalizer, preTokenizer,
                                            Native.MODEL_BYTE_ID, decoder);
            MemorySegment status = arena.allocate(Native.INT);
            MemorySegment handle = invokeNew(Native.ZTOK_PIPELINE_NEW, cfg, status);
            ZtokException.check(status.get(Native.INT, 0), "ztok_pipeline_new");
            return new Pipeline(handle);
        }
    }

    /** Load a .tiktoken vocab into a byte-level BPE pipeline (cl100k pre-tok by default). */
    public static Pipeline fromTiktoken(Path path) {
        return fromTiktoken(path, true);
    }

    public static Pipeline fromTiktoken(Path path, boolean cl100k) {
        return fromFile(Native.ZTOK_PIPELINE_NEW_BPE_FROM_TIKTOKEN, path,
                        Native.NORMALIZER_IDENTITY,
                        cl100k ? Native.PRETOK_CL100K : Native.PRETOK_IDENTITY,
                        Native.MODEL_BYTE_ID, Native.DECODER_CONCAT,
                        "ztok_pipeline_new_bpe_from_tiktoken");
    }

    /** Load a HuggingFace tokenizer.json BPE model. */
    public static Pipeline fromHfJson(Path path) {
        return fromHfJson(path, false);
    }

    public static Pipeline fromHfJson(Path path, boolean cl100k) {
        return fromFile(Native.ZTOK_PIPELINE_NEW_BPE_FROM_HF_JSON, path,
                        Native.NORMALIZER_IDENTITY,
                        cl100k ? Native.PRETOK_CL100K : Native.PRETOK_IDENTITY,
                        Native.MODEL_BYTE_ID, Native.DECODER_CONCAT,
                        "ztok_pipeline_new_bpe_from_hf_json");
    }

    /** Load a HuggingFace WordPiece model from tokenizer.json. */
    public static Pipeline fromWordPiece(Path path, int unkId) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment cfg = allocConfig(arena, Native.NORMALIZER_IDENTITY,
                                            Native.PRETOK_IDENTITY,
                                            Native.MODEL_BYTE_ID,
                                            Native.DECODER_WORDPIECE);
            MemorySegment status = arena.allocate(Native.INT);
            MemorySegment pathPtr = arena.allocateUtf8String(path.toString());
            MemorySegment handle;
            try {
                handle = (MemorySegment) Native.ZTOK_PIPELINE_NEW_WORDPIECE_FROM_HF_JSON
                    .invoke(pathPtr, unkId, cfg, status);
            } catch (Throwable t) {
                throw rethrow("ztok_pipeline_new_wordpiece_from_hf_json", t);
            }
            ZtokException.check(status.get(Native.INT, 0),
                "ztok_pipeline_new_wordpiece_from_hf_json");
            return new Pipeline(handle);
        }
    }

    /** Load a SentencePiece .model (Unigram) file. */
    public static Pipeline fromSentencePiece(Path path) {
        return fromSentencePiece(path, 0);
    }

    public static Pipeline fromSentencePiece(Path path, int unkId) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment cfg = allocConfig(arena, Native.NORMALIZER_IDENTITY,
                                            Native.PRETOK_IDENTITY,
                                            Native.MODEL_BYTE_ID,
                                            Native.DECODER_CONCAT);
            MemorySegment status = arena.allocate(Native.INT);
            MemorySegment pathPtr = arena.allocateUtf8String(path.toString());
            MemorySegment handle;
            try {
                handle = (MemorySegment) Native.ZTOK_PIPELINE_NEW_UNIGRAM_FROM_SP_MODEL
                    .invoke(pathPtr, unkId, cfg, status);
            } catch (Throwable t) {
                throw rethrow("ztok_pipeline_new_unigram_from_sp_model", t);
            }
            ZtokException.check(status.get(Native.INT, 0),
                "ztok_pipeline_new_unigram_from_sp_model");
            return new Pipeline(handle);
        }
    }

    /** Load a ztok TokenMonster .ztm vocab file. */
    public static Pipeline fromMonster(Path path) {
        return fromFile(Native.ZTOK_PIPELINE_NEW_MONSTER_FROM_FILE, path,
                        Native.NORMALIZER_IDENTITY, Native.PRETOK_IDENTITY,
                        Native.MODEL_BYTE_ID, Native.DECODER_CONCAT,
                        "ztok_pipeline_new_monster_from_file");
    }

    /**
     * Load an RWKV "World" vocab ({@code rwkv_vocab_v20230424.txt}). The
     * model is a greedy longest-match byte trie that runs over the whole
     * input as a single span — there is no pre-tokenizer, so the
     * pre-tokenizer slot is fixed to identity.
     */
    public static Pipeline fromRwkv(Path path) {
        return fromFile(Native.ZTOK_PIPELINE_NEW_RWKV_FROM_FILE, path,
                        Native.NORMALIZER_IDENTITY, Native.PRETOK_IDENTITY,
                        Native.MODEL_BYTE_ID, Native.DECODER_CONCAT,
                        "ztok_pipeline_new_rwkv_from_file");
    }

    /**
     * Load a Mistral Tekken {@code tekken.json} vocab (Nemo / Pixtral /
     * Devstral / Magistral, etc.). The loader lowers Tekken's base64 byte
     * vocab into a BPE with the special tokens packed into the bottom of
     * the id space. The default pre-tokenizer is the Tekken pattern
     * ({@code PRETOK_TEKKEN}) — NOT cl100k — and the decoder is concat
     * (pieces are raw bytes).
     */
    public static Pipeline fromTekken(Path path) {
        return fromFile(Native.ZTOK_PIPELINE_NEW_TEKKEN_FROM_FILE, path,
                        Native.NORMALIZER_IDENTITY, Native.PRETOK_TEKKEN,
                        Native.MODEL_BYTE_ID, Native.DECODER_CONCAT,
                        "ztok_pipeline_new_tekken_from_file");
    }

    /**
     * Auto-detect the file format and dispatch to the right loader. The
     * lowercase name matches the API used in the Python / Node bindings:
     * Python's {@code Pipeline.from_path}, Node's {@code Pipeline.fromPath},
     * Rust's {@code Pipeline::open}. We follow the Rust naming so the
     * top-level entry reads naturally in Java: {@code Pipeline.open(path)}.
     */
    public static Pipeline open(Path path) {
        Format fmt = autoDetect(path);
        return switch (fmt) {
            case TIKTOKEN -> fromTiktoken(path);
            case HF_JSON -> fromHfJson(path);
            case SENTENCEPIECE -> fromSentencePiece(path);
            case ZTM -> fromMonster(path);
            case TEKKEN -> fromTekken(path);
            case RWKV -> fromRwkv(path);
            case UNKNOWN -> throw new ZtokException.InvalidInput(
                "could not auto-detect tokenizer format for " + path
                + "; use a specific from* constructor instead",
                Native.ZTOK_ERR_INVALID_INPUT);
        };
    }

    /** Best-effort format sniffer via ztok_auto_detect. */
    public static Format autoDetect(Path path) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment p = arena.allocateUtf8String(path.toString());
            int code;
            try {
                code = (int) Native.ZTOK_AUTO_DETECT.invoke(p);
            } catch (Throwable t) {
                throw rethrow("ztok_auto_detect", t);
            }
            return Format.fromCode(code);
        }
    }

    // --- lifecycle --------------------------------------------------------

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        cleanable.clean();
    }

    public boolean isClosed() {
        return closed;
    }

    private MemorySegment handle() {
        if (closed || state.handle == null) {
            throw new ZtokException.Internal(
                "Pipeline is closed", Native.ZTOK_ERR_INTERNAL);
        }
        return state.handle;
    }

    // --- encode / decode --------------------------------------------------

    /** Encode a UTF-8 string into a fresh int[] of token ids. */
    public int[] encode(String text) {
        return encodeBytes(text.getBytes(StandardCharsets.UTF_8));
    }

    /** Encode raw bytes into a fresh int[] of token ids. */
    public int[] encodeBytes(byte[] data) {
        MemorySegment h = handle();
        if (data.length == 0) return new int[0];

        // Per-span maxTokensFor is conservative, so even a generously
        // sized buffer can trip BUFFER_TOO_SMALL mid-stream. Grow up to
        // 8 times before giving up — mirrors the Python/Node behavior.
        int cap = Math.max(data.length + 16, 64);
        for (int attempt = 0; attempt < 8; attempt++) {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment input = arena.allocate(data.length);
                MemorySegment.copy(data, 0, input, Native.BYTE, 0, data.length);
                MemorySegment out = arena.allocateArray(Native.TOKEN_ID, cap);
                MemorySegment outLen = arena.allocate(Native.SIZE_T);
                int rc;
                try {
                    rc = (int) Native.ZTOK_ENCODE.invoke(
                        h, input, (long) data.length, out, (long) cap, outLen);
                } catch (Throwable t) {
                    throw rethrow("ztok_encode", t);
                }
                if (rc == Native.ZTOK_OK) {
                    long n = outLen.get(Native.SIZE_T, 0);
                    int[] ids = new int[(int) n];
                    MemorySegment.copy(out, Native.TOKEN_ID, 0, ids, 0, (int) n);
                    return ids;
                }
                if (rc == Native.ZTOK_ERR_BUFFER_TOO_SMALL) {
                    long needed = outLen.get(Native.SIZE_T, 0);
                    cap = (int) Math.max((long) cap * 2L, needed + 16);
                    continue;
                }
                ZtokException.check(rc, "ztok_encode");
            }
        }
        throw new ZtokException.Internal(
            "ztok_encode kept reporting BUFFER_TOO_SMALL after 8 grow attempts",
            Native.ZTOK_ERR_INTERNAL);
    }

    /** Decode ids and interpret as UTF-8 (malformed bytes pass through to the JVM decoder). */
    public String decode(int[] ids) {
        return new String(decodeBytes(ids), StandardCharsets.UTF_8);
    }

    /** Decode ids without UTF-8 round-tripping. */
    public byte[] decodeBytes(int[] ids) {
        MemorySegment h = handle();
        if (ids.length == 0) return new byte[0];

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment idsSeg = arena.allocateArray(Native.TOKEN_ID, ids.length);
            MemorySegment.copy(ids, 0, idsSeg, Native.TOKEN_ID, 0, ids.length);
            MemorySegment outLen = arena.allocate(Native.SIZE_T);

            // Sizing pass: passing a 0-capacity output triggers
            // BUFFER_TOO_SMALL with *out_len set to the required size.
            MemorySegment probe = arena.allocate(1);
            int rc1;
            try {
                rc1 = (int) Native.ZTOK_DECODE.invoke(
                    h, idsSeg, (long) ids.length, probe, 0L, outLen);
            } catch (Throwable t) {
                throw rethrow("ztok_decode (sizing)", t);
            }
            if (rc1 != Native.ZTOK_OK && rc1 != Native.ZTOK_ERR_BUFFER_TOO_SMALL) {
                ZtokException.check(rc1, "ztok_decode (sizing)");
            }
            long nbytes = outLen.get(Native.SIZE_T, 0);
            if (nbytes == 0) return new byte[0];

            MemorySegment outBuf = arena.allocate(nbytes);
            int rc2;
            try {
                rc2 = (int) Native.ZTOK_DECODE.invoke(
                    h, idsSeg, (long) ids.length, outBuf, nbytes, outLen);
            } catch (Throwable t) {
                throw rethrow("ztok_decode", t);
            }
            ZtokException.check(rc2, "ztok_decode");
            long written = outLen.get(Native.SIZE_T, 0);
            byte[] out = new byte[(int) written];
            MemorySegment.copy(outBuf, Native.BYTE, 0, out, 0, (int) written);
            return out;
        }
    }

    // --- encode with overlays ---------------------------------------------

    /**
     * Result of {@link #encodeWithOverlays}: the token id stream plus a
     * map of each requested {@link OverlayKind} to its per-token value
     * array. Every channel array has the same length as {@link #ids()}.
     */
    public static final class OverlayResult {
        private final int[] ids;
        private final Map<OverlayKind, int[]> channels;

        OverlayResult(int[] ids, Map<OverlayKind, int[]> channels) {
            this.ids = ids;
            this.channels = channels;
        }

        /** Token ids — identical to what {@link Pipeline#encode} returns. */
        public int[] ids() { return ids; }

        /** Per-token overlay channels keyed by kind, aligned 1:1 with {@link #ids()}. */
        public Map<OverlayKind, int[]> channels() { return channels; }
    }

    // ztok_overlay_channel layout (64-bit): { int kind; <pad>; void* out; size_t out_cap; }
    // -> kind at 0, out (ptr) at 8, out_cap at 16; struct size 24, align 8.
    private static final long OVERLAY_CHANNEL_SIZE = 24L;
    private static final long OVERLAY_CHANNEL_KIND_OFFSET = 0L;
    private static final long OVERLAY_CHANNEL_OUT_OFFSET = 8L;
    private static final long OVERLAY_CHANNEL_CAP_OFFSET = 16L;

    /**
     * Select which domain normalizer populates the domain overlay channels
     * (OPCODE / OPERAND / SYMBOL_REF / HUNK). {@link OverlayDomain#NONE} (the
     * default) leaves those channels zero-filled; {@link OverlayDomain#X86_64}
     * decodes the input as x86-64 machine code. An unrecognized value leaves
     * the pipeline unchanged and throws {@link ZtokException.InvalidInput}.
     */
    public void setOverlayDomain(OverlayDomain domain) {
        MemorySegment h = handle();
        int rc;
        try {
            rc = (int) Native.ZTOK_PIPELINE_SET_OVERLAY_DOMAIN.invoke(h, domain.code());
        } catch (Throwable t) {
            throw rethrow("ztok_pipeline_set_overlay_domain", t);
        }
        ZtokException.check(rc, "ztok_pipeline_set_overlay_domain");
    }

    /** Encode a UTF-8 string and return ids plus the requested overlay channels. */
    public OverlayResult encodeWithOverlays(String text, OverlayKind... channels) {
        return encodeBytesWithOverlays(text.getBytes(StandardCharsets.UTF_8), channels);
    }

    /**
     * Encode raw bytes and return ids plus a map of per-token overlay
     * channels aligned 1:1 with the id stream. Requesting overlays never
     * changes tokenization — the ids match {@link #encode}. Cheap channels
     * carry encoder-derived values; domain channels come back zero-filled
     * until a domain plugin populates them. Duplicate kinds are rejected
     * with {@link ZtokException.InvalidInput}.
     */
    public OverlayResult encodeBytesWithOverlays(byte[] data, OverlayKind... channels) {
        MemorySegment h = handle();
        if (channels == null) channels = new OverlayKind[0];

        // Reject duplicate kinds — the result map would collapse them.
        for (int i = 0; i < channels.length; i++) {
            for (int j = 0; j < i; j++) {
                if (channels[i] == channels[j]) {
                    throw new ZtokException.InvalidInput(
                        "encodeWithOverlays: duplicate channel kind " + channels[i],
                        Native.ZTOK_ERR_INVALID_INPUT);
                }
            }
        }

        int nCh = channels.length;
        if (data.length == 0) {
            return new OverlayResult(new int[0], emptyChannels(channels));
        }

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment input = arena.allocate(data.length);
            MemorySegment.copy(data, 0, input, Native.BYTE, 0, data.length);
            MemorySegment outLen = arena.allocate(Native.SIZE_T);

            // Sizing pass: out_ids = NULL queries the token count. Build the
            // channels[] array with NULL out pointers so the C side just
            // reports the count.
            MemorySegment chans = nCh == 0
                ? MemorySegment.NULL
                : arena.allocate(OVERLAY_CHANNEL_SIZE * nCh, 8);
            for (int i = 0; i < nCh; i++) {
                long base = (long) i * OVERLAY_CHANNEL_SIZE;
                chans.set(Native.INT, base + OVERLAY_CHANNEL_KIND_OFFSET, channels[i].code());
                chans.set(Native.PTR, base + OVERLAY_CHANNEL_OUT_OFFSET, MemorySegment.NULL);
                chans.set(Native.SIZE_T, base + OVERLAY_CHANNEL_CAP_OFFSET, 0L);
            }

            int rc1;
            try {
                rc1 = (int) Native.ZTOK_ENCODE_WITH_OVERLAYS.invoke(
                    h, input, (long) data.length,
                    MemorySegment.NULL, 0L,
                    chans, (long) nCh, outLen);
            } catch (Throwable t) {
                throw rethrow("ztok_encode_with_overlays (sizing)", t);
            }
            if (rc1 != Native.ZTOK_OK && rc1 != Native.ZTOK_ERR_BUFFER_TOO_SMALL) {
                ZtokException.check(rc1, "ztok_encode_with_overlays (sizing)");
            }
            long count = outLen.get(Native.SIZE_T, 0);
            if (count == 0) {
                return new OverlayResult(new int[0], emptyChannels(channels));
            }

            // Fill pass: allocate the id buffer + one uint32 buffer per
            // channel, each sized to the exact count.
            MemorySegment outIds = arena.allocateArray(Native.TOKEN_ID, count);
            MemorySegment[] chanBufs = new MemorySegment[nCh];
            for (int i = 0; i < nCh; i++) {
                chanBufs[i] = arena.allocateArray(Native.TOKEN_ID, count);
                long base = (long) i * OVERLAY_CHANNEL_SIZE;
                chans.set(Native.PTR, base + OVERLAY_CHANNEL_OUT_OFFSET, chanBufs[i]);
                chans.set(Native.SIZE_T, base + OVERLAY_CHANNEL_CAP_OFFSET, count);
            }

            int rc2;
            try {
                rc2 = (int) Native.ZTOK_ENCODE_WITH_OVERLAYS.invoke(
                    h, input, (long) data.length,
                    outIds, count,
                    chans, (long) nCh, outLen);
            } catch (Throwable t) {
                throw rethrow("ztok_encode_with_overlays", t);
            }
            ZtokException.check(rc2, "ztok_encode_with_overlays");

            long n = outLen.get(Native.SIZE_T, 0);
            int[] ids = new int[(int) n];
            MemorySegment.copy(outIds, Native.TOKEN_ID, 0, ids, 0, (int) n);

            Map<OverlayKind, int[]> result = new EnumMap<>(OverlayKind.class);
            for (int i = 0; i < nCh; i++) {
                int[] vals = new int[(int) n];
                MemorySegment.copy(chanBufs[i], Native.TOKEN_ID, 0, vals, 0, (int) n);
                result.put(channels[i], vals);
            }
            return new OverlayResult(ids, result);
        }
    }

    private static Map<OverlayKind, int[]> emptyChannels(OverlayKind[] channels) {
        Map<OverlayKind, int[]> m = new EnumMap<>(OverlayKind.class);
        for (OverlayKind k : channels) m.put(k, new int[0]);
        return m;
    }

    // --- batch ------------------------------------------------------------

    /**
     * Encode many inputs in parallel via a persistent BatchPool.
     * Each output is a fresh int[]; the libztok-owned buffers are
     * materialized and immediately freed through ztok_ids_free (the
     * only safe path — buffers carry a length-prefix header).
     */
    public int[][] encodeBatch(BatchPool pool, List<String> inputs) {
        MemorySegment h = handle();
        int n = inputs.size();
        if (n == 0) return new int[0][];

        // Encode every string to UTF-8 bytes once; keep them alive.
        byte[][] bytes = new byte[n][];
        long[] lens = new long[n];
        for (int i = 0; i < n; i++) {
            bytes[i] = inputs.get(i).getBytes(StandardCharsets.UTF_8);
            lens[i] = bytes[i].length;
        }

        try (Arena arena = Arena.ofConfined()) {
            // inputs: array of (char*) — each entry points into a per-input segment.
            MemorySegment inputsArr  = arena.allocateArray(Native.PTR, n);
            MemorySegment lensArr    = arena.allocateArray(Native.SIZE_T, n);
            MemorySegment outIdsArr  = arena.allocateArray(Native.PTR, n);
            MemorySegment outLensArr = arena.allocateArray(Native.SIZE_T, n);

            for (int i = 0; i < n; i++) {
                MemorySegment buf = arena.allocate(bytes[i].length + 1L);
                if (bytes[i].length > 0) {
                    MemorySegment.copy(bytes[i], 0, buf, Native.BYTE, 0, bytes[i].length);
                }
                buf.set(Native.BYTE, bytes[i].length, (byte) 0);
                inputsArr.setAtIndex(Native.PTR, i, buf);
                lensArr.setAtIndex(Native.SIZE_T, i, lens[i]);
            }

            int rc;
            try {
                rc = (int) Native.ZTOK_ENCODE_BATCH_POOLED.invoke(
                    h, pool.handle(), inputsArr, lensArr, (long) n,
                    outIdsArr, outLensArr);
            } catch (Throwable t) {
                throw rethrow("ztok_encode_batch_pooled", t);
            }

            int[][] results = new int[n][];
            try {
                ZtokException.check(rc, "ztok_encode_batch_pooled");
                for (int i = 0; i < n; i++) {
                    long len = outLensArr.getAtIndex(Native.SIZE_T, i);
                    MemorySegment ptr = outIdsArr.getAtIndex(Native.PTR, i);
                    if (len == 0 || ptr.address() == 0) {
                        results[i] = new int[0];
                        continue;
                    }
                    MemorySegment view = ptr.reinterpret(len * Native.TOKEN_ID.byteSize());
                    int[] ids = new int[(int) len];
                    MemorySegment.copy(view, Native.TOKEN_ID, 0, ids, 0, (int) len);
                    results[i] = ids;
                }
            } finally {
                for (int i = 0; i < n; i++) {
                    MemorySegment ptr = outIdsArr.getAtIndex(Native.PTR, i);
                    if (ptr.address() != 0) {
                        try { Native.ZTOK_IDS_FREE.invoke(ptr); }
                        catch (Throwable ignored) { /* best-effort */ }
                    }
                }
            }
            return results;
        }
    }

    // --- chunking ---------------------------------------------------------

    // ztok_chunk_rec layout (64-bit): { token_id* ids; size_t ids_len;
    //   uint32 byte_start, byte_end, token_start, token_end; }
    // -> ids at 0, ids_len at 8, byte_start at 16, byte_end at 20,
    //    token_start at 24, token_end at 28; struct size 32, align 8.
    private static final long CHUNK_REC_SIZE = 32L;
    private static final long CHUNK_REC_IDS_OFFSET = 0L;
    private static final long CHUNK_REC_IDS_LEN_OFFSET = 8L;
    private static final long CHUNK_REC_BYTE_START_OFFSET = 16L;
    private static final long CHUNK_REC_BYTE_END_OFFSET = 20L;
    private static final long CHUNK_REC_TOKEN_START_OFFSET = 24L;
    private static final long CHUNK_REC_TOKEN_END_OFFSET = 28L;

    /**
     * Split {@code text} into non-overlapping token windows of at most
     * {@code maxTokens} ids each, with token-count boundaries.
     */
    public Chunk[] chunk(String text, int maxTokens) {
        return chunk(text, maxTokens, 0, ChunkBoundary.TOKEN);
    }

    /**
     * Split {@code text} into token windows of at most {@code maxTokens} ids
     * with {@code overlap} ids shared between neighbors, on token boundaries.
     */
    public Chunk[] chunk(String text, int maxTokens, int overlap) {
        return chunk(text, maxTokens, overlap, ChunkBoundary.TOKEN);
    }

    /**
     * Split {@code text} into overlapping token windows (late chunking).
     * Each window holds at most {@code maxTokens} ids with {@code overlap}
     * ids shared between neighbors (stride = {@code maxTokens - overlap}).
     * Returns an empty array for empty input. The native-owned id buffers
     * are copied into Java arrays and freed before returning.
     *
     * @throws ZtokException.InvalidInput if {@code maxTokens == 0} or
     *         {@code overlap >= maxTokens}.
     */
    public Chunk[] chunk(String text, int maxTokens, int overlap,
                         ChunkBoundary boundary) {
        MemorySegment h = handle();
        if (maxTokens <= 0 || overlap >= maxTokens) {
            throw new ZtokException.InvalidInput(
                "chunk: maxTokens must be > 0 and overlap < maxTokens",
                Native.ZTOK_ERR_INVALID_INPUT);
        }
        byte[] data = text.getBytes(StandardCharsets.UTF_8);
        if (data.length == 0) return new Chunk[0];
        int boundaryCode = boundary.code();

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment input = arena.allocate(data.length);
            MemorySegment.copy(data, 0, input, Native.BYTE, 0, data.length);
            MemorySegment outLen = arena.allocate(Native.SIZE_T);

            // Sizing pass: out_chunks = NULL -> *out_len = chunk count.
            int rc1;
            try {
                rc1 = (int) Native.ZTOK_CHUNK.invoke(
                    h, input, (long) data.length,
                    maxTokens, overlap, boundaryCode,
                    MemorySegment.NULL, 0L, outLen);
            } catch (Throwable t) {
                throw rethrow("ztok_chunk (sizing)", t);
            }
            if (rc1 != Native.ZTOK_OK && rc1 != Native.ZTOK_ERR_BUFFER_TOO_SMALL) {
                ZtokException.check(rc1, "ztok_chunk (sizing)");
            }
            long count = outLen.get(Native.SIZE_T, 0);
            if (count == 0) return new Chunk[0];

            // Fill pass: allocate the caller-owned record array.
            MemorySegment recs = arena.allocate(CHUNK_REC_SIZE * count, 8);
            int rc2;
            try {
                rc2 = (int) Native.ZTOK_CHUNK.invoke(
                    h, input, (long) data.length,
                    maxTokens, overlap, boundaryCode,
                    recs, count, outLen);
            } catch (Throwable t) {
                throw rethrow("ztok_chunk", t);
            }

            long got = outLen.get(Native.SIZE_T, 0);
            try {
                ZtokException.check(rc2, "ztok_chunk");
                Chunk[] chunks = new Chunk[(int) got];
                for (int i = 0; i < got; i++) {
                    long base = (long) i * CHUNK_REC_SIZE;
                    MemorySegment idsPtr = recs.get(Native.PTR, base + CHUNK_REC_IDS_OFFSET);
                    long idsLen = recs.get(Native.SIZE_T, base + CHUNK_REC_IDS_LEN_OFFSET);
                    int[] ids;
                    if (idsPtr.address() != 0 && idsLen > 0) {
                        MemorySegment view = idsPtr.reinterpret(idsLen * Native.TOKEN_ID.byteSize());
                        ids = new int[(int) idsLen];
                        MemorySegment.copy(view, Native.TOKEN_ID, 0, ids, 0, (int) idsLen);
                    } else {
                        ids = new int[0];
                    }
                    int byteStart  = recs.get(Native.INT, base + CHUNK_REC_BYTE_START_OFFSET);
                    int byteEnd    = recs.get(Native.INT, base + CHUNK_REC_BYTE_END_OFFSET);
                    int tokenStart = recs.get(Native.INT, base + CHUNK_REC_TOKEN_START_OFFSET);
                    int tokenEnd   = recs.get(Native.INT, base + CHUNK_REC_TOKEN_END_OFFSET);
                    chunks[i] = new Chunk(ids, byteStart, byteEnd, tokenStart, tokenEnd);
                }
                return chunks;
            } finally {
                // Release each record's ztok-allocated id buffer. The recs
                // array itself is arena-owned (freed by the try-with-resources).
                try { Native.ZTOK_CHUNKS_FREE.invoke(recs, got); }
                catch (Throwable ignored) { /* best-effort */ }
            }
        }
    }

    // --- streaming --------------------------------------------------------

    /** Stream-encode text via the default 64 KiB chunk size. */
    public StreamEncoder encodeStream(String text) {
        return encodeStream(text.getBytes(StandardCharsets.UTF_8),
                            StreamEncoder.DEFAULT_CHUNK_SIZE);
    }

    public StreamEncoder encodeStream(byte[] data, int chunkSize) {
        MemorySegment h = handle();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment status = arena.allocate(Native.INT);
            MemorySegment streamHandle;
            try {
                streamHandle = (MemorySegment) Native.ZTOK_STREAM_NEW.invoke(h, status);
            } catch (Throwable t) {
                throw rethrow("ztok_stream_new", t);
            }
            ZtokException.check(status.get(Native.INT, 0), "ztok_stream_new");
            if (streamHandle == null || streamHandle.address() == 0) {
                throw new ZtokException.Internal(
                    "ztok_stream_new returned NULL", Native.ZTOK_ERR_INTERNAL);
            }
            return new StreamEncoder(streamHandle, data, chunkSize);
        }
    }

    // --- fingerprint ------------------------------------------------------

    /** 32-byte deterministic tokenizer fingerprint. */
    public Fingerprint fingerprint() {
        MemorySegment h = handle();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment out = arena.allocate(Fingerprint.LENGTH);
            int rc;
            try {
                rc = (int) Native.ZTOK_FINGERPRINT.invoke(h, out);
            } catch (Throwable t) {
                throw rethrow("ztok_fingerprint", t);
            }
            ZtokException.check(rc, "ztok_fingerprint");
            byte[] bytes = new byte[Fingerprint.LENGTH];
            MemorySegment.copy(out, Native.BYTE, 0, bytes, 0, Fingerprint.LENGTH);
            return new Fingerprint(bytes);
        }
    }

    // --- internals --------------------------------------------------------

    /**
     * Layout of ztok_pipeline_config: four 32-bit uints, packed in
     * declaration order. We write the four fields by hand instead of
     * using a StructLayout so the descriptor file matches what
     * SymbolLookup expects (a flat 16-byte struct).
     */
    private static MemorySegment allocConfig(Arena arena, int normalizer,
                                             int preTokenizer, int model, int decoder) {
        MemorySegment cfg = arena.allocate(16, 4);
        cfg.set(Native.INT, 0, normalizer);
        cfg.set(Native.INT, 4, preTokenizer);
        cfg.set(Native.INT, 8, model);
        cfg.set(Native.INT, 12, decoder);
        return cfg;
    }

    private static MemorySegment invokeNew(java.lang.invoke.MethodHandle mh,
                                           MemorySegment cfg, MemorySegment status) {
        try {
            return (MemorySegment) mh.invoke(cfg, status);
        } catch (Throwable t) {
            throw rethrow("ztok_pipeline_new", t);
        }
    }

    private static Pipeline fromFile(java.lang.invoke.MethodHandle mh, Path path,
                                     int normalizer, int preTokenizer,
                                     int model, int decoder, String ctx) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment cfg = allocConfig(arena, normalizer, preTokenizer, model, decoder);
            MemorySegment status = arena.allocate(Native.INT);
            MemorySegment pathPtr = arena.allocateUtf8String(path.toString());
            MemorySegment handle;
            try {
                handle = (MemorySegment) mh.invoke(pathPtr, cfg, status);
            } catch (Throwable t) {
                throw rethrow(ctx, t);
            }
            ZtokException.check(status.get(Native.INT, 0), ctx);
            return new Pipeline(handle);
        }
    }

    private static ZtokException rethrow(String ctx, Throwable t) {
        if (t instanceof ZtokException ze) return ze;
        return new ZtokException.Internal(
            ctx + " threw: " + t.getClass().getSimpleName() + ": " + t.getMessage(),
            Native.ZTOK_ERR_INTERNAL);
    }

    /** Convenience: lowercase the OS — unused currently but kept for future ABI sniffing. */
    @SuppressWarnings("unused")
    private static String osName() {
        return System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
    }
}
