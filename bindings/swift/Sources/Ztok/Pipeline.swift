// Pipeline — high-level Swift wrapper around `ztok_pipeline*`.
//
// Construction goes through `Pipeline.open(path:)` (auto-detect) or
// one of the format-specific `from*` constructors. The handle is
// released in `deinit`, so callers don't have to remember to free it.
//
// Thread safety: libztok's encode/decode paths take a const pointer
// and the pipeline state is read-only after load (the worker pool
// lives on `BatchPool`, not here). Per src/c_api.zig encode/decode
// are safe to call from multiple threads against the same pipeline.
// We mark `Pipeline` as `@unchecked Sendable` for that reason —
// `unchecked` because the underlying type is a class with a mutable
// (closed) flag, not because the C-side guarantee is weak.

import CZtok
import Foundation

/// A loaded tokenizer pipeline.
///
/// Build via `Pipeline.open(path:)` (auto-detect) or a specific
/// `from*` constructor. The native handle is released when the
/// `Pipeline` is deallocated; you can also call `close()` explicitly
/// to release it earlier.
public final class Pipeline: @unchecked Sendable {
    /// Default version reported by `Pipeline.version()` if the C call
    /// returns NULL (shouldn't happen on a healthy build).
    public static let unknownVersion = "unknown"

    // The handle is mutable solely so `close()` can null it out.
    // After construction it is treated as immutable by the C ABI.
    private var handle: OpaquePointer?

    private init(handle: OpaquePointer) {
        self.handle = handle
    }

    deinit {
        if let h = handle {
            ztok_pipeline_free(h)
            handle = nil
        }
    }

    // MARK: - lifecycle

    /// Eagerly release the underlying native handle. Subsequent calls
    /// on this Pipeline will throw `ZtokError.closed`. Idempotent.
    public func close() {
        if let h = handle {
            ztok_pipeline_free(h)
            handle = nil
        }
    }

    /// `true` once `close()` has been called (or the pipeline failed
    /// to construct).
    public var isClosed: Bool {
        return handle == nil
    }

    /// Borrow the raw FFI handle. Internal — used by `BatchPool` and
    /// `StreamEncoder`. Throws if the pipeline is closed.
    internal func requireHandle(op: String) throws -> OpaquePointer {
        guard let h = handle else {
            throw ZtokError.closed(op: op)
        }
        return h
    }

    // MARK: - constructors

    /// Build the `byte_id` baseline pipeline (each input byte maps to
    /// its own id). Useful for tests / fuzzing because the round-trip
    /// is guaranteed exact for any byte sequence.
    public static func byteId(config: PipelineConfig = PipelineConfig()) throws -> Pipeline {
        var cfg = config.toC()
        var status: Int32 = 0
        let h = withUnsafePointer(to: &cfg) { cfgPtr in
            ztok_pipeline_new(cfgPtr, &status)
        }
        try checkStatus(status, op: "ztok_pipeline_new")
        guard let h else { throw ZtokError.nullHandle(op: "ztok_pipeline_new") }
        return Pipeline(handle: h)
    }

    /// Auto-detect `path`'s vocab format and dispatch to the right
    /// loader. WordPiece (which lives inside `tokenizer.json` but
    /// needs a specific `unkId`) is *not* covered — call
    /// `Pipeline.fromWordPiece` directly for that.
    public static func open(path: String) throws -> Pipeline {
        let fmt = detectFormat(path: path)
        switch fmt {
        case .tiktoken:
            return try fromTiktoken(path: path)
        case .hfJson, .tekken:
            // Tekken is BPE under the HF JSON loader in the other
            // bindings; route it through fromHfJson for symmetry.
            return try fromHfJson(path: path)
        case .sentencePiece:
            return try fromSentencePiece(path: path)
        case .ztm:
            return try fromMonster(path: path)
        case .rwkv:
            return try fromRwkv(path: path)
        case .unknown:
            throw ZtokError.unknownFormat(path: path)
        }
    }

    /// Convenience overload that takes a Foundation `URL` (file URLs only).
    public static func open(url: URL) throws -> Pipeline {
        return try open(path: url.path)
    }

    /// Load a `.tiktoken` vocab into a byte-level BPE pipeline. The
    /// CL100K pre-tokenizer is the default unless the caller passes a
    /// custom config.
    public static func fromTiktoken(path: String, config: PipelineConfig? = nil) throws -> Pipeline {
        let cfg = config ?? PipelineConfig(
            normalizer: .identity, preTokenizer: .cl100k, decoder: .concat)
        return try loadFile(
            path: path,
            config: cfg,
            op: "ztok_pipeline_new_bpe_from_tiktoken"
        ) { cPath, cfgPtr, statusPtr in
            ztok_pipeline_new_bpe_from_tiktoken(cPath, cfgPtr, statusPtr)
        }
    }

    /// Load a HuggingFace `tokenizer.json` BPE model.
    public static func fromHfJson(path: String, config: PipelineConfig? = nil) throws -> Pipeline {
        return try loadFile(
            path: path,
            config: config ?? PipelineConfig(),
            op: "ztok_pipeline_new_bpe_from_hf_json"
        ) { cPath, cfgPtr, statusPtr in
            ztok_pipeline_new_bpe_from_hf_json(cPath, cfgPtr, statusPtr)
        }
    }

    /// Load a HuggingFace WordPiece model from `tokenizer.json`.
    /// `unkId` is required (no sensible default for an unknown-token id).
    public static func fromWordPiece(
        path: String, unkId: UInt32, config: PipelineConfig? = nil
    ) throws -> Pipeline {
        let cfg = config ?? PipelineConfig(decoder: .wordPiece)
        return try loadFile(
            path: path,
            config: cfg,
            op: "ztok_pipeline_new_wordpiece_from_hf_json"
        ) { cPath, cfgPtr, statusPtr in
            ztok_pipeline_new_wordpiece_from_hf_json(cPath, unkId, cfgPtr, statusPtr)
        }
    }

    /// Load a SentencePiece `.model` (Unigram) file. `unkId` defaults
    /// to 0, matching the Python/Java bindings.
    public static func fromSentencePiece(
        path: String, unkId: UInt32 = 0, config: PipelineConfig? = nil
    ) throws -> Pipeline {
        return try loadFile(
            path: path,
            config: config ?? PipelineConfig(),
            op: "ztok_pipeline_new_unigram_from_sp_model"
        ) { cPath, cfgPtr, statusPtr in
            ztok_pipeline_new_unigram_from_sp_model(cPath, unkId, cfgPtr, statusPtr)
        }
    }

    /// Load a ztok TokenMonster `.ztm` vocab file.
    public static func fromMonster(path: String, config: PipelineConfig? = nil) throws -> Pipeline {
        return try loadFile(
            path: path,
            config: config ?? PipelineConfig(),
            op: "ztok_pipeline_new_monster_from_file"
        ) { cPath, cfgPtr, statusPtr in
            ztok_pipeline_new_monster_from_file(cPath, cfgPtr, statusPtr)
        }
    }

    /// Load an RWKV "World" vocab (`rwkv_vocab_v20230424.txt`) into a
    /// greedy longest-match byte-trie pipeline. The World scheme is
    /// byte-lossless (every byte 0..255 is a token), so it runs with an
    /// identity normalizer + identity pre-tokenizer + concat decoder;
    /// there is no pre-tokenizer to configure.
    public static func fromRwkv(path: String, config: PipelineConfig? = nil) throws -> Pipeline {
        return try loadFile(
            path: path,
            config: config ?? PipelineConfig(),
            op: "ztok_pipeline_new_rwkv_from_file"
        ) { cPath, cfgPtr, statusPtr in
            ztok_pipeline_new_rwkv_from_file(cPath, cfgPtr, statusPtr)
        }
    }

    /// Common loader plumbing: hand a C string + config pointer + status
    /// pointer to the constructor closure, then wrap the returned handle.
    private static func loadFile(
        path: String,
        config: PipelineConfig,
        op: String,
        body: (UnsafePointer<CChar>, UnsafePointer<ztok_pipeline_config>, UnsafeMutablePointer<ztok_status>) -> OpaquePointer?
    ) throws -> Pipeline {
        // Path must be NUL-terminated for the C ABI. CString conversion
        // catches interior NULs by returning nil from withCString.
        guard !path.isEmpty else { throw ZtokError.invalidPath(path) }
        var cfg = config.toC()
        var status: ztok_status = ZTOK_OK
        let handle: OpaquePointer? = path.withCString { cPath in
            withUnsafePointer(to: &cfg) { cfgPtr in
                body(cPath, cfgPtr, &status)
            }
        }
        try checkStatus(Int32(status.rawValue), op: op)
        guard let h = handle else { throw ZtokError.nullHandle(op: op) }
        return Pipeline(handle: h)
    }

    // MARK: - encode / decode

    /// Encode a UTF-8 string into a fresh `[UInt32]` of token ids.
    public func encode(_ text: String) throws -> [UInt32] {
        // Materialize once; reuse the byte view across the encode call.
        var bytes = Array(text.utf8)
        return try encodeBytes(&bytes)
    }

    /// Encode raw bytes into token ids. Use this for non-UTF-8 inputs;
    /// the underlying tokenizer treats input as a byte stream.
    public func encodeBytes(_ data: [UInt8]) throws -> [UInt32] {
        var copy = data
        return try encodeBytes(&copy)
    }

    /// Internal in-place encode that avoids one copy when the caller
    /// already owns a mutable `[UInt8]`.
    private func encodeBytes(_ data: inout [UInt8]) throws -> [UInt32] {
        let h = try requireHandle(op: "ztok_encode")
        if data.isEmpty { return [] }

        // libztok's per-span maxTokensFor upper bound is conservative,
        // so even a buffer sized to the true encoded length can trip
        // BUFFER_TOO_SMALL mid-stream. Start generous, double on retry,
        // hard-cap at 8 attempts (matches the Python/Go/Rust bindings).
        var cap = max(data.count + 16, 64)
        let maxAttempts = 8
        for _ in 0..<maxAttempts {
            var out = [UInt32](repeating: 0, count: cap)
            var outLen: Int = 0
            // Reinterpret &[UInt8] as `const char*` for the C ABI —
            // encode is a byte-level operation. withMemoryRebound is
            // the safe-pointer-shape conversion (CChar is signed
            // 8-bit, UInt8 is unsigned 8-bit, same layout).
            let rc: Int32 = data.withUnsafeBufferPointer { inBuf in
                inBuf.baseAddress!.withMemoryRebound(to: CChar.self, capacity: inBuf.count) { cIn in
                    out.withUnsafeMutableBufferPointer { outBuf in
                        let cStatus = ztok_encode(
                            h,
                            cIn,
                            inBuf.count,
                            outBuf.baseAddress,
                            outBuf.count,
                            &outLen
                        )
                        return Int32(cStatus.rawValue)
                    }
                }
            }
            if rc == Int32(ZTOK_OK.rawValue) {
                out.removeSubrange(outLen..<out.count)
                return out
            }
            if rc == Int32(ZTOK_ERR_BUFFER_TOO_SMALL.rawValue) {
                // out_len now carries the (conservative) required size.
                cap = max(cap * 2, outLen + 16)
                continue
            }
            try checkStatus(rc, op: "ztok_encode")
        }
        throw ZtokError.internalError(
            status: Int32(ZTOK_ERR_BUFFER_TOO_SMALL.rawValue),
            op: "ztok_encode: BUFFER_TOO_SMALL after \(maxAttempts) grow attempts"
        )
    }

    /// Decode token ids back to a UTF-8 string. Throws
    /// `ZtokError.invalidUtf8` if the result isn't valid UTF-8 — use
    /// `decodeBytes(_:)` to preserve raw bytes.
    public func decode(_ ids: [UInt32]) throws -> String {
        let bytes = try decodeBytes(ids)
        guard let s = String(bytes: bytes, encoding: .utf8) else {
            throw ZtokError.invalidUtf8
        }
        return s
    }

    /// Decode token ids to raw bytes (no UTF-8 round-tripping).
    public func decodeBytes(_ ids: [UInt32]) throws -> [UInt8] {
        let h = try requireHandle(op: "ztok_decode")
        if ids.isEmpty { return [] }

        // Sizing pass: pass out_cap=0 so the call writes the required
        // size into out_len and returns BUFFER_TOO_SMALL.
        var sized: Int = 0
        let rc1: Int32 = ids.withUnsafeBufferPointer { idsBuf in
            let cStatus = ztok_decode(
                h, idsBuf.baseAddress, idsBuf.count,
                nil, 0, &sized
            )
            return Int32(cStatus.rawValue)
        }
        if rc1 != Int32(ZTOK_OK.rawValue)
            && rc1 != Int32(ZTOK_ERR_BUFFER_TOO_SMALL.rawValue) {
            try checkStatus(rc1, op: "ztok_decode (sizing)")
        }
        if sized == 0 { return [] }

        var buf = [UInt8](repeating: 0, count: sized)
        var written: Int = 0
        let rc2: Int32 = ids.withUnsafeBufferPointer { idsBuf in
            buf.withUnsafeMutableBufferPointer { outBuf in
                // Same UInt8 <-> CChar rebound as the encode path.
                outBuf.baseAddress!.withMemoryRebound(to: CChar.self, capacity: outBuf.count) { cOut in
                    let cStatus = ztok_decode(
                        h, idsBuf.baseAddress, idsBuf.count,
                        cOut, outBuf.count, &written
                    )
                    return Int32(cStatus.rawValue)
                }
            }
        }
        try checkStatus(rc2, op: "ztok_decode")
        buf.removeSubrange(written..<buf.count)
        return buf
    }

    // MARK: - encode with overlays

    /// Result of `encodeWithOverlays`: the token id stream plus a map of
    /// each requested `OverlayKind` to its per-token value array. Every
    /// channel array has the same length as `ids`.
    public struct OverlayResult: Sendable {
        /// Token ids — identical to what `encode(_:)` returns.
        public let ids: [UInt32]
        /// Per-token overlay channels keyed by kind, aligned 1:1 with `ids`.
        public let channels: [OverlayKind: [UInt32]]
    }

    /// Encode a UTF-8 string and return the ids plus a map of per-token
    /// overlay channels aligned 1:1 with the id stream.
    ///
    /// Requesting overlays never changes tokenization — the ids are
    /// identical to `encode(_:)`. Cheap channels carry encoder-derived
    /// values; domain channels come back zero-filled until a domain
    /// plugin populates them. Duplicate kinds throw
    /// `ZtokError.invalidInput`.
    public func encodeWithOverlays(_ text: String, channels: [OverlayKind]) throws -> OverlayResult {
        return try encodeBytesWithOverlays(Array(text.utf8), channels: channels)
    }

    /// Raw-bytes form of `encodeWithOverlays(_:channels:)`. Use this for
    /// non-UTF-8 inputs; the underlying tokenizer treats input as a byte
    /// stream.
    public func encodeBytesWithOverlays(_ data: [UInt8], channels: [OverlayKind]) throws -> OverlayResult {
        let h = try requireHandle(op: "ztok_encode_with_overlays")

        // Reject duplicate kinds — the result map would silently collapse
        // them, which is almost certainly a caller bug.
        for (i, k) in channels.enumerated() where channels[..<i].contains(k) {
            throw ZtokError.invalidInput
        }

        func emptyResult() -> OverlayResult {
            var map = [OverlayKind: [UInt32]]()
            for k in channels { map[k] = [] }
            return OverlayResult(ids: [], channels: map)
        }

        if data.isEmpty { return emptyResult() }

        let nCh = channels.count

        // Sizing pass: out_ids = nil queries the token count. Build the
        // channels[] array with nil out pointers so the C side just
        // reports the count.
        var cChannels = channels.map { kind in
            ztok_overlay_channel(
                kind: ztok_overlay_kind(rawValue: kind.rawValue),
                out: nil,
                out_cap: 0)
        }

        var count: Int = 0
        let rc1: Int32 = data.withUnsafeBufferPointer { inBuf in
            inBuf.baseAddress!.withMemoryRebound(to: CChar.self, capacity: inBuf.count) { cIn in
                cChannels.withUnsafeMutableBufferPointer { chanBuf in
                    let cStatus = ztok_encode_with_overlays(
                        h,
                        cIn, inBuf.count,
                        nil, 0,
                        nCh == 0 ? nil : chanBuf.baseAddress, nCh,
                        &count)
                    return Int32(cStatus.rawValue)
                }
            }
        }
        if rc1 != Int32(ZTOK_OK.rawValue)
            && rc1 != Int32(ZTOK_ERR_BUFFER_TOO_SMALL.rawValue) {
            try checkStatus(rc1, op: "ztok_encode_with_overlays (sizing)")
        }
        if count == 0 { return emptyResult() }

        // Fill pass: allocate the id buffer plus one flat `[UInt32]`
        // backing store holding all channels contiguously (channel `i`
        // occupies `[i*count ..< (i+1)*count]`). A single flat buffer
        // sidesteps the exclusivity hazards of nesting
        // `withUnsafeMutableBufferPointer` over per-channel arrays.
        var ids = [UInt32](repeating: 0, count: count)
        var flat = [UInt32](repeating: 0, count: count * nCh)

        var outLen: Int = 0
        let rc2: Int32 = data.withUnsafeBufferPointer { inBuf in
            inBuf.baseAddress!.withMemoryRebound(to: CChar.self, capacity: inBuf.count) { cIn in
                ids.withUnsafeMutableBufferPointer { idsBuf in
                    flat.withUnsafeMutableBufferPointer { flatBuf in
                        // Point each channel descriptor at its slice of the
                        // flat backing store. With nCh == 0 the descriptor
                        // array is empty and we pass nil for channels.
                        var cChannels = (0..<nCh).map { i in
                            ztok_overlay_channel(
                                kind: ztok_overlay_kind(rawValue: channels[i].rawValue),
                                out: flatBuf.baseAddress.map { $0 + i * count },
                                out_cap: count)
                        }
                        return cChannels.withUnsafeMutableBufferPointer { chanBuf in
                            let cStatus = ztok_encode_with_overlays(
                                h,
                                cIn, inBuf.count,
                                idsBuf.baseAddress, count,
                                nCh == 0 ? nil : chanBuf.baseAddress, nCh,
                                &outLen)
                            return Int32(cStatus.rawValue)
                        }
                    }
                }
            }
        }
        try checkStatus(rc2, op: "ztok_encode_with_overlays")

        var map = [OverlayKind: [UInt32]]()
        for (i, k) in channels.enumerated() {
            map[k] = Array(flat[(i * count)..<((i + 1) * count)])
        }
        return OverlayResult(ids: ids, channels: map)
    }

    // MARK: - chunking

    /// Split `text` into overlapping token windows (late chunking). Each
    /// window holds at most `maxTokens` ids with `overlap` ids shared
    /// between neighbors (stride = `maxTokens - overlap`). Returns an
    /// empty array for empty input. Throws `ZtokError.invalidInput` when
    /// `maxTokens == 0` or `overlap >= maxTokens`.
    ///
    /// The C-owned id buffers are copied into Swift `[UInt32]` and freed
    /// via `ztok_chunks_free` before returning, so the result is fully
    /// owned by Swift.
    public func chunk(
        _ text: String,
        maxTokens: UInt32,
        overlap: UInt32 = 0,
        boundary: ChunkBoundary = .token
    ) throws -> [Chunk] {
        return try chunkBytes(
            Array(text.utf8),
            maxTokens: maxTokens,
            overlap: overlap,
            boundary: boundary)
    }

    /// Raw-bytes form of `chunk(_:maxTokens:overlap:boundary:)`. Use this
    /// for non-UTF-8 inputs; the underlying tokenizer treats input as a
    /// byte stream.
    public func chunkBytes(
        _ data: [UInt8],
        maxTokens: UInt32,
        overlap: UInt32 = 0,
        boundary: ChunkBoundary = .token
    ) throws -> [Chunk] {
        let h = try requireHandle(op: "ztok_chunk")

        // Validate args up front to surface a clean .invalidInput,
        // matching the Python/Go bindings (which check before the call).
        if maxTokens == 0 || overlap >= maxTokens {
            throw ZtokError.invalidInput
        }
        if data.isEmpty { return [] }

        return try data.withUnsafeBufferPointer { inBuf -> [Chunk] in
            try inBuf.baseAddress!.withMemoryRebound(to: CChar.self, capacity: inBuf.count) { cIn -> [Chunk] in
                // Sizing pass: out_chunks = nil -> *out_len = chunk count.
                var need: Int = 0
                let rc1 = ztok_chunk(
                    h, cIn, inBuf.count,
                    maxTokens, overlap, boundary.rawValue,
                    nil, 0, &need)
                let rc1i = Int32(rc1.rawValue)
                if rc1i != Int32(ZTOK_OK.rawValue)
                    && rc1i != Int32(ZTOK_ERR_BUFFER_TOO_SMALL.rawValue) {
                    try checkStatus(rc1i, op: "ztok_chunk (sizing)")
                }
                let count = need
                if count == 0 { return [] }

                // Fill pass: caller-owned record array of capacity `count`.
                var recs = [ztok_chunk_rec](
                    repeating: ztok_chunk_rec(),
                    count: count)
                var got: Int = 0
                let rc2: Int32 = recs.withUnsafeMutableBufferPointer { recBuf in
                    let cStatus = ztok_chunk(
                        h, cIn, inBuf.count,
                        maxTokens, overlap, boundary.rawValue,
                        recBuf.baseAddress, count, &got)
                    return Int32(cStatus.rawValue)
                }
                // On success, ztok_chunk allocated an id buffer per record;
                // copy the ids out, then release every buffer through
                // ztok_chunks_free (the only safe free path). Always run
                // the free, even if checkStatus throws.
                defer {
                    recs.withUnsafeMutableBufferPointer { recBuf in
                        ztok_chunks_free(recBuf.baseAddress, got)
                    }
                }
                try checkStatus(rc2, op: "ztok_chunk")

                var out = [Chunk]()
                out.reserveCapacity(got)
                for i in 0..<got {
                    let r = recs[i]
                    var ids = [UInt32]()
                    if let p = r.ids, r.ids_len > 0 {
                        ids = Array(UnsafeBufferPointer(start: p, count: r.ids_len))
                    }
                    out.append(Chunk(
                        ids: ids,
                        byteStart: r.byte_start,
                        byteEnd: r.byte_end,
                        tokenStart: r.token_start,
                        tokenEnd: r.token_end))
                }
                return out
            }
        }
    }

    // MARK: - fingerprint

    /// Compute the tokenizer fingerprint — a deterministic 32-byte
    /// SHA-256 digest over the pipeline's encoding behavior on a fixed
    /// canonical input set. Two pipelines that return the same
    /// fingerprint will produce bit-identical id streams for any
    /// input.
    public func fingerprint() throws -> Fingerprint {
        let h = try requireHandle(op: "ztok_fingerprint")
        var out = [UInt8](repeating: 0, count: Fingerprint.length)
        let rc: Int32 = out.withUnsafeMutableBufferPointer { buf in
            let cStatus = ztok_fingerprint(h, buf.baseAddress)
            return Int32(cStatus.rawValue)
        }
        try checkStatus(rc, op: "ztok_fingerprint")
        return Fingerprint(bytes: Data(out))
    }

    // MARK: - streaming convenience

    /// Convenience: open a streaming-encode session against this
    /// pipeline. Equivalent to `try StreamEncoder(pipeline: self,
    /// chunkSize: chunkSize)`.
    public func streamEncoder(chunkSize: Int = StreamEncoder.defaultChunkSize) throws -> StreamEncoder {
        return try StreamEncoder(pipeline: self, chunkSize: chunkSize)
    }
}

// MARK: - free functions

extension Pipeline {
    /// Return libztok's version string (e.g. `"1.24.0"`).
    public static func version() -> String {
        guard let raw = ztok_version() else { return Pipeline.unknownVersion }
        return String(cString: raw)
    }

    /// Sniff `path` for a known tokenizer format. Best-effort: any
    /// I/O error or unrecognized magic returns `.unknown` (matches
    /// the C contract — never raises).
    public static func detectFormat(path: String) -> Format {
        let code = path.withCString { ztok_auto_detect($0) }
        return Format.fromCode(code.rawValue)
    }
}
