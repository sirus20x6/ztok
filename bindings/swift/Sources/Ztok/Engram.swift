// Engram — deterministic multi-head token-n-gram hashing.
//
// Wraps `ztok_ngram_hash` / `ztok_ngram_hash_batch` (see src/ngram.zig).
// These operate on raw token ids and need no Pipeline; the output is
// row-major [position][head] raw UInt64 hashes which the caller masks to
// its own table width (hash & ((1 << bits) - 1)). The number of window
// positions for an `ids`-long stream is (ids.count - n + 1), or 0 if the
// stream is shorter than one window.
//
// Mirrors the Python `ngram_hash` / `ngram_hash_batch` free functions and
// the Go `HashNGrams` / `HashNGramsBatch` package functions. The batch
// path's per-doc UInt64 buffers are ztok-allocated and freed (each) via
// `ztok_u64s_free` before returning, so no foreign pointers leak.

import CZtok
import Foundation

/// Namespace for Engram n-gram hashing entry points.
public enum Engram {

    /// Hash every length-`n` window of `ids` under `heads` independent
    /// hash functions, returning the row-major `[position][head]` UInt64
    /// hashes (positions = `ids.count - n + 1`, or 0 if the stream is
    /// shorter than one window). Mask each hash to your table width
    /// (`hash & ((1 << bits) - 1)`). Deterministic: identical ids always
    /// yield identical hashes. Returns `[]` when there is nothing to hash
    /// (empty input, `n == 0`, `heads == 0`, or a stream shorter than one
    /// window) — never an error.
    public static func hashNGrams(_ ids: [UInt32], n: UInt32, heads: UInt32) throws -> [UInt64] {
        if ids.isEmpty || n == 0 || heads == 0 || ids.count < Int(n) {
            return []
        }
        let positions = ids.count - Int(n) + 1
        let want = positions * Int(heads)
        if want == 0 { return [] }

        // We size exactly, so the first call should succeed; honor the
        // BUFFER_TOO_SMALL contract with a single re-size for safety
        // (mirrors the Python binding's two-attempt loop).
        var cap = want
        for _ in 0..<2 {
            var out = [UInt64](repeating: 0, count: cap)
            var outLen: Int = 0
            let rc: Int32 = ids.withUnsafeBufferPointer { idsBuf in
                out.withUnsafeMutableBufferPointer { outBuf in
                    let cStatus = ztok_ngram_hash(
                        idsBuf.baseAddress, idsBuf.count,
                        n, heads,
                        outBuf.baseAddress, outBuf.count, &outLen)
                    return Int32(cStatus.rawValue)
                }
            }
            if rc == Int32(ZTOK_OK.rawValue) {
                out.removeSubrange(outLen..<out.count)
                return out
            }
            if rc == Int32(ZTOK_ERR_BUFFER_TOO_SMALL.rawValue) {
                if outLen == 0 { return [] }
                cap = outLen
                continue
            }
            try checkStatus(rc, op: "ztok_ngram_hash")
        }
        throw ZtokError.internalError(
            status: Int32(ZTOK_ERR_BUFFER_TOO_SMALL.rawValue),
            op: "ztok_ngram_hash: BUFFER_TOO_SMALL twice")
    }

    /// Hash many id streams in parallel across `pool`. `results[i]` holds
    /// the row-major hashes for `streams[i]` (`[]` for a stream shorter
    /// than one window). Equivalent to calling `hashNGrams` on each
    /// stream, fanned out across the pool's workers.
    public static func hashNGramsBatch(
        pool: BatchPool,
        _ streams: [[UInt32]],
        n: UInt32,
        heads: UInt32
    ) throws -> [[UInt64]] {
        let poolHandle = try pool.requireHandle(op: "ztok_ngram_hash_batch")

        let nDocs = streams.count
        if nDocs == 0 { return [] }

        // The C ABI takes parallel arrays of (const ztok_token_id* ptr,
        // size_t len). Copy every stream into one contiguous backing
        // buffer and compute offsets so each input pointer is stable
        // across the C call (same strategy as BatchPool.encodeBatchBytes).
        let lens: [Int] = streams.map { $0.count }
        let totalLen = lens.reduce(0, +)
        // At least one element so backing.baseAddress is non-nil even
        // when every stream is empty.
        var backing = [UInt32](repeating: 0, count: max(totalLen, 1))
        var offsets = [Int](repeating: 0, count: nDocs)
        do {
            var cursor = 0
            for i in 0..<nDocs {
                offsets[i] = cursor
                let s = streams[i]
                if !s.isEmpty {
                    backing.replaceSubrange(cursor..<(cursor + s.count), with: s)
                }
                cursor += s.count
            }
        }

        var outHashes = [UnsafeMutablePointer<UInt64>?](repeating: nil, count: nDocs)
        var outLens = [Int](repeating: 0, count: nDocs)

        let rc: Int32 = backing.withUnsafeBufferPointer { backingBuf in
            let basePtr = backingBuf.baseAddress!
            // Build the array of per-doc id pointers into the contiguous
            // backing store. A length-0 stream gets a nil pointer.
            var idPtrs: [UnsafePointer<UInt32>?] = []
            idPtrs.reserveCapacity(nDocs)
            for i in 0..<nDocs {
                if lens[i] == 0 {
                    idPtrs.append(nil)
                } else {
                    idPtrs.append(basePtr.advanced(by: offsets[i]))
                }
            }
            return idPtrs.withUnsafeBufferPointer { inPtrs in
                lens.withUnsafeBufferPointer { lenPtrs in
                    outHashes.withUnsafeMutableBufferPointer { outHashesBuf in
                        outLens.withUnsafeMutableBufferPointer { outLensBuf in
                            let cStatus = ztok_ngram_hash_batch(
                                poolHandle,
                                inPtrs.baseAddress,
                                lenPtrs.baseAddress,
                                nDocs,
                                n, heads,
                                outHashesBuf.baseAddress,
                                outLensBuf.baseAddress)
                            return Int32(cStatus.rawValue)
                        }
                    }
                }
            }
        }

        // ALWAYS materialize-and-free, even on error, so we never leak
        // C buffers from a partially-populated batch.
        var results = [[UInt64]]()
        results.reserveCapacity(nDocs)
        for i in 0..<nDocs {
            results.append(materializeHashesAndFree(ptr: outHashes[i], count: outLens[i]))
        }

        try checkStatus(rc, op: "ztok_ngram_hash_batch")
        return results
    }
}

/// Copy a libztok-owned UInt64 buffer into a Swift `[UInt64]` and free it
/// via `ztok_u64s_free`. The C buffer carries an opaque length-prefix
/// header (see src/c_api.zig::allocU64Buf) so this is the only safe free
/// path — never call `free()`.
@inline(__always)
internal func materializeHashesAndFree(ptr: UnsafeMutablePointer<UInt64>?, count: Int) -> [UInt64] {
    guard let ptr = ptr else { return [] }
    if count <= 0 {
        // Even an empty buffer may be a (possibly non-null) header to free.
        ztok_u64s_free(ptr)
        return []
    }
    let buf = UnsafeBufferPointer(start: ptr, count: count)
    let out = Array(buf)
    ztok_u64s_free(ptr)
    return out
}
