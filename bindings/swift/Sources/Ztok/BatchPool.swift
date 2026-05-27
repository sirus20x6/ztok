// BatchPool — persistent multi-threaded encode pool.
//
// Wraps `ztok_batch_pool*`. Create one, reuse across many batched
// encodes — each pool owns its own arenas and worker threads, so
// creating one per batch wastes setup work.
//
// Output handling: libztok returns each per-input id buffer with an
// opaque length-prefix header (see src/c_api.zig::allocIdBuf). The
// ONLY safe free is `ztok_ids_free`. We always materialize each buffer
// into a Swift `[UInt32]` and free it through `ztok_ids_free` before
// returning — no foreign pointers leak into safe Swift.

import CZtok
import Foundation

/// A persistent multi-threaded encode worker pool.
public final class BatchPool: @unchecked Sendable {
    private var handle: OpaquePointer?

    /// Build a persistent worker pool. `workers = 0` auto-detects the
    /// CPU count.
    public init(workers: UInt32 = 0) throws {
        var status: ztok_status = ZTOK_OK
        let h = ztok_batch_pool_new(workers, &status)
        try checkStatus(Int32(status.rawValue), op: "ztok_batch_pool_new")
        guard let h else {
            throw ZtokError.nullHandle(op: "ztok_batch_pool_new")
        }
        self.handle = h
    }

    deinit {
        if let h = handle {
            ztok_batch_pool_free(h)
            handle = nil
        }
    }

    /// Eagerly release the underlying native handle. Subsequent calls
    /// on this pool will throw `ZtokError.closed`. Idempotent.
    public func close() {
        if let h = handle {
            ztok_batch_pool_free(h)
            handle = nil
        }
    }

    public var isClosed: Bool { handle == nil }

    /// Borrow the raw FFI handle. Internal — used by `Engram`'s batch
    /// n-gram path. Throws if the pool is closed.
    internal func requireHandle(op: String) throws -> OpaquePointer {
        guard let h = handle else {
            throw ZtokError.closed(op: op)
        }
        return h
    }

    /// Actual worker count (resolves `workers = 0` to the detected
    /// CPU count).
    public var workerCount: Int {
        guard let h = handle else { return 0 }
        return ztok_batch_pool_worker_count(h)
    }

    /// Encode many UTF-8 strings in parallel. Returns one `[UInt32]`
    /// per input, in the same order. Empty inputs produce empty
    /// arrays.
    public func encodeBatch(pipeline: Pipeline, inputs: [String]) throws -> [[UInt32]] {
        let byteInputs = inputs.map { Array($0.utf8) }
        return try encodeBatchBytes(pipeline: pipeline, inputs: byteInputs)
    }

    /// Encode many byte arrays in parallel. Use for non-UTF-8 inputs.
    public func encodeBatchBytes(pipeline: Pipeline, inputs: [[UInt8]]) throws -> [[UInt32]] {
        guard let pool = handle else {
            throw ZtokError.closed(op: "ztok_encode_batch_pooled")
        }
        let pipeHandle = try pipeline.requireHandle(op: "ztok_encode_batch_pooled")

        let n = inputs.count
        if n == 0 { return [] }

        // The C ABI takes parallel arrays of (char* ptr, size_t len).
        // Inputs may legitimately contain NUL bytes — we pass plain
        // byte pointers with explicit lengths rather than CStrings.
        //
        // Strategy: copy all inputs into one contiguous backing buffer
        // and compute offsets. That gives us stable pointers without
        // a recursion-as-deep-as-inputs.count `withUnsafeBufferPointer`
        // chain and bounds the allocation to a single heap object.
        let lens: [Int] = inputs.map { $0.count }
        let totalLen = lens.reduce(0, +)
        // Allocate at least one byte so backing.baseAddress is non-nil
        // even when every input is empty.
        var backing = [UInt8](repeating: 0, count: max(totalLen, 1))
        var offsets = [Int](repeating: 0, count: n)
        do {
            var cursor = 0
            for i in 0..<n {
                offsets[i] = cursor
                let b = inputs[i]
                if !b.isEmpty {
                    backing.replaceSubrange(cursor..<(cursor + b.count), with: b)
                }
                cursor += b.count
            }
        }

        var outIds = [UnsafeMutablePointer<UInt32>?](repeating: nil, count: n)
        var outLens = [Int](repeating: 0, count: n)

        // Inside withUnsafeBufferPointer on `backing` we have a
        // stable base address for the contiguous storage. Build the
        // pointer array there so every `cInputPtrs[i]` points into a
        // region that lives across the C call.
        let rc: Int32 = backing.withUnsafeBufferPointer { backingBuf in
            let basePtr = backingBuf.baseAddress!
            var cInputPtrs: [UnsafePointer<CChar>?] = []
            cInputPtrs.reserveCapacity(n)
            for i in 0..<n {
                if lens[i] == 0 {
                    cInputPtrs.append(nil)
                } else {
                    let raw = UnsafeRawPointer(basePtr.advanced(by: offsets[i]))
                    cInputPtrs.append(raw.assumingMemoryBound(to: CChar.self))
                }
            }
            return cInputPtrs.withUnsafeBufferPointer { inPtrs in
                lens.withUnsafeBufferPointer { lenPtrs in
                    outIds.withUnsafeMutableBufferPointer { outIdsBuf in
                        outLens.withUnsafeMutableBufferPointer { outLensBuf in
                            let cStatus = ztok_encode_batch_pooled(
                                pipeHandle,
                                pool,
                                inPtrs.baseAddress,
                                lenPtrs.baseAddress,
                                n,
                                outIdsBuf.baseAddress,
                                outLensBuf.baseAddress
                            )
                            return Int32(cStatus.rawValue)
                        }
                    }
                }
            }
        }

        // ALWAYS materialize-and-free, even on error, so we never leak
        // C buffers from a partially-populated batch.
        var results = [[UInt32]]()
        results.reserveCapacity(n)
        for i in 0..<n {
            let ptr = outIds[i]
            let len = outLens[i]
            results.append(materializeAndFree(ptr: ptr, count: len))
        }

        try checkStatus(rc, op: "ztok_encode_batch_pooled")
        return results
    }
}

/// Copy a libztok-owned id buffer into a Swift `[UInt32]` and free it
/// via `ztok_ids_free`. The C buffer carries an opaque length-prefix
/// header (see src/c_api.zig::allocIdBuf) so this is the only safe
/// free path — never call `free()`.
@inline(__always)
internal func materializeAndFree(ptr: UnsafeMutablePointer<UInt32>?, count: Int) -> [UInt32] {
    guard let ptr = ptr else { return [] }
    if count == 0 {
        // Even for an empty buffer the C side may hand us a (possibly
        // non-null) header to free.
        ztok_ids_free(ptr)
        return []
    }
    let buf = UnsafeBufferPointer(start: ptr, count: count)
    let out = Array(buf)
    ztok_ids_free(ptr)
    return out
}
