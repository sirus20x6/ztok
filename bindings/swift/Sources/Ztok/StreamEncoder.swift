// StreamEncoder — streaming-encode wrapper around `ztok_stream*`.
//
// Two surfaces:
//
//   1. Direct sink-style API: `feed(_:)` / `finish()` returning
//      `[UInt32]` per call. Mirrors the Python/Rust/Java shape.
//
//   2. `IteratorProtocol`/`Sequence` over a fixed input buffer: drive
//      the encoder synchronously with `for chunk in stream { ... }`.
//
//   3. `AsyncSequence` adapter `IDStream` for native `for await id in
//      stream { ... }` ergonomics. Yields one `UInt32` at a time.
//
// The encoder defers a trailing partial UTF-8 codepoint /
// pre-tokenizer span up to a 1 MiB soft cap (see src/stream.zig);
// past that it force-cuts at the nearest codepoint boundary.

import CZtok
import Foundation

/// A streaming-encode session against a `Pipeline`.
///
/// Build via `Pipeline.streamEncoder(chunkSize:)` or the explicit
/// `StreamEncoder.init(pipeline:chunkSize:)`. Feed bytes via
/// `feed(_:)`, then `finish()` to drain the carry. `close()` releases
/// the native handle eagerly; the deinitializer does the same if the
/// caller forgets.
///
/// Not thread-safe — each task should own its own `StreamEncoder`.
public final class StreamEncoder {
    /// Default per-feed byte count (64 KiB) — matches the
    /// Python/Go/Rust/Java bindings.
    public static let defaultChunkSize: Int = 64 * 1024

    private var handle: OpaquePointer?
    // Keep the pipeline alive for the encoder's lifetime — the stream
    // holds a pointer to it internally (see src/stream.zig). If the
    // pipeline were freed first the stream would dangle on its next
    // feed/finish.
    private let _pipeline: Pipeline
    private var finished: Bool = false

    /// Open a new streaming session against `pipeline`. `chunkSize`
    /// is the per-feed byte count used when iterating over a fixed
    /// input buffer via `Sequence`/`AsyncSequence`. It does not bound
    /// what you may pass to `feed(_:)` directly.
    public init(pipeline: Pipeline, chunkSize: Int = StreamEncoder.defaultChunkSize) throws {
        precondition(chunkSize > 0, "chunkSize must be > 0")
        let pipeHandle = try pipeline.requireHandle(op: "ztok_stream_new")
        var status: ztok_status = ZTOK_OK
        let h = ztok_stream_new(pipeHandle, &status)
        try checkStatus(Int32(status.rawValue), op: "ztok_stream_new")
        guard let h else { throw ZtokError.nullHandle(op: "ztok_stream_new") }
        self.handle = h
        self._pipeline = pipeline
        self.chunkSize = chunkSize
    }

    deinit {
        if let h = handle {
            ztok_stream_free(h)
            handle = nil
        }
    }

    /// Eagerly release the native handle. Idempotent.
    public func close() {
        if let h = handle {
            ztok_stream_free(h)
            handle = nil
        }
    }

    public var isClosed: Bool { handle == nil }

    /// Chunk size used by the `Sequence`/`AsyncSequence` adapters.
    public let chunkSize: Int

    /// Feed `bytes` to the encoder. Returns any token ids newly
    /// emitted by this call (may be empty when the encoder is still
    /// buffering toward the next safe cut).
    public func feed(_ bytes: [UInt8]) throws -> [UInt32] {
        guard let h = handle else { throw ZtokError.closed(op: "ztok_stream_feed") }
        if finished {
            throw ZtokError.invalidInput
        }
        var outIds: UnsafeMutablePointer<UInt32>? = nil
        var outN: Int = 0
        let rc: Int32 = bytes.withUnsafeBufferPointer { buf in
            if buf.isEmpty {
                let cStatus = ztok_stream_feed(h, nil, 0, &outIds, &outN)
                return Int32(cStatus.rawValue)
            }
            return buf.baseAddress!.withMemoryRebound(
                to: CChar.self, capacity: buf.count
            ) { cIn in
                let cStatus = ztok_stream_feed(h, cIn, buf.count, &outIds, &outN)
                return Int32(cStatus.rawValue)
            }
        }
        // Even on error the C ABI may have allocated a partial buffer;
        // free it before propagating.
        if rc != Int32(ZTOK_OK.rawValue) {
            if let outIds {
                ztok_ids_free(outIds)
            }
            try checkStatus(rc, op: "ztok_stream_feed")
        }
        return materializeAndFree(ptr: outIds, count: outN)
    }

    /// Feed a UTF-8 string fragment. Equivalent to feeding `text.utf8`
    /// as `[UInt8]`.
    public func feed(_ text: String) throws -> [UInt32] {
        return try feed(Array(text.utf8))
    }

    /// Flush any remaining carry as a final encode. Idempotent: a
    /// second call returns an empty array.
    public func finish() throws -> [UInt32] {
        guard let h = handle else { throw ZtokError.closed(op: "ztok_stream_finish") }
        if finished { return [] }
        finished = true
        var outIds: UnsafeMutablePointer<UInt32>? = nil
        var outN: Int = 0
        let rc = Int32(ztok_stream_finish(h, &outIds, &outN).rawValue)
        if rc != Int32(ZTOK_OK.rawValue) {
            if let outIds {
                ztok_ids_free(outIds)
            }
            try checkStatus(rc, op: "ztok_stream_finish")
        }
        return materializeAndFree(ptr: outIds, count: outN)
    }
}

// MARK: - Sequence adapter over a fixed input buffer

extension Pipeline {
    /// Stream-encode `data` and produce one `[UInt32]` per non-empty
    /// emit; the final flush is yielded last. `chunkSize` controls
    /// the per-feed byte count.
    public func encodeStream(
        data: [UInt8], chunkSize: Int = StreamEncoder.defaultChunkSize
    ) throws -> StreamSequence {
        return try StreamSequence(pipeline: self, data: data, chunkSize: chunkSize)
    }

    /// Convenience overload taking a UTF-8 string.
    public func encodeStream(
        text: String, chunkSize: Int = StreamEncoder.defaultChunkSize
    ) throws -> StreamSequence {
        return try encodeStream(data: Array(text.utf8), chunkSize: chunkSize)
    }
}

/// Synchronous `Sequence` adapter over a `StreamEncoder` driven by a
/// fixed input buffer. Yields one non-empty `[UInt32]` per emit; the
/// final flush is yielded last.
public struct StreamSequence: Sequence {
    private let encoder: StreamEncoder
    private let data: [UInt8]
    private let chunkSize: Int

    internal init(pipeline: Pipeline, data: [UInt8], chunkSize: Int) throws {
        self.encoder = try StreamEncoder(pipeline: pipeline, chunkSize: chunkSize)
        self.data = data
        self.chunkSize = max(chunkSize, 1)
    }

    public func makeIterator() -> Iterator {
        return Iterator(encoder: encoder, data: data, chunkSize: chunkSize)
    }

    public struct Iterator: IteratorProtocol {
        public typealias Element = [UInt32]

        private let encoder: StreamEncoder
        private let data: [UInt8]
        private let chunkSize: Int
        private var cursor: Int = 0
        private var drained: Bool = false

        internal init(encoder: StreamEncoder, data: [UInt8], chunkSize: Int) {
            self.encoder = encoder
            self.data = data
            self.chunkSize = chunkSize
        }

        public mutating func next() -> [UInt32]? {
            // Loop until we either produce a non-empty batch, hit the
            // final flush, or exhaust everything. Empty emits between
            // cuts are skipped — exposing them would be noise.
            while true {
                if cursor < data.count {
                    let end = min(cursor + chunkSize, data.count)
                    let slice = Array(data[cursor..<end])
                    cursor = end
                    do {
                        let ids = try encoder.feed(slice)
                        if !ids.isEmpty { return ids }
                    } catch {
                        // The Sequence protocol can't propagate
                        // errors directly — fall through to end the
                        // iteration. Callers who care about errors
                        // should use the sink-style API instead.
                        return nil
                    }
                    continue
                }
                if !drained {
                    drained = true
                    do {
                        let tail = try encoder.finish()
                        return tail.isEmpty ? nil : tail
                    } catch {
                        return nil
                    }
                }
                return nil
            }
        }
    }
}

// MARK: - AsyncSequence adapter

extension StreamEncoder {
    /// Open an `AsyncSequence<UInt32>` view that yields ids one at a
    /// time as they're emitted by `feed(_:)` / `finish()` on `data`.
    /// Useful for native `for await id in stream { ... }` ergonomics.
    public func asyncEncode(data: [UInt8]) -> IDStream {
        return IDStream(encoder: self, data: data, chunkSize: chunkSize)
    }

    /// Convenience: take a UTF-8 string.
    public func asyncEncode(text: String) -> IDStream {
        return asyncEncode(data: Array(text.utf8))
    }
}

extension Pipeline {
    /// One-shot AsyncSequence convenience: open a fresh encoder and
    /// stream over `data`. The encoder is closed when iteration ends
    /// (or the task is cancelled).
    public func asyncEncodeStream(
        data: [UInt8], chunkSize: Int = StreamEncoder.defaultChunkSize
    ) throws -> IDStream {
        let enc = try StreamEncoder(pipeline: self, chunkSize: chunkSize)
        return IDStream(encoder: enc, data: data, chunkSize: chunkSize, ownsEncoder: true)
    }

    /// Convenience overload taking a UTF-8 string.
    public func asyncEncodeStream(
        text: String, chunkSize: Int = StreamEncoder.defaultChunkSize
    ) throws -> IDStream {
        return try asyncEncodeStream(data: Array(text.utf8), chunkSize: chunkSize)
    }
}

/// `AsyncSequence<UInt32>` adapter yielding one token id at a time.
/// Element-level granularity (rather than per-emit `[UInt32]`) gives
/// the canonical `for await id in stream` ergonomics; callers who
/// want batches can use the `Sequence` form (`StreamSequence`).
public struct IDStream: AsyncSequence {
    public typealias Element = UInt32

    private let encoder: StreamEncoder
    private let data: [UInt8]
    private let chunkSize: Int
    private let ownsEncoder: Bool

    internal init(encoder: StreamEncoder, data: [UInt8], chunkSize: Int, ownsEncoder: Bool = false) {
        self.encoder = encoder
        self.data = data
        self.chunkSize = chunkSize
        self.ownsEncoder = ownsEncoder
    }

    public func makeAsyncIterator() -> AsyncIterator {
        return AsyncIterator(encoder: encoder, data: data, chunkSize: chunkSize, ownsEncoder: ownsEncoder)
    }

    public final class AsyncIterator: AsyncIteratorProtocol {
        public typealias Element = UInt32

        private let encoder: StreamEncoder
        private let data: [UInt8]
        private let chunkSize: Int
        private let ownsEncoder: Bool
        private var cursor: Int = 0
        private var drained: Bool = false
        private var buffer: [UInt32] = []
        private var bufIdx: Int = 0

        internal init(encoder: StreamEncoder, data: [UInt8], chunkSize: Int, ownsEncoder: Bool) {
            self.encoder = encoder
            self.data = data
            self.chunkSize = chunkSize
            self.ownsEncoder = ownsEncoder
        }

        deinit {
            if ownsEncoder {
                encoder.close()
            }
        }

        public func next() async throws -> UInt32? {
            // Honor cooperative cancellation between emits.
            if Task.isCancelled { return nil }

            // Drain any buffered ids from the last C call first.
            if bufIdx < buffer.count {
                let id = buffer[bufIdx]
                bufIdx += 1
                return id
            }
            buffer.removeAll(keepingCapacity: true)
            bufIdx = 0

            while buffer.isEmpty {
                if cursor < data.count {
                    let end = min(cursor + chunkSize, data.count)
                    let slice = Array(data[cursor..<end])
                    cursor = end
                    buffer = try encoder.feed(slice)
                    continue
                }
                if !drained {
                    drained = true
                    buffer = try encoder.finish()
                    if buffer.isEmpty { return nil }
                    continue
                }
                return nil
            }

            let id = buffer[bufIdx]
            bufIdx += 1
            return id
        }
    }
}
