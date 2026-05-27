// Chunk — one token window produced by `Pipeline.chunk(...)`.
//
// Mirrors `ztok_chunk_rec` and the Python `Chunk` dataclass / Go `Chunk`
// struct. The `ids` are copied out of the ztok-allocated buffer (which is
// freed via `ztok_chunks_free` before the chunk is handed back), so a
// `Chunk` is fully Swift-owned with no lingering C pointers.

import Foundation

/// One token window from `Pipeline.chunk(_:maxTokens:overlap:boundary:)`.
///
/// `ids` are the token ids in this chunk. `byteStart`/`byteEnd` is the
/// half-open byte range the chunk covers in the ORIGINAL input;
/// `tokenStart`/`tokenEnd` the half-open token-index range in the full
/// encoding.
public struct Chunk: Sendable, Equatable {
    /// The token ids in this chunk (copied out of C memory).
    public let ids: [UInt32]
    /// Half-open byte range covered in the original input (start).
    public let byteStart: UInt32
    /// Half-open byte range covered in the original input (exclusive end).
    public let byteEnd: UInt32
    /// Half-open token-index range in the full encoding (start).
    public let tokenStart: UInt32
    /// Half-open token-index range in the full encoding (exclusive end).
    public let tokenEnd: UInt32

    public init(
        ids: [UInt32],
        byteStart: UInt32,
        byteEnd: UInt32,
        tokenStart: UInt32,
        tokenEnd: UInt32
    ) {
        self.ids = ids
        self.byteStart = byteStart
        self.byteEnd = byteEnd
        self.tokenStart = tokenStart
        self.tokenEnd = tokenEnd
    }
}
