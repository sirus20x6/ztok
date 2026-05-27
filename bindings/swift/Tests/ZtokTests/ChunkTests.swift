// ChunkTests — token-window chunking coverage. Mirrors the Python
// binding's tests/test_chunk.py. Runs over a byte_id pipeline (each input
// byte = one token) so chunk boundaries are predictable: "abcdefghij" is
// 10 tokens, one per byte.
//
// All cases skip via `try XCTSkipUnless(libztokAvailable)` when the
// shared library can't be loaded.

import XCTest

@testable import Ztok

final class ChunkTests: XCTestCase {

    override func setUpWithError() throws {
        try XCTSkipUnless(TestSupport.libztokAvailable,
                          "libztok not loadable (set LD_LIBRARY_PATH or build with `zig build`)")
    }

    func testNonOverlapping() throws {
        let pipe = try Pipeline.byteId()
        defer { pipe.close() }
        let chunks = try pipe.chunk("abcdefghij", maxTokens: 4, overlap: 0)
        // 10 tokens / window 4, stride 4 -> [0,4) [4,8) [8,10).
        XCTAssertEqual(chunks.count, 3)
        // (tokenStart, tokenEnd, byteStart, byteEnd, idsCount)
        let want: [(UInt32, UInt32, UInt32, UInt32, Int)] = [
            (0, 4, 0, 4, 4),
            (4, 8, 4, 8, 4),
            (8, 10, 8, 10, 2),
        ]
        for (c, w) in zip(chunks, want) {
            XCTAssertEqual(c.tokenStart, w.0)
            XCTAssertEqual(c.tokenEnd, w.1)
            XCTAssertEqual(c.byteStart, w.2)
            XCTAssertEqual(c.byteEnd, w.3)
            XCTAssertEqual(c.ids.count, w.4)
        }
    }

    func testOverlap() throws {
        let pipe = try Pipeline.byteId()
        defer { pipe.close() }
        let chunks = try pipe.chunk("abcdefghij", maxTokens: 4, overlap: 2)
        XCTAssertGreaterThanOrEqual(chunks.count, 2)
        // stride = 2, so the last 2 ids of chunk[i] equal the first 2 of
        // chunk[i+1].
        for (a, b) in zip(chunks, chunks.dropFirst()) {
            if a.ids.count >= 2 && b.ids.count >= 2 {
                XCTAssertEqual(Array(a.ids.suffix(2)), Array(b.ids.prefix(2)))
            }
        }
    }

    func testEmptyAndBadArgs() throws {
        let pipe = try Pipeline.byteId()
        defer { pipe.close() }
        XCTAssertEqual(try pipe.chunk("", maxTokens: 4), [])
        XCTAssertThrowsError(try pipe.chunk("abc", maxTokens: 0)) { error in
            guard case ZtokError.invalidInput = error else {
                return XCTFail("expected .invalidInput; got \(error)")
            }
        }
        XCTAssertThrowsError(try pipe.chunk("abc", maxTokens: 4, overlap: 4)) { error in
            guard case ZtokError.invalidInput = error else {
                return XCTFail("expected .invalidInput; got \(error)")
            }
        }
    }
}
