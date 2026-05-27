// NGramTests — Engram n-gram hashing coverage. Mirrors the Python
// binding's tests/test_ngram.py: deterministic multi-head token-n-gram
// hashes, row-major [position][head], positions = ids.count - n + 1.
//
// All cases skip via `try XCTSkipUnless(libztokAvailable)` when the
// shared library can't be loaded.

import XCTest

@testable import Ztok

final class NGramTests: XCTestCase {

    override func setUpWithError() throws {
        try XCTSkipUnless(TestSupport.libztokAvailable,
                          "libztok not loadable (set LD_LIBRARY_PATH or build with `zig build`)")
    }

    func testLengthMath() throws {
        let ids: [UInt32] = [1, 2, 3, 4, 5]
        // 5 ids, n=2 -> 4 positions; heads=3 -> 12 hashes.
        let out = try Engram.hashNGrams(ids, n: 2, heads: 3)
        XCTAssertEqual(out.count, 4 * 3)
    }

    func testDeterministic() throws {
        let ids: [UInt32] = [7, 8, 9, 10, 11, 12]
        let a = try Engram.hashNGrams(ids, n: 3, heads: 4)
        let b = try Engram.hashNGrams(ids, n: 3, heads: 4)
        XCTAssertEqual(a, b)
        XCTAssertFalse(a.isEmpty)
    }

    func testHeadIndependence() throws {
        // The heads of a single position should not all collide.
        let out = try Engram.hashNGrams([42, 43, 44], n: 2, heads: 4)
        let firstPosition = Array(out.prefix(4))
        XCTAssertGreaterThan(Set(firstPosition).count, 1)
    }

    func testShortAndBadArgs() throws {
        // Stream shorter than one window -> empty.
        XCTAssertEqual(try Engram.hashNGrams([1, 2], n: 3, heads: 2), [])
        // Degenerate args -> empty (no error).
        XCTAssertEqual(try Engram.hashNGrams([], n: 1, heads: 1), [])
        XCTAssertEqual(try Engram.hashNGrams([1, 2, 3], n: 0, heads: 1), [])
        XCTAssertEqual(try Engram.hashNGrams([1, 2, 3], n: 2, heads: 0), [])
    }

    func testBatchMatchesSingle() throws {
        let streams: [[UInt32]] = [
            [1, 2, 3, 4],
            [],             // empty -> no hashes
            [9],            // shorter than window -> no hashes
            [5, 6, 7, 8, 9],
        ]
        let pool = try BatchPool(workers: 2)
        defer { pool.close() }
        let batched = try Engram.hashNGramsBatch(pool: pool, streams, n: 2, heads: 3)
        XCTAssertEqual(batched.count, streams.count)
        for (s, got) in zip(streams, batched) {
            XCTAssertEqual(got, try Engram.hashNGrams(s, n: 2, heads: 3))
        }
    }
}
