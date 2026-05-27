// RwkvTests — RWKV "World" tokenizer coverage. Mirrors the Python
// binding's tests/test_rwkv.py: loads the real rwkv_vocab_v20230424.txt
// fixture (skipped when absent) and checks ztok reproduces the canonical
// reference encodings, matching the in-tree gate in src/rwkv_world.zig.
//
// All cases skip via `try XCTSkipUnless(libztokAvailable)` when the
// shared library can't be loaded, and via XCTSkip when the vocab fixture
// is not present.

import XCTest

@testable import Ztok

final class RwkvTests: XCTestCase {

    // Golden id sequences captured from BlinkDL's canonical reference
    // tokenizer (see bench/rwkv_parity.py / src/rwkv_world.zig).
    private static let golden: [(String, [UInt32])] = [
        ("Hello, world!", [33155, 45, 40213, 34]),
        ("emoji 😀🚀✨ test",
         [34295, 33, 3319, 153, 129, 3319, 155, 129, 10059, 32223]),
        ("0 1 2 10 99 100", [49, 284, 285, 3483, 3572, 3483, 49]),
    ]

    override func setUpWithError() throws {
        try XCTSkipUnless(TestSupport.libztokAvailable,
                          "libztok not loadable (set LD_LIBRARY_PATH or build with `zig build`)")
    }

    /// Locate `bench/vocabs/rwkv_vocab_v20230424.txt` relative to this
    /// source file: Tests/ZtokTests/RwkvTests.swift -> ... -> repo root.
    private static func vocabPath() -> String {
        // #filePath: <repo>/bindings/swift/Tests/ZtokTests/RwkvTests.swift
        // Drop: RwkvTests.swift, ZtokTests, Tests, swift, bindings -> repo.
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 {
            url.deleteLastPathComponent()
        }
        return url
            .appendingPathComponent("bench")
            .appendingPathComponent("vocabs")
            .appendingPathComponent("rwkv_vocab_v20230424.txt")
            .path
    }

    private func rwkvPipeline() throws -> Pipeline {
        let path = RwkvTests.vocabPath()
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("RWKV vocab fixture not present at \(path)")
        }
        return try Pipeline.fromRwkv(path: path)
    }

    func testMatchesReference() throws {
        let pipe = try rwkvPipeline()
        defer { pipe.close() }
        for (text, want) in RwkvTests.golden {
            XCTAssertEqual(try pipe.encode(text), want, "encode mismatch for \(text)")
        }
    }

    func testRoundTrips() throws {
        let pipe = try rwkvPipeline()
        defer { pipe.close() }
        for (text, _) in RwkvTests.golden {
            let ids = try pipe.encode(text)
            XCTAssertEqual(try pipe.decode(ids), text)
        }
    }
}
