// TekkenTests — Mistral Tekken tokenizer coverage. Mirrors the Python
// binding's tests/test_tekken.py: loads the real mistral_nemo_tekken.json
// fixture (skipped when absent) and checks ztok reproduces the golden ids
// verified against mistral_common 1.8.6.
//
// All cases skip via `try XCTSkipUnless(libztokAvailable)` when the
// shared library can't be loaded, and via XCTSkip when the vocab fixture
// is not present.

import XCTest

@testable import Ztok

final class TekkenTests: XCTestCase {

    // Golden id sequences verified against mistral_common 1.8.6 on
    // bench/vocabs/mistral_nemo_tekken.json.
    private static let golden: [(String, [UInt32])] = [
        ("Hello, world!", [22177, 1044, 4304, 1033]),
        ("The quick brown fox", [1784, 7586, 22980, 94137]),
        (" and the", [1321, 1278]),
    ]

    override func setUpWithError() throws {
        try XCTSkipUnless(TestSupport.libztokAvailable,
                          "libztok not loadable (set LD_LIBRARY_PATH or build with `zig build`)")
    }

    /// Locate `bench/vocabs/mistral_nemo_tekken.json` relative to this
    /// source file: Tests/ZtokTests/TekkenTests.swift -> ... -> repo root.
    private static func vocabPath() -> String {
        // #filePath: <repo>/bindings/swift/Tests/ZtokTests/TekkenTests.swift
        // Drop: TekkenTests.swift, ZtokTests, Tests, swift, bindings -> repo.
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 {
            url.deleteLastPathComponent()
        }
        return url
            .appendingPathComponent("bench")
            .appendingPathComponent("vocabs")
            .appendingPathComponent("mistral_nemo_tekken.json")
            .path
    }

    private func tekkenPipeline() throws -> Pipeline {
        let path = TekkenTests.vocabPath()
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("Tekken vocab fixture not present at \(path)")
        }
        return try Pipeline.fromTekken(path: path)
    }

    func testMatchesReference() throws {
        let pipe = try tekkenPipeline()
        defer { pipe.close() }
        for (text, want) in TekkenTests.golden {
            XCTAssertEqual(try pipe.encode(text), want, "encode mismatch for \(text)")
        }
    }

    func testAutoDetectViaOpen() throws {
        let path = TekkenTests.vocabPath()
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("Tekken vocab fixture not present at \(path)")
        }
        XCTAssertEqual(Pipeline.detectFormat(path: path), .tekken)
        let pipe = try Pipeline.open(path: path)
        defer { pipe.close() }
        XCTAssertEqual(try pipe.encode("Hello, world!"), [22177, 1044, 4304, 1033])
    }
}
