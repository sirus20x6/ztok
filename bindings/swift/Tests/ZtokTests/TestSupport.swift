// TestSupport — shared probe + synthetic tiktoken fixture used by the
// smoke and fuzz suites. Skipping happens at the XCTest level via
// `try XCTSkipUnless(...)`.

import Foundation
import XCTest

@testable import Ztok

enum TestSupport {
    /// True when libztok is loadable and `Pipeline.version()` returns a
    /// non-empty string. Cached after the first probe so each test
    /// pays the lookup once.
    static var libztokAvailable: Bool {
        if let cached = _cached { return cached }
        let v = Pipeline.version()
        let ok = !v.isEmpty && v != Pipeline.unknownVersion
        _cached = ok
        return ok
    }
    private static var _cached: Bool?

    /// Build a synthetic .tiktoken vocab covering all 256 single bytes
    /// plus a handful of merges. Same shape as the Python conftest and
    /// the Go/Rust binding fixtures — pick the same merges so a given
    /// input encodes to the same id sequence across bindings (modulo
    /// rank ordering).
    ///
    /// Returns nil if writing the file fails for any reason (read-only
    /// /tmp, permission denied) — the caller should `try XCTSkipIf`
    /// in that case.
    static func tiktokenFixture() -> URL? {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ztok-swift-\(ProcessInfo.processInfo.processIdentifier)")
        do {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        } catch {
            return nil
        }
        let path = dir.appendingPathComponent("synthetic_cl100k.tiktoken")
        var lines: [String] = []
        var rank: UInt32 = 0
        for b in 0..<256 {
            lines.append("\(base64Encode([UInt8(b)])) \(rank)")
            rank += 1
        }
        for extra in ["he", "hel", "hell", "hello",
                      " w", " wo", " wor", " worl", " world",
                      "th", "the", " th", " the",
                      "fo", "foo", "bar", "baz",
                      " quick", " brown", " fox"] {
            lines.append("\(base64Encode(Array(extra.utf8))) \(rank)")
            rank += 1
        }
        let body = lines.joined(separator: "\n") + "\n"
        do {
            try body.write(to: path, atomically: true, encoding: .utf8)
        } catch {
            return nil
        }
        return path
    }

    /// Tiny RFC 4648 base64 encoder — keeps the test suite free of
    /// external deps. tiktoken vocab files always use the standard
    /// alphabet with `=` padding.
    static func base64Encode(_ input: [UInt8]) -> String {
        let alphabet: [Character] = Array(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/")
        var out = String()
        out.reserveCapacity((input.count + 2) / 3 * 4)
        var i = 0
        while i < input.count {
            let b0 = input[i]
            let b1: UInt8 = (i + 1 < input.count) ? input[i + 1] : 0
            let b2: UInt8 = (i + 2 < input.count) ? input[i + 2] : 0
            let triple = (UInt32(b0) << 16) | (UInt32(b1) << 8) | UInt32(b2)
            out.append(alphabet[Int((triple >> 18) & 0x3F)])
            out.append(alphabet[Int((triple >> 12) & 0x3F)])
            out.append((i + 1 < input.count) ? alphabet[Int((triple >> 6) & 0x3F)] : "=")
            out.append((i + 2 < input.count) ? alphabet[Int(triple & 0x3F)] : "=")
            i += 3
        }
        return out
    }
}
