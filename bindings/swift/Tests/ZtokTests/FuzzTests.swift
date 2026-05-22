// FuzzTests — PRNG-driven round-trip fuzz harness for the Swift
// binding. Mirrors fuzz/encode_decode.zig and the corresponding
// Python/Node/Ruby/Rust/Java suites: a deterministic LCG mutates the
// byte input each iteration, encode-then-decode must round-trip
// exactly.
//
// The pipeline is byte_id (identity normalizer + identity pre-tok +
// byte_id model + concat decoder); by construction each input byte
// maps to one id and decoding concatenates them back, so for any
// byte sequence the invariant decode(encode(x)) == x must hold.
//
// Seed defaults to 0xFEEDB0B (matching the other bindings) — failures
// at a given iteration are cross-language reproducible. Override via
// `FUZZ_SEED` (decimal or `0x...` hex) and `ZTOK_FUZZ_ITERS`
// environment variables for nightly runs.

import XCTest

@testable import Ztok

final class FuzzTests: XCTestCase {
    private let defaultSeed: UInt64 = 0xFEEDB0B
    private let defaultIterations: Int = 1000
    private let maxLen: Int = 256

    override func setUpWithError() throws {
        try XCTSkipUnless(TestSupport.libztokAvailable,
                          "libztok not loadable (set LD_LIBRARY_PATH or build with `zig build`)")
    }

    func testByteIdRoundTripFuzz() throws {
        let env = ProcessInfo.processInfo.environment
        let seed = env["FUZZ_SEED"].flatMap(parseU64) ?? defaultSeed
        let iters = env["ZTOK_FUZZ_ITERS"].flatMap { Int($0) } ?? defaultIterations

        let pipe = try Pipeline.byteId()
        defer { pipe.close() }

        var rng = SplitMix64(seed: seed)
        var failures: [(Int, [UInt8], [UInt8])] = []

        for i in 0..<iters {
            let len = Int(rng.next() % UInt64(maxLen + 1))
            var data = [UInt8](repeating: 0, count: len)
            for j in 0..<len {
                data[j] = UInt8(rng.next() & 0xFF)
            }

            let ids = try pipe.encodeBytes(data)
            XCTAssertEqual(ids.count, data.count,
                           "iter \(i): byte_id produced \(ids.count) ids for \(data.count) bytes")
            let roundTrip = try pipe.decodeBytes(ids)
            if roundTrip != data {
                failures.append((i, data, roundTrip))
                if failures.count >= 5 { break }
            }
        }

        if !failures.isEmpty {
            var msg = "byte_id round-trip mismatches:\n"
            for (i, inBytes, outBytes) in failures {
                msg += "  iter \(i): in=\(inBytes) out=\(outBytes)\n"
            }
            XCTFail(msg)
        }
    }
}

/// Parse a u64 from decimal or `0x...` / `0X...` hex.
private func parseU64(_ raw: String) -> UInt64? {
    if raw.hasPrefix("0x") || raw.hasPrefix("0X") {
        return UInt64(raw.dropFirst(2), radix: 16)
    }
    return UInt64(raw)
}

/// Deterministic 64-bit PRNG (SplitMix64). Same shape across
/// bindings — picking any fixed PRNG is fine for fuzz coverage, and
/// SplitMix64 is small enough to inline here without pulling in
/// `swift-numerics`.
struct SplitMix64 {
    private var state: UInt64

    init(seed: UInt64) { self.state = seed }

    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}
