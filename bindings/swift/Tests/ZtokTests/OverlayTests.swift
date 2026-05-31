// OverlayTests — coverage for Pipeline.encodeWithOverlays
// (ztok_encode_with_overlays). Mirrors bindings/rust/tests/overlays.rs
// and the other bindings' overlay suites.
//
// All cases skip via `try XCTSkipUnless(libztokAvailable)` when the
// shared library can't be loaded.

import XCTest

@testable import Ztok

final class OverlayTests: XCTestCase {

    override func setUpWithError() throws {
        try XCTSkipUnless(TestSupport.libztokAvailable,
                          "libztok not loadable (set LD_LIBRARY_PATH or build with `zig build`)")
    }

    private func bpePipeline() throws -> Pipeline {
        guard let fixture = TestSupport.tiktokenFixture() else {
            throw XCTSkip("could not build tiktoken fixture in /tmp")
        }
        return try Pipeline.fromTiktoken(path: fixture.path)
    }

    func testIdsMatchPlainEncode() throws {
        let pipe = try bpePipeline()
        defer { pipe.close() }
        let text = "hello world"
        let plain = try pipe.encode(text)
        let res = try pipe.encodeWithOverlays(text, channels: [.byteStart, .byteEnd])
        XCTAssertEqual(res.ids, plain, "overlays must not change tokenization")
        XCTAssertEqual(res.channels.count, 2)
        XCTAssertNotNil(res.channels[.byteStart])
        XCTAssertNotNil(res.channels[.byteEnd])
    }

    func testChannelLengthsEqualIds() throws {
        let pipe = try bpePipeline()
        defer { pipe.close() }
        let res = try pipe.encodeWithOverlays(
            "the quick brown fox",
            channels: [.byteStart, .byteEnd, .boundary, .provenance])
        for (kind, values) in res.channels {
            XCTAssertEqual(values.count, res.ids.count, "channel \(kind) length mismatch")
        }
    }

    func testByteSpansAreSensible() throws {
        let pipe = try bpePipeline()
        defer { pipe.close() }
        let text = "hello world"
        let res = try pipe.encodeWithOverlays(text, channels: [.byteStart, .byteEnd])
        let starts = try XCTUnwrap(res.channels[.byteStart])
        let ends = try XCTUnwrap(res.channels[.byteEnd])
        let n = UInt32(Array(text.utf8).count)
        XCTAssertFalse(starts.isEmpty)
        for (s, e) in zip(starts, ends) {
            XCTAssertTrue(s < e && e <= n, "bad span (\(s), \(e)) for \(n) bytes")
        }
        XCTAssertEqual(starts.first, 0)
        XCTAssertEqual(ends.last, n)
        for i in 1..<starts.count {
            XCTAssertEqual(starts[i], ends[i - 1], "spans must tile left-to-right")
        }
    }

    func testByteIdSingleByteSpans() throws {
        let pipe = try Pipeline.byteId()
        defer { pipe.close() }
        let res = try pipe.encodeWithOverlays("hi", channels: [.byteStart, .byteEnd])
        XCTAssertEqual(res.ids, [0x68, 0x69])
        XCTAssertEqual(res.channels[.byteStart], [0, 1])
        XCTAssertEqual(res.channels[.byteEnd], [1, 2])
    }

    func testOpcodeDomainChannelIsAllZero() throws {
        let pipe = try bpePipeline()
        defer { pipe.close() }
        let res = try pipe.encodeWithOverlays("hello world", channels: [.opcode])
        let opcode = try XCTUnwrap(res.channels[.opcode])
        XCTAssertEqual(opcode.count, res.ids.count)
        XCTAssertTrue(opcode.allSatisfy { $0 == 0 },
                      "OPCODE must be zero-filled without a domain plugin")
    }

    func testEmptyInputReturnsEmptyChannels() throws {
        let pipe = try Pipeline.byteId()
        defer { pipe.close() }
        let res = try pipe.encodeWithOverlays("", channels: [.byteStart, .opcode])
        XCTAssertTrue(res.ids.isEmpty)
        XCTAssertEqual(res.channels[.byteStart], [])
        XCTAssertEqual(res.channels[.opcode], [])
    }

    func testNoChannelsReturnsJustIds() throws {
        let pipe = try bpePipeline()
        defer { pipe.close() }
        let res = try pipe.encodeWithOverlays("hello world", channels: [])
        XCTAssertEqual(res.ids, try pipe.encode("hello world"))
        XCTAssertTrue(res.channels.isEmpty)
    }

    func testDuplicateKindsRejected() throws {
        let pipe = try Pipeline.byteId()
        defer { pipe.close() }
        XCTAssertThrowsError(
            try pipe.encodeWithOverlays("hi", channels: [.byteStart, .byteStart])
        ) { error in
            guard case ZtokError.invalidInput = error else {
                return XCTFail("expected .invalidInput; got \(error)")
            }
        }
    }

    // x86-64 machine code: 48 89 d8 (mov rax,rbx) / e8 00000000 (call rel32) /
    // c3 (ret). With byteId each byte is its own token, so the OPCODE channel
    // carries one class per byte.
    private static let x86_64Code: [UInt8] =
        [0x48, 0x89, 0xd8, 0xe8, 0x00, 0x00, 0x00, 0x00, 0xc3]

    func testSetOverlayDomainX8664PopulatesOpcode() throws {
        let pipe = try Pipeline.byteId()
        defer { pipe.close() }

        // Default domain (.none): OPCODE is zero-filled.
        let none = try pipe.encodeBytesWithOverlays(
            OverlayTests.x86_64Code, channels: [.opcode])
        let opcodeNone = none.channels[.opcode] ?? []
        XCTAssertEqual(opcodeNone.count, OverlayTests.x86_64Code.count)
        XCTAssertTrue(opcodeNone.allSatisfy { $0 == 0 },
                      "OPCODE must be zero-filled with domain=.none")

        // After selecting x86-64 the OPCODE channel is populated.
        try pipe.setOverlayDomain(.x86_64)
        let x86 = try pipe.encodeBytesWithOverlays(
            OverlayTests.x86_64Code, channels: [.opcode])
        let opcodeX86 = x86.channels[.opcode] ?? []

        XCTAssertEqual(none.ids, x86.ids, "tokenization must be unchanged")
        XCTAssertNotEqual(opcodeNone, opcodeX86, "domain channel must differ from .none")
        XCTAssertTrue(opcodeX86.contains { $0 != 0 },
                      "OPCODE must be populated with domain=.x86_64")
    }

    func testSetOverlayDomainNoneRoundTrips() throws {
        let pipe = try Pipeline.byteId()
        defer { pipe.close() }
        try pipe.setOverlayDomain(.x86_64)
        try pipe.setOverlayDomain(.none)
        let res = try pipe.encodeBytesWithOverlays(
            OverlayTests.x86_64Code, channels: [.opcode])
        XCTAssertTrue((res.channels[.opcode] ?? []).allSatisfy { $0 == 0 })
    }
}
