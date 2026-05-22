// SmokeTests — basic API surface checks. Mirrors the smoke coverage
// in the Python / Node / Ruby / Go / Rust / .NET / Java bindings:
//   - version probe
//   - byte_id round-trip (ASCII + UTF-8)
//   - empty input
//   - BPE round-trip via the synthetic .tiktoken fixture
//   - decodeBytes returns raw bytes
//   - invalid input maps to a typed error
//   - batch pool round-trip
//   - streaming sync (Sequence) round-trip
//   - streaming async (AsyncSequence) round-trip
//   - fingerprint determinism
//   - detect-format on a missing file returns .unknown
//   - close() is idempotent
//
// All cases skip via `try XCTSkipUnless(libztokAvailable)` when the
// shared library can't be loaded — keeps the test target green on
// dev machines without a built libztok.

import XCTest

@testable import Ztok

final class SmokeTests: XCTestCase {

    override func setUpWithError() throws {
        try XCTSkipUnless(TestSupport.libztokAvailable,
                          "libztok not loadable (set LD_LIBRARY_PATH or build with `zig build`)")
    }

    func testVersionIsNonEmpty() {
        let v = Pipeline.version()
        XCTAssertFalse(v.isEmpty, "version string must be non-empty")
        XCTAssertTrue(v.contains("."), "version should look like X.Y.Z; got \(v)")
    }

    func testByteIdRoundTripAscii() throws {
        let pipe = try Pipeline.byteId()
        defer { pipe.close() }
        let ids = try pipe.encode("hello world")
        XCTAssertEqual(ids.count, 11, "byte_id is 1:1 with input bytes")
        XCTAssertEqual(try pipe.decode(ids), "hello world")
    }

    func testByteIdRoundTripUtf8() throws {
        let pipe = try Pipeline.byteId()
        defer { pipe.close() }
        let text = "héllo 😀"
        let expectedBytes = Array(text.utf8)
        let ids = try pipe.encode(text)
        XCTAssertEqual(ids.count, expectedBytes.count)
        XCTAssertEqual(try pipe.decodeBytes(ids), expectedBytes)
        XCTAssertEqual(try pipe.decode(ids), text)
    }

    func testEmptyInputRoundTrips() throws {
        let pipe = try Pipeline.byteId()
        defer { pipe.close() }
        XCTAssertEqual(try pipe.encode(""), [])
        XCTAssertEqual(try pipe.decode([]), "")
    }

    func testBpeRoundTrip() throws {
        guard let fixture = TestSupport.tiktokenFixture() else {
            throw XCTSkip("could not build tiktoken fixture in /tmp")
        }
        let pipe = try Pipeline.fromTiktoken(path: fixture.path)
        defer { pipe.close() }
        let text = "hello world"
        let ids = try pipe.encode(text)
        XCTAssertFalse(ids.isEmpty)
        XCTAssertEqual(try pipe.decode(ids), text)
    }

    func testDecodeBytesPreservesRawBytes() throws {
        let pipe = try Pipeline.byteId()
        defer { pipe.close() }
        let ids = try pipe.encode("ab")
        XCTAssertEqual(try pipe.decodeBytes(ids), [0x61, 0x62])
    }

    func testInvalidNormalizerMapsToInvalidInput() throws {
        // We can't construct an invalid Normalizer through the safe
        // enum, so we drop down to the C ABI directly via CZtok.
        // This is the same pattern the Rust smoke suite uses.
        var cfg = ztok_pipeline_config(
            normalizer: ztok_normalizer_kind(rawValue: 999),
            pre_tokenizer: ZTOK_PRETOK_IDENTITY,
            model: ZTOK_MODEL_BYTE_ID,
            decoder: ZTOK_DECODER_CONCAT
        )
        var status: ztok_status = ZTOK_OK
        let h = withUnsafePointer(to: &cfg) { ptr in
            ztok_pipeline_new(ptr, &status)
        }
        XCTAssertNil(h, "bogus normalizer must not produce a handle")
        XCTAssertEqual(status, ZTOK_ERR_INVALID_INPUT)
    }

    func testBatchPoolRoundTrip() throws {
        let pipe = try Pipeline.byteId()
        defer { pipe.close() }
        let pool = try BatchPool(workers: 2)
        defer { pool.close() }
        XCTAssertGreaterThanOrEqual(pool.workerCount, 1,
                                    "auto-detect must resolve to >=1 worker")
        let results = try pool.encodeBatch(
            pipeline: pipe, inputs: ["foo", "bar", "baz", ""])
        XCTAssertEqual(results.count, 4)
        XCTAssertEqual(results[0], [0x66, 0x6f, 0x6f])
        XCTAssertEqual(results[1], [0x62, 0x61, 0x72])
        XCTAssertEqual(results[2], [0x62, 0x61, 0x7a])
        XCTAssertEqual(results[3], [])
    }

    func testStreamEncoderSyncRoundTrip() throws {
        let pipe = try Pipeline.byteId()
        defer { pipe.close() }
        let text = "hello world"
        var collected = [UInt32]()
        for chunk in try pipe.encodeStream(text: text, chunkSize: 4) {
            collected.append(contentsOf: chunk)
        }
        let expected: [UInt32] = Array(text.utf8).map { UInt32($0) }
        XCTAssertEqual(collected, expected)
        XCTAssertEqual(try pipe.decode(collected), text)
    }

    func testStreamEncoderSinkApiAndIdempotentFinish() throws {
        let pipe = try Pipeline.byteId()
        defer { pipe.close() }
        let enc = try StreamEncoder(pipeline: pipe)
        defer { enc.close() }
        var collected = [UInt32]()
        // Feed in 3-byte chunks.
        let bytes = Array("hello world".utf8)
        var i = 0
        while i < bytes.count {
            let end = min(i + 3, bytes.count)
            collected.append(contentsOf: try enc.feed(Array(bytes[i..<end])))
            i = end
        }
        collected.append(contentsOf: try enc.finish())
        // finish() is idempotent.
        XCTAssertEqual(try enc.finish(), [])
        XCTAssertEqual(try pipe.decode(collected), "hello world")
    }

    func testStreamEncoderAsyncRoundTrip() async throws {
        let pipe = try Pipeline.byteId()
        defer { pipe.close() }
        let text = "stream me async"
        var collected = [UInt32]()
        for try await id in try pipe.asyncEncodeStream(text: text, chunkSize: 4) {
            collected.append(id)
        }
        let expected: [UInt32] = Array(text.utf8).map { UInt32($0) }
        XCTAssertEqual(collected, expected)
    }

    func testFingerprintIsDeterministic() throws {
        let a = try Pipeline.byteId()
        defer { a.close() }
        let b = try Pipeline.byteId()
        defer { b.close() }
        let fa = try a.fingerprint()
        let fb = try b.fingerprint()
        XCTAssertEqual(fa, fb, "same config must yield same fingerprint")
        XCTAssertEqual(fa.bytes.count, Fingerprint.length)
        XCTAssertFalse(fa.bytes.allSatisfy { $0 == 0 },
                       "all-zero fingerprint would indicate a hash bug")
        // hex is 64 lowercase characters.
        XCTAssertEqual(fa.hexString.count, 64)
        XCTAssertTrue(fa.hexString.allSatisfy {
            ("0"..."9").contains($0) || ("a"..."f").contains($0)
        })
    }

    func testDetectFormatUnknownForMissingFile() {
        let fmt = Pipeline.detectFormat(path: "/definitely/does/not/exist.bin")
        XCTAssertEqual(fmt, .unknown)
    }

    func testCloseIsIdempotent() throws {
        let pipe = try Pipeline.byteId()
        XCTAssertFalse(pipe.isClosed)
        pipe.close()
        XCTAssertTrue(pipe.isClosed)
        pipe.close()  // second call is a no-op
        XCTAssertTrue(pipe.isClosed)
        XCTAssertThrowsError(try pipe.encode("x")) { error in
            guard case ZtokError.closed = error else {
                return XCTFail("expected .closed; got \(error)")
            }
        }
    }
}
