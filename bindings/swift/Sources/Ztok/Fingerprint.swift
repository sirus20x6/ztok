// Fingerprint — 32-byte deterministic tokenizer fingerprint.
//
// Two pipelines that return the same fingerprint will emit
// bit-identical id streams for any input. Use as a cache key,
// KV-store discriminator, or training-pipeline guard.
//
// Stored as `Data`; the `.hexString` extension lazily formats the
// canonical lowercase hex representation (matching the Java binding's
// `Fingerprint.hex()` and the Python binding's `.hex()`).

import Foundation

/// 32-byte deterministic tokenizer fingerprint.
///
/// Equality is byte-wise; the hex representation is computed lazily
/// the first time `hexString` is accessed.
public struct Fingerprint: Hashable, Sendable, CustomStringConvertible {
    /// The fingerprint always occupies exactly 32 bytes (SHA-256
    /// digest size).
    public static let length = 32

    /// Raw 32-byte digest.
    public let bytes: Data

    /// Build a fingerprint from a 32-byte buffer. Traps on the wrong
    /// length — every libztok-provided fingerprint is 32 bytes by
    /// construction.
    public init(bytes: Data) {
        precondition(bytes.count == Fingerprint.length,
                     "Fingerprint must be exactly \(Fingerprint.length) bytes; got \(bytes.count)")
        self.bytes = bytes
    }

    /// Lowercase hex representation, e.g. `"a3f2..."`.
    public var hexString: String {
        // Hand-rolled hex formatter — Foundation has no first-party
        // hex API on Linux. Two characters per byte, lowercase.
        let table: [Character] = ["0", "1", "2", "3", "4", "5", "6", "7",
                                  "8", "9", "a", "b", "c", "d", "e", "f"]
        var out = String()
        out.reserveCapacity(bytes.count * 2)
        for b in bytes {
            out.append(table[Int(b >> 4)])
            out.append(table[Int(b & 0x0F)])
        }
        return out
    }

    public var description: String {
        return "Fingerprint(\(hexString))"
    }
}
