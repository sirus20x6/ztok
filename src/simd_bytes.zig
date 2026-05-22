//! SIMD byte-class scanners for the pretokenizer hot loops.
//!
//! All cl100k / GPT-2 pretokenizer character classes degenerate to
//! ASCII byte tests on the common path (English / code corpora are
//! >95% ASCII). Each non-trivial pretok pattern boils down to "scan
//! forward while byte ∈ class C", which is a textbook SIMD problem:
//!   * load 16 bytes into a `@Vector(16, u8)`,
//!   * compare against the class predicate to get a 16-lane bool,
//!   * find the first lane that violates the predicate.
//!
//! When the SIMD chunk contains ANY non-ASCII byte (>= 0x80) we stop
//! at that byte and let the caller fall back to the scalar UTF-8
//! decoder for that one codepoint, then resume SIMD on the next ASCII
//! run. That's the right policy because:
//!   1. UTF-8 leading bytes start at 0xC0 and continuation bytes at
//!      0x80; both look like "not a letter / not a digit / not a
//!      whitespace" to the ASCII tests, so the SIMD answer for that
//!      lane would be wrong anyway,
//!   2. the boundary report stays simple — the SIMD scan never crosses
//!      into a multibyte sequence.
//!
//! Vector width is chosen per target via `std.simd.suggestVectorLength`:
//!   * AVX2 x86_64  → 32 lanes (one `ymm`, `vpcmpeqb`/`vpminub`),
//!   * AVX-512      → 64 lanes (one `zmm`),
//!   * SSE2 / NEON / wasm128 → 16 lanes (one `xmm` / `q`-reg / `v128`).
//! The scan loop runs a widest-width stage, then a 16-lane stage for the
//! 16..(V-1)-byte remainder (compiled out when V == 16), then a scalar
//! tail. NEON / SSE2 / wasm therefore keep exactly the prior 16-lane
//! behaviour; AVX2 hosts process twice the bytes per iteration. On
//! targets without SIMD the compiler emits a scalar unrolling — correct,
//! just no speed win.

const std = @import("std");

/// Widest byte-vector the target benefits from (lanes). 16 on
/// SSE2/NEON/wasm128, 32 on AVX2, 64 on AVX-512.
pub const V: usize = std.simd.suggestVectorLength(u8) orelse 16;

/// Smallest unsigned int that holds one bit per lane of an `L`-wide
/// byte vector (the packed lane-predicate bitmap). `@bitCast` of a
/// `@Vector(L, u1)` lands here; `@ctz`/all-ones tests run on it.
fn MaskInt(comptime L: usize) type {
    return std.meta.Int(.unsigned, L);
}

/// Pack an `L`-lane bool vector into its bitmap and return the index of
/// the first false lane, or null if every lane is true. O(1) via
/// `@ctz` (TZCNT / `i8x16.bitmask`+ctz / NEON shrn+fmov).
inline fn firstFalse(comptime L: usize, mask: @Vector(L, bool)) ?usize {
    const M = MaskInt(L);
    const bits: M = @bitCast(@as(@Vector(L, u1), @intFromBool(mask)));
    if (bits == std.math.maxInt(M)) return null;
    return @ctz(~bits);
}

// ----- Class predicates (inline, comptime-foldable) -----

/// True if `b` is an ASCII byte that the cl100k/GPT-2 regex treats as
/// whitespace. Non-ASCII bytes return false — the caller must handle
/// them via the Unicode slow path.
pub inline fn isAsciiWs(b: u8) bool {
    // 0x09..0x0D OR 0x20.
    return (b >= 0x09 and b <= 0x0D) or b == 0x20;
}

/// True if `b` is an ASCII letter (A-Z, a-z).
pub inline fn isAsciiLetter(b: u8) bool {
    return (b >= 'A' and b <= 'Z') or (b >= 'a' and b <= 'z');
}

/// True if `b` is an ASCII digit (0-9).
pub inline fn isAsciiDigit(b: u8) bool {
    return b >= '0' and b <= '9';
}

/// True if `b` is ASCII AND not whitespace AND not a letter AND not a
/// digit — i.e. the body class for GPT-2 pattern ` ?[^\s\p{L}\p{N}]+`
/// on ASCII bytes. Non-ASCII bytes return false so the caller bails to
/// the Unicode slow path at the first non-ASCII byte.
pub inline fn isAsciiPunct(b: u8) bool {
    if (b >= 0x80) return false;
    return !isAsciiWs(b) and !isAsciiLetter(b) and !isAsciiDigit(b);
}

/// True if `b` is ASCII whitespace AND not \r or \n. Used by cl100k
/// pattern 5 (`\s*[\r\n]+`) where the leading `\s*` is taken from this
/// class. Non-ASCII bytes return false.
pub inline fn isAsciiWsNotNl(b: u8) bool {
    return b == 0x20 or b == 0x09 or b == 0x0B or b == 0x0C;
}

// ----- Direct entry points (specialised, comptime-cls dispatch) -----

const Class = enum { ws, letter, digit, punct, ws_not_nl };

inline fn laneMatches(comptime cls: Class, comptime L: usize, vec: @Vector(L, u8)) @Vector(L, bool) {
    return switch (cls) {
        .ws => blk: {
            // (b >= 0x09 AND b <= 0x0D) OR b == 0x20.
            const lo: @Vector(L, u8) = @splat(0x09);
            const hi: @Vector(L, u8) = @splat(0x0D);
            const sp: @Vector(L, u8) = @splat(0x20);
            const in_lo_hi = (vec >= lo) & (vec <= hi);
            const is_sp = vec == sp;
            break :blk in_lo_hi | is_sp;
        },
        .letter => blk: {
            // (b >= 'A' AND b <= 'Z') OR (b >= 'a' AND b <= 'z').
            const a_lo: @Vector(L, u8) = @splat('A');
            const a_hi: @Vector(L, u8) = @splat('Z');
            const b_lo: @Vector(L, u8) = @splat('a');
            const b_hi: @Vector(L, u8) = @splat('z');
            break :blk ((vec >= a_lo) & (vec <= a_hi)) | ((vec >= b_lo) & (vec <= b_hi));
        },
        .digit => blk: {
            const lo: @Vector(L, u8) = @splat('0');
            const hi: @Vector(L, u8) = @splat('9');
            break :blk (vec >= lo) & (vec <= hi);
        },
        .punct => blk: {
            // ASCII (<0x80) AND NOT ws AND NOT letter AND NOT digit.
            const ascii: @Vector(L, u8) = @splat(0x80);
            const is_ascii = vec < ascii;
            const is_ws = laneMatches(.ws, L, vec);
            const is_letter = laneMatches(.letter, L, vec);
            const is_digit = laneMatches(.digit, L, vec);
            const non_punct = is_ws | is_letter | is_digit;
            break :blk is_ascii & ~non_punct;
        },
        .ws_not_nl => blk: {
            // ws AND NOT (\r or \n) — i.e. \t, \v, \f, ' '.
            const cr: @Vector(L, u8) = @splat('\r');
            const nl: @Vector(L, u8) = @splat('\n');
            const is_ws = laneMatches(.ws, L, vec);
            const is_nl = (vec == cr) | (vec == nl);
            break :blk is_ws & ~is_nl;
        },
    };
}

/// Scan one `L`-lane chunk for class `cls`; returns the offset of the
/// first byte NOT in the class, or null if all L lanes are in-class.
inline fn classViolation(comptime cls: Class, comptime L: usize, chunk: @Vector(L, u8)) ?usize {
    return firstFalse(L, laneMatches(cls, L, chunk));
}

/// Scan one `L`-lane chunk for the printable-ASCII range 0x21..0x7E;
/// returns the offset of the first out-of-range byte, or null if all L
/// lanes are printable.
inline fn printableViolation(comptime L: usize, chunk: @Vector(L, u8)) ?usize {
    const lo: @Vector(L, u8) = @splat(0x21);
    const hi: @Vector(L, u8) = @splat(0x7E);
    return firstFalse(L, (chunk >= lo) & (chunk <= hi));
}

/// Inline scanner specialised by class. The compiler fully inlines
/// `laneMatches(cls, ...)` so each call site sees the v128 / pminub
/// pattern directly. Returns the count of contiguous in-class bytes
/// starting from index 0.
// Set this to true via a temporary compile-time flag to force the
// scalar tail for A/B benchmarking. Production builds keep it false.
const FORCE_SCALAR: bool = false;

inline fn scanWhileDirect(comptime cls: Class, bytes: []const u8) usize {
    var i: usize = 0;
    if (comptime !FORCE_SCALAR) {
        // Widest-width stage: V lanes/iteration (32 on AVX2, 64 on
        // AVX-512, 16 elsewhere). On a fully in-class chunk advance by
        // V; otherwise `classViolation` pinpoints the first out-of-class
        // lane in O(1).
        while (i + V <= bytes.len) : (i += V) {
            const chunk: @Vector(V, u8) = bytes[i..][0..V].*;
            if (classViolation(cls, V, chunk)) |off| return i + off;
        }
        // 16-lane stage for the 16..(V-1)-byte remainder. Compiles out
        // entirely when V == 16 (NEON / SSE2 / wasm128).
        if (comptime V > 16) {
            while (i + 16 <= bytes.len) : (i += 16) {
                const chunk: @Vector(16, u8) = bytes[i..][0..16].*;
                if (classViolation(cls, 16, chunk)) |off| return i + off;
            }
        }
    }
    // Scalar tail.
    while (i < bytes.len) : (i += 1) {
        const matches = switch (cls) {
            .ws => isAsciiWs(bytes[i]),
            .letter => isAsciiLetter(bytes[i]),
            .digit => isAsciiDigit(bytes[i]),
            .punct => isAsciiPunct(bytes[i]),
            .ws_not_nl => isAsciiWsNotNl(bytes[i]),
        };
        if (!matches) return i;
    }
    return i;
}

pub fn scanAsciiWs(bytes: []const u8) usize {
    return scanWhileDirect(.ws, bytes);
}

pub fn scanAsciiLetter(bytes: []const u8) usize {
    return scanWhileDirect(.letter, bytes);
}

pub fn scanAsciiDigit(bytes: []const u8) usize {
    return scanWhileDirect(.digit, bytes);
}

pub fn scanAsciiPunct(bytes: []const u8) usize {
    return scanWhileDirect(.punct, bytes);
}

pub fn scanAsciiWsNotNl(bytes: []const u8) usize {
    return scanWhileDirect(.ws_not_nl, bytes);
}

// ===================== byte-level mapping SIMD =====================
//
// The HF GPT-2 byte_to_unicode table maps each input byte to a
// codepoint in U+0021..U+0143. Codepoints < 0x80 emit 1 UTF-8 byte;
// 0x80..0x7FF emit 2 UTF-8 bytes. The simple per-byte loop in
// byte_level.encodeBytes spends most of its time in std.unicode.utf8Encode
// branching on those thresholds.
//
// We SIMD-fast-path the dominant case: long ASCII runs (every input
// byte 0x21..0x7E is a printable that maps to ITSELF as a 1-byte
// UTF-8). For one 16-byte chunk that is entirely in 0x21..0x7E we
// store it through verbatim and advance by 16 input + 16 output
// bytes — no LUT, no branch.

/// Try to fast-copy a run of printable-ASCII bytes from `input` to
/// `out`. Returns the number of input bytes consumed (== number of
/// output bytes written). Stops at the first byte that is NOT in
/// 0x21..0x7E. The caller must handle that byte via the scalar
/// `byte_to_unicode` + utf8Encode path, then resume calling this
/// helper for the next ASCII run.
///
/// Why 0x21..0x7E specifically: those are the seeded printable ASCII
/// codepoints in the canonical GPT-2 byte_to_unicode mapping. For
/// every byte in that range, `byte_to_unicode[b] == @as(u21, b)` AND
/// the UTF-8 encoding is the single byte `b` itself. So we copy them
/// verbatim. Space (0x20) is NOT in this range (it maps to U+0120
/// 'Ġ' = 2 UTF-8 bytes); neither is anything 0x00..0x20 or 0x7F..0xFF.
pub fn copyPrintableAscii(input: []const u8, out: []u8) usize {
    std.debug.assert(out.len >= input.len);
    var i: usize = 0;

    if (comptime FORCE_SCALAR) {
        while (i < input.len) : (i += 1) {
            const b = input[i];
            if (b < 0x21 or b > 0x7E) return i;
            out[i] = b;
        }
        return i;
    }

    // Widest-width stage: copy V verbatim per iteration while in range.
    while (i + V <= input.len) {
        const chunk: @Vector(V, u8) = input[i..][0..V].*;
        if (printableViolation(V, chunk)) |off| {
            if (off > 0) @memcpy(out[i .. i + off], input[i .. i + off]);
            return i + off;
        }
        out[i..][0..V].* = chunk;
        i += V;
    }
    // 16-lane stage for the 16..(V-1)-byte remainder; compiled out when
    // V == 16.
    if (comptime V > 16) {
        while (i + 16 <= input.len) {
            const chunk: @Vector(16, u8) = input[i..][0..16].*;
            if (printableViolation(16, chunk)) |off| {
                if (off > 0) @memcpy(out[i .. i + off], input[i .. i + off]);
                return i + off;
            }
            out[i..][0..16].* = chunk;
            i += 16;
        }
    }

    // Scalar tail.
    while (i < input.len) : (i += 1) {
        const b = input[i];
        if (b < 0x21 or b > 0x7E) return i;
        out[i] = b;
    }
    return i;
}

// =================== tests ===================

const testing = std.testing;

test "scanAsciiLetter consumes a long ASCII run" {
    const s = "abcdefghijklmnopqrstuvwxyz1";
    const n = scanAsciiLetter(s);
    try testing.expectEqual(@as(usize, 26), n);
}

test "scanAsciiLetter stops at first non-letter" {
    const s = "abc def";
    const n = scanAsciiLetter(s);
    try testing.expectEqual(@as(usize, 3), n);
}

test "scanAsciiLetter handles short input (scalar-only tail)" {
    try testing.expectEqual(@as(usize, 0), scanAsciiLetter(""));
    try testing.expectEqual(@as(usize, 3), scanAsciiLetter("abc"));
    try testing.expectEqual(@as(usize, 3), scanAsciiLetter("abc!"));
}

test "scanAsciiLetter stops at high byte (non-ASCII)" {
    const s = "abc\xc3\xa9def"; // "abcédef" — \xc3 is the leading byte
    const n = scanAsciiLetter(s);
    try testing.expectEqual(@as(usize, 3), n);
}

test "scanAsciiDigit" {
    try testing.expectEqual(@as(usize, 5), scanAsciiDigit("12345"));
    try testing.expectEqual(@as(usize, 3), scanAsciiDigit("123abc"));
    try testing.expectEqual(@as(usize, 0), scanAsciiDigit("abc"));
    try testing.expectEqual(@as(usize, 0), scanAsciiDigit("\xc3\xa9123"));
}

test "scanAsciiWs eats spaces and tabs but not \\r\\n only" {
    try testing.expectEqual(@as(usize, 5), scanAsciiWs("     "));
    try testing.expectEqual(@as(usize, 4), scanAsciiWs(" \t\n\r"));
    try testing.expectEqual(@as(usize, 5), scanAsciiWs(" \t\n\r " ++ "x"));
    try testing.expectEqual(@as(usize, 0), scanAsciiWs("abc"));
}

test "scanAsciiPunct stops at letter, digit, ws, and non-ASCII" {
    try testing.expectEqual(@as(usize, 4), scanAsciiPunct("!@#$abc"));
    try testing.expectEqual(@as(usize, 4), scanAsciiPunct("!@#$ "));
    try testing.expectEqual(@as(usize, 4), scanAsciiPunct("!@#$1"));
    // Non-ASCII (e.g. UTF-8 leading byte \xc3): stops at it.
    try testing.expectEqual(@as(usize, 3), scanAsciiPunct("!@#\xc3\xa9"));
}

test "scanAsciiWsNotNl eats space/tab/vt/ff but not \\r\\n" {
    try testing.expectEqual(@as(usize, 4), scanAsciiWsNotNl(" \t\x0B\x0C\r"));
    try testing.expectEqual(@as(usize, 0), scanAsciiWsNotNl("\n hello"));
}

test "scanAsciiLetter random equivalence with scalar" {
    var prng = std.Random.DefaultPrng.init(0x12345);
    const rng = prng.random();
    var buf: [256]u8 = undefined;
    var trial: usize = 0;
    while (trial < 200) : (trial += 1) {
        const len = rng.uintLessThan(usize, 257);
        for (buf[0..len]) |*b| b.* = rng.int(u8);
        const got = scanAsciiLetter(buf[0..len]);
        var want: usize = 0;
        for (buf[0..len]) |b| {
            if (!isAsciiLetter(b)) break;
            want += 1;
        }
        try testing.expectEqual(want, got);
    }
}

test "scanAsciiPunct random equivalence with scalar" {
    var prng = std.Random.DefaultPrng.init(0x67890);
    const rng = prng.random();
    var buf: [256]u8 = undefined;
    var trial: usize = 0;
    while (trial < 200) : (trial += 1) {
        const len = rng.uintLessThan(usize, 257);
        for (buf[0..len]) |*b| b.* = rng.int(u8);
        const got = scanAsciiPunct(buf[0..len]);
        var want: usize = 0;
        for (buf[0..len]) |b| {
            if (!isAsciiPunct(b)) break;
            want += 1;
        }
        try testing.expectEqual(want, got);
    }
}

test "copyPrintableAscii copies long printable runs verbatim" {
    var out: [128]u8 = undefined;
    const in = "Hello,World!ThisIsATest~";
    const n = copyPrintableAscii(in, &out);
    try testing.expectEqual(in.len, n);
    try testing.expectEqualSlices(u8, in, out[0..n]);
}

test "copyPrintableAscii stops at space (0x20)" {
    var out: [64]u8 = undefined;
    const in = "Hello World";
    const n = copyPrintableAscii(in, &out);
    try testing.expectEqual(@as(usize, 5), n);
    try testing.expectEqualSlices(u8, "Hello", out[0..n]);
}

test "copyPrintableAscii stops at non-ASCII byte" {
    var out: [64]u8 = undefined;
    const in = "abc\xc3\xa9def";
    const n = copyPrintableAscii(in, &out);
    try testing.expectEqual(@as(usize, 3), n);
}

test "copyPrintableAscii handles boundary crossing chunks correctly" {
    var out: [64]u8 = undefined;
    // First 16 bytes are printable, then a space, then more printable.
    const in = "abcdefghijklmnop qrstuv";
    const n = copyPrintableAscii(in, &out);
    try testing.expectEqual(@as(usize, 16), n);
    try testing.expectEqualSlices(u8, "abcdefghijklmnop", out[0..n]);
}

test "copyPrintableAscii random equivalence" {
    var prng = std.Random.DefaultPrng.init(0xDEAD);
    const rng = prng.random();
    var buf: [256]u8 = undefined;
    var out: [256]u8 = undefined;
    var trial: usize = 0;
    while (trial < 200) : (trial += 1) {
        const len = rng.uintLessThan(usize, 257);
        for (buf[0..len]) |*b| b.* = rng.int(u8);
        const got = copyPrintableAscii(buf[0..len], &out);
        var want: usize = 0;
        for (buf[0..len]) |b| {
            if (b < 0x21 or b > 0x7E) break;
            want += 1;
        }
        try testing.expectEqual(want, got);
        // The copied prefix must match.
        try testing.expectEqualSlices(u8, buf[0..got], out[0..got]);
    }
}

// --- wide-path (AVX2/AVX-512) boundary coverage ----------------------
//
// These force the scan past a single V-wide chunk so the 16-lane mid
// stage and scalar tail are all exercised regardless of host width.
// Behaviour is width-independent; on a 16-lane host the "wide" stage IS
// the 16-lane stage and these still hold.

test "scanAsciiLetter wide-path boundary lengths" {
    inline for (.{ 16, 17, 31, 32, 33, 48, 64, 65, 128 }) |len| {
        var buf: [len]u8 = undefined;
        for (&buf) |*b| b.* = 'a';
        // Whole buffer in-class.
        try testing.expectEqual(@as(usize, len), scanAsciiLetter(&buf));
        // One violation at the very last byte stops there.
        buf[len - 1] = '0';
        try testing.expectEqual(@as(usize, len - 1), scanAsciiLetter(&buf));
        // One violation at the very first byte stops at 0.
        buf[len - 1] = 'a';
        buf[0] = ' ';
        try testing.expectEqual(@as(usize, 0), scanAsciiLetter(&buf));
    }
}

test "scanAsciiPunct wide-path stops at high byte across chunks" {
    // 40 punct bytes then a non-ASCII lead byte: must stop at 40 even on
    // a 16-lane host (third chunk) and on a 32-lane host (second chunk).
    var buf: [41]u8 = undefined;
    for (&buf) |*b| b.* = '!';
    buf[40] = 0xC3;
    try testing.expectEqual(@as(usize, 40), scanAsciiPunct(&buf));
}

test "copyPrintableAscii wide-path long run + mid-stop" {
    var in: [200]u8 = undefined;
    var out: [200]u8 = undefined;
    for (&in) |*b| b.* = '~'; // 0x7E, printable
    try testing.expectEqual(@as(usize, 200), copyPrintableAscii(&in, &out));
    try testing.expectEqualSlices(u8, in[0..], out[0..]);

    // A space at index 70 stops the copy there (spans wide + 16 + tail).
    in[70] = ' ';
    const n = copyPrintableAscii(&in, &out);
    try testing.expectEqual(@as(usize, 70), n);
    try testing.expectEqualSlices(u8, in[0..70], out[0..70]);
}
