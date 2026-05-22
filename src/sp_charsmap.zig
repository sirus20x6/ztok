//! SentencePiece `precompiled_charsmap` parser + lookup.
//!
//! Each SP model can ship a `precompiled_charsmap` field on its
//! `NormalizerSpec` (proto field 2). It encodes a set of byte-sequence →
//! byte-sequence rewrite rules baked at training time (e.g. half-width
//! kana → full-width, Cyrillic homoglyph fixups, NMT-style cleanups).
//!
//! Blob layout (matches `Normalizer::EncodePrecompiledCharsMap` in
//! `refs/sentencepiece/src/normalizer.cc`):
//!
//!   ┌───────────────┬───────────────────────────┬──────────────────────┐
//!   │ trie_size  u32│ trie blob (i32 unit_size) │ normalized "\0" blob │
//!   │ little-endian │ trie_size bytes total     │ rest of the blob     │
//!   └───────────────┴───────────────────────────┴──────────────────────┘
//!
//! The trie blob is a Darts-clone double-array trie. Each unit is a
//! 32-bit packed value with the following bit layout (per `darts.h`):
//!   * bit 8       : has_leaf — child unit is a leaf carrying the value.
//!   * bit 9       : extension bit on `offset()` shift.
//!   * bits 0..=7  : node label byte (or, for leaf units, the low bits
//!                   carry the result value with bit 31 reserved).
//!   * bits 10..31 : offset / value (semantics depend on whether the
//!                   unit is a leaf or an internal node).
//!
//! The normalized blob is a sequence of replacement strings separated by
//! `'\0'` terminators. Each successful trie match yields a `value` which
//! is the byte offset into this blob where the replacement string starts
//! (run until the next NUL).
//!
//! Lookup is leftmost-longest: for each input position we walk the trie
//! and record the LAST leaf we hit (longest prefix). If we hit a leaf,
//! we emit the corresponding replacement; if not, we pass one UTF-8
//! codepoint of the input through unchanged. Replacement strings can be
//! empty (deletion).
//!
//! Validation lifted from the SP `DecodePrecompiledCharsMap` + `validate`
//! path:
//!   * Blob must be ≥ 4 bytes (for the header) plus a non-empty trie.
//!   * Trie size must be ≥ 1024 bytes and a multiple of 1024 (256-unit
//!     blocks).
//!   * Normalized blob must end with a `'\0'` terminator.
//!   * Every internal-unit offset must keep `(i ^ offset) | 0xFF` in
//!     range of the array length.
//!
//! Performance: the lookup loop is `O(match_len)` per starting position,
//! and we advance by at least one byte per outer iteration, so the full
//! `normalize` pass is `O(input_len × max_match_len)`. With realistic SP
//! charsmaps (T5: ~177 KB / 44k units, max match ~8 bytes) the constant
//! factor stays small. No quadratic blowups.

const std = @import("std");

const MIN_TRIE_BYTES: u32 = 1024;
const TRIE_BLOCK_BYTES: u32 = 1024; // 256 units × 4 bytes/unit
const UNIT_SIZE: u32 = 4;

pub const Error = error{
    BlobTooSmall,
    TrieSizeNotAligned,
    TrieSizeOverflow,
    NormalizedBlobUnterminated,
    NormalizedBlobEmpty,
    TrieValidationFailed,
} || std.mem.Allocator.Error;

/// A parsed SP precompiled charsmap. View over caller-supplied bytes —
/// owns no allocation of its own beyond the slices it borrows.
pub const PrecompiledCharsmap = struct {
    /// Darts trie units (each `i32` in source's signed view, used as
    /// `u32` for bit manipulation here). Borrowed slice.
    trie: []const u32,
    /// `'\0'`-delimited replacement-string blob. Borrowed slice.
    normalized: []const u8,
};

pub const Match = struct {
    /// Number of INPUT bytes the matched key spanned. Always > 0 when
    /// a match was found; lookup returns `null` otherwise.
    match_len: usize,
    /// Replacement bytes from `normalized`. May be empty (a "delete"
    /// rule) — callers must distinguish a null `Match` (no rule) from
    /// `Match{ match_len = N, replacement = "" }` (rule that deletes N
    /// input bytes).
    replacement: []const u8,
};

/// Parse a raw `precompiled_charsmap` blob into a `PrecompiledCharsmap`
/// view. The returned view borrows from `bytes` — the caller owns the
/// backing buffer and must keep it alive for as long as the view is in
/// use.
///
/// `bytes` is the proto field 2 of `NormalizerSpec` (see
/// `src/sp_model.zig` for the parsing layer). Returns an error on the
/// same conditions SP's reference decoder rejects:
///   * blob shorter than the 4-byte header,
///   * trie size header that overruns the blob,
///   * trie size < 1024 or not aligned to 1024 bytes,
///   * normalized blob missing the trailing NUL,
///   * any trie unit with an out-of-range offset.
pub fn parse(bytes: []const u8) Error!PrecompiledCharsmap {
    if (bytes.len <= @sizeOf(u32)) return Error.BlobTooSmall;

    const trie_blob_size = std.mem.readInt(u32, bytes[0..4], .little);

    if (trie_blob_size < MIN_TRIE_BYTES) return Error.TrieSizeOverflow;
    if ((trie_blob_size & (TRIE_BLOCK_BYTES - 1)) != 0) return Error.TrieSizeNotAligned;

    const header_bytes: usize = @sizeOf(u32);
    if (@as(usize, trie_blob_size) >= bytes.len) return Error.TrieSizeOverflow;

    const trie_end = header_bytes + @as(usize, trie_blob_size);
    if (trie_end > bytes.len) return Error.TrieSizeOverflow;

    // Trie units are i32-aligned within the (already 4-byte-aligned)
    // blob. We need an aligned view to reinterpret as []u32 without
    // copying. The slice we get from the proto reader is generally NOT
    // 4-byte aligned, so callers should pass through `parseAligned`
    // when they own the storage; this parse() requires the slice to be
    // 4-byte aligned at offset 4 (i.e. the bytes slice itself can be
    // unaligned but bytes[4..] must be 4-aligned).
    //
    // The common path is: `SpModel.precompiled_charsmap` is an owned
    // []u8 allocated via `allocator.dupe` — its base alignment is at
    // least 1, but Zig's default allocator returns alignment 8 or 16
    // pointers; the byte at offset 4 is therefore also 4-aligned.
    // For safety we route through `std.mem.bytesAsSlice` with
    // `@alignCast` after we've checked alignment.
    const trie_bytes = bytes[header_bytes..trie_end];
    if ((@intFromPtr(trie_bytes.ptr) % @alignOf(u32)) != 0) {
        return Error.TrieValidationFailed;
    }
    const aligned: []align(@alignOf(u32)) const u8 = @alignCast(trie_bytes);
    const trie: []const u32 = std.mem.bytesAsSlice(u32, aligned);

    const normalized = bytes[trie_end..];
    if (normalized.len == 0) return Error.NormalizedBlobEmpty;
    if (normalized[normalized.len - 1] != 0) return Error.NormalizedBlobUnterminated;

    // Sanity-check the trie array (mirrors SP's DoubleArray::validate):
    // every non-leaf unit's child offset must keep the next array index
    // in range. Leaf units (label > 0xFF) carry values, not offsets, so
    // skip them.
    for (trie, 0..) |unit, i| {
        if (unitLabel(unit) > 0xFF) continue;
        const offset = unitOffset(unit);
        if (offset == 0) continue;
        // SP uses `(i ^ offset) | 0xFF >= size`; reuse the same test.
        const base = i ^ offset;
        if ((base | 0xFF) >= trie.len) return Error.TrieValidationFailed;
    }

    return .{ .trie = trie, .normalized = normalized };
}

// ---- Unit accessors (mirror `Darts::DoubleArrayUnit`) ---------------------

inline fn unitHasLeaf(unit: u32) bool {
    return ((unit >> 8) & 1) == 1;
}

inline fn unitValue(unit: u32) u32 {
    return unit & ((@as(u32, 1) << 31) - 1);
}

inline fn unitLabel(unit: u32) u32 {
    // For non-leaf units, label is the low byte. For leaf units,
    // bit 31 sets the MSB so label() returns > 0xFF; this is the
    // same trick the C++ code uses to skip leaves in `validate()`.
    return unit & ((@as(u32, 1) << 31) | 0xFF);
}

inline fn unitOffset(unit: u32) u32 {
    // `offset = (unit >> 10) << ((unit >> 9) & 1) * 6`.
    // The extension bit at position 9 multiplies the shift by 64 (= <<6),
    // letting offsets exceed 22 bits when needed.
    const shift_bit: u5 = @intCast((unit >> 9) & 1);
    return (unit >> 10) << (shift_bit * @as(u5, 6));
}

// ---- Lookup ---------------------------------------------------------------

/// Look up the longest prefix of `input` that matches a rule in this
/// charsmap. Returns the matched length + replacement bytes; null if no
/// rule matches at the current position.
///
/// Mirrors `DoubleArray::commonPrefixSearch` followed by SP's longest-
/// match scan in `NormalizePrefix`. We don't materialise the full hit
/// list — we just keep the longest result seen.
pub fn lookup(self: PrecompiledCharsmap, input: []const u8) ?Match {
    if (self.trie.len == 0 or input.len == 0) return null;

    var node_pos: u32 = 0;
    var unit: u32 = self.trie[node_pos];
    node_pos ^= unitOffset(unit);

    var longest_len: usize = 0;
    var longest_value: u32 = 0;

    var i: usize = 0;
    while (i < input.len) : (i += 1) {
        node_pos ^= input[i];
        if (node_pos >= self.trie.len) return null;
        unit = self.trie[node_pos];
        if (unitLabel(unit) != input[i]) break;

        node_pos ^= unitOffset(unit);
        if (unitHasLeaf(unit)) {
            if (node_pos >= self.trie.len) break;
            const leaf = self.trie[node_pos];
            longest_len = i + 1;
            longest_value = unitValue(leaf);
        }
    }

    if (longest_len == 0) return null;
    if (longest_value >= self.normalized.len) return null;

    // Slice runs from longest_value up to the next NUL (exclusive).
    var end = longest_value;
    while (end < self.normalized.len and self.normalized[end] != 0) : (end += 1) {}
    return .{ .match_len = longest_len, .replacement = self.normalized[longest_value..end] };
}

// ---- Whole-string normalize ----------------------------------------------

const REPLACEMENT_CHAR = [_]u8{ 0xEF, 0xBF, 0xBD };

/// Apply the charsmap rules to `input`, returning a freshly-allocated
/// byte slice. The pass is leftmost-longest:
///
///   1. At position `i`, look up the longest matching rule.
///   2. If found, append the replacement and advance `i` by `match_len`.
///   3. Otherwise, consume one UTF-8 codepoint (or 1 byte on malformed
///      UTF-8 — emitting U+FFFD would change downstream encode behaviour
///      for vocabs without a U+FFFD token, so we pass the raw byte
///      through and let the encoder route via byte_fallback / unk).
///
/// The whole pass is `O(input_len × max_match_len)`. For T5's charsmap
/// the average is far closer to `O(input_len)` because most positions
/// either match a short rule (1–4 bytes) or fall through to the
/// codepoint advance.
pub fn normalize(
    allocator: std.mem.Allocator,
    self: PrecompiledCharsmap,
    input: []const u8,
) ![]u8 {
    if (input.len == 0) return allocator.alloc(u8, 0);

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    // Realistic charsmaps don't blow up input; +8 covers tiny inputs
    // where a single rule expansion can push past the literal length.
    try out.ensureTotalCapacity(allocator, input.len + 8);

    var i: usize = 0;
    while (i < input.len) {
        if (lookup(self, input[i..])) |m| {
            try out.appendSlice(allocator, m.replacement);
            i += m.match_len;
        } else {
            const lead = input[i];
            const cp_len_raw: u3 = std.unicode.utf8ByteSequenceLength(lead) catch 1;
            const cp_len: usize = cp_len_raw;
            const end = @min(i + cp_len, input.len);
            try out.appendSlice(allocator, input[i..end]);
            i = end;
        }
    }

    return out.toOwnedSlice(allocator);
}

/// Variant of `normalize` that also returns a per-output-byte map back
/// to the byte offset in `input` that produced each output byte. For
/// rules with N output bytes that consume M input bytes, all N output
/// bytes inherit the start offset of the matched input run — matches
/// the convention used by the rest of the SpNormalizer pipeline.
///
/// Caller frees both slices. Returns `error.OutOfMemory` on allocation
/// failure; never panics on malformed input.
pub const NormalizeResult = struct {
    bytes: []u8,
    origin: []u32,
};

pub fn normalizeWithOrigin(
    allocator: std.mem.Allocator,
    self: PrecompiledCharsmap,
    input: []const u8,
    /// Per-byte origin map of the INPUT slice (length == input.len).
    /// Each output byte's origin is looked up via this map at the
    /// position the matching rule consumed. Caller-supplied so the
    /// charsmap can compose with earlier normalizer stages (NFKC, etc.)
    /// that already maintain an origin map.
    input_origin: []const u32,
) !NormalizeResult {
    std.debug.assert(input_origin.len == input.len);
    if (input.len == 0) {
        return .{
            .bytes = try allocator.alloc(u8, 0),
            .origin = try allocator.alloc(u32, 0),
        };
    }

    var out_bytes: std.ArrayList(u8) = .empty;
    errdefer out_bytes.deinit(allocator);
    var out_origin: std.ArrayList(u32) = .empty;
    errdefer out_origin.deinit(allocator);
    try out_bytes.ensureTotalCapacity(allocator, input.len + 8);
    try out_origin.ensureTotalCapacity(allocator, input.len + 8);

    var i: usize = 0;
    while (i < input.len) {
        const anchor = input_origin[i];
        if (lookup(self, input[i..])) |m| {
            try out_bytes.appendSlice(allocator, m.replacement);
            var k: usize = 0;
            while (k < m.replacement.len) : (k += 1) {
                try out_origin.append(allocator, anchor);
            }
            i += m.match_len;
        } else {
            const lead = input[i];
            const cp_len_raw: u3 = std.unicode.utf8ByteSequenceLength(lead) catch 1;
            const cp_len: usize = cp_len_raw;
            const end = @min(i + cp_len, input.len);
            try out_bytes.appendSlice(allocator, input[i..end]);
            var k: usize = i;
            while (k < end) : (k += 1) {
                try out_origin.append(allocator, input_origin[k]);
            }
            i = end;
        }
    }

    return .{
        .bytes = try out_bytes.toOwnedSlice(allocator),
        .origin = try out_origin.toOwnedSlice(allocator),
    };
}

// ---- Test helpers --------------------------------------------------------

/// Build a synthetic 1-entry trie blob for unit testing. Encodes a
/// single 1-byte key with a value pointing at a normalized blob. The
/// trie is a minimal 256-unit (1024-byte) block: the root unit at
/// position 0 carries the offset, the labelled unit at the key byte
/// carries the leaf bit, and the leaf unit at `offset ^ value_pos`
/// carries the value.
///
/// This is a hand-built trie just sufficient for the lookup tests; it
/// does NOT exercise SP's `Darts::Builder` (which compacts many keys
/// across many blocks via the bit-9 extension shift). Real-world trie
/// blobs always pass through `Darts::Builder` upstream — we only need
/// to parse + walk them here.
fn buildSingleByteCharsmap(allocator: std.mem.Allocator, key: u8, replacement: []const u8) ![]u8 {
    const trie_units: u32 = 256;
    const trie_bytes: u32 = trie_units * UNIT_SIZE;

    // Trie walk for a 1-byte key:
    //   step 0: node_pos = 0; unit = trie[0]; node_pos ^= unit.offset();
    //   step 1: node_pos ^= key; unit = trie[node_pos]; check label;
    //           node_pos ^= unit.offset(); if has_leaf, value = trie[node_pos]
    //
    // Pick root_offset = 1 so after step 0 node_pos = 0 ^ 1 = 1. After
    // XORing the key byte (e.g. 'a' = 0x61): node_pos = 1 ^ 0x61 = 0x60.
    // We need trie[0x60] to be a labelled node carrying `key`. Set the
    // labelled unit's offset to 0x60 so it XORs back to position 0:
    // node_pos = 0x60 ^ 0x60 = 0. trie[0] is the root unit and its
    // label() returns 0 (since we set bits 0..7 = 0), so it can't be
    // re-used as a leaf. Put the leaf at position 2 instead: pick
    // labelled_offset = (0x60 ^ 2) = 0x62 so node_pos becomes
    // 0x60 ^ 0x62 = 2. trie[2] is the leaf — label MSB set, value = 0.
    //
    // For an arbitrary key byte we need labelled position = 1 ^ key
    // and leaf position = 2 (fixed). Labelled-node offset = (1 ^ key) ^ 2.
    const root_offset: u32 = 1;
    const labelled_pos: u32 = root_offset ^ @as(u32, key);
    const leaf_pos: u32 = 2;
    const labelled_offset: u32 = labelled_pos ^ leaf_pos;

    var trie: [256]u32 = @splat(0);
    trie[0] = root_offset << 10; // root: offset = 1, label = 0, bit 9 = 0
    trie[labelled_pos] = @as(u32, key) | (1 << 8) | (labelled_offset << 10);
    const value: u32 = 0;
    trie[leaf_pos] = (@as(u32, 1) << 31) | value;

    var bytes: std.ArrayList(u8) = .empty;
    errdefer bytes.deinit(allocator);

    var header: [4]u8 = undefined;
    std.mem.writeInt(u32, &header, trie_bytes, .little);
    try bytes.appendSlice(allocator, &header);

    // Append trie as raw bytes (little-endian on the host).
    const trie_raw: [*]const u8 = @ptrCast(&trie);
    try bytes.appendSlice(allocator, trie_raw[0 .. trie_units * UNIT_SIZE]);

    // Normalized blob: replacement + NUL.
    try bytes.appendSlice(allocator, replacement);
    try bytes.append(allocator, 0);

    return bytes.toOwnedSlice(allocator);
}

// ---- Tests ---------------------------------------------------------------

test "parse: rejects blob smaller than 4-byte header" {
    try std.testing.expectError(Error.BlobTooSmall, parse(""));
    try std.testing.expectError(Error.BlobTooSmall, parse("\x00\x00\x00\x00"));
}

test "parse: rejects trie size below 1024 minimum" {
    var buf: [16]u8 = @splat(0);
    std.mem.writeInt(u32, buf[0..4], 100, .little);
    try std.testing.expectError(Error.TrieSizeOverflow, parse(&buf));
}

test "parse: rejects trie size not divisible by 1024" {
    var buf: [2048]u8 = @splat(0);
    std.mem.writeInt(u32, buf[0..4], 1500, .little);
    try std.testing.expectError(Error.TrieSizeNotAligned, parse(&buf));
}

test "parse: rejects trie size that overruns the blob" {
    var buf: [200]u8 = @splat(0);
    std.mem.writeInt(u32, buf[0..4], 1024, .little); // header says 1024 but blob is 200
    try std.testing.expectError(Error.TrieSizeOverflow, parse(&buf));
}

test "parse: rejects normalized blob missing terminator" {
    const allocator = std.testing.allocator;
    var blob = try buildSingleByteCharsmap(allocator, 'a', "X");
    defer allocator.free(blob);
    // Strip the trailing NUL.
    blob[blob.len - 1] = 'Y';
    try std.testing.expectError(Error.NormalizedBlobUnterminated, parse(blob));
}

test "parse: single-byte synthetic charsmap parses to trie + normalized" {
    const allocator = std.testing.allocator;
    const blob = try buildSingleByteCharsmap(allocator, 'a', "Hi");
    defer allocator.free(blob);

    const cm = try parse(blob);
    try std.testing.expectEqual(@as(usize, 256), cm.trie.len);
    // normalized = "Hi\0"
    try std.testing.expectEqualStrings("Hi\x00", cm.normalized);
}

test "lookup: hits known single-byte key with the expected replacement" {
    const allocator = std.testing.allocator;
    const blob = try buildSingleByteCharsmap(allocator, 'a', "BB");
    defer allocator.free(blob);
    const cm = try parse(blob);

    const m = lookup(cm, "abc") orelse return error.TestExpectedMatch;
    try std.testing.expectEqual(@as(usize, 1), m.match_len);
    try std.testing.expectEqualStrings("BB", m.replacement);
}

test "lookup: returns null when no rule applies" {
    const allocator = std.testing.allocator;
    const blob = try buildSingleByteCharsmap(allocator, 'a', "BB");
    defer allocator.free(blob);
    const cm = try parse(blob);

    try std.testing.expect(lookup(cm, "xyz") == null);
    try std.testing.expect(lookup(cm, "") == null);
}

test "normalize: identity input (no matches) passes through unchanged" {
    const allocator = std.testing.allocator;
    const blob = try buildSingleByteCharsmap(allocator, 'q', "ZZ");
    defer allocator.free(blob);
    const cm = try parse(blob);

    const out = try normalize(allocator, cm, "hello world");
    defer allocator.free(out);
    try std.testing.expectEqualStrings("hello world", out);
}

test "normalize: input with multiple matches concatenates replacements" {
    const allocator = std.testing.allocator;
    // Rule: 'a' -> "AA". So "abracadabra" -> "AAbrAAcAAdAAbrAA".
    const blob = try buildSingleByteCharsmap(allocator, 'a', "AA");
    defer allocator.free(blob);
    const cm = try parse(blob);

    const out = try normalize(allocator, cm, "abracadabra");
    defer allocator.free(out);
    try std.testing.expectEqualStrings("AAbrAAcAAdAAbrAA", out);
}

test "normalize: empty rule deletes the matched input bytes" {
    const allocator = std.testing.allocator;
    // Rule: 'x' -> "" (delete).
    const blob = try buildSingleByteCharsmap(allocator, 'x', "");
    defer allocator.free(blob);
    const cm = try parse(blob);

    const out = try normalize(allocator, cm, "axbxcxd");
    defer allocator.free(out);
    try std.testing.expectEqualStrings("abcd", out);
}

test "normalizeWithOrigin: origin offsets anchor to start of matched run" {
    const allocator = std.testing.allocator;
    const blob = try buildSingleByteCharsmap(allocator, 'a', "AAA");
    defer allocator.free(blob);
    const cm = try parse(blob);

    const input = "ab";
    var input_origin: [2]u32 = .{ 0, 1 };
    const r = try normalizeWithOrigin(allocator, cm, input, &input_origin);
    defer allocator.free(r.bytes);
    defer allocator.free(r.origin);

    // Output is "AAAb" — first 3 bytes inherit origin 0 (the 'a'),
    // last byte inherits origin 1 (the 'b').
    try std.testing.expectEqualStrings("AAAb", r.bytes);
    try std.testing.expectEqualSlices(u32, &.{ 0, 0, 0, 1 }, r.origin);
}

test "normalize: real T5 charsmap loads and normalizes without crashing" {
    const allocator = std.testing.allocator;
    const path = "bench/vocabs/t5_unigram.model";
    const io = std.Io.Threaded.global_single_threaded.io();
    const data = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(data);

    const sp_model = @import("sp_model.zig");
    var sp = try sp_model.loadFromBytes(allocator, data);
    defer sp.deinit();

    const blob = sp.precompiled_charsmap orelse return error.SkipZigTest;
    const cm = try parse(blob);

    // Hit a few known inputs and make sure we don't crash. Real T5
    // charsmap normalizes "Hello world" mostly verbatim with the
    // dummy_prefix/escape passes running separately at a higher layer.
    const inputs = [_][]const u8{
        "Hello world",
        "café",
        "  multiple   spaces  ",
        // Full-width latin 'A' (U+FF21) — common SP NFKC-style target.
        "\xEF\xBC\xA1",
    };
    for (inputs) |s| {
        const out = try normalize(allocator, cm, s);
        defer allocator.free(out);
        // Just confirm we got something back (could be longer/shorter).
        try std.testing.expect(out.len > 0);
    }
}

test "real T5 charsmap rewrites U+FF21 -> 'A' (NFKC full-width fold)" {
    // The T5 nmt_nfkc charsmap includes the canonical full-width Latin
    // capital A (U+FF21, UTF-8 EF BC A1) -> ASCII 'A' mapping. This is
    // the headline rule users notice when they suspect the charsmap
    // pass is wired in. Confirms the lookup path emits the expected
    // single-byte replacement.
    const allocator = std.testing.allocator;
    const path = "bench/vocabs/t5_unigram.model";
    const io = std.Io.Threaded.global_single_threaded.io();
    const data = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(data);

    const sp_model = @import("sp_model.zig");
    var sp = try sp_model.loadFromBytes(allocator, data);
    defer sp.deinit();

    const blob = sp.precompiled_charsmap orelse return error.SkipZigTest;
    const cm = try parse(blob);

    // U+FF21 (full-width A). Either folds to "A" or to " A" — both are
    // valid NFKC-style rewrites depending on the trainer's prefix
    // handling. We just confirm the input doesn't pass through as-is
    // (the headline behaviour difference from the pre-charsmap code).
    const out = try normalize(allocator, cm, "\xEF\xBC\xA1");
    defer allocator.free(out);
    // Pre-charsmap behaviour would emit the 3 input bytes verbatim;
    // post-charsmap the trie expands them into "A" or " A".
    const passthrough = std.mem.eql(u8, out, "\xEF\xBC\xA1");
    try std.testing.expect(!passthrough);
    // The rewrite must end in 'A' regardless of the SP prefix space.
    try std.testing.expect(out[out.len - 1] == 'A');
}
