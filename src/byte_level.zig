// HuggingFace / GPT-2 byte-level BPE mapping.
// Forward: raw byte -> printable Unicode codepoint (UTF-8).
// Reverse: printable codepoint -> raw byte (sparse).
// Non-mapped codepoints pass through verbatim (e.g. special tokens).

const std = @import("std");
const simd_bytes = @import("simd_bytes.zig");

// Reverse table size: highest mapped codepoint is 0x143 (188 high cps
// start at 0x100, the original 188 printable cps are < 0x100), so a
// flat [0x144]?u8 covers the full range and is small.
const REV_SIZE: usize = 0x144;

// Comptime-built forward table mirroring the canonical GPT-2 ctor:
//   bs = printable ASCII (0x21..0x7E) + Latin-1 supplement minus a few
//        (0xA1..0xAC, 0xAE..0xFF). Remaining 256-N bytes are appended
//        and mapped to 0x100, 0x101, ... in order.
pub const byte_to_unicode: [256]u21 = blk: {
    @setEvalBranchQuota(20000);
    var mapped: [256]bool = .{false} ** 256;
    var fwd: [256]u21 = .{0} ** 256;

    // Seed with the printable ranges.
    var seed_bytes: [256]u8 = undefined;
    var seed_cps: [256]u21 = undefined;
    var n_seed: usize = 0;

    {
        var b: u32 = '!';
        while (b <= '~') : (b += 1) {
            seed_bytes[n_seed] = @intCast(b);
            seed_cps[n_seed] = @intCast(b);
            n_seed += 1;
        }
    }
    {
        var b: u32 = 0xA1;
        while (b <= 0xAC) : (b += 1) {
            seed_bytes[n_seed] = @intCast(b);
            seed_cps[n_seed] = @intCast(b);
            n_seed += 1;
        }
    }
    {
        var b: u32 = 0xAE;
        while (b <= 0xFF) : (b += 1) {
            seed_bytes[n_seed] = @intCast(b);
            seed_cps[n_seed] = @intCast(b);
            n_seed += 1;
        }
    }

    // Apply seeds.
    var i: usize = 0;
    while (i < n_seed) : (i += 1) {
        fwd[seed_bytes[i]] = seed_cps[i];
        mapped[seed_bytes[i]] = true;
    }

    // Append leftover bytes, mapped to 0x100 + n.
    var extra: u21 = 0;
    var b: u32 = 0;
    while (b < 256) : (b += 1) {
        if (!mapped[b]) {
            fwd[b] = 0x100 + extra;
            extra += 1;
        }
    }

    break :blk fwd;
};

// Sparse reverse table: cp -> raw byte, or null.
const unicode_to_byte: [REV_SIZE]?u8 = blk: {
    @setEvalBranchQuota(20000);
    var rev: [REV_SIZE]?u8 = .{null} ** REV_SIZE;
    var b: u32 = 0;
    while (b < 256) : (b += 1) {
        const cp = byte_to_unicode[b];
        rev[cp] = @intCast(b);
    }
    break :blk rev;
};

// Apply forward mapping. Output written into `out`; returns the
// populated slice. out.len must be >= input.len * 2.
//
// SIMD fast-path: bytes in 0x21..0x7E map to themselves as a single
// UTF-8 byte (`byte_to_unicode[b] == b` and `b < 0x80`), so we copy
// runs of printable ASCII verbatim using `simd_bytes.copyPrintableAscii`
// (16-lane `v128.load` + range compare + `v128.store`). Any byte
// outside that range falls back to the scalar `utf8Encode` path,
// which handles the 2-byte mapped codepoints (0x80..0x7FF — covers
// all of U+0100..U+0142).
pub fn encodeBytes(input: []const u8, out: []u8) []u8 {
    std.debug.assert(out.len >= input.len * 2);
    var r: usize = 0;
    var w: usize = 0;
    while (r < input.len) {
        // SIMD: copy any leading printable-ASCII run (1 input byte ->
        // 1 output byte).
        const fast_n = simd_bytes.copyPrintableAscii(input[r..], out[w..]);
        r += fast_n;
        w += fast_n;
        if (r >= input.len) break;
        // Scalar: emit one mapped codepoint, then loop back into SIMD.
        const cp = byte_to_unicode[input[r]];
        const n = std.unicode.utf8Encode(cp, out[w..]) catch unreachable;
        w += n;
        r += 1;
    }
    return out[0..w];
}

// Reverse the forward mapping. Non-mapped codepoints are passed
// through as their raw UTF-8 bytes. Returns the populated slice.
// out.len must be >= input.len (output is always <= input.len since
// every multi-byte mapped codepoint collapses to a single byte, and
// pass-through is byte-for-byte).
pub fn decodeBytes(input: []const u8, out: []u8) []u8 {
    std.debug.assert(out.len >= input.len);
    var r: usize = 0;
    var w: usize = 0;
    while (r < input.len) {
        const b0 = input[r];
        const seq_len = std.unicode.utf8ByteSequenceLength(b0) catch {
            // Malformed byte: pass through raw.
            out[w] = b0;
            w += 1;
            r += 1;
            continue;
        };
        if (r + seq_len > input.len) {
            // Truncated sequence at tail: pass through raw.
            out[w] = b0;
            w += 1;
            r += 1;
            continue;
        }
        const cp = std.unicode.utf8Decode(input[r .. r + seq_len]) catch {
            out[w] = b0;
            w += 1;
            r += 1;
            continue;
        };
        if (cp < REV_SIZE) {
            if (unicode_to_byte[cp]) |raw| {
                out[w] = raw;
                w += 1;
                r += seq_len;
                continue;
            }
        }
        // Non-mapped codepoint: copy raw UTF-8 bytes through.
        var k: usize = 0;
        while (k < seq_len) : (k += 1) {
            out[w + k] = input[r + k];
        }
        w += seq_len;
        r += seq_len;
    }
    return out[0..w];
}

pub const SplitMode = enum { whitespace, none };

pub const Span = struct { start: u32, end: u32 };

pub const SplitResult = struct {
    mapped: []u8,
    spans: []Span,

    pub fn deinit(self: *SplitResult, allocator: std.mem.Allocator) void {
        allocator.free(self.mapped);
        allocator.free(self.spans);
    }
};

// HF byte_level pre-tokenizer.
// .whitespace mode: split on runs of Unicode whitespace; each non-empty
//   non-whitespace run becomes a span. If add_prefix_space is true the
//   leading run is prefixed with a U+0020 before mapping (so the first
//   span starts with 'Ġ'); subsequent spans naturally absorb the
//   preceding whitespace via the leading-space convention.
// .none mode: single span covers the whole mapped buffer.
//
// HF's actual behavior glues the leading whitespace of each token onto
// the front of that token (so " hello world" -> ["Ġhello", "Ġworld"]).
// We replicate that: each span includes one leading space iff (a)
// add_prefix_space is true and this is the first span, OR (b) any
// whitespace preceded this span in the original input.
pub fn splitAndMap(
    allocator: std.mem.Allocator,
    input: []const u8,
    mode: SplitMode,
    add_prefix_space: bool,
) !SplitResult {
    // Worst case mapped size: each input byte -> 2 UTF-8 bytes. Add 2
    // more for an optional injected prefix space (mapped 0x20 -> 2 bytes).
    var mapped = try allocator.alloc(u8, input.len * 2 + 2);
    errdefer allocator.free(mapped);

    var spans: std.ArrayList(Span) = .empty;
    errdefer spans.deinit(allocator);

    if (mode == .none) {
        const written = if (add_prefix_space and (input.len == 0 or input[0] != ' ')) blk: {
            var tmp = try allocator.alloc(u8, input.len + 1);
            defer allocator.free(tmp);
            tmp[0] = ' ';
            @memcpy(tmp[1..], input);
            break :blk encodeBytes(tmp, mapped);
        } else encodeBytes(input, mapped);
        try spans.append(allocator, .{ .start = 0, .end = @intCast(written.len) });
        const shrunk = try allocator.realloc(mapped, written.len);
        return .{ .mapped = shrunk, .spans = try spans.toOwnedSlice(allocator) };
    }

    // .whitespace mode. Walk the input codepoint-by-codepoint, grouping
    // whitespace and non-whitespace runs. For each non-ws span, prepend
    // a single space iff there was preceding whitespace OR (first span
    // and add_prefix_space).
    var i: usize = 0;
    var w: usize = 0;
    var saw_ws_before = add_prefix_space;
    var first_span = true;

    while (i < input.len) {
        // Skip whitespace.
        while (i < input.len) {
            const cp_info = nextCp(input, i);
            if (!isWhitespace(cp_info.cp)) break;
            saw_ws_before = true;
            i += cp_info.len;
        }
        if (i >= input.len) break;

        // Collect non-whitespace run.
        const run_start = i;
        while (i < input.len) {
            const cp_info = nextCp(input, i);
            if (isWhitespace(cp_info.cp)) break;
            i += cp_info.len;
        }
        const run = input[run_start..i];

        const span_start = w;
        const prefix = saw_ws_before or (first_span and add_prefix_space);
        // Drop the "first_span + add_prefix_space" branch if no whitespace
        // actually preceded — handled by saw_ws_before for non-first.
        _ = prefix;

        const inject = saw_ws_before or (first_span and add_prefix_space);
        if (inject) {
            const n = std.unicode.utf8Encode(byte_to_unicode[' '], mapped[w..]) catch unreachable;
            w += n;
        }
        // Map the run itself.
        const written = encodeBytes(run, mapped[w..]);
        w += written.len;

        try spans.append(allocator, .{
            .start = @intCast(span_start),
            .end = @intCast(w),
        });

        first_span = false;
        saw_ws_before = false;
    }

    const shrunk = try allocator.realloc(mapped, w);
    return .{ .mapped = shrunk, .spans = try spans.toOwnedSlice(allocator) };
}

// Decode concatenated mapped-form bytes back to raw bytes.
pub fn decodeMapped(allocator: std.mem.Allocator, mapped: []const u8) ![]u8 {
    const out = try allocator.alloc(u8, mapped.len);
    const written = decodeBytes(mapped, out);
    return allocator.realloc(out, written.len);
}

// --- helpers ---

const CpInfo = struct { cp: u21, len: usize };

fn nextCp(s: []const u8, i: usize) CpInfo {
    const b0 = s[i];
    const seq = std.unicode.utf8ByteSequenceLength(b0) catch return .{ .cp = b0, .len = 1 };
    if (i + seq > s.len) return .{ .cp = b0, .len = 1 };
    const cp = std.unicode.utf8Decode(s[i .. i + seq]) catch return .{ .cp = b0, .len = 1 };
    return .{ .cp = cp, .len = seq };
}

// Unicode whitespace: covers ASCII ws, U+0085 NEL, U+00A0 NBSP,
// U+1680, U+2000..U+200A, U+2028, U+2029, U+202F, U+205F, U+3000.
fn isWhitespace(cp: u21) bool {
    return switch (cp) {
        0x09...0x0D, 0x20, 0x85, 0xA0, 0x1680, 0x2000...0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000 => true,
        else => false,
    };
}

// =================== tests ===================

test "byte_to_unicode is the canonical GPT-2 table" {
    // Space (0x20) is not in the seeded printable ranges, so it's
    // assigned 0x100 + n. Per GPT-2 the value is 0x120 ('Ġ').
    try std.testing.expectEqual(@as(u21, 0x120), byte_to_unicode[0x20]);
    // A couple more well-known anchors.
    try std.testing.expectEqual(@as(u21, '!'), byte_to_unicode[0x21]);
    try std.testing.expectEqual(@as(u21, '~'), byte_to_unicode[0x7E]);
    // 0x00..0x20 are not in seeds; they get 0x100..0x120 in order.
    try std.testing.expectEqual(@as(u21, 0x100), byte_to_unicode[0x00]);
    try std.testing.expectEqual(@as(u21, 0x101), byte_to_unicode[0x01]);
    // Leftovers in scan order (0..255 not in seeded ranges):
    //   0x00..0x20 -> 0x100..0x120 (33 entries)
    //   0x7F..0xA0 -> 0x121..0x142 (34 entries)
    //   0xAD       -> 0x143
    try std.testing.expectEqual(@as(u21, 0x121), byte_to_unicode[0x7F]);
    try std.testing.expectEqual(@as(u21, 0x122), byte_to_unicode[0x80]);
    try std.testing.expectEqual(@as(u21, 0x142), byte_to_unicode[0xA0]);
    try std.testing.expectEqual(@as(u21, 0x143), byte_to_unicode[0xAD]);
}

test "byte_to_unicode is bijective on 0..255" {
    var buf_enc: [4]u8 = undefined;
    var buf_dec: [4]u8 = undefined;
    var b: u32 = 0;
    while (b < 256) : (b += 1) {
        const in: [1]u8 = .{@intCast(b)};
        const enc = encodeBytes(&in, &buf_enc);
        const dec = decodeBytes(enc, &buf_dec);
        try std.testing.expectEqual(@as(usize, 1), dec.len);
        try std.testing.expectEqual(@as(u8, @intCast(b)), dec[0]);
    }
}

test "encodeBytes round-trip" {
    const in = "hello world";
    var buf_enc: [64]u8 = undefined;
    var buf_dec: [64]u8 = undefined;
    const enc = encodeBytes(in, &buf_enc);
    const dec = decodeBytes(enc, &buf_dec);
    try std.testing.expectEqualStrings(in, dec);
}

test "encodeBytes maps space to Gdot" {
    const in = " hello";
    var buf_enc: [32]u8 = undefined;
    const enc = encodeBytes(in, &buf_enc);
    // UTF-8 of U+0120 'Ġ' is 0xC4 0xA0.
    try std.testing.expect(enc.len >= 2);
    try std.testing.expectEqual(@as(u8, 0xC4), enc[0]);
    try std.testing.expectEqual(@as(u8, 0xA0), enc[1]);
}

test "splitAndMap whitespace mode with prefix space" {
    const allocator = std.testing.allocator;
    var res = try splitAndMap(allocator, "hello world", .whitespace, true);
    defer res.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 2), res.spans.len);
    // Each span's first 2 bytes should be the UTF-8 of 'Ġ' (0xC4 0xA0).
    for (res.spans) |sp| {
        try std.testing.expect(sp.end - sp.start >= 2);
        try std.testing.expectEqual(@as(u8, 0xC4), res.mapped[sp.start]);
        try std.testing.expectEqual(@as(u8, 0xA0), res.mapped[sp.start + 1]);
    }
}

test "splitAndMap whitespace mode without prefix space" {
    const allocator = std.testing.allocator;
    var res = try splitAndMap(allocator, "hello world", .whitespace, false);
    defer res.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 2), res.spans.len);
    // First span: "hello" — no prefix.
    const s0 = res.mapped[res.spans[0].start..res.spans[0].end];
    try std.testing.expectEqualStrings("hello", s0);
    // Second span: "Ġworld".
    const s1 = res.mapped[res.spans[1].start..res.spans[1].end];
    try std.testing.expectEqual(@as(u8, 0xC4), s1[0]);
    try std.testing.expectEqual(@as(u8, 0xA0), s1[1]);
    try std.testing.expectEqualStrings("world", s1[2..]);
}

test "decodeMapped reverses splitAndMap" {
    const allocator = std.testing.allocator;
    const inputs = [_][]const u8{
        "hello world",
        "the quick brown fox",
        "a",
        "multi   spaces",
        "tab\there",
    };
    for (inputs) |in| {
        var res = try splitAndMap(allocator, in, .whitespace, true);
        defer res.deinit(allocator);
        const round = try decodeMapped(allocator, res.mapped);
        defer allocator.free(round);
        // With add_prefix_space=true and whitespace splitting, the
        // round-trip yields a space-normalized form: every run of
        // whitespace collapses to a single ' '. We assert that the
        // round-trip equals the canonical whitespace-collapsed form
        // prefixed with a single space.
        var expected: std.ArrayList(u8) = .empty;
        defer expected.deinit(allocator);
        try expected.append(allocator, ' ');
        var i: usize = 0;
        var in_ws = true; // we already wrote the leading space
        while (i < in.len) {
            const ci = nextCp(in, i);
            if (isWhitespace(ci.cp)) {
                if (!in_ws) {
                    try expected.append(allocator, ' ');
                    in_ws = true;
                }
            } else {
                var k: usize = 0;
                while (k < ci.len) : (k += 1) try expected.append(allocator, in[i + k]);
                in_ws = false;
            }
            i += ci.len;
        }
        // Strip any trailing space that the loop might have produced
        // for trailing whitespace in `in`.
        var end: usize = expected.items.len;
        while (end > 0 and expected.items[end - 1] == ' ') end -= 1;
        try std.testing.expectEqualStrings(expected.items[0..end], round);
    }
}

test "decodeMapped passes through non-mapped codepoints" {
    const allocator = std.testing.allocator;
    // A codepoint well outside the forward set, e.g. U+2603 SNOWMAN.
    // UTF-8: 0xE2 0x98 0x83.
    const in = "\xe2\x98\x83";
    const out = try decodeMapped(allocator, in);
    defer allocator.free(out);
    try std.testing.expectEqualStrings(in, out);
}
