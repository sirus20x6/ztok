// HF ByteLevel pre-tokenization for GPT-2-family BPE.
// One pass: split with the canonical GPT-2 regex, then map each raw
// byte of each chunk through byte_to_unicode. Returns the mapped buffer
// plus spans over the mapped buffer.
//
// GPT-2 pattern (literal; case-sensitive contractions; unbounded digits):
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+

const std = @import("std");
const Span = @import("token.zig").Span;
const byte_level = @import("byte_level.zig");
const unicode_props = @import("unicode_props.zig");
const hf_regex = @import("hf_regex.zig");
const simd_bytes = @import("simd_bytes.zig");

pub const SplitMapResult = struct {
    mapped: []u8,
    spans: []Span,

    pub fn deinit(self: *SplitMapResult, allocator: std.mem.Allocator) void {
        allocator.free(self.mapped);
        allocator.free(self.spans);
    }
};

/// True if cutting `input` immediately before `pos` produces two halves
/// whose independently HF byte-level-pre-tokenized outputs concatenate to
/// the same span list as running the pre-tokenizer on `input` whole.
///
/// All GPT-2 alternatives are anchored prefix matchers with no lookbehind,
/// so the only divergence risk is a single-shot match that SPANS `pos`.
/// The patterns that can span a position are:
///   * whitespace eaters (e: `\s+(?!\S)`, f: `\s+`)
///   * `?\p{L}+`, ` ?\p{N}+`, ` ?[^\s\p{L}\p{N}]+` when a leading ASCII
///     space sits immediately before `pos` and the body starts at `pos`.
///
/// The byte→unicode remap is per-byte so it never reshapes across
/// regex-match boundaries: any safe regex boundary is also a safe cut at
/// the remap stage.
///
/// `pos` is a safe cut iff:
///   * `pos == 0` or `pos == input.len` (trivially), or
///   * the byte at `pos - 1` is `\n`, AND
///   * the byte at `pos` is NOT whitespace.
///
/// The "non-whitespace next" guard prevents cutting inside a whitespace
/// run that pattern e/f would absorb across the boundary in single-shot
/// mode (e.g. `\n` followed by spaces: single-shot pattern e would back
/// off and yield one match for `\n` + the spaces, but cutting between
/// them produces two separate ws spans).
pub fn isSafeCut(input: []const u8, pos: usize) bool {
    if (pos == 0 or pos == input.len) return true;
    if (pos > input.len) return false;
    if (input[pos - 1] != '\n') return false;
    const c = input[pos];
    // ASCII whitespace check first — covers \r, \t, ' ', \n, \v, \f.
    if (c == '\r' or c == '\n' or c == ' ' or c == '\t' or c == 0x0B or c == 0x0C) return false;
    // Non-ASCII: 0x80..0xBF is a continuation byte (mid-codepoint) — reject.
    // 0xC0..0xFF is a leading byte — decode and check Unicode whitespace.
    if (c >= 0x80) {
        if (c < 0xC0) return false;
        const d = decodeAt(input, pos) orelse return false;
        if (isWhitespace(d.cp)) return false;
    }
    return true;
}

/// Search for a safe HF byte-level cut position near `desired`. Scans
/// backwards first (up to `window` bytes) for the nearest safe position
/// at or before `desired`; if none is found, scans forward. Returns null
/// only if neither direction yields a safe cut within `window` bytes.
///
/// Mirrors the structure of `cl100k.findSafeCut`.
pub fn findSafeCut(input: []const u8, desired: usize, window: usize) ?usize {
    if (isSafeCut(input, desired)) return desired;

    var back: usize = 1;
    while (back <= window and back <= desired) : (back += 1) {
        const p = desired - back;
        if (isSafeCut(input, p)) return p;
    }

    var fwd: usize = 1;
    while (fwd <= window and desired + fwd <= input.len) : (fwd += 1) {
        const p = desired + fwd;
        if (isSafeCut(input, p)) return p;
    }

    return null;
}

pub fn splitAndMap(allocator: std.mem.Allocator, input: []const u8) !SplitMapResult {
    // Worst-case mapped: every input byte -> 2 UTF-8 bytes.
    // Worst-case spans: every input byte its own span.
    var mapped = try allocator.alloc(u8, input.len * 2);
    errdefer allocator.free(mapped);

    var spans = try allocator.alloc(Span, input.len);
    errdefer allocator.free(spans);

    var n_spans: usize = 0;
    var w: usize = 0;
    var i: usize = 0;

    while (i < input.len) {
        const took = matchOne(input, i);
        // matchOne always returns >= 1 on non-empty input because pattern d
        // (and patterns e/f for whitespace) cover every byte. Defensive guard.
        const len = if (took == 0) 1 else took;

        const span_start = w;
        // Map the raw bytes [i..i+len) through byte_to_unicode into mapped.
        // SIMD: fast-copy printable-ASCII runs verbatim (most pretok
        // chunks are pure ASCII words / numbers / leading-space-prefixed
        // words). Bytes outside 0x21..0x7E fall back to the scalar
        // utf8Encode for the 2-byte mapped codepoints.
        const written = byte_level.encodeBytes(input[i .. i + len], mapped[w..]);
        w += written.len;
        spans[n_spans] = .{ .start = @intCast(span_start), .end = @intCast(w) };
        n_spans += 1;
        i += len;
    }

    const mapped_shrunk = try allocator.realloc(mapped, w);
    const spans_shrunk = try allocator.realloc(spans, n_spans);
    return .{ .mapped = mapped_shrunk, .spans = spans_shrunk };
}

// Try the 7 GPT-2 alternatives in order at position `i`. Return raw match length.
fn matchOne(s: []const u8, i: usize) usize {
    if (matchContraction(s, i)) |len| return len;
    if (matchWord(s, i)) |len| return len;
    if (matchDigits(s, i)) |len| return len;
    if (matchPunct(s, i)) |len| return len;
    if (matchTrailingWs(s, i)) |len| return len;
    if (matchWs(s, i)) |len| return len;
    return 0;
}

// 's | 't | 're | 've | 'm | 'll | 'd  — literal lowercase, case-sensitive.
fn matchContraction(s: []const u8, i: usize) ?usize {
    if (i >= s.len or s[i] != '\'') return null;
    if (i + 1 >= s.len) return null;
    const c1 = s[i + 1];
    if (i + 2 < s.len) {
        const c2 = s[i + 2];
        if (c1 == 'r' and c2 == 'e') return 3;
        if (c1 == 'v' and c2 == 'e') return 3;
        if (c1 == 'l' and c2 == 'l') return 3;
    }
    if (c1 == 's' or c1 == 't' or c1 == 'm' or c1 == 'd') return 2;
    return null;
}

// ` ?\p{L}+` — optional ASCII space then 1+ Unicode letters.
fn matchWord(s: []const u8, i: usize) ?usize {
    var p = i;
    const had_space = p < s.len and s[p] == ' ';
    if (had_space) p += 1;
    const letters_start = p;
    // SIMD: vectorise the ASCII letter prefix, then fall back to scalar
    // for any non-ASCII codepoint and re-enter SIMD after.
    while (true) {
        p += simd_bytes.scanAsciiLetter(s[p..]);
        const d = decodeAt(s, p) orelse break;
        if (d.len == 1) break;
        if (!isLetter(d.cp)) break;
        p += d.len;
    }
    if (p == letters_start) return null;
    return p - i;
}

// ` ?\p{N}+` — optional ASCII space then 1+ Unicode numbers. Unbounded.
fn matchDigits(s: []const u8, i: usize) ?usize {
    var p = i;
    const had_space = p < s.len and s[p] == ' ';
    if (had_space) p += 1;
    const digits_start = p;
    while (true) {
        p += simd_bytes.scanAsciiDigit(s[p..]);
        const d = decodeAt(s, p) orelse break;
        if (d.len == 1) break;
        if (!isNumber(d.cp)) break;
        p += d.len;
    }
    if (p == digits_start) return null;
    return p - i;
}

// ` ?[^\s\p{L}\p{N}]+` — optional space then 1+ non-ws/letter/number.
fn matchPunct(s: []const u8, i: usize) ?usize {
    var p = i;
    const had_space = p < s.len and s[p] == ' ';
    if (had_space) p += 1;
    const body_start = p;
    while (true) {
        p += simd_bytes.scanAsciiPunct(s[p..]);
        const d = decodeAt(s, p) orelse break;
        if (d.len == 1) break;
        if (isWhitespace(d.cp) or isLetter(d.cp) or isNumber(d.cp)) break;
        p += d.len;
    }
    if (p == body_start) return null;
    return p - i;
}

// `\s+(?!\S)` — greedy ws run such that the char immediately after the match
// is whitespace or EOF. Standard regex backtracking semantics: take the
// longest prefix-of-the-ws-run that satisfies the lookahead.
fn matchTrailingWs(s: []const u8, i: usize) ?usize {
    var p = i;
    // Find end of the maximal whitespace run. SIMD-fast over ASCII ws;
    // scalar transitions cover any non-ASCII Unicode whitespace.
    while (true) {
        p += simd_bytes.scanAsciiWs(s[p..]);
        const d = decodeAt(s, p) orelse break;
        if (d.len == 1) break;
        if (!isWhitespace(d.cp)) break;
        p += d.len;
    }
    if (p == i) return null;
    // If the maximal run ends at EOF, the whole run matches.
    if (p == s.len) return p - i;
    // Otherwise the byte at p is the first non-ws byte. Back off by one
    // codepoint so that the char following the match is whitespace.
    // Decode the last codepoint of the run by scanning forward — but we
    // know the run consists of >=1 ws codepoint; find the start of the
    // final ws codepoint.
    var last_start: usize = i;
    var q = i;
    while (q < p) {
        const d = decodeAt(s, q) orelse break;
        last_start = q;
        q += d.len;
    }
    if (last_start == i) return null; // single-codepoint ws run; backing off makes it empty
    return last_start - i;
}

// `\s+` fallback.
fn matchWs(s: []const u8, i: usize) ?usize {
    var p = i;
    while (true) {
        p += simd_bytes.scanAsciiWs(s[p..]);
        const d = decodeAt(s, p) orelse break;
        if (d.len == 1) break;
        if (!isWhitespace(d.cp)) break;
        p += d.len;
    }
    if (p == i) return null;
    return p - i;
}

const Decoded = struct { cp: u21, len: usize };

fn decodeAt(s: []const u8, i: usize) ?Decoded {
    if (i >= s.len) return null;
    const b0 = s[i];
    if (b0 < 0x80) return .{ .cp = b0, .len = 1 };
    const seq_len = std.unicode.utf8ByteSequenceLength(b0) catch return .{ .cp = b0, .len = 1 };
    if (i + seq_len > s.len) return .{ .cp = b0, .len = 1 };
    const cp = std.unicode.utf8Decode(s[i .. i + seq_len]) catch return .{ .cp = b0, .len = 1 };
    return .{ .cp = cp, .len = seq_len };
}

fn isLetter(cp: u21) bool {
    if (cp < 0x80) return (cp >= 'A' and cp <= 'Z') or (cp >= 'a' and cp <= 'z');
    return unicode_props.isLetter(cp);
}

fn isNumber(cp: u21) bool {
    if (cp < 0x80) return cp >= '0' and cp <= '9';
    return unicode_props.isNumber(cp);
}

fn isWhitespace(cp: u21) bool {
    if (cp < 0x80) return cp == ' ' or cp == '\t' or cp == '\n' or cp == '\r' or cp == 0x0B or cp == 0x0C;
    return unicode_props.isWhitespace(cp);
}

// =================== chain executor ===================
//
// A `Chain` is an ordered list of pretokenizer operations parsed from a
// HF `pre_tokenizer.Sequence`. We execute them left-to-right on the
// input; each op maintains an implicit `(bytes, spans)` state. Some ops
// (Split, Digits, Punctuation, Whitespace, Metaspace, WhitespaceSplit)
// only subdivide spans; the `ByteLevel` op additionally re-maps bytes
// through `byte_to_unicode` (the resulting `mapped` buffer replaces
// `bytes` for all downstream ops).
//
// Lifetime: a `Chain` is built from a HF `tokenizer.json` at load time
// (`hf_json.parsePreTokenizer` populates `HFTokenizer.pretok_chain`) and
// the resulting bytecode + regex tables are stored on the
// `HFTokenizer.allocator`. A `PreTokenizer.chain` variant wraps a const
// pointer to a `Chain` value; encode-time runs the executor below.

pub const PretokOpKind = enum {
    split,
    byte_level,
    digits,
    punctuation,
    whitespace,
    whitespace_split,
    metaspace,
    bert,
};

pub const SplitBehavior = hf_regex.SplitBehavior;

pub const SplitOp = struct {
    /// Compiled regex pattern. Owned by the `Chain`'s allocator.
    re: *hf_regex.Regex,
    behavior: SplitBehavior,
    invert: bool,
};

pub const ByteLevelOp = struct {
    add_prefix_space: bool,
    /// Use the GPT-2 regex split before byte-mapping. If false, byte-map
    /// each input span as a single unit (no regex split).
    use_regex: bool,
    trim_offsets: bool = true,
};

pub const DigitsOp = struct {
    /// HF semantics: false = group digit runs into single spans; true = each
    /// digit becomes its own span.
    individual_digits: bool,
};

pub const MetaspaceOp = struct {
    replacement_cp: u21 = 0x2581, // ▁
    add_prefix_space: bool = true,
};

pub const PretokOp = union(PretokOpKind) {
    split: SplitOp,
    byte_level: ByteLevelOp,
    digits: DigitsOp,
    punctuation: SplitBehavior, // behavior controls grouping
    whitespace: void,
    whitespace_split: void,
    metaspace: MetaspaceOp,
    bert: void,
};

pub const Chain = struct {
    allocator: std.mem.Allocator,
    ops: []PretokOp,

    pub fn deinit(self: *Chain) void {
        for (self.ops) |op| {
            switch (op) {
                .split => |s| {
                    s.re.deinit();
                    self.allocator.destroy(s.re);
                },
                else => {},
            }
        }
        if (self.ops.len > 0) self.allocator.free(self.ops);
    }

    pub fn maxByteExpansion(self: *const Chain) usize {
        // Default 1; if any byte-level op is present, expand to 2.
        for (self.ops) |op| switch (op) {
            .byte_level => return 2,
            .metaspace => return 3, // ▁ is 3 UTF-8 bytes
            else => {},
        };
        return 1;
    }
};

/// Run the chain on `input`, returning the final `(bytes, spans)` pair.
/// Result is always allocator-owned (mapped buffer + spans slice). When
/// no op transformed bytes, the buffer is a verbatim copy of `input`.
pub fn runChain(
    allocator: std.mem.Allocator,
    chain: *const Chain,
    input: []const u8,
) !SplitMapResult {
    // State: a vector of (start, end) spans into `bytes`.
    var bytes_buf: std.ArrayList(u8) = .empty;
    errdefer bytes_buf.deinit(allocator);
    try bytes_buf.appendSlice(allocator, input);

    var spans: std.ArrayList(Span) = .empty;
    errdefer spans.deinit(allocator);
    try spans.append(allocator, .{ .start = 0, .end = @intCast(input.len) });

    for (chain.ops) |op| {
        try applyOp(allocator, op, &bytes_buf, &spans);
    }

    return .{
        .mapped = try bytes_buf.toOwnedSlice(allocator),
        .spans = try spans.toOwnedSlice(allocator),
    };
}

fn applyOp(
    allocator: std.mem.Allocator,
    op: PretokOp,
    bytes_buf: *std.ArrayList(u8),
    spans: *std.ArrayList(Span),
) !void {
    switch (op) {
        .split => |s| try applySplit(allocator, s, bytes_buf, spans),
        .byte_level => |b| try applyByteLevel(allocator, b, bytes_buf, spans),
        .digits => |d| try applyDigits(allocator, d, bytes_buf, spans),
        .punctuation => |behavior| try applyPunctuation(allocator, behavior, bytes_buf, spans),
        .whitespace => try applyWhitespace(allocator, bytes_buf, spans),
        .whitespace_split => try applyWhitespaceSplit(allocator, bytes_buf, spans),
        .metaspace => |m| try applyMetaspace(allocator, m, bytes_buf, spans),
        .bert => try applyBert(allocator, bytes_buf, spans),
    }
}

/// Apply a regex Split to every current span. The behavior controls
/// match-vs-gap grouping.
fn applySplit(
    allocator: std.mem.Allocator,
    op: SplitOp,
    bytes_buf: *std.ArrayList(u8),
    spans: *std.ArrayList(Span),
) !void {
    const bytes = bytes_buf.items;
    var out: std.ArrayList(Span) = .empty;
    errdefer out.deinit(allocator);

    for (spans.items) |sp| {
        const slice = bytes[sp.start..sp.end];
        const segs = try hf_regex.splitWith(allocator, op.re, slice, op.behavior);
        defer allocator.free(segs);
        if (op.invert) {
            // "invert: true" flips match vs non-match — emit only matches
            // when behavior would normally emit gaps, and vice versa.
            // For our use case (Falcon/Llama-3/Qwen2) invert is always
            // false; we emit all segments as spans regardless.
        }
        for (segs) |seg| {
            if (seg.end > seg.start) {
                try out.append(allocator, .{
                    .start = @intCast(sp.start + @as(u32, @intCast(seg.start))),
                    .end = @intCast(sp.start + @as(u32, @intCast(seg.end))),
                });
            }
        }
    }
    spans.deinit(allocator);
    spans.* = out;
}

/// Apply the HF ByteLevel transform: regex-split each span (if use_regex)
/// and byte_to_unicode-map every input byte. Optional `add_prefix_space`
/// inserts an ASCII space at the very front of the input before splitting.
fn applyByteLevel(
    allocator: std.mem.Allocator,
    op: ByteLevelOp,
    bytes_buf: *std.ArrayList(u8),
    spans: *std.ArrayList(Span),
) !void {
    const bytes = bytes_buf.items;
    var new_bytes: std.ArrayList(u8) = .empty;
    errdefer new_bytes.deinit(allocator);
    var new_spans: std.ArrayList(Span) = .empty;
    errdefer new_spans.deinit(allocator);

    var first = true;
    for (spans.items) |sp| {
        const slice = bytes[sp.start..sp.end];
        if (first and op.add_prefix_space and (slice.len == 0 or slice[0] != ' ')) {
            // Prepend a space byte to the FIRST span's mapped bytes.
            try mapAndEmit(allocator, &new_bytes, &new_spans, " ", op.use_regex);
        }
        first = false;
        try mapAndEmit(allocator, &new_bytes, &new_spans, slice, op.use_regex);
    }

    bytes_buf.deinit(allocator);
    bytes_buf.* = new_bytes;
    spans.deinit(allocator);
    spans.* = new_spans;
}

fn mapAndEmit(
    allocator: std.mem.Allocator,
    out_bytes: *std.ArrayList(u8),
    out_spans: *std.ArrayList(Span),
    slice: []const u8,
    use_regex: bool,
) !void {
    if (!use_regex) {
        // One span = the entire slice mapped through byte_to_unicode.
        const start = out_bytes.items.len;
        try mapBytes(allocator, out_bytes, slice);
        try out_spans.append(allocator, .{ .start = @intCast(start), .end = @intCast(out_bytes.items.len) });
        return;
    }
    // GPT-2 regex split + byte_to_unicode map: reuse the existing
    // hand-coded matcher.
    var i: usize = 0;
    while (i < slice.len) {
        const took = matchOne(slice, i);
        const len = if (took == 0) 1 else took;
        const start = out_bytes.items.len;
        try mapBytes(allocator, out_bytes, slice[i .. i + len]);
        try out_spans.append(allocator, .{ .start = @intCast(start), .end = @intCast(out_bytes.items.len) });
        i += len;
    }
}

fn mapBytes(allocator: std.mem.Allocator, out: *std.ArrayList(u8), slice: []const u8) !void {
    // Each byte expands to at most 2 UTF-8 bytes.
    try out.ensureUnusedCapacity(allocator, slice.len * 2);
    var r: usize = 0;
    var tmp: [4]u8 = undefined;
    while (r < slice.len) {
        // SIMD: copy printable-ASCII runs verbatim (1 byte -> 1 byte).
        // copyPrintableAscii needs an `out` slice big enough; ArrayList
        // capacity is already at least `slice.len * 2`, so we can write
        // straight into the items buffer past `.items.len`.
        const dst_start = out.items.len;
        const writable = out.capacity - dst_start;
        const fast_n = simd_bytes.copyPrintableAscii(
            slice[r..],
            out.allocatedSlice()[dst_start..][0..writable],
        );
        out.items.len += fast_n;
        r += fast_n;
        if (r >= slice.len) break;
        // Scalar: encode one mapped codepoint.
        const cp = byte_level.byte_to_unicode[slice[r]];
        const n = std.unicode.utf8Encode(cp, &tmp) catch unreachable;
        out.appendSliceAssumeCapacity(tmp[0..n]);
        r += 1;
    }
}

/// HF Digits: split each span on contiguous digit runs (or individual digits).
fn applyDigits(
    allocator: std.mem.Allocator,
    op: DigitsOp,
    bytes_buf: *std.ArrayList(u8),
    spans: *std.ArrayList(Span),
) !void {
    const bytes = bytes_buf.items;
    var out: std.ArrayList(Span) = .empty;
    errdefer out.deinit(allocator);

    for (spans.items) |sp| {
        const slice = bytes[sp.start..sp.end];
        var i: usize = 0;
        while (i < slice.len) {
            const d0 = decodeAt(slice, i) orelse break;
            const is_digit = isAsciiDigit(d0.cp);
            if (is_digit) {
                const seg_start = i;
                if (op.individual_digits) {
                    // Emit each digit as its own span.
                    try out.append(allocator, .{
                        .start = @intCast(sp.start + @as(u32, @intCast(seg_start))),
                        .end = @intCast(sp.start + @as(u32, @intCast(seg_start + d0.len))),
                    });
                    i += d0.len;
                } else {
                    // Emit the entire digit run as one span.
                    var j = i + d0.len;
                    while (j < slice.len) {
                        const dj = decodeAt(slice, j) orelse break;
                        if (!isAsciiDigit(dj.cp)) break;
                        j += dj.len;
                    }
                    try out.append(allocator, .{
                        .start = @intCast(sp.start + @as(u32, @intCast(seg_start))),
                        .end = @intCast(sp.start + @as(u32, @intCast(j))),
                    });
                    i = j;
                }
            } else {
                // Emit the non-digit run as one span.
                const seg_start = i;
                var j = i + d0.len;
                while (j < slice.len) {
                    const dj = decodeAt(slice, j) orelse break;
                    if (isAsciiDigit(dj.cp)) break;
                    j += dj.len;
                }
                try out.append(allocator, .{
                    .start = @intCast(sp.start + @as(u32, @intCast(seg_start))),
                    .end = @intCast(sp.start + @as(u32, @intCast(j))),
                });
                i = j;
            }
        }
    }
    spans.deinit(allocator);
    spans.* = out;
}

fn isAsciiDigit(cp: u21) bool {
    return cp >= '0' and cp <= '9';
}

/// HF Punctuation: split on punctuation characters. The HF behavior enum
/// controls how the matches are grouped — `Isolated` puts each punct in
/// its own span; `Contiguous` collapses consecutive puncts into one span;
/// `MergedWithPrevious` / `MergedWithNext` glue them to the adjacent text.
fn applyPunctuation(
    allocator: std.mem.Allocator,
    behavior: SplitBehavior,
    bytes_buf: *std.ArrayList(u8),
    spans: *std.ArrayList(Span),
) !void {
    const bytes = bytes_buf.items;
    var out: std.ArrayList(Span) = .empty;
    errdefer out.deinit(allocator);

    for (spans.items) |sp| {
        const slice = bytes[sp.start..sp.end];
        // Walk and emit alternating non-punct / punct spans, then post-
        // process per behavior.
        var i: usize = 0;
        var local: std.ArrayList(struct { start: usize, end: usize, is_punct: bool }) = .empty;
        defer local.deinit(allocator);

        while (i < slice.len) {
            const d0 = decodeAt(slice, i) orelse break;
            const is_punct = isPunct(d0.cp);
            const seg_start = i;
            var j = i + d0.len;
            while (j < slice.len) {
                const dj = decodeAt(slice, j) orelse break;
                if (isPunct(dj.cp) != is_punct) break;
                j += dj.len;
            }
            try local.append(allocator, .{ .start = seg_start, .end = j, .is_punct = is_punct });
            i = j;
        }

        // Behavior-driven emit.
        switch (behavior) {
            .Isolated, .Contiguous => {
                // Already coalesced consecutive runs above. Emit verbatim.
                for (local.items) |s| {
                    try out.append(allocator, .{
                        .start = @intCast(sp.start + @as(u32, @intCast(s.start))),
                        .end = @intCast(sp.start + @as(u32, @intCast(s.end))),
                    });
                }
            },
            .MergedWithPrevious => {
                var k: usize = 0;
                while (k < local.items.len) {
                    const s = local.items[k];
                    if (s.is_punct and k > 0) {
                        // Append onto the previous out segment.
                        if (out.items.len > 0) {
                            out.items[out.items.len - 1].end = @intCast(sp.start + @as(u32, @intCast(s.end)));
                        } else {
                            try out.append(allocator, .{
                                .start = @intCast(sp.start + @as(u32, @intCast(s.start))),
                                .end = @intCast(sp.start + @as(u32, @intCast(s.end))),
                            });
                        }
                    } else {
                        try out.append(allocator, .{
                            .start = @intCast(sp.start + @as(u32, @intCast(s.start))),
                            .end = @intCast(sp.start + @as(u32, @intCast(s.end))),
                        });
                    }
                    k += 1;
                }
            },
            .MergedWithNext => {
                // Buffer punct runs, glue to the next non-punct run.
                var k: usize = 0;
                var pending_start: ?usize = null;
                while (k < local.items.len) {
                    const s = local.items[k];
                    if (s.is_punct) {
                        if (pending_start == null) pending_start = s.start;
                    } else {
                        const start_ = pending_start orelse s.start;
                        try out.append(allocator, .{
                            .start = @intCast(sp.start + @as(u32, @intCast(start_))),
                            .end = @intCast(sp.start + @as(u32, @intCast(s.end))),
                        });
                        pending_start = null;
                    }
                    k += 1;
                }
                if (pending_start) |start_| {
                    try out.append(allocator, .{
                        .start = @intCast(sp.start + @as(u32, @intCast(start_))),
                        .end = sp.end,
                    });
                }
            },
            .Removed => {
                for (local.items) |s| {
                    if (!s.is_punct) {
                        try out.append(allocator, .{
                            .start = @intCast(sp.start + @as(u32, @intCast(s.start))),
                            .end = @intCast(sp.start + @as(u32, @intCast(s.end))),
                        });
                    }
                }
            },
        }
    }
    spans.deinit(allocator);
    spans.* = out;
}

fn isPunct(cp: u21) bool {
    if (cp < 0x80) {
        // `char::is_ascii_punctuation`: 0x21-0x2F, 0x3A-0x40, 0x5B-0x60, 0x7B-0x7E.
        return (cp >= '!' and cp <= '/') or
            (cp >= ':' and cp <= '@') or
            (cp >= '[' and cp <= '`') or
            (cp >= '{' and cp <= '~');
    }
    // HF `unicode_categories::is_punctuation` — General_Category P*
    // (Pc | Pd | Pe | Pf | Pi | Po | Ps). The previous block-range
    // approximation also matched Cf (e.g. U+2068/U+2069 bidi isolates),
    // Zs (U+2028/U+2029), Sm/So (most of 0x3000-0x303F), and Lo
    // (fullwidth letters in 0xFF00-0xFFEF), which corrupted the Falcon
    // pretok chain on bidi-isolated Arabic/Hebrew snippets in
    // unicode_stress.txt (-80/1000 lines) and a handful of code snippets.
    return unicode_props.isPunct(cp);
}

/// HF Whitespace pretok: split on `\w+` boundaries (group word chars,
/// emit punct as individual spans, drop whitespace). We approximate via
/// "split into runs of: letter/digit, punctuation, whitespace; drop
/// whitespace; emit alphas and individual puncts".
fn applyWhitespace(
    allocator: std.mem.Allocator,
    bytes_buf: *std.ArrayList(u8),
    spans: *std.ArrayList(Span),
) !void {
    const bytes = bytes_buf.items;
    var out: std.ArrayList(Span) = .empty;
    errdefer out.deinit(allocator);

    for (spans.items) |sp| {
        const slice = bytes[sp.start..sp.end];
        var i: usize = 0;
        while (i < slice.len) {
            const d0 = decodeAt(slice, i) orelse break;
            if (isWhitespace(d0.cp)) {
                i += d0.len;
                continue;
            }
            if (isLetter(d0.cp) or isAsciiDigit(d0.cp)) {
                const seg_start = i;
                var j = i + d0.len;
                while (j < slice.len) {
                    const dj = decodeAt(slice, j) orelse break;
                    if (!isLetter(dj.cp) and !isAsciiDigit(dj.cp)) break;
                    j += dj.len;
                }
                try out.append(allocator, .{
                    .start = @intCast(sp.start + @as(u32, @intCast(seg_start))),
                    .end = @intCast(sp.start + @as(u32, @intCast(j))),
                });
                i = j;
            } else {
                // Single punct codepoint.
                try out.append(allocator, .{
                    .start = @intCast(sp.start + @as(u32, @intCast(i))),
                    .end = @intCast(sp.start + @as(u32, @intCast(i + d0.len))),
                });
                i += d0.len;
            }
        }
    }
    spans.deinit(allocator);
    spans.* = out;
}

/// HF WhitespaceSplit: split on whitespace, drop the whitespace runs.
fn applyWhitespaceSplit(
    allocator: std.mem.Allocator,
    bytes_buf: *std.ArrayList(u8),
    spans: *std.ArrayList(Span),
) !void {
    const bytes = bytes_buf.items;
    var out: std.ArrayList(Span) = .empty;
    errdefer out.deinit(allocator);

    for (spans.items) |sp| {
        const slice = bytes[sp.start..sp.end];
        var i: usize = 0;
        while (i < slice.len) {
            const d0 = decodeAt(slice, i) orelse break;
            if (isWhitespace(d0.cp)) {
                i += d0.len;
                continue;
            }
            const seg_start = i;
            var j = i + d0.len;
            while (j < slice.len) {
                const dj = decodeAt(slice, j) orelse break;
                if (isWhitespace(dj.cp)) break;
                j += dj.len;
            }
            try out.append(allocator, .{
                .start = @intCast(sp.start + @as(u32, @intCast(seg_start))),
                .end = @intCast(sp.start + @as(u32, @intCast(j))),
            });
            i = j;
        }
    }
    spans.deinit(allocator);
    spans.* = out;
}

/// HF Metaspace: replace each space with the replacement codepoint
/// (typically U+2581 ▁), optionally prepending one at the start.
fn applyMetaspace(
    allocator: std.mem.Allocator,
    op: MetaspaceOp,
    bytes_buf: *std.ArrayList(u8),
    spans: *std.ArrayList(Span),
) !void {
    const bytes = bytes_buf.items;
    var new_bytes: std.ArrayList(u8) = .empty;
    errdefer new_bytes.deinit(allocator);
    var new_spans: std.ArrayList(Span) = .empty;
    errdefer new_spans.deinit(allocator);

    // Encode replacement once.
    var rep_buf: [4]u8 = undefined;
    const rep_len = std.unicode.utf8Encode(op.replacement_cp, &rep_buf) catch unreachable;
    const rep = rep_buf[0..rep_len];

    var first = true;
    for (spans.items) |sp| {
        const slice = bytes[sp.start..sp.end];
        const new_start = new_bytes.items.len;
        if (first and op.add_prefix_space) {
            try new_bytes.appendSlice(allocator, rep);
        }
        first = false;
        var i: usize = 0;
        while (i < slice.len) {
            if (slice[i] == ' ') {
                try new_bytes.appendSlice(allocator, rep);
                i += 1;
            } else {
                try new_bytes.append(allocator, slice[i]);
                i += 1;
            }
        }
        try new_spans.append(allocator, .{
            .start = @intCast(new_start),
            .end = @intCast(new_bytes.items.len),
        });
    }
    bytes_buf.deinit(allocator);
    bytes_buf.* = new_bytes;
    spans.deinit(allocator);
    spans.* = new_spans;
}

/// HF Bert pretok: split on whitespace + punctuation; each word + each
/// punct char becomes its own span.
fn applyBert(
    allocator: std.mem.Allocator,
    bytes_buf: *std.ArrayList(u8),
    spans: *std.ArrayList(Span),
) !void {
    try applyWhitespace(allocator, bytes_buf, spans);
}

// =================== tests ===================

test "hello world maps to two spans, second starts with Gdot" {
    const allocator = std.testing.allocator;
    var res = try splitAndMap(allocator, "hello world");
    defer res.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 2), res.spans.len);
    // First span: "hello" (no leading space mapping; 'h' is ASCII -> 1 byte).
    const s0 = res.mapped[res.spans[0].start..res.spans[0].end];
    try std.testing.expectEqualStrings("hello", s0);
    // Second span: " world" mapped -> "Ġworld". UTF-8 of U+0120 = 0xC4 0xA0.
    const s1 = res.mapped[res.spans[1].start..res.spans[1].end];
    try std.testing.expectEqual(@as(u8, 0xC4), s1[0]);
    try std.testing.expectEqual(@as(u8, 0xA0), s1[1]);
    try std.testing.expectEqualStrings("world", s1[2..]);
}

test "contractions split case-sensitive" {
    const allocator = std.testing.allocator;
    var res = try splitAndMap(allocator, "It's IT'S");
    defer res.deinit(allocator);
    // Expected raw splits: ["It", "'s", " IT", "'", "S"].
    // Decode each mapped span and assert.
    const expected = [_][]const u8{ "It", "'s", " IT", "'", "S" };
    try std.testing.expectEqual(expected.len, res.spans.len);
    var decode_buf: [32]u8 = undefined;
    for (res.spans, expected) |sp, want| {
        const piece = res.mapped[sp.start..sp.end];
        const decoded = byte_level.decodeBytes(piece, &decode_buf);
        try std.testing.expectEqualStrings(want, decoded);
    }
}

test "digits are unbounded (unlike cl100k 3-cap)" {
    const allocator = std.testing.allocator;
    var res = try splitAndMap(allocator, "12345");
    defer res.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 1), res.spans.len);
    const s0 = res.mapped[res.spans[0].start..res.spans[0].end];
    try std.testing.expectEqualStrings("12345", s0);
}

test "non-ASCII letters group via real p L" {
    const allocator = std.testing.allocator;
    var res = try splitAndMap(allocator, "café");
    defer res.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 1), res.spans.len);
    // Round-trip the mapped bytes back to raw via decodeBytes.
    var decode_buf: [32]u8 = undefined;
    const piece = res.mapped[res.spans[0].start..res.spans[0].end];
    const decoded = byte_level.decodeBytes(piece, &decode_buf);
    try std.testing.expectEqualStrings("café", decoded);
}

test "round-trip via decodeBytes recovers input" {
    const allocator = std.testing.allocator;
    const inputs = [_][]const u8{
        "hello world",
        "It's IT'S",
        "12345",
        "café",
        "  \n  word",
        "hi   ",
        "the quick brown fox jumps over 3 lazy dogs",
    };
    var decode_buf: [256]u8 = undefined;
    for (inputs) |in| {
        var res = try splitAndMap(allocator, in);
        defer res.deinit(allocator);
        // Concatenate all mapped spans (they are already contiguous in
        // `res.mapped` because we wrote them sequentially; the spans cover
        // [0..res.mapped.len) contiguously).
        const concat = res.mapped;
        const decoded = byte_level.decodeBytes(concat, &decode_buf);
        try std.testing.expectEqualStrings(in, decoded);
    }
}

test "trailing whitespace at EOF lands in one span" {
    const allocator = std.testing.allocator;
    var res = try splitAndMap(allocator, "hi   ");
    defer res.deinit(allocator);
    // "hi" then "   " (3 ws to EOF -> matched by \s+(?!\S) whole).
    try std.testing.expectEqual(@as(usize, 2), res.spans.len);
    var decode_buf: [32]u8 = undefined;
    const s0 = byte_level.decodeBytes(res.mapped[res.spans[0].start..res.spans[0].end], &decode_buf);
    try std.testing.expectEqualStrings("hi", s0);
    const s1 = byte_level.decodeBytes(res.mapped[res.spans[1].start..res.spans[1].end], &decode_buf);
    try std.testing.expectEqualStrings("   ", s1);
}

test "empty input yields zero spans" {
    const allocator = std.testing.allocator;
    var res = try splitAndMap(allocator, "");
    defer res.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 0), res.spans.len);
    try std.testing.expectEqual(@as(usize, 0), res.mapped.len);
}

test "isSafeCut true at end of input and at zero" {
    const s = "hello world";
    try std.testing.expect(isSafeCut(s, 0));
    try std.testing.expect(isSafeCut(s, s.len));
}

test "isSafeCut false mid-word" {
    const s = "hello world";
    // pos 3 is mid `hello` — prev is `l`, not `\n`.
    try std.testing.expect(!isSafeCut(s, 3));
    // pos 5 is between `o` and ` ` — prev `o` not `\n`.
    try std.testing.expect(!isSafeCut(s, 5));
    // pos 6 is between ` ` and `w` — prev ` ` not `\n` — unsafe even though
    // next is a letter (single-shot ` ?\p{L}+` would absorb the space).
    try std.testing.expect(!isSafeCut(s, 6));
}

test "isSafeCut true after newline when next char is non-whitespace" {
    const s = "foo\nbar\nbaz";
    // pos 4 = after first `\n`, before `b`. Safe.
    try std.testing.expect(isSafeCut(s, 4));
    // pos 8 = after second `\n`, before `b`. Safe.
    try std.testing.expect(isSafeCut(s, 8));
}

test "isSafeCut false after newline when next char is whitespace" {
    // Pattern e/f could absorb the trailing whitespace across the boundary.
    const s = "foo\n bar";
    try std.testing.expect(!isSafeCut(s, 4));
    const s2 = "foo\n\nbar";
    // pos 4: prev `\n`, next `\n` — unsafe.
    try std.testing.expect(!isSafeCut(s2, 4));
    // pos 5: prev `\n`, next `b` — safe.
    try std.testing.expect(isSafeCut(s2, 5));
}

test "isSafeCut handles UTF-8 continuation bytes after newline" {
    // "a\nĉ" — `ĉ` is U+0109 = 0xC4 0x89 (two bytes).
    const s = "a\n\xC4\x89";
    // pos 2 = after `\n`, before `\xC4` (leading byte). Safe.
    try std.testing.expect(isSafeCut(s, 2));
    // pos 3 = inside the multibyte sequence. Unsafe.
    try std.testing.expect(!isSafeCut(s, 3));
}

test "findSafeCut snaps to nearest safe cut within window" {
    const s = "abc\ndef\nghi";
    // Desired position 6 (mid-`def`). Backwards search hits pos 4 (after first `\n`).
    try std.testing.expectEqual(@as(?usize, 4), findSafeCut(s, 6, 16));
    // Desired position 0 is always safe.
    try std.testing.expectEqual(@as(?usize, 0), findSafeCut(s, 0, 16));
    // Tiny window may force forward scan.
    try std.testing.expectEqual(@as(?usize, 4), findSafeCut(s, 5, 1));
}

test "findSafeCut returns null when no safe boundary in window" {
    const s = "abcdefghij";
    // No newlines at all → only safe cuts are 0 and len. With desired=5 and
    // window=2, neither is reachable.
    try std.testing.expectEqual(@as(?usize, null), findSafeCut(s, 5, 2));
}

// =================== chain executor tests ===================

test "chain: Falcon-style Digits + Split([0-9][0-9][0-9])" {
    const a = std.testing.allocator;

    const re_ptr = try a.create(hf_regex.Regex);
    re_ptr.* = try hf_regex.compile(a, "[0-9][0-9][0-9]");
    const ops = try a.alloc(PretokOp, 2);
    ops[0] = .{ .digits = .{ .individual_digits = false } };
    ops[1] = .{ .split = .{ .re = re_ptr, .behavior = .Isolated, .invert = false } };
    var chain: Chain = .{ .allocator = a, .ops = ops };
    defer chain.deinit();

    const res = try runChain(a, &chain, "abc12345def");
    defer {
        a.free(res.mapped);
        a.free(res.spans);
    }
    // Expected: ["abc", "123", "45", "def"] (Digits split into "12345",
    // then the [0-9][0-9][0-9] split takes the first 3 of "12345" as a
    // separate span, leaving "45").
    const spans = res.spans;
    try std.testing.expect(spans.len >= 4);
    try std.testing.expectEqualStrings("abc", res.mapped[spans[0].start..spans[0].end]);
    try std.testing.expectEqualStrings("123", res.mapped[spans[1].start..spans[1].end]);
    try std.testing.expectEqualStrings("45", res.mapped[spans[2].start..spans[2].end]);
    try std.testing.expectEqualStrings("def", res.mapped[spans[3].start..spans[3].end]);
}

test "chain: Llama-3-style Split + ByteLevel(use_regex=false)" {
    const a = std.testing.allocator;

    // Simplified Llama-3 pretok: just \p{L}+ split + ByteLevel.
    const re_ptr = try a.create(hf_regex.Regex);
    re_ptr.* = try hf_regex.compile(a, "\\p{L}+|\\p{N}{1,3}|\\s+");
    const ops = try a.alloc(PretokOp, 2);
    ops[0] = .{ .split = .{ .re = re_ptr, .behavior = .Isolated, .invert = false } };
    ops[1] = .{ .byte_level = .{ .add_prefix_space = false, .use_regex = false } };
    var chain: Chain = .{ .allocator = a, .ops = ops };
    defer chain.deinit();

    const res = try runChain(a, &chain, "Hello 12345");
    defer {
        a.free(res.mapped);
        a.free(res.spans);
    }
    // Expected after Split: ["Hello", " ", "123", "45"].
    // After ByteLevel: each span byte-mapped; " " becomes "Ġ" (U+0120).
    try std.testing.expect(res.spans.len >= 4);
    try std.testing.expectEqualStrings("Hello", res.mapped[res.spans[0].start..res.spans[0].end]);
    // The second span is " " → "Ġ" (0xC4 0xA0).
    const s1 = res.mapped[res.spans[1].start..res.spans[1].end];
    try std.testing.expectEqual(@as(u8, 0xC4), s1[0]);
    try std.testing.expectEqual(@as(u8, 0xA0), s1[1]);
    try std.testing.expectEqualStrings("123", res.mapped[res.spans[2].start..res.spans[2].end]);
    try std.testing.expectEqualStrings("45", res.mapped[res.spans[3].start..res.spans[3].end]);
}

test "chain: ByteLevel(use_regex=true) matches the hand-coded GPT-2 path" {
    const a = std.testing.allocator;

    const ops = try a.alloc(PretokOp, 1);
    ops[0] = .{ .byte_level = .{ .add_prefix_space = false, .use_regex = true } };
    var chain: Chain = .{ .allocator = a, .ops = ops };
    defer chain.deinit();

    const res = try runChain(a, &chain, "hello world");
    defer {
        a.free(res.mapped);
        a.free(res.spans);
    }
    // Same as the splitAndMap test: ["hello", "Ġworld"].
    try std.testing.expectEqual(@as(usize, 2), res.spans.len);
    try std.testing.expectEqualStrings("hello", res.mapped[res.spans[0].start..res.spans[0].end]);
    const s1 = res.mapped[res.spans[1].start..res.spans[1].end];
    try std.testing.expectEqual(@as(u8, 0xC4), s1[0]);
    try std.testing.expectEqual(@as(u8, 0xA0), s1[1]);
    try std.testing.expectEqualStrings("world", s1[2..]);
}

test "chain: Punctuation Contiguous coalesces adjacent puncts" {
    const a = std.testing.allocator;

    const ops = try a.alloc(PretokOp, 1);
    ops[0] = .{ .punctuation = .Contiguous };
    var chain: Chain = .{ .allocator = a, .ops = ops };
    defer chain.deinit();

    const res = try runChain(a, &chain, "hello!?? world");
    defer {
        a.free(res.mapped);
        a.free(res.spans);
    }
    // Expected: ["hello", "!??", " world"] — consecutive ASCII puncts
    // glue into one span; the space between `??` and `world` is not
    // a punct so it's part of " world".
    try std.testing.expectEqual(@as(usize, 3), res.spans.len);
    try std.testing.expectEqualStrings("hello", res.mapped[res.spans[0].start..res.spans[0].end]);
    try std.testing.expectEqualStrings("!??", res.mapped[res.spans[1].start..res.spans[1].end]);
    try std.testing.expectEqualStrings(" world", res.mapped[res.spans[2].start..res.spans[2].end]);
}

test "chain: Metaspace replaces spaces with U+2581" {
    const a = std.testing.allocator;

    const ops = try a.alloc(PretokOp, 1);
    ops[0] = .{ .metaspace = .{ .replacement_cp = 0x2581, .add_prefix_space = true } };
    var chain: Chain = .{ .allocator = a, .ops = ops };
    defer chain.deinit();

    const res = try runChain(a, &chain, "hi world");
    defer {
        a.free(res.mapped);
        a.free(res.spans);
    }
    // U+2581 = 0xE2 0x96 0x81 (3 bytes). add_prefix_space=true prepends.
    // Result: "▁hi▁world" = 0xE2 0x96 0x81 'h' 'i' 0xE2 0x96 0x81 'w' 'o' 'r' 'l' 'd'.
    const want = "\xE2\x96\x81hi\xE2\x96\x81world";
    try std.testing.expectEqualStrings(want, res.mapped);
}

test "chain: Llama-3 verbatim regex digit groups are 1-3 not greedy" {
    // Regression for the multilingual stress sweep: the Llama-3
    // Split pattern carries `\p{N}{1,3}`, so a numeric run of 5 must
    // split into "123" + "45" — not one greedy "12345" span (which
    // is what `\p{N}+` would produce).
    const a = std.testing.allocator;

    // Verbatim pattern from bench/vocabs/llama3.json `pre_tokenizer.Sequence[0]`.
    const pat = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    const re_ptr = try a.create(hf_regex.Regex);
    re_ptr.* = try hf_regex.compile(a, pat);
    const ops = try a.alloc(PretokOp, 2);
    ops[0] = .{ .split = .{ .re = re_ptr, .behavior = .Isolated, .invert = false } };
    ops[1] = .{ .byte_level = .{ .add_prefix_space = false, .use_regex = false } };
    var chain: Chain = .{ .allocator = a, .ops = ops };
    defer chain.deinit();

    // A multilingual line with a trailing `n=907` numeric tag mirrors the
    // bench/corpora/multilingual.txt format that previously diverged from
    // HF on the 1.22 stress sweep. We just assert that the digit span "907"
    // appears as ONE span (length 3 still matches {1,3}) and is not glued
    // onto neighboring characters.
    const input = "Hello (n=907) and 12345 then 9876";
    const res = try runChain(a, &chain, input);
    defer {
        a.free(res.mapped);
        a.free(res.spans);
    }

    // Scan the spans for "907", "123", "45", "987", and "6".
    var saw_907 = false;
    var saw_123 = false;
    var saw_45 = false;
    var saw_987 = false;
    var saw_6 = false;
    for (res.spans) |sp| {
        const s = res.mapped[sp.start..sp.end];
        if (std.mem.eql(u8, s, "907")) saw_907 = true;
        if (std.mem.eql(u8, s, "123")) saw_123 = true;
        if (std.mem.eql(u8, s, "45")) saw_45 = true;
        if (std.mem.eql(u8, s, "987")) saw_987 = true;
        if (std.mem.eql(u8, s, "6")) saw_6 = true;
        // And critically, the greedy variant "12345" / "9876" must NEVER
        // appear — that's the `\p{N}+` bug.
        try std.testing.expect(!std.mem.eql(u8, s, "12345"));
        try std.testing.expect(!std.mem.eql(u8, s, "9876"));
    }
    try std.testing.expect(saw_907);
    try std.testing.expect(saw_123);
    try std.testing.expect(saw_45);
    try std.testing.expect(saw_987);
    try std.testing.expect(saw_6);
}

test "chain: empty input yields zero spans" {
    const a = std.testing.allocator;

    const ops = try a.alloc(PretokOp, 1);
    ops[0] = .{ .byte_level = .{ .add_prefix_space = false, .use_regex = true } };
    var chain: Chain = .{ .allocator = a, .ops = ops };
    defer chain.deinit();

    const res = try runChain(a, &chain, "");
    defer {
        a.free(res.mapped);
        a.free(res.spans);
    }
    try std.testing.expectEqual(@as(usize, 0), res.spans.len);
}

test "isPunct: bidi format characters U+2068/U+2069 are NOT punctuation" {
    // Regression for the Falcon-7B unicode_stress sweep: U+2068 (FIRST
    // STRONG ISOLATE) and U+2069 (POP DIRECTIONAL ISOLATE) are
    // General_Category Cf, not P*. The previous block-range
    // approximation `0x2000-0x206F` swept them up, causing the
    // Punctuation(Contiguous) pretok to split a leading-space run apart
    // from the bidi isolate that followed it, which then corrupted
    // every downstream ByteLevel BPE merge for the isolated Arabic /
    // Hebrew snippet. Other format / space chars in the same block
    // (U+200B ZWSP, U+200E LRM, U+202A LRE, U+2028 LINE SEP) are
    // covered by the same fix.
    try std.testing.expect(!isPunct(0x2068)); // FIRST STRONG ISOLATE (Cf)
    try std.testing.expect(!isPunct(0x2069)); // POP DIRECTIONAL ISOLATE (Cf)
    try std.testing.expect(!isPunct(0x2066)); // LEFT-TO-RIGHT ISOLATE (Cf)
    try std.testing.expect(!isPunct(0x2067)); // RIGHT-TO-LEFT ISOLATE (Cf)
    try std.testing.expect(!isPunct(0x200B)); // ZERO WIDTH SPACE (Cf)
    try std.testing.expect(!isPunct(0x200E)); // LEFT-TO-RIGHT MARK (Cf)
    try std.testing.expect(!isPunct(0x202A)); // LRE (Cf)
    try std.testing.expect(!isPunct(0x2028)); // LINE SEPARATOR (Zl)
    try std.testing.expect(!isPunct(0x2029)); // PARAGRAPH SEPARATOR (Zp)
    try std.testing.expect(!isPunct(0x2000)); // EN QUAD (Zs)
    // Sm/So in 0x3000-0x303F that the old range mismatched.
    try std.testing.expect(!isPunct(0x3000)); // IDEOGRAPHIC SPACE (Zs)
    try std.testing.expect(!isPunct(0x3012)); // POSTAL MARK (So)
    // Real P* characters that must still match.
    try std.testing.expect(isPunct(0x2010)); // HYPHEN (Pd)
    try std.testing.expect(isPunct(0x2014)); // EM DASH (Pd)
    try std.testing.expect(isPunct(0x2026)); // HORIZONTAL ELLIPSIS (Po)
    try std.testing.expect(isPunct(0x3001)); // IDEOGRAPHIC COMMA (Po)
    try std.testing.expect(isPunct(0x3008)); // LEFT ANGLE BRACKET (Ps)
    // ASCII branch unaffected.
    try std.testing.expect(isPunct('!'));
    try std.testing.expect(isPunct('?'));
    try std.testing.expect(!isPunct(' '));
    try std.testing.expect(!isPunct('a'));
}

test "chain: Falcon Punctuation+ByteLevel keeps bidi-isolate fused with leading space" {
    // Verbatim Falcon-7B pre_tokenizer prefix: Punctuation(Contiguous)
    // then ByteLevel(use_regex=true). On the exact line that diverged
    // from HF's reference (`Welcome to the project ⁨...`), the space
    // before U+2068 must stay glued to the bytes of U+2068 in a single
    // pretok span so the BPE merges see `Ġâ` as one piece (HF tok id
    // 2723) instead of `Ġ` + `âģ¨` (ztok's old 204 + 40005, 113).
    const a = std.testing.allocator;

    const ops = try a.alloc(PretokOp, 2);
    ops[0] = .{ .punctuation = .Contiguous };
    ops[1] = .{ .byte_level = .{ .add_prefix_space = false, .use_regex = true } };
    var chain: Chain = .{ .allocator = a, .ops = ops };
    defer chain.deinit();

    // " \u{2068}x" — single space, bidi isolate, ASCII letter.
    const input = " \xE2\x81\xA8x";
    const res = try runChain(a, &chain, input);
    defer {
        a.free(res.mapped);
        a.free(res.spans);
    }

    // Expected ByteLevel output for the leading-space + isolate run is
    // `Ġâģ¨` (U+0120 then the 3 mapped bytes of U+2068). The trailing
    // ASCII letter `x` is a separate ` ?\p{L}+` span... but there's no
    // leading space here, so just `x`. Total: 2 spans.
    //   span 0: " \u{2068}" mapped to "Ġâģ¨"  (U+0120 U+00E2 U+0123 U+00A8)
    //   span 1: "x" mapped to "x"
    try std.testing.expectEqual(@as(usize, 2), res.spans.len);
    const s0 = res.mapped[res.spans[0].start..res.spans[0].end];
    const s1 = res.mapped[res.spans[1].start..res.spans[1].end];
    // "Ġâģ¨" UTF-8: 0xC4 0xA0 0xC3 0xA2 0xC4 0xA3 0xC2 0xA8
    try std.testing.expectEqualStrings("\xC4\xA0\xC3\xA2\xC4\xA3\xC2\xA8", s0);
    try std.testing.expectEqualStrings("x", s1);
}

test "chain: Falcon Punctuation+ByteLevel splits on real Unicode punct" {
    // Positive control: characters that ARE P* (Pd hyphen, Po em dash)
    // must still create span boundaries under Punctuation(Contiguous),
    // matching HF's `unicode_categories::is_punctuation` exactly.
    const a = std.testing.allocator;

    const ops = try a.alloc(PretokOp, 2);
    ops[0] = .{ .punctuation = .Contiguous };
    ops[1] = .{ .byte_level = .{ .add_prefix_space = false, .use_regex = true } };
    var chain: Chain = .{ .allocator = a, .ops = ops };
    defer chain.deinit();

    // "ab\u{2014}cd" — em dash (U+2014, Pd) between two letter runs.
    // Punctuation pass yields ["ab", "\u{2014}", "cd"]; ByteLevel then
    // maps each span. The em dash itself becomes 3 mapped codepoints
    // (â\u{0122}\u{0136}), but the boundary between "ab" and the dash
    // must be present (would NOT be present if isPunct returned false
    // for U+2014).
    const input = "ab\xE2\x80\x94cd";
    const res = try runChain(a, &chain, input);
    defer {
        a.free(res.mapped);
        a.free(res.spans);
    }
    try std.testing.expectEqual(@as(usize, 3), res.spans.len);
    try std.testing.expectEqualStrings("ab", res.mapped[res.spans[0].start..res.spans[0].end]);
    try std.testing.expectEqualStrings("cd", res.mapped[res.spans[2].start..res.spans[2].end]);
}
