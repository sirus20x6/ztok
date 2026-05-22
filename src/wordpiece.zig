//! WordPiece model: longest-match-first subword tokenization as used by
//! BERT. Expects pre-tokenized input — `encodeWord` takes a single word
//! and never splits across word boundaries.
//!
//! Storage is SoA: flat `bytes` + `offsets` mirror src/vocab.zig so the
//! reverse lookup hash map can borrow keys directly from `bytes` without
//! per-token allocations.

const std = @import("std");
const TokenId = @import("token.zig").TokenId;
const Span = @import("token.zig").Span;

pub const WordPiece = struct {
    allocator: std.mem.Allocator,
    bytes: []u8,
    offsets: []u32,
    count: u32,
    by_bytes: std.StringHashMap(TokenId),
    unk_id: TokenId,
    continuing_subword_prefix: []u8,
    // NB: measured in BYTES, not Unicode chars. HF's reference counts chars
    // via sequence.chars().count(); revisit when full Unicode lands (Wave C).
    max_input_chars_per_word: u32,

    pub const Options = struct {
        unk_id: TokenId,
        continuing_subword_prefix: []const u8 = "##",
        max_input_chars_per_word: u32 = 100,
    };

    pub fn init(allocator: std.mem.Allocator, vocab: []const []const u8, opts: Options) !WordPiece {
        var total: usize = 0;
        for (vocab) |t| total += t.len;

        const bytes = try allocator.alloc(u8, total);
        errdefer allocator.free(bytes);

        const offsets = try allocator.alloc(u32, vocab.len + 1);
        errdefer allocator.free(offsets);

        const prefix = try allocator.alloc(u8, opts.continuing_subword_prefix.len);
        errdefer allocator.free(prefix);
        @memcpy(prefix, opts.continuing_subword_prefix);

        var by_bytes: std.StringHashMap(TokenId) = .init(allocator);
        errdefer by_bytes.deinit();
        try by_bytes.ensureTotalCapacity(@intCast(vocab.len));

        var cursor: u32 = 0;
        for (vocab, 0..) |t, i| {
            offsets[i] = cursor;
            @memcpy(bytes[cursor .. cursor + t.len], t);
            // Key borrows from `bytes`; lifetime tied to this struct.
            const key = bytes[cursor .. cursor + t.len];
            cursor += @intCast(t.len);
            // Last-write-wins matches HF when duplicate tokens appear in the file.
            try by_bytes.put(key, @intCast(i));
        }
        offsets[vocab.len] = cursor;

        std.debug.assert(opts.unk_id < vocab.len);

        return .{
            .allocator = allocator,
            .bytes = bytes,
            .offsets = offsets,
            .count = @intCast(vocab.len),
            .by_bytes = by_bytes,
            .unk_id = opts.unk_id,
            .continuing_subword_prefix = prefix,
            .max_input_chars_per_word = opts.max_input_chars_per_word,
        };
    }

    pub fn deinit(self: *WordPiece) void {
        self.by_bytes.deinit();
        if (self.bytes.len > 0) self.allocator.free(self.bytes);
        if (self.offsets.len > 0) self.allocator.free(self.offsets);
        if (self.continuing_subword_prefix.len > 0) self.allocator.free(self.continuing_subword_prefix);
        self.* = undefined;
    }

    pub fn idBytes(self: *const WordPiece, id: TokenId) []const u8 {
        std.debug.assert(id < self.count);
        const start = self.offsets[id];
        const end = self.offsets[id + 1];
        return self.bytes[start..end];
    }

    /// Tokenize a single pre-tokenized word. Out must have capacity
    /// >= word.len + 1 (worst case: every byte becomes its own subword
    /// piece; the +1 is slack — emitting `unk` only ever writes one id).
    pub fn encodeWord(self: *const WordPiece, word: []const u8, out: []TokenId) []TokenId {
        std.debug.assert(out.len >= word.len + 1);

        if (word.len > self.max_input_chars_per_word) {
            out[0] = self.unk_id;
            return out[0..1];
        }

        // Stack buffer for prefixed substring. Bound by max_input_chars_per_word
        // + prefix. 256 covers the default (100 + "##") with room to spare;
        // if a caller raises the limit beyond this, the assert trips loud.
        var buf: [256]u8 = undefined;
        std.debug.assert(word.len + self.continuing_subword_prefix.len <= buf.len);

        var n: usize = 0;
        var start: usize = 0;
        while (start < word.len) {
            var end: usize = word.len;
            var matched: ?TokenId = null;
            var matched_end: usize = 0;

            while (start < end) {
                const sub = word[start..end];
                const key: []const u8 = if (start > 0) blk: {
                    const p = self.continuing_subword_prefix;
                    @memcpy(buf[0..p.len], p);
                    @memcpy(buf[p.len .. p.len + sub.len], sub);
                    break :blk buf[0 .. p.len + sub.len];
                } else sub;

                if (self.by_bytes.get(key)) |id| {
                    matched = id;
                    matched_end = end;
                    break;
                }
                // Byte-wise shrink; HF shrinks by last codepoint length.
                // Acceptable for ASCII and round-trip-safe for UTF-8 because
                // unmatched multibyte prefixes will still fall through to unk.
                end -= 1;
            }

            if (matched) |id| {
                out[n] = id;
                n += 1;
                start = matched_end;
            } else {
                // Entire word becomes one unk — matches HF's is_bad reset.
                out[0] = self.unk_id;
                return out[0..1];
            }
        }

        return out[0..n];
    }

    /// Same as `encodeWord` but also emits one span per id covering the
    /// byte range in the source buffer the offsets index into.
    /// `word_offset` is the position of `word[0]` in that buffer.
    /// A single-UNK fallback (whole word) spans `[word_offset, word_offset+word.len)`.
    pub fn encodeWordWithOffsets(
        self: *const WordPiece,
        word: []const u8,
        word_offset: u32,
        out_ids: []TokenId,
        out_offsets: []Span,
    ) usize {
        std.debug.assert(out_ids.len >= word.len + 1);
        std.debug.assert(out_offsets.len >= word.len + 1);

        if (word.len > self.max_input_chars_per_word) {
            out_ids[0] = self.unk_id;
            out_offsets[0] = .{ .start = word_offset, .end = word_offset + @as(u32, @intCast(word.len)) };
            return 1;
        }

        var buf: [256]u8 = undefined;
        std.debug.assert(word.len + self.continuing_subword_prefix.len <= buf.len);

        var n: usize = 0;
        var start: usize = 0;
        while (start < word.len) {
            var end: usize = word.len;
            var matched: ?TokenId = null;
            var matched_end: usize = 0;

            while (start < end) {
                const sub = word[start..end];
                const key: []const u8 = if (start > 0) blk: {
                    const p = self.continuing_subword_prefix;
                    @memcpy(buf[0..p.len], p);
                    @memcpy(buf[p.len .. p.len + sub.len], sub);
                    break :blk buf[0 .. p.len + sub.len];
                } else sub;

                if (self.by_bytes.get(key)) |id| {
                    matched = id;
                    matched_end = end;
                    break;
                }
                end -= 1;
            }

            if (matched) |id| {
                out_ids[n] = id;
                out_offsets[n] = .{
                    .start = word_offset + @as(u32, @intCast(start)),
                    .end = word_offset + @as(u32, @intCast(matched_end)),
                };
                n += 1;
                start = matched_end;
            } else {
                out_ids[0] = self.unk_id;
                out_offsets[0] = .{ .start = word_offset, .end = word_offset + @as(u32, @intCast(word.len)) };
                return 1;
            }
        }

        return n;
    }
};

// --- tests ---

test "init + idBytes" {
    const vocab = [_][]const u8{ "[UNK]", "##s", "hello" };
    var wp = try WordPiece.init(std.testing.allocator, &vocab, .{ .unk_id = 0 });
    defer wp.deinit();
    try std.testing.expectEqualStrings("hello", wp.idBytes(2));
    try std.testing.expectEqualStrings("##s", wp.idBytes(1));
    try std.testing.expectEqualStrings("[UNK]", wp.idBytes(0));
}

test "encodeWord exact match" {
    const vocab = [_][]const u8{ "[UNK]", "hello" };
    var wp = try WordPiece.init(std.testing.allocator, &vocab, .{ .unk_id = 0 });
    defer wp.deinit();

    var out: [16]TokenId = undefined;
    const ids = wp.encodeWord("hello", &out);
    try std.testing.expectEqualSlices(TokenId, &.{1}, ids);
}

test "encodeWord greedy split" {
    const vocab = [_][]const u8{ "[UNK]", "un", "##aff", "##able" };
    var wp = try WordPiece.init(std.testing.allocator, &vocab, .{ .unk_id = 0 });
    defer wp.deinit();

    var out: [16]TokenId = undefined;
    const ids = wp.encodeWord("unaffable", &out);
    try std.testing.expectEqualSlices(TokenId, &.{ 1, 2, 3 }, ids);
}

test "encodeWord unknown -> single unk" {
    const vocab = [_][]const u8{ "[UNK]", "hello" };
    var wp = try WordPiece.init(std.testing.allocator, &vocab, .{ .unk_id = 0 });
    defer wp.deinit();

    var out: [16]TokenId = undefined;
    const ids = wp.encodeWord("xyz", &out);
    try std.testing.expectEqualSlices(TokenId, &.{0}, ids);
}

test "encodeWord too-long -> unk" {
    const vocab = [_][]const u8{"[UNK]"};
    var wp = try WordPiece.init(std.testing.allocator, &vocab, .{
        .unk_id = 0,
        .max_input_chars_per_word = 4,
    });
    defer wp.deinit();

    var out: [16]TokenId = undefined;
    const ids = wp.encodeWord("hello", &out);
    try std.testing.expectEqualSlices(TokenId, &.{0}, ids);
}

test "encodeWordWithOffsets attributes ## continuations correctly" {
    // {un, ##aff, ##able}. "unaffable" -> [un(0..2), ##aff(2..5), ##able(5..9)].
    const vocab = [_][]const u8{ "[UNK]", "un", "##aff", "##able" };
    var wp = try WordPiece.init(std.testing.allocator, &vocab, .{ .unk_id = 0 });
    defer wp.deinit();

    var out_ids: [16]TokenId = undefined;
    var out_off: [16]Span = undefined;
    const n = wp.encodeWordWithOffsets("unaffable", 0, &out_ids, &out_off);
    try std.testing.expectEqual(@as(usize, 3), n);
    try std.testing.expectEqualSlices(TokenId, &.{ 1, 2, 3 }, out_ids[0..n]);

    try std.testing.expectEqual(@as(u32, 0), out_off[0].start);
    try std.testing.expectEqual(@as(u32, 2), out_off[0].end);
    try std.testing.expectEqual(@as(u32, 2), out_off[1].start);
    try std.testing.expectEqual(@as(u32, 5), out_off[1].end);
    try std.testing.expectEqual(@as(u32, 5), out_off[2].start);
    try std.testing.expectEqual(@as(u32, 9), out_off[2].end);
}

test "encodeWordWithOffsets respects word_offset" {
    const vocab = [_][]const u8{ "[UNK]", "hello" };
    var wp = try WordPiece.init(std.testing.allocator, &vocab, .{ .unk_id = 0 });
    defer wp.deinit();

    var out_ids: [16]TokenId = undefined;
    var out_off: [16]Span = undefined;
    const n = wp.encodeWordWithOffsets("hello", 7, &out_ids, &out_off);
    try std.testing.expectEqual(@as(usize, 1), n);
    try std.testing.expectEqual(@as(u32, 7), out_off[0].start);
    try std.testing.expectEqual(@as(u32, 12), out_off[0].end);
}

test "encodeWordWithOffsets unknown -> single unk spans whole word" {
    const vocab = [_][]const u8{ "[UNK]", "hello" };
    var wp = try WordPiece.init(std.testing.allocator, &vocab, .{ .unk_id = 0 });
    defer wp.deinit();

    var out_ids: [16]TokenId = undefined;
    var out_off: [16]Span = undefined;
    const n = wp.encodeWordWithOffsets("xyz", 4, &out_ids, &out_off);
    try std.testing.expectEqual(@as(usize, 1), n);
    try std.testing.expectEqual(@as(TokenId, 0), out_ids[0]);
    try std.testing.expectEqual(@as(u32, 4), out_off[0].start);
    try std.testing.expectEqual(@as(u32, 7), out_off[0].end);
}
