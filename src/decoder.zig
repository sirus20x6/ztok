//! Decoders turn token ids back into bytes.
//!
//! Variants:
//!   * `concat`     — naive byte concatenation. Works for any model.
//!   * `wordpiece`  — strips the `##` continuation prefix; inserts a
//!                    single space before non-continuation pieces.
//!                    Matches BERT-style decoding.
//!   * `byte_level` — reverses the GPT-2 byte-to-printable-unicode map
//!                    after concatenation. Pairs with
//!                    `Normalizer.byte_level` for HF byte-level BPE.

const std = @import("std");
const TokenId = @import("token.zig").TokenId;
const Vocab = @import("vocab.zig").Vocab;
const Model = @import("model.zig").Model;
const byte_level = @import("byte_level.zig");

pub const Decoder = union(enum) {
    concat,
    wordpiece: WordPieceConfig,
    byte_level,

    pub const WordPieceConfig = struct {
        continuing_subword_prefix: []const u8 = "##",
        word_separator: []const u8 = " ",
    };

    pub fn decode(
        self: Decoder,
        allocator: std.mem.Allocator,
        ids: []const TokenId,
        m: Model,
        vocab: *const Vocab,
    ) ![]u8 {
        switch (self) {
            .concat => {
                var scratch: [1]u8 = undefined;
                var total: usize = 0;
                for (ids) |id| {
                    const bytes = m.idBytes(id, vocab, &scratch);
                    total = try std.math.add(usize, total, bytes.len);
                }
                const out = try allocator.alloc(u8, total);
                var offset: usize = 0;
                for (ids) |id| {
                    const bytes = m.idBytes(id, vocab, &scratch);
                    @memcpy(out[offset .. offset + bytes.len], bytes);
                    offset += bytes.len;
                }
                return out;
            },
            .wordpiece => |cfg| {
                var scratch: [1]u8 = undefined;
                var total: usize = 0;
                for (ids, 0..) |id, i| {
                    const bytes = m.idBytes(id, vocab, &scratch);
                    if (std.mem.startsWith(u8, bytes, cfg.continuing_subword_prefix)) {
                        total = try std.math.add(usize, total, bytes.len - cfg.continuing_subword_prefix.len);
                    } else {
                        if (i > 0) total = try std.math.add(usize, total, cfg.word_separator.len);
                        total = try std.math.add(usize, total, bytes.len);
                    }
                }
                const out = try allocator.alloc(u8, total);
                var offset: usize = 0;
                for (ids, 0..) |id, i| {
                    const bytes = m.idBytes(id, vocab, &scratch);
                    if (std.mem.startsWith(u8, bytes, cfg.continuing_subword_prefix)) {
                        const piece = bytes[cfg.continuing_subword_prefix.len..];
                        @memcpy(out[offset .. offset + piece.len], piece);
                        offset += piece.len;
                    } else {
                        if (i > 0) {
                            @memcpy(out[offset .. offset + cfg.word_separator.len], cfg.word_separator);
                            offset += cfg.word_separator.len;
                        }
                        @memcpy(out[offset .. offset + bytes.len], bytes);
                        offset += bytes.len;
                    }
                }
                return out;
            },
            .byte_level => {
                // First concatenate, then reverse the byte_to_unicode map.
                var scratch: [1]u8 = undefined;
                var total: usize = 0;
                for (ids) |id| {
                    const bytes = m.idBytes(id, vocab, &scratch);
                    total = try std.math.add(usize, total, bytes.len);
                }
                const concat_buf = try allocator.alloc(u8, total);
                defer allocator.free(concat_buf);
                var offset: usize = 0;
                for (ids) |id| {
                    const bytes = m.idBytes(id, vocab, &scratch);
                    @memcpy(concat_buf[offset .. offset + bytes.len], bytes);
                    offset += bytes.len;
                }
                return byte_level.decodeMapped(allocator, concat_buf);
            },
        }
    }
};

test "concat decoder round-trips byte_id" {
    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();
    const ids = [_]TokenId{ 'h', 'i' };
    const out = try (Decoder{ .concat = {} }).decode(
        std.testing.allocator,
        &ids,
        .{ .byte_id = {} },
        &v,
    );
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("hi", out);
}

test "wordpiece decoder strips ## and inserts word separators" {
    const WordPiece = @import("wordpiece.zig").WordPiece;
    const vocab = [_][]const u8{ "[UNK]", "un", "##aff", "##able", "the", "cat" };
    var wp = try WordPiece.init(std.testing.allocator, &vocab, .{ .unk_id = 0 });
    defer wp.deinit();

    var v = Vocab.empty(std.testing.allocator);
    defer v.deinit();

    const dec: Decoder = .{ .wordpiece = .{} };
    const m: Model = .{ .wordpiece = &wp };

    const ids1 = [_]TokenId{ 1, 2, 3 };
    const out1 = try dec.decode(std.testing.allocator, &ids1, m, &v);
    defer std.testing.allocator.free(out1);
    try std.testing.expectEqualStrings("unaffable", out1);

    const ids2 = [_]TokenId{ 4, 5 };
    const out2 = try dec.decode(std.testing.allocator, &ids2, m, &v);
    defer std.testing.allocator.free(out2);
    try std.testing.expectEqualStrings("the cat", out2);
}

test "byte_level decoder reverses byte_to_unicode" {
    const Normalizer = @import("normalizer.zig").Normalizer;
    const mapped = try (Normalizer{ .byte_level = {} }).normalize(std.testing.allocator, " hi");
    defer std.testing.allocator.free(mapped);

    const round = try byte_level.decodeMapped(std.testing.allocator, mapped);
    defer std.testing.allocator.free(round);
    try std.testing.expectEqualStrings(" hi", round);
}
