//! Encoder trace sink: one struct that all encoders feed per-step
//! decisions into when the user passes `--trace` (or builds a Pipeline
//! with a non-null `trace` field).
//!
//! The Trace itself owns nothing — it borrows a `*std.Io.Writer` and
//! writes line-oriented records to it. Each encoder calls the helper
//! that matches its decision shape (`merge` for BPE, `unigramPiece`
//! for Unigram backtrack, `monsterPiece` for TM Monster piece picks).
//!
//! The record format is deliberately stable and grep-friendly so the
//! trace can be diffed across runs and consumed by external tools.
//!
//! Hot-path contract: callers gate every Trace call behind
//! `if (pipe.trace) |t| try t.<helper>(...)`. When `pipe.trace == null`
//! the encoder pays a single null check per decision point, fully
//! predictable; no intermediate strings, no allocations.

const std = @import("std");

const TokenId = @import("token.zig").TokenId;

pub const Trace = struct {
    writer: *std.Io.Writer,
    indent: u32 = 0,

    /// BPE merge step. `pos` is the merge index (parts slot that was
    /// extended), `rank` is the merge priority (== token id for
    /// tiktoken / HF byte-level), and `left` / `right` are the byte
    /// payloads of the two parts that got fused.
    pub fn merge(
        self: *Trace,
        pos: usize,
        rank: u32,
        left: []const u8,
        right: []const u8,
    ) !void {
        try self.indentPrefix();
        try self.writer.print("bpe merge pos={d} rank={d} left=", .{ pos, rank });
        try writeQuoted(self.writer, left);
        try self.writer.writeAll(" right=");
        try writeQuoted(self.writer, right);
        try self.writer.writeAll("\n");
    }

    /// Optional follow-up: render the current parts list as a single
    /// space-separated piece dump. Used to make merge traces easier
    /// to follow by hand. Caller decides whether to emit before, after,
    /// or both.
    ///
    /// `parts` is a slice of byte slices. Empty list emits "pieces=[]".
    pub fn pieces(self: *Trace, label: []const u8, parts: []const []const u8) !void {
        try self.indentPrefix();
        try self.writer.print("{s}=[", .{label});
        for (parts, 0..) |p, i| {
            if (i > 0) try self.writer.writeAll(", ");
            try writeQuoted(self.writer, p);
        }
        try self.writer.writeAll("]\n");
    }

    /// Unigram Viterbi backtrack: one record per emitted piece, in
    /// the order they appear in the final encoding.
    pub fn unigramPiece(
        self: *Trace,
        pos: usize,
        piece_id: TokenId,
        piece: []const u8,
        score: f32,
    ) !void {
        try self.indentPrefix();
        try self.writer.print(
            "unigram pos={d} piece_id={d} piece=",
            .{ pos, piece_id },
        );
        try writeQuoted(self.writer, piece);
        try self.writer.print(" score={d:.6}\n", .{score});
    }

    /// TM Monster per-piece pick. `len` is the real input-byte advance
    /// (NOT vocab byte count — those can differ on lilbuf-prefixed
    /// pieces); `score` is the picked branch's tm score, sentinel
    /// values get reported as-is so callers can spot DEL emits.
    pub fn monsterPiece(
        self: *Trace,
        pos: usize,
        piece_id: TokenId,
        len: u32,
        score: i32,
    ) !void {
        try self.indentPrefix();
        try self.writer.print(
            "monster pos={d} piece_id={d} len={d} score={d}\n",
            .{ pos, piece_id, len, score },
        );
    }

    fn indentPrefix(self: *Trace) !void {
        var i: u32 = 0;
        while (i < self.indent) : (i += 1) try self.writer.writeAll("  ");
    }
};

/// Render `bytes` as a double-quoted, escaped string suitable for a
/// trace record. Control characters and non-printable bytes are
/// emitted as `\xNN`; backslash and double-quote are escaped.
fn writeQuoted(w: *std.Io.Writer, bytes: []const u8) !void {
    try w.writeAll("\"");
    for (bytes) |b| {
        switch (b) {
            '\\' => try w.writeAll("\\\\"),
            '"' => try w.writeAll("\\\""),
            '\n' => try w.writeAll("\\n"),
            '\r' => try w.writeAll("\\r"),
            '\t' => try w.writeAll("\\t"),
            0x20...0x21, 0x23...0x5B, 0x5D...0x7E => try w.writeByte(b),
            else => try w.print("\\x{x:0>2}", .{b}),
        }
    }
    try w.writeAll("\"");
}

// ---------------------------------------------------------------- tests

test "Trace.merge writes a parseable record" {
    var aw: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer aw.deinit();
    var t: Trace = .{ .writer = &aw.writer };
    try t.merge(0, 42, "h", "e");
    const out = aw.written();
    try std.testing.expect(std.mem.indexOf(u8, out, "bpe merge") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "pos=0") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "rank=42") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "left=\"h\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "right=\"e\"") != null);
}

test "Trace.merge feeds BPE merge stream end-to-end" {
    // Tiny vocab where 'h','e','l','o',' ','w','r','d' are bytes 0..7
    // and 'he' is a single merge with rank 8.
    const bpe_mod = @import("bpe.zig");
    const root = @import("root.zig");
    const Pipeline = root.Pipeline;
    const sample =
        "aA== 0\n" ++ // 'h' - actually any single-byte ids work; not used
        "aGU= 1\n"; // 'he'
    _ = sample;

    // Build a vocab via the BPE Builder pattern: each line in tiktoken
    // is `base64(bytes) rank`. We'll use a tiny vocab covering 'h','e',
    // 'l','o' as single bytes and 'he' as the merged token.
    var bpe_buf = std.ArrayList(u8).empty;
    defer bpe_buf.deinit(std.testing.allocator);
    const enc = std.base64.standard.Encoder;
    var b64buf: [16]u8 = undefined;
    inline for (.{ "h", "e", "l", "o" }, 0..) |s, id| {
        const e = enc.encode(&b64buf, s);
        try bpe_buf.print(std.testing.allocator, "{s} {d}\n", .{ e, id });
    }
    const e = enc.encode(&b64buf, "he");
    try bpe_buf.print(std.testing.allocator, "{s} {d}\n", .{ e, 4 });

    var bpe = try bpe_mod.Bpe.loadTiktokenBytes(std.testing.allocator, bpe_buf.items);
    defer bpe.deinit();

    var v = root.Vocab.empty(std.testing.allocator);
    defer v.deinit();

    var aw: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer aw.deinit();
    var tr: Trace = .{ .writer = &aw.writer };

    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .{ .bpe = &bpe },
        .decoder = .concat,
        .vocab = &v,
        .trace = &tr,
    };

    const ids = try pipe.encode(std.testing.allocator, "hello");
    defer std.testing.allocator.free(ids);

    const out = aw.written();
    // We expect at least one merge record (the 'h'+'e' pair merges).
    try std.testing.expect(std.mem.indexOf(u8, out, "bpe merge") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "left=\"h\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "right=\"e\"") != null);
}

test "Trace pieces helper emits piece list" {
    var aw: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer aw.deinit();
    var t: Trace = .{ .writer = &aw.writer };
    const ps = [_][]const u8{ "he", "ll", "o" };
    try t.pieces("before", &ps);
    const s = aw.written();
    try std.testing.expect(std.mem.indexOf(u8, s, "before=[") != null);
    try std.testing.expect(std.mem.indexOf(u8, s, "\"he\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, s, "\"ll\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, s, "\"o\"") != null);
}
