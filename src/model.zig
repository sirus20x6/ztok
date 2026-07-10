//! The model turns a byte span into a sequence of token ids and back.
//!
//! Tagged union over the concrete kinds — no vtables, no heap dispatch.
//! Variants today:
//!   * `byte_id`   — stub: each input byte is its own id (0..255).
//!   * `bpe`       — byte-level BPE (tiktoken-format vocabs).
//!   * `unigram`   — SentencePiece-style Viterbi over a unigram LM.
//!   * `wordpiece` — BERT-style longest-match-first.
//!   * `monster`   — TokenMonster-style 2-branch ungreedy.

const std = @import("std");
const TokenId = @import("token.zig").TokenId;
const Span = @import("token.zig").Span;
const Vocab = @import("vocab.zig").Vocab;
const Bpe = @import("bpe.zig").Bpe;
const Unigram = @import("unigram.zig").Unigram;
const WordPiece = @import("wordpiece.zig").WordPiece;
const Monster = @import("monster.zig").Monster;
const RwkvWorld = @import("rwkv_world.zig").RwkvWorld;
const trace_mod = @import("trace.zig");

pub const Model = union(enum) {
    byte_id,
    bpe: *const Bpe,
    unigram: *const Unigram,
    wordpiece: *const WordPiece,
    monster: *const Monster,
    rwkv_world: *const RwkvWorld,

    pub fn encode(
        self: Model,
        scratch: std.mem.Allocator,
        chunk: []const u8,
        out: []TokenId,
    ) ![]TokenId {
        return switch (self) {
            .byte_id => blk: {
                std.debug.assert(out.len >= chunk.len);
                for (chunk, 0..) |b, i| out[i] = b;
                break :blk out[0..chunk.len];
            },
            // Thread `scratch` into BPE so the per-thread arena absorbs
            // the rare heap-spillover allocations for chunks > 256 bytes
            // — keeps the GPA off the hot path on long unbroken runs.
            .bpe => |b| b.encodeChunkScratch(scratch, chunk, out),
            .unigram => |u| try u.encodeChunk(scratch, chunk, out),
            .wordpiece => |w| w.encodeWord(chunk, out),
            .monster => |m| try m.encodeChunk(scratch, chunk, out),
            // RWKV World: greedy longest-match over the whole chunk; no
            // scratch needed.
            .rwkv_world => |r| try r.encodeChunk(chunk, out),
        };
    }

    /// Same as `encode` but threads an optional `trace` sink into the
    /// encoder. When `trace == null`, the encoder hits the same hot
    /// path as `encode` (the BPE / Unigram / Monster fast functions
    /// take an optional pointer; the per-decision `if (trace) |t|`
    /// branch is predicted not-taken). When non-null, each decision
    /// point emits a record via the configured writer.
    pub fn encodeTraced(
        self: Model,
        scratch: std.mem.Allocator,
        chunk: []const u8,
        out: []TokenId,
        trace: ?*trace_mod.Trace,
    ) ![]TokenId {
        // Null-trace: skip the traced entry points entirely so the
        // existing hot path stays bit-identical.
        if (trace == null) return self.encode(scratch, chunk, out);
        return switch (self) {
            .byte_id => blk: {
                std.debug.assert(out.len >= chunk.len);
                for (chunk, 0..) |b, i| out[i] = b;
                break :blk out[0..chunk.len];
            },
            .bpe => |b| try b.encodeChunkScratchTrace(scratch, chunk, out, trace.?),
            .unigram => |u| try u.encodeChunkTrace(scratch, chunk, out, trace.?),
            .wordpiece => |w| w.encodeWord(chunk, out),
            .monster => |m| try m.encodeChunkTrace(scratch, chunk, out, trace.?),
            // RWKV World has no per-decision trace records; encode plainly.
            .rwkv_world => |r| try r.encodeChunk(chunk, out),
        };
    }

    pub fn maxTokensFor(self: Model, n: usize) usize {
        return switch (self) {
            // RWKV World emits at most one id per input byte (every byte
            // is itself a token), so `n` is the exact upper bound.
            .byte_id, .bpe, .unigram, .rwkv_world => n,
            // Monster's lilbuf path (a) can emit two ids (DEL + matched
            // piece) per input byte in the worst case — see
            // `monster.zig` lilbuf docs. Size for the worst case.
            .monster => n * 2,
            .wordpiece => n + 1,
        };
    }

    /// Encode `chunk` and also fill `out_offsets` with per-id byte spans
    /// in the buffer the caller indexes into. `chunk_offset` is the
    /// position of `chunk[0]` in that buffer. Returns the number of ids
    /// written. Both `out_ids` and `out_offsets` must hold at least
    /// `maxTokensFor(chunk.len)` entries.
    pub fn encodeWithOffsets(
        self: Model,
        scratch: std.mem.Allocator,
        chunk: []const u8,
        chunk_offset: u32,
        out_ids: []TokenId,
        out_offsets: []Span,
    ) !usize {
        return switch (self) {
            .byte_id => blk: {
                std.debug.assert(out_ids.len >= chunk.len);
                std.debug.assert(out_offsets.len >= chunk.len);
                for (chunk, 0..) |b, i| {
                    out_ids[i] = b;
                    const off: u32 = @intCast(i);
                    out_offsets[i] = .{
                        .start = chunk_offset + off,
                        .end = chunk_offset + off + 1,
                    };
                }
                break :blk chunk.len;
            },
            .bpe => |b| b.encodeChunkWithOffsetsScratch(scratch, chunk, chunk_offset, out_ids, out_offsets),
            .unigram => |u| try u.encodeChunkWithOffsets(scratch, chunk, chunk_offset, out_ids, out_offsets),
            .wordpiece => |w| w.encodeWordWithOffsets(chunk, chunk_offset, out_ids, out_offsets),
            .monster => |m| try m.encodeChunkWithOffsets(scratch, chunk, chunk_offset, out_ids, out_offsets),
            .rwkv_world => |r| try r.encodeChunkWithOffsets(chunk, chunk_offset, out_ids, out_offsets),
        };
    }

    /// Same as `encodeWithOffsets` but threads a trace sink. See
    /// `encodeTraced` for the zero-overhead-when-null contract.
    pub fn encodeWithOffsetsTraced(
        self: Model,
        scratch: std.mem.Allocator,
        chunk: []const u8,
        chunk_offset: u32,
        out_ids: []TokenId,
        out_offsets: []Span,
        trace: ?*trace_mod.Trace,
    ) !usize {
        if (trace == null) return self.encodeWithOffsets(scratch, chunk, chunk_offset, out_ids, out_offsets);
        return switch (self) {
            .byte_id => blk: {
                std.debug.assert(out_ids.len >= chunk.len);
                std.debug.assert(out_offsets.len >= chunk.len);
                for (chunk, 0..) |b, i| {
                    out_ids[i] = b;
                    const off: u32 = @intCast(i);
                    out_offsets[i] = .{
                        .start = chunk_offset + off,
                        .end = chunk_offset + off + 1,
                    };
                }
                break :blk chunk.len;
            },
            .bpe => |b| try b.encodeChunkWithOffsetsScratchTrace(scratch, chunk, chunk_offset, out_ids, out_offsets, trace.?),
            .unigram => |u| try u.encodeChunkWithOffsetsTrace(scratch, chunk, chunk_offset, out_ids, out_offsets, trace.?),
            .wordpiece => |w| w.encodeWordWithOffsets(chunk, chunk_offset, out_ids, out_offsets),
            .monster => |m| try m.encodeChunkWithOffsetsTrace(scratch, chunk, chunk_offset, out_ids, out_offsets, trace.?),
            .rwkv_world => |r| try r.encodeChunkWithOffsets(chunk, chunk_offset, out_ids, out_offsets),
        };
    }

    pub fn idBytes(self: Model, id: TokenId, vocab: *const Vocab, scratch: *[1]u8) []const u8 {
        return switch (self) {
            .byte_id => blk: {
                _ = vocab;
                scratch[0] = @intCast(id & 0xFF);
                break :blk scratch[0..1];
            },
            .bpe => |b| b.idBytes(id),
            .unigram => |u| u.idBytes(id),
            .wordpiece => |w| w.idBytes(id),
            .monster => |m| m.idBytes(id),
            .rwkv_world => |r| r.idBytes(id),
        };
    }
};

test "byte_id encode echoes bytes" {
    var buf: [16]TokenId = undefined;
    const ids = try (Model{ .byte_id = {} }).encode(std.testing.allocator, "abc", &buf);
    try std.testing.expectEqualSlices(TokenId, &.{ 'a', 'b', 'c' }, ids);
}

test "bpe variant dispatch" {
    const sample =
        "YQ== 0\n" ++ // 'a'
        "Yg== 1\n" ++ // 'b'
        "YWI= 2\n"; // 'ab'
    var b = try Bpe.loadTiktokenBytes(std.testing.allocator, sample);
    defer b.deinit();

    const m: Model = .{ .bpe = &b };
    var buf: [4]TokenId = undefined;
    const ids = try m.encode(std.testing.allocator, "ab", &buf);
    try std.testing.expectEqualSlices(TokenId, &.{2}, ids);
}

test "wordpiece variant dispatch" {
    const vocab = [_][]const u8{ "[UNK]", "un", "##aff", "##able" };
    var wp = try WordPiece.init(std.testing.allocator, &vocab, .{ .unk_id = 0 });
    defer wp.deinit();

    const m: Model = .{ .wordpiece = &wp };
    var buf: [16]TokenId = undefined;
    const ids = try m.encode(std.testing.allocator, "unaffable", &buf);
    try std.testing.expectEqualSlices(TokenId, &.{ 1, 2, 3 }, ids);
}

test "rwkv_world variant dispatch" {
    const entries = [_]RwkvWorld.Entry{
        .{ .id = 0, .bytes = "a" },
        .{ .id = 1, .bytes = "b" },
        .{ .id = 2, .bytes = "ab" },
        .{ .id = 3, .bytes = "abc" },
        .{ .id = 4, .bytes = "c" },
    };
    var r = try RwkvWorld.init(std.testing.allocator, &entries);
    defer r.deinit();

    const m: Model = .{ .rwkv_world = &r };
    var buf: [4]TokenId = undefined;
    const ids = try m.encode(std.testing.allocator, "abc", &buf);
    try std.testing.expectEqualSlices(TokenId, &.{3}, ids); // greedy longest
}

test "unigram variant dispatch" {
    var bld = Unigram.Builder.init(std.testing.allocator);
    defer bld.deinit();
    _ = try bld.addToken("<unk>", -100.0);
    _ = try bld.addToken("a", -2.0);
    _ = try bld.addToken("b", -2.0);
    _ = try bld.addToken("ab", -1.0);
    var u = try bld.finalize(0);
    defer u.deinit();

    const m: Model = .{ .unigram = &u };
    var buf: [4]TokenId = undefined;
    const ids = try m.encode(std.testing.allocator, "ab", &buf);
    try std.testing.expectEqualSlices(TokenId, &.{3}, ids);
}

test "monster variant dispatch" {
    var bld = Monster.Builder.init(std.testing.allocator);
    defer bld.deinit();
    _ = try bld.addToken("<unk>"); // id 0
    _ = try bld.addToken("a"); // id 1
    _ = try bld.addToken("ab"); // id 2
    _ = try bld.addToken("abc"); // id 3
    var mon = try bld.finalize(0);
    defer mon.deinit();

    const m: Model = .{ .monster = &mon };
    var buf: [4]TokenId = undefined;
    const ids = try m.encode(std.testing.allocator, "abc", &buf);
    try std.testing.expectEqualSlices(TokenId, &.{3}, ids); // ungreedy picks longest at tie
}
