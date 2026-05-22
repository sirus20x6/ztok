//! SentencePiece .model loader. Decodes the proto wire format using
//! proto.zig and packs pieces into a flat SoA layout matching vocab.zig.
//!
//! Field numbers verified against refs/sentencepiece/src/sentencepiece_model.proto:
//!   ModelProto:    pieces=1, trainer_spec=2, normalizer_spec=3
//!   SentencePiece: piece=1, score=2, type=3
//!   NormalizerSpec: name=1, precompiled_charsmap=2, add_dummy_prefix=3,
//!                   remove_extra_whitespaces=4, escape_whitespaces=5
//!   TrainerSpec:   model_type=9 (in spec it's actually 3 -- see note)
//!
//! NOTE: trainer_spec.model_type is field 3 (verified). The original task
//! description said field 9; the .proto has it at 3.

const std = @import("std");
const proto = @import("proto.zig");
const TokenId = @import("token.zig").TokenId;
const sp_charsmap = @import("sp_charsmap.zig");

pub const PieceType = enum(u8) {
    normal = 1,
    unknown = 2,
    control = 3,
    user_defined = 4,
    byte = 6,
    unused = 5,
};

pub const ModelKind = enum { unigram, bpe, word, char };

pub const SpModel = struct {
    allocator: std.mem.Allocator,
    bytes: []u8,
    offsets: []u32, // count + 1
    scores: []f32,
    types: []PieceType,
    count: u32,
    model_kind: ModelKind,
    normalizer_name: ?[]u8 = null,
    add_dummy_prefix: bool = true,
    remove_extra_whitespaces: bool = true,
    escape_whitespaces: bool = true,
    /// `trainer_spec.byte_fallback` (field 35). When true, the model
    /// guarantees 256 dedicated single-byte tokens of type `.byte`
    /// (`<0x00>`..`<0xFF>`) somewhere in the vocab; the SP encoder must
    /// fall back to those ids for bytes the merge loop couldn't cover.
    byte_fallback: bool = false,
    /// `normalizer_spec.precompiled_charsmap` (field 2). Raw serialized
    /// Darts double-array trie + normalized-string blob (see
    /// `src/sp_charsmap.zig` for the parser). Owned by `SpModel`; the
    /// alignment is at least 4 bytes (the loader copies into an
    /// `[]align(4) u8` buffer so `sp_charsmap.parse` can reinterpret
    /// `bytes[4..]` as `[]u32` without an extra copy).
    ///
    /// Most LLaMA/Gemma-style models omit this field (the spec only
    /// names the normalizer `nmt_nfkc` and SP applies NFKC at runtime).
    /// T5, mBART, and many Japanese/Chinese SP models bake additional
    /// per-model rules here.
    precompiled_charsmap: ?[]align(4) u8 = null,
    /// Lazily-parsed view over `precompiled_charsmap`. Heap-allocated so
    /// `*const PrecompiledCharsmap` is stable across `SpModel` copies /
    /// moves (the `Normalizer{ .sp_precompiled = ... }` variant stashes
    /// the pointer; we need it to stay valid across struct moves the
    /// pipeline owner might do). Populated on demand by `parsedCharsmap`.
    parsed_charsmap: ?*sp_charsmap.PrecompiledCharsmap = null,

    pub fn deinit(self: *SpModel) void {
        if (self.bytes.len > 0) self.allocator.free(self.bytes);
        if (self.offsets.len > 0) self.allocator.free(self.offsets);
        if (self.scores.len > 0) self.allocator.free(self.scores);
        if (self.types.len > 0) self.allocator.free(self.types);
        if (self.normalizer_name) |n| self.allocator.free(n);
        if (self.precompiled_charsmap) |c| self.allocator.free(c);
        if (self.parsed_charsmap) |p| self.allocator.destroy(p);
        self.bytes = &.{};
        self.offsets = &.{};
        self.scores = &.{};
        self.types = &.{};
        self.normalizer_name = null;
        self.precompiled_charsmap = null;
        self.parsed_charsmap = null;
        self.count = 0;
    }

    pub fn pieceBytes(self: *const SpModel, id: TokenId) []const u8 {
        std.debug.assert(id < self.count);
        const s = self.offsets[id];
        const e = self.offsets[id + 1];
        return self.bytes[s..e];
    }

    /// Parse `precompiled_charsmap` on demand and return a stable
    /// pointer to the resulting view. Returns `null` if the model has
    /// no charsmap. The pointee is owned by this `SpModel` — it stays
    /// valid until `deinit`. Subsequent calls return the same pointer.
    ///
    /// Errors surface from the charsmap parser; the most common cause
    /// is a model file with a malformed trie blob (e.g. truncated mid-
    /// transfer). Callers that want to degrade gracefully on parse
    /// failures should check the return + treat any error as "no
    /// charsmap" (same behaviour as a model that omits the field).
    pub fn parsedCharsmap(self: *SpModel) !?*const sp_charsmap.PrecompiledCharsmap {
        if (self.parsed_charsmap) |p| return p;
        const raw = self.precompiled_charsmap orelse return null;
        const parsed = try sp_charsmap.parse(raw);
        const slot = try self.allocator.create(sp_charsmap.PrecompiledCharsmap);
        slot.* = parsed;
        self.parsed_charsmap = slot;
        return slot;
    }
};

pub const Error = error{ MalformedProto, MissingField } || std.mem.Allocator.Error;

// Decode a single SentencePiece sub-message, appending to the SoA builders.
fn decodePiece(
    allocator: std.mem.Allocator,
    msg: *proto.Reader,
    bytes: *std.ArrayList(u8),
    offsets: *std.ArrayList(u32),
    scores: *std.ArrayList(f32),
    types: *std.ArrayList(PieceType),
) Error!void {
    var piece_bytes: []const u8 = &.{};
    var score: f32 = 0.0;
    var ty: PieceType = .normal;

    while (!msg.done()) {
        const tag = msg.nextTag() catch return Error.MalformedProto;
        switch (tag.field) {
            1 => { // piece (string)
                if (tag.wire != .len) return Error.MalformedProto;
                piece_bytes = msg.readLen() catch return Error.MalformedProto;
            },
            2 => { // score (float)
                if (tag.wire != .i32) return Error.MalformedProto;
                score = msg.readFloat() catch return Error.MalformedProto;
            },
            3 => { // type (enum/varint)
                if (tag.wire != .varint) return Error.MalformedProto;
                const v = msg.readVarint() catch return Error.MalformedProto;
                ty = switch (v) {
                    1 => .normal,
                    2 => .unknown,
                    3 => .control,
                    4 => .user_defined,
                    5 => .unused,
                    6 => .byte,
                    else => return Error.MalformedProto,
                };
            },
            else => msg.skip(tag.wire) catch return Error.MalformedProto,
        }
    }

    // Offsets carries count+1 entries; seed the leading 0 on the first piece.
    if (offsets.items.len == 0) try offsets.append(allocator, 0);
    try bytes.appendSlice(allocator, piece_bytes);
    try offsets.append(allocator, @intCast(bytes.items.len));
    try scores.append(allocator, score);
    try types.append(allocator, ty);
}

fn decodeNormalizer(out: *SpModel, msg: *proto.Reader) Error!void {
    while (!msg.done()) {
        const tag = msg.nextTag() catch return Error.MalformedProto;
        switch (tag.field) {
            1 => { // name
                if (tag.wire != .len) return Error.MalformedProto;
                const s = msg.readLen() catch return Error.MalformedProto;
                if (out.normalizer_name) |old| out.allocator.free(old);
                out.normalizer_name = try out.allocator.dupe(u8, s);
            },
            2 => { // precompiled_charsmap (bytes)
                if (tag.wire != .len) return Error.MalformedProto;
                const s = msg.readLen() catch return Error.MalformedProto;
                if (out.precompiled_charsmap) |old| out.allocator.free(old);
                // Allocate a 4-byte-aligned buffer so the trie segment
                // (bytes[4..4+trie_size]) reinterprets as []u32 without
                // a second copy. The default GeneralPurposeAllocator
                // gives us 8/16-aligned blocks anyway, but we want the
                // alignment guarantee in the type so `sp_charsmap.parse`
                // can use `@alignCast` safely.
                const buf = try out.allocator.alignedAlloc(u8, .of(u32), s.len);
                @memcpy(buf, s);
                out.precompiled_charsmap = buf;
            },
            3 => { // add_dummy_prefix
                if (tag.wire != .varint) return Error.MalformedProto;
                const v = msg.readVarint() catch return Error.MalformedProto;
                out.add_dummy_prefix = v != 0;
            },
            4 => { // remove_extra_whitespaces
                if (tag.wire != .varint) return Error.MalformedProto;
                const v = msg.readVarint() catch return Error.MalformedProto;
                out.remove_extra_whitespaces = v != 0;
            },
            5 => { // escape_whitespaces
                if (tag.wire != .varint) return Error.MalformedProto;
                const v = msg.readVarint() catch return Error.MalformedProto;
                out.escape_whitespaces = v != 0;
            },
            else => msg.skip(tag.wire) catch return Error.MalformedProto,
        }
    }
}

fn decodeTrainer(out: *SpModel, msg: *proto.Reader) Error!void {
    while (!msg.done()) {
        const tag = msg.nextTag() catch return Error.MalformedProto;
        switch (tag.field) {
            3 => { // model_type (enum)
                if (tag.wire != .varint) return Error.MalformedProto;
                const v = msg.readVarint() catch return Error.MalformedProto;
                out.model_kind = switch (v) {
                    1 => .unigram,
                    2 => .bpe,
                    3 => .word,
                    4 => .char,
                    else => return Error.MalformedProto,
                };
            },
            35 => { // byte_fallback
                if (tag.wire != .varint) return Error.MalformedProto;
                const v = msg.readVarint() catch return Error.MalformedProto;
                out.byte_fallback = v != 0;
            },
            else => msg.skip(tag.wire) catch return Error.MalformedProto,
        }
    }
}

pub fn loadFromBytes(allocator: std.mem.Allocator, contents: []const u8) Error!SpModel {
    var out: SpModel = .{
        .allocator = allocator,
        .bytes = &.{},
        .offsets = &.{},
        .scores = &.{},
        .types = &.{},
        .count = 0,
        .model_kind = .unigram,
    };
    errdefer out.deinit();

    var bytes_buf: std.ArrayList(u8) = .empty;
    defer bytes_buf.deinit(allocator);
    var offsets_buf: std.ArrayList(u32) = .empty;
    defer offsets_buf.deinit(allocator);
    var scores_buf: std.ArrayList(f32) = .empty;
    defer scores_buf.deinit(allocator);
    var types_buf: std.ArrayList(PieceType) = .empty;
    defer types_buf.deinit(allocator);

    var r = proto.Reader.init(contents);
    while (!r.done()) {
        const tag = r.nextTag() catch return Error.MalformedProto;
        switch (tag.field) {
            1 => { // pieces (repeated SentencePiece)
                if (tag.wire != .len) return Error.MalformedProto;
                var sub = r.readMessage() catch return Error.MalformedProto;
                try decodePiece(allocator, &sub, &bytes_buf, &offsets_buf, &scores_buf, &types_buf);
            },
            2 => { // trainer_spec
                if (tag.wire != .len) return Error.MalformedProto;
                var sub = r.readMessage() catch return Error.MalformedProto;
                try decodeTrainer(&out, &sub);
            },
            3 => { // normalizer_spec
                if (tag.wire != .len) return Error.MalformedProto;
                var sub = r.readMessage() catch return Error.MalformedProto;
                try decodeNormalizer(&out, &sub);
            },
            else => r.skip(tag.wire) catch return Error.MalformedProto,
        }
    }

    // If no pieces were parsed, offsets is empty -- seed the sentinel.
    if (offsets_buf.items.len == 0) try offsets_buf.append(allocator, 0);

    out.bytes = try bytes_buf.toOwnedSlice(allocator);
    out.offsets = try offsets_buf.toOwnedSlice(allocator);
    out.scores = try scores_buf.toOwnedSlice(allocator);
    out.types = try types_buf.toOwnedSlice(allocator);
    out.count = @intCast(out.scores.len);
    return out;
}

pub fn loadFromFile(allocator: std.mem.Allocator, path: []const u8) !SpModel {
    const io = std.Io.Threaded.global_single_threaded.io();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
    defer allocator.free(bytes);
    return loadFromBytes(allocator, bytes);
}

// --- test helpers / tests --------------------------------------------------

fn tvEncodeVarint(out: *std.ArrayList(u8), allocator: std.mem.Allocator, v: u64) !void {
    var x = v;
    while (x >= 0x80) {
        try out.append(allocator, @as(u8, @intCast(x & 0x7f)) | 0x80);
        x >>= 7;
    }
    try out.append(allocator, @intCast(x));
}

fn tvTag(out: *std.ArrayList(u8), allocator: std.mem.Allocator, field: u32, wire: u3) !void {
    try tvEncodeVarint(out, allocator, (@as(u64, field) << 3) | wire);
}

fn tvFloat(out: *std.ArrayList(u8), allocator: std.mem.Allocator, f: f32) !void {
    var b: [4]u8 = undefined;
    const u: u32 = @bitCast(f);
    std.mem.writeInt(u32, &b, u, .little);
    try out.appendSlice(allocator, &b);
}

// Build one SentencePiece sub-message body (no outer length-delim wrapper).
fn tvPieceBody(out: *std.ArrayList(u8), allocator: std.mem.Allocator, piece: []const u8, score: f32, ty: ?u64) !void {
    // field 1: piece (len)
    try tvTag(out, allocator, 1, 2);
    try tvEncodeVarint(out, allocator, piece.len);
    try out.appendSlice(allocator, piece);
    // field 2: score (i32)
    try tvTag(out, allocator, 2, 5);
    try tvFloat(out, allocator, score);
    // field 3: type (varint, optional)
    if (ty) |t| {
        try tvTag(out, allocator, 3, 0);
        try tvEncodeVarint(out, allocator, t);
    }
}

fn tvEmbedMsg(out: *std.ArrayList(u8), allocator: std.mem.Allocator, field: u32, body: []const u8) !void {
    try tvTag(out, allocator, field, 2);
    try tvEncodeVarint(out, allocator, body.len);
    try out.appendSlice(allocator, body);
}

test "decodes a 2-piece minimal model" {
    const allocator = std.testing.allocator;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);

    // piece 0
    var p0: std.ArrayList(u8) = .empty;
    defer p0.deinit(allocator);
    try tvPieceBody(&p0, allocator, "a", -1.0, 1);
    try tvEmbedMsg(&buf, allocator, 1, p0.items);

    // piece 1
    var p1: std.ArrayList(u8) = .empty;
    defer p1.deinit(allocator);
    try tvPieceBody(&p1, allocator, "ab", -2.0, null);
    try tvEmbedMsg(&buf, allocator, 1, p1.items);

    var m = try loadFromBytes(allocator, buf.items);
    defer m.deinit();

    try std.testing.expectEqual(@as(u32, 2), m.count);
    try std.testing.expectEqualStrings("a", m.pieceBytes(0));
    try std.testing.expectEqualStrings("ab", m.pieceBytes(1));
    try std.testing.expectEqual(@as(f32, -1.0), m.scores[0]);
    try std.testing.expectEqual(@as(f32, -2.0), m.scores[1]);
    try std.testing.expectEqual(PieceType.normal, m.types[0]);
    try std.testing.expectEqual(PieceType.normal, m.types[1]);
}

test "skips unknown fields without erroring" {
    const allocator = std.testing.allocator;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);

    // Phantom field 99 with varint value first
    try tvTag(&buf, allocator, 99, 0);
    try tvEncodeVarint(&buf, allocator, 12345);

    // One valid piece
    var p: std.ArrayList(u8) = .empty;
    defer p.deinit(allocator);
    try tvPieceBody(&p, allocator, "z", 0.5, 1);
    try tvEmbedMsg(&buf, allocator, 1, p.items);

    // Phantom field 77 with len value at the tail
    try tvTag(&buf, allocator, 77, 2);
    try tvEncodeVarint(&buf, allocator, 3);
    try buf.appendSlice(allocator, "abc");

    var m = try loadFromBytes(allocator, buf.items);
    defer m.deinit();

    try std.testing.expectEqual(@as(u32, 1), m.count);
    try std.testing.expectEqualStrings("z", m.pieceBytes(0));
}

test "captures normalizer_name" {
    const allocator = std.testing.allocator;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);

    // NormalizerSpec body
    var ns: std.ArrayList(u8) = .empty;
    defer ns.deinit(allocator);
    // field 1: name = "nmt_nfkc"
    try tvTag(&ns, allocator, 1, 2);
    try tvEncodeVarint(&ns, allocator, "nmt_nfkc".len);
    try ns.appendSlice(allocator, "nmt_nfkc");
    // field 3: add_dummy_prefix = false
    try tvTag(&ns, allocator, 3, 0);
    try tvEncodeVarint(&ns, allocator, 0);

    try tvEmbedMsg(&buf, allocator, 3, ns.items);

    var m = try loadFromBytes(allocator, buf.items);
    defer m.deinit();

    try std.testing.expect(m.normalizer_name != null);
    try std.testing.expectEqualStrings("nmt_nfkc", m.normalizer_name.?);
    try std.testing.expectEqual(false, m.add_dummy_prefix);
    try std.testing.expectEqual(true, m.remove_extra_whitespaces);
}

test "loadFromFile round-trips through a real path" {
    const allocator = std.testing.allocator;

    // Build a minimal 2-piece ModelProto in memory.
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);

    var p0: std.ArrayList(u8) = .empty;
    defer p0.deinit(allocator);
    try tvPieceBody(&p0, allocator, "x", -0.25, 1);
    try tvEmbedMsg(&buf, allocator, 1, p0.items);

    var p1: std.ArrayList(u8) = .empty;
    defer p1.deinit(allocator);
    try tvPieceBody(&p1, allocator, "xy", -0.5, 1);
    try tvEmbedMsg(&buf, allocator, 1, p1.items);

    // Drop the bytes onto /tmp via the same Io path loadFromFile uses.
    // Mirrors the writeFile/readFile pattern in monster_io.zig.
    const io = std.Io.Threaded.global_single_threaded.io();
    var rng = std.Random.DefaultPrng.init(0xDEAD_BEEF_BAD_F00D);
    const r = rng.random().int(u64);
    var path_buf: [80]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "/tmp/ztok_sp_model_loadfile_test_{x}.model", .{r});
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = buf.items });
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    var m = try loadFromFile(allocator, path);
    defer m.deinit();

    try std.testing.expectEqual(@as(u32, 2), m.count);
    try std.testing.expectEqualStrings("x", m.pieceBytes(0));
    try std.testing.expectEqualStrings("xy", m.pieceBytes(1));
    try std.testing.expectEqual(@as(f32, -0.25), m.scores[0]);
}

test "trainer_spec model_type captured" {
    const allocator = std.testing.allocator;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);

    // TrainerSpec body with model_type=BPE (2)
    var ts: std.ArrayList(u8) = .empty;
    defer ts.deinit(allocator);
    try tvTag(&ts, allocator, 3, 0);
    try tvEncodeVarint(&ts, allocator, 2);

    try tvEmbedMsg(&buf, allocator, 2, ts.items);

    var m = try loadFromBytes(allocator, buf.items);
    defer m.deinit();

    try std.testing.expectEqual(ModelKind.bpe, m.model_kind);
}
