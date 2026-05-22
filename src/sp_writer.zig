const std = @import("std");
const proto = @import("proto.zig");
const TokenId = @import("token.zig").TokenId;
const Unigram = @import("unigram.zig").Unigram;
const Bpe = @import("bpe.zig").Bpe;
const sp_model = @import("sp_model.zig");
const SpModel = sp_model.SpModel;
const PieceType = sp_model.PieceType;
const ModelKind = sp_model.ModelKind;

pub const WriteOptions = struct {
    normalizer_name: []const u8 = "identity",
    add_dummy_prefix: bool = false,
    remove_extra_whitespaces: bool = false,
    escape_whitespaces: bool = false,
    piece_types: ?[]const PieceType = null,
    unk_id: ?TokenId = null,
};

const ALLOC = std.mem.Allocator;
const W = std.ArrayList(u8);

fn writeVarint(out: *W, alloc: ALLOC, v: u64) !void {
    var x = v;
    while (x >= 0x80) {
        try out.append(alloc, @as(u8, @intCast(x & 0x7f)) | 0x80);
        x >>= 7;
    }
    try out.append(alloc, @intCast(x));
}

fn writeTag(out: *W, alloc: ALLOC, field: u32, wire: proto.WireType) !void {
    const w: u3 = @intFromEnum(wire);
    try writeVarint(out, alloc, (@as(u64, field) << 3) | w);
}

fn writeFloat(out: *W, alloc: ALLOC, field: u32, v: f32) !void {
    try writeTag(out, alloc, field, .i32);
    var b: [4]u8 = undefined;
    const u: u32 = @bitCast(v);
    std.mem.writeInt(u32, &b, u, .little);
    try out.appendSlice(alloc, &b);
}

fn writeString(out: *W, alloc: ALLOC, field: u32, s: []const u8) !void {
    try writeTag(out, alloc, field, .len);
    try writeVarint(out, alloc, s.len);
    try out.appendSlice(alloc, s);
}

fn writeEnum(out: *W, alloc: ALLOC, field: u32, v: u32) !void {
    try writeTag(out, alloc, field, .varint);
    try writeVarint(out, alloc, v);
}

fn writeBool(out: *W, alloc: ALLOC, field: u32, v: bool) !void {
    try writeTag(out, alloc, field, .varint);
    try writeVarint(out, alloc, if (v) 1 else 0);
}

fn writeMessage(out: *W, alloc: ALLOC, field: u32, body: []const u8) !void {
    try writeTag(out, alloc, field, .len);
    try writeVarint(out, alloc, body.len);
    try out.appendSlice(alloc, body);
}

fn pieceTypeEnum(t: PieceType) u32 {
    return switch (t) {
        .normal => 1,
        .unknown => 2,
        .control => 3,
        .user_defined => 4,
        .unused => 5,
        .byte => 6,
    };
}

fn modelKindEnum(k: ModelKind) u32 {
    return switch (k) {
        .unigram => 1,
        .bpe => 2,
        .word => 3,
        .char => 4,
    };
}

fn appendPiece(
    out: *W,
    alloc: ALLOC,
    piece_bytes: []const u8,
    score: f32,
    ty: ?PieceType,
) !void {
    var body: W = .empty;
    defer body.deinit(alloc);
    try writeString(&body, alloc, 1, piece_bytes);
    try writeFloat(&body, alloc, 2, score);
    if (ty) |t| {
        if (t != .normal) try writeEnum(&body, alloc, 3, pieceTypeEnum(t));
    }
    try writeMessage(out, alloc, 1, body.items);
}

fn appendTrainerSpec(out: *W, alloc: ALLOC, kind: ModelKind) !void {
    var body: W = .empty;
    defer body.deinit(alloc);
    try writeEnum(&body, alloc, 3, modelKindEnum(kind));
    try writeMessage(out, alloc, 2, body.items);
}

fn appendNormalizerSpec(out: *W, alloc: ALLOC, opts: WriteOptions) !void {
    var body: W = .empty;
    defer body.deinit(alloc);
    try writeString(&body, alloc, 1, opts.normalizer_name);
    try writeBool(&body, alloc, 3, opts.add_dummy_prefix);
    try writeBool(&body, alloc, 4, opts.remove_extra_whitespaces);
    try writeBool(&body, alloc, 5, opts.escape_whitespaces);
    try writeMessage(out, alloc, 3, body.items);
}

fn appendNormalizerSpecFromModel(out: *W, alloc: ALLOC, sp: *const SpModel) !void {
    var body: W = .empty;
    defer body.deinit(alloc);
    if (sp.normalizer_name) |n| try writeString(&body, alloc, 1, n);
    try writeBool(&body, alloc, 3, sp.add_dummy_prefix);
    try writeBool(&body, alloc, 4, sp.remove_extra_whitespaces);
    try writeBool(&body, alloc, 5, sp.escape_whitespaces);
    try writeMessage(out, alloc, 3, body.items);
}

pub fn writeUnigram(allocator: ALLOC, u: *const Unigram, opts: WriteOptions) ![]u8 {
    var out: W = .empty;
    errdefer out.deinit(allocator);

    var i: u32 = 0;
    while (i < u.count) : (i += 1) {
        var ty: ?PieceType = null;
        if (opts.unk_id) |uid| {
            if (uid == i) ty = .unknown;
        }
        if (ty == null) {
            if (opts.piece_types) |pt| {
                if (i < pt.len) ty = pt[i];
            }
        }
        try appendPiece(&out, allocator, u.idBytes(i), u.scores[i], ty);
    }

    try appendTrainerSpec(&out, allocator, .unigram);
    try appendNormalizerSpec(&out, allocator, opts);

    return out.toOwnedSlice(allocator);
}

pub fn writeBpe(allocator: ALLOC, bpe: *const Bpe, opts: WriteOptions) ![]u8 {
    var out: W = .empty;
    errdefer out.deinit(allocator);

    var i: u32 = 0;
    while (i < bpe.count) : (i += 1) {
        // SP-BPE convention: byte pieces (ids 0..255) score 0; non-bytes
        // get -(id - 256) so higher merge rank => more negative score.
        const score: f32 = if (i < 256) 0.0 else -@as(f32, @floatFromInt(i - 256));

        var ty: ?PieceType = null;
        if (i < 256) ty = .byte;
        if (opts.unk_id) |uid| {
            if (uid == i) ty = .unknown;
        }
        if (ty == null) {
            if (opts.piece_types) |pt| {
                if (i < pt.len) ty = pt[i];
            }
        }

        try appendPiece(&out, allocator, bpe.idBytes(i), score, ty);
    }

    try appendTrainerSpec(&out, allocator, .bpe);
    try appendNormalizerSpec(&out, allocator, opts);

    return out.toOwnedSlice(allocator);
}

pub fn writeSpModel(allocator: ALLOC, sp: *const SpModel) ![]u8 {
    var out: W = .empty;
    errdefer out.deinit(allocator);

    var i: u32 = 0;
    while (i < sp.count) : (i += 1) {
        try appendPiece(&out, allocator, sp.pieceBytes(i), sp.scores[i], sp.types[i]);
    }
    try appendTrainerSpec(&out, allocator, sp.model_kind);
    try appendNormalizerSpecFromModel(&out, allocator, sp);

    return out.toOwnedSlice(allocator);
}

fn writeBytesToPath(path: []const u8, buf: []const u8) !void {
    const io = std.Io.Threaded.global_single_threaded.io();
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = buf });
}

pub fn writeUnigramFile(allocator: ALLOC, u: *const Unigram, path: []const u8, opts: WriteOptions) !void {
    const buf = try writeUnigram(allocator, u, opts);
    defer allocator.free(buf);
    try writeBytesToPath(path, buf);
}

pub fn writeBpeFile(allocator: ALLOC, bpe: *const Bpe, path: []const u8, opts: WriteOptions) !void {
    const buf = try writeBpe(allocator, bpe, opts);
    defer allocator.free(buf);
    try writeBytesToPath(path, buf);
}

pub fn writeSpModelFile(allocator: ALLOC, sp: *const SpModel, path: []const u8) !void {
    const buf = try writeSpModel(allocator, sp);
    defer allocator.free(buf);
    try writeBytesToPath(path, buf);
}

// --- tests -----------------------------------------------------------------

test "writeUnigram round-trips via sp_model" {
    const alloc = std.testing.allocator;
    var b = Unigram.Builder.init(alloc);
    defer b.deinit();
    _ = try b.addToken("foo", -1.5);
    _ = try b.addToken("bar", -2.25);
    const unk = try b.addToken("<unk>", -10.0);
    var u = try b.finalize(unk);
    defer u.deinit();

    const bytes = try writeUnigram(alloc, &u, .{});
    defer alloc.free(bytes);

    var m = try sp_model.loadFromBytes(alloc, bytes);
    defer m.deinit();

    try std.testing.expectEqual(@as(u32, 3), m.count);
    try std.testing.expectEqualStrings("foo", m.pieceBytes(0));
    try std.testing.expectEqualStrings("bar", m.pieceBytes(1));
    try std.testing.expectEqualStrings("<unk>", m.pieceBytes(2));
    try std.testing.expectEqual(@as(f32, -1.5), m.scores[0]);
    try std.testing.expectEqual(@as(f32, -2.25), m.scores[1]);
    try std.testing.expectEqual(@as(f32, -10.0), m.scores[2]);
    try std.testing.expectEqual(ModelKind.unigram, m.model_kind);
}

test "writeBpe round-trips via sp_model with model_kind == .bpe" {
    const alloc = std.testing.allocator;
    // Build a tiny BPE: 256 byte tokens + 2 merges.
    const TestEntry = struct { bytes: []const u8, rank: u32 };
    var entries: std.ArrayList(TestEntry) = .empty;
    defer entries.deinit(alloc);

    var byte_holders: [256][1]u8 = undefined;
    var i: u32 = 0;
    while (i < 256) : (i += 1) {
        byte_holders[i][0] = @intCast(i);
        try entries.append(alloc, .{ .bytes = byte_holders[i][0..1], .rank = i });
    }
    try entries.append(alloc, .{ .bytes = "ab", .rank = 256 });
    try entries.append(alloc, .{ .bytes = "abc", .rank = 257 });

    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(alloc);
    const enc = std.base64.standard.Encoder;
    for (entries.items) |e| {
        const sz = enc.calcSize(e.bytes.len);
        const tmp = try alloc.alloc(u8, sz);
        defer alloc.free(tmp);
        const out = enc.encode(tmp, e.bytes);
        try src.appendSlice(alloc, out);
        try src.print(alloc, " {d}\n", .{e.rank});
    }
    var bpe = try Bpe.loadTiktokenBytes(alloc, src.items);
    defer bpe.deinit();

    const bytes = try writeBpe(alloc, &bpe, .{});
    defer alloc.free(bytes);

    var m = try sp_model.loadFromBytes(alloc, bytes);
    defer m.deinit();

    try std.testing.expectEqual(ModelKind.bpe, m.model_kind);
    try std.testing.expectEqual(@as(u32, 258), m.count);
    // First byte piece
    try std.testing.expectEqualStrings(bpe.idBytes(0), m.pieceBytes(0));
    try std.testing.expectEqualStrings("ab", m.pieceBytes(256));
    try std.testing.expectEqualStrings("abc", m.pieceBytes(257));
    // Byte pieces have type .byte, score 0
    try std.testing.expectEqual(PieceType.byte, m.types[0]);
    try std.testing.expectEqual(@as(f32, 0.0), m.scores[0]);
    // Merge pieces: score = -(id - 256)
    try std.testing.expectEqual(@as(f32, 0.0), m.scores[256]);
    try std.testing.expectEqual(@as(f32, -1.0), m.scores[257]);
}

test "writeSpModel reproduces input bytes" {
    const alloc = std.testing.allocator;

    // Hand-build a minimal SP model: 2 pieces + normalizer.
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(alloc);

    var p0: W = .empty;
    defer p0.deinit(alloc);
    try writeString(&p0, alloc, 1, "hi");
    try writeFloat(&p0, alloc, 2, -0.5);
    try writeMessage(&buf, alloc, 1, p0.items);

    var p1: W = .empty;
    defer p1.deinit(alloc);
    try writeString(&p1, alloc, 1, "<unk>");
    try writeFloat(&p1, alloc, 2, -9.0);
    try writeEnum(&p1, alloc, 3, 2); // UNKNOWN
    try writeMessage(&buf, alloc, 1, p1.items);

    // trainer_spec model_type = UNIGRAM
    var ts: W = .empty;
    defer ts.deinit(alloc);
    try writeEnum(&ts, alloc, 3, 1);
    try writeMessage(&buf, alloc, 2, ts.items);

    // normalizer_spec
    var ns: W = .empty;
    defer ns.deinit(alloc);
    try writeString(&ns, alloc, 1, "nmt_nfkc");
    try writeBool(&ns, alloc, 3, false);
    try writeBool(&ns, alloc, 4, true);
    try writeBool(&ns, alloc, 5, true);
    try writeMessage(&buf, alloc, 3, ns.items);

    var m = try sp_model.loadFromBytes(alloc, buf.items);
    defer m.deinit();

    const round = try writeSpModel(alloc, &m);
    defer alloc.free(round);

    // Structural round-trip: reload the written bytes and compare fields.
    var m2 = try sp_model.loadFromBytes(alloc, round);
    defer m2.deinit();

    try std.testing.expectEqual(m.count, m2.count);
    var k: u32 = 0;
    while (k < m.count) : (k += 1) {
        try std.testing.expectEqualStrings(m.pieceBytes(k), m2.pieceBytes(k));
        try std.testing.expectEqual(m.scores[k], m2.scores[k]);
        try std.testing.expectEqual(m.types[k], m2.types[k]);
    }
    try std.testing.expectEqual(m.model_kind, m2.model_kind);
    try std.testing.expect(m2.normalizer_name != null);
    try std.testing.expectEqualStrings(m.normalizer_name.?, m2.normalizer_name.?);
    try std.testing.expectEqual(m.add_dummy_prefix, m2.add_dummy_prefix);
    try std.testing.expectEqual(m.remove_extra_whitespaces, m2.remove_extra_whitespaces);
    try std.testing.expectEqual(m.escape_whitespaces, m2.escape_whitespaces);
}

test "varint encode + decode round-trip" {
    const alloc = std.testing.allocator;
    const cases = [_]u64{ 0, 1, 127, 128, 255, 16383, 16384, 0xFFFFFFFF, std.math.maxInt(u64) };
    for (cases) |v| {
        var out: W = .empty;
        defer out.deinit(alloc);
        try writeVarint(&out, alloc, v);
        var r = proto.Reader.init(out.items);
        const got = try r.readVarint();
        try std.testing.expectEqual(v, got);
        try std.testing.expect(r.done());
    }
}

test "float encode + decode round-trip" {
    const alloc = std.testing.allocator;
    const cases = [_]f32{
        0.0,
        -0.0,
        1.5,
        -3.14159,
        std.math.inf(f32),
        -std.math.inf(f32),
        std.math.nan(f32),
    };
    for (cases) |v| {
        var out: W = .empty;
        defer out.deinit(alloc);
        try writeFloat(&out, alloc, 7, v);

        var r = proto.Reader.init(out.items);
        const tag = try r.nextTag();
        try std.testing.expectEqual(@as(u32, 7), tag.field);
        try std.testing.expectEqual(proto.WireType.i32, tag.wire);
        const got = try r.readFloat();
        if (std.math.isNan(v)) {
            try std.testing.expect(std.math.isNan(got));
        } else {
            try std.testing.expectEqual(v, got);
            // -0.0 vs 0.0: bit-exact check
            const bv: u32 = @bitCast(v);
            const bg: u32 = @bitCast(got);
            try std.testing.expectEqual(bv, bg);
        }
        try std.testing.expect(r.done());
    }
}

test "writeUnigram with unk_id marks UNKNOWN type" {
    const alloc = std.testing.allocator;
    var b = Unigram.Builder.init(alloc);
    defer b.deinit();
    _ = try b.addToken("a", -1.0);
    _ = try b.addToken("b", -1.0);
    const unk = try b.addToken("<unk>", -10.0);
    var u = try b.finalize(unk);
    defer u.deinit();

    const bytes = try writeUnigram(alloc, &u, .{ .unk_id = unk });
    defer alloc.free(bytes);

    var m = try sp_model.loadFromBytes(alloc, bytes);
    defer m.deinit();

    try std.testing.expectEqual(PieceType.normal, m.types[0]);
    try std.testing.expectEqual(PieceType.normal, m.types[1]);
    try std.testing.expectEqual(PieceType.unknown, m.types[unk]);
}

test "writeUnigramFile + loadFromFile end-to-end" {
    const alloc = std.testing.allocator;
    var b = Unigram.Builder.init(alloc);
    defer b.deinit();
    _ = try b.addToken("alpha", -1.25);
    _ = try b.addToken("beta", -2.5);
    const unk = try b.addToken("<unk>", -8.0);
    var u = try b.finalize(unk);
    defer u.deinit();

    // Build a unique path in /tmp.
    var rng = std.Random.DefaultPrng.init(0xC0FFEE_DEADBEEF);
    const r = rng.random().int(u64);
    var path_buf: [64]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "/tmp/ztok_sp_writer_test_{x}.model", .{r});

    try writeUnigramFile(alloc, &u, path, .{ .unk_id = unk });
    const io = std.Io.Threaded.global_single_threaded.io();
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    var m = try sp_model.loadFromFile(alloc, path);
    defer m.deinit();

    try std.testing.expectEqual(@as(u32, 3), m.count);
    try std.testing.expectEqualStrings("alpha", m.pieceBytes(0));
    try std.testing.expectEqualStrings("beta", m.pieceBytes(1));
    try std.testing.expectEqualStrings("<unk>", m.pieceBytes(2));
    try std.testing.expectEqual(@as(f32, -1.25), m.scores[0]);
    try std.testing.expectEqual(PieceType.unknown, m.types[unk]);
}
