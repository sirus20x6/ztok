//! Minimal proto3/proto2 wire-format reader. Streaming, zero-alloc.
//! Decodes only what sentencepiece_model.proto needs.

const std = @import("std");

pub const WireType = enum(u3) {
    varint = 0,
    i64 = 1,
    len = 2,
    i32 = 5,
};

pub const Error = error{
    Truncated,
    BadWireType,
    VarintTooLong,
    LengthOverflow,
};

pub const Tag = struct { field: u32, wire: WireType };

pub const Reader = struct {
    buf: []const u8,
    pos: usize,

    pub fn init(buf: []const u8) Reader {
        return .{ .buf = buf, .pos = 0 };
    }

    pub fn done(self: *const Reader) bool {
        return self.pos >= self.buf.len;
    }

    fn readByte(self: *Reader) Error!u8 {
        if (self.pos >= self.buf.len) return Error.Truncated;
        const b = self.buf[self.pos];
        self.pos += 1;
        return b;
    }

    fn readRaw(self: *Reader, n: usize) Error![]const u8 {
        if (self.pos + n > self.buf.len) return Error.Truncated;
        const s = self.buf[self.pos .. self.pos + n];
        self.pos += n;
        return s;
    }

    pub fn readVarint(self: *Reader) Error!u64 {
        var result: u64 = 0;
        var shift: u6 = 0;
        var i: usize = 0;
        while (i < 10) : (i += 1) {
            const b = try self.readByte();
            // Last (10th) byte: only low bit of payload is meaningful for u64.
            result |= (@as(u64, b & 0x7f)) << shift;
            if ((b & 0x80) == 0) return result;
            if (i == 9) return Error.VarintTooLong;
            shift += 7;
        }
        return Error.VarintTooLong;
    }

    pub fn readSint(self: *Reader) Error!i64 {
        const u = try self.readVarint();
        // ZigZag decode: (u >> 1) ^ -(u & 1)
        const lo: i64 = @bitCast(u >> 1);
        const sign: i64 = -@as(i64, @bitCast(u & 1));
        return lo ^ sign;
    }

    pub fn nextTag(self: *Reader) Error!Tag {
        const t = try self.readVarint();
        const field_u64 = t >> 3;
        if (field_u64 > std.math.maxInt(u32)) return Error.BadWireType;
        const wire_bits: u3 = @intCast(t & 0x7);
        return switch (wire_bits) {
            0 => .{ .field = @intCast(field_u64), .wire = .varint },
            1 => .{ .field = @intCast(field_u64), .wire = .i64 },
            2 => .{ .field = @intCast(field_u64), .wire = .len },
            5 => .{ .field = @intCast(field_u64), .wire = .i32 },
            else => Error.BadWireType,
        };
    }

    pub fn readFloat(self: *Reader) Error!f32 {
        const s = try self.readRaw(4);
        const u: u32 = std.mem.readInt(u32, s[0..4], .little);
        return @bitCast(u);
    }

    pub fn readDouble(self: *Reader) Error!f64 {
        const s = try self.readRaw(8);
        const u: u64 = std.mem.readInt(u64, s[0..8], .little);
        return @bitCast(u);
    }

    pub fn readLen(self: *Reader) Error![]const u8 {
        const n_u64 = try self.readVarint();
        if (n_u64 > self.buf.len) return Error.LengthOverflow;
        const n: usize = @intCast(n_u64);
        return try self.readRaw(n);
    }

    pub fn readMessage(self: *Reader) Error!Reader {
        const s = try self.readLen();
        return Reader.init(s);
    }

    pub fn skip(self: *Reader, wire: WireType) Error!void {
        switch (wire) {
            .varint => _ = try self.readVarint(),
            .i64 => _ = try self.readRaw(8),
            .len => _ = try self.readLen(),
            .i32 => _ = try self.readRaw(4),
        }
    }
};

// --- tests -----------------------------------------------------------------

fn encodeVarint(out: *std.ArrayList(u8), allocator: std.mem.Allocator, v: u64) !void {
    var x = v;
    while (x >= 0x80) {
        try out.append(allocator, @as(u8, @intCast(x & 0x7f)) | 0x80);
        x >>= 7;
    }
    try out.append(allocator, @intCast(x));
}

test "varint encode 150 then decode" {
    const buf = [_]u8{ 0x96, 0x01 };
    var r = Reader.init(&buf);
    try std.testing.expectEqual(@as(u64, 150), try r.readVarint());
    try std.testing.expect(r.done());
}

test "tag decode" {
    const buf = [_]u8{0x08};
    var r = Reader.init(&buf);
    const t = try r.nextTag();
    try std.testing.expectEqual(@as(u32, 1), t.field);
    try std.testing.expectEqual(WireType.varint, t.wire);
}

test "skip varint" {
    // field=99 wire=varint -> tag = (99<<3)|0 = 792 -> varint: 0x98 0x06
    // value 300 -> varint 0xAC 0x02
    // followed by field=1 wire=varint -> 0x08, value 42 -> 0x2A
    const buf = [_]u8{ 0x98, 0x06, 0xAC, 0x02, 0x08, 0x2A };
    var r = Reader.init(&buf);
    const t1 = try r.nextTag();
    try std.testing.expectEqual(@as(u32, 99), t1.field);
    try r.skip(t1.wire);
    const t2 = try r.nextTag();
    try std.testing.expectEqual(@as(u32, 1), t2.field);
    try std.testing.expectEqual(@as(u64, 42), try r.readVarint());
    try std.testing.expect(r.done());
}

test "readMessage sub-reader" {
    const allocator = std.testing.allocator;
    var inner: std.ArrayList(u8) = .empty;
    defer inner.deinit(allocator);
    // inner: field=1 wire=varint, value=7
    try inner.append(allocator, 0x08);
    try inner.append(allocator, 0x07);
    // inner: field=2 wire=len, value="hi"
    try inner.append(allocator, 0x12);
    try inner.append(allocator, 0x02);
    try inner.appendSlice(allocator, "hi");

    var outer: std.ArrayList(u8) = .empty;
    defer outer.deinit(allocator);
    // outer: field=5 wire=len wrapping inner
    try outer.append(allocator, (5 << 3) | 2);
    try encodeVarint(&outer, allocator, inner.items.len);
    try outer.appendSlice(allocator, inner.items);

    var r = Reader.init(outer.items);
    const t = try r.nextTag();
    try std.testing.expectEqual(@as(u32, 5), t.field);
    try std.testing.expectEqual(WireType.len, t.wire);
    var sub = try r.readMessage();

    const it1 = try sub.nextTag();
    try std.testing.expectEqual(@as(u32, 1), it1.field);
    try std.testing.expectEqual(@as(u64, 7), try sub.readVarint());

    const it2 = try sub.nextTag();
    try std.testing.expectEqual(@as(u32, 2), it2.field);
    try std.testing.expectEqualStrings("hi", try sub.readLen());

    try std.testing.expect(sub.done());
    try std.testing.expect(r.done());
}

test "zigzag decode" {
    // encode 0 -> 0, -1 -> 1, 1 -> 2, -2 -> 3
    const cases = [_]struct { enc: u64, dec: i64 }{
        .{ .enc = 0, .dec = 0 },
        .{ .enc = 1, .dec = -1 },
        .{ .enc = 2, .dec = 1 },
        .{ .enc = 3, .dec = -2 },
        .{ .enc = 4294967294, .dec = 2147483647 },
    };
    for (cases) |c| {
        var ab: std.ArrayList(u8) = .empty;
        defer ab.deinit(std.testing.allocator);
        try encodeVarint(&ab, std.testing.allocator, c.enc);
        var r = Reader.init(ab.items);
        try std.testing.expectEqual(c.dec, try r.readSint());
    }
}

test "readFloat / readDouble" {
    const f: f32 = -1.5;
    const fu: u32 = @bitCast(f);
    var fb: [4]u8 = undefined;
    std.mem.writeInt(u32, &fb, fu, .little);
    var rf = Reader.init(&fb);
    try std.testing.expectEqual(f, try rf.readFloat());

    const d: f64 = 3.14159265358979;
    const du: u64 = @bitCast(d);
    var db: [8]u8 = undefined;
    std.mem.writeInt(u64, &db, du, .little);
    var rd = Reader.init(&db);
    try std.testing.expectEqual(d, try rd.readDouble());
}
