//! Minimal protobuf wire-format reader/writer used by `ztok grpc-serve`.
//!
//! This is intentionally NOT a general-purpose protobuf codec. It
//! implements only the slice of the wire format that the six gRPC
//! messages below need:
//!
//!   message EncodeRequest  { string text = 1; bytes raw = 2; }
//!   message EncodeResponse { repeated uint32 ids = 1; }
//!   message DecodeRequest  { repeated uint32 ids = 1; }
//!   message DecodeResponse { string text = 1; }
//!   message EvalRequest    { string text = 1; uint64 max_lines = 2; }
//!   message EvalResponse   { uint64 tokens = 1; uint64 bytes = 2; double fertility = 3; }
//!
//! That means we only need:
//!   * varint encode/decode  (wire type 0)
//!   * length-delimited       (wire type 2) for string + bytes + packed
//!   * fixed64 little-endian  (wire type 1) for `double`
//!   * tag = (field << 3) | wire_type
//!
//! We deliberately do NOT implement: groups (deprecated), sint/zigzag
//! (not needed by these messages), fixed32, packed encoding of repeated
//! uint32 fields (we emit one tag per element which is also valid
//! protobuf and what every common runtime accepts on the wire), or
//! unknown-field preservation.
//!
//! Hermetically tested against fixed-byte vectors so any drift from the
//! protobuf spec is caught immediately.

const std = @import("std");

pub const WireType = enum(u3) {
    varint = 0,
    fixed64 = 1,
    len = 2,
    fixed32 = 5,
};

pub const Error = error{
    Truncated,
    BadWireType,
    VarintTooLong,
    LengthOverflow,
    UnexpectedWireType,
};

// === Reader ============================================================

pub const Reader = struct {
    buf: []const u8,
    pos: usize,

    pub fn init(buf: []const u8) Reader {
        return .{ .buf = buf, .pos = 0 };
    }

    pub fn done(self: *const Reader) bool {
        return self.pos >= self.buf.len;
    }

    pub fn readVarint(self: *Reader) Error!u64 {
        var result: u64 = 0;
        var shift: u6 = 0;
        var i: usize = 0;
        while (i < 10) : (i += 1) {
            if (self.pos >= self.buf.len) return Error.Truncated;
            const b = self.buf[self.pos];
            self.pos += 1;
            result |= (@as(u64, b & 0x7f)) << shift;
            if ((b & 0x80) == 0) return result;
            if (i == 9) return Error.VarintTooLong;
            shift += 7;
        }
        return Error.VarintTooLong;
    }

    pub fn readFixed64(self: *Reader) Error!u64 {
        if (self.pos + 8 > self.buf.len) return Error.Truncated;
        const v = std.mem.readInt(u64, self.buf[self.pos..][0..8], .little);
        self.pos += 8;
        return v;
    }

    pub fn readDouble(self: *Reader) Error!f64 {
        const u = try self.readFixed64();
        return @bitCast(u);
    }

    pub fn readLen(self: *Reader) Error![]const u8 {
        const n_u64 = try self.readVarint();
        if (n_u64 > self.buf.len - self.pos) return Error.LengthOverflow;
        const n: usize = @intCast(n_u64);
        const s = self.buf[self.pos .. self.pos + n];
        self.pos += n;
        return s;
    }

    pub const Tag = struct { field: u32, wire: WireType };

    pub fn nextTag(self: *Reader) Error!Tag {
        const t = try self.readVarint();
        const wire_bits: u3 = @intCast(t & 0x7);
        const field_u64 = t >> 3;
        if (field_u64 > std.math.maxInt(u32)) return Error.BadWireType;
        const wt: WireType = switch (wire_bits) {
            0 => .varint,
            1 => .fixed64,
            2 => .len,
            5 => .fixed32,
            else => return Error.BadWireType,
        };
        return .{ .field = @intCast(field_u64), .wire = wt };
    }

    pub fn skip(self: *Reader, wire: WireType) Error!void {
        switch (wire) {
            .varint => _ = try self.readVarint(),
            .fixed64 => _ = try self.readFixed64(),
            .len => _ = try self.readLen(),
            .fixed32 => {
                if (self.pos + 4 > self.buf.len) return Error.Truncated;
                self.pos += 4;
            },
        }
    }
};

// === Writer ============================================================

/// Writes wire-format bytes into a growable byte list. Caller owns the
/// list and is responsible for its allocator.
pub const Writer = struct {
    list: *std.ArrayList(u8),
    allocator: std.mem.Allocator,

    pub fn init(list: *std.ArrayList(u8), allocator: std.mem.Allocator) Writer {
        return .{ .list = list, .allocator = allocator };
    }

    pub fn writeVarint(self: Writer, v: u64) !void {
        var x = v;
        while (x >= 0x80) {
            try self.list.append(self.allocator, @as(u8, @intCast(x & 0x7f)) | 0x80);
            x >>= 7;
        }
        try self.list.append(self.allocator, @intCast(x));
    }

    pub fn writeTag(self: Writer, field: u32, wire: WireType) !void {
        const t: u64 = (@as(u64, field) << 3) | @intFromEnum(wire);
        try self.writeVarint(t);
    }

    pub fn writeFixed64(self: Writer, v: u64) !void {
        var buf: [8]u8 = undefined;
        std.mem.writeInt(u64, &buf, v, .little);
        try self.list.appendSlice(self.allocator, &buf);
    }

    pub fn writeDouble(self: Writer, v: f64) !void {
        const u: u64 = @bitCast(v);
        try self.writeFixed64(u);
    }

    pub fn writeLenPrefixed(self: Writer, bytes: []const u8) !void {
        try self.writeVarint(@intCast(bytes.len));
        try self.list.appendSlice(self.allocator, bytes);
    }

    // -- High-level field helpers ---------------------------------------

    pub fn writeUint64Field(self: Writer, field: u32, v: u64) !void {
        try self.writeTag(field, .varint);
        try self.writeVarint(v);
    }

    pub fn writeUint32Field(self: Writer, field: u32, v: u32) !void {
        try self.writeTag(field, .varint);
        try self.writeVarint(v);
    }

    pub fn writeDoubleField(self: Writer, field: u32, v: f64) !void {
        try self.writeTag(field, .fixed64);
        try self.writeDouble(v);
    }

    pub fn writeStringField(self: Writer, field: u32, s: []const u8) !void {
        try self.writeTag(field, .len);
        try self.writeLenPrefixed(s);
    }

    pub fn writeBytesField(self: Writer, field: u32, s: []const u8) !void {
        try self.writeTag(field, .len);
        try self.writeLenPrefixed(s);
    }
};

// === Message types =====================================================

const TokenId = @import("token.zig").TokenId;

pub const EncodeRequest = struct {
    text: []const u8 = "",
    raw: []const u8 = "",

    /// Decode in place; returned slices borrow from `buf` (zero-copy).
    pub fn decode(buf: []const u8) Error!EncodeRequest {
        var r = Reader.init(buf);
        var out: EncodeRequest = .{};
        while (!r.done()) {
            const tag = try r.nextTag();
            switch (tag.field) {
                1 => {
                    if (tag.wire != .len) return Error.UnexpectedWireType;
                    out.text = try r.readLen();
                },
                2 => {
                    if (tag.wire != .len) return Error.UnexpectedWireType;
                    out.raw = try r.readLen();
                },
                else => try r.skip(tag.wire),
            }
        }
        return out;
    }

    pub fn encode(self: EncodeRequest, list: *std.ArrayList(u8), allocator: std.mem.Allocator) !void {
        const w = Writer.init(list, allocator);
        if (self.text.len > 0) try w.writeStringField(1, self.text);
        if (self.raw.len > 0) try w.writeBytesField(2, self.raw);
    }
};

pub const EncodeResponse = struct {
    ids: []const TokenId = &.{},

    pub fn encode(self: EncodeResponse, list: *std.ArrayList(u8), allocator: std.mem.Allocator) !void {
        const w = Writer.init(list, allocator);
        // Emit one tag per element. Valid protobuf — packed is the
        // canonical proto3 default for repeated uint32, but the
        // unpacked form is universally accepted on decode.
        for (self.ids) |id| {
            try w.writeUint32Field(1, @intCast(id));
        }
    }

    /// Decode into a freshly-allocated slice owned by `allocator`.
    pub fn decode(allocator: std.mem.Allocator, buf: []const u8) !EncodeResponse {
        var list: std.ArrayList(TokenId) = .empty;
        errdefer list.deinit(allocator);
        var r = Reader.init(buf);
        while (!r.done()) {
            const tag = try r.nextTag();
            switch (tag.field) {
                1 => switch (tag.wire) {
                    .varint => {
                        const v = try r.readVarint();
                        try list.append(allocator, @intCast(v));
                    },
                    // Packed form: a single length-delimited block of varints.
                    .len => {
                        const inner = try r.readLen();
                        var sub = Reader.init(inner);
                        while (!sub.done()) {
                            const v = try sub.readVarint();
                            try list.append(allocator, @intCast(v));
                        }
                    },
                    else => return Error.UnexpectedWireType,
                },
                else => try r.skip(tag.wire),
            }
        }
        return .{ .ids = try list.toOwnedSlice(allocator) };
    }
};

pub const DecodeRequest = struct {
    ids: []const TokenId = &.{},

    pub fn decode(allocator: std.mem.Allocator, buf: []const u8) !DecodeRequest {
        // Reuse EncodeResponse's decode — identical wire shape.
        const er = try EncodeResponse.decode(allocator, buf);
        return .{ .ids = er.ids };
    }

    pub fn encode(self: DecodeRequest, list: *std.ArrayList(u8), allocator: std.mem.Allocator) !void {
        const w = Writer.init(list, allocator);
        for (self.ids) |id| {
            try w.writeUint32Field(1, @intCast(id));
        }
    }
};

pub const DecodeResponse = struct {
    text: []const u8 = "",

    pub fn encode(self: DecodeResponse, list: *std.ArrayList(u8), allocator: std.mem.Allocator) !void {
        const w = Writer.init(list, allocator);
        if (self.text.len > 0) try w.writeStringField(1, self.text);
    }

    pub fn decode(buf: []const u8) Error!DecodeResponse {
        var r = Reader.init(buf);
        var out: DecodeResponse = .{};
        while (!r.done()) {
            const tag = try r.nextTag();
            switch (tag.field) {
                1 => {
                    if (tag.wire != .len) return Error.UnexpectedWireType;
                    out.text = try r.readLen();
                },
                else => try r.skip(tag.wire),
            }
        }
        return out;
    }
};

pub const EvalRequest = struct {
    text: []const u8 = "",
    max_lines: u64 = 0,

    pub fn decode(buf: []const u8) Error!EvalRequest {
        var r = Reader.init(buf);
        var out: EvalRequest = .{};
        while (!r.done()) {
            const tag = try r.nextTag();
            switch (tag.field) {
                1 => {
                    if (tag.wire != .len) return Error.UnexpectedWireType;
                    out.text = try r.readLen();
                },
                2 => {
                    if (tag.wire != .varint) return Error.UnexpectedWireType;
                    out.max_lines = try r.readVarint();
                },
                else => try r.skip(tag.wire),
            }
        }
        return out;
    }

    pub fn encode(self: EvalRequest, list: *std.ArrayList(u8), allocator: std.mem.Allocator) !void {
        const w = Writer.init(list, allocator);
        if (self.text.len > 0) try w.writeStringField(1, self.text);
        if (self.max_lines != 0) try w.writeUint64Field(2, self.max_lines);
    }
};

pub const EvalResponse = struct {
    tokens: u64 = 0,
    bytes: u64 = 0,
    fertility: f64 = 0,

    pub fn encode(self: EvalResponse, list: *std.ArrayList(u8), allocator: std.mem.Allocator) !void {
        const w = Writer.init(list, allocator);
        if (self.tokens != 0) try w.writeUint64Field(1, self.tokens);
        if (self.bytes != 0) try w.writeUint64Field(2, self.bytes);
        if (self.fertility != 0) try w.writeDoubleField(3, self.fertility);
    }

    pub fn decode(buf: []const u8) Error!EvalResponse {
        var r = Reader.init(buf);
        var out: EvalResponse = .{};
        while (!r.done()) {
            const tag = try r.nextTag();
            switch (tag.field) {
                1 => {
                    if (tag.wire != .varint) return Error.UnexpectedWireType;
                    out.tokens = try r.readVarint();
                },
                2 => {
                    if (tag.wire != .varint) return Error.UnexpectedWireType;
                    out.bytes = try r.readVarint();
                },
                3 => {
                    if (tag.wire != .fixed64) return Error.UnexpectedWireType;
                    out.fertility = try r.readDouble();
                },
                else => try r.skip(tag.wire),
            }
        }
        return out;
    }
};

// === Tests =============================================================

const testing = std.testing;

test "varint round-trip 150" {
    const a = testing.allocator;
    var list: std.ArrayList(u8) = .empty;
    defer list.deinit(a);
    const w = Writer.init(&list, a);
    try w.writeVarint(150);
    // 150 = 0x96, 0x01
    try testing.expectEqualSlices(u8, &.{ 0x96, 0x01 }, list.items);

    var r = Reader.init(list.items);
    try testing.expectEqual(@as(u64, 150), try r.readVarint());
    try testing.expect(r.done());
}

test "tag encode + decode" {
    const a = testing.allocator;
    var list: std.ArrayList(u8) = .empty;
    defer list.deinit(a);
    const w = Writer.init(&list, a);
    try w.writeTag(1, .varint);
    // (1<<3)|0 = 8
    try testing.expectEqualSlices(u8, &.{0x08}, list.items);

    var r = Reader.init(list.items);
    const t = try r.nextTag();
    try testing.expectEqual(@as(u32, 1), t.field);
    try testing.expectEqual(WireType.varint, t.wire);
}

test "EncodeRequest decodes hand-rolled bytes for 'hello'" {
    // field=1 (text) wire=len, "hello"
    //   tag = (1<<3)|2 = 0x0A
    //   len = 5
    //   payload = h e l l o
    const buf = [_]u8{ 0x0A, 0x05, 'h', 'e', 'l', 'l', 'o' };
    const decoded = try EncodeRequest.decode(&buf);
    try testing.expectEqualStrings("hello", decoded.text);
    try testing.expectEqualStrings("", decoded.raw);
}

test "EncodeRequest round-trip with text + raw" {
    const a = testing.allocator;
    var list: std.ArrayList(u8) = .empty;
    defer list.deinit(a);
    const req: EncodeRequest = .{ .text = "hi", .raw = "\x00\x01\x02" };
    try req.encode(&list, a);

    const decoded = try EncodeRequest.decode(list.items);
    try testing.expectEqualStrings("hi", decoded.text);
    try testing.expectEqualSlices(u8, &.{ 0, 1, 2 }, decoded.raw);
}

test "EncodeResponse round-trip ids" {
    const a = testing.allocator;
    var list: std.ArrayList(u8) = .empty;
    defer list.deinit(a);
    const resp: EncodeResponse = .{ .ids = &.{ 7, 11, 13, 300 } };
    try resp.encode(&list, a);

    // Decode and check
    const decoded = try EncodeResponse.decode(a, list.items);
    defer a.free(decoded.ids);
    try testing.expectEqualSlices(TokenId, &.{ 7, 11, 13, 300 }, decoded.ids);
}

test "EncodeResponse decodes packed form" {
    const a = testing.allocator;
    // Packed: tag = (1<<3)|2 = 0x0A, then a length-delimited block of varints.
    // Varints for {7, 11, 13}: 0x07, 0x0B, 0x0D
    const buf = [_]u8{ 0x0A, 0x03, 0x07, 0x0B, 0x0D };
    const decoded = try EncodeResponse.decode(a, &buf);
    defer a.free(decoded.ids);
    try testing.expectEqualSlices(TokenId, &.{ 7, 11, 13 }, decoded.ids);
}

test "DecodeRequest round-trip ids" {
    const a = testing.allocator;
    var list: std.ArrayList(u8) = .empty;
    defer list.deinit(a);
    const req: DecodeRequest = .{ .ids = &.{ 42, 1, 65535 } };
    try req.encode(&list, a);

    const decoded = try DecodeRequest.decode(a, list.items);
    defer a.free(decoded.ids);
    try testing.expectEqualSlices(TokenId, &.{ 42, 1, 65535 }, decoded.ids);
}

test "DecodeResponse round-trip text" {
    const a = testing.allocator;
    var list: std.ArrayList(u8) = .empty;
    defer list.deinit(a);
    const resp: DecodeResponse = .{ .text = "ok then" };
    try resp.encode(&list, a);
    const decoded = try DecodeResponse.decode(list.items);
    try testing.expectEqualStrings("ok then", decoded.text);
}

test "EvalRequest round-trip with max_lines" {
    const a = testing.allocator;
    var list: std.ArrayList(u8) = .empty;
    defer list.deinit(a);
    const req: EvalRequest = .{ .text = "abc", .max_lines = 1000 };
    try req.encode(&list, a);
    const decoded = try EvalRequest.decode(list.items);
    try testing.expectEqualStrings("abc", decoded.text);
    try testing.expectEqual(@as(u64, 1000), decoded.max_lines);
}

test "EvalRequest decoded ignores absent fields" {
    // Empty buffer is a valid (all-defaults) proto message.
    const decoded = try EvalRequest.decode("");
    try testing.expectEqualStrings("", decoded.text);
    try testing.expectEqual(@as(u64, 0), decoded.max_lines);
}

test "EvalResponse round-trip with double fertility" {
    const a = testing.allocator;
    var list: std.ArrayList(u8) = .empty;
    defer list.deinit(a);
    const resp: EvalResponse = .{ .tokens = 5, .bytes = 11, .fertility = 2.2 };
    try resp.encode(&list, a);
    const decoded = try EvalResponse.decode(list.items);
    try testing.expectEqual(@as(u64, 5), decoded.tokens);
    try testing.expectEqual(@as(u64, 11), decoded.bytes);
    try testing.expectApproxEqAbs(@as(f64, 2.2), decoded.fertility, 1e-12);
}

test "skip unknown field types" {
    const a = testing.allocator;
    var list: std.ArrayList(u8) = .empty;
    defer list.deinit(a);
    const w = Writer.init(&list, a);
    // field=999 wire=varint, value=42
    try w.writeTag(999, .varint);
    try w.writeVarint(42);
    // then field=1 wire=len, "x" (a real EncodeRequest text field)
    try w.writeStringField(1, "x");

    const decoded = try EncodeRequest.decode(list.items);
    try testing.expectEqualStrings("x", decoded.text);
}

// === gRPC-Web frame helpers ===========================================
//
// gRPC-Web frame format (https://github.com/grpc/grpc-web/blob/master/doc/binary-format.md):
//   1 byte:  flag (0x00 = data, 0x80 = trailers)
//   4 bytes: big-endian length of the payload
//   N bytes: payload (protobuf bytes for data frames; ASCII headers
//            terminated with \r\n for trailer frames)

pub const FrameFlag = enum(u8) {
    data = 0x00,
    trailers = 0x80,
};

/// Write a single gRPC-Web frame (flag + 4-byte BE length + payload) to
/// `list`. Caller owns the list.
pub fn writeFrame(list: *std.ArrayList(u8), allocator: std.mem.Allocator, flag: FrameFlag, payload: []const u8) !void {
    try list.append(allocator, @intFromEnum(flag));
    var len_buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &len_buf, @intCast(payload.len), .big);
    try list.appendSlice(allocator, &len_buf);
    try list.appendSlice(allocator, payload);
}

pub const ParsedFrame = struct {
    flag: u8,
    payload: []const u8,
    /// Number of input bytes consumed (5 + payload.len).
    consumed: usize,
};

/// Parse one gRPC-Web frame from the front of `buf`. Returns the flag
/// byte raw (so callers can distinguish 0x80 + compression bits if
/// needed) and a slice into `buf` for the payload.
pub fn readFrame(buf: []const u8) Error!ParsedFrame {
    if (buf.len < 5) return Error.Truncated;
    const flag = buf[0];
    const len = std.mem.readInt(u32, buf[1..5], .big);
    const total: usize = 5 + @as(usize, len);
    if (buf.len < total) return Error.Truncated;
    return .{ .flag = flag, .payload = buf[5..total], .consumed = total };
}

test "writeFrame produces flag + 4-byte big-endian length + payload" {
    const a = testing.allocator;
    var list: std.ArrayList(u8) = .empty;
    defer list.deinit(a);
    try writeFrame(&list, a, .data, "abc");
    try testing.expectEqualSlices(u8, &.{ 0x00, 0x00, 0x00, 0x00, 0x03, 'a', 'b', 'c' }, list.items);
}

test "readFrame parses a data frame" {
    const buf = [_]u8{ 0x00, 0x00, 0x00, 0x00, 0x05, 'h', 'e', 'l', 'l', 'o' };
    const frame = try readFrame(&buf);
    try testing.expectEqual(@as(u8, 0x00), frame.flag);
    try testing.expectEqualStrings("hello", frame.payload);
    try testing.expectEqual(@as(usize, 10), frame.consumed);
}

test "readFrame on short buffer returns Truncated" {
    const buf = [_]u8{ 0x00, 0x00, 0x00, 0x00, 0x05, 'h', 'i' };
    try testing.expectError(Error.Truncated, readFrame(&buf));
}

/// Helper to build a gRPC-Web trailer payload like
/// "grpc-status: 0\r\ngrpc-message: \r\n". Caller owns the returned slice.
pub fn buildTrailerPayload(allocator: std.mem.Allocator, grpc_status: u32, grpc_message: []const u8) ![]u8 {
    var list: std.ArrayList(u8) = .empty;
    errdefer list.deinit(allocator);
    var stat_buf: [16]u8 = undefined;
    const stat_str = try std.fmt.bufPrint(&stat_buf, "{d}", .{grpc_status});
    try list.appendSlice(allocator, "grpc-status: ");
    try list.appendSlice(allocator, stat_str);
    try list.appendSlice(allocator, "\r\ngrpc-message: ");
    try list.appendSlice(allocator, grpc_message);
    try list.appendSlice(allocator, "\r\n");
    return list.toOwnedSlice(allocator);
}

test "buildTrailerPayload formats trailers" {
    const a = testing.allocator;
    const t = try buildTrailerPayload(a, 0, "");
    defer a.free(t);
    try testing.expectEqualStrings("grpc-status: 0\r\ngrpc-message: \r\n", t);
}
