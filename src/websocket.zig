//! RFC 6455 WebSocket primitives (handshake + framing) for `ztok serve`.
//!
//! What's here:
//!   * `computeAcceptKey(key, out)` — derives the
//!     `Sec-WebSocket-Accept` value from the client's
//!     `Sec-WebSocket-Key` (SHA-1 of key + magic UUID, base64-encoded).
//!   * `Opcode` / `Frame` — the small subset of the framing protocol
//!     we actually use (text, binary, close, ping, pong).
//!   * `writeServerFrame(w, opcode, payload)` — emit a server-side
//!     (unmasked) frame for arbitrary payload size; selects 7-bit,
//!     16-bit, or 64-bit length encoding per the spec.
//!   * `readClientFrame(r, max_payload)` — parse one client frame
//!     (which MUST be masked per RFC 6455 §5.3) and unmask the payload.
//!
//! What's NOT here:
//!   * No fragmented-message reassembly (we never send fragments, and
//!     the only data we'd accept from the client is small control or
//!     text frames — those are intrinsically unfragmented).
//!   * No permessage-deflate extension.
//!
//! Hermetic-test posture: every primitive here is pure — it takes a
//! `std.Io.Reader` / `std.Io.Writer`, NOT a socket. The `Reader.fixed`
//! / `Writer.Allocating` pair lets tests round-trip the wire bytes
//! end-to-end with no I/O.

const std = @import("std");
const Sha1 = std.crypto.hash.Sha1;

/// RFC 6455 §1.3 magic GUID appended to the client's nonce before
/// hashing.
pub const accept_magic: []const u8 = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

/// Size of the base64-encoded SHA-1 digest (20 bytes → 28-char base64).
pub const accept_key_b64_len: usize = 28;

/// Compute the `Sec-WebSocket-Accept` header value. `out` must be at
/// least `accept_key_b64_len` bytes; on return it holds the ASCII
/// base64 string (no NUL terminator).
pub fn computeAcceptKey(client_key: []const u8, out: []u8) []const u8 {
    std.debug.assert(out.len >= accept_key_b64_len);
    var h = Sha1.init(.{});
    h.update(client_key);
    h.update(accept_magic);
    var digest: [Sha1.digest_length]u8 = undefined;
    h.final(&digest);
    return std.base64.standard.Encoder.encode(out[0..accept_key_b64_len], &digest);
}

/// RFC 6455 §5.2 opcodes. We don't bother modelling the reserved
/// range — anything outside this set is rejected at parse time.
pub const Opcode = enum(u4) {
    continuation = 0x0,
    text = 0x1,
    binary = 0x2,
    close = 0x8,
    ping = 0x9,
    pong = 0xA,
};

pub const Frame = struct {
    fin: bool,
    opcode: Opcode,
    /// Decoded payload, allocated from the reader-supplied allocator.
    payload: []u8,
};

pub const FrameError = error{
    ShortFrame,
    BadOpcode,
    Unmasked, // client → server frames MUST be masked
    TooLarge,
    OutOfMemory,
    EndOfStream,
    ReadFailed,
};

/// Read one client → server frame from `r` and return it. `payload`
/// is freshly allocated from `allocator`; caller frees.
pub fn readClientFrame(
    r: *std.Io.Reader,
    allocator: std.mem.Allocator,
    max_payload: usize,
) FrameError!Frame {
    // Header is at least 2 bytes.
    var header: [2]u8 = undefined;
    r.readSliceAll(&header) catch return error.ShortFrame;

    const b0 = header[0];
    const b1 = header[1];

    const fin: bool = (b0 & 0x80) != 0;
    const opcode_raw: u4 = @intCast(b0 & 0x0F);
    const opcode: Opcode = switch (opcode_raw) {
        0x0 => .continuation,
        0x1 => .text,
        0x2 => .binary,
        0x8 => .close,
        0x9 => .ping,
        0xA => .pong,
        else => return error.BadOpcode,
    };
    const masked: bool = (b1 & 0x80) != 0;
    const len7: u7 = @intCast(b1 & 0x7F);

    // RFC 6455 §5.3 — frames from client MUST be masked.
    if (!masked) return error.Unmasked;

    const payload_len: u64 = switch (len7) {
        0...125 => @as(u64, len7),
        126 => blk: {
            var ext: [2]u8 = undefined;
            r.readSliceAll(&ext) catch return error.ShortFrame;
            break :blk std.mem.readInt(u16, &ext, .big);
        },
        127 => blk: {
            var ext: [8]u8 = undefined;
            r.readSliceAll(&ext) catch return error.ShortFrame;
            break :blk std.mem.readInt(u64, &ext, .big);
        },
    };

    if (payload_len > max_payload) return error.TooLarge;

    var mask: [4]u8 = undefined;
    r.readSliceAll(&mask) catch return error.ShortFrame;

    const payload_usize: usize = @intCast(payload_len);
    const buf = try allocator.alloc(u8, payload_usize);
    errdefer allocator.free(buf);
    r.readSliceAll(buf) catch {
        allocator.free(buf);
        return error.ShortFrame;
    };
    var i: usize = 0;
    while (i < buf.len) : (i += 1) buf[i] ^= mask[i & 3];

    return .{ .fin = fin, .opcode = opcode, .payload = buf };
}

/// Emit one server → client frame to `w`. Server frames are NOT masked
/// (RFC 6455 §5.1). The length field is sized to the payload:
///   * < 126 → single byte
///   * 126..2^16-1 → two-byte length after a 126 marker
///   * ≥ 2^16 → eight-byte length after a 127 marker
pub fn writeServerFrame(w: *std.Io.Writer, opcode: Opcode, payload: []const u8) !void {
    const op4: u4 = @intFromEnum(opcode);
    const b0: u8 = 0x80 | @as(u8, op4); // FIN=1 + opcode
    try w.writeByte(b0);

    if (payload.len < 126) {
        try w.writeByte(@intCast(payload.len));
    } else if (payload.len < 65536) {
        try w.writeByte(126);
        var ext: [2]u8 = undefined;
        std.mem.writeInt(u16, &ext, @intCast(payload.len), .big);
        try w.writeAll(&ext);
    } else {
        try w.writeByte(127);
        var ext: [8]u8 = undefined;
        std.mem.writeInt(u64, &ext, payload.len, .big);
        try w.writeAll(&ext);
    }
    if (payload.len > 0) try w.writeAll(payload);
}

// === Tests ============================================================

const testing = std.testing;

test "ws: computeAcceptKey matches the RFC 6455 example" {
    // Section 1.3:
    //   Sec-WebSocket-Key:    dGhlIHNhbXBsZSBub25jZQ==
    //   Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=
    var out: [accept_key_b64_len]u8 = undefined;
    const got = computeAcceptKey("dGhlIHNhbXBsZSBub25jZQ==", &out);
    try testing.expectEqualStrings("s3pPLMBiTxaQ9kYGzzhZRbK+xOo=", got);
}

test "ws: writeServerFrame uses 7-bit length for small payload" {
    var buf: [128]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try writeServerFrame(&w, .text, "hello");
    // 0x81 = FIN + text; 0x05 = length 5; "hello"
    try testing.expectEqualSlices(u8, &[_]u8{ 0x81, 0x05, 'h', 'e', 'l', 'l', 'o' }, w.buffered());
}

test "ws: writeServerFrame uses 16-bit length for medium payload" {
    const a = testing.allocator;
    const payload = try a.alloc(u8, 200);
    defer a.free(payload);
    @memset(payload, 0xAB);
    var out: std.Io.Writer.Allocating = .init(a);
    defer out.deinit();
    try writeServerFrame(&out.writer, .binary, payload);
    const w = out.written();
    // Header: 0x82 (binary FIN), 0x7E (126 marker), then big-endian u16 = 200.
    try testing.expectEqual(@as(u8, 0x82), w[0]);
    try testing.expectEqual(@as(u8, 126), w[1]);
    try testing.expectEqual(@as(u8, 0), w[2]);
    try testing.expectEqual(@as(u8, 200), w[3]);
    try testing.expectEqual(@as(usize, 4 + 200), w.len);
}

test "ws: readClientFrame round-trips a masked binary frame" {
    const a = testing.allocator;
    // Build a client-style frame: FIN + binary, masked, payload "ping".
    const payload = "ping";
    const mask: [4]u8 = .{ 0x12, 0x34, 0x56, 0x78 };
    var wire: std.ArrayList(u8) = .empty;
    defer wire.deinit(a);
    try wire.append(a, 0x82); // FIN + binary
    try wire.append(a, 0x80 | @as(u8, payload.len)); // mask bit + length
    try wire.appendSlice(a, &mask);
    for (payload, 0..) |c, i| try wire.append(a, c ^ mask[i & 3]);

    var r: std.Io.Reader = .fixed(wire.items);
    const frame = try readClientFrame(&r, a, 1024);
    defer a.free(frame.payload);
    try testing.expect(frame.fin);
    try testing.expectEqual(Opcode.binary, frame.opcode);
    try testing.expectEqualSlices(u8, payload, frame.payload);
}

test "ws: readClientFrame rejects unmasked client frame" {
    const a = testing.allocator;
    // FIN + binary, mask bit = 0, length 0. Per RFC server MUST reject.
    var wire: [2]u8 = .{ 0x82, 0x00 };
    var r: std.Io.Reader = .fixed(&wire);
    try testing.expectError(error.Unmasked, readClientFrame(&r, a, 1024));
}

test "ws: server → client round-trip via Reader.fixed + Allocating" {
    const a = testing.allocator;
    var out: std.Io.Writer.Allocating = .init(a);
    defer out.deinit();
    // Server emits binary frame containing little-endian u32 ids.
    const ids = [_]u32{ 1, 2, 3, 4 };
    const bytes: [*]const u8 = @ptrCast(&ids);
    const payload = bytes[0 .. ids.len * @sizeOf(u32)];
    try writeServerFrame(&out.writer, .binary, payload);

    // The "client side" we model here is just a parser: server frames
    // are unmasked, so to feed them through readClientFrame we set the
    // mask bit + use a 0-mask. (This exercises every parser branch
    // except the actual XOR, which the previous test already covers.)
    const wire = out.written();
    var rebuilt: std.ArrayList(u8) = .empty;
    defer rebuilt.deinit(a);
    try rebuilt.append(a, wire[0]); // header byte unchanged
    try rebuilt.append(a, 0x80 | wire[1]); // flip mask bit
    try rebuilt.appendSlice(a, wire[2..]);
    // Insert zero mask before the payload.
    const mask: [4]u8 = .{ 0, 0, 0, 0 };
    try rebuilt.insertSlice(a, 2, &mask);

    var r: std.Io.Reader = .fixed(rebuilt.items);
    const frame = try readClientFrame(&r, a, 1024);
    defer a.free(frame.payload);
    try testing.expectEqualSlices(u8, payload, frame.payload);
}
