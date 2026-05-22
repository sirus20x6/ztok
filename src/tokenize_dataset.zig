//! Dataset tokenization for training.
//!
//! Encodes a text corpus into a packed token-id array ready for an LLM
//! training loader, in one of two formats:
//!   * `.bin` — a flat little-endian stream of all token ids (the
//!     nanoGPT / llm.c convention; memmap a `uint16`/`uint32` array and
//!     sample fixed windows at train time).
//!   * `.npy` — a 2D NumPy array `[n_sequences, seq_len]` of pre-cut,
//!     concat-packed training examples (the trailing partial sequence is
//!     dropped, or padded with `pad_id` when `pad_last`).
//!
//! Documents are separated by an optional BOS prefix and/or EOS suffix.
//! Input is split into documents either per non-empty line (default) or
//! treated as one document.
//!
//! dtype is `uint16` when the vocab fits (<= 65536 ids), else `uint32`.
//!
//! v1 buffers all token ids in memory before writing; for very large
//! corpora a streaming/sharded path is a planned follow-on.

const std = @import("std");
const Pipeline = @import("pipeline.zig").Pipeline;
const TokenId = @import("token.zig").TokenId;

pub const Format = enum { bin, npy };

pub const Error = error{
    /// A token id exceeds the chosen dtype's range (e.g. id >= 65536 for
    /// uint16). Pick `u32` / `auto`.
    IdTooLargeForDtype,
} || std.mem.Allocator.Error;

pub const Options = struct {
    seq_len: usize = 2048,
    format: Format = .bin,
    /// 2 (uint16) or 4 (uint32).
    dtype_bytes: u8 = 2,
    add_bos: bool = false,
    bos_id: u32 = 0,
    add_eos: bool = false,
    eos_id: u32 = 0,
    /// `.npy` only: pad the trailing partial sequence to `seq_len` with
    /// `pad_id` instead of dropping it.
    pad_last: bool = false,
    pad_id: u32 = 0,
    /// Split input into one document per non-empty line; otherwise the
    /// whole input is a single document.
    doc_per_line: bool = true,
};

pub const Stats = struct { docs: usize, tokens: usize, sequences: usize };

/// uint16 if the vocab fits in 16 bits, else uint32.
pub fn pickDtypeBytes(vocab_size: u32) u8 {
    return if (vocab_size <= 65536) 2 else 4;
}

fn encodeDoc(
    allocator: std.mem.Allocator,
    pipeline: *const Pipeline,
    doc: []const u8,
    opts: Options,
    out: *std.ArrayList(TokenId),
) !void {
    if (opts.add_bos) try out.append(allocator, opts.bos_id);
    const ids = try pipeline.encode(allocator, doc);
    defer allocator.free(ids);
    try out.appendSlice(allocator, ids);
    if (opts.add_eos) try out.append(allocator, opts.eos_id);
}

/// Encode `input` into a flat list of token ids (BOS/EOS inserted per
/// document). Returns the number of documents encoded.
pub fn tokenize(
    allocator: std.mem.Allocator,
    pipeline: *const Pipeline,
    input: []const u8,
    opts: Options,
    out: *std.ArrayList(TokenId),
) !usize {
    var docs: usize = 0;
    if (opts.doc_per_line) {
        var it = std.mem.splitScalar(u8, input, '\n');
        while (it.next()) |line| {
            const doc = std.mem.trim(u8, line, "\r");
            if (doc.len == 0) continue;
            try encodeDoc(allocator, pipeline, doc, opts, out);
            docs += 1;
        }
    } else if (input.len > 0) {
        try encodeDoc(allocator, pipeline, input, opts, out);
        docs = 1;
    }
    return docs;
}

inline fn writeToken(allocator: std.mem.Allocator, out: *std.ArrayList(u8), id: u32, dtype_bytes: u8) Error!void {
    if (dtype_bytes == 2) {
        if (id > std.math.maxInt(u16)) return Error.IdTooLargeForDtype;
        var b: [2]u8 = undefined;
        std.mem.writeInt(u16, &b, @intCast(id), .little);
        try out.appendSlice(allocator, &b);
    } else {
        var b: [4]u8 = undefined;
        std.mem.writeInt(u32, &b, id, .little);
        try out.appendSlice(allocator, &b);
    }
}

/// Flat little-endian token stream (`.bin`).
pub fn serializeBin(
    allocator: std.mem.Allocator,
    ids: []const TokenId,
    dtype_bytes: u8,
    out: *std.ArrayList(u8),
) Error!void {
    for (ids) |id| try writeToken(allocator, out, id, dtype_bytes);
}

/// Write a NumPy v1.0 array header for shape `(n_seq, seq_len)`.
fn npyHeader(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    dtype_bytes: u8,
    n_seq: usize,
    seq_len: usize,
) !void {
    const descr = if (dtype_bytes == 2) "<u2" else "<u4";
    var dict_buf: [128]u8 = undefined;
    const dict = try std.fmt.bufPrint(
        &dict_buf,
        "{{'descr': '{s}', 'fortran_order': False, 'shape': ({d}, {d}), }}",
        .{ descr, n_seq, seq_len },
    );
    // 10-byte preamble (magic 6 + version 2 + header-len 2). The header
    // (dict + spaces + '\n') is padded so the total is a 64-byte multiple.
    const base = 10;
    const unpadded = base + dict.len + 1; // + trailing '\n'
    const n_spaces = (64 - (unpadded % 64)) % 64;
    const header_len = dict.len + n_spaces + 1;

    try out.appendSlice(allocator, "\x93NUMPY");
    try out.append(allocator, 1); // major
    try out.append(allocator, 0); // minor
    var hl: [2]u8 = undefined;
    std.mem.writeInt(u16, &hl, @intCast(header_len), .little);
    try out.appendSlice(allocator, &hl);
    try out.appendSlice(allocator, dict);
    var s: usize = 0;
    while (s < n_spaces) : (s += 1) try out.append(allocator, ' ');
    try out.append(allocator, '\n');
}

/// 2D `[n_seq, seq_len]` NumPy array (`.npy`), concat-packed. Returns the
/// number of sequences written.
pub fn serializeNpy(
    allocator: std.mem.Allocator,
    ids: []const TokenId,
    seq_len: usize,
    dtype_bytes: u8,
    pad_last: bool,
    pad_id: u32,
    out: *std.ArrayList(u8),
) !usize {
    std.debug.assert(seq_len > 0);
    const n_full = ids.len / seq_len;
    const rem = ids.len % seq_len;
    const n_seq = if (pad_last and rem > 0) n_full + 1 else n_full;

    try npyHeader(allocator, out, dtype_bytes, n_seq, seq_len);

    const total = n_seq * seq_len;
    var i: usize = 0;
    while (i < total and i < ids.len) : (i += 1) try writeToken(allocator, out, ids[i], dtype_bytes);
    // pad the final partial row (only reached when pad_last and rem > 0)
    while (i < total) : (i += 1) try writeToken(allocator, out, pad_id, dtype_bytes);
    return n_seq;
}

// --- tests ----------------------------------------------------------------

const testing = std.testing;

fn byteIdPipeline(v: *@import("vocab.zig").Vocab) Pipeline {
    return .{ .normalizer = .identity, .pre_tokenizer = .identity, .model = .byte_id, .decoder = .concat, .vocab = v };
}

test "pickDtypeBytes" {
    try testing.expectEqual(@as(u8, 2), pickDtypeBytes(256));
    try testing.expectEqual(@as(u8, 2), pickDtypeBytes(65536));
    try testing.expectEqual(@as(u8, 4), pickDtypeBytes(65537));
    try testing.expectEqual(@as(u8, 4), pickDtypeBytes(100256));
}

test "tokenize: per-line docs with EOS" {
    var v = @import("vocab.zig").Vocab.empty(testing.allocator);
    defer v.deinit();
    const pipe = byteIdPipeline(&v);

    var ids: std.ArrayList(TokenId) = .empty;
    defer ids.deinit(testing.allocator);
    const opts: Options = .{ .add_eos = true, .eos_id = 999 };
    const docs = try tokenize(testing.allocator, &pipe, "ab\ncd\n", opts, &ids);

    try testing.expectEqual(@as(usize, 2), docs);
    // 'a','b',EOS,'c','d',EOS
    try testing.expectEqualSlices(TokenId, &.{ 'a', 'b', 999, 'c', 'd', 999 }, ids.items);
}

test "tokenize: whole-input single doc with BOS+EOS" {
    var v = @import("vocab.zig").Vocab.empty(testing.allocator);
    defer v.deinit();
    const pipe = byteIdPipeline(&v);

    var ids: std.ArrayList(TokenId) = .empty;
    defer ids.deinit(testing.allocator);
    const docs = try tokenize(testing.allocator, &pipe, "hi", .{ .doc_per_line = false, .add_bos = true, .bos_id = 1, .add_eos = true, .eos_id = 2 }, &ids);
    try testing.expectEqual(@as(usize, 1), docs);
    try testing.expectEqualSlices(TokenId, &.{ 1, 'h', 'i', 2 }, ids.items);
}

test "serializeBin: little-endian u16 / u32" {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(testing.allocator);
    try serializeBin(testing.allocator, &.{ 1, 258 }, 2, &out);
    // 1 -> 01 00, 258 -> 02 01
    try testing.expectEqualSlices(u8, &.{ 0x01, 0x00, 0x02, 0x01 }, out.items);

    out.clearRetainingCapacity();
    try serializeBin(testing.allocator, &.{0x01020304}, 4, &out);
    try testing.expectEqualSlices(u8, &.{ 0x04, 0x03, 0x02, 0x01 }, out.items);
}

test "serializeBin: u16 overflow is rejected" {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(testing.allocator);
    try testing.expectError(Error.IdTooLargeForDtype, serializeBin(testing.allocator, &.{70000}, 2, &out));
}

test "serializeNpy: header validity + drop-last vs pad-last" {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(testing.allocator);

    // 5 ids, seq_len 2: drop-last -> 2 sequences (4 tokens).
    const n = try serializeNpy(testing.allocator, &.{ 1, 2, 3, 4, 5 }, 2, 2, false, 0, &out);
    try testing.expectEqual(@as(usize, 2), n);
    // magic + version
    try testing.expectEqualSlices(u8, "\x93NUMPY", out.items[0..6]);
    try testing.expectEqual(@as(u8, 1), out.items[6]);
    try testing.expectEqual(@as(u8, 0), out.items[7]);
    const header_len = std.mem.readInt(u16, out.items[8..10], .little);
    // 10 + header_len must be a 64-byte multiple, header ends with '\n'.
    try testing.expectEqual(@as(usize, 0), (10 + @as(usize, header_len)) % 64);
    const hdr_end = 10 + @as(usize, header_len);
    try testing.expectEqual(@as(u8, '\n'), out.items[hdr_end - 1]);
    try testing.expect(std.mem.indexOf(u8, out.items[10..hdr_end], "'descr': '<u2'") != null);
    try testing.expect(std.mem.indexOf(u8, out.items[10..hdr_end], "(2, 2)") != null);
    // body: 4 tokens * 2 bytes
    try testing.expectEqual(hdr_end + 4 * 2, out.items.len);

    // pad-last -> 3 sequences (6 tokens), last padded with pad_id 7.
    out.clearRetainingCapacity();
    const n2 = try serializeNpy(testing.allocator, &.{ 1, 2, 3, 4, 5 }, 2, 2, true, 7, &out);
    try testing.expectEqual(@as(usize, 3), n2);
    const hl2 = std.mem.readInt(u16, out.items[8..10], .little);
    const body = out.items[10 + @as(usize, hl2) ..];
    try testing.expectEqual(@as(usize, 3 * 2 * 2), body.len);
    // last token is the pad id 7
    try testing.expectEqual(@as(u16, 7), std.mem.readInt(u16, body[body.len - 2 ..][0..2], .little));
}
