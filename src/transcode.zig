//! Cross-tokenizer transcoding — re-map a token-id stream produced by
//! tokenizer A into the equivalent id stream for tokenizer B.
//!
//! ## Why
//!
//! Distillation and dataset reuse frequently need a corpus that was
//! tokenized once (for model A) to be fed to a different model B with a
//! different vocabulary. Re-tokenizing from the raw text is the obvious
//! route, but pipelines often only keep the id stream around. This
//! module re-maps `ids_A -> ids_B` directly.
//!
//! ## Semantics — text is the bridge
//!
//! There is no general algebraic shortcut between two arbitrary vocabs:
//! a token in A may span a sub-piece, a whole word, or several words
//! that B splits differently. The only universal common ground is the
//! decoded byte stream. So transcoding is exactly:
//!
//!     bytes := decode_A(ids_A)
//!     ids_B := encode_B(bytes)
//!
//! `decode` and `encode` are the ordinary `Pipeline` operations, so the
//! re-mapping inherits whatever normalizer / pre-tokenizer / decoder the
//! caller wired into each pipeline.
//!
//! ## Round-trip invariant
//!
//!     decode_B(transcode(ids_A)) == decode_A(ids_A)
//!
//! i.e. the *text* is preserved exactly. We assert the text invariant in
//! the tests rather than asserting `ids_A == ids_B`, because the ids are
//! deliberately allowed to differ — that is the whole point.
//!
//! ## Honest exactness caveat
//!
//! The invariant above holds *exactly* only when **both** pipelines
//! round-trip text losslessly — i.e. the vocab is byte-level (or has a
//! byte-fallback path) *and* the pipeline is wired with a decoder that
//! reverses the model's byte representation, so `encode` then `decode`
//! is the identity on bytes. Concretely:
//!   * byte-level BPE (GPT-2 / cl100k / Llama-3 / Qwen2) needs the
//!     `byte_level` normalizer on encode and the `byte_level` decoder;
//!   * a raw `.tiktoken` BPE whose pieces are stored as literal bytes
//!     (cl100k via `loadTiktokenFile`) round-trips with `concat`;
//!   * a `byte_id` model round-trips any byte stream with `concat`.
//!
//! Note that SentencePiece byte-fallback models do *not* round-trip
//! through the plain `concat` decoder in this toolkit: their fallback
//! pieces are stored as the literal text `<0xNN>`, and `concat` emits
//! that literal rather than the byte. They need an SP-aware decoder to
//! qualify as a lossless bridge endpoint.
//!
//! When B *cannot* represent some bytes (no byte fallback, an `<unk>`
//! sink, lossy NFKC normalization, or a WordPiece-style decoder that
//! re-spaces output), the bridge is only as faithful as `encode_B` /
//! `decode_B` allow: bytes outside B's reach collapse to `<unk>` or are
//! dropped, and the equality degrades to "best effort". Likewise, if A's
//! own pipeline does not losslessly decode its ids (e.g. a lossy
//! normalizer baked into A), `decode_A(ids_A)` is already not the
//! original text and transcoding can only preserve *that* decoded text,
//! not the pre-A original. Callers that need a guarantee should transcode
//! between byte-level / byte-fallback vocabs and verify with `roundtrip`.

const std = @import("std");
const Pipeline = @import("pipeline.zig").Pipeline;
const TokenId = @import("token.zig").TokenId;

/// Re-map a single id stream from `pipeline_a` to `pipeline_b`.
///
/// Returns a freshly allocated `[]TokenId` owned by the caller (allocated
/// from `allocator`). The intermediate decoded bytes are released before
/// returning.
pub fn transcodeIds(
    allocator: std.mem.Allocator,
    pipeline_a: *const Pipeline,
    pipeline_b: *const Pipeline,
    ids_a: []const TokenId,
) ![]TokenId {
    const bytes = try pipeline_a.decode(allocator, ids_a);
    defer allocator.free(bytes);
    return pipeline_b.encode(allocator, bytes);
}

/// Output encoding for a transcoded corpus.
pub const Format = enum {
    /// One line of space-separated decimal ids per input line.
    text,
    /// One JSON array of ids per input line (JSON Lines).
    jsonl,
};

pub const CorpusOptions = struct {
    format: Format = .text,
    /// Skip blank input lines instead of emitting an empty output line.
    /// A blank line decodes to the empty string and re-encodes to zero
    /// ids, so by default it round-trips to a blank output line; set this
    /// to drop them entirely.
    skip_blank: bool = false,
    /// When set, ids `>= max_src_id` in the input are rejected with
    /// `error.IdOutOfRange` instead of being passed to `decode_A` (which
    /// would assert/panic for BPE models). Pass `vocab_size_of_A` here.
    /// `null` disables the check (e.g. for `byte_id` models, where any
    /// u32 maps to a byte).
    max_src_id: ?TokenId = null,
};

pub const Error = error{IdOutOfRange};

/// Stats returned by `transcodeCorpus` so callers (and the CLI) can
/// report progress without re-scanning the output.
pub const CorpusStats = struct {
    lines_in: usize = 0,
    lines_out: usize = 0,
    ids_in: usize = 0,
    ids_out: usize = 0,
};

/// Parse one whitespace-separated line of decimal ids from `line` into a
/// freshly allocated slice. Empty / whitespace-only lines yield a
/// zero-length slice. Errors on non-numeric tokens.
fn parseIdLine(allocator: std.mem.Allocator, line: []const u8) ![]TokenId {
    var list: std.ArrayList(TokenId) = .empty;
    errdefer list.deinit(allocator);

    var it = std.mem.tokenizeAny(u8, line, " \t\r");
    while (it.next()) |tok| {
        const id = try std.fmt.parseInt(TokenId, tok, 10);
        try list.append(allocator, id);
    }
    return list.toOwnedSlice(allocator);
}

fn writeIdLine(out: *std.Io.Writer, ids: []const TokenId, fmt: Format) !void {
    switch (fmt) {
        .text => {
            for (ids, 0..) |id, i| {
                if (i > 0) try out.writeByte(' ');
                try out.print("{d}", .{id});
            }
            try out.writeByte('\n');
        },
        .jsonl => {
            try out.writeByte('[');
            for (ids, 0..) |id, i| {
                if (i > 0) try out.writeByte(',');
                try out.print("{d}", .{id});
            }
            try out.writeAll("]\n");
        },
    }
}

/// Batch / streaming transcode over a corpus of id-lines.
///
/// `input` is the full corpus as a byte buffer; each line is a record of
/// space-separated A-ids (the format `ztok encode` emits). Each line is
/// decoded through `pipeline_a`, re-encoded through `pipeline_b`, and the
/// resulting B-ids are written to `out` one record per line in the chosen
/// `format`. Input may be JSON-Lines arrays as well — `parseIdLine`
/// tolerates the surrounding brackets/commas as whitespace-equivalent
/// separators are not assumed, so callers that emit `text` should feed
/// `text`; for `jsonl` input the brackets and commas are stripped by the
/// tolerant tokenizer (`[`, `]`, `,` are not digits and would error), so
/// this function expects **text** input lines. See the CLI for format
/// auto-handling of the *output* side.
///
/// Returns aggregate `CorpusStats`. The caller owns nothing returned by
/// reference; all per-line allocations are released internally.
pub fn transcodeCorpus(
    allocator: std.mem.Allocator,
    pipeline_a: *const Pipeline,
    pipeline_b: *const Pipeline,
    input: []const u8,
    out: *std.Io.Writer,
    opts: CorpusOptions,
) !CorpusStats {
    var stats: CorpusStats = .{};

    var lines = std.mem.splitScalar(u8, input, '\n');
    while (lines.next()) |raw_line| {
        // A trailing newline produces a final empty token from splitScalar;
        // don't treat it as a record.
        const is_last_empty = (lines.peek() == null and raw_line.len == 0);
        if (is_last_empty) break;

        const trimmed = std.mem.trim(u8, raw_line, " \t\r");
        if (trimmed.len == 0 and opts.skip_blank) continue;

        stats.lines_in += 1;

        const ids_a = try parseIdLine(allocator, trimmed);
        defer allocator.free(ids_a);
        stats.ids_in += ids_a.len;

        if (opts.max_src_id) |limit| {
            for (ids_a) |id| {
                if (id >= limit) return Error.IdOutOfRange;
            }
        }

        const ids_b = try transcodeIds(allocator, pipeline_a, pipeline_b, ids_a);
        defer allocator.free(ids_b);
        stats.ids_out += ids_b.len;

        try writeIdLine(out, ids_b, opts.format);
        stats.lines_out += 1;
    }

    return stats;
}

// --- tests -----------------------------------------------------------------

const testing = std.testing;
const Vocab = @import("vocab.zig").Vocab;
const Bpe = @import("bpe.zig").Bpe;

const test_io = std.Io.Threaded.global_single_threaded;

fn threadedIo() std.Io {
    return std.Io.Threaded.global_single_threaded.io();
}

/// A byte_id pipeline round-trips *any* byte stream losslessly: it maps
/// each byte to id == byte and concatenates on decode. This is the
/// canonical "lossless bridge" target B used in the cross-tokenizer
/// tests.
fn byteLevelPipeline(v: *const Vocab) Pipeline {
    return .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = v,
    };
}

test "identity A->A preserves decoded text" {
    // The real invariant: decode_A(transcode_{A->A}(ids)) == decode_A(ids).
    // Use byte_id for A and B so the bridge is exact for arbitrary bytes.
    var v = Vocab.empty(testing.allocator);
    defer v.deinit();
    const pipe = byteLevelPipeline(&v);

    const text = "Hello, world! 123 \t mixed\nbytes\xff\x00ok";
    const ids_a = try pipe.encode(testing.allocator, text);
    defer testing.allocator.free(ids_a);

    const ids_b = try transcodeIds(testing.allocator, &pipe, &pipe, ids_a);
    defer testing.allocator.free(ids_b);

    const decoded_a = try pipe.decode(testing.allocator, ids_a);
    defer testing.allocator.free(decoded_a);
    const decoded_b = try pipe.decode(testing.allocator, ids_b);
    defer testing.allocator.free(decoded_b);

    // Text invariant — the load-bearing assertion.
    try testing.expectEqualStrings(decoded_a, decoded_b);
    // And for byte_id<->byte_id specifically the ids match too.
    try testing.expectEqualSlices(TokenId, ids_a, ids_b);
}

test "cross: cl100k -> byte-level round-trips text" {
    // Encode ASCII text with cl100k_base (real byte-level BPE), transcode
    // to a byte-level (byte_id) vocab, and confirm decode_B == original.
    const path = "bench/vocabs/cl100k_base.tiktoken";
    const bytes = std.Io.Dir.cwd().readFileAlloc(threadedIo(), path, testing.allocator, .unlimited) catch
        return error.SkipZigTest;
    defer testing.allocator.free(bytes);

    var bpe = Bpe.loadTiktokenBytes(testing.allocator, bytes) catch return error.SkipZigTest;
    defer bpe.deinit();

    var va = Vocab.empty(testing.allocator);
    defer va.deinit();
    const pipe_a: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .cl100k,
        .model = .{ .bpe = &bpe },
        .decoder = .concat,
        .vocab = &va,
    };

    var vb = Vocab.empty(testing.allocator);
    defer vb.deinit();
    const pipe_b = byteLevelPipeline(&vb);

    const text = "The quick brown fox jumps over the lazy dog.";
    const ids_a = try pipe_a.encode(testing.allocator, text);
    defer testing.allocator.free(ids_a);

    // Sanity: A itself round-trips the text (precondition for the bridge).
    const decoded_a = try pipe_a.decode(testing.allocator, ids_a);
    defer testing.allocator.free(decoded_a);
    try testing.expectEqualStrings(text, decoded_a);

    const ids_b = try transcodeIds(testing.allocator, &pipe_a, &pipe_b, ids_a);
    defer testing.allocator.free(ids_b);

    const decoded_b = try pipe_b.decode(testing.allocator, ids_b);
    defer testing.allocator.free(decoded_b);

    // The invariant: decode_B(transcode(ids)) == decode_A(ids) == text.
    try testing.expectEqualStrings(text, decoded_b);
}

/// Build a GPT-2-style byte-level BPE pipeline from an HF tokenizer.json.
/// The byte_level normalizer maps raw bytes into the GPT-2 printable
/// alphabet on encode; the byte_level decoder reverses it. This pair
/// round-trips arbitrary text losslessly.
fn loadHfByteLevel(
    path: []const u8,
    bpe_out: *Bpe,
    v: *const Vocab,
) !Pipeline {
    const hf_json = @import("hf_json.zig");
    const hf_bridge = @import("hf_bridge.zig");
    const bytes = std.Io.Dir.cwd().readFileAlloc(threadedIo(), path, testing.allocator, .unlimited) catch
        return error.SkipZigTest;
    defer testing.allocator.free(bytes);
    var hf = hf_json.loadFromBytes(testing.allocator, bytes) catch return error.SkipZigTest;
    defer hf.deinit();
    if (hf.model_kind != .bpe) return error.SkipZigTest;
    bpe_out.* = hf_bridge.bpeFromHF(testing.allocator, &hf) catch return error.SkipZigTest;
    return .{
        .normalizer = .byte_level,
        .pre_tokenizer = .identity,
        .model = .{ .bpe = bpe_out },
        .decoder = .byte_level,
        .vocab = v,
    };
}

test "bench vocab pair: gpt2 -> llama3 text round-trips" {
    // A real cross-vocab pair from bench/vocabs on a short ASCII string.
    // gpt2 and llama3 are both byte-level BPE (HF ByteLevel) — every byte
    // is representable, so the text bridge is exact in both directions.
    var gpt2: Bpe = undefined;
    var va = Vocab.empty(testing.allocator);
    defer va.deinit();
    const pipe_a = loadHfByteLevel("bench/vocabs/gpt2_hf.json", &gpt2, &va) catch |e| {
        if (e == error.SkipZigTest) return e;
        return e;
    };
    defer gpt2.deinit();

    var llama3: Bpe = undefined;
    var vb = Vocab.empty(testing.allocator);
    defer vb.deinit();
    const pipe_b = loadHfByteLevel("bench/vocabs/llama3.json", &llama3, &vb) catch |e| {
        if (e == error.SkipZigTest) return e;
        return e;
    };
    defer llama3.deinit();

    const text = "the quick brown fox";

    const ids_a = try pipe_a.encode(testing.allocator, text);
    defer testing.allocator.free(ids_a);
    const decoded_a = try pipe_a.decode(testing.allocator, ids_a);
    defer testing.allocator.free(decoded_a);
    // Precondition: A round-trips its own text.
    try testing.expectEqualStrings(text, decoded_a);

    const ids_b = try transcodeIds(testing.allocator, &pipe_a, &pipe_b, ids_a);
    defer testing.allocator.free(ids_b);
    const decoded_b = try pipe_b.decode(testing.allocator, ids_b);
    defer testing.allocator.free(decoded_b);

    // The invariant: decode_B(transcode(ids)) == decode_A(ids) == text.
    try testing.expectEqualStrings(text, decoded_b);
    _ = test_io;
}

test "transcodeCorpus text format: per-line ids round-trip" {
    var va = Vocab.empty(testing.allocator);
    defer va.deinit();
    var vb = Vocab.empty(testing.allocator);
    defer vb.deinit();
    const pipe_a = byteLevelPipeline(&va);
    const pipe_b = byteLevelPipeline(&vb);

    // Two text lines -> their byte_id streams.
    const l1 = try pipe_a.encode(testing.allocator, "abc");
    defer testing.allocator.free(l1);
    const l2 = try pipe_a.encode(testing.allocator, "Z9!");
    defer testing.allocator.free(l2);

    var in_buf: std.ArrayList(u8) = .empty;
    defer in_buf.deinit(testing.allocator);
    var num_buf: [16]u8 = undefined;
    for (l1, 0..) |id, i| {
        if (i > 0) try in_buf.append(testing.allocator, ' ');
        try in_buf.appendSlice(testing.allocator, std.fmt.bufPrint(&num_buf, "{d}", .{id}) catch unreachable);
    }
    try in_buf.append(testing.allocator, '\n');
    for (l2, 0..) |id, i| {
        if (i > 0) try in_buf.append(testing.allocator, ' ');
        try in_buf.appendSlice(testing.allocator, std.fmt.bufPrint(&num_buf, "{d}", .{id}) catch unreachable);
    }
    try in_buf.append(testing.allocator, '\n');

    var out_buf: [4096]u8 = undefined;
    var w: std.Io.Writer = .fixed(&out_buf);
    const stats = try transcodeCorpus(testing.allocator, &pipe_a, &pipe_b, in_buf.items, &w, .{});

    try testing.expectEqual(@as(usize, 2), stats.lines_in);
    try testing.expectEqual(@as(usize, 2), stats.lines_out);

    // byte_id<->byte_id is identity, so output == input.
    try testing.expectEqualStrings(in_buf.items, w.buffered());
}

test "transcodeCorpus jsonl format brackets each record" {
    var va = Vocab.empty(testing.allocator);
    defer va.deinit();
    var vb = Vocab.empty(testing.allocator);
    defer vb.deinit();
    const pipe_a = byteLevelPipeline(&va);
    const pipe_b = byteLevelPipeline(&vb);

    // "AB" -> ids 65 66.
    const input = "65 66\n";
    var out_buf: [256]u8 = undefined;
    var w: std.Io.Writer = .fixed(&out_buf);
    const stats = try transcodeCorpus(testing.allocator, &pipe_a, &pipe_b, input, &w, .{ .format = .jsonl });
    try testing.expectEqual(@as(usize, 1), stats.lines_out);
    try testing.expectEqualStrings("[65,66]\n", w.buffered());
}
