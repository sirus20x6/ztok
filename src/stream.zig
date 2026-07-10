//! Streaming encode API — emit ids as input bytes arrive, instead of
//! waiting for the full buffer.
//!
//! Use case: partial chat responses, stdin pipelines, long-document
//! pre-tokenization. The streaming encoder defers any trailing partial
//! codepoint OR partial pre-tokenizer span at the end of each `feed`
//! call to a small carry buffer; on the next `feed` it prepends that
//! carry and re-encodes from the resulting safe prefix.
//!
//! Safety model — `findSafeCut`-based:
//!   The pre-tokenizer's `findSafeCut(input, desired, window)` tells us
//!   the largest position `<= desired` at which `input` can be split
//!   without changing the resulting token sequence on either half.
//!   `feed` runs `findSafeCut(buf, buf.len, window)` on its accumulated
//!   buffer; bytes up to that cut point are encoded and emitted, and
//!   the remainder becomes the new carry.
//!
//! Bounded carry: the carry has a hard cap (`max_carry_bytes`, default
//! 1 MiB). If a single pre-tokenizer span — or a stretch of input with
//! no safe-cut boundary — exceeds this, the encoder force-encodes the
//! carry to recover. The output is still legal token-wise but MAY
//! diverge from single-shot `Pipeline.encode` at the forced cut (the
//! same divergence `encodeChunked` documents when no safe cut is
//! found within the search window).
//!
//! Parity with single-shot `Pipeline.encode`:
//!   When the pre-tokenizer's `findSafeCut` finds a real boundary
//!   (cl100k regex match, HF byte-level whitespace), the streaming
//!   output is bit-identical to `encode(full_input)`. With the
//!   `identity` pre-tokenizer + a merge-based model (BPE/Monster),
//!   chunk boundaries fall on codepoint boundaries but NOT on BPE
//!   merge boundaries, so the streaming output MAY differ at chunk
//!   seams. This mirrors the same caveat `Pipeline.encodeChunked`
//!   documents — same boundary algorithm, same trade-off.
//!
//! Not used by `Pipeline.encode` itself; this is a complementary API
//! aimed at the `ztok serve` `/encode_stream` route and any library
//! caller that wants id-as-it-arrives semantics.

const std = @import("std");
const Pipeline = @import("pipeline.zig").Pipeline;
const ScratchArena = @import("pipeline.zig").ScratchArena;
const TokenId = @import("token.zig").TokenId;
const Vocab = @import("vocab.zig").Vocab;
const Bpe = @import("bpe.zig").Bpe;

/// Default cap on the carry buffer. If a single span exceeds this we
/// force a cut at the end of the carry; see the module doc comment.
pub const default_max_carry_bytes: usize = 1 * 1024 * 1024;

/// Default search window for `findSafeCut`. Mirrors the
/// `encodeChunked` window (16 KiB is plenty to find a regex boundary
/// in the long-line corpora we've seen — max line length ~11 KB on
/// Wikipedia).
pub const default_search_window: usize = 16 * 1024;

pub const Options = struct {
    /// Hard cap on the carry buffer. The encoder returns
    /// `error.CarryOverflow` only if the underlying allocator fails —
    /// this cap is a soft limit: when exceeded, the encoder force-
    /// encodes the carry and emits the resulting ids. See module doc.
    max_carry_bytes: usize = default_max_carry_bytes,
    /// Window passed to `pre_tokenizer.findSafeCut`. Larger windows
    /// can find more cut points but cost O(window) per `feed`.
    search_window: usize = default_search_window,
};

/// Persistent streaming-encode state. One per stream.
///
/// `feed(bytes, out)` appends to the carry, then encodes everything up
/// to the nearest safe cut, appending ids to `out` (any type that has
/// `append(allocator, TokenId)` and `appendSlice(allocator, []const
/// TokenId)` — `*std.ArrayList(TokenId)` is the obvious choice).
///
/// `finish(out)` flushes the remaining carry as a final encode.
///
/// The same allocator backs the carry and the internal scratch arena.
/// The caller owns whatever output container they pass in.
pub const StreamEncoder = struct {
    allocator: std.mem.Allocator,
    pipeline: *const Pipeline,
    scratch: ScratchArena,
    carry: std.ArrayList(u8),
    opts: Options,

    pub fn init(
        allocator: std.mem.Allocator,
        pipeline: *const Pipeline,
    ) StreamEncoder {
        return initWithOptions(allocator, pipeline, .{});
    }

    pub fn initWithOptions(
        allocator: std.mem.Allocator,
        pipeline: *const Pipeline,
        opts: Options,
    ) StreamEncoder {
        return .{
            .allocator = allocator,
            .pipeline = pipeline,
            .scratch = ScratchArena.init(allocator),
            .carry = .empty,
            .opts = opts,
        };
    }

    pub fn deinit(self: *StreamEncoder) void {
        self.carry.deinit(self.allocator);
        self.scratch.deinit();
    }

    /// Bytes currently buffered (not yet emitted).
    pub fn pendingBytes(self: *const StreamEncoder) usize {
        return self.carry.items.len;
    }

    /// Feed `bytes` to the encoder. Any ids that can now be safely
    /// emitted are appended to `out` via `out.append`/`out.appendSlice`.
    /// `out` must be `*std.ArrayList(TokenId)` (or anything with the
    /// same `append`/`appendSlice` shape — duck-typed).
    pub fn feed(
        self: *StreamEncoder,
        bytes: []const u8,
        out: anytype,
    ) !void {
        if (bytes.len == 0) return;
        try self.carry.appendSlice(self.allocator, bytes);
        try self.tryDrain(out, false);
    }

    /// Drop any held bytes into a final encode and emit them. Idempotent:
    /// callable twice in a row, second call is a no-op.
    pub fn finish(
        self: *StreamEncoder,
        out: anytype,
    ) !void {
        try self.tryDrain(out, true);
    }

    /// Encode `self.carry[0..cut]` through the pipeline, emit the ids,
    /// and shift `self.carry[cut..]` to the front of the carry.
    fn emitPrefix(
        self: *StreamEncoder,
        out: anytype,
        cut: usize,
    ) !void {
        if (cut == 0) return;
        const prefix = self.carry.items[0..cut];
        const ids = try self.pipeline.encodeWithScratch(self.allocator, prefix, &self.scratch);
        defer self.allocator.free(ids);
        try out.appendSlice(self.allocator, ids);

        const remaining = self.carry.items.len - cut;
        if (remaining > 0) {
            // Shift carry[cut..] to the front.
            std.mem.copyForwards(u8, self.carry.items[0..remaining], self.carry.items[cut..]);
        }
        self.carry.shrinkRetainingCapacity(remaining);
    }

    /// Drain emittable bytes. When `is_final` is true, flush everything
    /// remaining as a final encode.
    fn tryDrain(
        self: *StreamEncoder,
        out: anytype,
        is_final: bool,
    ) !void {
        if (is_final) {
            // Final flush: encode the entire carry, regardless of safe
            // cuts (this is the natural end-of-stream — any partial
            // codepoint left here is a bug in the upstream feeder, and
            // the model's UTF-8 handling will absorb it as best it can).
            if (self.carry.items.len > 0) {
                try self.emitPrefix(out, self.carry.items.len);
            }
            return;
        }

        if (self.carry.items.len == 0) return;

        // Find the largest safe cut <= carry.len. We never emit the very
        // last byte (it may be the start of a span that's still growing).
        // Search window is bounded by `opts.search_window`.
        const carry_len = self.carry.items.len;
        // `desired = carry_len - 1` searches backwards from the position
        // just before the last byte. findSafeCut will snap to the nearest
        // boundary <= desired.
        const desired = if (carry_len > 0) carry_len - 1 else 0;
        const maybe_cut = self.pipeline.pre_tokenizer.findSafeCut(
            self.carry.items,
            desired,
            self.opts.search_window,
        );

        var cut: usize = 0;
        if (maybe_cut) |c| {
            cut = c;
        } else if (carry_len >= self.opts.max_carry_bytes) {
            // Carry has grown past its soft cap and we still couldn't find
            // a safe cut in `search_window` bytes. Force a UTF-8-codepoint-
            // safe cut at the very end so we don't unbounded-grow.
            var p = carry_len;
            while (p > 0 and (self.carry.items[p - 1] & 0xC0) == 0x80) p -= 1;
            // back off the lead byte too if we're sitting mid-codepoint
            if (p > 0 and (self.carry.items[p - 1] & 0xC0) == 0xC0) p -= 1;
            cut = p;
        }
        if (cut > 0) try self.emitPrefix(out, cut);
    }
};

// === Tests ============================================================

const Normalizer = @import("normalizer.zig").Normalizer;
const PreTokenizer = @import("pretok.zig").PreTokenizer;
const Model = @import("model.zig").Model;
const Decoder = @import("decoder.zig").Decoder;
const added_tokens_mod = @import("added_tokens.zig");

test "StreamEncoder: 3-chunk feed matches single-shot encode (cl100k+bpe)" {
    const a = std.testing.allocator;

    // Build a small cl100k-style BPE vocab matching the existing
    // pipeline test: 256 byte ids + a few extra merges so "hello",
    // " world" tokenize to a couple of ids.
    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(a);
    const b64 = std.base64.standard.Encoder;
    var enc_buf: [16]u8 = undefined;
    var rank: u32 = 0;
    var b: u32 = 0;
    while (b < 256) : (b += 1) {
        const byte: [1]u8 = .{@intCast(b)};
        const encoded = b64.encode(&enc_buf, &byte);
        try src.print(a, "{s} {d}\n", .{ encoded, rank });
        rank += 1;
    }
    const extra = [_][]const u8{
        "he",     "hel",    "hell",   "hello",
        " w",     " wo",    " wor",   " worl",
        " world", " q",     " qu",    " qui",
        " quic",  " quick", " b",     " br",
        " bro",   " brow",  " brown", " f",
        " fo",    " fox",
    };
    for (extra) |bytes| {
        const encoded = b64.encode(&enc_buf, bytes);
        try src.print(a, "{s} {d}\n", .{ encoded, rank });
        rank += 1;
    }

    var bpe = try Bpe.loadTiktokenBytes(a, src.items);
    defer bpe.deinit();

    var v = Vocab.empty(a);
    defer v.deinit();

    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .cl100k,
        .model = .{ .bpe = &bpe },
        .decoder = .concat,
        .vocab = &v,
    };

    const full = "hello world the quick brown fox";
    const want = try pipe.encode(a, full);
    defer a.free(want);

    // Feed in three deliberately uneven chunks.
    var enc = StreamEncoder.init(a, &pipe);
    defer enc.deinit();
    var got: std.ArrayList(TokenId) = .empty;
    defer got.deinit(a);

    try enc.feed(full[0..11], &got); // "hello world"
    try enc.feed(full[11..21], &got); // " the quick"
    try enc.feed(full[21..], &got); // " brown fox"
    try enc.finish(&got);

    try std.testing.expectEqualSlices(TokenId, want, got.items);
}

test "StreamEncoder: mid-UTF-8 codepoint bytes do not produce garbage ids" {
    const a = std.testing.allocator;

    // byte_id model so we can predict ids exactly.
    var v = Vocab.empty(a);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };

    // "héllo" — the é is a 2-byte UTF-8 sequence (0xC3 0xA9). We feed
    // up to the first byte of é, then the rest.
    const full = "h\xC3\xA9llo";
    const want = try pipe.encode(a, full);
    defer a.free(want);

    var enc = StreamEncoder.init(a, &pipe);
    defer enc.deinit();
    var got: std.ArrayList(TokenId) = .empty;
    defer got.deinit(a);

    // Feed everything up to and INCLUDING the leading byte of the é.
    try enc.feed(full[0..2], &got);
    // The half-codepoint must NOT have been emitted yet. byte_id with
    // identity pretok will emit at most one byte of safe content (the
    // 'h') before the carry; the trailing 0xC3 must be deferred.
    // We assert nothing past 'h' has been emitted.
    try std.testing.expect(got.items.len <= 1);
    if (got.items.len == 1) {
        try std.testing.expectEqual(@as(TokenId, 'h'), got.items[0]);
    }

    try enc.feed(full[2..], &got);
    try enc.finish(&got);
    try std.testing.expectEqualSlices(TokenId, want, got.items);
}

test "StreamEncoder: empty feed + finish produces zero ids" {
    const a = std.testing.allocator;

    var v = Vocab.empty(a);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };

    var enc = StreamEncoder.init(a, &pipe);
    defer enc.deinit();
    var got: std.ArrayList(TokenId) = .empty;
    defer got.deinit(a);

    try enc.feed("", &got);
    try enc.finish(&got);
    try std.testing.expectEqual(@as(usize, 0), got.items.len);

    // Idempotent finish.
    try enc.finish(&got);
    try std.testing.expectEqual(@as(usize, 0), got.items.len);
}

test "StreamEncoder: many-tiny-chunks matches single-shot" {
    const a = std.testing.allocator;
    var v = Vocab.empty(a);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };

    const full = "the quick brown fox jumps over the lazy dog";
    const want = try pipe.encode(a, full);
    defer a.free(want);

    var enc = StreamEncoder.init(a, &pipe);
    defer enc.deinit();
    var got: std.ArrayList(TokenId) = .empty;
    defer got.deinit(a);

    // Feed one byte at a time.
    for (full) |c| {
        const one = [_]u8{c};
        try enc.feed(&one, &got);
    }
    try enc.finish(&got);
    try std.testing.expectEqualSlices(TokenId, want, got.items);
}
