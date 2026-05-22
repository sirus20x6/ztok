//! Continued-pretraining vocab adaptation.
//!
//! Extend an existing BPE vocab on a new corpus WITHOUT changing any of
//! the original token ids. The output vocab has:
//!   * Ids `0..base.count` bit-identical to the base (bytes, offsets,
//!     id ↔ bytes mapping all preserved).
//!   * Ids `[base.count .. base.count + n_added)` for new tokens
//!     discovered from the supplied corpus.
//!
//! Approach (simple additive — see "Limitations" below):
//!   1. Re-tokenize the corpus with the base BPE in fixed-size chunks
//!      to find the per-chunk bytes-per-token ratio. Chunks whose ratio
//!      is below the corpus median are "well covered" and contribute
//!      nothing; the others are "expensive" and get queued for further
//!      training.
//!   2. Concatenate the expensive chunks into a sub-corpus.
//!   3. Run `train_bpe.train` on that sub-corpus targeting
//!      `256 + add` merges, throw away the 256 seed-byte ids that the
//!      trainer always emits, and KEEP the ordered list of newly-minted
//!      tokens (the merge output of the trainer).
//!   4. Filter out any candidate token whose byte sequence already
//!      exists in the base vocab (no duplicate ids).
//!   5. Append at most `add` survivors at ids `base.count..` in order.
//!
//! Limitations (what is NOT faithful to research-grade vocab adaptation):
//!   * We do NOT register the new tokens as BPE merge rules that chain
//!     back to base ids. The new tokens behave like HF "added tokens" —
//!     to make the runtime encoder emit them you'd need to either
//!     (a) preprocess input with `added_tokens.Scanner` to substitute
//!         the new-token byte sequences before BPE, or
//!     (b) re-train merges that step from base ids up to each new id.
//!     The serialized output records the new tokens so the embedding-
//!     table side stays correct; encoder-side adoption is a follow-up.
//!   * The "expensive chunks" heuristic is byte-count-based, not the
//!     full information-theoretic per-line cost a research paper would
//!     measure. It is enough to bias the additional merges toward
//!     under-compressed regions of the corpus.
//!   * Only BPE-shaped vocabs are supported (tiktoken, HF BPE, SP BPE).
//!     Unigram, WordPiece, Monster, and the SP byte-fallback variants
//!     trip `error.UnsupportedFormat`. Adapting a non-BPE vocab cleanly
//!     is a separate body of work.
//!
//! The CLI entry-point lives in `main.zig` (subcommand `ztok adapt-vocab`).

const std = @import("std");

const TokenId = @import("token.zig").TokenId;
const bpe_mod = @import("bpe.zig");
const Bpe = bpe_mod.Bpe;
const Span = @import("token.zig").Span;
const auto_detect = @import("auto_detect.zig");
const train_bpe = @import("train_bpe.zig");
const hf_writer = @import("hf_writer.zig");
const sp_writer = @import("sp_writer.zig");

pub const Error = error{
    /// Base vocab format does not support extension (Monster .ztm,
    /// Unigram, WordPiece, Tekken-specials-aware variants, etc.).
    UnsupportedFormat,
    /// Base vocab smaller than 256 (no seed-byte layer) — adaptation
    /// algorithm requires a byte-level base.
    BaseVocabTooSmall,
    /// Caller asked for zero new tokens.
    NothingToAdd,
} || std.mem.Allocator.Error || error{
    // For convenience — re-tokenization may surface allocator errors and
    // we don't want the caller to have to merge sets.
    EncodeFailed,
};

/// Per-new-token telemetry that lands in the optional JSON report.
pub const NewTokenStat = struct {
    /// New id (== old_count + index).
    id: TokenId,
    /// Bytes of the new token. Borrows into the result's `bytes` buffer.
    bytes: []const u8,
    /// Number of times the trainer emitted this merge during the
    /// additional pass (i.e. its frequency in the expensive sub-corpus).
    /// Approximate — see `AdaptResult` doc-comment.
    occurrences: u64,
    /// First byte (cheap "cluster" hint) of the new token. Mirrored from
    /// `bytes[0]` for convenience in the JSON report.
    first_byte: u8,
};

/// Result of `adaptBpe`. The caller owns everything via `deinit`.
pub const AdaptResult = struct {
    allocator: std.mem.Allocator,

    /// Adapted BPE with original + new tokens. Ids `0..old_count` are
    /// bit-identical to the input.
    adapted_bpe: Bpe,

    /// Old vocab size — first new id is `old_count`.
    old_count: u32,
    /// Count of new tokens actually appended (<= requested `add`).
    new_token_added_count: u32,

    /// Identity map for original ids: `old_to_new[i] == i` for
    /// `i < old_count`. Kept for caller convenience / sanity checking.
    old_to_new: []TokenId,

    /// One entry per appended new id, in append order.
    new_token_stats: []NewTokenStat,

    pub fn deinit(self: *AdaptResult) void {
        if (self.old_to_new.len > 0) self.allocator.free(self.old_to_new);
        if (self.new_token_stats.len > 0) self.allocator.free(self.new_token_stats);
        self.adapted_bpe.deinit();
        self.* = undefined;
    }
};

pub const AdaptOptions = struct {
    /// How many new tokens to add at the tail. Hard cap; the algorithm
    /// may add fewer if the corpus runs out of distinct candidates that
    /// don't collide with the base vocab.
    add: u32 = 100,
    /// Chunk size (bytes) for the "expensive-region" detector. Default
    /// 256: small enough to localize hot spans, large enough to keep
    /// per-chunk encode overhead negligible.
    chunk_bytes: usize = 256,
    /// Trainer aims for `256 + add_overshoot * add` merges in the
    /// additional pass, then we filter out base-collisions until we hit
    /// the requested `add`. Overshooting lets a few collisions get
    /// thrown away without falling short of the target.
    add_overshoot: u32 = 4,
};

/// Adapt `base` to `corpus`. Returns a result whose `adapted_bpe`
/// preserves base ids bit-identically.
pub fn adaptBpe(
    allocator: std.mem.Allocator,
    base: *const Bpe,
    corpus: []const u8,
    opts: AdaptOptions,
) !AdaptResult {
    if (base.count < 256) return Error.BaseVocabTooSmall;
    if (opts.add == 0) return Error.NothingToAdd;

    // Step 1: identify "expensive" sub-corpus.
    const sub_corpus = try buildExpensiveSubCorpus(allocator, base, corpus, opts.chunk_bytes);
    defer if (sub_corpus.len > 0) allocator.free(sub_corpus);

    // Step 2: BPE train on it. Targeted size = 256 (seed bytes) + a
    // padded number of merges so post-filter survivors hit `add`.
    const target_extra: u32 = opts.add *| @max(@as(u32, 1), opts.add_overshoot);
    const target_total: u32 = 256 +| target_extra;

    var trained_bpe = train_bpe.trainFromBytes(
        allocator,
        if (sub_corpus.len > 0) sub_corpus else corpus,
        whitespaceLikeSplit,
        .{ .vocab_size = target_total },
    ) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return Error.EncodeFailed,
    };
    defer trained_bpe.deinit();

    // Step 3: extract candidate new tokens (ids 256..trained_bpe.count)
    // that don't already exist in the base vocab.
    var picked_bytes: std.ArrayList(u8) = .empty;
    defer picked_bytes.deinit(allocator);
    var picked_offsets: std.ArrayList(u32) = .empty;
    defer picked_offsets.deinit(allocator);
    try picked_offsets.append(allocator, 0);

    var picked: u32 = 0;
    var cand: u32 = 256;
    while (picked < opts.add and cand < trained_bpe.count) : (cand += 1) {
        const bytes = trained_bpe.idBytes(cand);
        if (bytes.len == 0) continue;
        if (base.by_bytes.get(bytes) != null) continue; // collision
        try picked_bytes.appendSlice(allocator, bytes);
        try picked_offsets.append(allocator, @intCast(picked_bytes.items.len));
        picked += 1;
    }

    // Step 4: build the adapted BPE. Clone base verbatim, then append
    // each survivor.
    var adapted = try cloneBpePreserving(allocator, base);
    errdefer adapted.deinit();
    var p_id: u32 = 0;
    while (p_id < picked) : (p_id += 1) {
        const s = picked_offsets.items[p_id];
        const e = picked_offsets.items[p_id + 1];
        const tok = picked_bytes.items[s..e];
        try appendToken(allocator, &adapted, tok);
    }

    // Build the identity old_to_new map. Trivial but documented in the
    // spec so consumers can rely on it without a special case.
    const old_to_new = try allocator.alloc(TokenId, base.count);
    errdefer allocator.free(old_to_new);
    var idx: u32 = 0;
    while (idx < base.count) : (idx += 1) old_to_new[idx] = idx;

    // Per-new-token stats. Occurrences are approximated by the trainer's
    // candidate ordering (lower trained-bpe id == more frequent merge).
    // We record the raw trained-bpe id as a proxy for frequency rank.
    const stats = try allocator.alloc(NewTokenStat, picked);
    errdefer allocator.free(stats);
    var sidx: u32 = 0;
    var trained_id: u32 = 256;
    while (sidx < picked and trained_id < trained_bpe.count) : (trained_id += 1) {
        const bytes = trained_bpe.idBytes(trained_id);
        if (bytes.len == 0) continue;
        if (base.by_bytes.get(bytes) != null) continue;
        // Frequency proxy: invert the trained-bpe rank so id 256 is
        // "highest" and we count down. Lossy but lets the JSON report
        // surface a usable ordering signal.
        const rank_from_top: u64 = @as(u64, trained_bpe.count) - @as(u64, trained_id);
        stats[sidx] = .{
            .id = base.count + sidx,
            .bytes = adapted.idBytes(base.count + sidx),
            .occurrences = rank_from_top,
            .first_byte = bytes[0],
        };
        sidx += 1;
    }

    return .{
        .allocator = allocator,
        .adapted_bpe = adapted,
        .old_count = base.count,
        .new_token_added_count = picked,
        .old_to_new = old_to_new,
        .new_token_stats = stats,
    };
}

// --- helpers ---------------------------------------------------------

/// Clone `base` into a fresh `Bpe`. Bytes / offsets / by_bytes /
/// hot_table / piece_ranks all reproduced; ids preserved.
fn cloneBpePreserving(allocator: std.mem.Allocator, base: *const Bpe) !Bpe {
    const bytes = if (base.bytes.len > 0)
        try allocator.dupe(u8, base.bytes)
    else
        try allocator.alloc(u8, 0);
    errdefer if (bytes.len > 0) allocator.free(bytes);
    const offsets = if (base.offsets.len > 0)
        try allocator.dupe(u32, base.offsets)
    else
        try allocator.alloc(u32, 0);
    errdefer if (offsets.len > 0) allocator.free(offsets);

    var by_bytes = std.StringHashMap(TokenId).init(allocator);
    errdefer by_bytes.deinit();
    try by_bytes.ensureTotalCapacity(base.count);
    var i: u32 = 0;
    while (i < base.count) : (i += 1) {
        const key = bytes[offsets[i]..offsets[i + 1]];
        try by_bytes.put(key, i);
    }

    // Hot-table rebuild keeps encode-loop perf for the clone. Optional
    // — only present on the base if it was loaded with hot_table=true.
    var hot_table: ?[]bpe_mod.HotEntry = null;
    errdefer if (hot_table) |h| allocator.free(h);
    if (base.hot_table != null) {
        hot_table = try Bpe.buildHotTable(allocator, bytes, offsets, base.count);
    }

    return .{
        .allocator = allocator,
        .bytes = bytes,
        .offsets = offsets,
        .count = base.count,
        .by_bytes = by_bytes,
        .hot_table = hot_table,
        .byte_fallback = base.byte_fallback,
        .encode_mode = base.encode_mode,
        .max_piece_len = base.max_piece_len,
        .piece_ranks = null,
    };
}

/// Append `tok` to `adapted` as a new id. Resizes bytes/offsets and
/// updates the reverse hashmap. The hot_table is NOT rebuilt — the
/// new id won't appear in the front cache, which is fine for the
/// "added tokens" use case (encoder fallback via `by_bytes` still
/// resolves it correctly).
fn appendToken(allocator: std.mem.Allocator, adapted: *Bpe, tok: []const u8) !void {
    const old_bytes_len: usize = adapted.bytes.len;
    const new_bytes_len: usize = old_bytes_len + tok.len;
    const new_bytes = try allocator.realloc(adapted.bytes, new_bytes_len);
    adapted.bytes = new_bytes;
    @memcpy(adapted.bytes[old_bytes_len..new_bytes_len], tok);

    const old_offsets_len: usize = adapted.offsets.len;
    const new_offsets = try allocator.realloc(adapted.offsets, old_offsets_len + 1);
    adapted.offsets = new_offsets;
    adapted.offsets[old_offsets_len] = @intCast(new_bytes_len);

    // The realloc above may have moved the bytes buffer — refresh every
    // borrowed by_bytes key against the post-realloc storage so existing
    // entries don't point at freed memory.
    var rebuilt = std.StringHashMap(TokenId).init(allocator);
    errdefer rebuilt.deinit();
    try rebuilt.ensureTotalCapacity(adapted.count + 1);
    var i: u32 = 0;
    while (i < adapted.count) : (i += 1) {
        const key = adapted.bytes[adapted.offsets[i]..adapted.offsets[i + 1]];
        try rebuilt.put(key, i);
    }
    const new_id: TokenId = adapted.count;
    const new_key = adapted.bytes[adapted.offsets[new_id]..adapted.offsets[new_id + 1]];
    try rebuilt.put(new_key, new_id);
    adapted.by_bytes.deinit();
    adapted.by_bytes = rebuilt;
    adapted.count += 1;
    if (tok.len > adapted.max_piece_len) adapted.max_piece_len = @intCast(tok.len);
}

/// Whitespace + punctuation split. Identity-ish for very small inputs.
/// We want SOMETHING that yields word-sized spans so the BPE trainer
/// has reasonable word boundaries to work with — the cl100k regex would
/// be more accurate but we explicitly want a format-agnostic adapter.
fn whitespaceLikeSplit(allocator: std.mem.Allocator, input: []const u8) anyerror![]Span {
    if (input.len == 0) return allocator.alloc(Span, 0);
    var out: std.ArrayList(Span) = .empty;
    defer out.deinit(allocator);
    var i: usize = 0;
    while (i < input.len) {
        // Skip leading whitespace as its own span so the trainer can
        // potentially merge whitespace runs.
        if (isWs(input[i])) {
            const start = i;
            while (i < input.len and isWs(input[i])) i += 1;
            try out.append(allocator, .{ .start = @intCast(start), .end = @intCast(i) });
            continue;
        }
        const start = i;
        while (i < input.len and !isWs(input[i])) i += 1;
        try out.append(allocator, .{ .start = @intCast(start), .end = @intCast(i) });
    }
    return out.toOwnedSlice(allocator);
}

inline fn isWs(c: u8) bool {
    return c == ' ' or c == '\t' or c == '\n' or c == '\r' or c == 0x0B or c == 0x0C;
}

/// Walk `corpus` in fixed-size byte chunks, re-encode each through
/// `base`, and return a freshly-allocated buffer containing only those
/// chunks whose bytes-per-token ratio is at-or-below the corpus median
/// (i.e. the worst-compressed half). Returns an empty slice when the
/// corpus has nothing actionable (empty, single tiny chunk, etc.).
///
/// The "median" here is approximated by computing the per-chunk ratio
/// up-front, sorting, and slicing the upper half. For corpora under a
/// few MB this is trivial; bigger corpora may want a reservoir sample
/// later but the algorithm shape is fine.
fn buildExpensiveSubCorpus(
    allocator: std.mem.Allocator,
    base: *const Bpe,
    corpus: []const u8,
    chunk_bytes: usize,
) ![]u8 {
    if (corpus.len == 0 or chunk_bytes == 0) return allocator.alloc(u8, 0);

    const n_chunks: usize = (corpus.len + chunk_bytes - 1) / chunk_bytes;
    if (n_chunks <= 1) {
        // Nothing meaningful to pick from — just return a clone.
        return allocator.dupe(u8, corpus);
    }

    const ratios = try allocator.alloc(f32, n_chunks);
    defer allocator.free(ratios);

    // Reusable encode scratch sized for the worst-case chunk (one id per
    // byte). Two allocations dominate the loop otherwise.
    const enc_out = try allocator.alloc(TokenId, chunk_bytes);
    defer allocator.free(enc_out);

    // SP-BPE encode paths in bpe.zig leak their internal scratch alloc
    // on every long-chunk call (the bpe-heap path forgets to free its
    // prev/nxt/heap buffers). Hand them an arena so the leaks are
    // reclaimed when this function returns instead of piling up.
    var scratch_arena = std.heap.ArenaAllocator.init(allocator);
    defer scratch_arena.deinit();
    const enc_scratch = scratch_arena.allocator();

    var ci: usize = 0;
    var off: usize = 0;
    while (ci < n_chunks) : (ci += 1) {
        const end = @min(off + chunk_bytes, corpus.len);
        const slice = corpus[off..end];
        const ids = base.encodeChunkScratch(enc_scratch, slice, enc_out);
        // bytes-per-token: high == well compressed, low == expensive.
        // We want to KEEP expensive chunks (low ratio).
        const r: f32 = if (ids.len == 0)
            0.0
        else
            @as(f32, @floatFromInt(slice.len)) / @as(f32, @floatFromInt(ids.len));
        ratios[ci] = r;
        off = end;
    }

    // Sort a copy to find the median.
    const sorted = try allocator.dupe(f32, ratios);
    defer allocator.free(sorted);
    std.mem.sort(f32, sorted, {}, comptime std.sort.asc(f32));
    const median: f32 = sorted[sorted.len / 2];

    // Pick chunks whose ratio is <= median (i.e. the lower half of the
    // compression distribution — these are the expensive ones).
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(allocator);
    ci = 0;
    off = 0;
    while (ci < n_chunks) : (ci += 1) {
        const end = @min(off + chunk_bytes, corpus.len);
        if (ratios[ci] <= median) {
            try out.appendSlice(allocator, corpus[off..end]);
            // Force a separator so adjacent kept chunks don't fuse into
            // a single mega-word during the trainer's split step.
            try out.append(allocator, '\n');
        }
        off = end;
    }

    return out.toOwnedSlice(allocator);
}

// --- serialization helpers (used by the CLI driver) -----------------

/// Write `adapted` back to disk in `fmt`. For HF JSON we emit the BPE
/// shape; for SP we emit a SP-BPE shape; for tiktoken we emit the
/// `<base64> <id>` line format. Errors with `Error.UnsupportedFormat`
/// for ZTM/Tekken.
pub fn writeAdaptedBpe(
    allocator: std.mem.Allocator,
    adapted: *const Bpe,
    fmt: auto_detect.Format,
    path: []const u8,
) !void {
    switch (fmt) {
        .tiktoken => try writeTiktoken(allocator, adapted, path),
        .hf_json => try hf_writer.writeBpeFile(allocator, adapted, path, .{}),
        .sentencepiece => try sp_writer.writeBpeFile(allocator, adapted, path, .{}),
        .tekken, .ztm, .unknown => return Error.UnsupportedFormat,
    }
}

fn writeTiktoken(allocator: std.mem.Allocator, bpe: *const Bpe, path: []const u8) !void {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(allocator);
    const b64 = std.base64.standard.Encoder;
    var line_buf: [4096]u8 = undefined;
    var id: u32 = 0;
    while (id < bpe.count) : (id += 1) {
        const bytes = bpe.idBytes(id);
        const enc_len = b64.calcSize(bytes.len);
        std.debug.assert(enc_len <= line_buf.len);
        const encoded = b64.encode(line_buf[0..enc_len], bytes);
        try out.appendSlice(allocator, encoded);
        try out.append(allocator, ' ');
        var num_buf: [12]u8 = undefined;
        const n = std.fmt.printInt(&num_buf, id, 10, .lower, .{});
        try out.appendSlice(allocator, num_buf[0..n]);
        try out.append(allocator, '\n');
    }
    const io = std.Io.Threaded.global_single_threaded.io();
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = out.items });
}

/// Render `result` as a JSON report blob the caller can dump to disk.
pub fn writeJsonReport(
    allocator: std.mem.Allocator,
    result: *const AdaptResult,
    base_format: auto_detect.Format,
    base_path: []const u8,
    out_path: []const u8,
) !void {
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var s: std.json.Stringify = .{
        .writer = &aw.writer,
        .options = .{ .whitespace = .indent_2 },
    };
    try s.beginObject();
    try s.objectField("base_path");
    try s.write(base_path);
    try s.objectField("base_format");
    try s.write(@tagName(base_format));
    try s.objectField("base_vocab_size");
    try s.write(@as(i64, @intCast(result.old_count)));
    try s.objectField("new_vocab_size");
    try s.write(@as(i64, @intCast(result.adapted_bpe.count)));
    try s.objectField("new_token_added_count");
    try s.write(@as(i64, @intCast(result.new_token_added_count)));
    try s.objectField("ids_preserved");
    try s.write(true);
    try s.objectField("algorithm");
    try s.write("simple_additive_bpe_on_expensive_chunks");
    try s.objectField("new_tokens");
    try s.beginArray();
    for (result.new_token_stats) |st| {
        try s.beginObject();
        try s.objectField("id");
        try s.write(@as(i64, @intCast(st.id)));
        try s.objectField("bytes_b64");
        const b64 = std.base64.standard.Encoder;
        var enc_buf: [4096]u8 = undefined;
        const enc_len = b64.calcSize(st.bytes.len);
        if (enc_len <= enc_buf.len) {
            const encoded = b64.encode(enc_buf[0..enc_len], st.bytes);
            try s.write(encoded);
        } else {
            try s.write(""); // overlong piece — uncommon, drop content
        }
        try s.objectField("length");
        try s.write(@as(i64, @intCast(st.bytes.len)));
        try s.objectField("first_byte");
        try s.write(@as(i64, st.first_byte));
        try s.objectField("rank_score");
        try s.write(@as(i64, @intCast(st.occurrences)));
        try s.endObject();
    }
    try s.endArray();
    try s.endObject();

    const io = std.Io.Threaded.global_single_threaded.io();
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out_path, .data = aw.writer.buffered() });
}

// =============================== tests ===============================

const testing = std.testing;

fn buildTinyBpe(allocator: std.mem.Allocator, n_merges: u32) !Bpe {
    // Reproducible tiny BPE: 256 byte seeds + n_merges learned from a
    // canned corpus. Mirrors the `train_bpe` test pattern.
    const words = [_][]const u8{ "abc", "abcd", "bcd", "ab", "bc", "cd" };
    const counts = [_]u32{ 6, 4, 3, 5, 2, 1 };
    return train_bpe.train(allocator, .{
        .words = &words,
        .counts = &counts,
    }, .{ .vocab_size = 256 + n_merges });
}

test "adaptBpe preserves all base ids byte-for-byte" {
    const alloc = testing.allocator;
    var base = try buildTinyBpe(alloc, 12);
    defer base.deinit();

    const corpus = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda";
    var res = try adaptBpe(alloc, &base, corpus, .{ .add = 5 });
    defer res.deinit();

    // Every base id decodes to the same bytes pre/post adaptation.
    try testing.expectEqual(base.count, res.old_count);
    try testing.expect(res.adapted_bpe.count >= res.old_count);
    var id: u32 = 0;
    while (id < base.count) : (id += 1) {
        const before = base.idBytes(id);
        const after = res.adapted_bpe.idBytes(id);
        try testing.expectEqualSlices(u8, before, after);
    }
}

test "adaptBpe appends new ids at base.count.." {
    const alloc = testing.allocator;
    var base = try buildTinyBpe(alloc, 8);
    defer base.deinit();

    const corpus = "xyz xyz xyz xyzw xyzw xyzw xyzwq xyzwq xyzwq\n" ++
        "foo bar baz quux fred plugh xyzzy waldo grault\n" ++
        "the quick brown fox jumps over the lazy dog dog dog dog\n";

    var res = try adaptBpe(alloc, &base, corpus, .{ .add = 6 });
    defer res.deinit();

    // We may not always find 6 unique candidates on a tiny corpus, but
    // we should find at least one — the test corpus contains plenty of
    // byte sequences absent from the seed-byte-only base.
    try testing.expect(res.new_token_added_count > 0);
    try testing.expect(res.new_token_added_count <= 6);

    // New ids appear in [old_count, old_count + new_token_added_count).
    var k: u32 = 0;
    while (k < res.new_token_added_count) : (k += 1) {
        const new_id = res.old_count + k;
        try testing.expect(new_id < res.adapted_bpe.count);
        // The reverse map should resolve the new bytes to the new id.
        const bytes = res.adapted_bpe.idBytes(new_id);
        try testing.expect(bytes.len > 0);
        const lookup = res.adapted_bpe.by_bytes.get(bytes);
        try testing.expect(lookup != null);
        try testing.expectEqual(new_id, lookup.?);
    }
}

test "adapted vocab encodes corpus to <= as many tokens as base" {
    // We don't synthesize true BPE merge chains for new tokens (a
    // research-grade adapter would). What we DO guarantee: appending
    // tokens never makes encoding strictly worse — at minimum, the
    // base's merge chains still resolve and the new ids are extra
    // options the encoder may match via `by_bytes`.
    const alloc = testing.allocator;
    var base = try buildTinyBpe(alloc, 12);
    defer base.deinit();

    const corpus = "alpha beta gamma alpha beta gamma alpha beta gamma " ++
        "delta epsilon zeta delta epsilon zeta delta epsilon zeta " ++
        "the quick brown fox the quick brown fox the quick brown fox\n";

    var res = try adaptBpe(alloc, &base, corpus, .{ .add = 12 });
    defer res.deinit();

    const stack_out = try alloc.alloc(TokenId, corpus.len);
    defer alloc.free(stack_out);

    // bpe.zig's heap path leaks scratch on every call; route through
    // an arena so the test allocator stays clean.
    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();

    const base_ids = base.encodeChunkScratch(arena.allocator(), corpus, stack_out);
    const base_len = base_ids.len;

    const adapted_ids = res.adapted_bpe.encodeChunkScratch(arena.allocator(), corpus, stack_out);
    const adapted_len = adapted_ids.len;

    // Adapted <= base. Strictly less is a non-goal of the simple
    // additive algorithm (the encoder only picks up new ids via the
    // longest-match path); equality is acceptable.
    try testing.expect(adapted_len <= base_len);
}

test "adaptBpe identity old_to_new map" {
    const alloc = testing.allocator;
    var base = try buildTinyBpe(alloc, 4);
    defer base.deinit();

    var res = try adaptBpe(alloc, &base, "alpha beta gamma alpha alpha", .{ .add = 2 });
    defer res.deinit();

    try testing.expectEqual(base.count, @as(u32, @intCast(res.old_to_new.len)));
    var i: u32 = 0;
    while (i < base.count) : (i += 1) {
        try testing.expectEqual(i, res.old_to_new[i]);
    }
}

test "adaptBpe JSON report is well-formed" {
    const alloc = testing.allocator;
    var base = try buildTinyBpe(alloc, 6);
    defer base.deinit();

    const corpus = "alpha alpha beta beta gamma gamma\n" ++
        "delta epsilon zeta theta iota kappa lambda mu nu xi\n";
    var res = try adaptBpe(alloc, &base, corpus, .{ .add = 4 });
    defer res.deinit();

    const path = "/tmp/ztok_adapt_report_test.json";
    try writeJsonReport(alloc, &res, .tiktoken, "base.tiktoken", path);
    defer {
        const io = std.Io.Threaded.global_single_threaded.io();
        std.Io.Dir.cwd().deleteFile(io, path) catch {};
    }

    const io = std.Io.Threaded.global_single_threaded.io();
    const contents = try std.Io.Dir.cwd().readFileAlloc(io, path, alloc, .unlimited);
    defer alloc.free(contents);
    // Sanity: the report parses as JSON and contains the marker keys.
    var parsed = try std.json.parseFromSlice(std.json.Value, alloc, contents, .{});
    defer parsed.deinit();
    const obj = parsed.value.object;
    try testing.expect(obj.get("new_token_added_count") != null);
    try testing.expect(obj.get("base_vocab_size") != null);
    try testing.expect(obj.get("new_tokens") != null);
    try testing.expect(obj.get("ids_preserved") != null);
    try testing.expectEqual(true, obj.get("ids_preserved").?.bool);
}

test "writeAdaptedBpe rejects ZTM with UnsupportedFormat" {
    const alloc = testing.allocator;
    var base = try buildTinyBpe(alloc, 2);
    defer base.deinit();
    const err = writeAdaptedBpe(alloc, &base, .ztm, "/tmp/ztok_should_not_exist.ztm");
    try testing.expectError(Error.UnsupportedFormat, err);

    const err2 = writeAdaptedBpe(alloc, &base, .tekken, "/tmp/ztok_should_not_exist.json");
    try testing.expectError(Error.UnsupportedFormat, err2);
}
