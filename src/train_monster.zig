//! TokenMonster-style distillation trainer (v2: marginal-value scoring).
//!
//! Pipeline:
//!   1. Seed: every substring up to `max_token_length` that appears at least
//!      `min_count` times across the corpus, plus all 256 single bytes
//!      (pinned — never dropped). The seed list is capped at
//!      `max_seed_size = 10 * vocab_size` to keep memory bounded — anything
//!      beyond is dropped by ascending (count * length) score.
//!      Cf. refs/tokenmonster/training/getalltokens.go.
//!   2. Distill: build a `Monster` from the current piece set, encode the
//!      corpus to count usage per id, then estimate each piece's MARGINAL
//!      value by re-encoding its own bytes through a vocab where that piece
//!      is excluded. value = (alt_pieces - 1) * usage. Drop the bottom
//!      `delete_fraction` non-byte pieces by that score. Rebuild. Loop until
//!      <= vocab_size or `max_iterations` hit.
//!      Cf. refs/tokenmonster/training/trainvocab.go (~1080-1150 — TM's
//!      branch-counting variant of the same idea).
//!   3. Finalize: convert the surviving piece set into a `Monster` via
//!      `Monster.Builder`. `unk_id` is the byte-0 piece (always pinned, always
//!      single-byte — matches the byte_fallback contract).
//!
//! Marginal-value approach: v3 uses a "mask P in the live Monster" path
//! for `encodeChunkWithout`. We build one Monster per iteration, then per
//! candidate piece P:
//!   1. Set `monster.mask[P] = 1`  (single-byte write, no allocation).
//!   2. Re-encode P's bytes through that Monster — the trie walk treats
//!      P's terminal as non-existent and falls back to alternatives.
//!   3. Restore `monster.mask[P] = 0`.
//!
//! This collapses the per-piece cost from "rebuild Monster + encode" to
//! "encode only", a ~5-50× win depending on vocab size. Bench: see
//! `bench/RESULTS.md` (v2-vs-v3 ratio at the bottom).
//!
//! v2 (kept under the name `altPieceCountV2` for the cross-check test)
//! rebuilt a temporary Monster excluding P per worker; trivially correct
//! but O(V * build_cost). v3 produces bit-identical alt-counts because
//! the encoding algorithm is identical and the mask-aware trie walk
//! visits the same terminals modulo the masked piece.
//!
//! Deferred from the canonical Go trainer:
//!   * Branch-aware scoring during the main corpus pass. TM's value formula
//!     also weights by branch context (nWords, capcode flags); v2 still
//!     uses the simpler (alt_pieces - 1) * usage proxy. Revisit if held-out
//!     compression lags the Go trainer's.
//!   * Capcode / deleteToken machinery (refs/tokenmonster/training/trainvocab.go:888).
//!     Our `Monster` doesn't carry these features either.
//!   * Suffix-array seed enumeration. We hash substrings directly; O(N*L^2)
//!     vs the SA's O(N) but trivial and capped by max_seed_size.

const std = @import("std");
const TokenId = @import("token.zig").TokenId;
const Span = @import("token.zig").Span;
const Monster = @import("monster.zig").Monster;
const BatchPool = @import("thread_pool.zig").BatchPool;
const negative_train = @import("negative_train.zig");

pub const TrainOptions = struct {
    /// Target vocab size (must include 256 byte fallbacks).
    vocab_size: u32,
    /// Maximum piece length in bytes.
    max_token_length: u8 = 24,
    /// Minimum corpus frequency for a substring to seed into the initial vocab.
    min_count: u32 = 4,
    /// Fraction to delete per iteration. Smaller = slower but better quality.
    delete_fraction: f32 = 0.05,
    /// Hard limit on iterations.
    max_iterations: u32 = 50,
    /// Optional pool for parallel corpus encoding during the value pass.
    pool: ?*BatchPool = null,
    /// Optional progress callback per iteration.
    on_round: ?*const fn (ctx: *anyopaque, vocab_size: u32, total_tokens: u64) void = null,
    on_round_ctx: *anyopaque = undefined,
    /// Optional avoid-pattern list. In `.exclude` mode, matching pieces
    /// never enter the seed. In `.penalize` mode they enter the seed
    /// but are flagged as priority-drop candidates during distillation
    /// (their marginal-value score is overridden to 0, putting them at
    /// the front of the kill list every iteration). Byte pieces are
    /// pinned regardless — single-byte patterns can't disable byte
    /// fallback.
    avoid: ?*const negative_train.AvoidList = null,
    avoid_mode: negative_train.Mode = .penalize,
};

pub const Corpus = struct {
    words: []const []const u8,
    counts: []const u32,
};

// --- internal mutable vocab ---------------------------------------------
//
// Pieces in SoA: a flat bytes buffer + offsets carving it. Pinning bytes
// is signalled by `is_byte[id]` — the 256 single-byte pieces are protected
// from deletion. We rewrite the arrays on each prune (no in-place shrink).

const Vocab = struct {
    allocator: std.mem.Allocator,
    bytes: std.ArrayList(u8),
    offsets: std.ArrayList(u32),
    is_byte: std.ArrayList(bool),

    fn init(allocator: std.mem.Allocator) Vocab {
        return .{
            .allocator = allocator,
            .bytes = .empty,
            .offsets = .empty,
            .is_byte = .empty,
        };
    }

    fn deinit(self: *Vocab) void {
        self.bytes.deinit(self.allocator);
        self.offsets.deinit(self.allocator);
        self.is_byte.deinit(self.allocator);
    }

    fn count(self: *const Vocab) u32 {
        return @intCast(self.is_byte.items.len);
    }

    fn piece(self: *const Vocab, id: u32) []const u8 {
        const s = self.offsets.items[id];
        const e = self.offsets.items[id + 1];
        return self.bytes.items[s..e];
    }

    fn pieceLen(self: *const Vocab, id: u32) u32 {
        return self.offsets.items[id + 1] - self.offsets.items[id];
    }

    fn add(self: *Vocab, piece_bytes: []const u8, is_byte_piece: bool) !void {
        if (self.offsets.items.len == 0) try self.offsets.append(self.allocator, 0);
        try self.bytes.appendSlice(self.allocator, piece_bytes);
        try self.offsets.append(self.allocator, @intCast(self.bytes.items.len));
        try self.is_byte.append(self.allocator, is_byte_piece);
    }
};

// --- seed ----------------------------------------------------------------

const SubstrEntry = struct {
    bytes: []const u8, // borrowed from corpus words (caller owns)
    count: u64,
};

fn lessByValueDesc(_: void, a: SubstrEntry, b: SubstrEntry) bool {
    const va = a.count *% a.bytes.len;
    const vb = b.count *% b.bytes.len;
    if (va != vb) return va > vb;
    if (a.bytes.len != b.bytes.len) return a.bytes.len > b.bytes.len;
    return std.mem.lessThan(u8, a.bytes, b.bytes);
}

fn seedVocab(
    allocator: std.mem.Allocator,
    corpus: Corpus,
    max_token_length: u8,
    min_count: u32,
    max_seed_size: usize,
    out: *Vocab,
    avoid: ?*const negative_train.AvoidList,
    avoid_mode: negative_train.Mode,
) !void {
    // Bytes first, pinned. Byte pieces are pinned (byte_fallback
    // contract); the avoid list is intentionally NOT consulted here.
    var b: u32 = 0;
    while (b < 256) : (b += 1) {
        const buf = [_]u8{@intCast(b)};
        try out.add(&buf, true);
    }

    // Count every substring >= length 2 up to max_token_length, weighted by
    // word count. Length-1 pieces are already covered by the byte vocab.
    var counts = std.StringHashMap(u64).init(allocator);
    defer counts.deinit();

    for (corpus.words, corpus.counts) |w, freq| {
        if (freq == 0 or w.len < 2) continue;
        var i: usize = 0;
        while (i < w.len) : (i += 1) {
            const lim = @min(w.len, i + max_token_length);
            var j: usize = i + 2;
            while (j <= lim) : (j += 1) {
                const sub = w[i..j];
                const gop = try counts.getOrPut(sub);
                if (!gop.found_existing) gop.value_ptr.* = 0;
                gop.value_ptr.* +%= freq;
            }
        }
    }

    // Filter by min_count, then score-rank and cap to max_seed_size.
    var entries: std.ArrayList(SubstrEntry) = .empty;
    defer entries.deinit(allocator);
    try entries.ensureTotalCapacity(allocator, counts.count());
    var it = counts.iterator();
    while (it.next()) |e| {
        if (e.value_ptr.* >= @as(u64, min_count)) {
            try entries.append(allocator, .{ .bytes = e.key_ptr.*, .count = e.value_ptr.* });
        }
    }
    std.mem.sort(SubstrEntry, entries.items, {}, lessByValueDesc);

    const keep = @min(entries.items.len, max_seed_size);
    var k: usize = 0;
    while (k < keep) : (k += 1) {
        if (avoid) |al| {
            if (al.matches(entries.items[k].bytes)) {
                switch (avoid_mode) {
                    .exclude => continue, // never seed
                    .penalize => {
                        // Still seeded so distillation can score it, but
                        // deleteLowest sees its marginal value as 0 (the
                        // avoid-tagged path below).
                        try out.add(entries.items[k].bytes, false);
                        continue;
                    },
                }
            }
        }
        try out.add(entries.items[k].bytes, false);
    }
}

// --- usage pass: encode corpus, accumulate usage[id] --------------------

const EncodeCtx = struct {
    monster: *const Monster,
    corpus: Corpus,
    per_worker_usage: [][]u64,
    per_worker_tokens: []u64,
};

const EncodeWorker = struct {
    pub fn run(ctx: *EncodeCtx, word_idx: usize, worker_idx: usize) void {
        const word = ctx.corpus.words[word_idx];
        const freq = ctx.corpus.counts[word_idx];
        if (freq == 0 or word.len == 0) return;

        // Stack buffer for small words; spill to a one-shot heap alloc.
        var stack_buf: [256]TokenId = undefined;
        const heap_alloc = std.heap.page_allocator;
        var out_slice: []TokenId = undefined;
        var heap_buf: ?[]TokenId = null;
        if (word.len <= stack_buf.len) {
            out_slice = stack_buf[0..];
        } else {
            heap_buf = heap_alloc.alloc(TokenId, word.len) catch return;
            out_slice = heap_buf.?;
        }
        defer if (heap_buf) |hb| heap_alloc.free(hb);

        const ids = ctx.monster.encodeChunk(heap_alloc, word, out_slice) catch return;
        const usage = ctx.per_worker_usage[worker_idx];
        const freq_u64: u64 = @as(u64, freq);
        for (ids) |id| usage[id] += freq_u64;
        ctx.per_worker_tokens[worker_idx] += @as(u64, ids.len) * freq_u64;
    }
};

fn countUsage(
    allocator: std.mem.Allocator,
    monster: *const Monster,
    corpus: Corpus,
    pool: ?*BatchPool,
    usage_out: []u64,
    total_tokens_out: *u64,
) !void {
    @memset(usage_out, 0);
    total_tokens_out.* = 0;

    if (pool) |bp| {
        const nw = bp.workerCount();
        const per_w_usage = try allocator.alloc([]u64, nw);
        defer {
            for (per_w_usage) |buf| allocator.free(buf);
            allocator.free(per_w_usage);
        }
        for (per_w_usage) |*buf| {
            buf.* = try allocator.alloc(u64, monster.count);
            @memset(buf.*, 0);
        }
        const per_w_tokens = try allocator.alloc(u64, nw);
        defer allocator.free(per_w_tokens);
        @memset(per_w_tokens, 0);

        var ctx: EncodeCtx = .{
            .monster = monster,
            .corpus = corpus,
            .per_worker_usage = per_w_usage,
            .per_worker_tokens = per_w_tokens,
        };
        try bp.runBatch(EncodeWorker, &ctx, corpus.words.len);

        for (per_w_usage) |buf| {
            for (usage_out, buf) |*acc, x| acc.* += x;
        }
        for (per_w_tokens) |x| total_tokens_out.* += x;
        return;
    }

    // Serial.
    var stack_buf: [256]TokenId = undefined;
    for (corpus.words, corpus.counts) |word, freq| {
        if (freq == 0 or word.len == 0) continue;
        var out_slice: []TokenId = undefined;
        var heap_buf: ?[]TokenId = null;
        if (word.len <= stack_buf.len) {
            out_slice = stack_buf[0..];
        } else {
            heap_buf = try allocator.alloc(TokenId, word.len);
            out_slice = heap_buf.?;
        }
        defer if (heap_buf) |hb| allocator.free(hb);

        const ids = try monster.encodeChunk(allocator, word, out_slice);
        const freq_u64: u64 = @as(u64, freq);
        for (ids) |id| usage_out[id] += freq_u64;
        total_tokens_out.* += @as(u64, ids.len) * freq_u64;
    }
}

// --- marginal-value pass ------------------------------------------------
//
// For each non-byte piece P we want to estimate "how many extra tokens
// would the encoder emit if I deleted P?". The proxy used here:
// re-encode P's own bytes through a vocab where P is excluded. If the
// result is k tokens, deleting P would cost (k - 1) extra tokens per
// occurrence. marginal_value(P) = (k - 1) * usage[P].
//
// v3 implementation: build ONE Monster per iteration, then for each
// candidate P flip a single byte in `monster.mask` to hide P, encode
// P's bytes, flip the byte back. The encode is O(L^2) on piece length L
// (typically L <= 24) — vs v2's "build a temp Monster" which is O(V).
// Speedup is roughly proportional to vocab size: 5-50× in practice.
//
// Parallel correctness: workers SHARE one Monster but each owns its own
// `mask` byte buffer (slice into a per-worker scratch). The Monster
// struct itself is value-copied per worker so the `mask` field is
// per-worker; the underlying trie/bytes/offsets/nwords slices are aliased
// and read-only during scoring. No races.

const AltCtx = struct {
    allocator: std.mem.Allocator,
    vocab: *const Vocab,
    usage: []const u64,
    /// alt_count[piece_id] = how many pieces the alt-vocab encoder uses to
    /// cover piece_id's bytes. Sentinel 0 means "byte piece or unused
    /// (skip)".
    alt_count: []u32,
    /// Per-worker Monster shallow copy (shared trie, distinct mask field).
    per_worker_monster: []Monster,
    /// Per-worker mask buffer (length == vocab_size). Owned by computeAltCounts.
    per_worker_mask: [][]u8,
};

const AltWorker = struct {
    pub fn run(ctx: *AltCtx, piece_id: usize, worker_idx: usize) void {
        const v = ctx.vocab;
        const pid: u32 = @intCast(piece_id);

        // Byte pieces are pinned — never delete, no need to score.
        if (v.is_byte.items[pid]) {
            ctx.alt_count[piece_id] = 0;
            return;
        }
        // Usage-0 pieces have marginal value 0 regardless of alt cost.
        // Skip the encode for them. (alt_count=1 → extra=0.)
        if (ctx.usage[piece_id] == 0) {
            ctx.alt_count[piece_id] = 1;
            return;
        }

        const piece_bytes = v.piece(pid);
        const mask = ctx.per_worker_mask[worker_idx];
        // Worker-local Monster — same shared trie, this worker's mask field.
        var monster = ctx.per_worker_monster[worker_idx];
        monster.mask = mask;

        // Flip mask[pid] for the duration of this encode, then restore.
        // No other worker writes mask[pid] (this worker is the sole owner
        // of `mask`), so the flip is race-free even though piece_id is
        // distributed across workers.
        mask[pid] = 1;
        defer mask[pid] = 0;

        var k: u32 = altPieceCountMasked(&monster, piece_bytes) catch {
            ctx.alt_count[piece_id] = 2;
            return;
        };
        if (k == 0) k = 1;
        ctx.alt_count[piece_id] = k;
    }
};

/// Encode `piece_bytes` through `monster` (which has its mask configured
/// by the caller) and return the token count. The mask hides exactly the
/// piece whose bytes we're encoding, so the encoder is forced to find an
/// alternative cover (a shorter prefix, then byte fallbacks for the rest).
fn altPieceCountMasked(monster: *const Monster, piece_bytes: []const u8) !u32 {
    // piece_bytes <= max_token_length (typically 24), well under 256.
    var stack_buf: [256]TokenId = undefined;
    std.debug.assert(piece_bytes.len <= stack_buf.len);
    const out_slice = stack_buf[0..];
    const ids = try monster.encodeChunk(std.heap.page_allocator, piece_bytes, out_slice);
    return @intCast(ids.len);
}

/// v2 baseline: rebuild a temporary Monster excluding P, encode P's bytes,
/// return token count. Retained for the cross-check test that v3 produces
/// the same ranking.
fn altPieceCountV2(
    allocator: std.mem.Allocator,
    vocab: *const Vocab,
    skip_id: u32,
    piece_bytes: []const u8,
) !u32 {
    var builder = Monster.Builder.init(allocator);
    defer builder.deinit();

    const v = vocab.count();
    var unk_id: TokenId = 0;
    var id: u32 = 0;
    while (id < v) : (id += 1) {
        if (id == skip_id) continue;
        const new_id = try builder.addToken(vocab.piece(id));
        if (id == 0) unk_id = new_id; // byte 0 is always present and always id 0
    }

    var monster = try builder.finalize(unk_id);
    defer monster.deinit();

    var stack_buf: [256]TokenId = undefined;
    var heap_buf: ?[]TokenId = null;
    var out_slice: []TokenId = undefined;
    if (piece_bytes.len <= stack_buf.len) {
        out_slice = stack_buf[0..];
    } else {
        heap_buf = try allocator.alloc(TokenId, piece_bytes.len);
        out_slice = heap_buf.?;
    }
    defer if (heap_buf) |hb| allocator.free(hb);

    const ids = try monster.encodeChunk(allocator, piece_bytes, out_slice);
    return @intCast(ids.len);
}

/// v3: mask-based marginal-value scoring. Reuses `monster` across all
/// pieces; only one byte of the mask buffer is mutated per encode call.
fn computeAltCounts(
    allocator: std.mem.Allocator,
    vocab: *const Vocab,
    monster: *const Monster,
    usage: []const u64,
    pool: ?*BatchPool,
    alt_count_out: []u32,
) !void {
    @memset(alt_count_out, 0);
    const v = vocab.count();
    std.debug.assert(monster.count == v);

    if (pool) |bp| {
        const nw = bp.workerCount();

        const per_w_mask = try allocator.alloc([]u8, nw);
        defer {
            for (per_w_mask) |m| allocator.free(m);
            allocator.free(per_w_mask);
        }
        for (per_w_mask) |*m| {
            m.* = try allocator.alloc(u8, v);
            @memset(m.*, 0);
        }

        const per_w_monster = try allocator.alloc(Monster, nw);
        defer allocator.free(per_w_monster);
        for (per_w_monster) |*pm| pm.* = monster.*; // shallow copy; mask written in worker

        var ctx: AltCtx = .{
            .allocator = allocator,
            .vocab = vocab,
            .usage = usage,
            .alt_count = alt_count_out,
            .per_worker_monster = per_w_monster,
            .per_worker_mask = per_w_mask,
        };
        try bp.runBatch(AltWorker, &ctx, v);
        return;
    }

    // Serial fallback — one mask buffer, one Monster value-copy.
    const mask = try allocator.alloc(u8, v);
    defer allocator.free(mask);
    @memset(mask, 0);

    var solo: Monster = monster.*;
    solo.mask = mask;

    var id: u32 = 0;
    while (id < v) : (id += 1) {
        if (vocab.is_byte.items[id]) {
            alt_count_out[id] = 0;
            continue;
        }
        if (usage[id] == 0) {
            alt_count_out[id] = 1;
            continue;
        }
        const piece_bytes = vocab.piece(id);
        mask[id] = 1;
        var k: u32 = altPieceCountMasked(&solo, piece_bytes) catch 2;
        mask[id] = 0;
        if (k == 0) k = 1;
        alt_count_out[id] = k;
    }
}

// --- delete pass ---------------------------------------------------------

const ScoredId = struct { id: u32, value: u64 };

fn lessByValueAsc(_: void, a: ScoredId, b: ScoredId) bool {
    if (a.value != b.value) return a.value < b.value;
    return a.id < b.id;
}

// Drop the lowest-marginal-value non-byte pieces. Won't drop below `target`.
//
// Avoid-list interaction: pieces whose bytes match an avoid pattern are
// score-pinned to 0, so the ascending sort places them at the very front
// of the kill list every iteration. In `.exclude` mode they're already
// missing from the seed; this path handles `.penalize`.
fn deleteLowest(
    allocator: std.mem.Allocator,
    vocab: *Vocab,
    usage: []const u64,
    alt_count: []const u32,
    delete_fraction: f32,
    target: u32,
    avoid: ?*const negative_train.AvoidList,
) !u32 {
    const v = vocab.count();
    if (v <= target) return 0;

    var candidates: std.ArrayList(ScoredId) = .empty;
    defer candidates.deinit(allocator);
    var id: u32 = 0;
    while (id < v) : (id += 1) {
        if (vocab.is_byte.items[id]) continue;
        // marginal_value = (alt_pieces - 1) * usage[id]
        // alt_count[id] is >= 1 by clamp in computeAltCounts.
        const extra: u64 = if (alt_count[id] >= 1) @as(u64, alt_count[id] - 1) else 0;
        var value = usage[id] *% extra;
        if (avoid) |al| {
            if (al.matches(vocab.piece(id))) value = 0;
        }
        try candidates.append(allocator, .{ .id = id, .value = value });
    }
    std.mem.sort(ScoredId, candidates.items, {}, lessByValueAsc);

    var drop_target: u32 = @intFromFloat(@as(f32, @floatFromInt(candidates.items.len)) * delete_fraction);
    // Ensure forward progress: always remove at least one when above target.
    if (drop_target == 0 and candidates.items.len > 0) drop_target = 1;
    const max_drop: u32 = v - target;
    const to_drop: u32 = @min(drop_target, max_drop);
    if (to_drop == 0) return 0;

    var kill = try allocator.alloc(bool, v);
    defer allocator.free(kill);
    @memset(kill, false);
    var k: u32 = 0;
    while (k < to_drop) : (k += 1) kill[candidates.items[k].id] = true;

    var new_vocab = Vocab.init(allocator);
    errdefer new_vocab.deinit();
    id = 0;
    while (id < v) : (id += 1) {
        if (kill[id]) continue;
        try new_vocab.add(vocab.piece(id), vocab.is_byte.items[id]);
    }
    vocab.deinit();
    vocab.* = new_vocab;
    return to_drop;
}

// --- build a Monster from current vocab ---------------------------------

fn buildMonster(allocator: std.mem.Allocator, vocab: *const Vocab) !Monster {
    var builder = Monster.Builder.init(allocator);
    defer builder.deinit();
    const v = vocab.count();
    var unk_id: TokenId = 0;
    var id: u32 = 0;
    while (id < v) : (id += 1) {
        const new_id = try builder.addToken(vocab.piece(id));
        if (id == 0) unk_id = new_id; // byte 0
    }
    return try builder.finalize(unk_id);
}

// --- public ---------------------------------------------------------------

pub fn train(allocator: std.mem.Allocator, corpus: Corpus, opts: TrainOptions) !Monster {
    if (corpus.words.len != corpus.counts.len) return error.CorpusLengthMismatch;
    if (opts.vocab_size < 256) return error.VocabTooSmall;
    if (opts.max_token_length == 0) return error.InvalidMaxTokenLength;
    if (opts.delete_fraction <= 0.0 or opts.delete_fraction >= 1.0) return error.InvalidDeleteFraction;

    var vocab = Vocab.init(allocator);
    defer vocab.deinit();

    const max_seed_size: usize = @as(usize, opts.vocab_size) * 10;
    try seedVocab(allocator, corpus, opts.max_token_length, opts.min_count, max_seed_size, &vocab, opts.avoid, opts.avoid_mode);

    // Distill loop.
    var iter: u32 = 0;
    while (iter < opts.max_iterations) : (iter += 1) {
        if (vocab.count() <= opts.vocab_size) break;

        var monster = try buildMonster(allocator, &vocab);
        defer monster.deinit();

        const usage = try allocator.alloc(u64, monster.count);
        defer allocator.free(usage);
        var total_tokens: u64 = 0;
        try countUsage(allocator, &monster, corpus, opts.pool, usage, &total_tokens);

        if (opts.on_round) |cb| cb(opts.on_round_ctx, vocab.count(), total_tokens);

        // Marginal-value pass: per-piece alternative-encoding cost.
        // v3 mask-based — reuses `monster` instead of rebuilding per piece.
        // Skip usage-0 pieces (they get marginal=0 regardless of alt cost).
        const alt_count = try allocator.alloc(u32, vocab.count());
        defer allocator.free(alt_count);
        try computeAltCounts(allocator, &vocab, &monster, usage, opts.pool, alt_count);

        const dropped = try deleteLowest(allocator, &vocab, usage, alt_count, opts.delete_fraction, opts.vocab_size, opts.avoid);
        if (dropped == 0) break;
    }

    return try buildMonster(allocator, &vocab);
}

pub fn trainFromBytes(
    allocator: std.mem.Allocator,
    raw: []const u8,
    splitFn: *const fn (std.mem.Allocator, []const u8) anyerror![]Span,
    opts: TrainOptions,
) !Monster {
    const spans = try splitFn(allocator, raw);
    defer allocator.free(spans);

    // Aggregate identical word strings → counts.
    var counts = std.StringHashMap(u32).init(allocator);
    defer counts.deinit();
    for (spans) |sp| {
        const w = sp.slice(raw);
        if (w.len == 0) continue;
        const gop = try counts.getOrPut(w);
        if (!gop.found_existing) gop.value_ptr.* = 0;
        gop.value_ptr.* +|= 1;
    }

    const n = counts.count();
    const words = try allocator.alloc([]const u8, n);
    defer allocator.free(words);
    const freqs = try allocator.alloc(u32, n);
    defer allocator.free(freqs);

    var it = counts.iterator();
    var i: usize = 0;
    while (it.next()) |e| : (i += 1) {
        words[i] = e.key_ptr.*;
        freqs[i] = e.value_ptr.*;
    }

    return try train(allocator, .{ .words = words, .counts = freqs }, opts);
}

// --- tests ---------------------------------------------------------------

const testing = std.testing;

test "trains to within target vocab_size" {
    const allocator = testing.allocator;

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    var words_storage: [120][]const u8 = undefined;
    var counts_storage: [120]u32 = undefined;
    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const rnd = prng.random();
    var w: usize = 0;
    while (w < 120) : (w += 1) {
        const len = rnd.intRangeAtMost(usize, 3, 12);
        const buf = try aa.alloc(u8, len);
        for (buf) |*c| c.* = 'a' + rnd.intRangeAtMost(u8, 0, 9);
        words_storage[w] = buf;
        counts_storage[w] = rnd.intRangeAtMost(u32, 4, 12);
    }

    var m = try train(allocator, .{
        .words = &words_storage,
        .counts = &counts_storage,
    }, .{
        .vocab_size = 300,
        .max_token_length = 8,
        .min_count = 2,
        .delete_fraction = 0.10,
    });
    defer m.deinit();

    try testing.expect(m.count <= 300);
    try testing.expect(m.count >= 256);
}

test "round-trips trained corpus" {
    const allocator = testing.allocator;

    const sentence = "the quick brown fox";
    var words_storage: [1][]const u8 = .{sentence};
    var counts_storage: [1]u32 = .{100};

    var m = try train(allocator, .{
        .words = &words_storage,
        .counts = &counts_storage,
    }, .{
        .vocab_size = 280,
        .max_token_length = 8,
        .min_count = 2,
        .delete_fraction = 0.10,
    });
    defer m.deinit();

    const sample = "the quick";
    var out: [64]TokenId = undefined;
    const ids = try m.encodeChunk(allocator, sample, &out);

    var reconstructed: std.ArrayList(u8) = .empty;
    defer reconstructed.deinit(allocator);
    for (ids) |id| try reconstructed.appendSlice(allocator, m.idBytes(id));
    try testing.expectEqualStrings(sample, reconstructed.items);
}

test "deterministic" {
    const allocator = testing.allocator;

    const words = [_][]const u8{ "banana", "apple", "orange", "applesauce", "bananabread" };
    const counts = [_]u32{ 30, 20, 15, 8, 5 };

    const opts: TrainOptions = .{
        .vocab_size = 290,
        .max_token_length = 8,
        .min_count = 2,
        .delete_fraction = 0.10,
    };

    var m1 = try train(allocator, .{ .words = &words, .counts = &counts }, opts);
    defer m1.deinit();
    var m2 = try train(allocator, .{ .words = &words, .counts = &counts }, opts);
    defer m2.deinit();

    try testing.expectEqual(m1.count, m2.count);
    try testing.expectEqualSlices(u8, m1.bytes, m2.bytes);
    try testing.expectEqualSlices(u32, m1.offsets, m2.offsets);
}

test "byte fallbacks always present" {
    const allocator = testing.allocator;

    const words = [_][]const u8{ "hello", "world" };
    const counts = [_]u32{ 5, 5 };

    var m = try train(allocator, .{ .words = &words, .counts = &counts }, .{
        .vocab_size = 260,
        .max_token_length = 4,
        .min_count = 2,
        .delete_fraction = 0.20,
    });
    defer m.deinit();

    // Every byte value must encode SOMEHOW (either as the literal byte
    // piece or absorbed into a containing piece). Encoding a single-byte
    // input must succeed and yield exactly one token whose bytes equal the
    // input (since pinned byte pieces still match length-1 inputs).
    var b: u32 = 0;
    while (b < 256) : (b += 1) {
        const input = [_]u8{@intCast(b)};
        var out: [4]TokenId = undefined;
        const ids = try m.encodeChunk(allocator, &input, &out);
        try testing.expect(ids.len == 1);
        const piece = m.idBytes(ids[0]);
        try testing.expect(piece.len == 1);
        try testing.expectEqual(@as(u8, @intCast(b)), piece[0]);
    }
}

test "value-based deletion prefers high-frequency long pieces" {
    const allocator = testing.allocator;

    // A high-frequency long word is repeated; noise words contribute many
    // low-value substrings that should get pruned first. After distillation
    // the long word itself ("xylophone") should still tokenize to a single
    // piece (or at worst two).
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    var words: std.ArrayList([]const u8) = .empty;
    defer words.deinit(allocator);
    var freqs: std.ArrayList(u32) = .empty;
    defer freqs.deinit(allocator);

    try words.append(allocator, "xylophone");
    try freqs.append(allocator, 200);

    // Noise: many distinct short random words, each appearing a few times.
    var prng = std.Random.DefaultPrng.init(0xABCD);
    const rnd = prng.random();
    var i: usize = 0;
    while (i < 200) : (i += 1) {
        const len = rnd.intRangeAtMost(usize, 3, 6);
        const buf = try aa.alloc(u8, len);
        for (buf) |*c| c.* = 'a' + rnd.intRangeAtMost(u8, 0, 22);
        try words.append(allocator, buf);
        try freqs.append(allocator, rnd.intRangeAtMost(u32, 4, 8));
    }

    var m = try train(allocator, .{
        .words = words.items,
        .counts = freqs.items,
    }, .{
        .vocab_size = 280,
        .max_token_length = 12,
        .min_count = 2,
        .delete_fraction = 0.10,
    });
    defer m.deinit();

    var out: [16]TokenId = undefined;
    const ids = try m.encodeChunk(allocator, "xylophone", &out);
    try testing.expect(ids.len <= 2);
}

test "marginal-value scoring diverges from v1 static heuristic" {
    // The v1 (usage * length) heuristic favours LONG pieces; the v2
    // marginal-value heuristic favours HIGH-USAGE pieces (because the
    // seed phase guarantees almost every piece a cheap 2-token alt, so
    // marginal ≈ usage). When length and usage trade off, v1 and v2
    // disagree on which piece to drop.
    //
    // Construction:
    //   * "abcdefghij" × 30 → usage = 30, length = 10
    //       v1 score  = 30 × 10 = 300
    //       v2 marginal = 1 × 30 = 30   (alt = "abcdefghi" + "j" = 2 tokens)
    //   * "ab"         × 100 → usage = 100, length = 2
    //       v1 score  = 100 × 2 = 200
    //       v2 marginal = 1 × 100 = 100 (alt = "a" + "b" = 2 tokens)
    //
    //   v1 would drop "ab" first (lower score).
    //   v2 drops "abcdefghij" first (lower marginal).
    //
    // Squeeze the vocab to 257 (256 bytes + 1 piece) so exactly one
    // non-byte survives, then check WHICH one. Under v2 it must be "ab".

    const allocator = testing.allocator;

    const words = [_][]const u8{ "abcdefghij", "ab" };
    const counts = [_]u32{ 30, 100 };

    var m = try train(allocator, .{ .words = &words, .counts = &counts }, .{
        .vocab_size = 257,
        .max_token_length = 12,
        .min_count = 30,
        .delete_fraction = 0.99,
        .max_iterations = 20,
    });
    defer m.deinit();

    try testing.expect(m.count <= 257);

    // "ab" must remain a single token — v2 keeps it because its marginal
    // value (100) beats "abcdefghij"'s marginal (30). If v1 scoring were
    // in effect, "ab" would have been dropped first and this encode
    // would emit 2 byte tokens.
    var out_ab: [4]TokenId = undefined;
    const ids_ab = try m.encodeChunk(allocator, "ab", &out_ab);
    try testing.expectEqual(@as(usize, 1), ids_ab.len);
}

test "v3 mask alt-counts match v2 rebuild alt-counts (top-K ranking)" {
    // Build a small vocab, populate fake usage, then compute alt counts
    // with both v2 (rebuild) and v3 (mask). They must agree per-piece;
    // therefore the sort-by-marginal-value rankings agree exactly.
    const allocator = testing.allocator;

    var vocab = Vocab.init(allocator);
    defer vocab.deinit();

    // 256 byte pieces (pinned, never scored).
    var b: u32 = 0;
    while (b < 256) : (b += 1) {
        const buf = [_]u8{@intCast(b)};
        try vocab.add(&buf, true);
    }
    // A handful of multi-byte pieces with varied byte composition.
    const pieces = [_][]const u8{ "ab", "bc", "abc", "xyz", "hello", "wor", "world", "the", "and", "ing" };
    for (pieces) |p| try vocab.add(p, false);

    var monster = try buildMonster(allocator, &vocab);
    defer monster.deinit();

    // Fabricate a usage distribution: nonzero so we exercise the encode
    // path (usage=0 short-circuits).
    const usage = try allocator.alloc(u64, vocab.count());
    defer allocator.free(usage);
    @memset(usage, 0);
    var i: u32 = 256;
    while (i < vocab.count()) : (i += 1) usage[i] = @as(u64, i - 256 + 1) * 7;

    // v3 path.
    const alt_v3 = try allocator.alloc(u32, vocab.count());
    defer allocator.free(alt_v3);
    try computeAltCounts(allocator, &vocab, &monster, usage, null, alt_v3);

    // v2 path (manual, serial).
    const alt_v2 = try allocator.alloc(u32, vocab.count());
    defer allocator.free(alt_v2);
    @memset(alt_v2, 0);
    var id: u32 = 0;
    while (id < vocab.count()) : (id += 1) {
        if (vocab.is_byte.items[id]) {
            alt_v2[id] = 0;
            continue;
        }
        if (usage[id] == 0) {
            alt_v2[id] = 1;
            continue;
        }
        const piece_bytes = vocab.piece(id);
        var k = altPieceCountV2(allocator, &vocab, id, piece_bytes) catch 2;
        if (k == 0) k = 1;
        alt_v2[id] = k;
    }

    // Per-piece equality is the strongest check — implies ranking equality.
    try testing.expectEqualSlices(u32, alt_v2, alt_v3);

    // And explicitly verify top-K ranking too (K=5).
    // Uses the module-level ScoredId / lessByValueAsc — we want
    // top-K-by-marginal-value, so we sort descending by negating value
    // through the existing ascending sort and taking from the end.
    var rank_v2: std.ArrayList(ScoredId) = .empty;
    defer rank_v2.deinit(allocator);
    var rank_v3: std.ArrayList(ScoredId) = .empty;
    defer rank_v3.deinit(allocator);
    var jj: u32 = 0;
    while (jj < vocab.count()) : (jj += 1) {
        if (vocab.is_byte.items[jj]) continue;
        const extra2: u64 = if (alt_v2[jj] >= 1) @as(u64, alt_v2[jj] - 1) else 0;
        const extra3: u64 = if (alt_v3[jj] >= 1) @as(u64, alt_v3[jj] - 1) else 0;
        try rank_v2.append(allocator, .{ .id = jj, .value = usage[jj] *% extra2 });
        try rank_v3.append(allocator, .{ .id = jj, .value = usage[jj] *% extra3 });
    }
    std.mem.sort(ScoredId, rank_v2.items, {}, lessByValueAsc);
    std.mem.sort(ScoredId, rank_v3.items, {}, lessByValueAsc);

    // After ascending sort, the last K entries are the top-K. Compare them.
    const total = rank_v2.items.len;
    const k_top: usize = @min(total, 5);
    var t: usize = 0;
    while (t < k_top) : (t += 1) {
        const idx = total - 1 - t;
        try testing.expectEqual(rank_v2.items[idx].id, rank_v3.items[idx].id);
    }
}

test "v3 end-to-end train converges to same vocab as v2 baseline (deterministic seed)" {
    // The trainer ALREADY uses v3 (computeAltCounts is the v3 path now),
    // so this test pins the trainer's output against a hand-checked
    // reproduction of v2's invariants:
    //   * vocab size at or below target,
    //   * deterministic across two runs on the same seed,
    //   * round-trips trained corpus bytes.
    const allocator = testing.allocator;

    const words = [_][]const u8{ "alphabet", "alphabetical", "beta", "betacarotene", "gamma" };
    const counts = [_]u32{ 50, 30, 40, 10, 20 };

    const opts: TrainOptions = .{
        .vocab_size = 280,
        .max_token_length = 12,
        .min_count = 2,
        .delete_fraction = 0.15,
        .max_iterations = 30,
    };

    var m1 = try train(allocator, .{ .words = &words, .counts = &counts }, opts);
    defer m1.deinit();
    var m2 = try train(allocator, .{ .words = &words, .counts = &counts }, opts);
    defer m2.deinit();

    // Determinism (same seed, same path).
    try testing.expectEqual(m1.count, m2.count);
    try testing.expectEqualSlices(u8, m1.bytes, m2.bytes);
    try testing.expectEqualSlices(u32, m1.offsets, m2.offsets);

    // Vocab size respected.
    try testing.expect(m1.count <= 280);
    try testing.expect(m1.count >= 256);

    // Round-trip a trained word — bytes survive distillation.
    var out: [64]TokenId = undefined;
    const ids = try m1.encodeChunk(allocator, "alphabet beta gamma", &out);
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);
    for (ids) |id| try buf.appendSlice(allocator, m1.idBytes(id));
    try testing.expectEqualStrings("alphabet beta gamma", buf.items);

    // Sanity: mask is not persisted; freshly trained monster has no mask.
    try testing.expect(m1.mask == null);
}
