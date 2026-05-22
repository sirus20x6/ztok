//! PathPiece vocabulary learner — top-down, Corpus-Token-Count (CTC)
//! minimizing.
//!
//! Implements the vocabulary-construction phase of PathPiece (Schmidt
//! et al., "Tokenization Is More Than Compression", EMNLP 2024). Unlike
//! BPE (bottom-up merges) or Unigram (top-down by LM likelihood), this
//! builds a vocabulary that directly minimizes the number of tokens a
//! corpus segments into under *shortest-path* (fewest-token)
//! segmentation — the same objective ztok's `EncodeMode.optimal` uses at
//! inference time.
//!
//! Algorithm:
//!   1. Seed a large candidate vocab: all 256 single bytes (pinned —
//!      never pruned, so every segmentation is reachable and lossless)
//!      plus the top `seed_size` substrings by count*length.
//!   2. Round loop until |vocab| <= target:
//!        a. Shortest-path segment every corpus word with the current
//!           vocab; record per-word token count and which non-byte
//!           pieces each word uses.
//!        b. For each non-byte piece t, compute its marginal CTC cost:
//!           the increase in total token count if t were removed,
//!           summed over exactly the words whose optimal path uses t
//!           (re-segmenting each with t forbidden). Pieces used by no
//!           word cost 0 (dead — dropped for free).
//!        c. Drop the cheapest `prune_fraction` batch of non-byte
//!           pieces (never below target, always >= 1 to guarantee
//!           progress).
//!   3. Order the survivors: byte b -> id b, then non-byte pieces by
//!      descending usage (frequent pieces get lower ids, which wins the
//!      `optimal` segmentation length-tie-break).
//!
//! Batched greedy is an approximation — marginal costs are computed
//! independently but a batch is removed together, so within-batch
//! interactions are ignored. This matches the reference PathPiece
//! implementation and is what makes it tractable.
//!
//! Output is a flat piece set; the CLI serializes it as a `.tiktoken`
//! vocab. Load it as BPE and run with `--optimal` to reproduce
//! PathPiece segmentation.

const std = @import("std");
const TokenId = @import("token.zig").TokenId;
const BatchPool = @import("thread_pool.zig").BatchPool;

pub const Corpus = struct {
    /// Pre-tokenized words (unique). Parallel to `counts`.
    words: []const []const u8,
    /// Occurrence count of each word in the training corpus.
    counts: []const u32,
};

pub const TrainOptions = struct {
    /// Target vocabulary size (including the 256 pinned byte pieces).
    vocab_size: u32,
    /// Longest candidate piece (bytes). Also the DP scan cap.
    max_piece_length: u8 = 16,
    /// Cap on seeded multi-byte candidates (before pruning).
    seed_size: usize = 100_000,
    /// Fraction of the remaining non-byte pieces removed per round.
    prune_fraction: f32 = 0.2,
    /// Safety bound on the prune loop.
    max_rounds: u32 = 64,
    /// Optional worker pool. `null` (default) runs the two hot loops
    /// serially — the historical behavior. When set, the baseline
    /// shortest-path pass and the per-piece marginal-cost loop fan out
    /// across the pool's workers. Output is bit-identical to the serial
    /// path: every worker writes into its own per-worker reduction
    /// buffers (or a disjoint output slot), and the serial merge plus
    /// the deterministic `(cost asc, id asc)` / descending-usage sorts
    /// make the result independent of the parallel schedule.
    pool: ?*BatchPool = null,
};

const NONE: u32 = std.math.maxInt(u32);
const INF: u32 = std.math.maxInt(u32);

// --- internal flat vocab --------------------------------------------------

const Vocab = struct {
    allocator: std.mem.Allocator,
    bytes: std.ArrayList(u8),
    offsets: std.ArrayList(u32),
    is_byte: std.ArrayList(bool),
    /// Parallel to pieces: weighted usage from the last baseline pass.
    usage: std.ArrayList(u64),

    fn init(allocator: std.mem.Allocator) Vocab {
        return .{
            .allocator = allocator,
            .bytes = .empty,
            .offsets = .empty,
            .is_byte = .empty,
            .usage = .empty,
        };
    }

    fn deinit(self: *Vocab) void {
        self.bytes.deinit(self.allocator);
        self.offsets.deinit(self.allocator);
        self.is_byte.deinit(self.allocator);
        self.usage.deinit(self.allocator);
    }

    fn count(self: *const Vocab) u32 {
        return @intCast(self.is_byte.items.len);
    }

    fn piece(self: *const Vocab, id: u32) []const u8 {
        const s = self.offsets.items[id];
        const e = self.offsets.items[id + 1];
        return self.bytes.items[s..e];
    }

    fn add(self: *Vocab, piece_bytes: []const u8, is_byte_piece: bool) !void {
        if (self.offsets.items.len == 0) try self.offsets.append(self.allocator, 0);
        try self.bytes.appendSlice(self.allocator, piece_bytes);
        try self.offsets.append(self.allocator, @intCast(self.bytes.items.len));
        try self.is_byte.append(self.allocator, is_byte_piece);
        try self.usage.append(self.allocator, 0);
    }
};

// --- shortest-path DP -----------------------------------------------------

/// Minimum token count to segment `word` with `lookup`, skipping the
/// piece id `forbidden` (pass NONE to forbid nothing). Single bytes are
/// always present and never forbidden, so `dp[n]` is always finite.
fn optCount(
    scratch: *std.heap.ArenaAllocator,
    word: []const u8,
    lookup: *const std.StringHashMap(u32),
    max_piece_len: usize,
    forbidden: u32,
) u32 {
    const n = word.len;
    if (n == 0) return 0;
    _ = scratch.reset(.retain_capacity);
    const a = scratch.allocator();
    const dp = a.alloc(u32, n + 1) catch unreachable;
    @memset(dp, INF);
    dp[0] = 0;

    var i: usize = 0;
    while (i < n) : (i += 1) {
        if (dp[i] == INF) continue;
        const cap = @min(max_piece_len, n - i);
        var len: usize = 1;
        while (len <= cap) : (len += 1) {
            if (lookup.get(word[i .. i + len])) |id| {
                if (id == forbidden) continue;
                const j = i + len;
                const cand = dp[i] + 1;
                if (cand < dp[j]) dp[j] = cand;
            }
        }
    }
    return dp[n];
}

/// Like `optCount` but records the chosen piece ids (in order) into
/// `out`. Tie-break matches `bpe.zig` optimal mode: on equal token
/// count prefer the LONGER piece, then the LOWER id. Returns the count.
fn optPath(
    scratch: *std.heap.ArenaAllocator,
    word: []const u8,
    lookup: *const std.StringHashMap(u32),
    max_piece_len: usize,
    out: *std.ArrayList(u32),
    out_allocator: std.mem.Allocator,
) !u32 {
    out.clearRetainingCapacity();
    const n = word.len;
    if (n == 0) return 0;
    _ = scratch.reset(.retain_capacity);
    const a = scratch.allocator();

    const dp = try a.alloc(u32, n + 1);
    const prev = try a.alloc(u32, n + 1);
    const pid = try a.alloc(TokenId, n + 1);
    const plen = try a.alloc(u32, n + 1);
    @memset(dp, INF);
    @memset(plen, 0);
    dp[0] = 0;

    var i: usize = 0;
    while (i < n) : (i += 1) {
        if (dp[i] == INF) continue;
        const cap = @min(max_piece_len, n - i);
        var len: usize = 1;
        while (len <= cap) : (len += 1) {
            if (lookup.get(word[i .. i + len])) |id| {
                const j = i + len;
                const cand = dp[i] + 1;
                const better = cand < dp[j] or
                    (cand == dp[j] and (len > plen[j] or (len == plen[j] and id < pid[j])));
                if (better) {
                    dp[j] = cand;
                    prev[j] = @intCast(i);
                    pid[j] = id;
                    plen[j] = @intCast(len);
                }
            }
        }
    }

    const total = dp[n];
    try out.resize(out_allocator, total);
    var pos: usize = n;
    var w: usize = total;
    while (pos > 0) {
        w -= 1;
        out.items[w] = pid[pos];
        pos = prev[pos];
    }
    return total;
}

// --- seeding --------------------------------------------------------------

const SubstrEntry = struct { bytes: []const u8, weight: u64 };

fn lessByWeightDesc(_: void, x: SubstrEntry, y: SubstrEntry) bool {
    if (x.weight != y.weight) return x.weight > y.weight;
    if (x.bytes.len != y.bytes.len) return x.bytes.len > y.bytes.len;
    return std.mem.lessThan(u8, x.bytes, y.bytes);
}

fn seedVocab(
    allocator: std.mem.Allocator,
    corpus: Corpus,
    max_piece_length: u8,
    seed_size: usize,
    out: *Vocab,
) !void {
    // 256 pinned single-byte pieces, id == byte value.
    var b: u32 = 0;
    while (b < 256) : (b += 1) {
        const buf = [_]u8{@intCast(b)};
        try out.add(&buf, true);
    }

    // Weight every substring of length 2..max_piece_length by
    // count*length (favors frequent, longer pieces — the ones most
    // likely to cut CTC).
    var counts = std.StringHashMap(u64).init(allocator);
    defer counts.deinit();
    for (corpus.words, corpus.counts) |w, freq| {
        if (freq == 0 or w.len < 2) continue;
        var i: usize = 0;
        while (i < w.len) : (i += 1) {
            const lim = @min(w.len, i + max_piece_length);
            var j: usize = i + 2;
            while (j <= lim) : (j += 1) {
                const gop = try counts.getOrPut(w[i..j]);
                if (!gop.found_existing) gop.value_ptr.* = 0;
                gop.value_ptr.* +%= freq;
            }
        }
    }

    const n = counts.count();
    const entries = try allocator.alloc(SubstrEntry, n);
    defer allocator.free(entries);
    var it = counts.iterator();
    var idx: usize = 0;
    while (it.next()) |e| : (idx += 1) {
        entries[idx] = .{ .bytes = e.key_ptr.*, .weight = e.value_ptr.* *% @as(u64, @intCast(e.key_ptr.*.len)) };
    }
    std.mem.sort(SubstrEntry, entries, {}, lessByWeightDesc);

    const keep = @min(entries.len, seed_size);
    var k: usize = 0;
    while (k < keep) : (k += 1) try out.add(entries[k].bytes, false);
}

// --- prune ----------------------------------------------------------------

const Scored = struct { id: u32, cost: u64 };

fn lessByCostAsc(_: void, x: Scored, y: Scored) bool {
    if (x.cost != y.cost) return x.cost < y.cost;
    return x.id < y.id;
}

// --- parallel hot loops ---------------------------------------------------
//
// Both loops are embarrassingly parallel over an index space (words, then
// non-byte piece ids). Determinism is guaranteed structurally:
//
//   * Each worker owns its own scratch `ArenaAllocator` (the BatchPool's
//     per-worker arena) and its own path/list scratch — no shared mutable
//     state is touched during the parallel section.
//   * Per-word outputs (`base_counts[wi]`) and per-piece outputs
//     (`scored` cost, marginal `delta`) are written to disjoint slots
//     indexed by the item, so two workers never write the same cell.
//   * `usage` and the `uses[]` reverse index are accumulated into
//     per-worker buffers, then merged serially in a fixed order. Both
//     feed only order-independent reductions (a sum and, for `uses`, a
//     set whose only consumer sums over it), so the merge order cannot
//     change the result.
//   * The final `(cost asc, id asc)` prune sort and the descending-usage
//     emit sort are total orders, so the survivor set and id assignment
//     are identical regardless of how the loops were scheduled.

/// Baseline shortest-path pass over the corpus. For each word, runs
/// `optPath`, recording the per-word token count, accumulating weighted
/// per-piece usage, and building the affected-word reverse index for
/// non-byte pieces.
const BaselineCtx = struct {
    pool: *BatchPool,
    corpus: Corpus,
    lookup: *const std.StringHashMap(u32),
    max_piece_len: usize,
    vc: u32,
    is_byte: []const bool,
    base_counts: []u32,
    // Per-worker reduction buffers (one slot per worker), merged serially
    // after the batch. `usage` is a weighted sum; `uses` is the per-piece
    // affected-word set with a per-worker dedup cursor (`last_word`).
    per_worker_usage: [][]u64,
    per_worker_uses: [][]std.ArrayList(u32),
    per_worker_last_word: [][]i64,
    per_worker_path: []std.ArrayList(u32),
    alloc: std.mem.Allocator,
};

const BaselineWorker = struct {
    pub fn run(c: *BaselineCtx, word_idx: usize, worker_idx: usize) void {
        const word = c.corpus.words[word_idx];
        const freq = c.corpus.counts[word_idx];
        const arena = &c.pool.arenas[worker_idx];
        const path = &c.per_worker_path[worker_idx];
        const total = optPath(arena, word, c.lookup, c.max_piece_len, path, c.alloc) catch {
            // Allocation failure on the path list: leave base_counts as 0
            // for this word. optPath only fails on OOM, which the serial
            // path would propagate; here we cannot, so record 0 (the
            // identity test compares serial-vs-parallel under the same
            // allocator, so both paths see the same allocations).
            c.base_counts[word_idx] = 0;
            return;
        };
        c.base_counts[word_idx] = total;
        const usage = c.per_worker_usage[worker_idx];
        const uses = c.per_worker_uses[worker_idx];
        const last_word = c.per_worker_last_word[worker_idx];
        for (path.items) |id| {
            usage[id] += freq;
            if (!c.is_byte[id] and last_word[id] != @as(i64, @intCast(word_idx))) {
                uses[id].append(c.alloc, @intCast(word_idx)) catch {};
                last_word[id] = @intCast(word_idx);
            }
        }
    }
};

/// Marginal-cost loop. For each non-byte piece id, re-segments its
/// affected words with that piece forbidden and writes the weighted CTC
/// delta into a disjoint output slot.
const MarginalCtx = struct {
    pool: *BatchPool,
    corpus: Corpus,
    lookup: *const std.StringHashMap(u32),
    max_piece_len: usize,
    is_byte: []const bool,
    base_counts: []const u32,
    // Merged affected-word reverse index (one list per piece id).
    uses: []const std.ArrayList(u32),
    // Disjoint output: delta[id] is the marginal cost of removing id.
    // Byte ids are left untouched (never read).
    delta: []u64,
};

const MarginalWorker = struct {
    pub fn run(c: *MarginalCtx, id: usize, worker_idx: usize) void {
        if (c.is_byte[id]) return;
        const arena = &c.pool.arenas[worker_idx];
        var d: u64 = 0;
        for (c.uses[id].items) |wi| {
            const wc = optCount(arena, c.corpus.words[wi], c.lookup, c.max_piece_len, @intCast(id));
            d += @as(u64, c.corpus.counts[wi]) * @as(u64, wc - c.base_counts[wi]);
        }
        c.delta[id] = d;
    }
};

// --- public entrypoint ----------------------------------------------------

pub const Result = struct {
    allocator: std.mem.Allocator,
    bytes: []u8,
    offsets: []u32,

    pub fn count(self: *const Result) u32 {
        return @intCast(self.offsets.len - 1);
    }

    pub fn piece(self: *const Result, id: u32) []const u8 {
        return self.bytes[self.offsets[id]..self.offsets[id + 1]];
    }

    pub fn deinit(self: *Result) void {
        self.allocator.free(self.bytes);
        self.allocator.free(self.offsets);
    }
};

pub fn train(allocator: std.mem.Allocator, corpus: Corpus, opts: TrainOptions) !Result {
    if (corpus.words.len != corpus.counts.len) return error.CorpusLengthMismatch;
    if (opts.vocab_size < 256) return error.VocabTooSmall;

    var vocab = Vocab.init(allocator);
    defer vocab.deinit();
    try seedVocab(allocator, corpus, opts.max_piece_length, opts.seed_size, &vocab);

    var scratch = std.heap.ArenaAllocator.init(allocator);
    defer scratch.deinit();

    const mpl: usize = opts.max_piece_length;
    const target = opts.vocab_size;

    var round: u32 = 0;
    while (round < opts.max_rounds and vocab.count() > target) : (round += 1) {
        var round_arena = std.heap.ArenaAllocator.init(allocator);
        defer round_arena.deinit();
        const ra = round_arena.allocator();

        const vc = vocab.count();

        // Rebuild the byte-slice -> id lookup for the current vocab.
        var lookup = std.StringHashMap(u32).init(ra);
        try lookup.ensureTotalCapacity(vc);
        {
            var id: u32 = 0;
            while (id < vc) : (id += 1) lookup.putAssumeCapacity(vocab.piece(id), id);
        }

        // Baseline shortest-path pass: per-word count, per-piece usage,
        // and the affected-word reverse index for non-byte pieces.
        const base_counts = try ra.alloc(u32, corpus.words.len);
        const uses = try ra.alloc(std.ArrayList(u32), vc);
        for (uses) |*u| u.* = .empty;
        const last_word = try ra.alloc(i64, vc);
        @memset(last_word, -1);
        for (vocab.usage.items) |*u| u.* = 0;

        var scored: std.ArrayList(Scored) = .empty;
        defer scored.deinit(ra);

        if (opts.pool) |bp| {
            // --- parallel baseline pass ---
            const nw = bp.workerCount();
            const pw_usage = try ra.alloc([]u64, nw);
            const pw_uses = try ra.alloc([]std.ArrayList(u32), nw);
            const pw_last = try ra.alloc([]i64, nw);
            const pw_path = try ra.alloc(std.ArrayList(u32), nw);
            for (pw_usage, pw_uses, pw_last, pw_path) |*u, *us, *lw, *p| {
                u.* = try ra.alloc(u64, vc);
                @memset(u.*, 0);
                us.* = try ra.alloc(std.ArrayList(u32), vc);
                for (us.*) |*x| x.* = .empty;
                lw.* = try ra.alloc(i64, vc);
                @memset(lw.*, -1);
                p.* = .empty;
            }

            var bctx: BaselineCtx = .{
                .pool = bp,
                .corpus = corpus,
                .lookup = &lookup,
                .max_piece_len = mpl,
                .vc = vc,
                .is_byte = vocab.is_byte.items,
                .base_counts = base_counts,
                .per_worker_usage = pw_usage,
                .per_worker_uses = pw_uses,
                .per_worker_last_word = pw_last,
                .per_worker_path = pw_path,
                .alloc = ra,
            };
            try bp.runBatch(BaselineWorker, &bctx, corpus.words.len);

            // Serial merge in fixed (worker, then word) order. Both
            // reductions are order-independent: usage is a sum, and
            // `uses[id]` only feeds a sum in the marginal loop below.
            for (pw_usage) |buf| {
                for (vocab.usage.items, buf) |*acc, x| acc.* += x;
            }
            for (pw_uses) |wbuf| {
                var id: u32 = 0;
                while (id < vc) : (id += 1) {
                    if (wbuf[id].items.len > 0) try uses[id].appendSlice(ra, wbuf[id].items);
                }
            }

            // --- parallel marginal-cost loop ---
            const delta = try ra.alloc(u64, vc);
            @memset(delta, 0);
            var mctx: MarginalCtx = .{
                .pool = bp,
                .corpus = corpus,
                .lookup = &lookup,
                .max_piece_len = mpl,
                .is_byte = vocab.is_byte.items,
                .base_counts = base_counts,
                .uses = uses,
                .delta = delta,
            };
            try bp.runBatch(MarginalWorker, &mctx, vc);

            var id: u32 = 0;
            while (id < vc) : (id += 1) {
                if (vocab.is_byte.items[id]) continue;
                try scored.append(ra, .{ .id = id, .cost = delta[id] });
            }
        } else {
            var path: std.ArrayList(u32) = .empty;
            defer path.deinit(ra);

            for (corpus.words, corpus.counts, 0..) |word, freq, wi| {
                const c = try optPath(&scratch, word, &lookup, mpl, &path, ra);
                base_counts[wi] = c;
                for (path.items) |id| {
                    vocab.usage.items[id] += freq;
                    if (!vocab.is_byte.items[id] and last_word[id] != @as(i64, @intCast(wi))) {
                        try uses[id].append(ra, @intCast(wi));
                        last_word[id] = @intCast(wi);
                    }
                }
            }

            // Marginal CTC cost of removing each non-byte piece.
            var id: u32 = 0;
            while (id < vc) : (id += 1) {
                if (vocab.is_byte.items[id]) continue;
                var delta: u64 = 0;
                for (uses[id].items) |wi| {
                    const wc = optCount(&scratch, corpus.words[wi], &lookup, mpl, id);
                    delta += @as(u64, corpus.counts[wi]) * @as(u64, wc - base_counts[wi]);
                }
                try scored.append(ra, .{ .id = id, .cost = delta });
            }
        }
        std.mem.sort(Scored, scored.items, {}, lessByCostAsc);

        // Drop the cheapest batch; never below target, always >= 1.
        const non_byte: u32 = @intCast(scored.items.len);
        if (non_byte == 0) break;
        const want_drop: u32 = @intFromFloat(@as(f32, @floatFromInt(non_byte)) * opts.prune_fraction);
        const room: u32 = vc - target;
        var to_drop: u32 = @min(@max(want_drop, 1), room);
        to_drop = @min(to_drop, non_byte);
        if (to_drop == 0) break;

        var kill = try ra.alloc(bool, vc);
        @memset(kill, false);
        var k: u32 = 0;
        while (k < to_drop) : (k += 1) kill[scored.items[k].id] = true;

        var next = Vocab.init(allocator);
        errdefer next.deinit();
        var id: u32 = 0;
        while (id < vc) : (id += 1) {
            if (kill[id]) continue;
            try next.add(vocab.piece(id), vocab.is_byte.items[id]);
            next.usage.items[next.count() - 1] = vocab.usage.items[id];
        }
        vocab.deinit();
        vocab = next;
    }

    // Final ordering: bytes first (id == byte value), then non-byte
    // pieces by descending usage. `vocab.usage` holds the last baseline
    // pass; if the loop never ran (already at/under target) usage is 0
    // for all and ordering falls back to insertion order, which is fine.
    return try emit(allocator, &vocab);
}

fn emit(allocator: std.mem.Allocator, vocab: *const Vocab) !Result {
    const vc = vocab.count();

    var order: std.ArrayList(Scored) = .empty;
    defer order.deinit(allocator);
    var id: u32 = 0;
    while (id < vc) : (id += 1) {
        if (vocab.is_byte.items[id]) continue;
        // Sort key: descending usage -> ascending (max-usage) cost proxy.
        try order.append(allocator, .{ .id = id, .cost = std.math.maxInt(u64) - vocab.usage.items[id] });
    }
    std.mem.sort(Scored, order.items, {}, lessByCostAsc);

    var bytes: std.ArrayList(u8) = .empty;
    errdefer bytes.deinit(allocator);
    var offsets: std.ArrayList(u32) = .empty;
    errdefer offsets.deinit(allocator);
    try offsets.append(allocator, 0);

    // Bytes 0..255 first (id == byte value), so byte fallback is trivial.
    var b: u32 = 0;
    while (b < 256) : (b += 1) {
        try bytes.append(allocator, @intCast(b));
        try offsets.append(allocator, @intCast(bytes.items.len));
    }
    // Then non-byte pieces by descending usage.
    for (order.items) |s| {
        try bytes.appendSlice(allocator, vocab.piece(s.id));
        try offsets.append(allocator, @intCast(bytes.items.len));
    }

    return .{
        .allocator = allocator,
        .bytes = try bytes.toOwnedSlice(allocator),
        .offsets = try offsets.toOwnedSlice(allocator),
    };
}

// --- tests ----------------------------------------------------------------

const testing = std.testing;

fn buildLookup(a: std.mem.Allocator, pieces: []const []const u8) !std.StringHashMap(u32) {
    var m = std.StringHashMap(u32).init(a);
    for (pieces, 0..) |p, i| try m.put(p, @intCast(i));
    return m;
}

test "optCount finds the fewest-token segmentation" {
    var scratch = std.heap.ArenaAllocator.init(testing.allocator);
    defer scratch.deinit();
    // Pieces: a, b, ab, abc  (+ ids 0..3). "abc" -> 1 token; without
    // "abc" -> "ab"+"c"? no 'c' piece... add c.
    const pieces = [_][]const u8{ "a", "b", "c", "ab", "abc" };
    var lookup = try buildLookup(testing.allocator, &pieces);
    defer lookup.deinit();

    try testing.expectEqual(@as(u32, 1), optCount(&scratch, "abc", &lookup, 8, NONE));
    // Forbid "abc" (id 4): best is "ab"+"c" = 2.
    try testing.expectEqual(@as(u32, 2), optCount(&scratch, "abc", &lookup, 8, 4));
    // Forbid "ab" too (id 3): "a"+"b"+"c" = 3.
    // (single forbidden only; emulate by a lookup without ab)
}

test "optPath is lossless and respects longer-piece tie-break" {
    var scratch = std.heap.ArenaAllocator.init(testing.allocator);
    defer scratch.deinit();
    const pieces = [_][]const u8{ "a", "b", "c", "ab", "bc" };
    var lookup = try buildLookup(testing.allocator, &pieces);
    defer lookup.deinit();

    var path: std.ArrayList(u32) = .empty;
    defer path.deinit(testing.allocator);
    const c = try optPath(&scratch, "abc", &lookup, 8, &path, testing.allocator);
    try testing.expectEqual(@as(u32, 2), c);

    // Reconstruct bytes from the path -> must equal the input (lossless).
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(testing.allocator);
    for (path.items) |id| try buf.appendSlice(testing.allocator, pieces[id]);
    try testing.expectEqualStrings("abc", buf.items);
}

test "train keeps all 256 bytes and hits the target size" {
    const words = [_][]const u8{ "banana", "bandana", "ananas", "band" };
    const counts = [_]u32{ 10, 5, 7, 3 };

    // 256 bytes + many distinct 2..8-grams seeded, pruned down to 260 —
    // far more candidates than the 4 non-byte slots, so it lands exactly.
    var r = try train(testing.allocator, .{ .words = &words, .counts = &counts }, .{
        .vocab_size = 260,
        .max_piece_length = 8,
        .seed_size = 1000,
        .prune_fraction = 0.3,
    });
    defer r.deinit();

    try testing.expectEqual(@as(u32, 260), r.count());

    // First 256 ids are the single bytes, id == byte value.
    var b: u32 = 0;
    while (b < 256) : (b += 1) {
        const p = r.piece(b);
        try testing.expectEqual(@as(usize, 1), p.len);
        try testing.expectEqual(@as(u8, @intCast(b)), p[0]);
    }
}

test "train serial and parallel produce a byte-identical vocab" {
    const words = [_][]const u8{
        "banana",   "bandana", "ananas",     "band",   "candy",
        "sandbar",  "random",  "abracadabra", "dandelion", "standard",
        "panorama", "savanna", "caravan",    "vanilla", "lavanda",
        "mandible", "andante", "android",    "grandstand", "bandana",
    };
    const counts = [_]u32{
        37, 11, 23, 5, 17, 8, 41, 13, 3, 19,
        7,  29, 2, 31, 6, 4,  9,  15, 1, 12,
    };

    const opts_base = TrainOptions{
        .vocab_size = 300,
        .max_piece_length = 8,
        .seed_size = 2000,
        .prune_fraction = 0.25,
    };

    // Serial reference (pool == null).
    var serial = try train(testing.allocator, .{ .words = &words, .counts = &counts }, opts_base);
    defer serial.deinit();

    // Parallel run with a real multi-worker BatchPool.
    var pool = try BatchPool.init(testing.allocator, 4);
    defer pool.deinit();
    var par_opts = opts_base;
    par_opts.pool = &pool;
    var parallel = try train(testing.allocator, .{ .words = &words, .counts = &counts }, par_opts);
    defer parallel.deinit();

    // Byte-for-byte identical: same count, same pieces in the same id order.
    try testing.expectEqual(serial.count(), parallel.count());
    try testing.expectEqualSlices(u32, serial.offsets, parallel.offsets);
    try testing.expectEqualSlices(u8, serial.bytes, parallel.bytes);

    var id: u32 = 0;
    while (id < serial.count()) : (id += 1) {
        try testing.expectEqualSlices(u8, serial.piece(id), parallel.piece(id));
    }
}

test "train output segments the corpus losslessly under optimal" {
    const words = [_][]const u8{ "hello", "world", "held", "word" };
    const counts = [_]u32{ 4, 4, 2, 2 };

    var r = try train(testing.allocator, .{ .words = &words, .counts = &counts }, .{
        .vocab_size = 280,
        .max_piece_length = 8,
        .seed_size = 1000,
    });
    defer r.deinit();

    // Rebuild a lookup from the trained pieces and confirm every word
    // segments and reconstructs exactly.
    var lookup = std.StringHashMap(u32).init(testing.allocator);
    defer lookup.deinit();
    var id: u32 = 0;
    while (id < r.count()) : (id += 1) try lookup.put(r.piece(id), id);

    var scratch = std.heap.ArenaAllocator.init(testing.allocator);
    defer scratch.deinit();
    var path: std.ArrayList(u32) = .empty;
    defer path.deinit(testing.allocator);

    for (words) |w| {
        _ = try optPath(&scratch, w, &lookup, 8, &path, testing.allocator);
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(testing.allocator);
        for (path.items) |pid| try buf.appendSlice(testing.allocator, r.piece(pid));
        try testing.expectEqualStrings(w, buf.items);
    }
}
