//! Byte-level BPE training, incremental variant.
//!
//! Algorithm (HF tokenizers `models/bpe/trainer.rs`, tiktoken-style):
//!   1. Seed vocab with 256 single-byte tokens.
//!   2. Represent each word as a doubly-linked list of part ids over flat
//!      arrays (`parts`, `prev`, `next`). One big arena slab per corpus.
//!   3. INITIAL PASS (multithreaded): walk every word once, populate the
//!      pair-count map and a reverse index `pair -> [(word_idx, pos)]`.
//!   4. MERGE LOOP (single-threaded, V iterations):
//!        a. Linear scan of pair_counts for the highest count; tiebreak by
//!           smallest (left_id, right_id).
//!        b. For each occurrence (word_idx, pos) of the winning pair —
//!           sorted by (word_idx, pos) for left-to-right determinism —
//!           validate (parts[pos] still == left, next[pos] alive, parts of
//!           next still == right), then splice out the right neighbor.
//!           Decrement the two adjacent pair counts (prev_pos, pos) and
//!           (right_pos, right_neighbor); increment the new pairs
//!           (prev_pos, new_id) and (new_id, right_neighbor), pushing
//!           those positions into the reverse index.
//!        c. Drop the merged pair's reverse-index list and count.
//!
//! Complexity: O(N) for the initial pass (N = total corpus part count),
//! then O(V * (P + K)) for the merge loop where P is the average live
//! pair-count map size and K is the avg occurrences per winning pair.
//! Previously this was O(V * N) per merge — quadratic in vocab size.
//!
//! Lazy-delete scheme for the reverse index:
//!   The per-pair occurrence list is an ArrayList(Occurrence). When a
//!   neighbor pair is invalidated we DO NOT scan its list to remove the
//!   stale entry; we just let it sit. At iteration time we re-validate
//!   each occurrence by checking the linked-list state. Counts in
//!   `pair_counts` stay authoritative because every neighbor change emits
//!   a paired decrement+increment. This keeps the merge step O(K) per
//!   pair occurrence with no list-scan overhead.
//!
//! Threading: only the initial pair-counting pass uses `BatchPool` (per-
//! worker maps merged at the end). Merge iterations touch a handful of
//! positions each — the spawn cost dwarfs the work.
//!
//! SuperBPE mode implements the two-stage pretokenization curriculum from
//! Liu et al., COLM 2025 (https://arxiv.org/abs/2503.13423): phase one blocks
//! merges at pretoken boundaries, then phase two lifts those boundaries while
//! preserving the learned merge ranks.  The phase transition/recount uses the
//! same live linked-list arena, avoiding a second raw-document representation.
//! The frequency-aggregation direction and explicit two-phase formulation are
//! also informed by Schmidt et al., *Faster Superword Tokenization* (2026),
//! https://arxiv.org/abs/2604.05192.

const std = @import("std");
const TokenId = @import("token.zig").TokenId;
const Span = @import("token.zig").Span;
const Bpe = @import("bpe.zig").Bpe;
const BatchPool = @import("thread_pool.zig").BatchPool;
const negative_train = @import("negative_train.zig");

pub const TrainOptions = struct {
    vocab_size: u32,
    pool: ?*BatchPool = null,
    on_merge: ?*const fn (ctx: *anyopaque, vocab_size: u32, pair_count: u32) void = null,
    on_merge_ctx: *anyopaque = undefined,
    /// Optional avoid-pattern list. When non-null, merge candidates whose
    /// concatenated bytes contain any pattern are demoted (`.penalize`)
    /// or filtered (`.exclude`). See `negative_train.zig`.
    avoid: ?*const negative_train.AvoidList = null,
    avoid_mode: negative_train.Mode = .penalize,
    /// 0 is ordinary BPE. Otherwise, pretoken boundaries are enforced until
    /// this vocabulary size, then lifted for SuperBPE superword merges.
    superword_phase_vocab: u32 = 0,
};

pub const Corpus = struct {
    words: []const []const u8,
    counts: []const u32,
    /// Optional per-document masks. `mask[i] == true` blocks the adjacency
    /// immediately before byte i during phase one only.
    phase1_blocked_before: ?[]const []const bool = null,
};

inline fn packPair(left: u32, right: u32) u64 {
    return (@as(u64, left) << 32) | @as(u64, right);
}

inline fn unpackLeft(key: u64) u32 {
    return @intCast(key >> 32);
}

inline fn unpackRight(key: u64) u32 {
    return @intCast(key & 0xffff_ffff);
}

const Occurrence = packed struct(u64) {
    word_idx: u32,
    pos: u32,
};

const PairCountMap = std.AutoHashMap(u64, u64);
const OccList = std.ArrayList(Occurrence);
const PairOccMap = std.AutoHashMap(u64, OccList);

/// Slab layout for all words' linked-list state. Each word occupies a
/// contiguous span `[starts[i], starts[i]+lens[i])` in the flat arrays.
const WordArena = struct {
    parts: []u32,
    prev: []i32,
    next: []i32,
    starts: []u32,
    lens: []u32, // original byte-length; the linked list never resizes
    phase1_blocked_before: ?[]bool,

    fn slabSize(corpus: Corpus) usize {
        var sum: usize = 0;
        for (corpus.words) |w| sum += w.len;
        return sum;
    }

    fn init(allocator: std.mem.Allocator, corpus: Corpus) !WordArena {
        if (corpus.phase1_blocked_before) |masks|
            if (masks.len != corpus.words.len) return error.CorpusLengthMismatch;
        const total = slabSize(corpus);
        const parts = try allocator.alloc(u32, total);
        errdefer allocator.free(parts);
        const prev = try allocator.alloc(i32, total);
        errdefer allocator.free(prev);
        const next = try allocator.alloc(i32, total);
        errdefer allocator.free(next);
        const starts = try allocator.alloc(u32, corpus.words.len);
        errdefer allocator.free(starts);
        const lens = try allocator.alloc(u32, corpus.words.len);
        errdefer allocator.free(lens);
        const blocked = if (corpus.phase1_blocked_before != null)
            try allocator.alloc(bool, total)
        else
            null;
        errdefer if (blocked) |value| allocator.free(value);

        var cursor: u32 = 0;
        for (corpus.words, 0..) |w, i| {
            if (corpus.phase1_blocked_before) |masks|
                if (masks[i].len != w.len) return error.CorpusLengthMismatch;
            starts[i] = cursor;
            lens[i] = @intCast(w.len);
            for (w, 0..) |byte, j| {
                const idx = cursor + @as(u32, @intCast(j));
                parts[idx] = byte;
                prev[idx] = if (j == 0) -1 else @intCast(idx - 1);
                next[idx] = if (j + 1 == w.len) -1 else @intCast(idx + 1);
                if (blocked) |value| value[idx] = corpus.phase1_blocked_before.?[i][j];
            }
            cursor += @intCast(w.len);
        }
        return .{
            .parts = parts,
            .prev = prev,
            .next = next,
            .starts = starts,
            .lens = lens,
            .phase1_blocked_before = blocked,
        };
    }

    fn deinit(self: *WordArena, allocator: std.mem.Allocator) void {
        allocator.free(self.parts);
        allocator.free(self.prev);
        allocator.free(self.next);
        allocator.free(self.starts);
        allocator.free(self.lens);
        if (self.phase1_blocked_before) |value| allocator.free(value);
    }
};

inline fn phase1Allows(arena: *const WordArena, right_pos: u32) bool {
    const blocked = arena.phase1_blocked_before orelse return true;
    return !blocked[right_pos];
}

const CountCtx = struct {
    arena: *const WordArena,
    counts: []const u32,
    pair_maps: []PairCountMap,
    occ_maps: []PairOccMap,
    phase1: bool,
};

const CountWorker = struct {
    pub fn run(ctx: *CountCtx, word_idx: usize, worker_idx: usize) void {
        const freq = ctx.counts[word_idx];
        if (freq == 0) return;
        const ln = ctx.arena.lens[word_idx];
        if (ln < 2) return;
        const start = ctx.arena.starts[word_idx];
        const parts = ctx.arena.parts;
        const pair_map = &ctx.pair_maps[worker_idx];
        const occ_map = &ctx.occ_maps[worker_idx];
        var i: u32 = 0;
        while (i + 1 < ln) : (i += 1) {
            const pos = start + i;
            if (ctx.phase1 and !phase1Allows(ctx.arena, pos + 1)) continue;
            const key = packPair(parts[pos], parts[pos + 1]);
            const pgop = pair_map.getOrPut(key) catch return;
            if (!pgop.found_existing) pgop.value_ptr.* = 0;
            pgop.value_ptr.* +%= freq;

            const ogop = occ_map.getOrPut(key) catch return;
            if (!ogop.found_existing) ogop.value_ptr.* = .empty;
            ogop.value_ptr.append(pair_map.allocator, .{
                .word_idx = @intCast(word_idx),
                .pos = pos,
            }) catch return;
        }
    }
};

fn initialCount(
    allocator: std.mem.Allocator,
    arena: *const WordArena,
    counts: []const u32,
    pool: ?*BatchPool,
    out_pairs: *PairCountMap,
    out_occ: *PairOccMap,
    phase1: bool,
) !void {
    if (pool) |bp| {
        const n = bp.workerCount();
        const pmaps = try allocator.alloc(PairCountMap, n);
        defer {
            for (pmaps) |*m| m.deinit();
            allocator.free(pmaps);
        }
        const omaps = try allocator.alloc(PairOccMap, n);
        defer {
            for (omaps) |*m| {
                var it = m.iterator();
                while (it.next()) |e| e.value_ptr.deinit(allocator);
                m.deinit();
            }
            allocator.free(omaps);
        }
        for (pmaps) |*m| m.* = .init(allocator);
        for (omaps) |*m| m.* = .init(allocator);

        var ctx: CountCtx = .{
            .arena = arena,
            .counts = counts,
            .pair_maps = pmaps,
            .occ_maps = omaps,
            .phase1 = phase1,
        };
        try bp.runBatch(CountWorker, &ctx, arena.starts.len);

        for (pmaps) |*m| {
            var it = m.iterator();
            while (it.next()) |e| {
                const gop = try out_pairs.getOrPut(e.key_ptr.*);
                if (!gop.found_existing) gop.value_ptr.* = 0;
                gop.value_ptr.* +%= e.value_ptr.*;
            }
        }
        for (omaps) |*m| {
            var it = m.iterator();
            while (it.next()) |e| {
                const gop = try out_occ.getOrPut(e.key_ptr.*);
                if (!gop.found_existing) gop.value_ptr.* = .empty;
                try gop.value_ptr.appendSlice(allocator, e.value_ptr.items);
            }
        }
    } else {
        const parts = arena.parts;
        for (arena.starts, arena.lens, counts, 0..) |start, ln, freq, word_idx| {
            if (freq == 0 or ln < 2) continue;
            var i: u32 = 0;
            while (i + 1 < ln) : (i += 1) {
                const pos = start + i;
                if (phase1 and !phase1Allows(arena, pos + 1)) continue;
                const key = packPair(parts[pos], parts[pos + 1]);
                const pgop = try out_pairs.getOrPut(key);
                if (!pgop.found_existing) pgop.value_ptr.* = 0;
                pgop.value_ptr.* +%= freq;

                const ogop = try out_occ.getOrPut(key);
                if (!ogop.found_existing) ogop.value_ptr.* = .empty;
                try ogop.value_ptr.append(allocator, .{
                    .word_idx = @intCast(word_idx),
                    .pos = pos,
                });
            }
        }
    }
}

fn clearOccurrences(allocator: std.mem.Allocator, map: *PairOccMap) void {
    var it = map.iterator();
    while (it.next()) |entry| entry.value_ptr.deinit(allocator);
    map.clearRetainingCapacity();
}

/// Recount the current linked-list token stream when SuperBPE lifts the
/// pretoken boundary restriction. This preserves every phase-one merge rank.
fn recountLive(allocator: std.mem.Allocator, arena: *const WordArena, counts: []const u32, phase1: bool, pairs: *PairCountMap, occurrences: *PairOccMap) !void {
    pairs.clearRetainingCapacity();
    clearOccurrences(allocator, occurrences);
    for (arena.starts, arena.lens, counts, 0..) |start, len, freq, word_idx| {
        if (len < 2 or freq == 0) continue;
        var pos = start;
        while (arena.next[pos] >= 0) {
            const right: u32 = @intCast(arena.next[pos]);
            if (!phase1 or phase1Allows(arena, right)) {
                const key = packPair(arena.parts[pos], arena.parts[right]);
                try incPair(pairs, key, freq);
                try pushOcc(allocator, occurrences, key, .{
                    .word_idx = @intCast(word_idx),
                    .pos = pos,
                });
            }
            pos = right;
        }
    }
}

/// Context passed to `pickBestPair` so it can evaluate whether a
/// candidate merge would produce avoid-pattern bytes. The bytes for any
/// candidate pair (l,r) are reconstructed from `token_bytes` and
/// `token_offsets` (the same SoA the merge loop writes into).
const AvoidCtx = struct {
    avoid: *const negative_train.AvoidList,
    mode: negative_train.Mode,
    token_bytes: []const u8,
    token_offsets: []const u32,
    /// Scratch buffer reused per candidate to hold (left ++ right). The
    /// caller owns it; sized to >= longest merged-piece length seen so
    /// far. We reallocate inside `pickBestPair` if a longer pair appears
    /// to keep call-site code simple.
    scratch: *std.ArrayList(u8),
    allocator: std.mem.Allocator,
};

/// Builds the concatenated (left ++ right) bytes into `ctx.scratch`,
/// returning a slice. Cheap because token byte-strings are typically
/// short and the buffer is reused.
fn buildMergedBytes(ctx: *AvoidCtx, left: u32, right: u32) ![]const u8 {
    const a_start = ctx.token_offsets[left];
    const a_end = ctx.token_offsets[left + 1];
    const b_start = ctx.token_offsets[right];
    const b_end = ctx.token_offsets[right + 1];
    const total = (a_end - a_start) + (b_end - b_start);
    ctx.scratch.clearRetainingCapacity();
    try ctx.scratch.ensureTotalCapacity(ctx.allocator, total);
    ctx.scratch.appendSliceAssumeCapacity(ctx.token_bytes[a_start..a_end]);
    ctx.scratch.appendSliceAssumeCapacity(ctx.token_bytes[b_start..b_end]);
    return ctx.scratch.items;
}

/// Effective score for a candidate pair under the avoid list.
///   exclude  → matching pairs report score 0 (skipped by the > 0 guard).
///   penalize → matching pairs report `count - PENALTY` (saturating at 0).
/// Non-matching candidates pass through unchanged.
fn effectiveCount(ctx: ?*AvoidCtx, key: u64, raw: u64) u64 {
    const c = ctx orelse return raw;
    const left = unpackLeft(key);
    const right = unpackRight(key);
    const merged = buildMergedBytes(c, left, right) catch return raw;
    if (!c.avoid.matches(merged)) return raw;
    return switch (c.mode) {
        .exclude => 0,
        .penalize => blk: {
            const p: u64 = @intCast(@max(@as(i64, 0), -negative_train.PENALTY));
            break :blk if (raw > p) raw - p else 0;
        },
    };
}

/// Linear scan of pair counts. Deterministic: highest count wins;
/// tiebreak by smallest left id, then smallest right id. When `avoid`
/// is set, candidates whose merged bytes contain an avoid-pattern are
/// demoted (penalize) or skipped entirely (exclude).
fn pickBestPair(map: *const PairCountMap, avoid: ?*AvoidCtx) ?u64 {
    var best_key: u64 = 0;
    var best_count: u64 = 0;
    var have_best: bool = false;
    var it = map.iterator();
    while (it.next()) |e| {
        const raw = e.value_ptr.*;
        if (raw == 0) continue;
        const k = e.key_ptr.*;
        const c = effectiveCount(avoid, k, raw);
        if (c == 0) continue;
        if (!have_best or c > best_count) {
            best_key = k;
            best_count = c;
            have_best = true;
            continue;
        }
        if (c == best_count) {
            const new_l = unpackLeft(k);
            const new_r = unpackRight(k);
            const cur_l = unpackLeft(best_key);
            const cur_r = unpackRight(best_key);
            if (new_l < cur_l or (new_l == cur_l and new_r < cur_r)) {
                best_key = k;
            }
        }
    }
    return if (have_best) best_key else null;
}

fn occLessThan(_: void, a: Occurrence, b: Occurrence) bool {
    if (a.word_idx != b.word_idx) return a.word_idx < b.word_idx;
    return a.pos < b.pos;
}

/// Decrement a pair's count. If it falls to zero, drop the entry so
/// `pickBestPair` skips it and the merge loop terminates cleanly.
fn decPair(map: *PairCountMap, key: u64, delta: u64) void {
    const e = map.getEntry(key) orelse return;
    if (e.value_ptr.* > delta) {
        e.value_ptr.* -= delta;
    } else {
        _ = map.remove(key);
    }
}

fn incPair(map: *PairCountMap, key: u64, delta: u64) !void {
    const gop = try map.getOrPut(key);
    if (!gop.found_existing) gop.value_ptr.* = 0;
    gop.value_ptr.* +%= delta;
}

fn pushOcc(
    allocator: std.mem.Allocator,
    map: *PairOccMap,
    key: u64,
    occ: Occurrence,
) !void {
    const gop = try map.getOrPut(key);
    if (!gop.found_existing) gop.value_ptr.* = .empty;
    try gop.value_ptr.append(allocator, occ);
}

pub fn train(allocator: std.mem.Allocator, corpus: Corpus, opts: TrainOptions) !Bpe {
    if (opts.vocab_size < 256) return error.VocabTooSmall;
    if (corpus.words.len != corpus.counts.len) return error.CorpusLengthMismatch;
    if (opts.superword_phase_vocab != 0 and
        (opts.superword_phase_vocab <= 256 or opts.superword_phase_vocab >= opts.vocab_size))
        return error.InvalidSuperwordTransition;
    if (opts.superword_phase_vocab != 0 and corpus.phase1_blocked_before == null)
        return error.MissingPretokenBoundaries;

    var token_bytes: std.ArrayList(u8) = .empty;
    defer token_bytes.deinit(allocator);
    var token_offsets: std.ArrayList(u32) = .empty;
    defer token_offsets.deinit(allocator);

    try token_bytes.ensureTotalCapacity(allocator, 256);
    try token_offsets.ensureTotalCapacity(allocator, 257);
    try token_offsets.append(allocator, 0);
    var b: u32 = 0;
    while (b < 256) : (b += 1) {
        try token_bytes.append(allocator, @intCast(b));
        try token_offsets.append(allocator, @intCast(token_bytes.items.len));
    }

    var arena = try WordArena.init(allocator, corpus);
    defer arena.deinit(allocator);

    var pair_counts: PairCountMap = .init(allocator);
    defer pair_counts.deinit();
    var pair_occ: PairOccMap = .init(allocator);
    defer {
        var it = pair_occ.iterator();
        while (it.next()) |e| e.value_ptr.deinit(allocator);
        pair_occ.deinit();
    }

    var phase1 = opts.superword_phase_vocab != 0;
    try initialCount(allocator, &arena, corpus.counts, opts.pool, &pair_counts, &pair_occ, phase1);

    var winning: std.ArrayList(Occurrence) = .empty;
    defer winning.deinit(allocator);

    // Avoid-list scratch — shared across all pickBestPair calls so we
    // don't reallocate per candidate. Only present when opts.avoid is set.
    var avoid_scratch: std.ArrayList(u8) = .empty;
    defer avoid_scratch.deinit(allocator);
    var avoid_ctx: AvoidCtx = undefined;
    var avoid_ptr: ?*AvoidCtx = null;
    if (opts.avoid) |al| {
        avoid_ctx = .{
            .avoid = al,
            .mode = opts.avoid_mode,
            .token_bytes = &.{},
            .token_offsets = &.{},
            .scratch = &avoid_scratch,
            .allocator = allocator,
        };
        avoid_ptr = &avoid_ctx;
    }

    var cur_vocab: u32 = 256;
    while (cur_vocab < opts.vocab_size) {
        if (phase1 and cur_vocab >= opts.superword_phase_vocab) {
            phase1 = false;
            try recountLive(allocator, &arena, corpus.counts, false, &pair_counts, &pair_occ);
        }
        // Refresh the avoid context's byte/offset views — token_bytes
        // grows each merge as new pieces are appended.
        if (avoid_ptr) |ap| {
            ap.token_bytes = token_bytes.items;
            ap.token_offsets = token_offsets.items;
        }
        var best_opt = pickBestPair(&pair_counts, avoid_ptr);
        // A tiny corpus can exhaust all within-pretoken pairs before the
        // requested transition. Lift early instead of stopping below target.
        if (best_opt == null and phase1) {
            phase1 = false;
            try recountLive(allocator, &arena, corpus.counts, false, &pair_counts, &pair_occ);
            best_opt = pickBestPair(&pair_counts, avoid_ptr);
        }
        const best = best_opt orelse break;
        const left = unpackLeft(best);
        const right = unpackRight(best);
        const new_id = cur_vocab;

        // Append the new token's bytes (left ++ right) into the staging
        // buffer. Done up front so a mid-loop allocation failure leaves
        // the arena consistent enough for cleanup.
        const a_start = token_offsets.items[left];
        const a_end = token_offsets.items[left + 1];
        const b_start = token_offsets.items[right];
        const b_end = token_offsets.items[right + 1];
        try token_bytes.appendSlice(allocator, token_bytes.items[a_start..a_end]);
        try token_bytes.appendSlice(allocator, token_bytes.items[b_start..b_end]);
        try token_offsets.append(allocator, @intCast(token_bytes.items.len));

        // Gather the winning pair's occurrences, validate, sort. Sorting
        // by (word_idx, pos) replays a left-to-right merge over each word
        // which matches HF's `merge` pass and keeps overlapping-pair
        // resolution (e.g. `aaaa` with pair `(a,a)`) deterministic.
        const occ_entry_ptr = pair_occ.getPtr(best);
        winning.clearRetainingCapacity();
        if (occ_entry_ptr) |list| {
            for (list.items) |occ| {
                const pos = occ.pos;
                if (arena.parts[pos] != left) continue;
                const np = arena.next[pos];
                if (np < 0) continue;
                if (arena.parts[@intCast(np)] != right) continue;
                try winning.append(allocator, occ);
            }
            list.deinit(allocator);
            _ = pair_occ.remove(best);
        }
        _ = pair_counts.remove(best);

        std.mem.sort(Occurrence, winning.items, {}, occLessThan);

        for (winning.items) |occ| {
            const pos = occ.pos;
            const word_idx = occ.word_idx;
            // Re-validate after prior in-loop merges may have splatted
            // overlapping positions in the same word.
            if (arena.parts[pos] != left) continue;
            const right_pos_i = arena.next[pos];
            if (right_pos_i < 0) continue;
            const right_pos: u32 = @intCast(right_pos_i);
            if (arena.parts[right_pos] != right) continue;

            const freq = corpus.counts[word_idx];

            const prev_pos_i = arena.prev[pos];
            const right_neighbor_i = arena.next[right_pos];

            // Decrement the two old neighbor-pair counts.
            if (prev_pos_i >= 0) {
                const pp: u32 = @intCast(prev_pos_i);
                const old_key = packPair(arena.parts[pp], left);
                decPair(&pair_counts, old_key, freq);
                // Don't bother purging the stale (pp) entry from the
                // old_key list — lazy-delete handles it at iteration.
            }
            if (right_neighbor_i >= 0) {
                const rn: u32 = @intCast(right_neighbor_i);
                const old_key = packPair(right, arena.parts[rn]);
                decPair(&pair_counts, old_key, freq);
            }

            // Splice: pos becomes new_id, right_pos is unlinked.
            arena.parts[pos] = new_id;
            arena.next[pos] = right_neighbor_i;
            if (right_neighbor_i >= 0) {
                arena.prev[@intCast(right_neighbor_i)] = @intCast(pos);
            }
            // Leave parts[right_pos] alone — its mismatch on the next
            // iteration's validation is what marks it dead.

            // Increment the two new neighbor pairs and push positions
            // into their reverse-index lists.
            if (prev_pos_i >= 0) {
                const pp: u32 = @intCast(prev_pos_i);
                if (!phase1 or phase1Allows(&arena, pos)) {
                    const new_key = packPair(arena.parts[pp], new_id);
                    try incPair(&pair_counts, new_key, freq);
                    try pushOcc(allocator, &pair_occ, new_key, .{
                        .word_idx = word_idx,
                        .pos = pp,
                    });
                }
            }
            if (right_neighbor_i >= 0) {
                const rn: u32 = @intCast(right_neighbor_i);
                if (!phase1 or phase1Allows(&arena, rn)) {
                    const new_key = packPair(new_id, arena.parts[rn]);
                    try incPair(&pair_counts, new_key, freq);
                    try pushOcc(allocator, &pair_occ, new_key, .{
                        .word_idx = word_idx,
                        .pos = pos,
                    });
                }
            }
        }

        cur_vocab += 1;
        if (opts.on_merge) |cb| cb(opts.on_merge_ctx, cur_vocab, @intCast(pair_counts.count()));
    }

    const final_bytes = try token_bytes.toOwnedSlice(allocator);
    errdefer allocator.free(final_bytes);
    const final_offsets = try token_offsets.toOwnedSlice(allocator);
    errdefer allocator.free(final_offsets);

    const final_count: u32 = @intCast(final_offsets.len - 1);

    var by_bytes: std.StringHashMap(TokenId) = .init(allocator);
    errdefer by_bytes.deinit();
    try by_bytes.ensureTotalCapacity(final_count);
    var r: u32 = 0;
    while (r < final_count) : (r += 1) {
        const key = final_bytes[final_offsets[r]..final_offsets[r + 1]];
        try by_bytes.put(key, r);
    }

    return .{
        .allocator = allocator,
        .bytes = final_bytes,
        .offsets = final_offsets,
        .count = final_count,
        .by_bytes = by_bytes,
    };
}

/// Train SuperBPE on newline-delimited documents. Phase one uses `splitFn`
/// boundaries; phase two lifts only those boundaries, never document lines.
pub fn trainSuperwordFromBytes(
    allocator: std.mem.Allocator,
    raw: []const u8,
    splitFn: *const fn (allocator: std.mem.Allocator, input: []const u8) anyerror![]Span,
    opts: TrainOptions,
) !Bpe {
    if (opts.superword_phase_vocab == 0) return error.InvalidSuperwordTransition;
    var words: std.ArrayList([]const u8) = .empty;
    defer words.deinit(allocator);
    var counts: std.ArrayList(u32) = .empty;
    defer counts.deinit(allocator);
    var masks: std.ArrayList([]const bool) = .empty;
    defer {
        for (masks.items) |mask| allocator.free(mask);
        masks.deinit(allocator);
    }
    var lines = std.mem.splitScalar(u8, raw, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        const spans = try splitFn(allocator, line);
        defer allocator.free(spans);
        const mask = try allocator.alloc(bool, line.len);
        @memset(mask, false);
        for (spans) |span| {
            if (span.start > 0 and span.start < line.len) mask[span.start] = true;
        }
        try words.append(allocator, line);
        try counts.append(allocator, 1);
        try masks.append(allocator, mask);
    }
    if (words.items.len == 0) return error.EmptyCorpus;
    return train(allocator, .{ .words = words.items, .counts = counts.items, .phase1_blocked_before = masks.items }, opts);
}

pub fn trainFromBytes(
    allocator: std.mem.Allocator,
    raw: []const u8,
    splitFn: *const fn (allocator: std.mem.Allocator, input: []const u8) anyerror![]Span,
    opts: TrainOptions,
) !Bpe {
    const spans = try splitFn(allocator, raw);
    defer allocator.free(spans);

    var freq: std.StringHashMap(u32) = .init(allocator);
    defer freq.deinit();
    for (spans) |s| {
        const key = s.slice(raw);
        const gop = try freq.getOrPut(key);
        if (!gop.found_existing) gop.value_ptr.* = 0;
        gop.value_ptr.* +%= 1;
    }

    const n = freq.count();
    const words = try allocator.alloc([]const u8, n);
    defer allocator.free(words);
    const counts = try allocator.alloc(u32, n);
    defer allocator.free(counts);

    var it = freq.iterator();
    var i: usize = 0;
    while (it.next()) |e| : (i += 1) {
        words[i] = e.key_ptr.*;
        counts[i] = e.value_ptr.*;
    }

    return train(allocator, .{ .words = words, .counts = counts }, opts);
}

// --- tests -----------------------------------------------------------

const testing = std.testing;

test "trains to exactly vocab_size" {
    const allocator = testing.allocator;

    var words_storage: [100][]const u8 = undefined;
    var counts_storage: [100]u32 = undefined;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    var seed: u64 = 0xC0FFEE;
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    _ = &seed;

    var w: usize = 0;
    while (w < 100) : (w += 1) {
        const len = rnd.intRangeAtMost(usize, 3, 12);
        const buf = try aa.alloc(u8, len);
        for (buf) |*c| c.* = 'a' + rnd.intRangeAtMost(u8, 0, 7);
        words_storage[w] = buf;
        counts_storage[w] = rnd.intRangeAtMost(u32, 1, 9);
    }

    var bpe = try train(allocator, .{
        .words = &words_storage,
        .counts = &counts_storage,
    }, .{ .vocab_size = 300 });
    defer bpe.deinit();

    try testing.expectEqual(@as(u32, 300), bpe.count);
}

test "round-trips a simple corpus" {
    const allocator = testing.allocator;

    const words = [_][]const u8{ "abc", "bcd" };
    const counts = [_]u32{ 5, 3 };

    var bpe = try train(allocator, .{
        .words = &words,
        .counts = &counts,
    }, .{ .vocab_size = 260 });
    defer bpe.deinit();

    try testing.expect(bpe.count >= 256);
    try testing.expect(bpe.count <= 260);

    var out: [16]TokenId = undefined;
    const ids = bpe.encodeChunk("abc", &out);
    try testing.expect(ids.len > 0);

    var decoded: std.ArrayList(u8) = .empty;
    defer decoded.deinit(allocator);
    for (ids) |id| try decoded.appendSlice(allocator, bpe.idBytes(id));
    try testing.expectEqualStrings("abc", decoded.items);
}

test "deterministic across runs" {
    const allocator = testing.allocator;

    const words = [_][]const u8{ "hello", "world", "help", "word", "wood" };
    const counts = [_]u32{ 4, 3, 2, 2, 1 };

    var bpe_a = try train(allocator, .{
        .words = &words,
        .counts = &counts,
    }, .{ .vocab_size = 280 });
    defer bpe_a.deinit();

    var bpe_b = try train(allocator, .{
        .words = &words,
        .counts = &counts,
    }, .{ .vocab_size = 280 });
    defer bpe_b.deinit();

    try testing.expectEqual(bpe_a.count, bpe_b.count);
    try testing.expectEqualSlices(u8, bpe_a.bytes, bpe_b.bytes);
    try testing.expectEqualSlices(u32, bpe_a.offsets, bpe_b.offsets);
}

test "multithreaded matches single-threaded" {
    const allocator = testing.allocator;

    const words = [_][]const u8{ "abracadabra", "cadaver", "abacus", "barbara", "rhubarb" };
    const counts = [_]u32{ 5, 3, 2, 4, 1 };

    var bpe_serial = try train(allocator, .{
        .words = &words,
        .counts = &counts,
    }, .{ .vocab_size = 290 });
    defer bpe_serial.deinit();

    var pool = try BatchPool.init(allocator, 2);
    defer pool.deinit();

    var bpe_parallel = try train(allocator, .{
        .words = &words,
        .counts = &counts,
    }, .{ .vocab_size = 290, .pool = &pool });
    defer bpe_parallel.deinit();

    try testing.expectEqual(bpe_serial.count, bpe_parallel.count);
    try testing.expectEqualSlices(u8, bpe_serial.bytes, bpe_parallel.bytes);
    try testing.expectEqualSlices(u32, bpe_serial.offsets, bpe_parallel.offsets);
}

fn whitespaceSpans(allocator: std.mem.Allocator, input: []const u8) ![]Span {
    var spans: std.ArrayList(Span) = .empty;
    errdefer spans.deinit(allocator);
    var start: usize = 0;
    var i: usize = 0;
    while (i < input.len) : (i += 1) {
        if (input[i] != ' ') continue;
        if (start < i) try spans.append(allocator, .{ .start = @intCast(start), .end = @intCast(i) });
        try spans.append(allocator, .{ .start = @intCast(i), .end = @intCast(i + 1) });
        start = i + 1;
    }
    if (start < input.len) try spans.append(allocator, .{ .start = @intCast(start), .end = @intCast(input.len) });
    return spans.toOwnedSlice(allocator);
}

test "SuperBPE preserves subwords then learns cross-whitespace tokens" {
    const raw = "new york city\nnew york state\nnew york city\nnew jersey city";
    var bpe = try trainSuperwordFromBytes(testing.allocator, raw, &whitespaceSpans, .{
        .vocab_size = 280,
        .superword_phase_vocab = 268,
    });
    defer bpe.deinit();
    var found_superword = false;
    var id: u32 = 268;
    while (id < bpe.count) : (id += 1) {
        if (std.mem.indexOfScalar(u8, bpe.idBytes(id), ' ') != null) {
            found_superword = true;
            break;
        }
    }
    try testing.expect(found_superword);
    var out: [64]TokenId = undefined;
    const ids = bpe.encodeChunk("new york city", &out);
    var decoded: std.ArrayList(u8) = .empty;
    defer decoded.deinit(testing.allocator);
    for (ids) |token| try decoded.appendSlice(testing.allocator, bpe.idBytes(token));
    try testing.expectEqualStrings("new york city", decoded.items);
}

// Hand-traced 3-merge BPE on the corpus
//   "ab" x4, "ac" x3, "bc" x2
// Byte ids: 'a'=0x61, 'b'=0x62, 'c'=0x63.
//
// Initial pair counts (weighted):
//   (a,b)=4, (a,c)=3, (b,c)=2.
// Merge 1 -> id 256 = "ab". After: word "ab"=[256]; "ac"=[a,c]; "bc"=[b,c].
// Counts now: (a,c)=3, (b,c)=2.
// Merge 2 -> id 257 = "ac" (3 > 2). After: words "ac"=[257]; "bc"=[b,c].
// Counts now: (b,c)=2.
// Merge 3 -> id 258 = "bc".
//
// Final extra tokens (in order): "ab", "ac", "bc". Pinning this trace
// catches any future drift in merge ordering, tiebreak, or splicing.
test "incremental matches hand-traced merges" {
    const allocator = testing.allocator;

    const words = [_][]const u8{ "ab", "ac", "bc" };
    const counts = [_]u32{ 4, 3, 2 };

    var bpe = try train(allocator, .{
        .words = &words,
        .counts = &counts,
    }, .{ .vocab_size = 259 });
    defer bpe.deinit();

    try testing.expectEqual(@as(u32, 259), bpe.count);
    try testing.expectEqualStrings("ab", bpe.idBytes(256));
    try testing.expectEqualStrings("ac", bpe.idBytes(257));
    try testing.expectEqualStrings("bc", bpe.idBytes(258));

    // Cross-check with multithreaded run: must produce identical bytes
    // and offsets, including the new tokens appended in the same order.
    var pool = try BatchPool.init(allocator, 3);
    defer pool.deinit();
    var bpe_par = try train(allocator, .{
        .words = &words,
        .counts = &counts,
    }, .{ .vocab_size = 259, .pool = &pool });
    defer bpe_par.deinit();
    try testing.expectEqualSlices(u8, bpe.bytes, bpe_par.bytes);
    try testing.expectEqualSlices(u32, bpe.offsets, bpe_par.offsets);
}
