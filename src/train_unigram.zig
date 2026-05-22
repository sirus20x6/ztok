//! Unigram language-model trainer (SentencePiece-style EM).
//!
//! Pipeline:
//!   1. Seed vocab = 256 single bytes (byte_fallback) + all unique substrings
//!      of the corpus up to `max_piece_length` bytes, scored by their
//!      occurrence count weighted by piece length (matches SP's
//!      `MakeSeedSentencePieces`, refs/sentencepiece/src/unigram_model_trainer.cc:113).
//!      The seed list is capped at `seed_size` (default 1,000,000) — anything
//!      beyond is dropped by ascending score.
//!   2. EM rounds. Each round runs `em_iters_per_round` E/M passes then prunes
//!      `prune_fraction` of the smallest-score pieces. Repeat until vocab is
//!      within ~5% of the target.
//!   3. E-step: per word, build a lattice over byte positions, compute
//!      forward `alpha[]` and backward `beta[]` in log space, accumulate the
//!      posterior expected count of each piece (`alpha[i] + score + beta[i+L] - Z`).
//!      All per-state buffers are flat `[]f32` SoA, allocated once per word.
//!      When `subword_reg_samples > 0`, the analytic posterior is replaced by
//!      a Monte-Carlo estimate: sample N segmentations per word and count
//!      empirical piece occurrences, divided by N.
//!   4. M-step: new_score[p] = log(expected[p] / sum_expected). Pieces whose
//!      expected count is < 0.5 are dropped (SP convention, trainer.cc:464).
//!      The 256 byte pieces are pinned — never pruned — so byte_fallback always
//!      works.
//!   5. Final pruning trims to exactly `vocab_size`, then converts the SoA
//!      arrays into a `Unigram` via its `Builder`.
//!
//! Simplifications vs SentencePiece (refs/sentencepiece/src/unigram_model_trainer.cc):
//!   * Seed enumeration uses a hashmap over substrings, not a suffix array
//!     (SP uses esaxx, trainer.cc:148). Slower asymptotically (O(N·L²)) but
//!     trivial to write and bounded by `seed_size`.
//!   * Pruning is naive "drop lowest-score" rather than SP's loss-based
//!     `PruneSentencePieces` (trainer.cc:266) which re-segments each piece
//!     with its 2nd-best path and computes marginal contribution. This means
//!     we keep a few more low-quality pieces than SP would but converges to
//!     the same target size.
//!   * M-step uses plain MLE renormalization, not SP's Bayesian/digamma
//!     EM (trainer.cc:497). Effect is minor for v1 and avoids a digamma
//!     implementation.
//!   * No `splitDigits`, no `splitByWhitespace` etc. — input is bytes in,
//!     bytes out. Pre-tokenization is the caller's job (same convention as
//!     `train_bpe`).
//!
//! Threading: E-step is parallelized via `BatchPool`. Each worker fills a
//! private `[]f64` expected-counts buffer; we reduce at the end. M-step and
//! pruning are single-threaded — both O(|vocab|).
//!
//! Complexity per round: seed is O(|corpus| · L²) one-shot. E-step per
//! iteration is O(|corpus| · L · |vocab_avg_per_pos|) — for a 1 MB corpus
//! with target 8k vocab and 5 rounds expect ~30 s single-threaded ReleaseFast.

const std = @import("std");
const TokenId = @import("token.zig").TokenId;
const Unigram = @import("unigram.zig").Unigram;
const BatchPool = @import("thread_pool.zig").BatchPool;
const negative_train = @import("negative_train.zig");

const NEG_INF: f32 = -std.math.inf(f32);
const UNK_SCORE: f32 = -100.0;
const MIN_EXPECTED: f64 = 0.5;

pub const TrainOptions = struct {
    vocab_size: u32,
    max_piece_length: u8 = 16,
    em_iters_per_round: u32 = 2,
    prune_fraction: f32 = 0.20,
    seed_size: usize = 1_000_000,
    pool: ?*BatchPool = null,
    on_round: ?*const fn (ctx: *anyopaque, vocab_size: u32, avg_loglik: f32) void = null,
    on_round_ctx: *anyopaque = undefined,
    /// If non-zero, encode each corpus word `subword_reg_samples` times
    /// per E-step iteration using subword-regularization sampling. The
    /// resulting posterior counts are averaged across samples. alpha
    /// controls the sampling sharpness.
    subword_reg_samples: u32 = 0,
    subword_reg_alpha: f32 = 1.0,
    /// Seed for the RNG. Same seed → same training run.
    subword_reg_seed: u64 = 0x5A_54_4F_4B_5A_54_4F_4B,
    /// Optional avoid-pattern list. Seed pieces containing any pattern
    /// are demoted (`.penalize` adds `negative_train.PENALTY` to the
    /// log-prob, pushing them to the bottom of every prune sort) or
    /// dropped (`.exclude` keeps them out of the seed entirely).
    avoid: ?*const negative_train.AvoidList = null,
    avoid_mode: negative_train.Mode = .penalize,
};

pub const Corpus = struct {
    words: []const []const u8,
    counts: []const u32,
};

// --- helpers -------------------------------------------------------------

inline fn logSumExp(a: f32, b: f32) f32 {
    if (a == NEG_INF) return b;
    if (b == NEG_INF) return a;
    const hi = @max(a, b);
    const lo = @min(a, b);
    if (hi - lo > 50.0) return hi;
    return hi + @log(1.0 + @exp(lo - hi));
}

inline fn gumbel(rng: std.Random) f32 {
    const u_raw = rng.float(f32);
    const u = if (u_raw < 1e-20) @as(f32, 1e-20) else u_raw;
    return -@log(-@log(u));
}

// --- internal mutable vocab ----------------------------------------------
//
// All piece state lives in SoA arrays. `bytes` is the concatenation of all
// piece byte strings; `offsets` carves it into pieces. We never resize a
// piece in place — pruning just rewrites the arrays into a fresh layout.

const Vocab = struct {
    allocator: std.mem.Allocator,
    bytes: std.ArrayList(u8),
    offsets: std.ArrayList(u32),
    scores: std.ArrayList(f32),
    is_byte: std.ArrayList(bool),

    fn init(allocator: std.mem.Allocator) Vocab {
        return .{
            .allocator = allocator,
            .bytes = .empty,
            .offsets = .empty,
            .scores = .empty,
            .is_byte = .empty,
        };
    }

    fn deinit(self: *Vocab) void {
        self.bytes.deinit(self.allocator);
        self.offsets.deinit(self.allocator);
        self.scores.deinit(self.allocator);
        self.is_byte.deinit(self.allocator);
    }

    fn count(self: *const Vocab) u32 {
        return @intCast(self.scores.items.len);
    }

    fn piece(self: *const Vocab, id: u32) []const u8 {
        const s = self.offsets.items[id];
        const e = self.offsets.items[id + 1];
        return self.bytes.items[s..e];
    }

    fn add(self: *Vocab, piece_bytes: []const u8, score: f32, is_byte_piece: bool) !void {
        if (self.offsets.items.len == 0) try self.offsets.append(self.allocator, 0);
        try self.bytes.appendSlice(self.allocator, piece_bytes);
        try self.offsets.append(self.allocator, @intCast(self.bytes.items.len));
        try self.scores.append(self.allocator, score);
        try self.is_byte.append(self.allocator, is_byte_piece);
    }
};

// --- piece trie (mirrors Unigram inference, but mutable) -----------------
//
// For the E-step we need: at byte position `i`, enumerate every piece that
// matches starting at `i`. A trie keyed by byte gives us this in
// O(matching_pieces) per position. We rebuild the trie every round (after
// pruning) so the indices are tight.

const Trie = struct {
    const NodeId = u32;
    const NO_TOK: u32 = 0xFFFF_FFFF;

    allocator: std.mem.Allocator,
    // Per-node: token id, plus a 256-entry inline child table — flat for
    // O(1) walk. Memory is O(|nodes|·256·4) which for a typical seed of
    // ~100k pieces and ~500k nodes runs ~500 MB — too big. Use a sparse
    // child map instead: a `std.AutoHashMap(u64, u32)` keyed by
    // `(node_id<<8) | byte`.
    token_id: std.ArrayList(u32),
    children: std.AutoHashMap(u64, u32),

    fn init(allocator: std.mem.Allocator) !Trie {
        var t: Trie = .{
            .allocator = allocator,
            .token_id = .empty,
            .children = std.AutoHashMap(u64, u32).init(allocator),
        };
        try t.token_id.append(allocator, NO_TOK); // root
        return t;
    }

    fn deinit(self: *Trie) void {
        self.token_id.deinit(self.allocator);
        self.children.deinit();
    }

    inline fn key(parent: u32, byte: u8) u64 {
        return (@as(u64, parent) << 8) | @as(u64, byte);
    }

    fn insert(self: *Trie, piece_bytes: []const u8, id: u32) !void {
        var cur: u32 = 0;
        for (piece_bytes) |b| {
            const k = key(cur, b);
            const gop = try self.children.getOrPut(k);
            if (!gop.found_existing) {
                const new_id: u32 = @intCast(self.token_id.items.len);
                try self.token_id.append(self.allocator, NO_TOK);
                gop.value_ptr.* = new_id;
            }
            cur = gop.value_ptr.*;
        }
        self.token_id.items[cur] = id;
    }

    inline fn child(self: *const Trie, parent: u32, byte: u8) ?u32 {
        return self.children.get(key(parent, byte));
    }

    inline fn tokenAt(self: *const Trie, node: u32) u32 {
        return self.token_id.items[node];
    }
};

fn buildTrie(allocator: std.mem.Allocator, vocab: *const Vocab) !Trie {
    var t = try Trie.init(allocator);
    errdefer t.deinit();
    var id: u32 = 0;
    while (id < vocab.count()) : (id += 1) {
        try t.insert(vocab.piece(id), id);
    }
    return t;
}

// --- seed vocab ----------------------------------------------------------
//
// Two-stage: (1) count substring occurrences in a hashmap, (2) score each
// by `count * length` (SP convention, longer-better tie-breaks) and keep
// the top `seed_size`. Single bytes are inserted unconditionally so
// byte_fallback always works.

const SubstrEntry = struct {
    bytes: []const u8, // borrowed from a backing arena
    count: u64,
};

fn lessByScoreDesc(_: void, a: SubstrEntry, b: SubstrEntry) bool {
    const sa = a.count *% a.bytes.len;
    const sb = b.count *% b.bytes.len;
    if (sa != sb) return sa > sb;
    if (a.bytes.len != b.bytes.len) return a.bytes.len > b.bytes.len;
    return std.mem.lessThan(u8, a.bytes, b.bytes);
}

fn seedVocab(
    allocator: std.mem.Allocator,
    corpus: Corpus,
    max_piece_length: u8,
    seed_size: usize,
    out: *Vocab,
    avoid: ?*const negative_train.AvoidList,
    avoid_mode: negative_train.Mode,
) !void {
    // 256 byte pieces first, score = 0 (will be replaced by log_prob after M-step).
    // Byte pieces are pinned (byte_fallback contract); the avoid list is
    // intentionally NOT consulted here — a single-byte avoid pattern would
    // otherwise make encoding impossible for that byte.
    var b: u32 = 0;
    while (b < 256) : (b += 1) {
        const buf = [_]u8{@intCast(b)};
        try out.add(&buf, 0.0, true);
    }

    // Count every substring up to max_piece_length, weighted by word count.
    var counts = std.StringHashMap(u64).init(allocator);
    defer counts.deinit();
    try counts.ensureTotalCapacity(@min(seed_size * 4, 1 << 20));

    for (corpus.words, corpus.counts) |w, freq| {
        if (freq == 0 or w.len < 2) continue;
        var i: usize = 0;
        while (i < w.len) : (i += 1) {
            const lim = @min(w.len, i + max_piece_length);
            var j: usize = i + 2; // skip length-1 (already covered by byte vocab)
            while (j <= lim) : (j += 1) {
                const sub = w[i..j];
                const gop = try counts.getOrPut(sub);
                if (!gop.found_existing) gop.value_ptr.* = 0;
                gop.value_ptr.* +%= freq;
            }
        }
    }

    // Collect, score, sort, take top `seed_size`.
    const n = counts.count();
    const entries = try allocator.alloc(SubstrEntry, n);
    defer allocator.free(entries);
    var it = counts.iterator();
    var idx: usize = 0;
    while (it.next()) |e| : (idx += 1) {
        entries[idx] = .{ .bytes = e.key_ptr.*, .count = e.value_ptr.* };
    }
    std.mem.sort(SubstrEntry, entries, {}, lessByScoreDesc);

    const keep = @min(entries.len, seed_size);
    // Score = log(count / total). Compute total over kept entries.
    var total: f64 = 0;
    var k: usize = 0;
    while (k < keep) : (k += 1) total += @floatFromInt(entries[k].count);
    if (total == 0) total = 1.0;
    const log_total: f64 = @log(total);

    k = 0;
    while (k < keep) : (k += 1) {
        var lp: f32 = @floatCast(@log(@as(f64, @floatFromInt(entries[k].count))) - log_total);
        if (avoid) |al| {
            if (al.matches(entries[k].bytes)) {
                switch (avoid_mode) {
                    .exclude => continue, // never seed
                    .penalize => lp += @floatFromInt(negative_train.PENALTY),
                }
            }
        }
        try out.add(entries[k].bytes, lp, false);
    }

    // Re-normalize byte scores to a uniform low log-prob — they get
    // updated by the first M-step anyway. Use min of kept scores.
    var min_lp: f32 = 0.0;
    if (keep > 0) min_lp = out.scores.items[256];
    for (out.scores.items[0..256]) |*s| s.* = min_lp;
}

// --- E-step: forward-backward over a word --------------------------------
//
// For each word we allocate `alpha[0..n+1]` and `beta[0..n+1]`, both
// initialised to NEG_INF except `alpha[0]=0`, `beta[n]=0`. Then:
//   forward: for each end-position `e` in [1..n], for each piece (s,e)
//     with `s<e`, `alpha[e] = logSumExp(alpha[e], alpha[s] + score[piece])`.
//   backward: symmetric.
//   posterior count of piece (s,e) = exp(alpha[s] + score + beta[e] - Z) * freq.
// `Z = alpha[n]`. If `alpha[n] == NEG_INF` the word is unreachable; we fall
// back to byte-level (always reachable since bytes are in vocab) so this
// branch only fires if the byte vocab was elided.
//
// EnumerateMatches: walk the trie from position `i`, recording (length, id)
// for each terminal hit. Limit length to `max_piece_length`.

const Match = struct {
    end: u32, // exclusive end byte offset
    id: u32,
    score: f32,
};

fn enumerateMatches(
    trie: *const Trie,
    vocab: *const Vocab,
    word: []const u8,
    start: u32,
    max_len: u8,
    out: *std.ArrayList(Match),
    allocator: std.mem.Allocator,
) !void {
    out.clearRetainingCapacity();
    var node: u32 = 0;
    var k: u32 = 0;
    const lim: u32 = @min(@as(u32, @intCast(word.len)) - start, max_len);
    while (k < lim) : (k += 1) {
        const next = trie.child(node, word[start + k]) orelse return;
        node = next;
        const id = trie.tokenAt(node);
        if (id != Trie.NO_TOK) {
            try out.append(allocator, .{
                .end = start + k + 1,
                .id = id,
                .score = vocab.scores.items[id],
            });
        }
    }
}

// Forward-backward one word into `expected` (length == vocab.count()).
// Returns log-likelihood Z (or NEG_INF if word is unreachable).
fn forwardBackwardWord(
    allocator: std.mem.Allocator,
    trie: *const Trie,
    vocab: *const Vocab,
    word: []const u8,
    freq: u32,
    max_piece_length: u8,
    expected: []f64,
) !f32 {
    const n: u32 = @intCast(word.len);
    if (n == 0) return 0.0;

    const alpha = try allocator.alloc(f32, n + 1);
    defer allocator.free(alpha);
    const beta = try allocator.alloc(f32, n + 1);
    defer allocator.free(beta);
    for (alpha) |*v| v.* = NEG_INF;
    for (beta) |*v| v.* = NEG_INF;
    alpha[0] = 0.0;
    beta[n] = 0.0;

    // We rebuild match lists position-by-position twice (forward then
    // backward). Cheap relative to logSumExp work, and skips the need to
    // cache (n × |matches|) tuples.
    var matches: std.ArrayList(Match) = .empty;
    defer matches.deinit(allocator);

    // Forward.
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        if (alpha[i] == NEG_INF) {
            // Unreachable position — force a byte UNK to keep DP alive.
            // Score doesn't really matter since posterior will be tiny.
            alpha[i + 1] = logSumExp(alpha[i + 1], alpha[i] + UNK_SCORE);
            continue;
        }
        try enumerateMatches(trie, vocab, word, i, max_piece_length, &matches, allocator);
        if (matches.items.len == 0) {
            // Byte fallback should prevent this; if it fires, treat as UNK.
            alpha[i + 1] = logSumExp(alpha[i + 1], alpha[i] + UNK_SCORE);
            continue;
        }
        for (matches.items) |m| {
            const cand = alpha[i] + m.score;
            alpha[m.end] = logSumExp(alpha[m.end], cand);
        }
    }

    const Z = alpha[n];
    if (Z == NEG_INF) return NEG_INF;

    // Backward.
    var ii: i32 = @intCast(n);
    while (ii > 0) {
        ii -= 1;
        const s: u32 = @intCast(ii);
        if (beta[s + 1] == NEG_INF and s + 1 != n) {
            // Won't contribute to Z, but propagate to keep paths alive.
        }
        try enumerateMatches(trie, vocab, word, s, max_piece_length, &matches, allocator);
        for (matches.items) |m| {
            const cand = m.score + beta[m.end];
            beta[s] = logSumExp(beta[s], cand);
        }
    }

    // Posterior counts: for each piece (s,e), contribution =
    // exp(alpha[s] + score + beta[e] - Z) * freq.
    const freq_f: f64 = @floatFromInt(freq);
    i = 0;
    while (i < n) : (i += 1) {
        if (alpha[i] == NEG_INF) continue;
        try enumerateMatches(trie, vocab, word, i, max_piece_length, &matches, allocator);
        for (matches.items) |m| {
            if (beta[m.end] == NEG_INF) continue;
            const x: f32 = alpha[i] + m.score + beta[m.end] - Z;
            // Guard against tiny numerical positives above 0 (would imply
            // posterior > 1). Cap at 0.
            const xc = if (x > 0.0) 0.0 else x;
            const p: f64 = @exp(@as(f64, xc));
            expected[m.id] += freq_f * p;
        }
    }

    return Z;
}

// --- Monte-Carlo posterior via forward-filter / backward-sample ----------
//
// Replaces forwardBackwardWord when subword_reg_samples > 0. For each
// sample we run the same forward pass (giving an estimate of log-likelihood
// Z = alpha[n]) and then draw one segmentation via Gumbel-max over pieces
// that end at each position. The empirical count of piece p in N samples,
// divided by N, is an unbiased estimator of its analytic posterior — and as
// alpha → 1 the expectation converges to the same exp(alpha+score+beta-Z).
// Z is computed once (forward pass only) and reused across all N samples.

const SampleMatch = struct {
    start: u32,
    end: u32,
    id: u32,
    score: f32,
};

fn forwardSampleWord(
    allocator: std.mem.Allocator,
    trie: *const Trie,
    vocab: *const Vocab,
    word: []const u8,
    freq: u32,
    max_piece_length: u8,
    samples: u32,
    alpha_temp: f32,
    rng: std.Random,
    expected: []f64,
) !f32 {
    const n: u32 = @intCast(word.len);
    if (n == 0) return 0.0;

    const fwd = try allocator.alloc(f32, n + 1);
    defer allocator.free(fwd);
    for (fwd) |*v| v.* = NEG_INF;
    fwd[0] = 0.0;

    // Collect every match seen during forward pass. Single ArrayList that
    // we later index by end position via counting sort.
    var matches: std.ArrayList(SampleMatch) = .empty;
    defer matches.deinit(allocator);

    var step: std.ArrayList(Match) = .empty;
    defer step.deinit(allocator);

    var i: u32 = 0;
    while (i < n) : (i += 1) {
        if (fwd[i] == NEG_INF) {
            // Force-advance via UNK. Use the first-byte token as the
            // forced piece so backward sampling has something to pick.
            const uid: u32 = @intCast(word[i]); // byte == its byte-piece id
            const sc: f32 = if (vocab.count() > uid) vocab.scores.items[uid] else UNK_SCORE;
            fwd[i + 1] = logSumExp(fwd[i + 1], fwd[i] + sc);
            try matches.append(allocator, .{
                .start = i,
                .end = i + 1,
                .id = uid,
                .score = sc,
            });
            continue;
        }
        try enumerateMatches(trie, vocab, word, i, max_piece_length, &step, allocator);
        if (step.items.len == 0) {
            const uid: u32 = @intCast(word[i]);
            const sc: f32 = if (vocab.count() > uid) vocab.scores.items[uid] else UNK_SCORE;
            fwd[i + 1] = logSumExp(fwd[i + 1], fwd[i] + sc);
            try matches.append(allocator, .{
                .start = i,
                .end = i + 1,
                .id = uid,
                .score = sc,
            });
            continue;
        }
        for (step.items) |m| {
            const cand = fwd[i] + m.score;
            fwd[m.end] = logSumExp(fwd[m.end], cand);
            try matches.append(allocator, .{
                .start = i,
                .end = m.end,
                .id = m.id,
                .score = m.score,
            });
        }
    }

    const Z = fwd[n];
    if (Z == NEG_INF) return NEG_INF;

    // Counting-sort matches by end position so the backward sampler can do
    // O(matches_per_pos) lookups instead of a full scan. After the prefix
    // sum, matches with end == p live in `by_end[ends_off[p]..ends_off[p+1]]`.
    const ends_off = try allocator.alloc(u32, n + 2);
    defer allocator.free(ends_off);
    @memset(ends_off, 0);
    for (matches.items) |m| ends_off[m.end] += 1;
    var acc: u32 = 0;
    for (ends_off) |*c| {
        const v = c.*;
        c.* = acc;
        acc += v;
    }
    const by_end = try allocator.alloc(SampleMatch, matches.items.len);
    defer allocator.free(by_end);
    const cursor = try allocator.alloc(u32, n + 2);
    defer allocator.free(cursor);
    @memcpy(cursor, ends_off);
    for (matches.items) |m| {
        const idx = cursor[m.end];
        by_end[idx] = m;
        cursor[m.end] = idx + 1;
    }

    const freq_f: f64 = @floatFromInt(freq);
    const samples_f: f64 = @floatFromInt(samples);
    const weight: f64 = freq_f / samples_f;
    const inv_alpha: f32 = if (alpha_temp > 0.0) 1.0 / alpha_temp else 1e9;

    var s_idx: u32 = 0;
    while (s_idx < samples) : (s_idx += 1) {
        var pos: u32 = n;
        while (pos > 0) {
            const lo = ends_off[pos];
            const hi = ends_off[pos + 1];
            std.debug.assert(hi > lo);
            var best_g: f32 = NEG_INF;
            var best_idx: u32 = lo;
            var j: u32 = lo;
            while (j < hi) : (j += 1) {
                const m = by_end[j];
                const sp = fwd[m.start];
                if (sp == NEG_INF) continue;
                const lw = (sp + m.score) * inv_alpha;
                const g = lw + gumbel(rng);
                if (g > best_g) {
                    best_g = g;
                    best_idx = j;
                }
            }
            const chosen = by_end[best_idx];
            expected[chosen.id] += weight;
            pos = chosen.start;
        }
    }

    return Z;
}

// --- E-step driver (single + multithreaded) ------------------------------

const EStepResult = struct {
    expected: []f64,
    total_loglik: f64,
};

const EStepCtx = struct {
    trie: *const Trie,
    vocab: *const Vocab,
    corpus: Corpus,
    max_piece_length: u8,
    pool: *BatchPool,
    per_worker_expected: [][]f64,
    per_worker_loglik: []f64,
    // Sampling config. samples == 0 → analytic forward-backward.
    samples: u32,
    alpha_temp: f32,
    // Per-worker RNG state to avoid contention. Each worker has its own
    // DefaultPrng seeded deterministically from `subword_reg_seed` ⊕ idx.
    per_worker_rng: []std.Random.DefaultPrng,
};

const EStepWorker = struct {
    pub fn run(ctx: *EStepCtx, word_idx: usize, worker_idx: usize) void {
        const word = ctx.corpus.words[word_idx];
        const freq = ctx.corpus.counts[word_idx];
        if (freq == 0 or word.len == 0) return;
        const scratch = ctx.pool.resetArena(worker_idx);
        // Note: resetArena() resets the WHOLE arena. That's fine because
        // the BatchPool guarantees each worker_idx is owned by one thread
        // for the duration of the batch — but we still need a stable
        // allocator inside this call. We allocate alpha/beta on `scratch`,
        // which is the arena. Multiple words per worker would each re-reset.
        // To avoid losing memory across words on the same worker, we
        // allocate manually below and defer-free.
        _ = scratch;
        // Use the underlying allocator directly to make per-word
        // alloc/free symmetric (arena reset would otherwise reset state
        // mid-batch when this fn is called again for the same worker).
        const alloc = ctx.pool.allocator;
        if (ctx.samples == 0) {
            const z = forwardBackwardWord(
                alloc,
                ctx.trie,
                ctx.vocab,
                word,
                freq,
                ctx.max_piece_length,
                ctx.per_worker_expected[worker_idx],
            ) catch return;
            if (z != NEG_INF) {
                ctx.per_worker_loglik[worker_idx] += @as(f64, z) * @as(f64, @floatFromInt(freq));
            }
        } else {
            const rng = ctx.per_worker_rng[worker_idx].random();
            const z = forwardSampleWord(
                alloc,
                ctx.trie,
                ctx.vocab,
                word,
                freq,
                ctx.max_piece_length,
                ctx.samples,
                ctx.alpha_temp,
                rng,
                ctx.per_worker_expected[worker_idx],
            ) catch return;
            if (z != NEG_INF) {
                ctx.per_worker_loglik[worker_idx] += @as(f64, z) * @as(f64, @floatFromInt(freq));
            }
        }
    }
};

fn runEStep(
    allocator: std.mem.Allocator,
    trie: *const Trie,
    vocab: *const Vocab,
    corpus: Corpus,
    max_piece_length: u8,
    pool: ?*BatchPool,
    samples: u32,
    alpha_temp: f32,
    rng_seed: u64,
    iter_idx: u32,
) !EStepResult {
    const v = vocab.count();
    const expected = try allocator.alloc(f64, v);
    @memset(expected, 0.0);
    var total_loglik: f64 = 0.0;

    if (pool) |bp| {
        const nw = bp.workerCount();
        const per_w_expected = try allocator.alloc([]f64, nw);
        defer {
            for (per_w_expected) |buf| allocator.free(buf);
            allocator.free(per_w_expected);
        }
        for (per_w_expected) |*buf| {
            buf.* = try allocator.alloc(f64, v);
            @memset(buf.*, 0.0);
        }
        const per_w_loglik = try allocator.alloc(f64, nw);
        defer allocator.free(per_w_loglik);
        @memset(per_w_loglik, 0.0);

        // Seed per-worker PRNGs deterministically. We mix in iter_idx so
        // successive E-step iterations are not identical samples.
        const per_w_rng = try allocator.alloc(std.Random.DefaultPrng, nw);
        defer allocator.free(per_w_rng);
        var w: usize = 0;
        while (w < nw) : (w += 1) {
            const seed = rng_seed ^ (@as(u64, w) *% 0x9E37_79B9_7F4A_7C15) ^
                (@as(u64, iter_idx) *% 0xBF58_476D_1CE4_E5B9);
            per_w_rng[w] = std.Random.DefaultPrng.init(seed);
        }

        var ctx: EStepCtx = .{
            .trie = trie,
            .vocab = vocab,
            .corpus = corpus,
            .max_piece_length = max_piece_length,
            .pool = bp,
            .per_worker_expected = per_w_expected,
            .per_worker_loglik = per_w_loglik,
            .samples = samples,
            .alpha_temp = alpha_temp,
            .per_worker_rng = per_w_rng,
        };
        try bp.runBatch(EStepWorker, &ctx, corpus.words.len);

        for (per_w_expected) |buf| {
            for (expected, buf) |*acc, x| acc.* += x;
        }
        for (per_w_loglik) |x| total_loglik += x;
    } else {
        var prng = std.Random.DefaultPrng.init(rng_seed ^
            (@as(u64, iter_idx) *% 0xBF58_476D_1CE4_E5B9));
        const rng = prng.random();
        for (corpus.words, corpus.counts) |word, freq| {
            if (freq == 0 or word.len == 0) continue;
            const z = if (samples == 0)
                try forwardBackwardWord(
                    allocator,
                    trie,
                    vocab,
                    word,
                    freq,
                    max_piece_length,
                    expected,
                )
            else
                try forwardSampleWord(
                    allocator,
                    trie,
                    vocab,
                    word,
                    freq,
                    max_piece_length,
                    samples,
                    alpha_temp,
                    rng,
                    expected,
                );
            if (z != NEG_INF) {
                total_loglik += @as(f64, z) * @as(f64, @floatFromInt(freq));
            }
        }
    }

    return .{ .expected = expected, .total_loglik = total_loglik };
}

// --- M-step: renormalize -------------------------------------------------
//
// Drop pieces whose expected count is below MIN_EXPECTED (except pinned
// bytes). Renormalize the survivors into new log-probs. Rewrites vocab in
// place by building a fresh Vocab and swapping.

fn runMStep(
    allocator: std.mem.Allocator,
    vocab: *Vocab,
    expected: []const f64,
) !void {
    var new_vocab = Vocab.init(allocator);
    errdefer new_vocab.deinit();

    var sum: f64 = 0.0;
    const v = vocab.count();
    var id: u32 = 0;
    while (id < v) : (id += 1) {
        if (vocab.is_byte.items[id]) {
            // Pinned: counted toward sum with a floor so byte log_prob
            // doesn't collapse to -inf when a byte never appears.
            const c = @max(expected[id], MIN_EXPECTED);
            sum += c;
        } else if (expected[id] >= MIN_EXPECTED) {
            sum += expected[id];
        }
    }
    if (sum <= 0.0) sum = 1.0;
    const log_sum: f64 = @log(sum);

    id = 0;
    while (id < v) : (id += 1) {
        if (vocab.is_byte.items[id]) {
            const c = @max(expected[id], MIN_EXPECTED);
            const lp: f32 = @floatCast(@log(c) - log_sum);
            try new_vocab.add(vocab.piece(id), lp, true);
        } else if (expected[id] >= MIN_EXPECTED) {
            const lp: f32 = @floatCast(@log(expected[id]) - log_sum);
            try new_vocab.add(vocab.piece(id), lp, false);
        }
    }

    vocab.deinit();
    vocab.* = new_vocab;
}

// --- pruning -------------------------------------------------------------
//
// Naive: sort non-byte pieces by score ascending, drop the bottom
// `prune_fraction * non_byte_count`. Never below `target_size`. Always
// keep all 256 bytes.

const ScoredId = struct { id: u32, score: f32 };

fn lessByScoreAsc(_: void, a: ScoredId, b: ScoredId) bool {
    if (a.score != b.score) return a.score < b.score;
    return a.id < b.id;
}

fn prune(
    allocator: std.mem.Allocator,
    vocab: *Vocab,
    prune_fraction: f32,
    target_size: u32,
) !void {
    const v = vocab.count();
    if (v <= target_size) return;

    // Collect non-byte pieces sorted by score ascending.
    var non_byte: std.ArrayList(ScoredId) = .empty;
    defer non_byte.deinit(allocator);
    var id: u32 = 0;
    while (id < v) : (id += 1) {
        if (!vocab.is_byte.items[id]) {
            try non_byte.append(allocator, .{ .id = id, .score = vocab.scores.items[id] });
        }
    }
    std.mem.sort(ScoredId, non_byte.items, {}, lessByScoreAsc);

    // How many to drop?
    const drop_target: u32 = @intFromFloat(@as(f32, @floatFromInt(non_byte.items.len)) * prune_fraction);
    // Don't go below `target_size` total.
    const byte_count: u32 = v - @as(u32, @intCast(non_byte.items.len));
    const max_drop: u32 = if (v > target_size) v - target_size else 0;
    const to_drop: u32 = @min(drop_target, max_drop);
    if (to_drop == 0) return;
    _ = byte_count;

    // Build a kill set.
    var kill = try allocator.alloc(bool, v);
    defer allocator.free(kill);
    @memset(kill, false);
    var k: u32 = 0;
    while (k < to_drop) : (k += 1) kill[non_byte.items[k].id] = true;

    // Rewrite the vocab keeping survivors.
    var new_vocab = Vocab.init(allocator);
    errdefer new_vocab.deinit();
    id = 0;
    while (id < v) : (id += 1) {
        if (kill[id]) continue;
        try new_vocab.add(vocab.piece(id), vocab.scores.items[id], vocab.is_byte.items[id]);
    }

    vocab.deinit();
    vocab.* = new_vocab;
}

// --- final shrink to exactly `vocab_size` --------------------------------
//
// After EM converges we might be slightly above the target. Greedy: drop
// lowest-score non-byte pieces until we hit the target.

fn finalShrink(
    allocator: std.mem.Allocator,
    vocab: *Vocab,
    target: u32,
) !void {
    if (vocab.count() <= target) return;
    var non_byte: std.ArrayList(ScoredId) = .empty;
    defer non_byte.deinit(allocator);
    var id: u32 = 0;
    while (id < vocab.count()) : (id += 1) {
        if (!vocab.is_byte.items[id]) {
            try non_byte.append(allocator, .{ .id = id, .score = vocab.scores.items[id] });
        }
    }
    std.mem.sort(ScoredId, non_byte.items, {}, lessByScoreAsc);

    const need_drop = vocab.count() - target;
    if (need_drop > non_byte.items.len) return; // can't shrink past bytes

    var kill = try allocator.alloc(bool, vocab.count());
    defer allocator.free(kill);
    @memset(kill, false);
    var k: u32 = 0;
    while (k < need_drop) : (k += 1) kill[non_byte.items[k].id] = true;

    var new_vocab = Vocab.init(allocator);
    errdefer new_vocab.deinit();
    id = 0;
    while (id < vocab.count()) : (id += 1) {
        if (kill[id]) continue;
        try new_vocab.add(vocab.piece(id), vocab.scores.items[id], vocab.is_byte.items[id]);
    }
    vocab.deinit();
    vocab.* = new_vocab;
}

// --- public entrypoint ---------------------------------------------------

pub fn train(allocator: std.mem.Allocator, corpus: Corpus, opts: TrainOptions) !Unigram {
    if (corpus.words.len != corpus.counts.len) return error.CorpusLengthMismatch;
    if (opts.vocab_size < 257) return error.VocabTooSmall; // 256 bytes + unk

    var vocab = Vocab.init(allocator);
    defer vocab.deinit();

    try seedVocab(allocator, corpus, opts.max_piece_length, opts.seed_size, &vocab, opts.avoid, opts.avoid_mode);

    // Ensure we have at least a target+1 to allow an UNK slot.
    const em_target = opts.vocab_size; // we'll insert UNK at finalize time

    // Round loop: EM + prune until at-or-near target.
    var round: u32 = 0;
    const max_rounds: u32 = 32;
    var iter_counter: u32 = 0;
    while (round < max_rounds) : (round += 1) {
        var sub: u32 = 0;
        var avg_loglik: f32 = 0.0;
        while (sub < opts.em_iters_per_round) : (sub += 1) {
            var trie = try buildTrie(allocator, &vocab);
            defer trie.deinit();

            const e = try runEStep(
                allocator,
                &trie,
                &vocab,
                corpus,
                opts.max_piece_length,
                opts.pool,
                opts.subword_reg_samples,
                opts.subword_reg_alpha,
                opts.subword_reg_seed,
                iter_counter,
            );
            defer allocator.free(e.expected);
            iter_counter += 1;

            var total_freq: f64 = 0;
            for (corpus.counts) |c| total_freq += @floatFromInt(c);
            if (total_freq == 0) total_freq = 1.0;
            avg_loglik = @floatCast(e.total_loglik / total_freq);

            try runMStep(allocator, &vocab, e.expected);
        }
        if (opts.on_round) |cb| cb(opts.on_round_ctx, vocab.count(), avg_loglik);

        if (vocab.count() <= em_target) break;
        try prune(allocator, &vocab, opts.prune_fraction, em_target);
        if (vocab.count() <= em_target) break;
    }

    try finalShrink(allocator, &vocab, em_target);

    // Convert into Unigram via Builder. Add a sentinel UNK piece at the end
    // if the vocab doesn't already have one named "<unk>"; otherwise the
    // first byte piece (byte 0) acts as the UNK fallback.
    //
    // Convention: pick byte 0 as the UNK id. It's always present (byte
    // pieces are pinned), it's a single byte so encode never blows up, and
    // it matches the byte_fallback contract.
    var builder = Unigram.Builder.init(allocator);
    defer builder.deinit();
    var unk_id: TokenId = 0;
    var id: u32 = 0;
    const v = vocab.count();
    while (id < v) : (id += 1) {
        const new_id = try builder.addToken(vocab.piece(id), vocab.scores.items[id]);
        if (id == 0) unk_id = new_id; // byte 0
    }
    return try builder.finalize(unk_id);
}

// --- tests ---------------------------------------------------------------

const testing = std.testing;

test "train converges on a trivial corpus" {
    const allocator = testing.allocator;
    const words = [_][]const u8{"ababab"};
    const counts = [_]u32{100};

    var u = try train(allocator, .{
        .words = &words,
        .counts = &counts,
    }, .{
        .vocab_size = 270,
        .max_piece_length = 6,
        .em_iters_per_round = 2,
    });
    defer u.deinit();

    // After training, "ababab" should encode into ≤3 pieces.
    var out: [16]TokenId = undefined;
    const ids = try u.encodeChunk(allocator, "ababab", &out);
    try testing.expect(ids.len <= 3);
}

test "train reaches target vocab_size" {
    const allocator = testing.allocator;

    // Use an arena for word storage.
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    var words_storage: [100][]const u8 = undefined;
    var counts_storage: [100]u32 = undefined;
    var prng = std.Random.DefaultPrng.init(0xDEADBEEF);
    const rnd = prng.random();
    var w: usize = 0;
    while (w < 100) : (w += 1) {
        const len = rnd.intRangeAtMost(usize, 3, 10);
        const buf = try aa.alloc(u8, len);
        for (buf) |*c| c.* = 'a' + rnd.intRangeAtMost(u8, 0, 7);
        words_storage[w] = buf;
        counts_storage[w] = rnd.intRangeAtMost(u32, 1, 5);
    }

    var u = try train(allocator, .{
        .words = &words_storage,
        .counts = &counts_storage,
    }, .{
        .vocab_size = 300,
        .max_piece_length = 8,
        .em_iters_per_round = 1,
    });
    defer u.deinit();

    try testing.expectEqual(@as(u32, 300), u.count);
}

test "train produces a valid Unigram (idBytes works)" {
    const allocator = testing.allocator;
    const words = [_][]const u8{ "hello", "world", "help", "wood" };
    const counts = [_]u32{ 4, 3, 2, 1 };

    var u = try train(allocator, .{
        .words = &words,
        .counts = &counts,
    }, .{
        .vocab_size = 280,
        .max_piece_length = 6,
        .em_iters_per_round = 1,
    });
    defer u.deinit();

    try testing.expect(u.count >= 256);
    var id: u32 = 0;
    while (id < u.count) : (id += 1) {
        const b = u.idBytes(id);
        try testing.expect(b.len > 0);
    }
}

test "train UNK is set" {
    const allocator = testing.allocator;
    const words = [_][]const u8{"abcabc"};
    const counts = [_]u32{10};

    var u = try train(allocator, .{
        .words = &words,
        .counts = &counts,
    }, .{
        .vocab_size = 260,
        .max_piece_length = 4,
        .em_iters_per_round = 1,
    });
    defer u.deinit();

    // unk_id is a TokenId — it must be valid (< count) and resolvable.
    try testing.expect(u.unk_id < u.count);
    const unk_bytes = u.idBytes(u.unk_id);
    try testing.expect(unk_bytes.len == 1); // single byte
}

test "multithreaded matches single-threaded" {
    const allocator = testing.allocator;
    const words = [_][]const u8{ "abracadabra", "cadaver", "abacus", "barbara", "rhubarb" };
    const counts = [_]u32{ 5, 3, 2, 4, 1 };

    var u_serial = try train(allocator, .{
        .words = &words,
        .counts = &counts,
    }, .{
        .vocab_size = 280,
        .max_piece_length = 6,
        .em_iters_per_round = 1,
    });
    defer u_serial.deinit();

    var pool = try BatchPool.init(allocator, 2);
    defer pool.deinit();

    var u_par = try train(allocator, .{
        .words = &words,
        .counts = &counts,
    }, .{
        .vocab_size = 280,
        .max_piece_length = 6,
        .em_iters_per_round = 1,
        .pool = &pool,
    });
    defer u_par.deinit();

    // Allow tiny float divergence — compare token counts on a sample.
    try testing.expectEqual(u_serial.count, u_par.count);

    var out_s: [32]TokenId = undefined;
    var out_p: [32]TokenId = undefined;
    const ids_s = try u_serial.encodeChunk(allocator, "abracadabra", &out_s);
    const ids_p = try u_par.encodeChunk(allocator, "abracadabra", &out_p);
    // Counts shouldn't diverge wildly. Allow ±1 to absorb float jitter.
    const diff: i32 = @as(i32, @intCast(ids_s.len)) - @as(i32, @intCast(ids_p.len));
    try testing.expect(@abs(diff) <= 1);
}

test "train with subword_reg_samples=5 still converges to target vocab size" {
    const allocator = testing.allocator;

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    var words_storage: [100][]const u8 = undefined;
    var counts_storage: [100]u32 = undefined;
    var prng = std.Random.DefaultPrng.init(0xDEADBEEF);
    const rnd = prng.random();
    var w: usize = 0;
    while (w < 100) : (w += 1) {
        const len = rnd.intRangeAtMost(usize, 3, 10);
        const buf = try aa.alloc(u8, len);
        for (buf) |*c| c.* = 'a' + rnd.intRangeAtMost(u8, 0, 7);
        words_storage[w] = buf;
        counts_storage[w] = rnd.intRangeAtMost(u32, 1, 5);
    }

    var u = try train(allocator, .{
        .words = &words_storage,
        .counts = &counts_storage,
    }, .{
        .vocab_size = 300,
        .max_piece_length = 8,
        .em_iters_per_round = 1,
        .subword_reg_samples = 5,
        .subword_reg_alpha = 1.0,
        .subword_reg_seed = 0xC0FFEE,
    });
    defer u.deinit();

    try testing.expectEqual(@as(u32, 300), u.count);
}

test "train with subword_reg_samples=0 matches existing behavior" {
    const allocator = testing.allocator;
    const words = [_][]const u8{ "abracadabra", "cadaver", "abacus", "barbara", "rhubarb" };
    const counts = [_]u32{ 5, 3, 2, 4, 1 };

    var u_default = try train(allocator, .{
        .words = &words,
        .counts = &counts,
    }, .{
        .vocab_size = 280,
        .max_piece_length = 6,
        .em_iters_per_round = 1,
    });
    defer u_default.deinit();

    var u_explicit = try train(allocator, .{
        .words = &words,
        .counts = &counts,
    }, .{
        .vocab_size = 280,
        .max_piece_length = 6,
        .em_iters_per_round = 1,
        .subword_reg_samples = 0,
        .subword_reg_alpha = 1.0,
        .subword_reg_seed = 0,
    });
    defer u_explicit.deinit();

    try testing.expectEqual(u_default.count, u_explicit.count);
    // Bit-identical scores and bytes.
    try testing.expectEqualSlices(f32, u_default.scores, u_explicit.scores);
    try testing.expectEqualSlices(u8, u_default.bytes, u_explicit.bytes);
    try testing.expectEqualSlices(u32, u_default.offsets, u_explicit.offsets);
}
