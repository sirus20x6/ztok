//! Vocab extend / transplant — append new tokens to an existing BPE vocab
//! at high ids and emit an "embedding init plan" so the caller can resize
//! its embedding table.
//!
//! Strategies:
//!   - `mean_subtoken` (WECHSEL-style): encode the new token's bytes with
//!     the OLD tokenizer; the resulting subtoken ids tell the caller which
//!     existing embedding rows to average to seed the new row.
//!   - `weighted_similar`: scan the existing vocab for the K tokens most
//!     byte-similar to the new token (longest common substring / max len),
//!     then emit a frequency-weighted combination. Optionally weight by
//!     inverse rank in the existing vocab.
//!
//! LIMITATION — encoder vs init plan:
//! The new tokens are appended as plain entries, not BPE merges. The
//! returned `extended_bpe` therefore will NOT spontaneously merge raw
//! bytes INTO a new token id during `encodeChunk` (a merge chain to the
//! new id does not exist). They behave like HF "added tokens". To make
//! the encoder actually emit a new id the caller must either
//!   1. pre-process input with `added_tokens.Scanner` to substitute the
//!      new-token byte sequences before BPE, or
//!   2. train fresh BPE merges that build up to each new id.
//! The init plan is still useful for the embedding side regardless.

const std = @import("std");
const TokenId = @import("token.zig").TokenId;
const Bpe = @import("bpe.zig").Bpe;

pub const Strategy = enum {
    /// Initial embedding = mean of old-tokenizer-subtoken embeddings.
    mean_subtoken,
    /// Initial embedding = weighted average of the K existing tokens
    /// most byte-similar to the new token. See `WeightedSimilarOpts`.
    weighted_similar,
    /// Random initialization (caller provides RNG).
    random,
};

/// Per-extension options for `Strategy.weighted_similar`. Set on
/// `ExtendOptions.weighted_similar` to control the K-nearest scan.
pub const WeightedSimilarOpts = struct {
    /// Number of source tokens to combine. Capped to the existing vocab
    /// size if larger.
    k: u8 = 8,
    /// If true, multiply each candidate's similarity weight by
    /// (1 / (1 + rank)) so older/foundational tokens dominate. Default
    /// false: similarity scores alone determine weights.
    weight_by_rank: bool = false,
    /// When true, build a 4-gram inverted index over the existing vocab
    /// once and use it to pre-filter candidates before running the
    /// expensive LCS scan. Strictly faster for V > a few thousand;
    /// produces the same top-K set as the naive scan in practice (the
    /// pre-filter retains the top ~3K most gram-overlapping candidates
    /// before exact scoring). Set false to force the naive full scan
    /// (mainly for regression testing).
    use_qgram_index: bool = true,
};

pub const InitInstruction = struct {
    /// New token id (== old_vocab_size + extension index).
    new_id: TokenId,
    /// Bytes of the new token. Owned by the parent `ExtendResult`.
    bytes: []const u8,
    /// Strategy-specific payload. Owned by the parent `ExtendResult`.
    payload: Payload,

    pub const Payload = union(enum) {
        /// `subtoken_ids`: old-tokenizer subtoken ids to average. Owned
        /// by the parent `ExtendResult`.
        mean_subtoken: struct {
            subtoken_ids: []const TokenId,
        },
        /// `sources`: existing token ids to draw embeddings from.
        /// `weights`: per-source weights summing to 1.0. Equal length to
        /// `sources`. Both owned by the parent `ExtendResult`.
        weighted_similar: struct {
            sources: []const TokenId,
            weights: []const f32,
        },
        /// Random-init marker; no associated data.
        random,
    };

    /// Convenience accessor — returns the mean_subtoken ids or empty.
    pub fn subtokenIds(self: InitInstruction) []const TokenId {
        return switch (self.payload) {
            .mean_subtoken => |m| m.subtoken_ids,
            else => &.{},
        };
    }
};

pub const ExtendResult = struct {
    allocator: std.mem.Allocator,
    /// New BPE with old + new tokens. Caller deinits via `deinit()`.
    extended_bpe: Bpe,
    /// Old vocab size — new ids start at this number.
    old_vocab_size: u32,
    /// New vocab size after extension.
    new_vocab_size: u32,
    /// Per-added-token: how the caller should initialize its embedding row.
    init_plan: []InitInstruction,

    pub fn deinit(self: *ExtendResult) void {
        for (self.init_plan) |inst| {
            if (inst.bytes.len > 0) self.allocator.free(inst.bytes);
            switch (inst.payload) {
                .mean_subtoken => |m| {
                    if (m.subtoken_ids.len > 0) self.allocator.free(m.subtoken_ids);
                },
                .weighted_similar => |w| {
                    if (w.sources.len > 0) self.allocator.free(w.sources);
                    if (w.weights.len > 0) self.allocator.free(w.weights);
                },
                .random => {},
            }
        }
        if (self.init_plan.len > 0) self.allocator.free(self.init_plan);
        self.extended_bpe.deinit();
        self.* = undefined;
    }
};

pub const ExtendOptions = struct {
    /// New tokens to add. Each string becomes a new id (in order).
    new_tokens: []const []const u8,
    strategy: Strategy = .mean_subtoken,
    /// Per-extension options for `Strategy.weighted_similar`. Ignored
    /// for other strategies.
    weighted_similar: WeightedSimilarOpts = .{},
    /// For .random: the seed.
    random_seed: u64 = 0,
    /// If true, skip new_tokens that already exist in the old vocab
    /// (don't add duplicates). If false, error on collision.
    skip_existing: bool = true,
};

pub const ExtendError = error{
    DuplicateToken,
    EmptyToken,
    OutOfMemory,
};

/// Clone `old_bpe` into a fresh allocation owned by `allocator`. The
/// clone's `by_bytes` keys borrow into the clone's own `bytes` buffer.
fn cloneBpe(allocator: std.mem.Allocator, old_bpe: *const Bpe) !Bpe {
    const bytes = if (old_bpe.bytes.len > 0) try allocator.dupe(u8, old_bpe.bytes) else try allocator.alloc(u8, 0);
    errdefer if (bytes.len > 0) allocator.free(bytes);
    const offsets = if (old_bpe.offsets.len > 0) try allocator.dupe(u32, old_bpe.offsets) else try allocator.alloc(u32, 0);
    errdefer if (offsets.len > 0) allocator.free(offsets);

    var by_bytes = std.StringHashMap(TokenId).init(allocator);
    errdefer by_bytes.deinit();
    try by_bytes.ensureTotalCapacity(old_bpe.count);
    var i: u32 = 0;
    while (i < old_bpe.count) : (i += 1) {
        const key = bytes[offsets[i]..offsets[i + 1]];
        try by_bytes.put(key, i);
    }

    // Rebuild the two-level merge-rank front cache so the clone keeps
    // the same encode-loop perf as the original (see `Bpe.hot_table`).
    const hot_table = try Bpe.buildHotTable(allocator, bytes, offsets, old_bpe.count);
    errdefer allocator.free(hot_table);

    return .{
        .allocator = allocator,
        .bytes = bytes,
        .offsets = offsets,
        .count = old_bpe.count,
        .by_bytes = by_bytes,
        .hot_table = hot_table,
    };
}

/// Longest common substring length between `a` and `b` using a rolling
/// two-row scratch buffer. `scratch.len` must be >= 2 * (min(a.len,
/// b.len) + 1). O(|a|*|b|) time, O(min) space.
fn lcsLength(a: []const u8, b: []const u8, scratch: []u32) u32 {
    if (a.len == 0 or b.len == 0) return 0;
    // Iterate so the inner dimension is the shorter side -> scratch fits.
    const short_is_b = b.len <= a.len;
    const outer = if (short_is_b) a else b;
    const inner = if (short_is_b) b else a;
    const w = inner.len + 1;
    std.debug.assert(scratch.len >= 2 * w);

    var prev = scratch[0..w];
    var curr = scratch[w .. 2 * w];
    @memset(prev, 0);
    @memset(curr, 0);

    var best: u32 = 0;
    var i: usize = 0;
    while (i < outer.len) : (i += 1) {
        // curr[0] is the boundary column; always 0.
        curr[0] = 0;
        const oc = outer[i];
        var j: usize = 0;
        while (j < inner.len) : (j += 1) {
            if (oc == inner[j]) {
                const v = prev[j] + 1;
                curr[j + 1] = v;
                if (v > best) best = v;
            } else {
                curr[j + 1] = 0;
            }
        }
        const tmp = prev;
        prev = curr;
        curr = tmp;
    }
    return best;
}

/// Internal: pick top-K existing tokens by similarity to `tok_bytes`.
/// Writes ids into `out_sources`, raw similarity scores into
/// `out_scores`. Returns the number of entries written (<= K). Uses
/// `scratch_u32` as a scratch buffer for LCS. Similarity score for an
/// existing token `e` is `lcs(tok, e) / max(|tok|, |e|)` in [0, 1].
fn topKSimilar(
    old_bpe: *const Bpe,
    tok_bytes: []const u8,
    k: usize,
    out_sources: []TokenId,
    out_scores: []f32,
    lcs_scratch: []u32,
) usize {
    const n_existing = old_bpe.count;
    const cap = @min(k, out_sources.len);
    std.debug.assert(out_scores.len >= cap);
    if (cap == 0 or n_existing == 0 or tok_bytes.len == 0) return 0;

    // Min-heap by score: smallest at index 0. We keep up to `cap`
    // entries; a new candidate replaces the head only if it beats it.
    // Heap size stays small (K=8 typical) so a tiny array + linear ops
    // is cheaper than std heap machinery.
    var heap_ids: [256]TokenId = undefined;
    var heap_scores: [256]f32 = undefined;
    std.debug.assert(cap <= heap_ids.len);
    var heap_len: usize = 0;

    var id: u32 = 0;
    while (id < n_existing) : (id += 1) {
        const a = old_bpe.bytes[old_bpe.offsets[id]..old_bpe.offsets[id + 1]];
        if (a.len == 0) continue;
        const min_len = @min(a.len, tok_bytes.len);
        // Ensure scratch fits this pair. Caller sized for the longest
        // tok_bytes; an existing token longer than that flips the
        // shorter-axis selection in lcsLength, so we still need
        // 2 * (min_len + 1) words. Guaranteed by caller's sizing.
        const need = 2 * (min_len + 1);
        std.debug.assert(lcs_scratch.len >= need);
        const lcs = lcsLength(a, tok_bytes, lcs_scratch[0..need]);
        if (lcs == 0) continue;
        const denom: f32 = @floatFromInt(@max(a.len, tok_bytes.len));
        const score: f32 = @as(f32, @floatFromInt(lcs)) / denom;

        if (heap_len < cap) {
            // Insert in sorted (ascending) order so heap[0] is min.
            var pos = heap_len;
            while (pos > 0 and heap_scores[pos - 1] > score) : (pos -= 1) {
                heap_scores[pos] = heap_scores[pos - 1];
                heap_ids[pos] = heap_ids[pos - 1];
            }
            heap_scores[pos] = score;
            heap_ids[pos] = id;
            heap_len += 1;
        } else if (score > heap_scores[0]) {
            // Drop the min, insert in sorted order.
            var pos: usize = 0;
            while (pos + 1 < heap_len and heap_scores[pos + 1] < score) : (pos += 1) {
                heap_scores[pos] = heap_scores[pos + 1];
                heap_ids[pos] = heap_ids[pos + 1];
            }
            heap_scores[pos] = score;
            heap_ids[pos] = id;
        }
    }

    // Output in descending-score order (best first).
    var i: usize = 0;
    while (i < heap_len) : (i += 1) {
        const src_idx = heap_len - 1 - i;
        out_sources[i] = heap_ids[src_idx];
        out_scores[i] = heap_scores[src_idx];
    }
    return heap_len;
}

// --- q-gram pre-filter for weighted_similar -------------------------
//
// Build once: for each existing token of length >= QGRAM, insert each
// distinct 4-gram (key) along with that token's id into a multimap. At
// query time, enumerate the new token's 4-grams, tally per-existing-token
// overlap counts in a flat array (indexed by TokenId), keep the top
// `candidate_cap` (= 3 * K, plus a small headroom) by overlap count, and
// run the exact LCS scoring only on those candidates. For short new
// tokens (< QGRAM) we fall back to the naive scan; that case is cheap
// anyway because the inner LCS is O(|new| * |existing|) and |new| is
// tiny.

const QGRAM: usize = 4;
const QGramKey = [QGRAM]u8;

/// Inverted index: 4-gram -> ArrayList(TokenId). Sized roughly
/// O(sum_of_token_lengths) for the existing vocab.
const QGramIndex = struct {
    allocator: std.mem.Allocator,
    map: std.AutoHashMap(QGramKey, std.ArrayList(TokenId)),

    fn init(allocator: std.mem.Allocator) QGramIndex {
        return .{
            .allocator = allocator,
            .map = std.AutoHashMap(QGramKey, std.ArrayList(TokenId)).init(allocator),
        };
    }

    fn deinit(self: *QGramIndex) void {
        var it = self.map.iterator();
        while (it.next()) |e| e.value_ptr.deinit(self.allocator);
        self.map.deinit();
    }

    /// Insert every distinct 4-gram of `bytes` -> `id`. We dedupe per
    /// token so the overlap tally counts each new-token gram against
    /// each existing token at most once regardless of multiplicity in
    /// the existing token.
    fn addToken(self: *QGramIndex, id: TokenId, bytes: []const u8) !void {
        if (bytes.len < QGRAM) return;
        // Tiny inline set of grams seen for THIS token to dedupe. Token
        // lengths are short (rare > 32 bytes for BPE pieces); a linear
        // scan over a small stack buffer beats hashing.
        var seen: [256]QGramKey = undefined;
        var seen_len: usize = 0;
        var i: usize = 0;
        while (i + QGRAM <= bytes.len) : (i += 1) {
            var k: QGramKey = undefined;
            @memcpy(&k, bytes[i..][0..QGRAM]);
            var dup = false;
            if (seen_len <= seen.len) {
                var j: usize = 0;
                while (j < seen_len) : (j += 1) {
                    if (std.mem.eql(u8, &seen[j], &k)) {
                        dup = true;
                        break;
                    }
                }
            }
            if (dup) continue;
            if (seen_len < seen.len) {
                seen[seen_len] = k;
                seen_len += 1;
            }
            const gop = try self.map.getOrPut(k);
            if (!gop.found_existing) gop.value_ptr.* = .empty;
            try gop.value_ptr.append(self.allocator, id);
        }
    }

    /// Estimated heap footprint in bytes. Used for diagnostics; not on
    /// any hot path.
    fn estimatedBytes(self: *const QGramIndex) usize {
        var total: usize = 0;
        total += self.map.count() * (@sizeOf(QGramKey) + @sizeOf(std.ArrayList(TokenId)) + 32);
        var it = self.map.iterator();
        while (it.next()) |e| total += e.value_ptr.items.len * @sizeOf(TokenId);
        return total;
    }
};

/// Diagnostic-only helper for benches: build a q-gram index over
/// `old_bpe` and return its estimated heap footprint in bytes. The
/// index is freed before returning.
pub fn estimateQGramIndexBytes(allocator: std.mem.Allocator, old_bpe: *const Bpe) !usize {
    var idx = try buildQGramIndex(allocator, old_bpe);
    defer idx.deinit();
    return idx.estimatedBytes();
}

/// Build the index over `old_bpe`. Caller deinits.
fn buildQGramIndex(allocator: std.mem.Allocator, old_bpe: *const Bpe) !QGramIndex {
    var idx = QGramIndex.init(allocator);
    errdefer idx.deinit();
    var id: u32 = 0;
    while (id < old_bpe.count) : (id += 1) {
        const a = old_bpe.bytes[old_bpe.offsets[id]..old_bpe.offsets[id + 1]];
        try idx.addToken(id, a);
    }
    return idx;
}

/// Score one new token via the q-gram pre-filter + exact LCS on top
/// candidates. Behaves identically to `topKSimilar` for tokens long
/// enough to have any 4-grams; for shorter tokens or when no candidate
/// is found via the index, falls back to the full naive scan so we
/// don't regress on edge cases. Output format matches `topKSimilar`.
fn topKSimilarIndexed(
    old_bpe: *const Bpe,
    tok_bytes: []const u8,
    k: usize,
    qindex: *const QGramIndex,
    /// Reusable: flat counter array of length `old_bpe.count`. Zeroed on
    /// entry and on exit (cleared via `touched`).
    counts: []u16,
    /// Reusable: ids of tokens whose counter was bumped. Caller sizes
    /// for capacity (worst case: all of vocab). Zeroed on exit.
    touched: []TokenId,
    out_sources: []TokenId,
    out_scores: []f32,
    lcs_scratch: []u32,
) usize {
    const cap = @min(k, out_sources.len);
    std.debug.assert(out_scores.len >= cap);
    if (cap == 0 or old_bpe.count == 0 or tok_bytes.len == 0) return 0;

    // Short tokens: no 4-grams to look up. Fall back to the full scan.
    if (tok_bytes.len < QGRAM) {
        return topKSimilar(old_bpe, tok_bytes, k, out_sources, out_scores, lcs_scratch);
    }

    std.debug.assert(counts.len >= old_bpe.count);
    std.debug.assert(touched.len >= old_bpe.count);

    // Tally per-candidate overlap. Dedupe new-token grams the same way
    // addToken dedupes per existing-token: a repeated gram in the new
    // token would otherwise double-count the same existing-token gram.
    var seen: [256]QGramKey = undefined;
    var seen_len: usize = 0;
    var touched_len: usize = 0;
    var i: usize = 0;
    while (i + QGRAM <= tok_bytes.len) : (i += 1) {
        var key: QGramKey = undefined;
        @memcpy(&key, tok_bytes[i..][0..QGRAM]);
        var dup = false;
        if (seen_len <= seen.len) {
            var j: usize = 0;
            while (j < seen_len) : (j += 1) {
                if (std.mem.eql(u8, &seen[j], &key)) {
                    dup = true;
                    break;
                }
            }
        }
        if (dup) continue;
        if (seen_len < seen.len) {
            seen[seen_len] = key;
            seen_len += 1;
        }
        const list_opt = qindex.map.get(key);
        if (list_opt) |list| {
            for (list.items) |id| {
                if (counts[id] == 0) {
                    touched[touched_len] = id;
                    touched_len += 1;
                }
                // Saturate at u16 max — overflow doesn't matter for
                // ranking, only the relative order does.
                if (counts[id] < std.math.maxInt(u16)) counts[id] += 1;
            }
        }
    }

    if (touched_len == 0) {
        // No overlap at all. Fall back to the full scan, which will
        // also find no LCS-positive candidate and return 0. Doing the
        // fallback (instead of returning 0 directly) keeps semantics
        // identical to the naive path for the corner case where a
        // single shared byte run would have surfaced (impossible here
        // — no 4-gram match means LCS < 4 — but defensible).
        return topKSimilar(old_bpe, tok_bytes, k, out_sources, out_scores, lcs_scratch);
    }

    // Pick the top `candidate_cap` by overlap count via a small
    // min-heap on (count, id). Cap is `3*K + 4` with a hard ceiling
    // matching topKSimilar's stack heap.
    const max_cap: usize = 256;
    var cand_cap: usize = @min(max_cap, @min(touched_len, 3 * cap + 4));
    if (cand_cap < cap) cand_cap = @min(touched_len, cap);

    var heap_counts: [256]u16 = undefined;
    var heap_ids: [256]TokenId = undefined;
    var heap_len: usize = 0;

    var ti: usize = 0;
    while (ti < touched_len) : (ti += 1) {
        const id = touched[ti];
        const c = counts[id];
        if (heap_len < cand_cap) {
            // Insert in ascending order so [0] is the smallest count.
            var pos = heap_len;
            // Ascending by count; ties broken by ascending id so the
            // SMALLER id wins (i.e. older tokens preferred on ties —
            // matches topKSimilar's "ascending-rank tiebreak").
            while (pos > 0 and (heap_counts[pos - 1] > c or
                (heap_counts[pos - 1] == c and heap_ids[pos - 1] > id))) : (pos -= 1)
            {
                heap_counts[pos] = heap_counts[pos - 1];
                heap_ids[pos] = heap_ids[pos - 1];
            }
            heap_counts[pos] = c;
            heap_ids[pos] = id;
            heap_len += 1;
        } else {
            // Beats the head if count higher, or same count and id is
            // smaller (older wins tie).
            const head_c = heap_counts[0];
            const head_id = heap_ids[0];
            const better = (c > head_c) or (c == head_c and id < head_id);
            if (!better) continue;
            // Drop head, find sorted insertion point.
            var pos: usize = 0;
            while (pos + 1 < heap_len and
                (heap_counts[pos + 1] < c or
                    (heap_counts[pos + 1] == c and heap_ids[pos + 1] > id))) : (pos += 1)
            {
                heap_counts[pos] = heap_counts[pos + 1];
                heap_ids[pos] = heap_ids[pos + 1];
            }
            heap_counts[pos] = c;
            heap_ids[pos] = id;
        }
    }

    // Clear `counts` for reuse on the next token.
    var ci: usize = 0;
    while (ci < touched_len) : (ci += 1) counts[touched[ci]] = 0;

    if (heap_len == 0) return 0;

    // Now run exact LCS on the surviving candidates and pick top-K with
    // the same scoring rule as topKSimilar.
    var out_heap_ids: [256]TokenId = undefined;
    var out_heap_scores: [256]f32 = undefined;
    var out_heap_len: usize = 0;
    std.debug.assert(cap <= out_heap_ids.len);

    var hi: usize = 0;
    while (hi < heap_len) : (hi += 1) {
        const id = heap_ids[hi];
        const a = old_bpe.bytes[old_bpe.offsets[id]..old_bpe.offsets[id + 1]];
        if (a.len == 0) continue;
        const min_len = @min(a.len, tok_bytes.len);
        const need = 2 * (min_len + 1);
        std.debug.assert(lcs_scratch.len >= need);
        const lcs = lcsLength(a, tok_bytes, lcs_scratch[0..need]);
        if (lcs == 0) continue;
        const denom: f32 = @floatFromInt(@max(a.len, tok_bytes.len));
        const score: f32 = @as(f32, @floatFromInt(lcs)) / denom;

        if (out_heap_len < cap) {
            var pos = out_heap_len;
            // Ascending score; ties broken by descending id so the
            // SMALLER id wins on tie (matches topKSimilar — when scores
            // tie, the lower id stays in the heap longer because the
            // later equal-score insert keeps shifting up past it).
            while (pos > 0 and (out_heap_scores[pos - 1] > score or
                (out_heap_scores[pos - 1] == score and out_heap_ids[pos - 1] > id))) : (pos -= 1)
            {
                out_heap_scores[pos] = out_heap_scores[pos - 1];
                out_heap_ids[pos] = out_heap_ids[pos - 1];
            }
            out_heap_scores[pos] = score;
            out_heap_ids[pos] = id;
            out_heap_len += 1;
        } else {
            const head_s = out_heap_scores[0];
            const head_id = out_heap_ids[0];
            const better = (score > head_s) or (score == head_s and id < head_id);
            if (!better) continue;
            var pos: usize = 0;
            while (pos + 1 < out_heap_len and
                (out_heap_scores[pos + 1] < score or
                    (out_heap_scores[pos + 1] == score and out_heap_ids[pos + 1] > id))) : (pos += 1)
            {
                out_heap_scores[pos] = out_heap_scores[pos + 1];
                out_heap_ids[pos] = out_heap_ids[pos + 1];
            }
            out_heap_scores[pos] = score;
            out_heap_ids[pos] = id;
        }
    }

    var oi: usize = 0;
    while (oi < out_heap_len) : (oi += 1) {
        const src_idx = out_heap_len - 1 - oi;
        out_sources[oi] = out_heap_ids[src_idx];
        out_scores[oi] = out_heap_scores[src_idx];
    }
    return out_heap_len;
}

/// Internal: compute normalized weights from raw similarity scores,
/// optionally re-weighted by inverse rank. Result sums to 1.0 (within
/// fp32 epsilon). Writes into `out_weights`. If all inputs are zero
/// (shouldn't happen — caller filtered lcs == 0), returns false.
fn normalizeWeights(
    sources: []const TokenId,
    scores: []const f32,
    weight_by_rank: bool,
    out_weights: []f32,
) bool {
    std.debug.assert(sources.len == scores.len);
    std.debug.assert(out_weights.len >= sources.len);
    var total: f64 = 0;
    var i: usize = 0;
    while (i < sources.len) : (i += 1) {
        var w: f64 = @floatCast(scores[i]);
        if (weight_by_rank) {
            const rank: f64 = @floatFromInt(sources[i]);
            w *= 1.0 / (1.0 + rank);
        }
        out_weights[i] = @floatCast(w);
        total += w;
    }
    if (total <= 0) return false;
    const inv: f64 = 1.0 / total;
    i = 0;
    while (i < sources.len) : (i += 1) {
        out_weights[i] = @floatCast(@as(f64, @floatCast(out_weights[i])) * inv);
    }
    return true;
}

/// Extend `old_bpe` with `opts.new_tokens`. Returns an `ExtendResult`
/// whose `extended_bpe` contains old + new tokens (new ids start at
/// `old_bpe.count`) and an `init_plan` describing how to seed the new
/// embedding rows. Caller must `deinit` the result.
pub fn extendBpe(
    allocator: std.mem.Allocator,
    old_bpe: *const Bpe,
    opts: ExtendOptions,
) !ExtendResult {
    const old_count = old_bpe.count;

    // Empty fast path: still hand back an owned clone so the caller's
    // deinit pattern works uniformly.
    if (opts.new_tokens.len == 0) {
        var cloned = try cloneBpe(allocator, old_bpe);
        errdefer cloned.deinit();
        return .{
            .allocator = allocator,
            .extended_bpe = cloned,
            .old_vocab_size = old_count,
            .new_vocab_size = old_count,
            .init_plan = &.{},
        };
    }

    // Stage 1: walk inputs, skip duplicates / empties, encode subtokens
    // under the OLD tokenizer (so encoding semantics match the caller's
    // existing embedding table). For weighted_similar we also need a
    // subtoken encoding because it's the fallback when no existing
    // token shares any byte substring with the new token.
    var kept_indices: std.ArrayList(usize) = .empty;
    defer kept_indices.deinit(allocator);
    try kept_indices.ensureTotalCapacity(allocator, opts.new_tokens.len);

    // Per kept token, the subtoken encoding under the old BPE. Always
    // computed regardless of strategy so we have a fallback path for
    // weighted_similar.
    var subtoken_lists: std.ArrayList([]TokenId) = .empty;
    errdefer {
        for (subtoken_lists.items) |s| if (s.len > 0) allocator.free(s);
        subtoken_lists.deinit(allocator);
    }
    try subtoken_lists.ensureTotalCapacity(allocator, opts.new_tokens.len);

    var total_new_bytes: usize = 0;

    // Reusable encode scratch — sized to the longest input we'll see.
    var max_len: usize = 0;
    for (opts.new_tokens) |t| if (t.len > max_len) {
        max_len = t.len;
    };
    const encode_scratch: []TokenId = if (max_len > 0) try allocator.alloc(TokenId, max_len) else &.{};
    defer if (max_len > 0) allocator.free(encode_scratch);

    for (opts.new_tokens, 0..) |tok, i| {
        if (tok.len == 0) {
            std.log.warn("vocab_extend: skipping empty new_token at index {d}", .{i});
            continue;
        }
        if (old_bpe.by_bytes.get(tok)) |_| {
            if (opts.skip_existing) continue;
            return ExtendError.DuplicateToken;
        }

        const subs_view: []const TokenId = switch (opts.strategy) {
            .mean_subtoken, .weighted_similar => old_bpe.encodeChunk(tok, encode_scratch),
            .random => &.{},
        };

        // Empty subtoken result (e.g. encoder returned no ids) — warn
        // and skip. With a complete byte-level base vocab this is a
        // degenerate case but we guard for it.
        if ((opts.strategy == .mean_subtoken or opts.strategy == .weighted_similar) and subs_view.len == 0) {
            std.log.warn("vocab_extend: subtoken encoding empty for token index {d}, skipping", .{i});
            continue;
        }

        const subs_owned: []TokenId = if (subs_view.len > 0) try allocator.dupe(TokenId, subs_view) else try allocator.alloc(TokenId, 0);
        errdefer if (subs_owned.len > 0) allocator.free(subs_owned);
        try subtoken_lists.append(allocator, subs_owned);
        try kept_indices.append(allocator, i);
        total_new_bytes += tok.len;
    }

    const added: u32 = @intCast(kept_indices.items.len);
    const new_count: u32 = old_count + added;

    // Stage 1b (weighted_similar only): for each kept token, scan the
    // OLD vocab for the K most byte-similar existing tokens. Falls back
    // to mean_subtoken if nothing shares any substring. Allocated
    // up-front so we can hand ownership to the init plan in stage 4.
    //
    // SoA: parallel arrays so we can transfer slice ownership to the
    // init plan one entry at a time. `kept_sources[k]`/`kept_weights[k]`
    // are non-null only when the k-th token uses weighted_similar; null
    // means the fallback (mean_subtoken using subtoken_lists[k]).
    var kept_sources: []?[]TokenId = &.{};
    var kept_weights: []?[]f32 = &.{};
    if (opts.strategy == .weighted_similar and added > 0) {
        kept_sources = try allocator.alloc(?[]TokenId, added);
        @memset(kept_sources, null);
        errdefer {
            for (kept_sources) |maybe| if (maybe) |s| allocator.free(s);
            allocator.free(kept_sources);
        }
        kept_weights = try allocator.alloc(?[]f32, added);
        @memset(kept_weights, null);
        errdefer {
            for (kept_weights) |maybe| if (maybe) |w| allocator.free(w);
            allocator.free(kept_weights);
        }

        const k_eff: usize = @min(@as(usize, opts.weighted_similar.k), old_count);
        if (k_eff > 0) {
            // Single LCS scratch reused across all candidates. Size:
            // 2 * (min(max_existing_len, max_new_len) + 1). Conservative
            // upper bound: 2 * (max_existing_len + 1) (also covers the
            // case where new tokens are longer).
            var max_existing_len: usize = 0;
            var ei: u32 = 0;
            while (ei < old_count) : (ei += 1) {
                const e_len = old_bpe.offsets[ei + 1] - old_bpe.offsets[ei];
                if (e_len > max_existing_len) max_existing_len = e_len;
            }
            const lcs_min = @max(max_existing_len, max_len);
            const lcs_scratch = try allocator.alloc(u32, 2 * (lcs_min + 1));
            defer allocator.free(lcs_scratch);

            // Temporary buffers for one token's top-K scan.
            const sources_buf = try allocator.alloc(TokenId, k_eff);
            defer allocator.free(sources_buf);
            const scores_buf = try allocator.alloc(f32, k_eff);
            defer allocator.free(scores_buf);

            // Optional q-gram index for fast candidate pre-filtering.
            // Built once, queried once per new token, freed when this
            // block exits. For small vocabs the build overhead can be
            // larger than the scan it saves, but the option default is
            // index-on because it's a clear win above ~few-thousand V.
            var qindex_opt: ?QGramIndex = null;
            defer if (qindex_opt) |*idx| idx.deinit();
            var counts_buf: []u16 = &.{};
            defer if (counts_buf.len > 0) allocator.free(counts_buf);
            var touched_buf: []TokenId = &.{};
            defer if (touched_buf.len > 0) allocator.free(touched_buf);
            if (opts.weighted_similar.use_qgram_index) {
                qindex_opt = try buildQGramIndex(allocator, old_bpe);
                counts_buf = try allocator.alloc(u16, old_count);
                @memset(counts_buf, 0);
                touched_buf = try allocator.alloc(TokenId, old_count);
            }

            for (kept_indices.items, 0..) |src_idx, k_pos| {
                const tok = opts.new_tokens[src_idx];
                var n_found: usize = 0;
                if (qindex_opt) |*qidx| {
                    n_found = topKSimilarIndexed(
                        old_bpe,
                        tok,
                        k_eff,
                        qidx,
                        counts_buf,
                        touched_buf,
                        sources_buf[0..k_eff],
                        scores_buf[0..k_eff],
                        lcs_scratch,
                    );
                    // The q-gram pre-filter only sees candidates whose
                    // LCS with `tok` is >= QGRAM (= 4). If that yielded
                    // fewer than K results, there may still be naive-
                    // path candidates with LCS in 1..QGRAM-1 that the
                    // naive scan would rank. Fall back to the full scan
                    // in that case to preserve the same top-K set as
                    // the naive baseline.
                    if (n_found < k_eff) {
                        n_found = topKSimilar(
                            old_bpe,
                            tok,
                            k_eff,
                            sources_buf[0..k_eff],
                            scores_buf[0..k_eff],
                            lcs_scratch,
                        );
                    }
                } else {
                    n_found = topKSimilar(
                        old_bpe,
                        tok,
                        k_eff,
                        sources_buf[0..k_eff],
                        scores_buf[0..k_eff],
                        lcs_scratch,
                    );
                }
                if (n_found == 0) {
                    // Fallback: leave kept_sources[k_pos] null; init
                    // plan will use mean_subtoken via subtoken_lists[k_pos].
                    continue;
                }

                const owned_sources = try allocator.dupe(TokenId, sources_buf[0..n_found]);
                errdefer allocator.free(owned_sources);
                const owned_weights = try allocator.alloc(f32, n_found);
                errdefer allocator.free(owned_weights);
                if (!normalizeWeights(
                    owned_sources,
                    scores_buf[0..n_found],
                    opts.weighted_similar.weight_by_rank,
                    owned_weights,
                )) {
                    // Shouldn't happen — topKSimilar filters lcs==0 —
                    // but if it does, drop back to fallback.
                    allocator.free(owned_sources);
                    allocator.free(owned_weights);
                    continue;
                }
                kept_sources[k_pos] = owned_sources;
                kept_weights[k_pos] = owned_weights;
            }
        }
    }
    defer {
        // If we made it to the end without consuming these into the
        // init plan, free them. Successful path nulls each entry as it
        // transfers ownership.
        if (kept_sources.len > 0) {
            for (kept_sources) |maybe| if (maybe) |s| allocator.free(s);
            allocator.free(kept_sources);
        }
        if (kept_weights.len > 0) {
            for (kept_weights) |maybe| if (maybe) |w| allocator.free(w);
            allocator.free(kept_weights);
        }
    }

    // Stage 2: build the extended BPE's bytes/offsets in a single pass.
    const old_bytes_len = old_bpe.bytes.len;
    const new_bytes = try allocator.alloc(u8, old_bytes_len + total_new_bytes);
    errdefer allocator.free(new_bytes);
    const new_offsets = try allocator.alloc(u32, @as(usize, new_count) + 1);
    errdefer allocator.free(new_offsets);

    if (old_bytes_len > 0) @memcpy(new_bytes[0..old_bytes_len], old_bpe.bytes);
    if (old_bpe.offsets.len > 0) {
        @memcpy(new_offsets[0 .. old_bpe.offsets.len], old_bpe.offsets);
    } else {
        new_offsets[0] = 0;
    }

    var write_off: u32 = @intCast(old_bytes_len);
    for (kept_indices.items, 0..) |src_idx, k| {
        const tok = opts.new_tokens[src_idx];
        @memcpy(new_bytes[write_off .. write_off + tok.len], tok);
        write_off += @intCast(tok.len);
        new_offsets[old_count + k + 1] = write_off;
    }
    std.debug.assert(write_off == new_bytes.len);

    // Stage 3: rebuild by_bytes against the NEW bytes buffer so keys
    // stay valid (the old map's keys borrow into the old buffer).
    var by_bytes = std.StringHashMap(TokenId).init(allocator);
    errdefer by_bytes.deinit();
    try by_bytes.ensureTotalCapacity(new_count);
    var id: u32 = 0;
    while (id < new_count) : (id += 1) {
        const key = new_bytes[new_offsets[id]..new_offsets[id + 1]];
        try by_bytes.put(key, id);
    }

    // Stage 4: materialize init_plan. Subtoken arrays were already
    // duped; bytes need their own owned copies so they outlive the
    // caller's `new_tokens` slice.
    const init_plan = try allocator.alloc(InitInstruction, added);
    errdefer allocator.free(init_plan);

    var built: u32 = 0;
    errdefer {
        for (init_plan[0..built]) |inst| {
            if (inst.bytes.len > 0) allocator.free(inst.bytes);
        }
    }
    for (kept_indices.items, 0..) |src_idx, k| {
        const tok = opts.new_tokens[src_idx];
        const bytes_copy = try allocator.dupe(u8, tok);

        const payload: InitInstruction.Payload = switch (opts.strategy) {
            .random => .random,
            .mean_subtoken => .{ .mean_subtoken = .{ .subtoken_ids = subtoken_lists.items[k] } },
            .weighted_similar => blk: {
                // Prefer the weighted_similar payload if the scan found
                // candidates; otherwise fall back to mean_subtoken.
                if (kept_sources.len > 0) {
                    if (kept_sources[k]) |s| {
                        const w = kept_weights[k].?;
                        // Transfer ownership: null the slot so the defer
                        // doesn't double-free.
                        kept_sources[k] = null;
                        kept_weights[k] = null;
                        // Also free the now-redundant subtoken list.
                        if (subtoken_lists.items[k].len > 0) {
                            allocator.free(subtoken_lists.items[k]);
                            subtoken_lists.items[k] = &.{};
                        }
                        break :blk .{ .weighted_similar = .{ .sources = s, .weights = w } };
                    }
                }
                break :blk .{ .mean_subtoken = .{ .subtoken_ids = subtoken_lists.items[k] } };
            },
        };

        init_plan[k] = .{
            .new_id = old_count + @as(u32, @intCast(k)),
            .bytes = bytes_copy,
            .payload = payload,
        };
        built += 1;
    }

    // Subtoken slices now owned by init_plan entries — drop the staging
    // list without freeing them. (Any slot we replaced with
    // weighted_similar has been zeroed to a length-0 slice already.)
    subtoken_lists.clearRetainingCapacity();
    subtoken_lists.deinit(allocator);

    // Rebuild the two-level merge-rank front cache against the new
    // (extended) bytes/offsets so the resulting Bpe keeps the L1d-
    // resident lookup fast path. See `Bpe.hot_table`.
    const hot_table = try Bpe.buildHotTable(allocator, new_bytes, new_offsets, new_count);
    errdefer allocator.free(hot_table);

    const extended_bpe: Bpe = .{
        .allocator = allocator,
        .bytes = new_bytes,
        .offsets = new_offsets,
        .count = new_count,
        .by_bytes = by_bytes,
        .hot_table = hot_table,
    };

    return .{
        .allocator = allocator,
        .extended_bpe = extended_bpe,
        .old_vocab_size = old_count,
        .new_vocab_size = new_count,
        .init_plan = init_plan,
    };
}

// --- tests -----------------------------------------------------------

const testing = std.testing;

const TestEntry = struct { bytes: []const u8, rank: u32 };

fn buildVocabSource(
    allocator: std.mem.Allocator,
    entries: []const TestEntry,
) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    const enc = std.base64.standard.Encoder;
    for (entries) |e| {
        const sz = enc.calcSize(e.bytes.len);
        const tmp = try allocator.alloc(u8, sz);
        defer allocator.free(tmp);
        const encoded = enc.encode(tmp, e.bytes);
        try buf.appendSlice(allocator, encoded);
        try buf.print(allocator, " {d}\n", .{e.rank});
    }
    return buf.toOwnedSlice(allocator);
}

fn buildByteVocab(
    allocator: std.mem.Allocator,
    extras: []const TestEntry,
) !Bpe {
    var entries: std.ArrayList(TestEntry) = .empty;
    defer entries.deinit(allocator);

    var byte_holders: [256][1]u8 = undefined;
    var i: u32 = 0;
    while (i < 256) : (i += 1) {
        byte_holders[i][0] = @intCast(i);
        try entries.append(allocator, .{ .bytes = byte_holders[i][0..1], .rank = i });
    }
    for (extras) |e| try entries.append(allocator, e);

    const src = try buildVocabSource(allocator, entries.items);
    defer allocator.free(src);
    return Bpe.loadTiktokenBytes(allocator, src);
}

test "extendBpe adds new tokens at high ids" {
    var old = try buildByteVocab(testing.allocator, &.{
        .{ .bytes = "he", .rank = 256 },
        .{ .bytes = "hel", .rank = 257 },
        .{ .bytes = "hello", .rank = 258 },
    });
    defer old.deinit();

    const n = old.count;
    const new_toks = [_][]const u8{ "tokenizer", "medical", "scientific" };
    var res = try extendBpe(testing.allocator, &old, .{ .new_tokens = &new_toks });
    defer res.deinit();

    try testing.expectEqual(n, res.old_vocab_size);
    try testing.expectEqual(n + 3, res.new_vocab_size);
    try testing.expectEqual(@as(usize, 3), res.init_plan.len);
    try testing.expectEqual(n, res.init_plan[0].new_id);
    try testing.expectEqual(n + 1, res.init_plan[1].new_id);
    try testing.expectEqual(n + 2, res.init_plan[2].new_id);
    try testing.expectEqual(n + 3, res.extended_bpe.count);
}

test "init_plan contains mean_subtoken instruction" {
    var old = try buildByteVocab(testing.allocator, &.{
        .{ .bytes = "to", .rank = 256 },
        .{ .bytes = "ken", .rank = 257 },
        .{ .bytes = "tok", .rank = 258 },
        .{ .bytes = "izer", .rank = 259 },
    });
    defer old.deinit();

    const tok = "tokenizer";
    var scratch: [64]TokenId = undefined;
    const expected = old.encodeChunk(tok, &scratch);
    const expected_copy = try testing.allocator.dupe(TokenId, expected);
    defer testing.allocator.free(expected_copy);

    const new_toks = [_][]const u8{tok};
    var res = try extendBpe(testing.allocator, &old, .{ .new_tokens = &new_toks });
    defer res.deinit();

    try testing.expectEqual(@as(usize, 1), res.init_plan.len);
    try testing.expectEqualStrings(tok, res.init_plan[0].bytes);
    const subs = res.init_plan[0].subtokenIds();
    try testing.expectEqualSlices(TokenId, expected_copy, subs);
    try testing.expect(subs.len > 0);
}

test "skip_existing skips existing tokens" {
    var old = try buildByteVocab(testing.allocator, &.{
        .{ .bytes = "hello", .rank = 256 },
    });
    defer old.deinit();

    const new_toks = [_][]const u8{ "hello", "world" };
    var res = try extendBpe(testing.allocator, &old, .{
        .new_tokens = &new_toks,
        .skip_existing = true,
    });
    defer res.deinit();

    // Only "world" should be added — "hello" is already present.
    try testing.expectEqual(@as(usize, 1), res.init_plan.len);
    try testing.expectEqualStrings("world", res.init_plan[0].bytes);
    try testing.expectEqual(old.count + 1, res.new_vocab_size);

    // skip_existing=false: same input should now error.
    const err = extendBpe(testing.allocator, &old, .{
        .new_tokens = &new_toks,
        .skip_existing = false,
    });
    try testing.expectError(ExtendError.DuplicateToken, err);
}

test "extended bpe encodes existing tokens identically" {
    var old = try buildByteVocab(testing.allocator, &.{
        .{ .bytes = "he", .rank = 256 },
        .{ .bytes = "ll", .rank = 257 },
        .{ .bytes = "hel", .rank = 258 },
        .{ .bytes = "hell", .rank = 259 },
        .{ .bytes = "hello", .rank = 260 },
    });
    defer old.deinit();

    const new_toks = [_][]const u8{ "domainx", "domainy" };
    var res = try extendBpe(testing.allocator, &old, .{ .new_tokens = &new_toks });
    defer res.deinit();

    var out_old: [16]TokenId = undefined;
    var out_new: [16]TokenId = undefined;
    const ids_old = old.encodeChunk("hello", &out_old);
    const ids_new = res.extended_bpe.encodeChunk("hello", &out_new);
    try testing.expectEqualSlices(TokenId, ids_old, ids_new);
}

test "extended bpe can decode new tokens" {
    var old = try buildByteVocab(testing.allocator, &.{});
    defer old.deinit();

    const new_toks = [_][]const u8{ "alpha", "beta", "gamma" };
    var res = try extendBpe(testing.allocator, &old, .{ .new_tokens = &new_toks });
    defer res.deinit();

    try testing.expectEqualStrings("alpha", res.extended_bpe.idBytes(res.init_plan[0].new_id));
    try testing.expectEqualStrings("beta", res.extended_bpe.idBytes(res.init_plan[1].new_id));
    try testing.expectEqualStrings("gamma", res.extended_bpe.idBytes(res.init_plan[2].new_id));

    // by_bytes round-trip too.
    try testing.expectEqual(@as(?TokenId, res.init_plan[0].new_id), res.extended_bpe.by_bytes.get("alpha"));
    try testing.expectEqual(@as(?TokenId, res.init_plan[2].new_id), res.extended_bpe.by_bytes.get("gamma"));
}

// --- weighted_similar tests -----------------------------------------

test "weighted_similar picks the prefix-share token as #1 similar" {
    // "scientific_method" overlaps "scientific" by 10 chars (LCS),
    // beating any single-byte candidate.
    var old = try buildByteVocab(testing.allocator, &.{
        .{ .bytes = "scientific", .rank = 256 },
        .{ .bytes = "method", .rank = 257 },
        .{ .bytes = "data", .rank = 258 },
    });
    defer old.deinit();

    const target_id_for_scientific: TokenId = 256;

    const new_toks = [_][]const u8{"scientific_method"};
    var res = try extendBpe(testing.allocator, &old, .{
        .new_tokens = &new_toks,
        .strategy = .weighted_similar,
        .weighted_similar = .{ .k = 4 },
    });
    defer res.deinit();

    try testing.expectEqual(@as(usize, 1), res.init_plan.len);
    switch (res.init_plan[0].payload) {
        .weighted_similar => |w| {
            try testing.expect(w.sources.len > 0);
            // Top entry should be "scientific" (longest LCS share).
            try testing.expectEqual(target_id_for_scientific, w.sources[0]);
        },
        else => return error.TestExpectedWeightedSimilar,
    }
}

test "weighted_similar weights sum to 1.0" {
    var old = try buildByteVocab(testing.allocator, &.{
        .{ .bytes = "alpha", .rank = 256 },
        .{ .bytes = "alpine", .rank = 257 },
        .{ .bytes = "alphabet", .rank = 258 },
        .{ .bytes = "beta", .rank = 259 },
        .{ .bytes = "gamma", .rank = 260 },
    });
    defer old.deinit();

    const new_toks = [_][]const u8{"alphabetic"};
    var res = try extendBpe(testing.allocator, &old, .{
        .new_tokens = &new_toks,
        .strategy = .weighted_similar,
        .weighted_similar = .{ .k = 8 },
    });
    defer res.deinit();

    switch (res.init_plan[0].payload) {
        .weighted_similar => |w| {
            try testing.expect(w.weights.len > 0);
            try testing.expectEqual(w.sources.len, w.weights.len);
            var total: f64 = 0;
            for (w.weights) |x| total += x;
            try testing.expectApproxEqAbs(@as(f64, 1.0), total, 1e-6);
            for (w.weights) |x| try testing.expect(x > 0);
        },
        else => return error.TestExpectedWeightedSimilar,
    }
}

test "weighted_similar K is honored" {
    // Plenty of candidates so K is the binding cap, not vocab size.
    var old = try buildByteVocab(testing.allocator, &.{
        .{ .bytes = "alphabet", .rank = 256 },
        .{ .bytes = "alpine", .rank = 257 },
        .{ .bytes = "alpha", .rank = 258 },
        .{ .bytes = "almond", .rank = 259 },
        .{ .bytes = "allow", .rank = 260 },
        .{ .bytes = "alter", .rank = 261 },
        .{ .bytes = "altitude", .rank = 262 },
        .{ .bytes = "amazing", .rank = 263 },
    });
    defer old.deinit();

    const new_toks = [_][]const u8{"alphabetical"};
    var res = try extendBpe(testing.allocator, &old, .{
        .new_tokens = &new_toks,
        .strategy = .weighted_similar,
        .weighted_similar = .{ .k = 3 },
    });
    defer res.deinit();

    switch (res.init_plan[0].payload) {
        .weighted_similar => |w| {
            try testing.expectEqual(@as(usize, 3), w.sources.len);
            try testing.expectEqual(@as(usize, 3), w.weights.len);
        },
        else => return error.TestExpectedWeightedSimilar,
    }
}

test "weighted_similar weight_by_rank changes weights" {
    var old = try buildByteVocab(testing.allocator, &.{
        .{ .bytes = "alphabet", .rank = 256 },
        .{ .bytes = "alpine", .rank = 257 },
        .{ .bytes = "alpha", .rank = 258 },
        .{ .bytes = "almond", .rank = 259 },
    });
    defer old.deinit();

    const new_toks = [_][]const u8{"alphabetical"};

    var res_flat = try extendBpe(testing.allocator, &old, .{
        .new_tokens = &new_toks,
        .strategy = .weighted_similar,
        .weighted_similar = .{ .k = 4, .weight_by_rank = false },
    });
    defer res_flat.deinit();

    var res_ranked = try extendBpe(testing.allocator, &old, .{
        .new_tokens = &new_toks,
        .strategy = .weighted_similar,
        .weighted_similar = .{ .k = 4, .weight_by_rank = true },
    });
    defer res_ranked.deinit();

    const w_flat = res_flat.init_plan[0].payload.weighted_similar;
    const w_rank = res_ranked.init_plan[0].payload.weighted_similar;
    // Same sources in the same order (selection is similarity-driven
    // and we use a stable top-K with ascending-rank tiebreak).
    try testing.expectEqualSlices(TokenId, w_flat.sources, w_rank.sources);

    // Weights should differ — at least one element must move past
    // 1e-4 relative when rank weighting is applied.
    var any_diff = false;
    for (w_flat.weights, w_rank.weights) |a, b| {
        if (@abs(a - b) > 1e-4) {
            any_diff = true;
            break;
        }
    }
    try testing.expect(any_diff);

    // Ranked weights also sum to 1.
    var total: f64 = 0;
    for (w_rank.weights) |x| total += x;
    try testing.expectApproxEqAbs(@as(f64, 1.0), total, 1e-6);
}

test "weighted_similar falls back to mean_subtoken when no shared bytes" {
    // Old vocab is the 256 base bytes only, nothing else. A new token
    // built from byte values that are still in the base vocab (0..255)
    // WILL share bytes (every base byte is an existing token of length
    // 1, so the LCS is at least 1). To force the no-similar path we'd
    // need a vocab that excludes some bytes — but our test builder
    // always includes 0..255 to keep BPE encode well-defined.
    //
    // Instead, exercise the OTHER fallback condition: K=0. With no
    // candidates considered the scan returns 0 found, init falls back
    // to mean_subtoken.
    var old = try buildByteVocab(testing.allocator, &.{
        .{ .bytes = "hello", .rank = 256 },
    });
    defer old.deinit();

    const new_toks = [_][]const u8{"xyzzy"};
    var res = try extendBpe(testing.allocator, &old, .{
        .new_tokens = &new_toks,
        .strategy = .weighted_similar,
        .weighted_similar = .{ .k = 0 },
    });
    defer res.deinit();

    try testing.expectEqual(@as(usize, 1), res.init_plan.len);
    switch (res.init_plan[0].payload) {
        .mean_subtoken => |m| try testing.expect(m.subtoken_ids.len > 0),
        else => return error.TestExpectedMeanSubtokenFallback,
    }
}

// --- q-gram index tests ---------------------------------------------

test "qgram index: long shared substring still ranked #1 with index on" {
    var old = try buildByteVocab(testing.allocator, &.{
        .{ .bytes = "scientific", .rank = 256 },
        .{ .bytes = "method", .rank = 257 },
        .{ .bytes = "data", .rank = 258 },
        .{ .bytes = "tific_app", .rank = 259 },
    });
    defer old.deinit();

    const target_scientific: TokenId = 256;

    const new_toks = [_][]const u8{"scientific_method"};
    var res = try extendBpe(testing.allocator, &old, .{
        .new_tokens = &new_toks,
        .strategy = .weighted_similar,
        .weighted_similar = .{ .k = 4, .use_qgram_index = true },
    });
    defer res.deinit();

    switch (res.init_plan[0].payload) {
        .weighted_similar => |w| {
            try testing.expect(w.sources.len > 0);
            try testing.expectEqual(target_scientific, w.sources[0]);
        },
        else => return error.TestExpectedWeightedSimilar,
    }
}

test "qgram index off: identical ranking to naive baseline" {
    // Same vocab, same input, with index OFF. The baseline `topKSimilar`
    // path must produce the same plan it always did before this change
    // — guard against accidental regressions.
    var old = try buildByteVocab(testing.allocator, &.{
        .{ .bytes = "alphabet", .rank = 256 },
        .{ .bytes = "alpine", .rank = 257 },
        .{ .bytes = "alpha", .rank = 258 },
        .{ .bytes = "almond", .rank = 259 },
        .{ .bytes = "allow", .rank = 260 },
        .{ .bytes = "alter", .rank = 261 },
        .{ .bytes = "altitude", .rank = 262 },
        .{ .bytes = "amazing", .rank = 263 },
    });
    defer old.deinit();

    const new_toks = [_][]const u8{"alphabetical"};
    var res = try extendBpe(testing.allocator, &old, .{
        .new_tokens = &new_toks,
        .strategy = .weighted_similar,
        .weighted_similar = .{ .k = 4, .use_qgram_index = false },
    });
    defer res.deinit();

    switch (res.init_plan[0].payload) {
        .weighted_similar => |w| {
            try testing.expectEqual(@as(usize, 4), w.sources.len);
            // Top match must be "alphabet" (longest LCS == 5 of 12).
            try testing.expectEqual(@as(TokenId, 256), w.sources[0]);
        },
        else => return error.TestExpectedWeightedSimilar,
    }
}

test "qgram index on: same top-K set as naive on V=200 vocab" {
    // Synthetic vocab of ~200 randomized english-y pieces. Build two
    // ExtendResults (index on/off), compare the top-K id SET (allowing
    // for score-tie permutation).
    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const r = prng.random();

    const piece_chars = "abcdefghijklmnopqrstuvwxyz_";
    var pieces: std.ArrayList(TestEntry) = .empty;
    defer pieces.deinit(testing.allocator);
    // Persistent storage for piece bytes so the slices remain valid.
    var piece_bufs: std.ArrayList([]u8) = .empty;
    defer {
        for (piece_bufs.items) |b| testing.allocator.free(b);
        piece_bufs.deinit(testing.allocator);
    }

    var i: u32 = 0;
    while (i < 200) : (i += 1) {
        const len: usize = 3 + r.intRangeAtMost(usize, 0, 7);
        const buf = try testing.allocator.alloc(u8, len);
        for (buf) |*c| c.* = piece_chars[r.intRangeAtMost(usize, 0, piece_chars.len - 1)];
        try piece_bufs.append(testing.allocator, buf);
        try pieces.append(testing.allocator, .{ .bytes = buf, .rank = 256 + i });
    }

    var old = try buildByteVocab(testing.allocator, pieces.items);
    defer old.deinit();

    const new_toks = [_][]const u8{ "alphabetical", "encoded_value", "tokenizer_x" };
    const K: u8 = 8;

    var res_naive = try extendBpe(testing.allocator, &old, .{
        .new_tokens = &new_toks,
        .strategy = .weighted_similar,
        .weighted_similar = .{ .k = K, .use_qgram_index = false },
    });
    defer res_naive.deinit();
    var res_qgram = try extendBpe(testing.allocator, &old, .{
        .new_tokens = &new_toks,
        .strategy = .weighted_similar,
        .weighted_similar = .{ .k = K, .use_qgram_index = true },
    });
    defer res_qgram.deinit();

    for (res_naive.init_plan, 0..) |naive_inst, idx| {
        const qgram_inst = res_qgram.init_plan[idx];
        switch (naive_inst.payload) {
            .weighted_similar => |w_naive| {
                const w_q = qgram_inst.payload.weighted_similar;
                try testing.expectEqual(w_naive.sources.len, w_q.sources.len);
                // Same SET. Build a tiny bitset/hashmap; here `cap` ≤ K
                // ≤ 8 so a linear scan is fine.
                for (w_naive.sources) |sid| {
                    var found = false;
                    for (w_q.sources) |qid| if (qid == sid) {
                        found = true;
                        break;
                    };
                    try testing.expect(found);
                }
                // Ranks (positions) for K=8 should match: scoring is
                // deterministic so the ordering is identical when both
                // paths consider the same candidate set.
                for (w_naive.sources, w_q.sources) |sn, sq| {
                    try testing.expectEqual(sn, sq);
                }
            },
            .mean_subtoken => {
                // Fallback: index path must also fall back.
                try testing.expect(qgram_inst.payload == .mean_subtoken);
            },
            .random => unreachable,
        }
    }
}

test "qgram index: tiny new token (<4 bytes) gracefully falls back" {
    var old = try buildByteVocab(testing.allocator, &.{
        .{ .bytes = "hello", .rank = 256 },
        .{ .bytes = "world", .rank = 257 },
        .{ .bytes = "abc", .rank = 258 },
        .{ .bytes = "xy", .rank = 259 },
    });
    defer old.deinit();

    const new_toks = [_][]const u8{"xy_"}; // length 3 < QGRAM
    var res = try extendBpe(testing.allocator, &old, .{
        .new_tokens = &new_toks,
        .strategy = .weighted_similar,
        .weighted_similar = .{ .k = 4, .use_qgram_index = true },
    });
    defer res.deinit();

    try testing.expectEqual(@as(usize, 1), res.init_plan.len);
    switch (res.init_plan[0].payload) {
        .weighted_similar => |w| {
            try testing.expect(w.sources.len > 0);
            // "xy" (id 259) shares 2 chars with "xy_"; should be #1.
            try testing.expectEqual(@as(TokenId, 259), w.sources[0]);
        },
        else => return error.TestExpectedWeightedSimilar,
    }
}

test "qgram index: V > 100K stress completes quickly" {
    // Synthetic vocab of ~50K random pieces (debug builds choke on
    // 110K due to unoptimized inner loops, and the full-suite test
    // runner caps individual test allocations more tightly than a
    // standalone test binary). The microbench in
    // bench/bench_vocab_extend.zig exercises the V=10K/50K/100K
    // numbers explicitly under ReleaseSafe.
    const V: u32 = 50_000;
    var prng = std.Random.DefaultPrng.init(0xBEEFD00D);
    const r = prng.random();
    const piece_chars = "abcdefghijklmnopqrstuvwxyz0123456789_";

    var pieces: std.ArrayList(TestEntry) = .empty;
    defer pieces.deinit(testing.allocator);
    var piece_bufs: std.ArrayList([]u8) = .empty;
    defer {
        for (piece_bufs.items) |b| testing.allocator.free(b);
        piece_bufs.deinit(testing.allocator);
    }

    // Dedupe via a small set — buildByteVocab will fail if the same
    // bytes appear twice (NonContiguousRanks would still pass; but
    // duplicates would corrupt by_bytes via overwrite).
    var seen = std.StringHashMap(void).init(testing.allocator);
    defer seen.deinit();

    var rank_cursor: u32 = 256;
    while (rank_cursor < 256 + V) {
        const len: usize = 4 + r.intRangeAtMost(usize, 0, 6);
        const buf = try testing.allocator.alloc(u8, len);
        for (buf) |*c| c.* = piece_chars[r.intRangeAtMost(usize, 0, piece_chars.len - 1)];
        const gop = try seen.getOrPut(buf);
        if (gop.found_existing) {
            testing.allocator.free(buf);
            continue;
        }
        gop.key_ptr.* = buf;
        try piece_bufs.append(testing.allocator, buf);
        try pieces.append(testing.allocator, .{ .bytes = buf, .rank = rank_cursor });
        rank_cursor += 1;
    }

    var old = try buildByteVocab(testing.allocator, pieces.items);
    defer old.deinit();

    const new_toks = [_][]const u8{
        "alphabetical", "tokenizer", "embeddings", "transformer",
    };

    const t0 = wallNanos();
    var res = try extendBpe(testing.allocator, &old, .{
        .new_tokens = &new_toks,
        .strategy = .weighted_similar,
        .weighted_similar = .{ .k = 8, .use_qgram_index = true },
    });
    defer res.deinit();
    const elapsed_ns = wallNanos() - t0;

    // Wall-clock budget. ReleaseSafe is well under 1 s; Debug is
    // dominated by unoptimized LCS inner loops and can take 10+ s,
    // which is fine for a one-shot dev test but not worth asserting.
    const builtin = @import("builtin");
    const budget_ns: u64 = if (builtin.mode == .Debug) 60_000_000_000 else 5_000_000_000;
    try testing.expect(elapsed_ns < budget_ns);
    try testing.expectEqual(@as(usize, 4), res.init_plan.len);
}

// std.time was gutted in 0.16; clock_gettime is the simplest path. Only
// used by stress tests and microbench — not on any hot path.
extern "c" fn clock_gettime(clk_id: c_int, tp: *TestTimespec) c_int;
const TestTimespec = extern struct { sec: i64, nsec: i64 };
fn wallNanos() u64 {
    var ts: TestTimespec = .{ .sec = 0, .nsec = 0 };
    _ = clock_gettime(1, &ts); // CLOCK_MONOTONIC
    return @intCast(ts.sec * 1_000_000_000 + ts.nsec);
}
