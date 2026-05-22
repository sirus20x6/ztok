//! Unigram language model encoder. Viterbi over input bytes against a flat
//! trie of vocab pieces. Layout mirrors `vocab.zig` (SoA, no per-token
//! allocations). Trie nodes live in one slice, children entries in another;
//! NodeId is a u32 index. Children are stored sorted by byte so prefix walks
//! use a binary search — vocab is fixed after `finalize()`, so the cost is
//! paid once.
//!
//! `encodeChunk` returns the single best segmentation (argmax). `sampleChunk`
//! draws a random segmentation from the lattice using forward-filter /
//! backward-sample, controlled by `alpha` (Kudo 2018 subword regularization).
//!
//! Viterbi mirrors `refs/sentencepiece/src/unigram_model.cc::Encode`:
//!   * `unk_score = min_score - kUnkPenalty` (10.0) — guarantees unk only
//!     wins when no real piece covers the position. Using the literal
//!     `scores[unk_id]` (often 0.0 in T5/LLaMA SP exports) instead would
//!     make unk strictly better than any normal piece (which carry
//!     negative scores) and collapse the lattice into all-unk.
//!   * Position advances by one UTF-8 codepoint (`OneCharLen`) when the
//!     unk / byte-fallback edge fires, so the segmentation aligns with
//!     SP's character-level lattice. Within a position the trie walk
//!     still operates on raw bytes so multi-byte pieces match exactly.
//!   * When `byte_fallback` is non-null (set by `sp_bridge.unigramFromSP`
//!     for SP models with `trainer_spec.byte_fallback`), the fallback
//!     edge replaces the unk edge for the SINGLE byte at the current
//!     position, advancing one byte and emitting the dedicated byte
//!     token id. Each byte in the lattice gets its own fallback edge so
//!     a multi-byte unknown sequence emits one byte-fallback id per byte
//!     (matching SP's `<0xNN>` expansion behavior).

const std = @import("std");
const TokenId = @import("token.zig").TokenId;
const Span = @import("token.zig").Span;

const NO_TOKEN: u32 = 0xFFFF_FFFF;
const NEG_INF: f32 = -std.math.inf(f32);

/// Penalty added on top of the model's minimum piece score when computing
/// the unk_score for the Viterbi fallback edge. Mirrors SP's
/// `kUnkPenalty = 10.0` (unigram_model.cc:42). The constant is what
/// guarantees unk only wins when nothing else covers the position —
/// raising it makes unk-emit rarer; lowering it makes it more common.
pub const K_UNK_PENALTY: f32 = 10.0;

/// Returns the UTF-8 encoded length (1..4) for the codepoint starting at
/// `b`. Treats invalid leading bytes (10xxxxxx continuation) as length 1
/// so the encoder can still advance past garbage instead of looping.
/// Matches SP's `OneCharLen` in `string_util.h`.
inline fn oneCharLen(b: u8) u8 {
    if (b < 0x80) return 1;
    if (b < 0xC0) return 1; // stray continuation byte — make forward progress
    if (b < 0xE0) return 2;
    if (b < 0xF0) return 3;
    return 4;
}

inline fn logSumExp(a: f32, b: f32) f32 {
    if (a == NEG_INF) return b;
    if (b == NEG_INF) return a;
    const hi = @max(a, b);
    const lo = @min(a, b);
    if (hi - lo > 50.0) return hi;
    return hi + @log(1.0 + @exp(lo - hi));
}

// Gumbel(0,1) sample from a uniform draw. Clamp away from 0 to avoid log(0).
inline fn gumbel(rng: std.Random) f32 {
    const u_raw = rng.float(f32);
    const u = if (u_raw < 1e-20) @as(f32, 1e-20) else u_raw;
    return -@log(-@log(u));
}

pub const Unigram = struct {
    allocator: std.mem.Allocator,
    bytes: []u8,
    offsets: []u32,
    scores: []f32,
    count: u32,
    unk_id: TokenId,
    trie_nodes: []TrieNode,
    trie_children: []TrieChild,
    bos_id: ?TokenId = null,
    eos_id: ?TokenId = null,

    /// Cached minimum score across all (real) pieces in `scores`. Used
    /// to derive the Viterbi unk fallback cost as
    /// `min_score - K_UNK_PENALTY`. Populated by `Builder.finalize` from
    /// the actual piece scores; defaults to 0.0 when the vocab is empty
    /// (then unk_score collapses to `-K_UNK_PENALTY`, still strictly
    /// worse than the start-of-path score of 0.0).
    min_score: f32 = 0.0,

    /// Optional byte-fallback table: byte value -> token id. Populated by
    /// SP-Unigram loaders when the source model has
    /// `trainer_spec.byte_fallback` and dedicated `<0xNN>` byte tokens
    /// in the vocab. When non-null, the Viterbi DP adds a fallback edge
    /// at every position from byte `p` to byte `p+1` whose cost is
    /// `min_score - K_UNK_PENALTY` and whose emitted id is
    /// `byte_fallback[input[p]]`. The edge is added in addition to any
    /// piece matches at that position, so a piece covering the byte (or
    /// any prefix starting from it) still wins on score. Unmatched
    /// bytes — including the bytes inside an unknown multi-byte UTF-8
    /// sequence — fall through to the byte-fallback id one byte at a
    /// time, matching SP's `<0xNN>` expansion behavior.
    ///
    /// `null` for non-SP Unigram vocabs or SP vocabs without
    /// `byte_fallback`; in that case the encoder uses the legacy
    /// per-codepoint unk edge so existing callers stay bit-identical.
    byte_fallback: ?[256]TokenId = null,

    pub const TrieNode = struct {
        token_id: u32,
        children_start: u32,
        children_len: u32,
    };

    pub const TrieChild = struct {
        byte: u8,
        node: u32,
    };

    pub const Builder = struct {
        allocator: std.mem.Allocator,
        bytes_buf: std.ArrayList(u8),
        offsets_buf: std.ArrayList(u32),
        scores_buf: std.ArrayList(f32),
        /// Token ids that occupy a vocab slot but must NOT be matched as
        /// text by the trie walk. SP-derived loaders flag pieces of type
        /// `.unknown`, `.byte`, `.control`, and `.unused` here so they
        /// keep stable ids (downstream code may look up the byte image
        /// via `idBytes`) without competing with real merges during
        /// Viterbi. Owned by the builder; consumed by `finalize`.
        excluded_from_trie: std.ArrayList(TokenId),

        pub fn init(allocator: std.mem.Allocator) Builder {
            return .{
                .allocator = allocator,
                .bytes_buf = .empty,
                .offsets_buf = .empty,
                .scores_buf = .empty,
                .excluded_from_trie = .empty,
            };
        }

        pub fn deinit(self: *Builder) void {
            self.bytes_buf.deinit(self.allocator);
            self.offsets_buf.deinit(self.allocator);
            self.scores_buf.deinit(self.allocator);
            self.excluded_from_trie.deinit(self.allocator);
        }

        pub fn addToken(self: *Builder, bytes: []const u8, score: f32) !TokenId {
            if (self.offsets_buf.items.len == 0) {
                try self.offsets_buf.append(self.allocator, 0);
            }
            const id: TokenId = @intCast(self.scores_buf.items.len);
            try self.bytes_buf.appendSlice(self.allocator, bytes);
            try self.offsets_buf.append(self.allocator, @intCast(self.bytes_buf.items.len));
            try self.scores_buf.append(self.allocator, score);
            return id;
        }

        /// Mark a previously-added id as excluded from the trie. Use for
        /// SP control/unknown/byte/unused pieces whose byte image must
        /// never be matched by the Viterbi walk (otherwise they win
        /// score-0 ties against real merges — see the t5_unigram code-
        /// corpus regression where `<unk>` inside text would otherwise
        /// resolve to unk-id instead of `<`, `unk`, `>`).
        pub fn excludeFromTrie(self: *Builder, id: TokenId) !void {
            try self.excluded_from_trie.append(self.allocator, id);
        }

        pub fn finalize(self: *Builder, unk_id: TokenId) !Unigram {
            const count: u32 = @intCast(self.scores_buf.items.len);
            std.debug.assert(count == 0 or unk_id < count);

            // Move SoA buffers into the new struct; clear the builder's
            // arrays so deinit() is a no-op for them afterwards.
            const bytes = try self.bytes_buf.toOwnedSlice(self.allocator);
            errdefer self.allocator.free(bytes);
            const offsets_initial = if (self.offsets_buf.items.len == 0) blk: {
                // No tokens were added; synthesize a single offset of 0.
                var tmp = try self.allocator.alloc(u32, 1);
                tmp[0] = 0;
                break :blk tmp;
            } else try self.offsets_buf.toOwnedSlice(self.allocator);
            errdefer self.allocator.free(offsets_initial);
            const scores = try self.scores_buf.toOwnedSlice(self.allocator);
            errdefer self.allocator.free(scores);

            // Build the trie skipping any ids the caller flagged via
            // `excludeFromTrie`. Empty exclusion list collapses to the
            // legacy "every id is indexed" path.
            const excl = self.excluded_from_trie.items;
            const trie = try buildTrieExcluding(self.allocator, bytes, offsets_initial, count, excl);
            self.excluded_from_trie.deinit(self.allocator);
            self.excluded_from_trie = .empty;

            // Cache the min piece score for unk fallback edges.
            // Specials (unk/bos/eos/pad) sit at score 0.0 in T5/LLaMA
            // vocabs; including them would push min_score up to 0 and
            // collapse unk_score to -K_UNK_PENALTY, which is still less
            // than any real piece's negative score so the behavior is
            // unaffected — but we exclude unk_id explicitly anyway for
            // clarity, since SP's unigram_model.cc:CalcMinScore loops
            // over all pieces and the equivalence is moot.
            var min_score: f32 = 0.0;
            if (count > 0) {
                var k: u32 = 0;
                var seen: bool = false;
                while (k < count) : (k += 1) {
                    if (k == unk_id) continue;
                    const s = scores[k];
                    if (!seen or s < min_score) {
                        min_score = s;
                        seen = true;
                    }
                }
                if (!seen) {
                    // Only unk in the vocab; fall back to its score.
                    min_score = scores[unk_id];
                }
            }

            return .{
                .allocator = self.allocator,
                .bytes = bytes,
                .offsets = offsets_initial,
                .scores = scores,
                .count = count,
                .unk_id = unk_id,
                .trie_nodes = trie.nodes,
                .trie_children = trie.children,
                .min_score = min_score,
            };
        }
    };

    pub fn deinit(self: *Unigram) void {
        if (self.bytes.len > 0) self.allocator.free(self.bytes);
        if (self.offsets.len > 0) self.allocator.free(self.offsets);
        if (self.scores.len > 0) self.allocator.free(self.scores);
        if (self.trie_nodes.len > 0) self.allocator.free(self.trie_nodes);
        if (self.trie_children.len > 0) self.allocator.free(self.trie_children);
        self.bytes = &.{};
        self.offsets = &.{};
        self.scores = &.{};
        self.trie_nodes = &.{};
        self.trie_children = &.{};
        self.count = 0;
        self.min_score = 0.0;
        self.byte_fallback = null;
    }

    pub fn idBytes(self: *const Unigram, id: TokenId) []const u8 {
        std.debug.assert(id < self.count);
        const start = self.offsets[id];
        const end = self.offsets[id + 1];
        return self.bytes[start..end];
    }

    pub fn encodeChunk(
        self: *const Unigram,
        allocator: std.mem.Allocator,
        chunk: []const u8,
        out: []TokenId,
    ) ![]TokenId {
        std.debug.assert(out.len >= chunk.len);
        if (chunk.len == 0) return out[0..0];

        const n = chunk.len;
        // Cumulative Viterbi scores are accumulated in f64. The stored
        // piece scores are f32 (matches the on-disk model precision),
        // but the running sum needs the extra mantissa: HF tokenizers
        // and SentencePiece both use f64 here, and on near-tie paths
        // (e.g. "4|44" vs "44|4" where the two paths differ only in
        // edge-ordering) f32 accumulation can flip the strict `>`
        // tie-break and diverge by one token. See the multilingual
        // corpus regression in `bench/equivalence_check.py` (Russian
        // line with trailing "=444)").
        const best_score = try allocator.alloc(f64, n + 1);
        defer allocator.free(best_score);
        const best_prev = try allocator.alloc(u32, n + 1);
        defer allocator.free(best_prev);
        const best_id = try allocator.alloc(u32, n + 1);
        defer allocator.free(best_id);

        // Mirror SP's BestPathNode sentinel pattern: `starts_at == -1`
        // means "no path yet". We use `best_prev[p] == NO_TOKEN` as the
        // equivalent — fresh positions look unreached, and the unk edge
        // takes 0 as the prior score (not NEG_INF) so a brand-new path
        // can spawn from a position the trie walk couldn't reach.
        @memset(best_score, 0.0);
        @memset(best_prev, NO_TOKEN);
        @memset(best_id, self.unk_id);
        best_score[0] = 0.0;
        best_prev[0] = 0;
        best_id[0] = NO_TOKEN;

        // SP's effective unk cost — `min_score - 10.0` — keeps unk
        // strictly worse than any real piece. Empty vocab still produces
        // a valid (sentinel) cost so the DP doesn't degenerate. Promoted
        // to f64 to match the cumulative arithmetic.
        const unk_score: f64 = if (self.count == 0) @as(f64, NEG_INF) else @as(f64, self.min_score) - @as(f64, K_UNK_PENALTY);
        const bf = self.byte_fallback;

        // Iterate `starts_at` by codepoints (mblen), matching
        // SP's `starts_at += mblen` outer loop. Inside a codepoint we
        // never start new lattice nodes — only complete codepoint
        // boundaries are first-class lattice positions.
        var i: usize = 0;
        while (i < n) {
            const mblen = @min(@as(usize, oneCharLen(chunk[i])), n - i);

            // Walk trie from root from position i, recording every
            // terminal hit at end = i + 1 .. i + max_piece_len.
            var has_single_node = false;
            var node_idx: u32 = 0;
            var k: usize = 0;
            while (i + k < n) : (k += 1) {
                const child = self.findChild(node_idx, chunk[i + k]) orelse break;
                node_idx = child;
                const node = self.trie_nodes[node_idx];
                if (node.token_id != NO_TOKEN) {
                    const end = i + k + 1;
                    const cand: f64 = best_score[i] + @as(f64, self.scores[node.token_id]);
                    if (best_prev[end] == NO_TOKEN or cand > best_score[end]) {
                        best_score[end] = cand;
                        best_prev[end] = @intCast(i);
                        best_id[end] = node.token_id;
                    }
                    if (k + 1 == mblen) has_single_node = true;
                }
            }

            // No piece covered the leading codepoint — emit unk (or one
            // byte-fallback id) at this position. SP advances by mblen
            // here regardless; byte-fallback advances by 1 byte at a
            // time so each byte of an unknown multi-byte UTF-8 sequence
            // becomes its own `<0xNN>` id.
            if (!has_single_node) {
                if (bf) |table| {
                    var b: usize = 0;
                    while (b < mblen) : (b += 1) {
                        const src = i + b;
                        const end = src + 1;
                        const cand: f64 = best_score[src] + unk_score;
                        if (best_prev[end] == NO_TOKEN or cand > best_score[end]) {
                            best_score[end] = cand;
                            best_prev[end] = @intCast(src);
                            best_id[end] = table[chunk[src]];
                        }
                    }
                } else {
                    const end = i + mblen;
                    const cand: f64 = best_score[i] + unk_score;
                    if (best_prev[end] == NO_TOKEN or cand > best_score[end]) {
                        best_score[end] = cand;
                        best_prev[end] = @intCast(i);
                        best_id[end] = self.unk_id;
                    }
                }
            }

            i += mblen;
        }

        // Walk back from n to 0 recovering ids in reverse, then flip.
        // Without byte-fallback, SP's sentencepiece_processor merges
        // consecutive unk pieces into one (so a 3-codepoint unknown run
        // emits a single `<unk>` id instead of three). With
        // byte-fallback, every byte gets its own `<0xNN>` id so no
        // merging occurs. We mirror that by suppressing emission of an
        // unk id when the previous (reverse-order) id was also unk.
        var count_out: usize = 0;
        var pos: usize = n;
        var prev_unk = false;
        while (pos > 0) {
            std.debug.assert(count_out < out.len);
            const id = best_id[pos];
            const is_unk = (bf == null) and (id == self.unk_id);
            if (!(prev_unk and is_unk)) {
                out[count_out] = id;
                count_out += 1;
            }
            prev_unk = is_unk;
            const next = best_prev[pos];
            // Either we found a path back or we landed on an
            // unreached final position (only possible for empty input
            // segments after special tokens — guard so we don't loop).
            if (next == NO_TOKEN or next >= pos) break;
            pos = next;
        }
        std.mem.reverse(TokenId, out[0..count_out]);
        return out[0..count_out];
    }

    /// Same as encodeChunk but also writes one Span per emitted id into
    /// `out_offsets`. Offsets are relative to the buffer the encoder's
    /// caller indexes into; `chunk_offset` is added to each piece's start
    /// to translate from chunk-local to buffer-global coordinates.
    /// Returns the number of ids written.
    pub fn encodeChunkWithOffsets(
        self: *const Unigram,
        allocator: std.mem.Allocator,
        chunk: []const u8,
        chunk_offset: u32,
        out_ids: []TokenId,
        out_offsets: []Span,
    ) !usize {
        std.debug.assert(out_ids.len >= chunk.len);
        std.debug.assert(out_offsets.len >= chunk.len);
        if (chunk.len == 0) return 0;

        const n = chunk.len;
        // f64 cumulative — see `encodeChunk` for the rationale.
        const best_score = try allocator.alloc(f64, n + 1);
        defer allocator.free(best_score);
        const best_prev = try allocator.alloc(u32, n + 1);
        defer allocator.free(best_prev);
        const best_id = try allocator.alloc(u32, n + 1);
        defer allocator.free(best_id);

        @memset(best_score, 0.0);
        @memset(best_prev, NO_TOKEN);
        @memset(best_id, self.unk_id);
        best_score[0] = 0.0;
        best_prev[0] = 0;
        best_id[0] = NO_TOKEN;

        // See `encodeChunk` for the rationale on `min_score - K_UNK_PENALTY`
        // (mirrors SP's `kUnkPenalty`).
        const unk_score: f64 = if (self.count == 0) @as(f64, NEG_INF) else @as(f64, self.min_score) - @as(f64, K_UNK_PENALTY);
        const bf = self.byte_fallback;

        var i: usize = 0;
        while (i < n) {
            const mblen = @min(@as(usize, oneCharLen(chunk[i])), n - i);
            var has_single_node = false;
            var node_idx: u32 = 0;
            var k: usize = 0;
            while (i + k < n) : (k += 1) {
                const child = self.findChild(node_idx, chunk[i + k]) orelse break;
                node_idx = child;
                const node = self.trie_nodes[node_idx];
                if (node.token_id != NO_TOKEN) {
                    const end = i + k + 1;
                    const cand: f64 = best_score[i] + @as(f64, self.scores[node.token_id]);
                    if (best_prev[end] == NO_TOKEN or cand > best_score[end]) {
                        best_score[end] = cand;
                        best_prev[end] = @intCast(i);
                        best_id[end] = node.token_id;
                    }
                    if (k + 1 == mblen) has_single_node = true;
                }
            }

            if (!has_single_node) {
                if (bf) |table| {
                    var b: usize = 0;
                    while (b < mblen) : (b += 1) {
                        const src = i + b;
                        const end = src + 1;
                        const cand: f64 = best_score[src] + unk_score;
                        if (best_prev[end] == NO_TOKEN or cand > best_score[end]) {
                            best_score[end] = cand;
                            best_prev[end] = @intCast(src);
                            best_id[end] = table[chunk[src]];
                        }
                    }
                } else {
                    const end = i + mblen;
                    const cand: f64 = best_score[i] + unk_score;
                    if (best_prev[end] == NO_TOKEN or cand > best_score[end]) {
                        best_score[end] = cand;
                        best_prev[end] = @intCast(i);
                        best_id[end] = self.unk_id;
                    }
                }
            }

            i += mblen;
        }

        // Walk back; record id + span (start = best_prev[pos], end = pos).
        // Consecutive unks coalesce into one (matches SP without
        // byte-fallback). When merging, the span extends leftward to
        // cover the entire run.
        var count_out: usize = 0;
        var pos: usize = n;
        var prev_unk = false;
        while (pos > 0) {
            std.debug.assert(count_out < out_ids.len);
            const start_raw: u32 = best_prev[pos];
            const end_u32: u32 = @intCast(pos);
            const id = best_id[pos];
            const is_unk = (bf == null) and (id == self.unk_id);

            if (prev_unk and is_unk) {
                // Extend the previous (reverse-order) span leftward
                // through this unk's bytes — i.e. drop its left edge.
                const last_idx = count_out - 1;
                out_offsets[last_idx].start = chunk_offset + start_raw;
            } else {
                out_ids[count_out] = id;
                out_offsets[count_out] = .{
                    .start = chunk_offset + start_raw,
                    .end = chunk_offset + end_u32,
                };
                count_out += 1;
            }
            prev_unk = is_unk;
            if (start_raw == NO_TOKEN or start_raw >= pos) break;
            pos = start_raw;
        }
        std.mem.reverse(TokenId, out_ids[0..count_out]);
        std.mem.reverse(Span, out_offsets[0..count_out]);
        return count_out;
    }

    // -------- trace variants ---------------------------------------------

    /// Same shape as `encodeChunk` plus a non-null `trace` sink. After
    /// the Viterbi DP runs to completion, each backtracked piece emits
    /// `unigram pos=… piece_id=… piece=… score=…` in the order ids
    /// appear in the final output (left to right). Pieces coalesced
    /// via the consecutive-unk merge rule emit ONE record covering
    /// the merged span.
    pub fn encodeChunkTrace(
        self: *const Unigram,
        allocator: std.mem.Allocator,
        chunk: []const u8,
        out: []TokenId,
        trace: *@import("trace.zig").Trace,
    ) ![]TokenId {
        std.debug.assert(out.len >= chunk.len);
        if (chunk.len == 0) return out[0..0];

        const n = chunk.len;
        // f64 cumulative — see `encodeChunk` for the rationale.
        const best_score = try allocator.alloc(f64, n + 1);
        defer allocator.free(best_score);
        const best_prev = try allocator.alloc(u32, n + 1);
        defer allocator.free(best_prev);
        const best_id = try allocator.alloc(u32, n + 1);
        defer allocator.free(best_id);

        @memset(best_score, 0.0);
        @memset(best_prev, NO_TOKEN);
        @memset(best_id, self.unk_id);
        best_score[0] = 0.0;
        best_prev[0] = 0;
        best_id[0] = NO_TOKEN;

        const unk_score: f64 = if (self.count == 0) @as(f64, NEG_INF) else @as(f64, self.min_score) - @as(f64, K_UNK_PENALTY);
        const bf = self.byte_fallback;

        var i: usize = 0;
        while (i < n) {
            const mblen = @min(@as(usize, oneCharLen(chunk[i])), n - i);
            var has_single_node = false;
            var node_idx: u32 = 0;
            var k: usize = 0;
            while (i + k < n) : (k += 1) {
                const child = self.findChild(node_idx, chunk[i + k]) orelse break;
                node_idx = child;
                const node = self.trie_nodes[node_idx];
                if (node.token_id != NO_TOKEN) {
                    const end = i + k + 1;
                    const cand: f64 = best_score[i] + @as(f64, self.scores[node.token_id]);
                    if (best_prev[end] == NO_TOKEN or cand > best_score[end]) {
                        best_score[end] = cand;
                        best_prev[end] = @intCast(i);
                        best_id[end] = node.token_id;
                    }
                    if (k + 1 == mblen) has_single_node = true;
                }
            }

            if (!has_single_node) {
                if (bf) |table| {
                    var b: usize = 0;
                    while (b < mblen) : (b += 1) {
                        const src = i + b;
                        const end = src + 1;
                        const cand: f64 = best_score[src] + unk_score;
                        if (best_prev[end] == NO_TOKEN or cand > best_score[end]) {
                            best_score[end] = cand;
                            best_prev[end] = @intCast(src);
                            best_id[end] = table[chunk[src]];
                        }
                    }
                } else {
                    const end = i + mblen;
                    const cand: f64 = best_score[i] + unk_score;
                    if (best_prev[end] == NO_TOKEN or cand > best_score[end]) {
                        best_score[end] = cand;
                        best_prev[end] = @intCast(i);
                        best_id[end] = self.unk_id;
                    }
                }
            }

            i += mblen;
        }

        // Backtrack: record id + (start, end) pairs in reverse, then
        // emit trace records in forward order. We need the start/end
        // pairs to render the piece bytes for the trace stream.
        var count_out: usize = 0;
        var starts_buf = try allocator.alloc(u32, n + 1);
        defer allocator.free(starts_buf);
        var ends_buf = try allocator.alloc(u32, n + 1);
        defer allocator.free(ends_buf);
        var scores_buf = try allocator.alloc(f32, n + 1);
        defer allocator.free(scores_buf);

        var pos: usize = n;
        var prev_unk = false;
        while (pos > 0) {
            std.debug.assert(count_out < out.len);
            const id = best_id[pos];
            const is_unk = (bf == null) and (id == self.unk_id);
            const prev_start = best_prev[pos];
            const piece_score: f32 = if (id == self.unk_id) @floatCast(unk_score) else self.scores[id];
            if (prev_unk and is_unk) {
                // Extend the previous (reverse-order) entry leftward.
                starts_buf[count_out - 1] = prev_start;
            } else {
                out[count_out] = id;
                starts_buf[count_out] = prev_start;
                ends_buf[count_out] = @intCast(pos);
                scores_buf[count_out] = piece_score;
                count_out += 1;
            }
            prev_unk = is_unk;
            if (prev_start == NO_TOKEN or prev_start >= pos) break;
            pos = prev_start;
        }
        std.mem.reverse(TokenId, out[0..count_out]);
        std.mem.reverse(u32, starts_buf[0..count_out]);
        std.mem.reverse(u32, ends_buf[0..count_out]);
        std.mem.reverse(f32, scores_buf[0..count_out]);

        // Emit trace records left-to-right in the order they appear in
        // the final encoding.
        var t: usize = 0;
        while (t < count_out) : (t += 1) {
            const s = starts_buf[t];
            const e = ends_buf[t];
            try trace.unigramPiece(
                @intCast(s),
                out[t],
                chunk[s..e],
                scores_buf[t],
            );
        }
        return out[0..count_out];
    }

    /// Offsets-emitting trace variant.
    pub fn encodeChunkWithOffsetsTrace(
        self: *const Unigram,
        allocator: std.mem.Allocator,
        chunk: []const u8,
        chunk_offset: u32,
        out_ids: []TokenId,
        out_offsets: []Span,
        trace: *@import("trace.zig").Trace,
    ) !usize {
        // For the trace variant we delegate to the non-trace offsets
        // path AFTER emitting trace records via the simpler trace path.
        // The two paths agree by construction (same DP); the trace path
        // walks the same lattice and produces the same id sequence.
        const ids_scratch = try allocator.alloc(TokenId, chunk.len + 1);
        defer allocator.free(ids_scratch);
        _ = try self.encodeChunkTrace(allocator, chunk, ids_scratch, trace);
        return self.encodeChunkWithOffsets(allocator, chunk, chunk_offset, out_ids, out_offsets);
    }

    /// Sample a segmentation from the lattice instead of taking the best.
    /// `alpha > 0` controls sharpness: 1.0 = sample proportional to prob,
    /// 0.1 = nearly greedy, 10.0 = nearly uniform.
    /// `rng` is a `std.Random` pointer for reproducibility.
    pub fn sampleChunk(
        self: *const Unigram,
        allocator: std.mem.Allocator,
        chunk: []const u8,
        out: []TokenId,
        alpha: f32,
        rng: std.Random,
    ) ![]TokenId {
        std.debug.assert(out.len >= chunk.len);
        if (chunk.len == 0) return out[0..0];

        // alpha <= 0 collapses to greedy; mirror encodeChunk exactly.
        if (!(alpha > 0.0)) return self.encodeChunk(allocator, chunk, out);

        const n = chunk.len;
        const inv_alpha: f32 = 1.0 / alpha;
        // Same unk-score derivation as encodeChunk (`min_score - K_UNK_PENALTY`).
        const unk_score: f32 = if (self.count == 0) NEG_INF else self.min_score - K_UNK_PENALTY;
        const bf = self.byte_fallback;

        // Forward log-probabilities: fwd[e] = log sum over all segmentations
        // of chunk[0..e]. fwd[0] = 0.
        const fwd = try allocator.alloc(f32, n + 1);
        defer allocator.free(fwd);
        for (fwd) |*v| v.* = NEG_INF;
        fwd[0] = 0.0;

        // Collect all (start, end, id, score) matches during the forward
        // pass and build a per-end-position index for the backward sample.
        var matches: std.ArrayList(Match) = .empty;
        defer matches.deinit(allocator);

        // Codepoint-aware outer loop. An unreached position (`fwd[i] ==
        // NEG_INF`) is treated as the start of a fresh path with prior
        // score 0 — SP's `best_path_ends_at[i].best_path_score = 0`
        // default semantics. The sampler isn't called for cross-tokenizer
        // parity (encodeChunk owns that path); keeping its lattice shape
        // consistent with encodeChunk avoids surprises for callers who
        // sample/encode the same chunk.
        var i: usize = 0;
        while (i < n) {
            const fi: f32 = if (fwd[i] == NEG_INF) 0.0 else fwd[i];
            const mblen = @min(@as(usize, oneCharLen(chunk[i])), n - i);
            var has_single_node = false;
            var node_idx: u32 = 0;
            var k: usize = 0;
            while (i + k < n) : (k += 1) {
                const child = self.findChild(node_idx, chunk[i + k]) orelse break;
                node_idx = child;
                const node = self.trie_nodes[node_idx];
                if (node.token_id != NO_TOKEN) {
                    const end = i + k + 1;
                    const sc = self.scores[node.token_id];
                    fwd[end] = logSumExp(fwd[end], fi + sc);
                    try matches.append(allocator, .{
                        .start = @intCast(i),
                        .end = @intCast(end),
                        .id = node.token_id,
                        .score = sc,
                    });
                    if (k + 1 == mblen) has_single_node = true;
                }
            }
            if (!has_single_node) {
                if (bf) |table| {
                    var b: usize = 0;
                    while (b < mblen) : (b += 1) {
                        const src = i + b;
                        const end = src + 1;
                        const fsrc: f32 = if (fwd[src] == NEG_INF) 0.0 else fwd[src];
                        fwd[end] = logSumExp(fwd[end], fsrc + unk_score);
                        try matches.append(allocator, .{
                            .start = @intCast(src),
                            .end = @intCast(end),
                            .id = table[chunk[src]],
                            .score = unk_score,
                        });
                    }
                } else {
                    fwd[i + mblen] = logSumExp(fwd[i + mblen], fi + unk_score);
                    try matches.append(allocator, .{
                        .start = @intCast(i),
                        .end = @intCast(i + mblen),
                        .id = self.unk_id,
                        .score = unk_score,
                    });
                }
            }
            i += mblen;
        }

        // Index matches by end position via counting sort. After the prefix
        // sum, matches with `m.end == p` live in `by_end[ends_off[p]..ends_off[p+1]]`.
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
        const by_end = try allocator.alloc(Match, matches.items.len);
        defer allocator.free(by_end);
        // Use a transient cursor copy to know where to place each match.
        const cursor = try allocator.alloc(u32, n + 2);
        defer allocator.free(cursor);
        @memcpy(cursor, ends_off);
        for (matches.items) |m| {
            const idx = cursor[m.end];
            by_end[idx] = m;
            cursor[m.end] = idx + 1;
        }

        // Backward sample: at position p, look at every match (s, p, id, sc).
        // Unnormalized log-weight = (fwd[s] + sc) / alpha. Gumbel-max sample.
        // An NEG_INF prior score is replaced by 0 (start-of-fresh-path)
        // so unreached positions still produce a valid weight.
        var count_out: usize = 0;
        var pos: usize = n;
        while (pos > 0) {
            const lo = ends_off[pos];
            const hi = ends_off[pos + 1];
            std.debug.assert(hi > lo); // forward pass guarantees ≥1 match
            var best_g: f32 = NEG_INF;
            var best_idx: u32 = lo;
            var j: u32 = lo;
            while (j < hi) : (j += 1) {
                const m = by_end[j];
                const s_prev_raw = fwd[m.start];
                const s_prev: f32 = if (s_prev_raw == NEG_INF) 0.0 else s_prev_raw;
                const lw = (s_prev + m.score) * inv_alpha;
                const g = lw + gumbel(rng);
                if (g > best_g) {
                    best_g = g;
                    best_idx = j;
                }
            }
            const chosen = by_end[best_idx];
            std.debug.assert(count_out < out.len);
            out[count_out] = chosen.id;
            count_out += 1;
            pos = chosen.start;
        }
        std.mem.reverse(TokenId, out[0..count_out]);
        return out[0..count_out];
    }

    fn findChild(self: *const Unigram, node_idx: u32, byte: u8) ?u32 {
        const node = self.trie_nodes[node_idx];
        if (node.children_len == 0) return null;
        const lo = node.children_start;
        const hi = lo + node.children_len;
        var l: u32 = lo;
        var r: u32 = hi;
        while (l < r) {
            const m = l + (r - l) / 2;
            const b = self.trie_children[m].byte;
            if (b == byte) return self.trie_children[m].node;
            if (b < byte) l = m + 1 else r = m;
        }
        return null;
    }
};

// --- shared match record used by sampleChunk ---

const Match = struct {
    start: u32,
    end: u32,
    id: u32,
    score: f32,
};

// --- trie construction ---

const TrieBuild = struct {
    nodes: []Unigram.TrieNode,
    children: []Unigram.TrieChild,
};

// Intermediate node form with a growable child list. We collapse to flat
// SoA at the end so the hot path never chases a pointer.
const BuildNode = struct {
    token_id: u32,
    children: std.ArrayList(Unigram.TrieChild),
};

fn buildTrie(
    allocator: std.mem.Allocator,
    bytes: []const u8,
    offsets: []const u32,
    count: u32,
) !TrieBuild {
    return buildTrieExcluding(allocator, bytes, offsets, count, &.{});
}

fn buildTrieExcluding(
    allocator: std.mem.Allocator,
    bytes: []const u8,
    offsets: []const u32,
    count: u32,
    excluded: []const TokenId,
) !TrieBuild {
    var nodes: std.ArrayList(BuildNode) = .empty;
    defer {
        for (nodes.items) |*n| n.children.deinit(allocator);
        nodes.deinit(allocator);
    }
    try nodes.append(allocator, .{ .token_id = NO_TOKEN, .children = .empty });

    // O(count + excluded.len) excluded-id check via a small bitmap.
    // Excluded list size is at most a few hundred entries (SP control +
    // user_defined counts), so the bitmap-per-bit cost is negligible
    // versus the trie nodes themselves.
    var excl_mask: ?[]u64 = null;
    defer if (excl_mask) |m| allocator.free(m);
    if (excluded.len > 0 and count > 0) {
        const words = (@as(usize, count) + 63) / 64;
        const m = try allocator.alloc(u64, words);
        @memset(m, 0);
        for (excluded) |eid| {
            if (eid < count) m[eid / 64] |= (@as(u64, 1) << @intCast(eid % 64));
        }
        excl_mask = m;
    }

    var id: u32 = 0;
    while (id < count) : (id += 1) {
        if (excl_mask) |m| {
            if ((m[id / 64] & (@as(u64, 1) << @intCast(id % 64))) != 0) continue;
        }
        const start = offsets[id];
        const end = offsets[id + 1];
        const piece = bytes[start..end];
        var cur: u32 = 0;
        for (piece) |b| {
            const next = try descendOrCreate(allocator, &nodes, cur, b);
            cur = next;
        }
        // Note: if two vocab pieces share bytes (shouldn't happen in a
        // well-formed vocab), last writer wins.
        nodes.items[cur].token_id = id;
    }

    // Flatten: each node's children become a contiguous slice in `children`.
    const node_count = nodes.items.len;
    var total_children: usize = 0;
    for (nodes.items) |n| total_children += n.children.items.len;

    const flat_nodes = try allocator.alloc(Unigram.TrieNode, node_count);
    errdefer allocator.free(flat_nodes);
    const flat_children = try allocator.alloc(Unigram.TrieChild, total_children);
    errdefer allocator.free(flat_children);

    var write: u32 = 0;
    for (nodes.items, 0..) |n, idx| {
        // Children inserted in sorted order by descendOrCreate, so we can
        // just copy. Binary search at lookup time relies on this.
        const len: u32 = @intCast(n.children.items.len);
        flat_nodes[idx] = .{
            .token_id = n.token_id,
            .children_start = write,
            .children_len = len,
        };
        @memcpy(flat_children[write .. write + len], n.children.items);
        write += len;
    }

    return .{ .nodes = flat_nodes, .children = flat_children };
}

fn descendOrCreate(
    allocator: std.mem.Allocator,
    nodes: *std.ArrayList(BuildNode),
    parent: u32,
    byte: u8,
) !u32 {
    // Sorted insert — keeps children array searchable without a post-sort.
    // NOTE: nodes.append below may reallocate nodes.items, so we cannot
    // hold a pointer into it across that call. Search up-front, then do
    // the new-node append, then re-look-up the children list by index.
    {
        const kids = nodes.items[parent].children.items;
        var l: usize = 0;
        var r: usize = kids.len;
        while (l < r) {
            const m = l + (r - l) / 2;
            const b = kids[m].byte;
            if (b == byte) return kids[m].node;
            if (b < byte) l = m + 1 else r = m;
        }
    }

    const new_idx: u32 = @intCast(nodes.items.len);
    try nodes.append(allocator, .{ .token_id = NO_TOKEN, .children = .empty });

    // Re-do the binary search post-realloc to find the insert position.
    const kids_ptr = &nodes.items[parent].children;
    var l2: usize = 0;
    var r2: usize = kids_ptr.items.len;
    while (l2 < r2) {
        const m = l2 + (r2 - l2) / 2;
        const b = kids_ptr.items[m].byte;
        if (b < byte) l2 = m + 1 else r2 = m;
    }
    try kids_ptr.insert(allocator, l2, .{ .byte = byte, .node = new_idx });
    return new_idx;
}

// --- tests ---

test "builder + idBytes round-trips" {
    var b = Unigram.Builder.init(std.testing.allocator);
    defer b.deinit();
    _ = try b.addToken("foo", -1.0);
    _ = try b.addToken("bar", -2.0);
    const unk = try b.addToken("<unk>", -10.0);
    var u = try b.finalize(unk);
    defer u.deinit();

    try std.testing.expectEqualStrings("foo", u.idBytes(0));
    try std.testing.expectEqualStrings("bar", u.idBytes(1));
    try std.testing.expectEqualStrings("<unk>", u.idBytes(2));
    try std.testing.expectEqual(@as(u32, 3), u.count);
    try std.testing.expectEqual(@as(TokenId, 2), u.unk_id);
}

test "encodeChunk picks higher-score segmentation" {
    var b = Unigram.Builder.init(std.testing.allocator);
    defer b.deinit();
    const a_id = try b.addToken("a", -2.0);
    const b_id = try b.addToken("b", -2.0);
    const ab_id = try b.addToken("ab", -1.0);
    const unk = try b.addToken("<unk>", -10.0);
    var u = try b.finalize(unk);
    defer u.deinit();

    _ = a_id;
    _ = b_id;
    var out: [4]TokenId = undefined;
    const ids = try u.encodeChunk(std.testing.allocator, "ab", &out);
    try std.testing.expectEqualSlices(TokenId, &.{ab_id}, ids);
}

test "encodeChunk uses unk for unknown bytes" {
    var b = Unigram.Builder.init(std.testing.allocator);
    defer b.deinit();
    const a_id = try b.addToken("a", -1.0);
    const unk = try b.addToken("<unk>", -10.0);
    var u = try b.finalize(unk);
    defer u.deinit();

    var out: [4]TokenId = undefined;
    const ids = try u.encodeChunk(std.testing.allocator, "az", &out);
    try std.testing.expectEqualSlices(TokenId, &.{ a_id, unk }, ids);
}

test "encodeChunk handles longer match preference" {
    // Spec note: vocab {a:-1, ab:-1, abc:-3} encoding "aab" — the only paths
    // that fully tokenize "aab" without UNK are a+ab (-2) and a+a+? (no
    // single-letter b without a 'b' token). a+ab wins over the alternative
    // a+UNK+UNK. Note: input is "aab" so we exercise the a→ab path; "abc"
    // wouldn't be reachable by a+ab.
    var b = Unigram.Builder.init(std.testing.allocator);
    defer b.deinit();
    const a_id = try b.addToken("a", -1.0);
    const ab_id = try b.addToken("ab", -1.0);
    _ = try b.addToken("abc", -3.0);
    const unk = try b.addToken("<unk>", -10.0);
    var u = try b.finalize(unk);
    defer u.deinit();

    var out: [4]TokenId = undefined;
    const ids = try u.encodeChunk(std.testing.allocator, "aab", &out);
    try std.testing.expectEqualSlices(TokenId, &.{ a_id, ab_id }, ids);
}

test "trie matches all prefixes" {
    var b = Unigram.Builder.init(std.testing.allocator);
    defer b.deinit();
    _ = try b.addToken("a", -1.0);
    _ = try b.addToken("ab", -1.0);
    _ = try b.addToken("abc", -1.0);
    const unk = try b.addToken("<unk>", -10.0);
    var u = try b.finalize(unk);
    defer u.deinit();

    // Walk trie manually: root -> 'a' -> 'b' -> 'c', each step terminal.
    const a_node = u.findChild(0, 'a') orelse return error.TestExpectedFound;
    try std.testing.expectEqual(@as(u32, 0), u.trie_nodes[a_node].token_id);

    const b_node = u.findChild(a_node, 'b') orelse return error.TestExpectedFound;
    try std.testing.expectEqual(@as(u32, 1), u.trie_nodes[b_node].token_id);

    const c_node = u.findChild(b_node, 'c') orelse return error.TestExpectedFound;
    try std.testing.expectEqual(@as(u32, 2), u.trie_nodes[c_node].token_id);

    // Negative lookup should miss.
    try std.testing.expect(u.findChild(0, 'z') == null);
}

test "sampleChunk with alpha=0.01 reproduces best path" {
    // Near-zero alpha sharpens the Gumbel-max sampler into argmax, so the
    // result should equal the Viterbi best path.
    var b = Unigram.Builder.init(std.testing.allocator);
    defer b.deinit();
    _ = try b.addToken("a", -2.0);
    _ = try b.addToken("b", -2.0);
    const ab_id = try b.addToken("ab", -1.0);
    const unk = try b.addToken("<unk>", -10.0);
    var u = try b.finalize(unk);
    defer u.deinit();

    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const rng = prng.random();
    var out_best: [4]TokenId = undefined;
    var out_samp: [4]TokenId = undefined;
    const best = try u.encodeChunk(std.testing.allocator, "ab", &out_best);
    var k: u32 = 0;
    while (k < 50) : (k += 1) {
        const samp = try u.sampleChunk(std.testing.allocator, "ab", &out_samp, 0.01, rng);
        try std.testing.expectEqualSlices(TokenId, best, samp);
        try std.testing.expectEqualSlices(TokenId, &.{ab_id}, samp);
    }
}

test "sampleChunk with alpha=1.0 produces varied output across calls" {
    // Vocab has equal-score competing segmentations of the same string.
    // 100 samples with different seeds should produce ≥2 distinct outputs.
    var b = Unigram.Builder.init(std.testing.allocator);
    defer b.deinit();
    _ = try b.addToken("a", -1.0);
    _ = try b.addToken("b", -1.0);
    _ = try b.addToken("ab", -2.0);
    const unk = try b.addToken("<unk>", -10.0);
    var u = try b.finalize(unk);
    defer u.deinit();

    var seen_a_a: bool = false;
    var seen_ab: bool = false;
    var seed: u64 = 1;
    while (seed <= 100) : (seed += 1) {
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        var out: [4]TokenId = undefined;
        const ids = try u.sampleChunk(std.testing.allocator, "ab", &out, 1.0, rng);
        if (ids.len == 2) seen_a_a = true;
        if (ids.len == 1) seen_ab = true;
        if (seen_a_a and seen_ab) break;
    }
    try std.testing.expect(seen_a_a and seen_ab);
}

test "sampleChunk respects vocab boundaries" {
    var b = Unigram.Builder.init(std.testing.allocator);
    defer b.deinit();
    _ = try b.addToken("a", -1.0);
    _ = try b.addToken("b", -1.0);
    _ = try b.addToken("c", -1.0);
    _ = try b.addToken("ab", -2.0);
    _ = try b.addToken("bc", -2.0);
    const unk = try b.addToken("<unk>", -10.0);
    var u = try b.finalize(unk);
    defer u.deinit();

    var prng = std.Random.DefaultPrng.init(0xBADC0DE);
    const rng = prng.random();
    var out: [16]TokenId = undefined;
    var iter: u32 = 0;
    while (iter < 100) : (iter += 1) {
        const ids = try u.sampleChunk(std.testing.allocator, "abcabc", &out, 1.0, rng);
        for (ids) |id| try std.testing.expect(id < u.count);
        try std.testing.expect(ids.len > 0);
    }
}

test "encodeChunkWithOffsets covers input exactly" {
    var b = Unigram.Builder.init(std.testing.allocator);
    defer b.deinit();
    _ = try b.addToken("a", -2.0);
    _ = try b.addToken("b", -2.0);
    _ = try b.addToken("ab", -1.0);
    _ = try b.addToken("c", -2.0);
    const unk = try b.addToken("<unk>", -10.0);
    var u = try b.finalize(unk);
    defer u.deinit();

    var out_ids: [8]TokenId = undefined;
    var out_off: [8]Span = undefined;
    const n = try u.encodeChunkWithOffsets(std.testing.allocator, "abc", 0, &out_ids, &out_off);
    try std.testing.expect(n > 0);

    // Non-overlapping, non-decreasing, covering [0, 3).
    try std.testing.expectEqual(@as(u32, 0), out_off[0].start);
    try std.testing.expectEqual(@as(u32, 3), out_off[n - 1].end);
    var idx: usize = 0;
    while (idx + 1 < n) : (idx += 1) {
        try std.testing.expect(out_off[idx].start <= out_off[idx].end);
        try std.testing.expectEqual(out_off[idx].end, out_off[idx + 1].start);
    }
}

test "encodeChunkWithOffsets respects chunk_offset" {
    var b = Unigram.Builder.init(std.testing.allocator);
    defer b.deinit();
    _ = try b.addToken("a", -1.0);
    _ = try b.addToken("b", -1.0);
    const unk = try b.addToken("<unk>", -10.0);
    var u = try b.finalize(unk);
    defer u.deinit();

    var out_ids: [4]TokenId = undefined;
    var out_off: [4]Span = undefined;
    const n = try u.encodeChunkWithOffsets(std.testing.allocator, "ab", 7, &out_ids, &out_off);
    try std.testing.expectEqual(@as(usize, 2), n);
    try std.testing.expectEqual(@as(u32, 7), out_off[0].start);
    try std.testing.expectEqual(@as(u32, 8), out_off[0].end);
    try std.testing.expectEqual(@as(u32, 8), out_off[1].start);
    try std.testing.expectEqual(@as(u32, 9), out_off[1].end);
}

// --- byte_fallback + min_score tests --------------------------------------

test "finalize caches min_score across normal pieces" {
    // min_score must be the min of all real piece scores (excludes the
    // unk slot, whose 0.0 score in T5/LLaMA exports would otherwise
    // pull min_score up and break the unk-penalty derivation).
    var b = Unigram.Builder.init(std.testing.allocator);
    defer b.deinit();
    _ = try b.addToken("a", -1.0);
    _ = try b.addToken("b", -5.5);
    _ = try b.addToken("c", -3.0);
    const unk = try b.addToken("<unk>", 0.0);
    var u = try b.finalize(unk);
    defer u.deinit();

    try std.testing.expectEqual(@as(f32, -5.5), u.min_score);
}

test "encodeChunk with byte_fallback emits per-byte ids for unknown bytes" {
    // Vocab: a "▁hello" multi-byte piece + 256 byte-fallback ids. Encoding
    // "▁hello!" should emit the piece id for "▁hello" then a byte-fallback
    // id for '!' (which isn't in any piece). We use a multi-byte piece
    // to avoid the trie collision between a 1-byte piece and the
    // single-byte byte_fallback token for the same byte (the builder's
    // trie collapses both to the byte-fallback id since byte tokens are
    // added last and the trie uses last-writer-wins for terminal ids).
    var b = Unigram.Builder.init(std.testing.allocator);
    defer b.deinit();
    const hello_id = try b.addToken("hello", -1.0);
    const unk = try b.addToken("<unk>", 0.0);
    var bf_table: [256]TokenId = undefined;
    var i: u32 = 0;
    while (i < 256) : (i += 1) {
        const id = try b.addToken(&[1]u8{@intCast(i)}, -20.0);
        bf_table[i] = id;
    }
    var u = try b.finalize(unk);
    defer u.deinit();
    u.byte_fallback = bf_table;

    var out: [8]TokenId = undefined;
    const ids = try u.encodeChunk(std.testing.allocator, "hello!", &out);
    try std.testing.expectEqual(@as(usize, 2), ids.len);
    try std.testing.expectEqual(hello_id, ids[0]);
    // '!' = 0x21 → byte-fallback id (2 + 0x21 under this layout: indices
    // 0=hello, 1=<unk>, then 2..257 = byte tokens).
    try std.testing.expectEqual(@as(TokenId, 2 + 0x21), ids[1]);
}

test "encodeChunk with byte_fallback null preserves unk-id behavior" {
    // Same vocab without byte_fallback set — encoder must emit the
    // unk_id for the unknown byte (legacy contract).
    var b = Unigram.Builder.init(std.testing.allocator);
    defer b.deinit();
    const a_id = try b.addToken("a", -1.0);
    const unk = try b.addToken("<unk>", -10.0);
    var u = try b.finalize(unk);
    defer u.deinit();
    try std.testing.expect(u.byte_fallback == null);

    var out: [4]TokenId = undefined;
    const ids = try u.encodeChunk(std.testing.allocator, "az", &out);
    try std.testing.expectEqualSlices(TokenId, &.{ a_id, unk }, ids);
}

test "encodeChunk merges consecutive unks without byte_fallback" {
    // Three consecutive unknown ASCII chars should collapse to a single
    // unk_id (SP's sentencepiece_processor.cc behavior). With
    // byte_fallback enabled the same input would emit three separate
    // byte-fallback ids.
    var b = Unigram.Builder.init(std.testing.allocator);
    defer b.deinit();
    const a_id = try b.addToken("a", -1.0);
    const unk = try b.addToken("<unk>", -10.0);
    var u = try b.finalize(unk);
    defer u.deinit();

    var out: [8]TokenId = undefined;
    const ids = try u.encodeChunk(std.testing.allocator, "axyz", &out);
    // Expected: a + single merged unk for "xyz" (not three unks).
    try std.testing.expectEqualSlices(TokenId, &.{ a_id, unk }, ids);
}

test "encodeChunk byte_fallback emits one id per byte for multi-byte unknowns" {
    // A 3-byte UTF-8 codepoint (U+2603 = ☃, bytes E2 98 83) that's not
    // a vocab piece should expand to three byte-fallback ids — one for
    // each byte — when byte_fallback is set, matching SP's
    // `<0xNN><0xNN><0xNN>` decomposition.
    var b = Unigram.Builder.init(std.testing.allocator);
    defer b.deinit();
    _ = try b.addToken("a", -1.0);
    const unk = try b.addToken("<unk>", 0.0);
    var bf_table: [256]TokenId = undefined;
    var i: u32 = 0;
    while (i < 256) : (i += 1) {
        const id = try b.addToken(&[1]u8{@intCast(i)}, -20.0);
        bf_table[i] = id;
    }
    var u = try b.finalize(unk);
    defer u.deinit();
    u.byte_fallback = bf_table;

    var out: [8]TokenId = undefined;
    const ids = try u.encodeChunk(std.testing.allocator, "\xE2\x98\x83", &out);
    try std.testing.expectEqual(@as(usize, 3), ids.len);
    try std.testing.expectEqual(@as(TokenId, 2 + 0xE2), ids[0]);
    try std.testing.expectEqual(@as(TokenId, 2 + 0x98), ids[1]);
    try std.testing.expectEqual(@as(TokenId, 2 + 0x83), ids[2]);
}

test "Builder.excludeFromTrie keeps the id slot but skips trie registration" {
    // The SP bridge uses this to keep BOS/EOS/unk piece slots addressable
    // (callers may look up `.idBytes` on them) while preventing the
    // Viterbi trie walk from matching their byte image — mirroring SP's
    // encoder behavior. Test in isolation by adding three pieces and
    // excluding the score-0 special: the encoder should split the
    // literal "ab" path into per-char pieces rather than match the 2-char
    // special even though it'd be score-0 (the best possible).
    var b = Unigram.Builder.init(std.testing.allocator);
    defer b.deinit();
    const unk = try b.addToken("<unk>", -10.0);
    const a_id = try b.addToken("a", -2.0);
    const b_id = try b.addToken("b", -2.0);
    const ab_id = try b.addToken("ab", 0.0); // score-0 → would always win
    try b.excludeFromTrie(ab_id);
    var u = try b.finalize(unk);
    defer u.deinit();

    // All four pieces are queryable by id even though `ab` isn't in trie.
    try std.testing.expectEqual(@as(u32, 4), u.count);
    try std.testing.expectEqualStrings("ab", u.idBytes(ab_id));

    var out: [4]TokenId = undefined;
    const ids = try u.encodeChunk(std.testing.allocator, "ab", &out);
    try std.testing.expectEqualSlices(TokenId, &.{ a_id, b_id }, ids);
}
