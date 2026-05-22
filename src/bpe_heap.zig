const std = @import("std");
const TokenId = @import("token.zig").TokenId;
const bpe_mod = @import("bpe.zig");
const HotEntry = bpe_mod.HotEntry;
const hotHash = bpe_mod.hotHash;
const HOT_MASK = bpe_mod.HOT_MASK;
const HOT_KEY_INLINE_MAX = bpe_mod.HOT_KEY_INLINE_MAX;

pub const RANK_INVALID: u32 = std.math.maxInt(u32);
pub const NONE: u32 = std.math.maxInt(u32);

pub const HeapNode = struct { rank: u32, part_idx: u32 };

/// Inline hot-table lookup mirroring `Bpe.hotLookup` — duplicated here
/// (instead of taking a `*const Bpe` pointer) so the heap encoder stays
/// decoupled from `bpe.zig`'s struct layout and so the compiler sees
/// the lookup body open at every call site in the merge loop.
inline fn hotLookup(hot_table: ?[]const HotEntry, key: []const u8) ?TokenId {
    if (key.len == 0 or key.len > HOT_KEY_INLINE_MAX) return null;
    const table = hot_table orelse return null;
    const h = hotHash(key);
    const slot = table[h & HOT_MASK];
    if (slot.hash != h) return null;
    if (slot.key_len != key.len) return null;
    var i: u8 = 0;
    while (i < slot.key_len) : (i += 1) {
        if (slot.key_bytes[i] != key[i]) return null;
    }
    return slot.id;
}

inline fn lessNode(a: HeapNode, b: HeapNode) bool {
    if (a.rank != b.rank) return a.rank < b.rank;
    return a.part_idx < b.part_idx;
}

inline fn parent(i: usize) usize {
    return (i - 1) >> 2;
}

inline fn firstChild(i: usize) usize {
    return (i << 2) + 1;
}

fn siftDown(buf: []HeapNode, n: usize, start: usize) void {
    var i = start;
    while (true) {
        const c0 = firstChild(i);
        if (c0 >= n) return;
        var best = c0;
        const c_end = @min(c0 + 4, n);
        var c = c0 + 1;
        while (c < c_end) : (c += 1) {
            if (lessNode(buf[c], buf[best])) best = c;
        }
        if (!lessNode(buf[best], buf[i])) return;
        const tmp = buf[i];
        buf[i] = buf[best];
        buf[best] = tmp;
        i = best;
    }
}

fn siftUp(buf: []HeapNode, start: usize) void {
    var i = start;
    while (i > 0) {
        const p = parent(i);
        if (!lessNode(buf[i], buf[p])) return;
        const tmp = buf[i];
        buf[i] = buf[p];
        buf[p] = tmp;
        i = p;
    }
}

fn heapify(buf: []HeapNode, n: usize) void {
    if (n < 2) return;
    var i = parent(n - 1) + 1;
    while (i > 0) {
        i -= 1;
        siftDown(buf, n, i);
    }
}

fn push(buf: []HeapNode, n: *usize, node: HeapNode) void {
    std.debug.assert(n.* < buf.len);
    buf[n.*] = node;
    n.* += 1;
    siftUp(buf, n.* - 1);
}

fn pop(buf: []HeapNode, n: *usize) ?HeapNode {
    if (n.* == 0) return null;
    const top = buf[0];
    n.* -= 1;
    if (n.* > 0) {
        buf[0] = buf[n.*];
        siftDown(buf, n.*, 0);
    }
    return top;
}

inline fn pairRank(
    chunk: []const u8,
    parts_start: []const u32,
    parts_len: []const u32,
    a: u32,
    b: u32,
    by_bytes: *const std.StringHashMap(TokenId),
    hot_table: ?[]const HotEntry,
) u32 {
    const s = parts_start[a];
    const total = parts_len[a] + parts_len[b];
    const key = chunk[s .. s + total];
    if (hotLookup(hot_table, key)) |id| return id;
    if (by_bytes.get(key)) |id| return id;
    return RANK_INVALID;
}

pub fn encode(
    chunk: []const u8,
    n: usize,
    parts_start: []u32,
    parts_len: []u32,
    ranks: []u32,
    prev: []u32,
    next: []u32,
    by_bytes: *const std.StringHashMap(TokenId),
    heap_buf: []HeapNode,
    out: []TokenId,
) usize {
    return encodeWithFallback(chunk, n, parts_start, parts_len, ranks, prev, next, by_bytes, null, heap_buf, out, null);
}

/// SP-style heap encoder for long chunks. Differs from `encode` in two
/// ways:
///   1. Pair rank lookup goes through `piece_ranks[id]` — SP's merge
///      priority — instead of using the raw id. Lower rank wins.
///   2. The emit phase pulls each part's id from a parallel `part_ids`
///      array that the caller pre-populated (typically from the initial
///      codepoint-level segmentation), instead of re-hashing the bytes.
///      Each merge updates the surviving part's id to the lookup of the
///      merged byte span; that id MUST exist in `by_bytes` by SP
///      construction (the same lookup that produced the rank for the
///      heap entry).
pub fn encodeSpBpe(
    chunk: []const u8,
    n: usize,
    parts_start: []u32,
    parts_len: []u32,
    part_ids: []TokenId,
    ranks: []u32,
    prev: []u32,
    next: []u32,
    by_bytes: *const std.StringHashMap(TokenId),
    piece_ranks: []const u32,
    heap_buf: []HeapNode,
    out: []TokenId,
) usize {
    if (n == 0) return 0;

    var heap_n: usize = 0;
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        if (ranks[i] != RANK_INVALID) {
            heap_buf[heap_n] = .{ .rank = ranks[i], .part_idx = i };
            heap_n += 1;
        }
    }
    heapify(heap_buf, heap_n);

    var live_count: usize = n;

    outer: while (live_count >= 2) {
        var a: u32 = undefined;
        while (true) {
            const t = pop(heap_buf, &heap_n) orelse break :outer;
            if (ranks[t.part_idx] != t.rank) continue;
            if (next[t.part_idx] == NONE) continue;
            a = t.part_idx;
            break;
        }
        const b = next[a];

        // Merge b into a. Look up the merged piece's id once and stash
        // it in part_ids[a] so the emit loop doesn't have to re-hash.
        const ms = parts_start[a];
        const merged_total = parts_len[a] + parts_len[b];
        const merged_key = chunk[ms .. ms + merged_total];
        const merged_id = by_bytes.get(merged_key) orelse unreachable;
        part_ids[a] = merged_id;
        parts_len[a] = merged_total;

        const b_next = next[b];
        next[a] = b_next;
        if (b_next != NONE) prev[b_next] = a;

        ranks[b] = RANK_INVALID;
        next[b] = NONE;
        prev[b] = NONE;

        // Recompute neighbour ranks. Pair rank = piece_ranks[lookup id]
        // — uses the SP merge priority, not the raw id.
        if (b_next != NONE) {
            const s = parts_start[a];
            const total = parts_len[a] + parts_len[b_next];
            const key = chunk[s .. s + total];
            const r = if (by_bytes.get(key)) |id| piece_ranks[id] else RANK_INVALID;
            ranks[a] = r;
            if (r != RANK_INVALID) {
                push(heap_buf, &heap_n, .{ .rank = r, .part_idx = a });
            }
        } else {
            ranks[a] = RANK_INVALID;
        }

        const pa = prev[a];
        if (pa != NONE) {
            const s = parts_start[pa];
            const total = parts_len[pa] + parts_len[a];
            const key = chunk[s .. s + total];
            const r = if (by_bytes.get(key)) |id| piece_ranks[id] else RANK_INVALID;
            ranks[pa] = r;
            if (r != RANK_INVALID) {
                push(heap_buf, &heap_n, .{ .rank = r, .part_idx = pa });
            }
        }

        live_count -= 1;
    }

    var w: usize = 0;
    var cur: u32 = 0;
    while (cur != NONE) {
        out[w] = part_ids[cur];
        w += 1;
        cur = next[cur];
    }
    return w;
}

/// Variant of `encode` that applies an SP-style byte-fallback table to
/// length-1 chunks after the merge loop. See `Bpe.byte_fallback`.
///
/// `hot_table` is the optional two-level merge-rank front cache from
/// `bpe.zig` (see `Bpe.hot_table`). When non-null, every `by_bytes.get`
/// is preceded by a hot-table probe that hits in L1d for the common
/// high-frequency merges, cutting StringHashMap pressure under SMT-
/// sibling cache contention. Pass `null` to disable (synthetic-vocab
/// test paths).
pub fn encodeWithFallback(
    chunk: []const u8,
    n: usize,
    parts_start: []u32,
    parts_len: []u32,
    ranks: []u32,
    prev: []u32,
    next: []u32,
    by_bytes: *const std.StringHashMap(TokenId),
    byte_fallback: ?*const [256]TokenId,
    heap_buf: []HeapNode,
    out: []TokenId,
    hot_table: ?[]const HotEntry,
) usize {
    if (n == 0) return 0;

    // Build initial heap from all populated ranks. ranks[i] is the rank
    // of the pair (i, i+1); ranks[n-1] is always RANK_INVALID.
    var heap_n: usize = 0;
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        if (ranks[i] != RANK_INVALID) {
            heap_buf[heap_n] = .{ .rank = ranks[i], .part_idx = i };
            heap_n += 1;
        }
    }
    heapify(heap_buf, heap_n);

    var live_count: usize = n;

    outer: while (live_count >= 2) {
        // Pop and skip stale entries. An entry is stale if (1) the
        // recorded rank no longer matches the current pair-rank of that
        // index, or (2) the left part has been merged away (next ==
        // NONE means it's either dead or the tail of the list).
        var a: u32 = undefined;
        while (true) {
            const t = pop(heap_buf, &heap_n) orelse break :outer;
            if (ranks[t.part_idx] != t.rank) continue;
            if (next[t.part_idx] == NONE) continue;
            a = t.part_idx;
            break;
        }
        const b = next[a];

        // Merge b into a. The byte ranges in `chunk` stay contiguous
        // because we only ever merge adjacent live parts.
        parts_len[a] += parts_len[b];

        const b_next = next[b];
        next[a] = b_next;
        if (b_next != NONE) prev[b_next] = a;

        // Mark b dead so any stale heap entry referencing it is skipped.
        ranks[b] = RANK_INVALID;
        next[b] = NONE;
        prev[b] = NONE;

        // Recompute rank for a's new right pair. ranks[a] MUST be
        // written before the push so the stale-skip sees the new value.
        if (b_next != NONE) {
            const r = pairRank(chunk, parts_start, parts_len, a, b_next, by_bytes, hot_table);
            ranks[a] = r;
            if (r != RANK_INVALID) {
                push(heap_buf, &heap_n, .{ .rank = r, .part_idx = a });
            }
        } else {
            ranks[a] = RANK_INVALID;
        }

        // Recompute rank for the left pair (prev[a], a).
        const pa = prev[a];
        if (pa != NONE) {
            const r = pairRank(chunk, parts_start, parts_len, pa, a, by_bytes, hot_table);
            ranks[pa] = r;
            if (r != RANK_INVALID) {
                push(heap_buf, &heap_n, .{ .rank = r, .part_idx = pa });
            }
        }

        live_count -= 1;
    }

    // Walk the live list left-to-right and emit ids. Index 0 is always
    // alive (it can only be removed as the right half of a merge, but
    // there is no part to its left, so it is never merged away).
    //
    // SP byte-fallback: a length-1 part whose raw byte has a mapped id
    // in the fallback table is rewritten to that id (see `Bpe.byte_fallback`).
    var w: usize = 0;
    var cur: u32 = 0;
    while (cur != NONE) {
        const s = parts_start[cur];
        const len = parts_len[cur];
        const key = chunk[s .. s + len];
        var id: TokenId = if (hotLookup(hot_table, key)) |r|
            r
        else
            by_bytes.get(key) orelse std.math.maxInt(TokenId);
        if (len == 1) {
            if (byte_fallback) |tbl| {
                const mapped = tbl[chunk[s]];
                if (mapped != std.math.maxInt(TokenId)) id = mapped;
            }
        }
        out[w] = id;
        w += 1;
        cur = next[cur];
    }
    return w;
}

// --- tests -----------------------------------------------------------

const testing = std.testing;

test "heap ops: push/pop maintain min order" {
    var buf: [32]HeapNode = undefined;
    var heap_n: usize = 0;
    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const rng = prng.random();
    const k: usize = 20;
    var i: usize = 0;
    while (i < k) : (i += 1) {
        const r = rng.int(u16);
        push(&buf, &heap_n, .{ .rank = @intCast(r), .part_idx = @intCast(i) });
    }
    var last: u32 = 0;
    i = 0;
    while (i < k) : (i += 1) {
        const top = pop(&buf, &heap_n) orelse return error.UnexpectedEmpty;
        try testing.expect(top.rank >= last);
        last = top.rank;
    }
    try testing.expectEqual(@as(?HeapNode, null), pop(&buf, &heap_n));
}

test "heap ops: heapify produces valid heap" {
    var buf: [64]HeapNode = undefined;
    var prng = std.Random.DefaultPrng.init(0xDEADBEEF);
    const rng = prng.random();
    for (&buf, 0..) |*node, idx| {
        node.* = .{ .rank = rng.int(u16), .part_idx = @intCast(idx) };
    }
    heapify(&buf, buf.len);
    var i: usize = 0;
    while (i < buf.len) : (i += 1) {
        const c0 = firstChild(i);
        var c = c0;
        const c_end = @min(c0 + 4, buf.len);
        while (c < c_end) : (c += 1) {
            try testing.expect(!lessNode(buf[c], buf[i]));
        }
    }
}

test "encode empty input" {
    var by_bytes = std.StringHashMap(TokenId).init(testing.allocator);
    defer by_bytes.deinit();
    var heap_buf: [4]HeapNode = undefined;
    var out: [4]TokenId = undefined;
    var parts_start: [0]u32 = undefined;
    var parts_len: [0]u32 = undefined;
    var ranks: [0]u32 = undefined;
    var prev_arr: [0]u32 = undefined;
    var next_arr: [0]u32 = undefined;
    const w = encode("", 0, &parts_start, &parts_len, &ranks, &prev_arr, &next_arr, &by_bytes, &heap_buf, &out);
    try testing.expectEqual(@as(usize, 0), w);
}

test "encode single byte" {
    var by_bytes = std.StringHashMap(TokenId).init(testing.allocator);
    defer by_bytes.deinit();
    try by_bytes.put("a", 0);

    const chunk = "a";
    var parts_start = [_]u32{0};
    var parts_len = [_]u32{1};
    var ranks = [_]u32{RANK_INVALID};
    var prev_arr = [_]u32{NONE};
    var next_arr = [_]u32{NONE};
    var heap_buf: [4]HeapNode = undefined;
    var out: [4]TokenId = undefined;

    const w = encode(chunk, 1, &parts_start, &parts_len, &ranks, &prev_arr, &next_arr, &by_bytes, &heap_buf, &out);
    try testing.expectEqual(@as(usize, 1), w);
    try testing.expectEqual(@as(TokenId, 0), out[0]);
}

// Helper: build initial parts/prev/next/ranks arrays for a chunk where
// each byte starts as a single part.
fn initParts(
    chunk: []const u8,
    parts_start: []u32,
    parts_len: []u32,
    ranks: []u32,
    prev_arr: []u32,
    next_arr: []u32,
    by_bytes: *const std.StringHashMap(TokenId),
) void {
    const n = chunk.len;
    for (0..n) |i| {
        parts_start[i] = @intCast(i);
        parts_len[i] = 1;
        prev_arr[i] = if (i == 0) NONE else @intCast(i - 1);
        next_arr[i] = if (i + 1 == n) NONE else @intCast(i + 1);
    }
    for (0..n) |i| {
        if (i + 1 < n) {
            const key = chunk[i .. i + 2];
            ranks[i] = if (by_bytes.get(key)) |id| id else RANK_INVALID;
        } else {
            ranks[i] = RANK_INVALID;
        }
    }
}

test "encode greedy merge chain" {
    var by_bytes = std.StringHashMap(TokenId).init(testing.allocator);
    defer by_bytes.deinit();
    try by_bytes.put("a", 0);
    try by_bytes.put("b", 1);
    try by_bytes.put("c", 2);
    try by_bytes.put("d", 3);
    try by_bytes.put("e", 4);
    try by_bytes.put("f", 5);
    try by_bytes.put("ab", 6);
    try by_bytes.put("cd", 7);
    try by_bytes.put("ef", 8);
    try by_bytes.put("abcd", 9);
    try by_bytes.put("abcdef", 10);

    const chunk = "abcdef";
    const n: usize = 6;
    var parts_start: [6]u32 = undefined;
    var parts_len: [6]u32 = undefined;
    var ranks: [6]u32 = undefined;
    var prev_arr: [6]u32 = undefined;
    var next_arr: [6]u32 = undefined;
    initParts(chunk, &parts_start, &parts_len, &ranks, &prev_arr, &next_arr, &by_bytes);

    var heap_buf: [18]HeapNode = undefined;
    var out: [6]TokenId = undefined;
    const w = encode(chunk, n, &parts_start, &parts_len, &ranks, &prev_arr, &next_arr, &by_bytes, &heap_buf, &out);
    try testing.expectEqual(@as(usize, 1), w);
    try testing.expectEqual(@as(TokenId, 10), out[0]);
}

test "encode lazy-delete handles stale heap entries" {
    var by_bytes = std.StringHashMap(TokenId).init(testing.allocator);
    defer by_bytes.deinit();
    try by_bytes.put("a", 0);
    try by_bytes.put("b", 1);
    try by_bytes.put("c", 2);
    try by_bytes.put("ab", 3);
    try by_bytes.put("bc", 4);

    const chunk = "abc";
    const n: usize = 3;
    var parts_start: [3]u32 = undefined;
    var parts_len: [3]u32 = undefined;
    var ranks: [3]u32 = undefined;
    var prev_arr: [3]u32 = undefined;
    var next_arr: [3]u32 = undefined;
    initParts(chunk, &parts_start, &parts_len, &ranks, &prev_arr, &next_arr, &by_bytes);

    var heap_buf: [9]HeapNode = undefined;
    var out: [3]TokenId = undefined;
    const w = encode(chunk, n, &parts_start, &parts_len, &ranks, &prev_arr, &next_arr, &by_bytes, &heap_buf, &out);
    try testing.expectEqual(@as(usize, 2), w);
    try testing.expectEqual(@as(TokenId, 3), out[0]);
    try testing.expectEqual(@as(TokenId, 2), out[1]);
}

// Regression: a long all-same-byte chunk must collapse the way tiktoken
// does. Pre-fix this returned a garbled mix because all initial pair
// ranks tied and the heap's order-of-insertion tiebreak picked
// non-leftmost merges, diverging from the scalar leftmost-first rule.
test "long all-same-byte chunk merges deterministically" {
    var by_bytes = std.StringHashMap(TokenId).init(testing.allocator);
    defer by_bytes.deinit();
    try by_bytes.put("a", 0);
    try by_bytes.put("aa", 1);
    try by_bytes.put("aaaa", 2);
    try by_bytes.put("aaaaaaaa", 3);

    const n: usize = 64;
    var chunk: [n]u8 = undefined;
    @memset(&chunk, 'a');

    var parts_start: [n]u32 = undefined;
    var parts_len: [n]u32 = undefined;
    var ranks: [n]u32 = undefined;
    var prev_arr: [n]u32 = undefined;
    var next_arr: [n]u32 = undefined;
    initParts(&chunk, &parts_start, &parts_len, &ranks, &prev_arr, &next_arr, &by_bytes);

    var heap_buf: [3 * n]HeapNode = undefined;
    var out: [n]TokenId = undefined;
    const w = encode(&chunk, n, &parts_start, &parts_len, &ranks, &prev_arr, &next_arr, &by_bytes, &heap_buf, &out);

    // Expected: 8 copies of "aaaaaaaa" (token id 3).
    try testing.expectEqual(@as(usize, 8), w);
    var k: usize = 0;
    while (k < 8) : (k += 1) try testing.expectEqual(@as(TokenId, 3), out[k]);
}

// Two pairs share the minimum rank; the scalar path merges the leftmost
// first. The heap must too, via lexicographic (rank, part_idx) ordering.
test "tie-break uses leftmost index when ranks are equal" {
    var by_bytes = std.StringHashMap(TokenId).init(testing.allocator);
    defer by_bytes.deinit();
    try by_bytes.put("a", 0);
    try by_bytes.put("b", 1);
    try by_bytes.put("ab", 2);
    try by_bytes.put("ba", 3);

    const chunk = "abab";
    const n: usize = 4;
    var parts_start: [4]u32 = undefined;
    var parts_len: [4]u32 = undefined;
    var ranks: [4]u32 = undefined;
    var prev_arr: [4]u32 = undefined;
    var next_arr: [4]u32 = undefined;
    initParts(chunk, &parts_start, &parts_len, &ranks, &prev_arr, &next_arr, &by_bytes);

    var heap_buf: [12]HeapNode = undefined;
    var out: [4]TokenId = undefined;
    const w = encode(chunk, n, &parts_start, &parts_len, &ranks, &prev_arr, &next_arr, &by_bytes, &heap_buf, &out);

    // Merge pos-0 first (leftmost (a,b) at rank 2) -> [ab, a, b].
    // Then (a,b) at pos 2 wins -> [ab, ab].
    try testing.expectEqual(@as(usize, 2), w);
    try testing.expectEqual(@as(TokenId, 2), out[0]);
    try testing.expectEqual(@as(TokenId, 2), out[1]);
}

// Reference scalar BPE that mirrors the algorithm in `bpe.zig`.
// Re-implemented here so this test stays self-contained.
fn scalarRef(
    chunk: []const u8,
    by_bytes: *const std.StringHashMap(TokenId),
    out: []TokenId,
) usize {
    if (chunk.len == 0) return 0;
    var parts_start: [256]u32 = undefined;
    var parts_len: [256]u32 = undefined;
    var ranks: [256]u32 = undefined;
    std.debug.assert(chunk.len <= 256);

    var live: u32 = @intCast(chunk.len);
    for (0..chunk.len) |i| {
        parts_start[i] = @intCast(i);
        parts_len[i] = 1;
    }
    if (live >= 2) {
        var i: u32 = 0;
        while (i + 1 < live) : (i += 1) {
            const s = parts_start[i];
            const total = parts_len[i] + parts_len[i + 1];
            const key = chunk[s .. s + total];
            ranks[i] = if (by_bytes.get(key)) |r| r else RANK_INVALID;
        }
    }
    if (live > 0) ranks[live - 1] = RANK_INVALID;

    while (live >= 2) {
        // Leftmost min scan.
        var mi: u32 = 0;
        var best: u32 = RANK_INVALID;
        var k: u32 = 0;
        while (k + 1 < live) : (k += 1) {
            if (ranks[k] < best) {
                best = ranks[k];
                mi = k;
            }
        }
        if (best == RANK_INVALID) break;

        parts_len[mi] += parts_len[mi + 1];
        var j: u32 = mi + 1;
        while (j + 1 < live) : (j += 1) {
            parts_start[j] = parts_start[j + 1];
            parts_len[j] = parts_len[j + 1];
            ranks[j] = ranks[j + 1];
        }
        live -= 1;

        if (mi > 0) {
            const li = mi - 1;
            const s = parts_start[li];
            const total = parts_len[li] + parts_len[mi];
            const key = chunk[s .. s + total];
            ranks[li] = if (by_bytes.get(key)) |r| r else RANK_INVALID;
        }
        if (mi + 1 < live) {
            const s = parts_start[mi];
            const total = parts_len[mi] + parts_len[mi + 1];
            const key = chunk[s .. s + total];
            ranks[mi] = if (by_bytes.get(key)) |r| r else RANK_INVALID;
        } else {
            ranks[mi] = RANK_INVALID;
        }
    }

    var w: usize = 0;
    var idx: u32 = 0;
    while (idx < live) : (idx += 1) {
        const s = parts_start[idx];
        const len = parts_len[idx];
        const key = chunk[s .. s + len];
        out[w] = by_bytes.get(key) orelse std.math.maxInt(TokenId);
        w += 1;
    }
    return w;
}

test "regression: scalar vs heap agree on random long chunks" {
    var by_bytes = std.StringHashMap(TokenId).init(testing.allocator);
    defer by_bytes.deinit();
    // Small vocab over the alphabet {a,b,c,d} plus a handful of merges
    // with intentionally clustered ranks to provoke ties.
    try by_bytes.put("a", 0);
    try by_bytes.put("b", 1);
    try by_bytes.put("c", 2);
    try by_bytes.put("d", 3);
    try by_bytes.put("aa", 10);
    try by_bytes.put("ab", 10);
    try by_bytes.put("bc", 11);
    try by_bytes.put("cd", 12);
    try by_bytes.put("ba", 13);
    try by_bytes.put("dd", 14);
    try by_bytes.put("aaa", 15);
    try by_bytes.put("abc", 16);
    try by_bytes.put("bcd", 17);
    try by_bytes.put("cda", 18);
    try by_bytes.put("aabb", 19);
    try by_bytes.put("abcd", 20);

    const n: usize = 100;
    var chunk: [n]u8 = undefined;
    var parts_start: [n]u32 = undefined;
    var parts_len: [n]u32 = undefined;
    var ranks: [n]u32 = undefined;
    var prev_arr: [n]u32 = undefined;
    var next_arr: [n]u32 = undefined;
    var heap_buf: [3 * n]HeapNode = undefined;
    var heap_out: [n]TokenId = undefined;
    var scalar_out: [n]TokenId = undefined;

    const alphabet = "abcd";
    var prng = std.Random.DefaultPrng.init(0xA5A5_5A5A);
    const rng = prng.random();

    var trial: usize = 0;
    while (trial < 10) : (trial += 1) {
        for (0..n) |i| chunk[i] = alphabet[rng.uintLessThan(usize, alphabet.len)];

        initParts(&chunk, &parts_start, &parts_len, &ranks, &prev_arr, &next_arr, &by_bytes);
        const wh = encode(&chunk, n, &parts_start, &parts_len, &ranks, &prev_arr, &next_arr, &by_bytes, &heap_buf, &heap_out);
        const ws = scalarRef(&chunk, &by_bytes, &scalar_out);

        try testing.expectEqual(ws, wh);
        try testing.expectEqualSlices(TokenId, scalar_out[0..ws], heap_out[0..wh]);
    }
}
