//! Deterministic multi-head token-n-gram hashing.
//!
//! Produces the kind of memory addresses DeepSeek's "Engram" conditional-
//! memory module needs: for a token-id stream, hash every length-`n`
//! window under `heads` independent hash functions. The output depends
//! only on the input token ids (never on model activations), so callers
//! can compute it ahead of time and prefetch the corresponding embedding
//! rows while the GPU is busy with early layers.
//!
//! We emit **raw u64 hashes** rather than table indices: the caller masks
//! `hash & ((1 << table_bits) - 1)` to land in a table of its chosen size.
//! That keeps this layer policy-free and lets one pass feed tables of
//! different widths.
//!
//! Output layout for `hashNGrams` is row-major `[position][head]`: the
//! `heads` hashes for window position 0 come first, then position 1, etc.
//!
//! The mixer is a splitmix64-style finalizer seeded by the 64-bit
//! Fibonacci constant (the same `0x9E3779B97F4A7C15` `bpe.hotHash` uses).
//! `hotHash` itself is *not* reused — it is tuned for ≤7-byte string keys,
//! whereas here we fold a sequence of 32-bit token ids.

const std = @import("std");
const TokenId = @import("token.zig").TokenId;
const BatchPool = @import("thread_pool.zig").BatchPool;

/// 64-bit Fibonacci/golden-ratio constant — the avalanche seed.
const GOLDEN: u64 = 0x9E3779B97F4A7C15;

/// splitmix64 finalizer. Bijective, strong avalanche; the workhorse mix.
inline fn smix(x: u64) u64 {
    var z = x +% GOLDEN;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

/// Independent per-head salt. Head 0,1,2,… each yield an unrelated hash
/// family, so collisions in one head are uncorrelated with another.
pub inline fn headSalt(head: u32) u64 {
    return smix(GOLDEN ^ (@as(u64, head) *% GOLDEN));
}

/// Hash one n-gram (a window of token ids) under `head_salt`. Folding each
/// id through `smix` makes the result order-sensitive, and the trailing
/// length fold guards against a shorter gram hashing equal to a prefix of
/// a longer one when `n` varies across calls.
pub inline fn mixNGram(ids: []const TokenId, head_salt: u64) u64 {
    var acc: u64 = head_salt ^ GOLDEN;
    for (ids) |id| acc = smix(acc ^ @as(u64, id));
    return smix(acc ^ @as(u64, ids.len));
}

/// Number of length-`n` window positions in a stream of `ids_len` ids.
/// Zero when `n == 0` or the stream is shorter than one window.
pub fn hashNGramsLen(ids_len: usize, n: u32) usize {
    if (n == 0 or ids_len < n) return 0;
    return ids_len - n + 1;
}

/// Total u64s `hashNGrams` writes: positions × heads.
pub fn hashNGramsOutLen(ids_len: usize, n: u32, heads: u32) usize {
    return hashNGramsLen(ids_len, n) * heads;
}

/// Hash every length-`n` window of `ids` under `heads` hash functions,
/// writing row-major `[position][head]` into `out`. Returns the number of
/// window **positions** written; the count of u64s written is that times
/// `heads`. `out` must hold at least `hashNGramsOutLen(ids.len, n, heads)`
/// elements (asserted in safe builds).
pub fn hashNGrams(ids: []const TokenId, n: u32, heads: u32, out: []u64) usize {
    const positions = hashNGramsLen(ids.len, n);
    if (positions == 0 or heads == 0) return positions;
    std.debug.assert(out.len >= positions * heads);

    // Per-head salts are derived once; the window walk then costs one
    // `mixNGram` per (position, head).
    var w: usize = 0;
    var p: usize = 0;
    while (p < positions) : (p += 1) {
        const gram = ids[p .. p + n];
        var h: u32 = 0;
        while (h < heads) : (h += 1) {
            out[w] = mixNGram(gram, headSalt(h));
            w += 1;
        }
    }
    return positions;
}

/// Hash many id streams in parallel across `pool`. `results[i]` is set to a
/// freshly allocated slice (owned by `result_allocator`) holding the
/// row-major hashes for `id_streams[i]`; an empty stream yields an empty
/// (zero-length) slice. On error, every slice allocated so far is freed.
pub fn hashNGramsBatch(
    result_allocator: std.mem.Allocator,
    pool: *BatchPool,
    id_streams: []const []const TokenId,
    n: u32,
    heads: u32,
    results: [][]u64,
) !void {
    std.debug.assert(results.len == id_streams.len);
    for (results) |*r| r.* = &.{};

    const Ctx = struct {
        streams: []const []const TokenId,
        results: [][]u64,
        alloc: std.mem.Allocator,
        n: u32,
        heads: u32,
        errored: std.atomic.Value(u32),

        const Self = @This();

        pub fn run(c: *Self, idx: usize, worker_idx: usize) void {
            _ = worker_idx;
            const out_len = hashNGramsOutLen(c.streams[idx].len, c.n, c.heads);
            if (out_len == 0) return;
            const buf = c.alloc.alloc(u64, out_len) catch {
                _ = c.errored.fetchAdd(1, .acq_rel);
                return;
            };
            _ = hashNGrams(c.streams[idx], c.n, c.heads, buf);
            c.results[idx] = buf;
        }
    };

    var ctx: Ctx = .{
        .streams = id_streams,
        .results = results,
        .alloc = result_allocator,
        .n = n,
        .heads = heads,
        .errored = std.atomic.Value(u32).init(0),
    };

    pool.runBatch(Ctx, &ctx, id_streams.len) catch |e| {
        for (results) |r| if (r.len > 0) result_allocator.free(r);
        for (results) |*r| r.* = &.{};
        return e;
    };

    if (ctx.errored.load(.acquire) != 0) {
        for (results) |r| if (r.len > 0) result_allocator.free(r);
        for (results) |*r| r.* = &.{};
        return error.OutOfMemory;
    }
}

// --- tests ------------------------------------------------------------

test "hashNGramsLen window math" {
    try std.testing.expectEqual(@as(usize, 0), hashNGramsLen(0, 3));
    try std.testing.expectEqual(@as(usize, 0), hashNGramsLen(2, 3)); // shorter than window
    try std.testing.expectEqual(@as(usize, 1), hashNGramsLen(3, 3));
    try std.testing.expectEqual(@as(usize, 8), hashNGramsLen(10, 3));
    try std.testing.expectEqual(@as(usize, 0), hashNGramsLen(10, 0)); // n=0 degenerate
}

test "hashNGrams is deterministic across calls" {
    const ids = [_]TokenId{ 7, 42, 1000, 3, 99, 7, 42 };
    var a: [16]u64 = undefined;
    var b: [16]u64 = undefined;
    const pa = hashNGrams(&ids, 3, 2, &a);
    const pb = hashNGrams(&ids, 3, 2, &b);
    try std.testing.expectEqual(pa, pb);
    try std.testing.expectEqualSlices(u64, a[0 .. pa * 2], b[0 .. pb * 2]);
}

test "hashNGrams short input writes nothing" {
    const ids = [_]TokenId{ 1, 2 };
    var out: [8]u64 = undefined;
    try std.testing.expectEqual(@as(usize, 0), hashNGrams(&ids, 3, 4, &out));
}

test "hashNGrams heads are independent" {
    const ids = [_]TokenId{ 5, 6, 7, 8 };
    var out: [16]u64 = undefined; // 2 positions (n=3) × up to 8 heads
    const positions = hashNGrams(&ids, 3, 3, &out);
    try std.testing.expectEqual(@as(usize, 2), positions);
    // The 3 heads for position 0 must all differ from each other.
    try std.testing.expect(out[0] != out[1]);
    try std.testing.expect(out[1] != out[2]);
    try std.testing.expect(out[0] != out[2]);
    // Same gram value appearing elsewhere would hash identically per head;
    // here positions differ so their head-0 hashes should differ.
    try std.testing.expect(out[0] != out[3]);
}

test "hashNGrams order sensitivity" {
    const fwd = [_]TokenId{ 1, 2, 3 };
    const rev = [_]TokenId{ 3, 2, 1 };
    var a: [4]u64 = undefined;
    var b: [4]u64 = undefined;
    _ = hashNGrams(&fwd, 3, 1, &a);
    _ = hashNGrams(&rev, 3, 1, &b);
    try std.testing.expect(a[0] != b[0]);
}

test "hashNGrams collision rate sane on random ids" {
    var prng = std.Random.DefaultPrng.init(0xA11CE);
    const rng = prng.random();
    const N = 4000;
    var ids: [N]TokenId = undefined;
    for (&ids) |*x| x.* = rng.uintLessThan(TokenId, 50000);

    const heads = 1;
    const positions = hashNGramsLen(N, 3);
    const out = try std.testing.allocator.alloc(u64, positions * heads);
    defer std.testing.allocator.free(out);
    _ = hashNGrams(&ids, 3, heads, out);

    // Count duplicate 32-bit-folded buckets; with ~4k items into 2^32 the
    // expected collision count is tiny. Allow generous slack.
    var seen = std.AutoHashMap(u32, void).init(std.testing.allocator);
    defer seen.deinit();
    var collisions: usize = 0;
    for (out) |h| {
        const folded: u32 = @truncate(h ^ (h >> 32));
        const gop = try seen.getOrPut(folded);
        if (gop.found_existing) collisions += 1;
    }
    try std.testing.expect(collisions < positions / 100 + 4);
}

test "hashNGramsBatch matches single-stream path" {
    var pool = try BatchPool.init(std.testing.allocator, 2);
    defer pool.deinit();

    const s0 = [_]TokenId{ 1, 2, 3, 4, 5 };
    const s1 = [_]TokenId{ 9, 8, 7 };
    const s2 = [_]TokenId{ 1, 1 }; // shorter than n -> empty
    const streams = [_][]const TokenId{ &s0, &s1, &s2 };

    var results: [3][]u64 = undefined;
    try hashNGramsBatch(std.testing.allocator, &pool, &streams, 3, 2, &results);
    defer for (results) |r| if (r.len > 0) std.testing.allocator.free(r);

    // Compare each against the serial result.
    for (streams, 0..) |stream, i| {
        const want_len = hashNGramsOutLen(stream.len, 3, 2);
        try std.testing.expectEqual(want_len, results[i].len);
        if (want_len == 0) continue;
        const ref = try std.testing.allocator.alloc(u64, want_len);
        defer std.testing.allocator.free(ref);
        _ = hashNGrams(stream, 3, 2, ref);
        try std.testing.expectEqualSlices(u64, ref, results[i]);
    }
}
