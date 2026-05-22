const std = @import("std");
const TokenId = @import("token.zig").TokenId;
const Bpe = @import("bpe.zig").Bpe;
const BatchPool = @import("thread_pool.zig").BatchPool;

pub const NULL_ID: TokenId = std.math.maxInt(TokenId);

pub const PruneStrategy = enum {
    by_min_count,
    by_bottom_fraction,
    unused_only,
};

pub const PruneOptions = struct {
    strategy: PruneStrategy = .unused_only,
    min_count: u64 = 1,
    bottom_fraction: f32 = 0.10,
    preserve_ids: []const TokenId = &.{},
    preserve_byte_fallback: bool = true,
    pool: ?*BatchPool = null,
};

pub const PruneResult = struct {
    allocator: std.mem.Allocator,
    pruned_bpe: Bpe,
    old_to_new: []const TokenId,
    new_to_old: []const TokenId,
    dropped_count: u32,

    pub fn deinit(self: *PruneResult) void {
        self.pruned_bpe.deinit();
        self.allocator.free(self.old_to_new);
        self.allocator.free(self.new_to_old);
        self.old_to_new = &.{};
        self.new_to_old = &.{};
        self.dropped_count = 0;
    }
};

const CHUNK_SIZE: usize = 64 * 1024;
const SMALL_CORPUS: usize = 1024 * 1024;

const EncodeCtx = struct {
    bpe: *const Bpe,
    corpus: []const u8,
    chunks: []const ChunkRange,
    per_worker_usage: [][]u64,
};

const ChunkRange = struct { start: usize, end: usize };

const EncodeWorker = struct {
    pub fn run(c: *EncodeCtx, idx: usize, worker_idx: usize) void {
        const range = c.chunks[idx];
        const slice = c.corpus[range.start..range.end];
        if (slice.len == 0) return;
        const out = c.bpe.allocator.alloc(TokenId, slice.len) catch return;
        defer c.bpe.allocator.free(out);
        const ids = c.bpe.encodeChunk(slice, out);
        const usage = c.per_worker_usage[worker_idx];
        for (ids) |id| {
            if (id < usage.len) usage[id] += 1;
        }
    }
};

fn buildChunks(allocator: std.mem.Allocator, corpus: []const u8) ![]ChunkRange {
    if (corpus.len == 0) return allocator.alloc(ChunkRange, 0);
    const n_chunks = (corpus.len + CHUNK_SIZE - 1) / CHUNK_SIZE;
    const chunks = try allocator.alloc(ChunkRange, n_chunks);
    var i: usize = 0;
    var off: usize = 0;
    while (i < n_chunks) : (i += 1) {
        const end = @min(off + CHUNK_SIZE, corpus.len);
        chunks[i] = .{ .start = off, .end = end };
        off = end;
    }
    return chunks;
}

fn countUsage(
    allocator: std.mem.Allocator,
    bpe: *const Bpe,
    corpus: []const u8,
    pool: ?*BatchPool,
    usage_out: []u64,
) !void {
    @memset(usage_out, 0);
    if (corpus.len == 0) return;

    const use_pool = pool != null and corpus.len >= SMALL_CORPUS;
    if (!use_pool) {
        const out = try allocator.alloc(TokenId, corpus.len);
        defer allocator.free(out);
        const ids = bpe.encodeChunk(corpus, out);
        for (ids) |id| {
            if (id < usage_out.len) usage_out[id] += 1;
        }
        return;
    }

    const bp = pool.?;
    const chunks = try buildChunks(allocator, corpus);
    defer allocator.free(chunks);

    const nw = bp.workerCount();
    const per_w = try allocator.alloc([]u64, nw);
    defer {
        for (per_w) |buf| allocator.free(buf);
        allocator.free(per_w);
    }
    for (per_w) |*buf| {
        buf.* = try allocator.alloc(u64, bpe.count);
        @memset(buf.*, 0);
    }

    var ctx: EncodeCtx = .{
        .bpe = bpe,
        .corpus = corpus,
        .chunks = chunks,
        .per_worker_usage = per_w,
    };
    try bp.runBatch(EncodeWorker, &ctx, chunks.len);

    for (per_w) |buf| {
        for (usage_out, buf) |*acc, x| acc.* += x;
    }
}

// Find the merge split for piece `id` using the same scan-for-valid-split
// approach as the HF writer: leftmost-shortest split where both halves
// already exist in the vocab with smaller ids.
fn findMergeSplit(bpe: *const Bpe, id: TokenId) ?struct { left_id: TokenId, right_id: TokenId } {
    const piece = bpe.idBytes(id);
    if (piece.len < 2) return null;
    var sp: usize = 1;
    while (sp < piece.len) : (sp += 1) {
        const lid = bpe.by_bytes.get(piece[0..sp]) orelse continue;
        const rid = bpe.by_bytes.get(piece[sp..]) orelse continue;
        if (lid < id and rid < id) return .{ .left_id = lid, .right_id = rid };
    }
    return null;
}

// Propagate "dropped" forward through merge dependencies. Any piece (id>=256)
// whose merge halves contain a dropped id becomes a drop itself. Iterate
// to fixed point.
fn propagateDrops(
    allocator: std.mem.Allocator,
    bpe: *const Bpe,
    keep: []bool,
    preserve_mask: []const bool,
) !void {
    const n = bpe.count;
    const deps_l = try allocator.alloc(TokenId, n);
    defer allocator.free(deps_l);
    const deps_r = try allocator.alloc(TokenId, n);
    defer allocator.free(deps_r);
    const has_deps = try allocator.alloc(bool, n);
    defer allocator.free(has_deps);
    @memset(has_deps, false);

    var i: u32 = 256;
    while (i < n) : (i += 1) {
        if (findMergeSplit(bpe, i)) |sp| {
            deps_l[i] = sp.left_id;
            deps_r[i] = sp.right_id;
            has_deps[i] = true;
        }
    }

    var changed = true;
    while (changed) {
        changed = false;
        var k: u32 = 256;
        while (k < n) : (k += 1) {
            if (!keep[k]) continue;
            if (preserve_mask[k]) continue;
            if (!has_deps[k]) continue;
            if (!keep[deps_l[k]] or !keep[deps_r[k]]) {
                keep[k] = false;
                changed = true;
            }
        }
    }
}

pub fn pruneBpe(
    allocator: std.mem.Allocator,
    old_bpe: *const Bpe,
    corpus: []const u8,
    opts: PruneOptions,
) !PruneResult {
    const n = old_bpe.count;

    // 1+2: encode corpus, build usage table.
    const usage = try allocator.alloc(u64, n);
    defer allocator.free(usage);
    try countUsage(allocator, old_bpe, corpus, opts.pool, usage);

    // 3: initial keep set from strategy.
    var keep = try allocator.alloc(bool, n);
    defer allocator.free(keep);

    switch (opts.strategy) {
        .unused_only => {
            for (keep, usage) |*k, u| k.* = u > 0;
        },
        .by_min_count => {
            for (keep, usage) |*k, u| k.* = u >= opts.min_count;
        },
        .by_bottom_fraction => {
            // Sort ids by usage ascending, drop bottom floor(fraction * N).
            const order = try allocator.alloc(u32, n);
            defer allocator.free(order);
            for (order, 0..) |*o, i| o.* = @intCast(i);
            const SortCtx = struct {
                usage: []const u64,
                pub fn lessThan(ctx: @This(), a: u32, b: u32) bool {
                    return ctx.usage[a] < ctx.usage[b];
                }
            };
            std.mem.sort(u32, order, SortCtx{ .usage = usage }, SortCtx.lessThan);
            const drop_n: usize = @intFromFloat(@floor(@as(f32, @floatFromInt(n)) * opts.bottom_fraction));
            for (keep) |*k| k.* = true;
            var d: usize = 0;
            while (d < drop_n) : (d += 1) {
                keep[order[d]] = false;
            }
        },
    }

    // 4: preserve rules. Build a parallel preserve_mask so the merge-prune
    // pass can refuse to drop ids that were forced-kept.
    const preserve_mask = try allocator.alloc(bool, n);
    defer allocator.free(preserve_mask);
    @memset(preserve_mask, false);

    if (opts.preserve_byte_fallback) {
        const upto: u32 = @min(@as(u32, 256), n);
        var i: u32 = 0;
        while (i < upto) : (i += 1) {
            keep[i] = true;
            preserve_mask[i] = true;
        }
    }
    for (opts.preserve_ids) |id| {
        if (id < n) {
            keep[id] = true;
            preserve_mask[id] = true;
        }
    }

    // 5: merge-dependency propagation. If a piece's halves were dropped,
    // the piece itself is unreachable; drop it. preserve_mask ids are
    // never demoted even if their deps vanish — caller has to live with
    // the fact that those ids may not be reachable via merges anymore
    // (they remain in the vocab but encoding won't synthesise them).
    try propagateDrops(allocator, old_bpe, keep, preserve_mask);

    // 6: renumber in original order.
    const old_to_new = try allocator.alloc(TokenId, n);
    errdefer allocator.free(old_to_new);

    var new_count: u32 = 0;
    {
        var i: u32 = 0;
        while (i < n) : (i += 1) {
            if (keep[i]) {
                old_to_new[i] = new_count;
                new_count += 1;
            } else {
                old_to_new[i] = NULL_ID;
            }
        }
    }

    const new_to_old = try allocator.alloc(TokenId, new_count);
    errdefer allocator.free(new_to_old);
    {
        var i: u32 = 0;
        while (i < n) : (i += 1) {
            if (keep[i]) new_to_old[old_to_new[i]] = i;
        }
    }

    // 7: build pruned_bpe. Compute total bytes, then allocate + emit.
    var total_bytes: usize = 0;
    {
        var i: u32 = 0;
        while (i < n) : (i += 1) {
            if (!keep[i]) continue;
            total_bytes += old_bpe.idBytes(i).len;
        }
    }

    const new_bytes = try allocator.alloc(u8, total_bytes);
    errdefer allocator.free(new_bytes);
    const new_offsets = try allocator.alloc(u32, @as(usize, new_count) + 1);
    errdefer allocator.free(new_offsets);

    new_offsets[0] = 0;
    var write_off: u32 = 0;
    {
        var i: u32 = 0;
        while (i < n) : (i += 1) {
            if (!keep[i]) continue;
            const piece = old_bpe.idBytes(i);
            @memcpy(new_bytes[write_off .. write_off + piece.len], piece);
            write_off += @intCast(piece.len);
            const nid = old_to_new[i];
            new_offsets[nid + 1] = write_off;
        }
    }
    std.debug.assert(write_off == total_bytes);

    var by_bytes = std.StringHashMap(TokenId).init(allocator);
    errdefer by_bytes.deinit();
    try by_bytes.ensureTotalCapacity(new_count);
    {
        var i: u32 = 0;
        while (i < new_count) : (i += 1) {
            const key = new_bytes[new_offsets[i]..new_offsets[i + 1]];
            try by_bytes.put(key, i);
        }
    }

    // Rebuild the two-level merge-rank front cache against the pruned
    // bytes/offsets so the surviving pieces stay on the L1d-resident
    // fast path. See `Bpe.hot_table`.
    const hot_table = try Bpe.buildHotTable(allocator, new_bytes, new_offsets, new_count);
    errdefer allocator.free(hot_table);

    const pruned: Bpe = .{
        .allocator = allocator,
        .bytes = new_bytes,
        .offsets = new_offsets,
        .count = new_count,
        .by_bytes = by_bytes,
        .hot_table = hot_table,
    };

    const dropped: u32 = n - new_count;
    return .{
        .allocator = allocator,
        .pruned_bpe = pruned,
        .old_to_new = old_to_new,
        .new_to_old = new_to_old,
        .dropped_count = dropped,
    };
}

// --- tests -----------------------------------------------------------

const testing = std.testing;

const TestEntry = struct { bytes: []const u8, rank: u32 };

fn buildVocabSource(allocator: std.mem.Allocator, entries: []const TestEntry) ![]u8 {
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

fn buildByteVocab(allocator: std.mem.Allocator, extras: []const TestEntry) !Bpe {
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

test "unused_only keeps used + drops unused" {
    var bpe = try buildByteVocab(testing.allocator, &.{
        .{ .bytes = "ab", .rank = 256 },
        .{ .bytes = "cd", .rank = 257 },
        .{ .bytes = "ef", .rank = 258 },
        .{ .bytes = "gh", .rank = 259 },
    });
    defer bpe.deinit();

    // Corpus uses "ab" and "cd" merges; ef and gh untouched.
    const corpus = "abcdabcd";
    var res = try pruneBpe(testing.allocator, &bpe, corpus, .{ .strategy = .unused_only });
    defer res.deinit();

    // 256 byte tokens (always preserved) + 2 used merges = 258
    try testing.expectEqual(@as(u32, 258), res.pruned_bpe.count);
    try testing.expectEqual(@as(u32, 2), res.dropped_count);

    // Kept ids map to valid new ids.
    try testing.expect(res.old_to_new[256] < res.pruned_bpe.count);
    try testing.expect(res.old_to_new[257] < res.pruned_bpe.count);
    // Unused merges map to NULL_ID.
    try testing.expectEqual(NULL_ID, res.old_to_new[258]);
    try testing.expectEqual(NULL_ID, res.old_to_new[259]);
}

test "byte fallbacks preserved by default" {
    var bpe = try buildByteVocab(testing.allocator, &.{});
    defer bpe.deinit();

    // Empty corpus: no usage at all. Byte fallbacks should still survive.
    var res = try pruneBpe(testing.allocator, &bpe, "", .{ .strategy = .unused_only });
    defer res.deinit();

    try testing.expectEqual(@as(u32, 256), res.pruned_bpe.count);
    try testing.expectEqual(@as(u32, 0), res.dropped_count);
    var i: u32 = 0;
    while (i < 256) : (i += 1) {
        try testing.expectEqual(i, res.old_to_new[i]);
    }
}

test "preserve_ids respected" {
    var bpe = try buildByteVocab(testing.allocator, &.{
        .{ .bytes = "xy", .rank = 256 },
        .{ .bytes = "zw", .rank = 257 },
    });
    defer bpe.deinit();

    // Corpus that uses neither.
    const corpus = "hello";
    const preserve = [_]TokenId{257};
    var res = try pruneBpe(testing.allocator, &bpe, corpus, .{
        .strategy = .unused_only,
        .preserve_ids = &preserve,
    });
    defer res.deinit();

    // 256 = "xy" was dropped, 257 = "zw" was preserved.
    try testing.expectEqual(NULL_ID, res.old_to_new[256]);
    try testing.expect(res.old_to_new[257] != NULL_ID);
    try testing.expect(res.old_to_new[257] < res.pruned_bpe.count);
}

test "by_min_count drops below threshold" {
    var bpe = try buildByteVocab(testing.allocator, &.{
        .{ .bytes = "ab", .rank = 256 },
    });
    defer bpe.deinit();

    // "ab" appears 3 times.
    const corpus = "ababab";
    var res = try pruneBpe(testing.allocator, &bpe, corpus, .{
        .strategy = .by_min_count,
        .min_count = 5,
    });
    defer res.deinit();

    // ab dropped (3 < 5), bytes preserved by default.
    try testing.expectEqual(NULL_ID, res.old_to_new[256]);
    try testing.expectEqual(@as(u32, 256), res.pruned_bpe.count);
}

test "merge-dependency propagation" {
    // Vocab: 256 bytes + "ab"(256) + "abc"(257). If "ab" is dropped,
    // "abc" depends on (ab, c) and must also be dropped.
    var bpe = try buildByteVocab(testing.allocator, &.{
        .{ .bytes = "ab", .rank = 256 },
        .{ .bytes = "abc", .rank = 257 },
    });
    defer bpe.deinit();

    // Corpus where "abc" appears (so it would survive unused_only on its
    // own) but "ab" never appears as its own emitted token because the
    // greedy encoder prefers "abc". With unused_only, ab has usage=0 and
    // gets dropped; propagation must then drop abc too.
    const corpus = "abcabcabc";
    var res = try pruneBpe(testing.allocator, &bpe, corpus, .{ .strategy = .unused_only });
    defer res.deinit();

    try testing.expectEqual(NULL_ID, res.old_to_new[256]); // ab dropped
    try testing.expectEqual(NULL_ID, res.old_to_new[257]); // abc dropped via propagation
    try testing.expectEqual(@as(u32, 256), res.pruned_bpe.count);
}

test "encoding via pruned_bpe respects compacted ids" {
    var bpe = try buildByteVocab(testing.allocator, &.{
        .{ .bytes = "ab", .rank = 256 },
        .{ .bytes = "cd", .rank = 257 },
        .{ .bytes = "ef", .rank = 258 },
    });
    defer bpe.deinit();

    const corpus = "abcdabcd";
    var res = try pruneBpe(testing.allocator, &bpe, corpus, .{ .strategy = .unused_only });
    defer res.deinit();

    var out: [32]TokenId = undefined;
    const ids = res.pruned_bpe.encodeChunk("abcdab", &out);
    try testing.expect(ids.len > 0);
    for (ids) |id| {
        try testing.expect(id < res.pruned_bpe.count);
    }

    // Round-trip: bytes from compacted ids should match input.
    var decoded: std.ArrayList(u8) = .empty;
    defer decoded.deinit(testing.allocator);
    for (ids) |id| try decoded.appendSlice(testing.allocator, res.pruned_bpe.idBytes(id));
    try testing.expectEqualSlices(u8, "abcdab", decoded.items);
}
