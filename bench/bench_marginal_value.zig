//! Marginal-value scoring microbench: v2 (rebuild Monster per piece)
//! vs v3 (mask a piece in a single shared Monster).
//!
//! Builds a synthetic vocab of `vocab_size` pieces over a ~`corpus_kb` KB
//! corpus, runs `iters` rounds of marginal-value scoring with each path,
//! and prints the ratio. Self-contained — no external corpus needed.
//!
//! Run via:  zig build bench-marginal-value -Doptimize=ReleaseFast
//! Or directly: zig run -OReleaseFast bench/bench_marginal_value.zig --dep ztok -Mztok=src/root.zig

const std = @import("std");
const ztok = @import("ztok");
const Monster = ztok.Monster;
const TokenId = ztok.TokenId;
const BatchPool = ztok.thread_pool.BatchPool;

extern "c" fn clock_gettime(clk_id: c_int, tp: *Timespec) c_int;
const Timespec = extern struct { sec: i64, nsec: i64 };
const CLOCK_MONOTONIC: c_int = 1;

fn nanosNow() u64 {
    var ts: Timespec = .{ .sec = 0, .nsec = 0 };
    _ = clock_gettime(CLOCK_MONOTONIC, &ts);
    return @intCast(ts.sec * 1_000_000_000 + ts.nsec);
}

fn buildVocab(a: std.mem.Allocator, vocab_size: u32, corpus: []const u8) !struct {
    bytes: []u8,
    offsets: []u32,
    is_byte: []bool,
} {
    var bytes: std.ArrayList(u8) = .empty;
    errdefer bytes.deinit(a);
    var offsets: std.ArrayList(u32) = .empty;
    errdefer offsets.deinit(a);
    var is_byte: std.ArrayList(bool) = .empty;
    errdefer is_byte.deinit(a);
    try offsets.append(a, 0);

    // 256 byte pieces (pinned).
    var b: u32 = 0;
    while (b < 256) : (b += 1) {
        try bytes.append(a, @intCast(b));
        try offsets.append(a, @intCast(bytes.items.len));
        try is_byte.append(a, true);
    }

    // Mine substrings (length 2..8) by frequency.
    var counts = std.StringHashMap(u32).init(a);
    defer counts.deinit();
    var i: usize = 0;
    while (i < corpus.len) : (i += 1) {
        const limit = @min(corpus.len, i + 8);
        var j: usize = i + 2;
        while (j <= limit) : (j += 1) {
            const sub = corpus[i..j];
            const gop = try counts.getOrPut(sub);
            if (!gop.found_existing) gop.value_ptr.* = 0;
            gop.value_ptr.* +%= 1;
        }
    }

    const Entry = struct { sub: []const u8, c: u32 };
    var entries: std.ArrayList(Entry) = .empty;
    defer entries.deinit(a);
    try entries.ensureTotalCapacity(a, counts.count());
    var it = counts.iterator();
    while (it.next()) |e| {
        if (e.value_ptr.* >= 2) try entries.append(a, .{ .sub = e.key_ptr.*, .c = e.value_ptr.* });
    }
    const lessByScore = struct {
        fn lt(_: void, x: Entry, y: Entry) bool {
            const sx = @as(u64, x.c) * x.sub.len;
            const sy = @as(u64, y.c) * y.sub.len;
            if (sx != sy) return sx > sy;
            return std.mem.lessThan(u8, x.sub, y.sub);
        }
    };
    std.mem.sort(Entry, entries.items, {}, lessByScore.lt);

    const target_extra = vocab_size - 256;
    const keep = @min(entries.items.len, @as(usize, target_extra));
    var k: usize = 0;
    while (k < keep) : (k += 1) {
        try bytes.appendSlice(a, entries.items[k].sub);
        try offsets.append(a, @intCast(bytes.items.len));
        try is_byte.append(a, false);
    }

    return .{
        .bytes = try bytes.toOwnedSlice(a),
        .offsets = try offsets.toOwnedSlice(a),
        .is_byte = try is_byte.toOwnedSlice(a),
    };
}

fn buildMonsterFromArrays(
    a: std.mem.Allocator,
    bytes: []const u8,
    offsets: []const u32,
) !Monster {
    var builder = Monster.Builder.init(a);
    defer builder.deinit();
    var id: u32 = 0;
    const n: u32 = @intCast(offsets.len - 1);
    while (id < n) : (id += 1) {
        const start = offsets[id];
        const end = offsets[id + 1];
        _ = try builder.addToken(bytes[start..end]);
    }
    return try builder.finalize(0);
}

fn pieceOf(bytes: []const u8, offsets: []const u32, id: u32) []const u8 {
    return bytes[offsets[id]..offsets[id + 1]];
}

// v2: build a temp Monster excluding piece P, encode P's bytes.
fn scoreV2(
    a: std.mem.Allocator,
    bytes: []const u8,
    offsets: []const u32,
    is_byte: []const bool,
    out_alt: []u32,
) !void {
    const v: u32 = @intCast(is_byte.len);
    var pid: u32 = 0;
    while (pid < v) : (pid += 1) {
        if (is_byte[pid]) {
            out_alt[pid] = 0;
            continue;
        }
        const piece_bytes = pieceOf(bytes, offsets, pid);

        // Build excluding pid.
        var builder = Monster.Builder.init(a);
        defer builder.deinit();
        var id: u32 = 0;
        while (id < v) : (id += 1) {
            if (id == pid) continue;
            _ = try builder.addToken(pieceOf(bytes, offsets, id));
        }
        var m = try builder.finalize(0);
        defer m.deinit();

        var buf: [256]TokenId = undefined;
        const ids = try m.encodeChunk(a, piece_bytes, buf[0..]);
        out_alt[pid] = @intCast(ids.len);
    }
}

// v3: reuse a single Monster; flip mask[P] per encode.
fn scoreV3(
    a: std.mem.Allocator,
    monster: *const Monster,
    bytes: []const u8,
    offsets: []const u32,
    is_byte: []const bool,
    out_alt: []u32,
) !void {
    const v: u32 = @intCast(is_byte.len);
    const mask = try a.alloc(u8, v);
    defer a.free(mask);
    @memset(mask, 0);
    var solo: Monster = monster.*;
    solo.mask = mask;

    var pid: u32 = 0;
    while (pid < v) : (pid += 1) {
        if (is_byte[pid]) {
            out_alt[pid] = 0;
            continue;
        }
        mask[pid] = 1;
        var buf: [256]TokenId = undefined;
        const piece_bytes = pieceOf(bytes, offsets, pid);
        const ids = try solo.encodeChunk(a, piece_bytes, buf[0..]);
        mask[pid] = 0;
        out_alt[pid] = @intCast(ids.len);
    }
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    // Synthetic corpus: ~50 KB of pseudo-text.
    const corpus_kb: u32 = 50;
    const corpus_bytes = corpus_kb * 1024;
    const corpus = try gpa.alloc(u8, corpus_bytes);
    defer gpa.free(corpus);
    var prng = std.Random.DefaultPrng.init(0xCAFE);
    const rnd = prng.random();
    // Generate text-ish content: words of 3-9 lowercase letters separated by spaces.
    var p: usize = 0;
    while (p < corpus.len) {
        const word_len = rnd.intRangeAtMost(usize, 3, 9);
        const end = @min(corpus.len, p + word_len);
        var q: usize = p;
        while (q < end) : (q += 1) corpus[q] = 'a' + rnd.intRangeAtMost(u8, 0, 25);
        p = end;
        if (p < corpus.len) {
            corpus[p] = ' ';
            p += 1;
        }
    }

    var stdout_buf: [4096]u8 = undefined;
    var stdout_w = std.Io.File.stdout().writer(io, &stdout_buf);
    const out = &stdout_w.interface;
    defer out.flush() catch {};

    try out.print("v2 (rebuild) vs v3 (mask) marginal-value scoring microbench\n", .{});
    try out.print("  corpus: {d} bytes\n", .{corpus.len});
    try out.print("  pieces      v2 ms/iter   v3 ms/iter   speedup   alt-match\n", .{});

    const sizes = [_]u32{ 100, 250, 500, 1000 };
    const warm_iters: u32 = 1;
    const iters: u32 = 3;

    for (sizes) |non_byte| {
        const vocab_size: u32 = non_byte + 256;
        const arrays = try buildVocab(gpa, vocab_size, corpus);
        defer gpa.free(arrays.bytes);
        defer gpa.free(arrays.offsets);
        defer gpa.free(arrays.is_byte);

        var monster = try buildMonsterFromArrays(gpa, arrays.bytes, arrays.offsets);
        defer monster.deinit();

        const alt_v2 = try gpa.alloc(u32, monster.count);
        defer gpa.free(alt_v2);
        const alt_v3 = try gpa.alloc(u32, monster.count);
        defer gpa.free(alt_v3);

        // Warm-up
        var w: u32 = 0;
        while (w < warm_iters) : (w += 1) {
            try scoreV2(gpa, arrays.bytes, arrays.offsets, arrays.is_byte, alt_v2);
            try scoreV3(gpa, &monster, arrays.bytes, arrays.offsets, arrays.is_byte, alt_v3);
        }

        const t2_start = nanosNow();
        var k: u32 = 0;
        while (k < iters) : (k += 1) {
            try scoreV2(gpa, arrays.bytes, arrays.offsets, arrays.is_byte, alt_v2);
        }
        const t2 = nanosNow() - t2_start;

        const t3_start = nanosNow();
        k = 0;
        while (k < iters) : (k += 1) {
            try scoreV3(gpa, &monster, arrays.bytes, arrays.offsets, arrays.is_byte, alt_v3);
        }
        const t3 = nanosNow() - t3_start;

        var match: usize = 0;
        var mismatch: usize = 0;
        var i: usize = 0;
        while (i < monster.count) : (i += 1) {
            if (alt_v2[i] == alt_v3[i]) match += 1 else mismatch += 1;
        }

        const t2_ms = @as(f64, @floatFromInt(t2)) / 1e6 / @as(f64, @floatFromInt(iters));
        const t3_ms = @as(f64, @floatFromInt(t3)) / 1e6 / @as(f64, @floatFromInt(iters));
        const ratio = t2_ms / t3_ms;

        try out.print(
            "  {d:>6}     {d:>10.2}   {d:>10.3}   {d:>6.2}x   {d}/{d}\n",
            .{ non_byte, t2_ms, t3_ms, ratio, match, mismatch },
        );
    }
}
