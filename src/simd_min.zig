//! SIMD min-rank scan for the BPE merge loop.
//!
//! `scanMin(ranks)` returns the position of the lowest u32 in `ranks`
//! (first occurrence wins on ties), or `null` if every element equals
//! `RANK_INVALID`. The merge loop calls this once per merge; on a
//! typical cl100k chunk (5-10 bytes) the span scanned is tiny, but on
//! long unbroken identifiers / base64 / `aaaa...` the span can run
//! into the dozens before the heap path takes over at >64 bytes.
//!
//! Two SIMD widths are exposed:
//!   * `V = 16`  — `@Vector(16, u32)` (64 B). On AVX2 the compiler
//!                 lowers `@reduce(.Min, ...)` to two 256-bit
//!                 `vpminud` ops; on AVX-512 it folds into one
//!                 512-bit `vpminud`. On NEON it lowers to a sequence
//!                 ending in `uminv`.
//!   * `V_WIDE = 32` — `@Vector(32, u32)` (128 B). Only enabled when
//!                 the build target has AVX-512F (covers the `zmm`
//!                 register file vpminud needs); otherwise the wide
//!                 path compiles out and `scanMin` is identical to
//!                 the V=16 path. Halves the loop count on long
//!                 spans.
//!
//! Non-x86 targets (aarch64 / wasm32 / etc.) keep the V=16 path.

const std = @import("std");
const builtin = @import("builtin");

pub const Result = struct { idx: u32, rank: u32 };
pub const RANK_INVALID: u32 = std.math.maxInt(u32);

const V: usize = 16;
const V_WIDE: usize = 32;

/// Comptime gate: emit the wider `@Vector(32, u32)` path only when the
/// target CPU model advertises AVX-512F. We require F (not just VL)
/// because the 512-bit `vpminud zmm, zmm, zmm` form is in the F
/// extension; VL only matters for using AVX-512 instructions on
/// xmm/ymm widths, which we don't need here. On Zig 0.16 the
/// `std.Target.x86.featureSetHas` API takes a feature set and a
/// feature enum and returns bool.
pub const has_avx512: bool = blk: {
    if (builtin.cpu.arch != .x86_64 and builtin.cpu.arch != .x86) break :blk false;
    break :blk std.Target.x86.featureSetHas(builtin.cpu.features, .avx512f);
};

pub fn scanMinScalar(ranks: []const u32) ?Result {
    var best_idx: u32 = 0;
    var best: u32 = std.math.maxInt(u32);
    for (ranks, 0..) |r, i| {
        if (r < best) {
            best = r;
            best_idx = @intCast(i);
        }
    }
    return if (best == std.math.maxInt(u32)) null else .{ .idx = best_idx, .rank = best };
}

/// Narrow path: 16-lane reduction. Always available.
pub fn scanMinNarrow(ranks: []const u32) ?Result {
    var i: usize = 0;
    var best_rank: u32 = RANK_INVALID;
    var best_idx: u32 = 0;

    while (i + V <= ranks.len) : (i += V) {
        const vec: @Vector(V, u32) = ranks[i..][0..V].*;
        const chunk_min = @reduce(.Min, vec);
        if (chunk_min < best_rank) {
            const splat: @Vector(V, u32) = @splat(chunk_min);
            const mask = vec == splat;
            var lane: u32 = 0;
            inline for (0..V) |k| {
                if (mask[k]) {
                    lane = k;
                    break;
                }
            }
            best_rank = chunk_min;
            best_idx = @intCast(i + lane);
        }
    }

    while (i < ranks.len) : (i += 1) {
        if (ranks[i] < best_rank) {
            best_rank = ranks[i];
            best_idx = @intCast(i);
        }
    }

    return if (best_rank == RANK_INVALID) null else .{ .idx = best_idx, .rank = best_rank };
}

/// Wide path: 32-lane reduction, with a 16-lane fallthrough for the
/// middle of the buffer and a scalar tail. Only meaningful on AVX-512.
/// Compiles to the same machine code as the narrow path on AVX2 or
/// non-x86 targets because the compiler still lowers a 32-wide vector
/// to two 256-bit ops; we still go through it on those targets to
/// keep one entry point, but the wider lane count adds register
/// pressure for no win — so we gate this path off entirely there.
pub fn scanMinWide(ranks: []const u32) ?Result {
    var i: usize = 0;
    var best_rank: u32 = RANK_INVALID;
    var best_idx: u32 = 0;

    // 32-lane sweep over the long prefix.
    while (i + V_WIDE <= ranks.len) : (i += V_WIDE) {
        const vec: @Vector(V_WIDE, u32) = ranks[i..][0..V_WIDE].*;
        const chunk_min = @reduce(.Min, vec);
        if (chunk_min < best_rank) {
            const splat: @Vector(V_WIDE, u32) = @splat(chunk_min);
            const mask = vec == splat;
            var lane: u32 = 0;
            inline for (0..V_WIDE) |k| {
                if (mask[k]) {
                    lane = k;
                    break;
                }
            }
            best_rank = chunk_min;
            best_idx = @intCast(i + lane);
        }
    }

    // 16-lane tail.
    while (i + V <= ranks.len) : (i += V) {
        const vec: @Vector(V, u32) = ranks[i..][0..V].*;
        const chunk_min = @reduce(.Min, vec);
        if (chunk_min < best_rank) {
            const splat: @Vector(V, u32) = @splat(chunk_min);
            const mask = vec == splat;
            var lane: u32 = 0;
            inline for (0..V) |k| {
                if (mask[k]) {
                    lane = k;
                    break;
                }
            }
            best_rank = chunk_min;
            best_idx = @intCast(i + lane);
        }
    }

    // Scalar tail.
    while (i < ranks.len) : (i += 1) {
        if (ranks[i] < best_rank) {
            best_rank = ranks[i];
            best_idx = @intCast(i);
        }
    }

    return if (best_rank == RANK_INVALID) null else .{ .idx = best_idx, .rank = best_rank };
}

/// Public entry point. Comptime-selects the widest path the target
/// will benefit from. On AVX-512 hosts the wide path is used for
/// spans long enough to fit a 32-lane chunk; everywhere else this is
/// a direct alias for `scanMinNarrow`.
pub inline fn scanMin(ranks: []const u32) ?Result {
    if (comptime has_avx512) {
        // Short spans (the common case on cl100k: live=2..20) don't
        // even fill one 32-lane vector — go straight to the narrow
        // path so we don't pay the wide-path's extra branches for
        // nothing.
        if (ranks.len < V_WIDE) return scanMinNarrow(ranks);
        return scanMinWide(ranks);
    } else {
        return scanMinNarrow(ranks);
    }
}

// --- tests ------------------------------------------------------------

test "scanMin empty returns null" {
    const ranks = [_]u32{};
    try std.testing.expect(scanMin(&ranks) == null);
}

test "scanMin all-INVALID returns null" {
    const ranks = [_]u32{ RANK_INVALID, RANK_INVALID, RANK_INVALID, RANK_INVALID, RANK_INVALID };
    try std.testing.expect(scanMin(&ranks) == null);

    var big: [64]u32 = undefined;
    for (&big) |*x| x.* = RANK_INVALID;
    try std.testing.expect(scanMin(&big) == null);
}

test "scanMin single element" {
    const ranks = [_]u32{5};
    const r = scanMin(&ranks).?;
    try std.testing.expectEqual(@as(u32, 0), r.idx);
    try std.testing.expectEqual(@as(u32, 5), r.rank);
}

test "scanMin finds minimum at boundary" {
    const ranks = [_]u32{ 10, 10, 5, 10 };
    const r = scanMin(&ranks).?;
    try std.testing.expectEqual(@as(u32, 2), r.idx);
    try std.testing.expectEqual(@as(u32, 5), r.rank);
}

test "scanMin ties prefer lower index" {
    const ranks = [_]u32{ 5, 5, 5 };
    const r = scanMin(&ranks).?;
    try std.testing.expectEqual(@as(u32, 0), r.idx);
    try std.testing.expectEqual(@as(u32, 5), r.rank);

    var aligned: [16]u32 = undefined;
    for (&aligned) |*x| x.* = 7;
    const r2 = scanMin(&aligned).?;
    try std.testing.expectEqual(@as(u32, 0), r2.idx);
    try std.testing.expectEqual(@as(u32, 7), r2.rank);

    var long: [40]u32 = undefined;
    for (&long) |*x| x.* = 100;
    long[5] = 3;
    long[20] = 3;
    long[33] = 3;
    const r3 = scanMin(&long).?;
    try std.testing.expectEqual(@as(u32, 5), r3.idx);
    try std.testing.expectEqual(@as(u32, 3), r3.rank);
}

test "scanMin matches scanMinScalar on random data" {
    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const rng = prng.random();
    var buf: [256]u32 = undefined;

    var trial: usize = 0;
    while (trial < 1000) : (trial += 1) {
        const len = rng.uintLessThan(usize, 257);
        for (buf[0..len]) |*x| {
            if (rng.uintLessThan(u8, 5) == 0) {
                x.* = RANK_INVALID;
            } else {
                x.* = rng.int(u32);
            }
        }
        const a = scanMin(buf[0..len]);
        const b = scanMinScalar(buf[0..len]);
        if (a == null or b == null) {
            try std.testing.expect(a == null and b == null);
        } else {
            try std.testing.expectEqual(b.?.idx, a.?.idx);
            try std.testing.expectEqual(b.?.rank, a.?.rank);
        }
    }
}

test "scanMin handles V-aligned and tail" {
    {
        var a: [16]u32 = undefined;
        for (&a, 0..) |*x, i| x.* = @intCast(100 + i);
        a[9] = 1;
        const r = scanMin(&a).?;
        try std.testing.expectEqual(@as(u32, 9), r.idx);
        try std.testing.expectEqual(@as(u32, 1), r.rank);
    }
    {
        var a: [17]u32 = undefined;
        for (&a, 0..) |*x, i| x.* = @intCast(100 + i);
        a[16] = 1;
        const r = scanMin(&a).?;
        try std.testing.expectEqual(@as(u32, 16), r.idx);
        try std.testing.expectEqual(@as(u32, 1), r.rank);
    }
    {
        var a: [31]u32 = undefined;
        for (&a, 0..) |*x, i| x.* = @intCast(100 + i);
        a[30] = 1;
        const r = scanMin(&a).?;
        try std.testing.expectEqual(@as(u32, 30), r.idx);
        try std.testing.expectEqual(@as(u32, 1), r.rank);
    }
    {
        var a: [32]u32 = undefined;
        for (&a, 0..) |*x, i| x.* = @intCast(100 + i);
        a[17] = 1;
        const r = scanMin(&a).?;
        try std.testing.expectEqual(@as(u32, 17), r.idx);
        try std.testing.expectEqual(@as(u32, 1), r.rank);
    }
}

test "scanMin tail-only path" {
    const ranks = [_]u32{ 9, 4, 7 };
    const r = scanMin(&ranks).?;
    try std.testing.expectEqual(@as(u32, 1), r.idx);
    try std.testing.expectEqual(@as(u32, 4), r.rank);
}

// --- AVX-512 wide-path equivalence tests -----------------------------
//
// These exercise scanMinNarrow and scanMinWide directly, regardless of
// the host's AVX-512 status. The wide-path implementation is portable
// (it just builds wider Zig vectors); the perf benefit is hardware-
// dependent but the result should be bit-identical to the narrow path
// on every architecture.

test "wide path matches narrow on aligned span" {
    // 64 elements: hits one V_WIDE iteration plus one V iteration.
    var a: [64]u32 = undefined;
    for (&a, 0..) |*x, i| x.* = @intCast(1000 + i);
    a[37] = 0; // unique minimum mid-buffer
    const narrow = scanMinNarrow(&a).?;
    const wide = scanMinWide(&a).?;
    try std.testing.expectEqual(narrow.idx, wide.idx);
    try std.testing.expectEqual(narrow.rank, wide.rank);
    try std.testing.expectEqual(@as(u32, 37), wide.idx);
}

test "wide path edge cases length 1/15/16/17/31/32/33" {
    inline for (.{ 1, 15, 16, 17, 31, 32, 33 }) |len| {
        var a: [len]u32 = undefined;
        for (&a, 0..) |*x, i| x.* = @intCast(500 + i);
        // Put the min at the last position to force the scan to walk
        // the full span.
        a[len - 1] = 1;
        const narrow = scanMinNarrow(&a).?;
        const wide = scanMinWide(&a).?;
        try std.testing.expectEqual(narrow.idx, wide.idx);
        try std.testing.expectEqual(narrow.rank, wide.rank);
        try std.testing.expectEqual(@as(u32, len - 1), wide.idx);
    }
}

test "wide path all-equal preserves first-min tiebreak" {
    inline for (.{ 16, 31, 32, 33, 64, 96 }) |len| {
        var a: [len]u32 = undefined;
        for (&a) |*x| x.* = 7;
        const wide = scanMinWide(&a).?;
        try std.testing.expectEqual(@as(u32, 0), wide.idx);
        try std.testing.expectEqual(@as(u32, 7), wide.rank);
    }
}

test "wide path random equivalence up to 8192 elements" {
    var prng = std.Random.DefaultPrng.init(0xDEADBEEF);
    const rng = prng.random();
    var buf: [8192]u32 = undefined;

    var trial: usize = 0;
    while (trial < 200) : (trial += 1) {
        const len = rng.uintLessThan(usize, 8193);
        for (buf[0..len]) |*x| {
            if (rng.uintLessThan(u8, 8) == 0) {
                x.* = RANK_INVALID;
            } else {
                x.* = rng.int(u32);
            }
        }
        const narrow = scanMinNarrow(buf[0..len]);
        const wide = scanMinWide(buf[0..len]);
        const scalar = scanMinScalar(buf[0..len]);
        if (narrow == null or wide == null or scalar == null) {
            try std.testing.expect(narrow == null and wide == null and scalar == null);
        } else {
            try std.testing.expectEqual(scalar.?.idx, narrow.?.idx);
            try std.testing.expectEqual(scalar.?.idx, wide.?.idx);
            try std.testing.expectEqual(scalar.?.rank, narrow.?.rank);
            try std.testing.expectEqual(scalar.?.rank, wide.?.rank);
        }
    }
}
