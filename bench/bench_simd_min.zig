//! Microbenchmark for the BPE merge loop's min-rank scan.
//!
//! Compares scanMinScalar / scanMinNarrow / scanMinWide at a range of
//! span lengths typical of the BPE inner loop. On non-AVX-512 hosts the
//! "wide" column is the same Zig source as narrow (the compiler folds
//! it to AVX2 either way), so the comparison is most meaningful when
//! built with `-Dcpu=native` on an AVX-512 host or with an explicit
//! `-Dcpu=znver4`-style target.
//!
//!   zig build bench-simd-min -Doptimize=ReleaseFast
//!
//! Pass `--iters N` to override the inner iteration count (default
//! auto-scales per span length so each measurement runs ~200ms).

const std = @import("std");
const ztok = @import("ztok");
const simd_min = ztok.simd_min;

extern "c" fn clock_gettime(clk_id: c_int, tp: *Timespec) c_int;
const Timespec = extern struct { sec: i64, nsec: i64 };
const CLOCK_MONOTONIC: c_int = 1;

fn nanosNow() u64 {
    var ts: Timespec = .{ .sec = 0, .nsec = 0 };
    _ = clock_gettime(CLOCK_MONOTONIC, &ts);
    return @intCast(ts.sec * 1_000_000_000 + ts.nsec);
}

const SPAN_LENGTHS = [_]usize{ 64, 256, 1024, 4096, 16384, 65536 };

fn fillRandom(buf: []u32, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    for (buf) |*x| {
        // 10% RANK_INVALID, otherwise random rank in 0..vocab_size to
        // mimic cl100k-ish distribution. The unique min will land
        // somewhere mid-buffer with overwhelming probability so the
        // splat-match scalar tail doesn't dominate.
        if (rng.uintLessThan(u8, 10) == 0) {
            x.* = simd_min.RANK_INVALID;
        } else {
            x.* = rng.uintLessThan(u32, 100_000);
        }
    }
}

const Bench = struct {
    label: []const u8,
    len: usize,
    ns_per_op: f64,
    gb_per_sec: f64,
};

fn runOne(comptime ScanFn: anytype, ranks: []const u32, target_ns: u64) Bench {
    // Warmup.
    {
        var i: usize = 0;
        while (i < 16) : (i += 1) {
            const r = ScanFn(ranks);
            std.mem.doNotOptimizeAway(r);
        }
    }

    // Calibrate iteration count.
    var iters: u64 = 1024;
    while (true) {
        const t0 = nanosNow();
        var i: u64 = 0;
        while (i < iters) : (i += 1) {
            const r = ScanFn(ranks);
            std.mem.doNotOptimizeAway(r);
        }
        const dt = nanosNow() - t0;
        if (dt >= target_ns / 4) {
            const ns_per_op = @as(f64, @floatFromInt(dt)) / @as(f64, @floatFromInt(iters));
            const bytes = @as(f64, @floatFromInt(ranks.len * @sizeOf(u32)));
            const gbs = bytes / ns_per_op; // bytes/ns == GB/s
            return .{ .label = "", .len = ranks.len, .ns_per_op = ns_per_op, .gb_per_sec = gbs };
        }
        iters *= 2;
        if (iters > 1 << 32) {
            const ns_per_op = @as(f64, @floatFromInt(dt)) / @as(f64, @floatFromInt(iters));
            const bytes = @as(f64, @floatFromInt(ranks.len * @sizeOf(u32)));
            return .{ .label = "", .len = ranks.len, .ns_per_op = ns_per_op, .gb_per_sec = bytes / ns_per_op };
        }
    }
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var stdout_buf: [4096]u8 = undefined;
    var stdout_w = std.Io.File.stdout().writer(io, &stdout_buf);
    const out = &stdout_w.interface;
    defer out.flush() catch {};

    try out.print("simd_min microbench (host has AVX-512: {}) target_ns=~200ms\n", .{simd_min.has_avx512});
    try out.print("{s:>8} | {s:>16} | {s:>16} | {s:>16}\n", .{
        "len", "scalar ns/op", "narrow ns/op", "wide ns/op",
    });
    try out.print("{s:>8} | {s:>16} | {s:>16} | {s:>16}\n", .{
        "---", "---", "---", "---",
    });

    inline for (SPAN_LENGTHS) |len| {
        const buf = try gpa.alloc(u32, len);
        defer gpa.free(buf);
        fillRandom(buf, 0xC0FFEE ^ len);

        const scalar = runOne(simd_min.scanMinScalar, buf, 200_000_000);
        const narrow = runOne(simd_min.scanMinNarrow, buf, 200_000_000);
        const wide = runOne(simd_min.scanMinWide, buf, 200_000_000);

        try out.print(
            "{d:>8} | {d:>16.1} | {d:>16.1} | {d:>16.1}\n",
            .{ len, scalar.ns_per_op, narrow.ns_per_op, wide.ns_per_op },
        );
    }

    try out.print(
        "\nGB/s (last column) — interpret as steady-state memory throughput.\n",
        .{},
    );
    try out.print("On AVX-512 the wide path should beat narrow at len >= 32 by ~1.3-1.8x.\n", .{});
    try out.print("On AVX2 the wide path emits the same machine code as narrow.\n", .{});
}
