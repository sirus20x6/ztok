//! Capcode normalizer profiling harness (post-1.20 agent C).
//!
//! Measures throughput of the three capcode-normalizer hot paths on a
//! 10 MB corpus:
//!
//!   1. Identity (no capcode; passthrough alloc+memcpy baseline)
//!   2. Nocapcode TM-compat (`tm_norm.normalizeNocapcode`)
//!   3. Full capcode TM-printable (`tm_norm.normalizeCapcode`)
//!
//! Each scenario runs --iters times after a single warmup. Reports
//! avg ms/iter, MB/s, and bytes-out / bytes-in.
//!
//! Usage:
//!   bench_capcode_profile --corpus PATH [--iters N]

const std = @import("std");
const ztok = @import("ztok");

extern "c" fn clock_gettime(clk_id: c_int, tp: *Timespec) c_int;
const Timespec = extern struct { sec: i64, nsec: i64 };
const CLOCK_MONOTONIC: c_int = 1;

fn nanosNow() u64 {
    var ts: Timespec = .{ .sec = 0, .nsec = 0 };
    _ = clock_gettime(CLOCK_MONOTONIC, &ts);
    return @intCast(ts.sec * 1_000_000_000 + ts.nsec);
}

const Args = struct {
    corpus: []const u8,
    iters: u32 = 10,
};

fn parseArgs(owned: *std.ArrayList([]const u8)) ?Args {
    var corpus: ?[]const u8 = null;
    var iters: u32 = 10;
    var i: usize = 1;
    while (i < owned.items.len) : (i += 1) {
        const a = owned.items[i];
        if (std.mem.eql(u8, a, "--corpus")) {
            corpus = owned.items[i + 1];
            i += 1;
        } else if (std.mem.eql(u8, a, "--iters")) {
            iters = std.fmt.parseInt(u32, owned.items[i + 1], 10) catch return null;
            i += 1;
        }
    }
    if (corpus == null) return null;
    return .{ .corpus = corpus.?, .iters = iters };
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var arg_iter = try std.process.Args.Iterator.initAllocator(init.minimal.args, gpa);
    defer arg_iter.deinit();

    var owned: std.ArrayList([]const u8) = .empty;
    defer {
        for (owned.items) |s| gpa.free(s);
        owned.deinit(gpa);
    }
    while (arg_iter.next()) |a| {
        const s = try gpa.dupe(u8, a);
        try owned.append(gpa, s);
    }

    const args = parseArgs(&owned) orelse {
        std.debug.print(
            "usage: bench_capcode_profile --corpus PATH [--iters N]\n",
            .{},
        );
        std.process.exit(2);
    };

    const corpus = try std.Io.Dir.cwd().readFileAlloc(io, args.corpus, gpa, .unlimited);
    defer gpa.free(corpus);

    std.debug.print(
        "bench_capcode_profile\n  corpus:  {d} bytes\n  iters:   {d}\n\n",
        .{ corpus.len, args.iters },
    );

    // ---- Scenario 1: identity (alloc+memcpy baseline) ----
    {
        // Warmup.
        const w = try gpa.alloc(u8, corpus.len);
        @memcpy(w, corpus);
        gpa.free(w);

        var total_ns: u64 = 0;
        var iter: u32 = 0;
        while (iter < args.iters) : (iter += 1) {
            const t0 = nanosNow();
            const out = try gpa.alloc(u8, corpus.len);
            @memcpy(out, corpus);
            total_ns += nanosNow() - t0;
            gpa.free(out);
        }
        const avg_ms: f64 = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(args.iters)) / 1_000_000.0;
        const mb_per_sec: f64 = (@as(f64, @floatFromInt(corpus.len)) / 1_048_576.0) / (avg_ms / 1000.0);
        std.debug.print("identity (alloc+memcpy):    {d:.2} ms/iter  {d:.2} MB/s\n", .{ avg_ms, mb_per_sec });
    }

    // ---- Scenario 2: nocapcode (tm_norm.normalizeNocapcode) ----
    {
        const w = try ztok.tm_norm.normalizeNocapcode(gpa, corpus);
        gpa.free(w);

        var total_ns: u64 = 0;
        var iter: u32 = 0;
        var last_len: usize = 0;
        while (iter < args.iters) : (iter += 1) {
            const t0 = nanosNow();
            const out = try ztok.tm_norm.normalizeNocapcode(gpa, corpus);
            total_ns += nanosNow() - t0;
            last_len = out.len;
            gpa.free(out);
        }
        const avg_ms: f64 = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(args.iters)) / 1_000_000.0;
        const mb_per_sec: f64 = (@as(f64, @floatFromInt(corpus.len)) / 1_048_576.0) / (avg_ms / 1000.0);
        std.debug.print("nocapcode (tm-compat):      {d:.2} ms/iter  {d:.2} MB/s  expand={d:.3}x\n", .{
            avg_ms, mb_per_sec, @as(f64, @floatFromInt(last_len)) / @as(f64, @floatFromInt(corpus.len)),
        });
    }

    // ---- Scenario 3: full capcode tm_printable ----
    {
        const w = try ztok.tm_norm.normalizeCapcode(gpa, corpus, .tm_printable);
        gpa.free(w);

        var total_ns: u64 = 0;
        var iter: u32 = 0;
        var last_len: usize = 0;
        while (iter < args.iters) : (iter += 1) {
            const t0 = nanosNow();
            const out = try ztok.tm_norm.normalizeCapcode(gpa, corpus, .tm_printable);
            total_ns += nanosNow() - t0;
            last_len = out.len;
            gpa.free(out);
        }
        const avg_ms: f64 = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(args.iters)) / 1_000_000.0;
        const mb_per_sec: f64 = (@as(f64, @floatFromInt(corpus.len)) / 1_048_576.0) / (avg_ms / 1000.0);
        std.debug.print("capcode (.tm_printable):    {d:.2} ms/iter  {d:.2} MB/s  expand={d:.3}x\n", .{
            avg_ms, mb_per_sec, @as(f64, @floatFromInt(last_len)) / @as(f64, @floatFromInt(corpus.len)),
        });
    }

    // ---- Scenario 4: full capcode ztok marker (.ztok) for comparison ----
    {
        const w = try ztok.capcode.encodeStyled(gpa, corpus, .ztok);
        gpa.free(w);

        var total_ns: u64 = 0;
        var iter: u32 = 0;
        var last_len: usize = 0;
        while (iter < args.iters) : (iter += 1) {
            const t0 = nanosNow();
            const out = try ztok.capcode.encodeStyled(gpa, corpus, .ztok);
            total_ns += nanosNow() - t0;
            last_len = out.len;
            gpa.free(out);
        }
        const avg_ms: f64 = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(args.iters)) / 1_000_000.0;
        const mb_per_sec: f64 = (@as(f64, @floatFromInt(corpus.len)) / 1_048_576.0) / (avg_ms / 1000.0);
        std.debug.print("capcode (.ztok, ASCII):     {d:.2} ms/iter  {d:.2} MB/s  expand={d:.3}x\n", .{
            avg_ms, mb_per_sec, @as(f64, @floatFromInt(last_len)) / @as(f64, @floatFromInt(corpus.len)),
        });
    }
}
