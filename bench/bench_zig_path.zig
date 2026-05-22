//! In-process Zig equivalent of bench_c_api.c. Mirrors the same three
//! scenarios so we can compute a C-vs-Zig ratio per scenario.

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

fn fillPseudo(buf: []u8, seed: u32) void {
    const alpha = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ " ++
        "0123456789 .,;:!? \n\n";
    var s: u32 = if (seed == 0) 1 else seed;
    for (buf) |*b| {
        s = s *% 1664525 +% 1013904223;
        b.* = alpha[(s >> 16) % alpha.len];
    }
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var owned: std.ArrayList([]const u8) = .empty;
    defer {
        for (owned.items) |s| gpa.free(s);
        owned.deinit(gpa);
    }
    var arg_iter = try std.process.Args.Iterator.initAllocator(init.minimal.args, gpa);
    defer arg_iter.deinit();
    while (arg_iter.next()) |a| try owned.append(gpa, try gpa.dupe(u8, a));

    var model_path: ?[]const u8 = null;
    var corpus_path: ?[]const u8 = null;
    var scenario: []const u8 = "all";
    var small_iters: u32 = 100_000;
    var large_bytes: usize = 10 * 1024 * 1024;
    var batch_n: usize = 10_000;
    var batch_per: usize = 1024;

    var i: usize = 1;
    while (i < owned.items.len) : (i += 1) {
        const a = owned.items[i];
        if (std.mem.eql(u8, a, "--model")) { model_path = owned.items[i + 1]; i += 1; }
        else if (std.mem.eql(u8, a, "--corpus")) { corpus_path = owned.items[i + 1]; i += 1; }
        else if (std.mem.eql(u8, a, "--scenario")) { scenario = owned.items[i + 1]; i += 1; }
        else if (std.mem.eql(u8, a, "--small-iters")) { small_iters = try std.fmt.parseInt(u32, owned.items[i + 1], 10); i += 1; }
        else if (std.mem.eql(u8, a, "--large-bytes")) { large_bytes = try std.fmt.parseInt(usize, owned.items[i + 1], 10); i += 1; }
        else if (std.mem.eql(u8, a, "--batch-n")) { batch_n = try std.fmt.parseInt(usize, owned.items[i + 1], 10); i += 1; }
        else if (std.mem.eql(u8, a, "--batch-per")) { batch_per = try std.fmt.parseInt(usize, owned.items[i + 1], 10); i += 1; }
    }

    const mp = model_path orelse {
        std.debug.print("usage: bench_zig_path --model PATH [--corpus PATH] [--scenario X]\n", .{});
        std.process.exit(2);
    };

    // Load/synthesize corpus.
    var corpus: []u8 = &.{};
    defer if (corpus.len > 0) gpa.free(corpus);
    if (corpus_path) |cp| {
        const data = try std.Io.Dir.cwd().readFileAlloc(io, cp, gpa, .unlimited);
        corpus = data;
    }
    var need: usize = large_bytes;
    if (batch_n * batch_per > need) need = batch_n * batch_per;
    if (corpus.len < need) {
        const old = corpus.len;
        corpus = try gpa.realloc(corpus, need);
        fillPseudo(corpus[old..], 0xDECAFBAD);
    }

    var bpe = try ztok.Bpe.loadTiktokenFile(gpa, mp);
    defer bpe.deinit();
    var v = ztok.Vocab.empty(gpa);
    defer v.deinit();

    const pipe: ztok.Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .cl100k,
        .model = .{ .bpe = &bpe },
        .decoder = .concat,
        .vocab = &v,
    };

    var pool = try ztok.thread_pool.BatchPool.init(gpa, null);
    defer pool.deinit();

    var stdout_buf: [4096]u8 = undefined;
    var stdout_w = std.Io.File.stdout().writer(io, &stdout_buf);
    const out = &stdout_w.interface;
    defer out.flush() catch {};

    try out.print("ztok zig-path  workers={d}  corpus={d}\n", .{ pool.workerCount(), corpus.len });

    const do_all = std.mem.eql(u8, scenario, "all");

    // single_small: 100K calls to pipe.encode with 50-byte inputs.
    if (do_all or std.mem.eql(u8, scenario, "single_small")) {
        var input: [50]u8 = undefined;
        fillPseudo(&input, 0xC0FFEE);

        var total_ids: u64 = 0;
        const t0 = nanosNow();
        var k: u32 = 0;
        while (k < small_iters) : (k += 1) {
            const ids = try pipe.encode(gpa, &input);
            total_ids += ids.len;
            gpa.free(ids);
        }
        const dt = nanosNow() - t0;
        try out.print("single_small  iters={d}  time={d:.3} ms  per-call={d:.3} us  ids/call={d:.1}  ns/op={d:.1}\n", .{
            small_iters,
            @as(f64, @floatFromInt(dt)) / 1e6,
            @as(f64, @floatFromInt(dt)) / 1e3 / @as(f64, @floatFromInt(small_iters)),
            @as(f64, @floatFromInt(total_ids)) / @as(f64, @floatFromInt(small_iters)),
            @as(f64, @floatFromInt(dt)) / @as(f64, @floatFromInt(small_iters)),
        });
    }

    // single_large: 1 call on the corpus prefix.
    if (do_all or std.mem.eql(u8, scenario, "single_large")) {
        const t0 = nanosNow();
        const ids = try pipe.encode(gpa, corpus[0..large_bytes]);
        const dt = nanosNow() - t0;
        defer gpa.free(ids);
        const mb_per_sec = (@as(f64, @floatFromInt(large_bytes)) / @as(f64, @floatFromInt(dt))) * 1e3;
        try out.print("single_large  bytes={d}  time={d:.2} ms  MB/s={d:.1}  ids={d}\n", .{
            large_bytes,
            @as(f64, @floatFromInt(dt)) / 1e6,
            mb_per_sec,
            ids.len,
        });
    }

    // batch_pooled: 10K inputs of 1 KB each.
    if (do_all or std.mem.eql(u8, scenario, "batch_pooled")) {
        const inputs = try gpa.alloc([]const u8, batch_n);
        defer gpa.free(inputs);
        for (inputs, 0..) |*s, idx| s.* = corpus[idx * batch_per .. (idx + 1) * batch_per];
        const results = try gpa.alloc([]ztok.TokenId, batch_n);
        defer gpa.free(results);
        for (results) |*r| r.* = &.{};

        const t0 = nanosNow();
        try pipe.encodeBatch(gpa, &pool, inputs, results);
        const dt = nanosNow() - t0;
        defer for (results) |r| if (r.len > 0) gpa.free(r);

        var total_ids: u64 = 0;
        for (results) |r| total_ids += r.len;
        const total_bytes: u64 = @as(u64, batch_n) * @as(u64, batch_per);
        const mb_per_sec = (@as(f64, @floatFromInt(total_bytes)) / @as(f64, @floatFromInt(dt))) * 1e3;
        try out.print("batch_pooled  n={d}  per={d}  time={d:.2} ms  MB/s={d:.1}  ids={d}\n", .{
            batch_n, batch_per,
            @as(f64, @floatFromInt(dt)) / 1e6,
            mb_per_sec,
            total_ids,
        });
    }
}
