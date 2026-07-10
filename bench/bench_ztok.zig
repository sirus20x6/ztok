//! ztok benchmark vs tiktoken / HF tokenizers.
//! Loads cl100k_base, encodes the corpus N times, reports throughput.
//! Single-threaded by default; pass --batch N to test batch encode.

const std = @import("std");
const ztok = @import("ztok");

// Zig 0.16's std.time was gutted; std.Io.Clock works but is heavy for a
// simple wall-clock-elapsed measurement. clock_gettime via libc is two
// lines and accurate enough.
extern "c" fn clock_gettime(clk_id: c_int, tp: *Timespec) c_int;
const Timespec = extern struct { sec: i64, nsec: i64 };
const CLOCK_MONOTONIC: c_int = 1;

fn nanosNow() u64 {
    var ts: Timespec = .{ .sec = 0, .nsec = 0 };
    _ = clock_gettime(CLOCK_MONOTONIC, &ts);
    return @intCast(ts.sec * 1_000_000_000 + ts.nsec);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var arg_iter = try std.process.Args.Iterator.initAllocator(init.minimal.args, gpa);
    defer arg_iter.deinit();

    var corpus_path: ?[]const u8 = null;
    var model_path: ?[]const u8 = null;
    var iters: u32 = 5;
    var batch: u32 = 0;
    var workers: ?u32 = null;
    var perworker: bool = false;
    var pin_physical: bool = false;
    var use_hugepages: bool = false;
    var numa_aware: bool = false;
    // --small-strings N: synthetic mode. Slice the corpus into ~50-byte
    // fragments and encode each one N times through both `encode` (one-
    // shot arena per call) and `encodeWithScratch` (persistent arena
    // reused across calls). Reveals the per-chunk allocation overhead
    // that the persistent-scratch path eliminates.
    var small_strings: u32 = 0;
    // 1.16 hot-table opt-out: bench loads via the bare `loadTiktokenFile`
    // which now defaults the 1.15 hot table OFF. Flip `--hot-table` to
    // re-enable it (the right choice for batch ×N + pin shapes).
    var hot_table: bool = false;
    var owned: std.ArrayList([]const u8) = .empty;
    defer {
        for (owned.items) |s| gpa.free(s);
        owned.deinit(gpa);
    }
    while (arg_iter.next()) |a| {
        const s = try gpa.dupe(u8, a);
        try owned.append(gpa, s);
    }
    var i: usize = 1;
    while (i < owned.items.len) : (i += 1) {
        const a = owned.items[i];
        if (std.mem.eql(u8, a, "--corpus")) {
            corpus_path = owned.items[i + 1];
            i += 1;
        } else if (std.mem.eql(u8, a, "--model")) {
            model_path = owned.items[i + 1];
            i += 1;
        } else if (std.mem.eql(u8, a, "--iters")) {
            iters = try std.fmt.parseInt(u32, owned.items[i + 1], 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--batch")) {
            batch = try std.fmt.parseInt(u32, owned.items[i + 1], 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--workers")) {
            workers = try std.fmt.parseInt(u32, owned.items[i + 1], 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--per-worker")) {
            perworker = true;
        } else if (std.mem.eql(u8, a, "--pin-physical")) {
            pin_physical = true;
        } else if (std.mem.eql(u8, a, "--hugepages")) {
            use_hugepages = true;
        } else if (std.mem.eql(u8, a, "--numa-aware")) {
            numa_aware = true;
        } else if (std.mem.eql(u8, a, "--small-strings")) {
            small_strings = try std.fmt.parseInt(u32, owned.items[i + 1], 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--hot-table")) {
            hot_table = true;
        }
    }

    const cp = corpus_path orelse {
        std.debug.print("usage: bench --model PATH --corpus PATH [--iters N] [--batch N]\n", .{});
        std.process.exit(2);
    };
    const mp = model_path orelse {
        std.debug.print("usage: bench --model PATH --corpus PATH [--iters N] [--batch N]\n", .{});
        std.process.exit(2);
    };

    const corpus = try std.Io.Dir.cwd().readFileAlloc(io, cp, gpa, .unlimited);
    defer gpa.free(corpus);

    var bpe = try ztok.Bpe.loadTiktokenFileWithOptions(gpa, mp, .{ .hot_table = hot_table });
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

    var stdout_buf: [4096]u8 = undefined;
    var stdout_w = std.Io.File.stdout().writer(io, &stdout_buf);
    const out = &stdout_w.interface;
    defer out.flush() catch {};

    try out.print("corpus:  {d} bytes\n", .{corpus.len});
    try out.print("vocab:   {d} tokens, {d} bytes\n", .{ bpe.count, bpe.bytes.len });

    if (small_strings > 0) {
        // Carve the corpus into ~50-byte fragments. `cl100k` pretok will
        // re-split inside each one, so this mirrors the small-string
        // workload (one chat message per encode call). We run `iters`
        // passes over `small_strings` slices, comparing the one-shot
        // arena path (`encode`) to the persistent-arena path
        // (`encodeWithScratch`).
        //
        // Larger slice sizes (300+) trip the BPE heap-spillover path
        // inside `encodeChunkScratch` for long unbroken runs, which is
        // where the per-call alloc savings are biggest. Set
        // `slice_bytes` via `--workers` (re-purposed in this mode) to
        // override the default 50.
        const N = small_strings;
        const slice_len: usize = if (workers) |w| @max(@as(u32, 8), w) else 50;
        const slices = try gpa.alloc([]const u8, N);
        defer gpa.free(slices);
        for (slices, 0..) |*s, idx| {
            const start = (idx * slice_len) % @max(1, corpus.len - slice_len);
            const end = @min(start + slice_len, corpus.len);
            s.* = corpus[start..end];
        }

        // Path A: legacy. One-shot arena per call via `encode`.
        var total_a: u64 = 0;
        const t_a0 = nanosNow();
        var ka: u32 = 0;
        while (ka < iters) : (ka += 1) {
            for (slices) |inp| {
                const ids = try pipe.encode(gpa, inp);
                total_a += ids.len;
                gpa.free(ids);
            }
        }
        const ns_a = nanosNow() - t_a0;

        // Path B: persistent ScratchArena reused across all calls.
        var scratch = ztok.pipeline.ScratchArena.init(gpa);
        defer scratch.deinit();
        var total_b: u64 = 0;
        const t_b0 = nanosNow();
        var kb: u32 = 0;
        while (kb < iters) : (kb += 1) {
            for (slices) |inp| {
                const ids = try pipe.encodeWithScratch(gpa, inp, &scratch);
                total_b += ids.len;
                gpa.free(ids);
            }
        }
        const ns_b = nanosNow() - t_b0;

        const bytes_total = @as(u64, N) * slice_len * iters;
        const mb_a = @as(f64, @floatFromInt(bytes_total)) / (@as(f64, @floatFromInt(ns_a)) / 1e9) / 1e6;
        const mb_b = @as(f64, @floatFromInt(bytes_total)) / (@as(f64, @floatFromInt(ns_b)) / 1e9) / 1e6;
        try out.print("mode:           small-strings N={d} iters={d} slice={d}B\n", .{ N, iters, slice_len });
        try out.print("encode:         {d:.2} ms  ({d:.1} MB/s)  ids={d}\n", .{
            @as(f64, @floatFromInt(ns_a)) / 1e6, mb_a, total_a,
        });
        try out.print("encodeWithScr:  {d:.2} ms  ({d:.1} MB/s)  ids={d}\n", .{
            @as(f64, @floatFromInt(ns_b)) / 1e6, mb_b, total_b,
        });
        try out.print("speedup:        {d:.3}x\n", .{
            @as(f64, @floatFromInt(ns_a)) / @as(f64, @floatFromInt(ns_b)),
        });
        try out.print("scratch peak:   {d} bytes\n", .{scratch.peakBytes()});
    } else if (batch == 0) {
        // Single-threaded encode loop
        var total_ids: u64 = 0;
        const t0 = nanosNow();
        var k: u32 = 0;
        while (k < iters) : (k += 1) {
            const ids = try pipe.encode(gpa, corpus);
            total_ids += ids.len;
            gpa.free(ids);
        }
        const elapsed_ns = nanosNow() - t0;
        const bytes_total = @as(u64, corpus.len) * iters;
        const mb_per_sec = @as(f64, @floatFromInt(bytes_total)) / (@as(f64, @floatFromInt(elapsed_ns)) / 1e9) / 1e6;
        const tokens_per_sec = @as(f64, @floatFromInt(total_ids)) / (@as(f64, @floatFromInt(elapsed_ns)) / 1e9);
        try out.print("mode:    single-thread, iters={d}\n", .{iters});
        try out.print("ids/run: {d}\n", .{total_ids / iters});
        try out.print("time:    {d:.2} ms\n", .{@as(f64, @floatFromInt(elapsed_ns)) / 1e6});
        try out.print("MB/s:    {d:.1}\n", .{mb_per_sec});
        try out.print("tok/s:   {d:.0}\n", .{tokens_per_sec});
    } else {
        // Chunked encode: split corpus into `batch` pre-tokenizer-safe
        // chunks via `Pipeline.encodeChunked`, which snaps cut points to
        // boundaries the pre-tokenizer would have produced anyway. Output
        // is bit-identical to single-shot.
        var pool = try ztok.thread_pool.BatchPool.initWithOptions(gpa, workers, .{
            .pin_to_physical_cores = pin_physical,
            .use_hugepages = use_hugepages,
            .numa_aware = numa_aware,
        });
        defer pool.deinit();

        var total_ids: u64 = 0;
        const t0 = nanosNow();
        var k: u32 = 0;
        while (k < iters) : (k += 1) {
            const ids = try pipe.encodeChunked(gpa, &pool, corpus, batch);
            total_ids += ids.len;
            gpa.free(ids);
        }
        const elapsed_ns = nanosNow() - t0;
        const bytes_total = @as(u64, corpus.len) * iters;
        const mb_per_sec = @as(f64, @floatFromInt(bytes_total)) / (@as(f64, @floatFromInt(elapsed_ns)) / 1e9) / 1e6;
        const tokens_per_sec = @as(f64, @floatFromInt(total_ids)) / (@as(f64, @floatFromInt(elapsed_ns)) / 1e9);
        try out.print("mode:    batch, chunks={d}, workers={d}, iters={d}, pin={s}\n", .{
            batch, pool.workerCount(), iters,
            switch (pool.pin_diagnostic) {
                .disabled => "off",
                .pinned => "physical",
                .fallback_no_topology => "fallback_no_topology",
                .fallback_partial => "fallback_partial",
            },
        });
        try out.print("ids/run: {d}\n", .{total_ids / iters});
        try out.print("time:    {d:.2} ms\n", .{@as(f64, @floatFromInt(elapsed_ns)) / 1e6});
        try out.print("MB/s:    {d:.1}\n", .{mb_per_sec});
        try out.print("tok/s:   {d:.0}\n", .{tokens_per_sec});

        if (perworker) {
            try runPerWorkerHistogram(gpa, io, out, &pipe, &pool, corpus, batch, iters);
        }
    }
}

// === Per-worker instrumentation ===
//
// Splits the corpus into `n_chunks` byte-equal slices (snapped to
// pre-tok-safe boundaries the same way encodeChunked does), then calls
// encodeBatch with timing wrappers around each work item. Reports:
//   - per-worker active time (sum of work items run by that worker)
//   - per-worker job count
//   - wall time vs sum-of-active (overhead = spawn + atomic-cursor stalls)

fn runPerWorkerHistogram(
    gpa: std.mem.Allocator,
    io: std.Io,
    out: anytype,
    pipe: *const ztok.Pipeline,
    pool: *ztok.thread_pool.BatchPool,
    corpus: []const u8,
    n_chunks: u32,
    iters: u32,
) !void {
    _ = io;
    const n_workers = pool.workerCount();

    // Compute boundaries the same way encodeChunked does (nominal-share
    // snapped via findSafeCut).
    const chunks: usize = if (n_chunks == 0) 1 else n_chunks;
    const boundaries = try gpa.alloc(usize, chunks + 1);
    defer gpa.free(boundaries);
    boundaries[0] = 0;
    boundaries[chunks] = corpus.len;

    const nominal = corpus.len / chunks;
    const window = @min(@max(nominal / 2, 256), 64 * 1024);
    var bi: usize = 1;
    while (bi < chunks) : (bi += 1) {
        const desired = bi * nominal;
        const snapped = pipe.pre_tokenizer.findSafeCut(corpus, desired, window) orelse blk: {
            var p = desired;
            while (p > 0 and (corpus[p] & 0xC0) == 0x80) p -= 1;
            break :blk p;
        };
        boundaries[bi] = @max(snapped, boundaries[bi - 1]);
    }

    const inputs = try gpa.alloc([]const u8, chunks);
    defer gpa.free(inputs);
    for (0..chunks) |k| inputs[k] = corpus[boundaries[k]..boundaries[k + 1]];

    const Stats = struct {
        active_ns: u64 = 0,
        gpa_ns: u64 = 0,
        jobs: u64 = 0,
        bytes: u64 = 0,
        min_ns: u64 = std.math.maxInt(u64),
        max_ns: u64 = 0,
    };
    // 64-byte-padded slots to remove false sharing for the histogram itself.
    const Padded = struct {
        s: Stats align(64) = .{},
        _pad: [64 - @sizeOf(Stats) % 64]u8 = undefined,
    };
    const stats = try gpa.alloc(Padded, n_workers);
    defer gpa.free(stats);
    for (stats) |*p| p.* = .{};

    const InstrCtx = struct {
        pipe: *const ztok.Pipeline,
        inputs: []const []const u8,
        results: [][]ztok.TokenId,
        ra: std.mem.Allocator,
        pool: *ztok.thread_pool.BatchPool,
        stats: []Padded,
        errored: std.atomic.Value(u32) = .init(0),

        const Self = @This();

        pub fn run(c: *Self, idx: usize, worker_idx: usize) void {
            const t_start = nanosNow();

            // Mirror encodeBatch's per-job path exactly: per-worker arena
            // scratch + GPA result_allocator for the `out` buffer. This is
            // the production hot path for parallel encode.
            const scratch = c.pool.resetArena(worker_idx);

            const t_alloc_start = nanosNow();
            const ids = c.pipe.encode(c.ra, c.inputs[idx]) catch {
                _ = c.errored.fetchAdd(1, .acq_rel);
                return;
            };
            const t_alloc_end = nanosNow();
            _ = scratch; // (production encodeBatch uses scratch for normalizer/pretok via encodeText — pipe.encode takes only one allocator, so this is best-case for now)

            c.results[idx] = ids;

            const elapsed = nanosNow() - t_start;
            const s = &c.stats[worker_idx].s;
            s.active_ns += elapsed;
            s.gpa_ns += t_alloc_end - t_alloc_start;
            s.jobs += 1;
            s.bytes += c.inputs[idx].len;
            if (elapsed < s.min_ns) s.min_ns = elapsed;
            if (elapsed > s.max_ns) s.max_ns = elapsed;
        }
    };

    const partials = try gpa.alloc([]ztok.TokenId, chunks);
    defer gpa.free(partials);

    var ctx: InstrCtx = .{
        .pipe = pipe,
        .inputs = inputs,
        .results = partials,
        .ra = gpa,
        .pool = pool,
        .stats = stats,
    };

    // Warm up arenas.
    {
        try pool.runBatch(InstrCtx, &ctx, chunks);
        for (partials) |p| gpa.free(p);
        for (stats) |*p| p.* = .{};
    }

    var total_wall: u64 = 0;
    var k: u32 = 0;
    while (k < iters) : (k += 1) {
        const t0 = nanosNow();
        try pool.runBatch(InstrCtx, &ctx, chunks);
        total_wall += nanosNow() - t0;
        for (partials) |p| gpa.free(p);
    }

    try out.print("\n=== per-worker breakdown ({d} iters, {d} workers, {d} chunks/iter) ===\n", .{ iters, n_workers, chunks });
    try out.print("wall (sum of iters): {d:.2} ms\n", .{@as(f64, @floatFromInt(total_wall)) / 1e6});

    var sum_active: u64 = 0;
    var sum_jobs: u64 = 0;
    var sum_bytes: u64 = 0;
    var min_active: u64 = std.math.maxInt(u64);
    var max_active: u64 = 0;
    for (stats) |*p| {
        sum_active += p.s.active_ns;
        sum_jobs += p.s.jobs;
        sum_bytes += p.s.bytes;
        if (p.s.active_ns < min_active) min_active = p.s.active_ns;
        if (p.s.active_ns > max_active) max_active = p.s.active_ns;
    }
    const ideal_wall = if (n_workers > 0) sum_active / n_workers else 0;
    const efficiency = if (total_wall > 0)
        @as(f64, @floatFromInt(ideal_wall)) / @as(f64, @floatFromInt(total_wall)) * 100.0
    else
        0.0;

    try out.print("sum active ns: {d:.2} ms  (jobs={d}, bytes={d})\n", .{
        @as(f64, @floatFromInt(sum_active)) / 1e6,
        sum_jobs,
        sum_bytes,
    });
    try out.print("ideal wall (sum/N): {d:.2} ms  efficiency: {d:.1}%\n", .{
        @as(f64, @floatFromInt(ideal_wall)) / 1e6,
        efficiency,
    });
    try out.print("worker active spread: min={d:.2} ms  max={d:.2} ms  imbalance={d:.2}x\n", .{
        @as(f64, @floatFromInt(min_active)) / 1e6,
        @as(f64, @floatFromInt(max_active)) / 1e6,
        if (min_active > 0) @as(f64, @floatFromInt(max_active)) / @as(f64, @floatFromInt(min_active)) else 0.0,
    });
    var sum_gpa: u64 = 0;
    for (stats) |*p| sum_gpa += p.s.gpa_ns;
    try out.print("sum encode_ns: {d:.2} ms  (note: includes encode+GPA alloc+realloc; >100% of active means timing overhead is small)\n", .{
        @as(f64, @floatFromInt(sum_gpa)) / 1e6,
    });
    try out.print("\nworker | jobs |   bytes  | active_ms | encode_ms | min_ms | max_ms\n", .{});
    for (stats, 0..) |*p, wi| {
        try out.print("{d:>6} | {d:>4} | {d:>8} | {d:>9.2} | {d:>9.2} | {d:>6.3} | {d:>6.3}\n", .{
            wi,                                           p.s.jobs,                                  p.s.bytes,
            @as(f64, @floatFromInt(p.s.active_ns)) / 1e6, @as(f64, @floatFromInt(p.s.gpa_ns)) / 1e6, @as(f64, @floatFromInt(p.s.min_ns)) / 1e6,
            @as(f64, @floatFromInt(p.s.max_ns)) / 1e6,
        });
    }
}
