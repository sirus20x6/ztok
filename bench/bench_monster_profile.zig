//! Monster-encoder profiling harness (post-1.18 agent E).
//!
//! Runs `Monster.encodeChunk` on a 10 MB corpus N times and reports the
//! per-iteration time and throughput. When `monster.profile_enabled`
//! is flipped on at build time, also dumps per-phase nanosecond
//! breakdown from the in-`encodeChunkImpl` counters.
//!
//! Usage:
//!   bench_monster_profile --model VOCAB.ztm --corpus PATH [--iters N]

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
    model: []const u8,
    corpus: []const u8,
    iters: u32 = 5,
};

fn parseArgs(gpa: std.mem.Allocator, owned: *std.ArrayList([]const u8)) !?Args {
    _ = gpa;
    var model: ?[]const u8 = null;
    var corpus: ?[]const u8 = null;
    var iters: u32 = 5;
    var i: usize = 1;
    while (i < owned.items.len) : (i += 1) {
        const a = owned.items[i];
        if (std.mem.eql(u8, a, "--model")) {
            model = owned.items[i + 1];
            i += 1;
        } else if (std.mem.eql(u8, a, "--corpus")) {
            corpus = owned.items[i + 1];
            i += 1;
        } else if (std.mem.eql(u8, a, "--iters")) {
            iters = try std.fmt.parseInt(u32, owned.items[i + 1], 10);
            i += 1;
        }
    }
    if (model == null or corpus == null) return null;
    return .{
        .model = model.?,
        .corpus = corpus.?,
        .iters = iters,
    };
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

    const args = (try parseArgs(gpa, &owned)) orelse {
        std.debug.print(
            "usage: bench_monster_profile --model VOCAB.ztm --corpus PATH [--iters N]\n",
            .{},
        );
        std.process.exit(2);
    };

    const corpus = try std.Io.Dir.cwd().readFileAlloc(io, args.corpus, gpa, .unlimited);
    defer gpa.free(corpus);

    const ztm_bytes = try std.Io.Dir.cwd().readFileAlloc(io, args.model, gpa, .unlimited);
    defer gpa.free(ztm_bytes);

    var monster = try ztok.monster_io.readBytes(gpa, ztm_bytes);
    defer monster.deinit();

    const out_cap = corpus.len * 2;
    const out_buf = try gpa.alloc(ztok.TokenId, out_cap);
    defer gpa.free(out_buf);

    std.debug.print(
        "bench_monster_profile\n  model:   {s}\n  corpus:  {d} bytes\n  iters:   {d}\n  vocab:   {d} tokens, max_token_len={d}\n  lilbuf:  enabled={}  s2b3b={}  goto_chk={}  precomp_alts={}\n\n",
        .{
            args.model,
            corpus.len,
            args.iters,
            monster.count,
            monster.max_token_len,
            monster.lilbuf_enabled,
            monster.score2b3b_enabled,
            monster.goto_checkpoint_enabled,
            monster.use_precomputed_alts,
        },
    );

    // Warm-up.
    _ = try monster.encodeChunk(gpa, corpus, out_buf);

    // Reset counters after warm-up.
    if (ztok.monster.profile_enabled) {
        ztok.monster.profile_counters = .{};
    }

    var total_ns: u64 = 0;
    var ids_per_iter: usize = 0;
    var iter: u32 = 0;
    while (iter < args.iters) : (iter += 1) {
        const t0 = nanosNow();
        const ids = try monster.encodeChunk(gpa, corpus, out_buf);
        total_ns += nanosNow() - t0;
        ids_per_iter = ids.len;
    }

    const avg_ms: f64 = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(args.iters)) / 1_000_000.0;
    const bytes_total = corpus.len;
    const mb_per_sec: f64 = (@as(f64, @floatFromInt(bytes_total)) / 1_048_576.0) /
        (avg_ms / 1000.0);

    std.debug.print(
        "  avg ms/iter:   {d:.2}\n  MB/s:          {d:.2}\n  ids/iter:      {d}\n  bytes/tok:     {d:.2}\n",
        .{ avg_ms, mb_per_sec, ids_per_iter, @as(f64, @floatFromInt(bytes_total)) / @as(f64, @floatFromInt(ids_per_iter)) },
    );

    if (ztok.monster.profile_enabled) {
        const c = ztok.monster.profile_counters;
        std.debug.print(
            "\nPer-phase ns (totals across {d} iters):\n  collectPrefixMatches: {d} ns ({d:.1}%)\n  lilbuf_walks:         {d} ns ({d:.1}%)\n  branch_score:         {d} ns ({d:.1}%)\n  scoreb_extra:         {d} ns ({d:.1}%)\n  emit_advance:         {d} ns ({d:.1}%)\n",
            .{
                args.iters,
                c.collect_ns,
                100.0 * @as(f64, @floatFromInt(c.collect_ns)) / @as(f64, @floatFromInt(total_ns)),
                c.lilbuf_ns,
                100.0 * @as(f64, @floatFromInt(c.lilbuf_ns)) / @as(f64, @floatFromInt(total_ns)),
                c.branch_ns,
                100.0 * @as(f64, @floatFromInt(c.branch_ns)) / @as(f64, @floatFromInt(total_ns)),
                c.scoreb_ns,
                100.0 * @as(f64, @floatFromInt(c.scoreb_ns)) / @as(f64, @floatFromInt(total_ns)),
                c.emit_ns,
                100.0 * @as(f64, @floatFromInt(c.emit_ns)) / @as(f64, @floatFromInt(total_ns)),
            },
        );
    }
}
