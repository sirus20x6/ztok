//! Capcode pipeline profile: measures normalizer-only and
//! encoder-against-pre-normalized-input timings on a 10 MB corpus.
//! Tells us how much of the full-pipeline time is normalizer vs encoder.

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
    while (arg_iter.next()) |a| try owned.append(gpa, try gpa.dupe(u8, a));

    var corpus_path: []const u8 = "";
    var model_path: []const u8 = "";
    var iters: u32 = 10;
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
        }
    }

    const corpus = try std.Io.Dir.cwd().readFileAlloc(io, corpus_path, gpa, .unlimited);
    defer gpa.free(corpus);
    const ztm = try std.Io.Dir.cwd().readFileAlloc(io, model_path, gpa, .unlimited);
    defer gpa.free(ztm);

    var monster = try ztok.monster_io.readBytes(gpa, ztm);
    defer monster.deinit();

    // Pre-normalize once.
    const normed = try ztok.tm_norm.normalizeCapcode(gpa, corpus, .tm_printable);
    defer gpa.free(normed);

    std.debug.print("corpus: {d} bytes, normalized: {d} bytes ({d:.3}x)\n", .{
        corpus.len, normed.len, @as(f64, @floatFromInt(normed.len)) / @as(f64, @floatFromInt(corpus.len)),
    });

    // --- Normalizer only ---
    {
        const w = try ztok.tm_norm.normalizeCapcode(gpa, corpus, .tm_printable);
        gpa.free(w);
        var ns: u64 = 0;
        var it: u32 = 0;
        while (it < iters) : (it += 1) {
            const t0 = nanosNow();
            const out = try ztok.tm_norm.normalizeCapcode(gpa, corpus, .tm_printable);
            ns += nanosNow() - t0;
            gpa.free(out);
        }
        const ms: f64 = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(iters)) / 1_000_000.0;
        const mbps: f64 = (@as(f64, @floatFromInt(corpus.len)) / 1_048_576.0) / (ms / 1000.0);
        std.debug.print("normalizer:      {d:.2} ms/iter  {d:.2} MB/s  (per orig MB)\n", .{ ms, mbps });
    }

    // --- Encoder only against pre-normalized input ---
    {
        const buf = try gpa.alloc(ztok.TokenId, normed.len * 2 + 16);
        defer gpa.free(buf);
        const w = try monster.encodeChunk(gpa, normed, buf);
        std.debug.print("encoder ids on normalized: {d} ({d:.2} bytes/tok of normalized)\n", .{ w.len, @as(f64, @floatFromInt(normed.len)) / @as(f64, @floatFromInt(w.len)) });

        var ns: u64 = 0;
        var it: u32 = 0;
        while (it < iters) : (it += 1) {
            const t0 = nanosNow();
            _ = try monster.encodeChunk(gpa, normed, buf);
            ns += nanosNow() - t0;
        }
        const ms: f64 = @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(iters)) / 1_000_000.0;
        const mbps_norm: f64 = (@as(f64, @floatFromInt(normed.len)) / 1_048_576.0) / (ms / 1000.0);
        const mbps_orig: f64 = (@as(f64, @floatFromInt(corpus.len)) / 1_048_576.0) / (ms / 1000.0);
        std.debug.print("encoder (normed):{d:.2} ms/iter  {d:.2} MB/s of normalized bytes  ({d:.2} MB/s per orig MB)\n", .{ ms, mbps_norm, mbps_orig });
    }
}
