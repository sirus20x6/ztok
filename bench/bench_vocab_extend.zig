//! Vocab-extend microbench: naive O(V·L²) full LCS scan vs q-gram
//! pre-filter on synthetic vocabs of V = 10K / 50K / 100K with 100
//! new tokens per round.
//!
//! Self-contained — no external corpus needed. Run via:
//!   zig build bench-vocab-extend -Doptimize=ReleaseSafe
//! Or directly:
//!   zig run -OReleaseSafe bench/bench_vocab_extend.zig \
//!       --dep ztok -Mztok=src/root.zig

const std = @import("std");
const ztok = @import("ztok");
const TokenId = ztok.TokenId;
const Bpe = ztok.Bpe;
const vocab_extend = ztok.vocab_extend;

extern "c" fn clock_gettime(clk_id: c_int, tp: *Timespec) c_int;
const Timespec = extern struct { sec: i64, nsec: i64 };
const CLOCK_MONOTONIC: c_int = 1;

fn nanosNow() u64 {
    var ts: Timespec = .{ .sec = 0, .nsec = 0 };
    _ = clock_gettime(CLOCK_MONOTONIC, &ts);
    return @intCast(ts.sec * 1_000_000_000 + ts.nsec);
}

/// Build a synthetic byte-level BPE with `vocab_size` distinct pieces.
/// To approximate real BPE vocab structure (where many tokens share
/// common subwords like "ing", "tion", "ed", "ly"), each multi-byte
/// piece is built as `prefix + stem + suffix` from small recurring
/// pools. This produces realistic 4-gram collision rates rather than
/// the pathological "every gram is unique" regime of pure random
/// pieces.
fn buildSyntheticBpe(a: std.mem.Allocator, vocab_size: u32, seed: u64) !Bpe {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();

    // Recurring pools so subwords collide between tokens. We chain
    // TWO stems plus a numeric tail to expand the cardinality beyond
    // the 100K vocab size while keeping 4-gram overlap realistic.
    const prefixes = [_][]const u8{
        "un", "re", "in", "dis", "pre", "non", "anti", "auto",
        "co", "de", "en", "ex", "il", "im", "ir", "mis",
        "over", "post", "semi", "sub", "super", "trans", "ultra", "under",
        "for", "fore", "out", "up", "with", "be", "em", "ab",
    };
    const stems = [_][]const u8{
        "act", "ate", "able", "ance", "ant", "ary",
        "ence", "ent", "ess", "fic", "ful", "ial",
        "ify", "ion", "ish", "ism", "ist", "ity", "ive", "ize",
        "less", "ment", "ness", "ous", "ship", "tion", "ward",
        "form", "graph", "logy", "meter", "phon", "scope", "tech", "type",
        "play", "work", "load", "code", "data", "node", "path", "name",
    };
    const suffixes = [_][]const u8{
        "", "s", "ed", "ing", "er", "est", "ly", "able", "ish", "ful",
        "_x", "_y", "_z", "_a", "_b", "_c",
    };

    // Step 1: generate the piece-bytes list in a way that doesn't alias
    // hashmap keys into a re-allocating buffer. Each piece is owned in
    // its own slab; the final flat `bytes` array is concatenated once
    // we know the full size.
    var pieces: std.ArrayList([]u8) = .empty;
    errdefer {
        for (pieces.items) |p| a.free(p);
        pieces.deinit(a);
    }
    var seen = std.StringHashMap(void).init(a);
    defer seen.deinit();

    var count: u32 = 256;
    var total_bytes: usize = 256;
    try pieces.ensureTotalCapacity(a, vocab_size);
    // 256 single-byte pieces (id == byte value).
    var b: u32 = 0;
    while (b < 256) : (b += 1) {
        const buf = try a.alloc(u8, 1);
        buf[0] = @intCast(b);
        try pieces.append(a, buf);
        try seen.put(buf, {});
    }

    // Two-stem chain plus an optional numeric tail gives more than
    // enough unique combinations for V=100K while keeping 4-grams
    // realistic — back-to-back stems share more grams than random
    // bytes.
    var attempts: u64 = 0;
    while (count < vocab_size) {
        attempts += 1;
        if (attempts > @as(u64, vocab_size) * 100) return error.SyntheticVocabExhausted;
        const p = prefixes[r.intRangeAtMost(usize, 0, prefixes.len - 1)];
        const s1 = stems[r.intRangeAtMost(usize, 0, stems.len - 1)];
        const s2 = stems[r.intRangeAtMost(usize, 0, stems.len - 1)];
        const sf = suffixes[r.intRangeAtMost(usize, 0, suffixes.len - 1)];
        // Numeric tail: 0..999 string. Adds 1-3 digits, ~1K distinct
        // values, boosts combinations enormously without harming
        // 4-gram realism (digits are common in real tokenizers too).
        var tail_buf: [8]u8 = undefined;
        const need_tail: bool = count >= 20_000;
        const tail_v: u32 = if (need_tail)
            r.intRangeAtMost(u32, 0, if (count < 60_000) 999 else 999_999)
        else
            0;
        const tail_str: []const u8 = if (need_tail)
            std.fmt.bufPrint(&tail_buf, "_{d}", .{tail_v}) catch unreachable
        else
            "";
        const len = p.len + s1.len + s2.len + sf.len + tail_str.len;
        if (len < 4) continue;
        const buf = try a.alloc(u8, len);
        var off: usize = 0;
        @memcpy(buf[off .. off + p.len], p);
        off += p.len;
        @memcpy(buf[off .. off + s1.len], s1);
        off += s1.len;
        @memcpy(buf[off .. off + s2.len], s2);
        off += s2.len;
        @memcpy(buf[off .. off + sf.len], sf);
        off += sf.len;
        @memcpy(buf[off..], tail_str);
        const gop = try seen.getOrPut(buf);
        if (gop.found_existing) {
            a.free(buf);
            continue;
        }
        gop.key_ptr.* = buf;
        try pieces.append(a, buf);
        total_bytes += len;
        count += 1;
    }

    // Step 2: concatenate.
    const owned_bytes = try a.alloc(u8, total_bytes);
    errdefer a.free(owned_bytes);
    const owned_offsets = try a.alloc(u32, @as(usize, count) + 1);
    errdefer a.free(owned_offsets);
    var cursor: u32 = 0;
    owned_offsets[0] = 0;
    for (pieces.items, 0..) |p, i| {
        @memcpy(owned_bytes[cursor .. cursor + p.len], p);
        cursor += @intCast(p.len);
        owned_offsets[i + 1] = cursor;
    }
    // Free the per-piece slabs now that bytes are concatenated.
    for (pieces.items) |p| a.free(p);
    pieces.deinit(a);

    var by_bytes = std.StringHashMap(TokenId).init(a);
    errdefer by_bytes.deinit();
    try by_bytes.ensureTotalCapacity(count);
    var id: u32 = 0;
    while (id < count) : (id += 1) {
        const key = owned_bytes[owned_offsets[id]..owned_offsets[id + 1]];
        try by_bytes.put(key, id);
    }

    return .{
        .allocator = a,
        .bytes = owned_bytes,
        .offsets = owned_offsets,
        .count = count,
        .by_bytes = by_bytes,
    };
}

/// Build `n` new tokens that do NOT collide with the existing vocab.
/// Uses the same prefix+stem+suffix template as buildSyntheticBpe so
/// the new tokens share realistic 4-grams with the existing vocab.
fn buildNewTokens(a: std.mem.Allocator, old: *const Bpe, n: usize, seed: u64) ![][]u8 {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const prefixes = [_][]const u8{
        "domain_", "medical_", "scientific_", "legal_", "financial_",
        "novel_", "synthetic_", "test_", "rare_", "private_",
    };
    const stems = [_][]const u8{
        "encoder", "transformer", "embedding", "tokenizer", "decoder",
        "context", "attention", "weights", "features", "patterns",
        "actuator", "classifier", "function", "instance", "method",
    };
    const suffixes = [_][]const u8{ "", "_v2", "_v3", "_alt", "_x", "_y", "_pro" };

    var out: std.ArrayList([]u8) = .empty;
    errdefer {
        for (out.items) |s| a.free(s);
        out.deinit(a);
    }

    var made: usize = 0;
    while (made < n) {
        const p = prefixes[r.intRangeAtMost(usize, 0, prefixes.len - 1)];
        const s = stems[r.intRangeAtMost(usize, 0, stems.len - 1)];
        const sf = suffixes[r.intRangeAtMost(usize, 0, suffixes.len - 1)];
        const len = p.len + s.len + sf.len;
        const buf = try a.alloc(u8, len);
        @memcpy(buf[0..p.len], p);
        @memcpy(buf[p.len .. p.len + s.len], s);
        @memcpy(buf[p.len + s.len ..], sf);
        if (old.by_bytes.get(buf)) |_| {
            a.free(buf);
            continue;
        }
        try out.append(a, buf);
        made += 1;
    }
    return out.toOwnedSlice(a);
}

fn runOne(
    a: std.mem.Allocator,
    old: *const Bpe,
    new_toks: []const []const u8,
    use_index: bool,
) !u64 {
    // Cast to []const []const u8.
    const t0 = nanosNow();
    var res = try vocab_extend.extendBpe(a, old, .{
        .new_tokens = new_toks,
        .strategy = .weighted_similar,
        .weighted_similar = .{
            .k = 8,
            .use_qgram_index = use_index,
        },
    });
    const t1 = nanosNow();
    res.deinit();
    return t1 - t0;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var stdout_buf: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buf);
    const out = &stdout_writer.interface;

    try out.print("vocab_extend microbench — weighted_similar, K=8, 100 new tokens\n", .{});
    try out.print("{s:>10} | {s:>12} | {s:>12} | {s:>10} | {s:>12}\n", .{ "V", "naive (ms)", "qgram (ms)", "speedup", "idx (KB)" });
    try out.print("-----------+--------------+--------------+------------+-------------\n", .{});
    try out.flush();

    const sizes = [_]u32{ 10_000, 50_000, 100_000 };
    const n_new: usize = 100;

    for (sizes) |v| {
        var bpe = try buildSyntheticBpe(gpa, v, 0xDEAD_BEEFC0DE);
        defer bpe.deinit();

        const new_owned = try buildNewTokens(gpa, &bpe, n_new, 0xCAFE_BABE);
        defer {
            for (new_owned) |s| gpa.free(s);
            gpa.free(new_owned);
        }
        // Cast to []const []const u8 for the extendBpe signature.
        const new_const: [][]const u8 = try gpa.alloc([]const u8, n_new);
        defer gpa.free(new_const);
        for (new_owned, 0..) |s, i| new_const[i] = s;

        // Run each path twice and take the min — wall-clock noise on
        // first run can swamp small differences.
        const ns_naive_a = try runOne(gpa, &bpe, new_const, false);
        const ns_naive_b = try runOne(gpa, &bpe, new_const, false);
        const ns_qgram_a = try runOne(gpa, &bpe, new_const, true);
        const ns_qgram_b = try runOne(gpa, &bpe, new_const, true);
        const ns_naive = @min(ns_naive_a, ns_naive_b);
        const ns_qgram = @min(ns_qgram_a, ns_qgram_b);

        const idx_bytes = try vocab_extend.estimateQGramIndexBytes(gpa, &bpe);

        const ms_naive = @as(f64, @floatFromInt(ns_naive)) / 1_000_000.0;
        const ms_qgram = @as(f64, @floatFromInt(ns_qgram)) / 1_000_000.0;
        const speedup = if (ns_qgram > 0)
            @as(f64, @floatFromInt(ns_naive)) / @as(f64, @floatFromInt(ns_qgram))
        else
            0.0;
        const idx_kb = @as(f64, @floatFromInt(idx_bytes)) / 1024.0;

        try out.print("{d:>10} | {d:>12.2} | {d:>12.2} | {d:>9.2}x | {d:>12.0}\n", .{ v, ms_naive, ms_qgram, speedup, idx_kb });
        try out.flush();
    }
}
