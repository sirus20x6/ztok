//! ztok encode/decode fuzz harness.
//!
//! Two complementary harnesses share one binary:
//!
//! 1. `byte_id round-trip` — identity normalizer + identity pre-tok +
//!    byte_id model + concat decoder. By construction every byte maps
//!    to exactly one id and decoding concatenates them back, so for
//!    arbitrary input bytes we MUST have `decode(encode(input)) ==
//!    input`. Any divergence is a bug in the model encode bookkeeping
//!    or the decoder's byte-reassembly path.
//!
//! 2. `cl100k BPE round-trip` — loads a tiny synthetic tiktoken vocab
//!    covering all 256 bytes + a handful of merges (matching the
//!    fixture used in src/c_api.zig and pipeline.zig tests). The
//!    cl100k pre-tokenizer applies the GPT-4 regex; combined with
//!    concat decode it MUST also round-trip exactly, because the
//!    vocab covers every byte and the decoder is byte-clean. This
//!    exercises:
//!      * UTF-8 codepoint segmentation in cl100k.zig
//!      * BPE merge selection in bpe.zig
//!      * the SoA scratch reuse in pipeline.zig
//!      * the cl100k pre-tok's "carry the trailing partial codepoint
//!        forward" rule
//!
//! Both modes are wrapped in `std.testing.fuzz` so libFuzzer (when
//! ZTOK builds with `--fuzz`) can mutate the input bytes intelligently.
//! In the default `zig build fuzz` invocation we run for a wall-clock
//! budget (env `ZTOK_FUZZ_BUDGET_SECS`, default 60) using a manual
//! PRNG-driven loop — this works on any Zig 0.16 build without the
//! libFuzzer runtime.
//!
//! Equivalence relation note: round-trip equality holds for byte_id
//! (always) and for the synthetic cl100k vocab built here (because it
//! is complete on all 256 bytes). With a NORMALIZER (NFC/NFKC) or a
//! lossy decoder (byte_level over a vocab missing some bytes) the
//! round-trip is only equivalent modulo the normalizer's mapping —
//! NOT a fuzz target until we add expected-output bookkeeping.
//!
//! This file does NOT touch bench/vocabs/*. The only fixture data is
//! the synthetic vocab built in-memory below; corpus seeds live under
//! fuzz/corpus/ and are read-only.

const std = @import("std");
const ztok = @import("ztok");

const Bpe = ztok.Bpe;
const Pipeline = ztok.Pipeline;
const Vocab = ztok.vocab.Vocab;
const TokenId = ztok.TokenId;

// Zig 0.16 retired `std.time.milliTimestamp` / `nanoTimestamp` /
// `std.time.Timer`. Mirror the pattern the bench harnesses use
// (bench/bench_simd_min.zig etc.) and shell out to libc directly. We
// link libc anyway via the build module config — no extra runtime cost.
const Timespec = extern struct { sec: i64, nsec: i64 };
const CLOCK_MONOTONIC: c_int = 1;
extern "c" fn clock_gettime(clk_id: c_int, tp: *Timespec) c_int;

fn nanosNow() i64 {
    var ts: Timespec = .{ .sec = 0, .nsec = 0 };
    _ = clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.sec * std.time.ns_per_s + ts.nsec;
}

fn millisNow() i64 {
    return @divTrunc(nanosNow(), std.time.ns_per_ms);
}

/// Build a synthetic cl100k-style tiktoken vocab covering all 256
/// bytes + a small set of merges. Same fixture shape as the unit
/// tests in src/c_api.zig and src/pipeline.zig — we duplicate it here
/// so the fuzz harness can be run without dragging in test-only
/// helpers.
fn buildSyntheticBpe(a: std.mem.Allocator) !Bpe {
    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(a);
    const b64 = std.base64.standard.Encoder;
    var enc_buf: [32]u8 = undefined;
    var rank: u32 = 0;
    var b: u32 = 0;
    while (b < 256) : (b += 1) {
        const byte: [1]u8 = .{@intCast(b)};
        const encoded = b64.encode(&enc_buf, &byte);
        try src.print(a, "{s} {d}\n", .{ encoded, rank });
        rank += 1;
    }
    const extra = [_][]const u8{
        "he",     "hel",   "hell",  "hello",
        " w",     " wo",   " wor",  " worl",
        " world", "the",   " the",  " quick",
        " brown", " fox",  "foo",   "bar",
        "baz",    "zig",   " zig",  "test",
        " test",  "fuzz",  " fuzz",
    };
    for (extra) |bytes| {
        const encoded = b64.encode(&enc_buf, bytes);
        try src.print(a, "{s} {d}\n", .{ encoded, rank });
        rank += 1;
    }
    return Bpe.loadTiktokenBytes(a, src.items);
}

/// byte_id round-trip on arbitrary bytes.
fn checkByteIdRoundTrip(a: std.mem.Allocator, input: []const u8) !void {
    var v = Vocab.empty(a);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };
    const ids = try pipe.encode(a, input);
    defer a.free(ids);
    // byte_id maps 1:1.
    if (ids.len != input.len) return error.LengthMismatch;
    const round = try pipe.decode(a, ids);
    defer a.free(round);
    if (!std.mem.eql(u8, round, input)) return error.RoundTripMismatch;
}

/// cl100k BPE round-trip on arbitrary bytes — must equal input
/// because the synthetic vocab covers every byte.
fn checkBpeRoundTrip(a: std.mem.Allocator, bpe: *Bpe, vocab: *Vocab, input: []const u8) !void {
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .cl100k,
        .model = .{ .bpe = bpe },
        .decoder = .concat,
        .vocab = vocab,
    };
    const ids = try pipe.encode(a, input);
    defer a.free(ids);
    const round = try pipe.decode(a, ids);
    defer a.free(round);
    if (!std.mem.eql(u8, round, input)) return error.RoundTripMismatch;
}

/// Per-iteration target for std.testing.fuzz. The `context` carries
/// both the BPE pipeline (cached across iterations) and the GPA the
/// per-call allocations go through.
const FuzzContext = struct {
    a: std.mem.Allocator,
    bpe: *Bpe,
    vocab: *Vocab,
};

fn fuzzOne(ctx: FuzzContext, smith: *std.testing.Smith) !void {
    @disableInstrumentation();
    // Cap each input at 8 KiB. The encoder is sub-linear past a few
    // KiB; longer inputs just slow the fuzzer without finding bugs we
    // wouldn't also find in the 8 KiB range.
    var buf: [8192]u8 = undefined;
    const len = smith.sliceWeightedBytes(buf[0..], &.{
        // Cover the full byte range — adversarial inputs (control
        // chars, NUL, malformed UTF-8) are the whole point.
        .rangeAtMost(u8, 0x00, 0xff, 1),
        // Bias toward printable ASCII so the BPE merge logic sees
        // realistic-shaped tokens too.
        .rangeAtMost(u8, 0x20, 0x7e, 4),
        // Whitespace bias — cl100k's regex treats whitespace runs
        // specially, so hammer those boundaries.
        .value(u8, ' ', 6),
        .value(u8, '\n', 4),
        .value(u8, '\t', 2),
        // Multibyte UTF-8 lead bytes — drives the cl100k carry path.
        .rangeAtMost(u8, 0xc2, 0xdf, 2), // 2-byte leads
        .rangeAtMost(u8, 0xe0, 0xef, 2), // 3-byte leads
        .rangeAtMost(u8, 0xf0, 0xf4, 1), // 4-byte leads
    });
    const input = buf[0..len];

    try checkByteIdRoundTrip(ctx.a, input);
    try checkBpeRoundTrip(ctx.a, ctx.bpe, ctx.vocab, input);
}

// ---------------------------------------------------------------------
// Free-standing harness: when libFuzzer isn't linked in we still want
// `zig build fuzz` to do *something* useful. Drive `fuzzOne` from a
// PRNG-mutated corpus of byte buffers for ZTOK_FUZZ_BUDGET_SECS seconds.
//
// Seeds are read from fuzz/corpus/ (if present) on startup. Each
// iteration picks a seed (or a fresh random buffer when the corpus is
// empty), applies a random mutation, and runs the byte_id + BPE
// round-trip checks. Any panic / unreachable / assertion failure
// crashes the process and is reported by the build system.
// ---------------------------------------------------------------------

const default_budget_secs: u64 = 60;

const Mutator = struct {
    prng: std.Random.DefaultPrng,

    fn init(seed: u64) Mutator {
        return .{ .prng = std.Random.DefaultPrng.init(seed) };
    }

    fn rng(self: *Mutator) std.Random {
        return self.prng.random();
    }

    /// Produce a fresh buffer derived from `seed` (which may be empty).
    /// Mutations: bit-flip, byte-randomize, truncate, extend with
    /// random tail. The output length is bounded by `cap`.
    fn mutate(self: *Mutator, a: std.mem.Allocator, seed: []const u8, cap: usize) ![]u8 {
        const r = self.rng();
        const target_len = blk: {
            // 25% empty, 25% length match, 50% random length up to cap.
            const roll = r.intRangeAtMost(u8, 0, 99);
            if (roll < 25) break :blk 0;
            if (roll < 50) break :blk @min(seed.len, cap);
            break :blk r.intRangeAtMost(usize, 0, cap);
        };
        var out = try a.alloc(u8, target_len);
        errdefer a.free(out);

        // Copy as much of the seed as fits, then apply mutations.
        const carry = @min(seed.len, target_len);
        @memcpy(out[0..carry], seed[0..carry]);
        // Fill the tail with random bytes.
        if (target_len > carry) {
            r.bytes(out[carry..]);
        }
        // Apply a handful of bit-flips so the seed shape is perturbed.
        if (target_len > 0) {
            var flips: u32 = r.intRangeAtMost(u32, 0, 8);
            while (flips > 0) : (flips -= 1) {
                const idx = r.intRangeLessThan(usize, 0, target_len);
                const bit: u3 = @intCast(r.intRangeAtMost(u8, 0, 7));
                out[idx] ^= @as(u8, 1) << bit;
            }
        }
        return out;
    }
};

fn readEnvU64(env: *const std.process.Environ.Map, name: []const u8, default: u64) u64 {
    const v = env.get(name) orelse return default;
    return std.fmt.parseInt(u64, v, 0) catch default;
}

fn loadCorpusSeeds(a: std.mem.Allocator, io: std.Io) !std.ArrayList([]u8) {
    var seeds: std.ArrayList([]u8) = .empty;
    errdefer {
        for (seeds.items) |s| a.free(s);
        seeds.deinit(a);
    }
    var dir = std.Io.Dir.cwd().openDir(io, "fuzz/corpus", .{ .iterate = true }) catch {
        // Corpus directory missing is fine — fall back to all-random.
        return seeds;
    };
    defer dir.close(io);
    var it = dir.iterate();
    while (it.next(io) catch null) |entry| {
        if (entry.kind != .file) continue;
        if (std.mem.startsWith(u8, entry.name, ".")) continue;
        // Cap individual seed at 64 KiB so a stray big file doesn't
        // blow the corpus loader; the fuzzer caps inputs at 8 KiB
        // downstream anyway.
        const bytes = dir.readFileAlloc(io, entry.name, a, .limited(64 * 1024)) catch continue;
        try seeds.append(a, bytes);
    }
    return seeds;
}

pub fn main(init: std.process.Init) !void {
    // Use the harness GPA from the Zig 0.16 `std.process.Init`. The
    // harness leaks small per-iteration allocations on purpose (we want
    // ReleaseSafe + GPA's leak detection to fire if encode/decode
    // leaks); we therefore drain the arena ourselves at loop exit
    // rather than relying on a defer here.
    const a = init.gpa;
    const io = init.io;
    const env = init.environ_map;

    const budget_secs = readEnvU64(env, "ZTOK_FUZZ_BUDGET_SECS", default_budget_secs);
    // Seed precedence: FUZZ_SEED (nightly rotation) > ZTOK_FUZZ_SEED
    // (legacy/local) > wall-clock fallback. Both are parsed as u64 via
    // std.fmt.parseInt with base 0 so hex like "0xdeadbeef" works.
    const seed_arg = if (env.get("FUZZ_SEED")) |_|
        readEnvU64(env, "FUZZ_SEED", @bitCast(nanosNow()))
    else
        readEnvU64(env, "ZTOK_FUZZ_SEED", @bitCast(nanosNow()));

    var bpe = try buildSyntheticBpe(a);
    defer bpe.deinit();
    var v = Vocab.empty(a);
    defer v.deinit();

    var seeds = try loadCorpusSeeds(a, io);
    defer {
        for (seeds.items) |s| a.free(s);
        seeds.deinit(a);
    }

    var mutator: Mutator = .init(seed_arg);

    var stderr_file = std.Io.File.stderr();
    var stderr_buf: [4096]u8 = undefined;
    var stderr_w = stderr_file.writer(io, &stderr_buf);
    const stderr = &stderr_w.interface;

    try stderr.print(
        "ztok fuzz: budget={d}s seed={d} corpus_size={d}\n",
        .{ budget_secs, seed_arg, seeds.items.len },
    );
    try stderr.flush();

    const start = millisNow();
    const deadline = start + @as(i64, @intCast(budget_secs)) * 1000;
    var iterations: u64 = 0;
    var failures: u64 = 0;

    while (millisNow() < deadline) {
        const seed_slice: []const u8 = blk: {
            if (seeds.items.len == 0) break :blk &.{};
            const idx = mutator.rng().intRangeLessThan(usize, 0, seeds.items.len);
            break :blk seeds.items[idx];
        };
        const input = try mutator.mutate(a, seed_slice, 8192);
        defer a.free(input);

        checkByteIdRoundTrip(a, input) catch |e| {
            failures += 1;
            try stderr.print(
                "ztok fuzz: byte_id mismatch iter={d} input_len={d} err={s}\n",
                .{ iterations, input.len, @errorName(e) },
            );
            try stderr.flush();
            // Surface the offending input so re-running with the same
            // seed reproduces; then propagate the failure.
            return e;
        };
        checkBpeRoundTrip(a, &bpe, &v, input) catch |e| {
            failures += 1;
            try stderr.print(
                "ztok fuzz: bpe mismatch iter={d} input_len={d} err={s}\n",
                .{ iterations, input.len, @errorName(e) },
            );
            try stderr.flush();
            return e;
        };
        iterations += 1;
        if (iterations % 1000 == 0) {
            try stderr.print(
                "ztok fuzz: iter={d} elapsed={d}ms failures={d}\n",
                .{ iterations, millisNow() - start, failures },
            );
            try stderr.flush();
        }
    }

    try stderr.print(
        "ztok fuzz: done iters={d} failures={d} elapsed_ms={d}\n",
        .{ iterations, failures, millisNow() - start },
    );
    try stderr.flush();
}

// ---------------------------------------------------------------------
// std.testing.fuzz entry: when running under `zig build test --fuzz`,
// the libFuzzer runtime drives `fuzzOne` directly with mutated inputs.
// Outside `--fuzz`, this just runs the harness once on an empty input
// as a smoke-test of the fuzzOne body — the corpus-driven loop above
// is what `zig build fuzz` invokes.
// ---------------------------------------------------------------------

test "fuzz: encode/decode round-trip" {
    const a = std.testing.allocator;
    var bpe = try buildSyntheticBpe(a);
    defer bpe.deinit();
    var v = Vocab.empty(a);
    defer v.deinit();
    const ctx: FuzzContext = .{ .a = a, .bpe = &bpe, .vocab = &v };
    try std.testing.fuzz(ctx, fuzzOne, .{});
}
