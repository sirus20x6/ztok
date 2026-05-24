//! Library-level helpers backing the `ztok bench` CLI subcommand. Runs
//! the canonical perf suite against the vocab fixtures vendored under
//! `bench/vocabs/` and emits a tabular text report (mirroring
//! `bench/RESULTS.md`'s style) or a stable JSON document for CI
//! consumers.
//!
//! Kept out of `src/main.zig` so the scenario-discovery, timing, and
//! formatting logic can be unit-tested against the root module without
//! spawning a subprocess.
//!
//! Scenarios:
//!   * `cl100k`        — cl100k_base.tiktoken (BPE)
//!   * `sp-bpe`        — LLaMA-2 SP-BPE (`llama2.model`)
//!   * `sp-unigram`    — T5 Unigram (`t5_unigram.model`)
//!   * `tm`            — TokenMonster nocapcode (`tm_englishcode_32k.ztm`)
//!   * `hf-bpe`        — HF GPT-2 BPE (`gpt2_hf.json`)
//!
//! Post-1.17 extended scenarios (agent E). All optional — each one
//! reports `skipped` cleanly when its fixture is absent so the default
//! `ztok bench` run still works without `bench/fetch_vocabs.py --extended`.
//!   * `mistral7b`     — SP-BPE (`mistral7b.model`, 32K vocab)
//!   * `yi6b`          — SP-BPE (`yi6b.model`, 64K vocab)
//!   * `phi3`          — HF-BPE with byte_fallback (`phi3.json`)
//!   * `falcon7b`      — HF-BPE ByteLevel (`falcon7b.json`, 65K vocab)
//!   * `deepseek-v2`   — HF-BPE ByteLevel (`deepseek_v2.json`, 100K vocab)
//!   * `qwen2`         — HF-BPE ByteLevel (`qwen2.json`, 151K vocab)
//!   * `llama3`        — HF-BPE ByteLevel (`llama3.json`, 128K vocab)
//!
//! Each scenario reports throughput (single-thread MB/s, batch ×8 MB/s,
//! batch ×48 with `--pin-physical` MB/s) and the encoded id count for
//! spot-check correctness. A scenario whose fixture is missing is
//! reported as `skipped` rather than aborting the run.
//!
//! JSON schema (stable, version-tagged at top level):
//!
//!   {
//!     "version": 1,
//!     "tool": "ztok",
//!     "iters": 5,
//!     "corpus_bytes": 1048576,
//!     "scenarios": [
//!       {
//!         "name": "cl100k",
//!         "status": "ok",          // "ok" | "skipped" | "error"
//!         "vocab_path": "bench/vocabs/...",
//!         "vocab_size": 100277,
//!         "ids": 327680,
//!         "results": [
//!           {"shape": "single",      "mb_per_sec": 26.1, "ms": 40.1, "ids": 327680},
//!           {"shape": "batch_8",     "mb_per_sec": 92.0, "ms": 11.4, "ids": 327680},
//!           {"shape": "batch_48_pin","mb_per_sec": 350.0,"ms": 2.99, "ids": 327680}
//!         ],
//!         "error": null
//!       }
//!     ]
//!   }
//!
//! Field names are stable across releases — downstream CI tools that
//! track throughput regressions can rely on them.

const std = @import("std");
const TokenId = @import("token.zig").TokenId;
const Bpe = @import("bpe.zig").Bpe;
const Unigram = @import("unigram.zig").Unigram;
const WordPiece = @import("wordpiece.zig").WordPiece;
const Monster = @import("monster.zig").Monster;
const RwkvWorld = @import("rwkv_world.zig").RwkvWorld;
const Pipeline = @import("pipeline.zig").Pipeline;
const Vocab = @import("vocab.zig").Vocab;
const Model = @import("model.zig").Model;
const Normalizer = @import("normalizer.zig").Normalizer;
const PreTokenizer = @import("pretok.zig").PreTokenizer;
const Decoder = @import("decoder.zig").Decoder;
const thread_pool = @import("thread_pool.zig");
const auto_detect = @import("auto_detect.zig");
const hf_json = @import("hf_json.zig");
const hf_bridge = @import("hf_bridge.zig");
const sp_model = @import("sp_model.zig");
const sp_bridge = @import("sp_bridge.zig");
const monster_io = @import("monster_io.zig");

pub const Format = enum { text, json };

/// Stable scenario tag. The on-disk fixture name + pre-tokenizer +
/// normalizer recipe are inferred per-tag inside `runOne`.
pub const Scenario = enum {
    cl100k,
    sp_bpe,
    sp_unigram,
    tm,
    hf_bpe,
    // Extended cross-bench fixtures (post-1.17 agent E). Each one is
    // optional and skips cleanly when its vocab file isn't vendored.
    mistral7b,
    yi6b,
    phi3,
    falcon7b,
    deepseek_v2,
    qwen2,
    llama3,

    pub fn cliName(self: Scenario) []const u8 {
        return switch (self) {
            .cl100k => "cl100k",
            .sp_bpe => "sp-bpe",
            .sp_unigram => "sp-unigram",
            .tm => "tm",
            .hf_bpe => "hf-bpe",
            .mistral7b => "mistral7b",
            .yi6b => "yi6b",
            .phi3 => "phi3",
            .falcon7b => "falcon7b",
            .deepseek_v2 => "deepseek-v2",
            .qwen2 => "qwen2",
            .llama3 => "llama3",
        };
    }

    pub fn parse(s: []const u8) ?Scenario {
        if (std.mem.eql(u8, s, "cl100k")) return .cl100k;
        if (std.mem.eql(u8, s, "sp-bpe")) return .sp_bpe;
        if (std.mem.eql(u8, s, "sp-unigram")) return .sp_unigram;
        if (std.mem.eql(u8, s, "tm")) return .tm;
        if (std.mem.eql(u8, s, "hf-bpe")) return .hf_bpe;
        if (std.mem.eql(u8, s, "mistral7b")) return .mistral7b;
        if (std.mem.eql(u8, s, "yi6b")) return .yi6b;
        if (std.mem.eql(u8, s, "phi3")) return .phi3;
        if (std.mem.eql(u8, s, "falcon7b")) return .falcon7b;
        if (std.mem.eql(u8, s, "deepseek-v2")) return .deepseek_v2;
        if (std.mem.eql(u8, s, "qwen2")) return .qwen2;
        if (std.mem.eql(u8, s, "llama3")) return .llama3;
        return null;
    }
};

pub const all_scenarios = [_]Scenario{
    .cl100k,
    .sp_bpe,
    .sp_unigram,
    .tm,
    .hf_bpe,
    .mistral7b,
    .yi6b,
    .phi3,
    .falcon7b,
    .deepseek_v2,
    .qwen2,
    .llama3,
};

pub const BenchOptions = struct {
    /// Iterations per (scenario × shape). Defaults to 5; `--quick` =
    /// 1.
    iters: u32 = 5,
    /// Optional filter — when non-empty, only these scenarios run.
    /// Pass null to run every scenario whose vocab fixture exists.
    include: ?[]const Scenario = null,
    /// Root directory holding the vocab fixtures. Defaults to
    /// `bench/vocabs` (relative to cwd) — the same path
    /// `bench/fetch_vocabs.py` writes to.
    vocab_root: []const u8 = "bench/vocabs",
    /// Output format.
    format: Format = .text,
    /// Bytes of synthetic corpus generated per scenario. The corpus
    /// is fixed text (lorem-style) so different scenario runs are
    /// comparable on the same input size. 1 MiB is large enough to
    /// hit the encode hot path and small enough to keep `--quick`
    /// runs sub-second.
    corpus_bytes: usize = 1024 * 1024,
    /// Read the benchmark corpus from this file instead of synthesizing
    /// one. Lets ztok and an external tokenizer (e.g. tiktoken via
    /// `bench/bench_competitors.py --corpus PATH`) be timed on identical
    /// bytes. When set, `corpus_bytes` is ignored.
    corpus_file: ?[]const u8 = null,
    /// Skip batch shapes (single-thread only) — useful for
    /// non-batch hardware or for tracking single-thread regressions
    /// independently.
    single_only: bool = false,
};

pub const ShapeKind = enum { single, batch_8, batch_48_pin };

pub const ShapeResult = struct {
    shape: ShapeKind,
    mb_per_sec: f64,
    ms_per_iter: f64,
    ids: usize,
};

pub const ScenarioStatus = enum { ok, skipped, error_ };

pub const ScenarioResult = struct {
    name: Scenario,
    status: ScenarioStatus,
    vocab_path: ?[]u8 = null,
    vocab_size: u32 = 0,
    ids: usize = 0,
    results: []ShapeResult = &.{},
    error_message: ?[]u8 = null,

    pub fn deinit(self: *ScenarioResult, allocator: std.mem.Allocator) void {
        if (self.vocab_path) |p| allocator.free(p);
        if (self.error_message) |m| allocator.free(m);
        if (self.results.len > 0) allocator.free(self.results);
    }
};

pub const BenchResult = struct {
    iters: u32,
    corpus_bytes: usize,
    scenarios: []ScenarioResult,

    pub fn deinit(self: *BenchResult, allocator: std.mem.Allocator) void {
        for (self.scenarios) |*s| s.deinit(allocator);
        allocator.free(self.scenarios);
    }
};

// ----------------------------------------------------------------------
// Public entry points.
// ----------------------------------------------------------------------

/// Top-level driver. Runs every requested scenario, prints a report,
/// and returns the structured result so callers (CLI, tests, CI) can
/// inspect throughput numbers without re-parsing the printout.
pub fn runBench(
    allocator: std.mem.Allocator,
    opts: BenchOptions,
    out: *std.Io.Writer,
) !BenchResult {
    // Either read a real corpus file (for cross-tokenizer comparison on
    // identical bytes) or synthesize a fixed one. Every scenario runs
    // against the same bytes so per-scenario throughputs are comparable.
    const corpus = if (opts.corpus_file) |path| blk: {
        const io = std.Io.Threaded.global_single_threaded.io();
        break :blk try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
    } else try synthesizeCorpus(allocator, opts.corpus_bytes);
    defer allocator.free(corpus);

    const selected: []const Scenario = if (opts.include) |inc| inc else &all_scenarios;

    var out_scenarios: std.ArrayList(ScenarioResult) = .empty;
    errdefer {
        for (out_scenarios.items) |*s| s.deinit(allocator);
        out_scenarios.deinit(allocator);
    }

    for (selected) |sc| {
        const r = runOne(allocator, sc, opts, corpus) catch |err| blk: {
            const msg = std.fmt.allocPrint(allocator, "{s}", .{@errorName(err)}) catch null;
            break :blk ScenarioResult{
                .name = sc,
                .status = .error_,
                .error_message = msg,
            };
        };
        try out_scenarios.append(allocator, r);
    }

    var result = BenchResult{
        .iters = opts.iters,
        .corpus_bytes = corpus.len,
        .scenarios = try out_scenarios.toOwnedSlice(allocator),
    };
    errdefer result.deinit(allocator);

    switch (opts.format) {
        .text => try writeText(out, &result),
        .json => try writeJson(out, &result),
    }
    return result;
}

// ----------------------------------------------------------------------
// Per-scenario runner.
// ----------------------------------------------------------------------

/// Which pre-tokenizer the scenario should run with. Different vocabs
/// expect different splits (cl100k regex, HF ByteLevel, or identity for
/// SP / TM / Monster which tokenize raw bytes). The bench harness only
/// cares about throughput shape — exact equivalence with the upstream
/// tool is verified by `bench/equivalence_check.py`, not here.
pub const PretokChoice = enum { identity, cl100k, hf_byte_level };

const ScenarioSpec = struct {
    fixture: []const u8,
    pretok: PretokChoice = .identity,
    /// Format hint for the loader — null means auto-detect.
    format: ?auto_detect.Format = null,
};

fn specFor(sc: Scenario) ScenarioSpec {
    return switch (sc) {
        // cl100k_base BPE expects to run on cl100k-pretokenized
        // input — running it identity-split makes the encoder swallow
        // the whole 1 MB corpus as one chunk (~10× slower). Match the
        // production deployment shape.
        .cl100k => .{ .fixture = "cl100k_base.tiktoken", .pretok = .cl100k, .format = .tiktoken },
        // SP-BPE / SP-Unigram / TM all use the model's own segmentation
        // over raw bytes — identity pre-tok matches production shape.
        .sp_bpe => .{ .fixture = "llama2.model", .pretok = .identity, .format = .sentencepiece },
        .sp_unigram => .{ .fixture = "t5_unigram.model", .pretok = .identity, .format = .sentencepiece },
        .tm => .{ .fixture = "tm_englishcode_32k.ztm", .pretok = .identity, .format = .ztm },
        // HF-BPE byte-level fixtures use the GPT-2-style ByteLevel
        // pre-tokenizer (regex split + byte_to_unicode mapping in one
        // pass). Even Phi-3 (Llama-2-style tokenizer wrapped in HF
        // JSON) runs the byte_level pretok here for throughput shape;
        // its custom Prepend(U+2581)+Replace normalizer would only
        // matter for the equivalence-check path, which lives in
        // bench/equivalence_check.py.
        .hf_bpe => .{ .fixture = "gpt2_hf.json", .pretok = .hf_byte_level, .format = .hf_json },
        // ---- Extended (post-1.17 agent E) ----
        .mistral7b => .{ .fixture = "mistral7b.model", .pretok = .identity, .format = .sentencepiece },
        .yi6b => .{ .fixture = "yi6b.model", .pretok = .identity, .format = .sentencepiece },
        .phi3 => .{ .fixture = "phi3.json", .pretok = .hf_byte_level, .format = .hf_json },
        .falcon7b => .{ .fixture = "falcon7b.json", .pretok = .hf_byte_level, .format = .hf_json },
        .deepseek_v2 => .{ .fixture = "deepseek_v2.json", .pretok = .hf_byte_level, .format = .hf_json },
        .qwen2 => .{ .fixture = "qwen2.json", .pretok = .hf_byte_level, .format = .hf_json },
        .llama3 => .{ .fixture = "llama3.json", .pretok = .hf_byte_level, .format = .hf_json },
    };
}

fn runOne(
    allocator: std.mem.Allocator,
    sc: Scenario,
    opts: BenchOptions,
    corpus: []const u8,
) !ScenarioResult {
    const spec = specFor(sc);

    // Resolve fixture path. Skip cleanly if the vocab isn't present.
    const primary = try std.fs.path.join(allocator, &.{ opts.vocab_root, spec.fixture });
    errdefer allocator.free(primary);

    // Determine the path we'll actually load. For cl100k we fall back
    // to /tmp/cl100k_base.tiktoken (the tiktoken Python wheel's cache
    // location) when the vendored fixture isn't present.
    var resolved_path: []u8 = primary;
    if (!fileExists(resolved_path)) {
        if (sc == .cl100k) {
            const alt = try allocator.dupe(u8, "/tmp/cl100k_base.tiktoken");
            if (fileExists(alt)) {
                allocator.free(resolved_path);
                resolved_path = alt;
            } else {
                allocator.free(alt);
                return ScenarioResult{
                    .name = sc,
                    .status = .skipped,
                    .vocab_path = resolved_path,
                };
            }
        } else {
            return ScenarioResult{
                .name = sc,
                .status = .skipped,
                .vocab_path = resolved_path,
            };
        }
    }

    // Load the model. `OwnedModel` mirrors `main.zig`'s autodetect
    // wrapper; we inline a stripped-down version here so this module
    // stays self-contained.
    var loaded = try loadModel(allocator, sc, resolved_path);
    defer loaded.deinit();

    var vocab = Vocab.empty(allocator);
    defer vocab.deinit();

    const pretok: PreTokenizer = switch (spec.pretok) {
        .identity => .identity,
        .cl100k => .cl100k,
        .hf_byte_level => .hf_byte_level,
    };
    const decoder: Decoder = switch (spec.pretok) {
        // The HF byte_level pretok maps raw bytes to U+0021..U+0142
        // printable codepoints; decoding back to the original bytes
        // requires the matching byte_level decoder. Other shapes
        // round-trip through `.concat`.
        .hf_byte_level => .byte_level,
        else => .concat,
    };
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = pretok,
        .model = loaded.modelValue(),
        .decoder = decoder,
        .vocab = &vocab,
    };

    var shapes: std.ArrayList(ShapeResult) = .empty;
    errdefer shapes.deinit(allocator);

    // Single-thread shape always runs.
    const single = try timeSingle(allocator, &pipe, corpus, opts.iters);
    try shapes.append(allocator, single);

    if (!opts.single_only) {
        // Batch ×8, no pinning.
        if (timeBatch(allocator, &pipe, corpus, opts.iters, 8, false)) |b8| {
            try shapes.append(allocator, b8);
        } else |_| {} // batch failures (e.g. OOM) don't tank the scenario

        // Batch ×48 + pin-physical. On hosts without 48 logical CPUs
        // this falls back to whatever the OS allows — the BatchPool
        // already degrades gracefully.
        if (timeBatch(allocator, &pipe, corpus, opts.iters, 48, true)) |b48| {
            try shapes.append(allocator, b48);
        } else |_| {}
    }

    return ScenarioResult{
        .name = sc,
        .status = .ok,
        .vocab_path = resolved_path,
        .vocab_size = loaded.vocabSize(),
        .ids = single.ids,
        .results = try shapes.toOwnedSlice(allocator),
    };
}

// ----------------------------------------------------------------------
// Timing helpers.
// ----------------------------------------------------------------------

// Zig 0.16's std.time was gutted; clock_gettime via libc is two lines
// and accurate enough for our coarse-grained timing — we always run
// >=1ms of work per shape per iter. Mirrors `bench/bench_ztok.zig`.
const Timespec = extern struct { sec: i64, nsec: i64 };
const CLOCK_MONOTONIC: c_int = 1;
extern "c" fn clock_gettime(clk_id: c_int, tp: *Timespec) c_int;

fn nanosNow() u64 {
    var ts: Timespec = .{ .sec = 0, .nsec = 0 };
    _ = clock_gettime(CLOCK_MONOTONIC, &ts);
    return @intCast(ts.sec * 1_000_000_000 + ts.nsec);
}

fn timeSingle(
    allocator: std.mem.Allocator,
    pipe: *const Pipeline,
    corpus: []const u8,
    iters: u32,
) !ShapeResult {
    // Warm-up: one encode primes the per-thread arena + any
    // lazily-initialized hot tables.
    {
        const warm = try pipe.encode(allocator, corpus);
        allocator.free(warm);
    }

    var total_ids: usize = 0;
    const t0 = nanosNow();
    var k: u32 = 0;
    while (k < iters) : (k += 1) {
        const ids = try pipe.encode(allocator, corpus);
        total_ids = ids.len;
        allocator.free(ids);
    }
    const ns = nanosNow() - t0;

    return computeShape(.single, corpus.len, iters, ns, total_ids);
}

fn timeBatch(
    allocator: std.mem.Allocator,
    pipe: *const Pipeline,
    corpus: []const u8,
    iters: u32,
    chunks: u32,
    pin_physical: bool,
) !ShapeResult {
    var pool = try thread_pool.BatchPool.initWithOptions(allocator, null, .{
        .pin_to_physical_cores = pin_physical,
    });
    defer pool.deinit();

    // Warm-up: one batch primes worker arenas.
    {
        const warm = try pipe.encodeChunked(allocator, &pool, corpus, chunks);
        allocator.free(warm);
    }

    var total_ids: usize = 0;
    const t0 = nanosNow();
    var k: u32 = 0;
    while (k < iters) : (k += 1) {
        const ids = try pipe.encodeChunked(allocator, &pool, corpus, chunks);
        total_ids = ids.len;
        allocator.free(ids);
    }
    const ns = nanosNow() - t0;

    const shape: ShapeKind = if (chunks >= 48 and pin_physical) .batch_48_pin else .batch_8;
    return computeShape(shape, corpus.len, iters, ns, total_ids);
}

fn computeShape(
    shape: ShapeKind,
    corpus_bytes: usize,
    iters: u32,
    elapsed_ns: u64,
    ids: usize,
) ShapeResult {
    const bytes_total = @as(u64, corpus_bytes) * iters;
    const seconds = @as(f64, @floatFromInt(elapsed_ns)) / 1e9;
    const mb_per_sec = if (seconds > 0)
        (@as(f64, @floatFromInt(bytes_total)) / seconds) / 1e6
    else
        0.0;
    const ms_per_iter = (@as(f64, @floatFromInt(elapsed_ns)) / 1e6) /
        @as(f64, @floatFromInt(@max(iters, 1)));
    return .{
        .shape = shape,
        .mb_per_sec = mb_per_sec,
        .ms_per_iter = ms_per_iter,
        .ids = ids,
    };
}

// ----------------------------------------------------------------------
// Model loading.
// ----------------------------------------------------------------------

const OwnedModel = union(enum) {
    bpe: Bpe,
    unigram: Unigram,
    wordpiece: WordPiece,
    monster: Monster,
    rwkv_world: RwkvWorld,

    fn deinit(self: *OwnedModel) void {
        switch (self.*) {
            .bpe => |*b| b.deinit(),
            .unigram => |*u| u.deinit(),
            .wordpiece => |*w| w.deinit(),
            .monster => |*m| m.deinit(),
            .rwkv_world => |*r| r.deinit(),
        }
    }

    fn modelValue(self: *const OwnedModel) Model {
        return switch (self.*) {
            .bpe => |*b| .{ .bpe = b },
            .unigram => |*u| .{ .unigram = u },
            .wordpiece => |*w| .{ .wordpiece = w },
            .monster => |*m| .{ .monster = m },
            .rwkv_world => |*r| .{ .rwkv_world = r },
        };
    }

    fn vocabSize(self: *const OwnedModel) u32 {
        return switch (self.*) {
            .bpe => |*b| b.count,
            .unigram => |*u| u.count,
            .wordpiece => |*w| w.count,
            .monster => |*m| m.count,
            .rwkv_world => |*r| r.count,
        };
    }
};

fn loadModel(allocator: std.mem.Allocator, sc: Scenario, path: []const u8) !OwnedModel {
    const spec = specFor(sc);
    const fmt: auto_detect.Format = spec.format orelse blk: {
        const detected = auto_detect.detectFile(path) catch auto_detect.Format.unknown;
        break :blk detected;
    };

    switch (fmt) {
        .tiktoken => return .{ .bpe = try Bpe.loadTiktokenFile(allocator, path) },
        .hf_json => {
            var hf = try hf_json.loadFromFile(allocator, path);
            defer hf.deinit();
            return switch (hf.model_kind) {
                .bpe => .{ .bpe = try hf_bridge.bpeFromHF(allocator, &hf) },
                .unigram => .{ .unigram = try hf_bridge.unigramFromHF(allocator, &hf) },
                .wordpiece => .{ .wordpiece = try hf_bridge.wordPieceFromHF(allocator, &hf, .{ .unk_id = 0 }) },
            };
        },
        .sentencepiece => {
            var sp = try sp_model.loadFromFile(allocator, path);
            defer sp.deinit();
            return switch (sp.model_kind) {
                .bpe => .{ .bpe = try sp_bridge.bpeFromSP(allocator, &sp) },
                .unigram => .{ .unigram = try sp_bridge.unigramFromSP(allocator, &sp) },
                else => return error.UnsupportedSpModelKind,
            };
        },
        .ztm => {
            const loaded = try monster_io.readFileMeta(allocator, path);
            return .{ .monster = loaded.monster };
        },
        .rwkv => return .{ .rwkv_world = try RwkvWorld.loadFromFile(allocator, path) },
        .tekken => {
            // Tekken loader returns a wrapper that owns a Bpe + special
            // tokens + pattern string. The bench harness only needs the
            // raw Bpe — move it out before dropping the wrapper.
            const tekken_mod = @import("tekken.zig");
            var tk = try tekken_mod.loadTekkenFile(allocator, path);
            const bpe_out = tk.bpe;
            tk.bpe = .{
                .allocator = allocator,
                .bytes = &.{},
                .offsets = &.{},
                .count = 0,
                .by_bytes = std.StringHashMap(TokenId).init(allocator),
            };
            tk.deinit();
            return .{ .bpe = bpe_out };
        },
        .unknown => return error.UnknownModelFormat,
    }
}

// ----------------------------------------------------------------------
// Filesystem + corpus helpers.
// ----------------------------------------------------------------------

fn fileExists(path: []const u8) bool {
    // Zig 0.16 removed std.fs.cwd(); use std.Io.Dir.cwd() with a
    // throwaway open to test for existence. `openFile` errors with
    // FileNotFound (or any I/O error) → not present from our pov.
    const io = std.Io.Threaded.global_single_threaded.io();
    var f = std.Io.Dir.cwd().openFile(io, path, .{}) catch return false;
    f.close(io);
    return true;
}

/// Synthesize a deterministic, ASCII-heavy corpus of approximately
/// `target_bytes`. We use a fixed lorem-style seed text repeated until
/// the target length is reached so every scenario sees the same input.
fn synthesizeCorpus(allocator: std.mem.Allocator, target_bytes: usize) ![]u8 {
    const seed =
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " ++
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. " ++
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris " ++
        "nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in " ++
        "reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla " ++
        "pariatur. Excepteur sint occaecat cupidatat non proident, sunt in " ++
        "culpa qui officia deserunt mollit anim id est laborum.\n" ++
        "The quick brown fox jumps over the lazy dog. " ++
        "Pack my box with five dozen liquor jugs. " ++
        "Sphinx of black quartz, judge my vow.\n" ++
        "Multi-byte sample: caf\xc3\xa9 na\xc3\xafve fa\xc3\xa7ade r\xc3\xa9sum\xc3\xa9. " ++
        "CJK sample: \xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e \xe4\xb8\xad\xe6\x96\x87.\n";

    var buf = try allocator.alloc(u8, target_bytes);
    errdefer allocator.free(buf);

    var written: usize = 0;
    while (written < target_bytes) {
        const remaining = target_bytes - written;
        const n = @min(remaining, seed.len);
        @memcpy(buf[written .. written + n], seed[0..n]);
        written += n;
    }
    return buf;
}

// ----------------------------------------------------------------------
// Text formatter.
// ----------------------------------------------------------------------

fn shapeLabel(s: ShapeKind) []const u8 {
    return switch (s) {
        .single => "single-thread",
        .batch_8 => "batch x8",
        .batch_48_pin => "batch x48 +pin",
    };
}

fn statusLabel(s: ScenarioStatus) []const u8 {
    return switch (s) {
        .ok => "OK",
        .skipped => "SKIP",
        .error_ => "ERROR",
    };
}

fn writeText(out: *std.Io.Writer, r: *const BenchResult) !void {
    try out.print("ztok bench (iters={d}, corpus={d} bytes)\n\n", .{ r.iters, r.corpus_bytes });
    try out.writeAll(
        "  scenario      status  vocab_size       shape       MB/s   ms/iter      ids\n" ++
            "  ------------- ------  ----------  --------------  -------  --------  -------\n",
    );

    for (r.scenarios) |s| {
        const name = s.name.cliName();
        const status = statusLabel(s.status);
        switch (s.status) {
            .skipped => {
                try out.print("  {s:<13} {s:<6}  {s:<10}  {s:<14}    {s:<6}  {s:<8}  {s:<7}\n", .{
                    name, status, "-", "-", "-", "-", "-",
                });
                if (s.vocab_path) |vp| {
                    try out.print("      (missing fixture: {s})\n", .{vp});
                }
            },
            .error_ => {
                try out.print("  {s:<13} {s:<6}  {s:<10}  {s:<14}    {s:<6}  {s:<8}  {s:<7}\n", .{
                    name, status, "-", "-", "-", "-", "-",
                });
                if (s.error_message) |msg| {
                    try out.print("      error: {s}\n", .{msg});
                }
            },
            .ok => {
                // First row carries vocab_size + status; subsequent shapes
                // indent under the same scenario.
                var first: bool = true;
                for (s.results) |sr| {
                    if (first) {
                        try out.print("  {s:<13} {s:<6}  {d:<10}  {s:<14}  {d:>7.1}  {d:>8.2}  {d:>7}\n", .{
                            name, status, s.vocab_size, shapeLabel(sr.shape),
                            sr.mb_per_sec, sr.ms_per_iter, sr.ids,
                        });
                        first = false;
                    } else {
                        try out.print("  {s:<13} {s:<6}  {s:<10}  {s:<14}  {d:>7.1}  {d:>8.2}  {d:>7}\n", .{
                            "", "", "", shapeLabel(sr.shape),
                            sr.mb_per_sec, sr.ms_per_iter, sr.ids,
                        });
                    }
                }
                if (first) {
                    // No shapes recorded — shouldn't happen but keep the
                    // table consistent.
                    try out.print("  {s:<13} {s:<6}  {d:<10}  {s:<14}    {s:<6}  {s:<8}  {s:<7}\n", .{
                        name, status, s.vocab_size, "(no shapes)", "-", "-", "-",
                    });
                }
            },
        }
    }

    // Footer summary.
    var ok_count: u32 = 0;
    var skipped_count: u32 = 0;
    var error_count: u32 = 0;
    for (r.scenarios) |s| switch (s.status) {
        .ok => ok_count += 1,
        .skipped => skipped_count += 1,
        .error_ => error_count += 1,
    };
    try out.print("\nSummary: {d} ok, {d} skipped, {d} errors\n", .{ ok_count, skipped_count, error_count });
}

// ----------------------------------------------------------------------
// JSON formatter.
// ----------------------------------------------------------------------

fn writeJsonString(out: *std.Io.Writer, s: []const u8) !void {
    try out.writeByte('"');
    for (s) |c| switch (c) {
        '"' => try out.writeAll("\\\""),
        '\\' => try out.writeAll("\\\\"),
        '\n' => try out.writeAll("\\n"),
        '\r' => try out.writeAll("\\r"),
        '\t' => try out.writeAll("\\t"),
        0x08 => try out.writeAll("\\b"),
        0x0C => try out.writeAll("\\f"),
        0x00...0x07, 0x0B, 0x0E...0x1F => try out.print("\\u{x:0>4}", .{c}),
        else => try out.writeByte(c),
    };
    try out.writeByte('"');
}

fn writeJsonFloat(out: *std.Io.Writer, v: f64) !void {
    if (std.math.isNan(v) or std.math.isInf(v)) {
        try out.writeAll("null");
    } else {
        try out.print("{d:.4}", .{v});
    }
}

fn shapeJsonName(s: ShapeKind) []const u8 {
    return switch (s) {
        .single => "single",
        .batch_8 => "batch_8",
        .batch_48_pin => "batch_48_pin",
    };
}

fn statusJsonName(s: ScenarioStatus) []const u8 {
    return switch (s) {
        .ok => "ok",
        .skipped => "skipped",
        .error_ => "error",
    };
}

fn writeJson(out: *std.Io.Writer, r: *const BenchResult) !void {
    try out.writeAll("{");
    try out.writeAll("\"version\":1,");
    try out.writeAll("\"tool\":\"ztok\",");
    try out.print("\"iters\":{d},", .{r.iters});
    try out.print("\"corpus_bytes\":{d},", .{r.corpus_bytes});
    try out.writeAll("\"scenarios\":[");

    var first_sc = true;
    for (r.scenarios) |s| {
        if (!first_sc) try out.writeAll(",");
        first_sc = false;
        try out.writeAll("{");
        try out.writeAll("\"name\":");
        try writeJsonString(out, s.name.cliName());
        try out.writeAll(",\"status\":");
        try writeJsonString(out, statusJsonName(s.status));
        try out.writeAll(",\"vocab_path\":");
        if (s.vocab_path) |p| {
            try writeJsonString(out, p);
        } else {
            try out.writeAll("null");
        }
        try out.print(",\"vocab_size\":{d},\"ids\":{d},\"results\":[", .{ s.vocab_size, s.ids });
        var first_sh = true;
        for (s.results) |sh| {
            if (!first_sh) try out.writeAll(",");
            first_sh = false;
            try out.writeAll("{\"shape\":");
            try writeJsonString(out, shapeJsonName(sh.shape));
            try out.writeAll(",\"mb_per_sec\":");
            try writeJsonFloat(out, sh.mb_per_sec);
            try out.writeAll(",\"ms_per_iter\":");
            try writeJsonFloat(out, sh.ms_per_iter);
            try out.print(",\"ids\":{d}", .{sh.ids});
            try out.writeAll("}");
        }
        try out.writeAll("],\"error\":");
        if (s.error_message) |m| {
            try writeJsonString(out, m);
        } else {
            try out.writeAll("null");
        }
        try out.writeAll("}");
    }
    try out.writeAll("]}\n");
}

// ======================================================================
// Tests
// ======================================================================

const testing = std.testing;

fn captureOutput(comptime cap: usize) type {
    return struct {
        buf: [cap]u8 = undefined,
        writer: std.Io.Writer = undefined,

        const Self = @This();
        fn init(self: *Self) void {
            self.writer = .fixed(&self.buf);
        }
        fn slice(self: *Self) []const u8 {
            return self.writer.buffered();
        }
    };
}

test "runBench cl100k completes and reports >0 MB/s when fixture available" {
    var cap: captureOutput(16384) = .{};
    cap.init();

    var result = try runBench(testing.allocator, .{
        .iters = 1,
        .include = &[_]Scenario{.cl100k},
        .vocab_root = "/tmp",
        .corpus_bytes = 64 * 1024,
        .single_only = true,
    }, &cap.writer);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 1), result.scenarios.len);
    const sc = result.scenarios[0];
    try testing.expectEqual(Scenario.cl100k, sc.name);

    // The test runs only if /tmp/cl100k_base.tiktoken exists locally
    // (CI environments without the cached tiktoken won't trigger the
    // throughput assertion; we still validate the skipped-status path
    // exits cleanly).
    switch (sc.status) {
        .ok => {
            try testing.expect(sc.results.len >= 1);
            try testing.expect(sc.results[0].mb_per_sec > 0.0);
            try testing.expect(sc.ids > 0);
        },
        .skipped => {
            // Acceptable — fixture not present in this env.
            try testing.expectEqual(@as(usize, 0), sc.results.len);
        },
        .error_ => return error.UnexpectedError,
    }
}

test "runBench --format json output parses as valid JSON" {
    var cap: captureOutput(16384) = .{};
    cap.init();

    // Use an intentionally-empty vocab root so every scenario reports
    // as skipped — the test asserts the JSON shape independently of
    // whether fixtures are present.
    var result = try runBench(testing.allocator, .{
        .iters = 1,
        .vocab_root = "/nonexistent_dir_for_test",
        .format = .json,
        .corpus_bytes = 1024,
        .single_only = true,
    }, &cap.writer);
    defer result.deinit(testing.allocator);

    const s = cap.slice();
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, s, .{});
    defer parsed.deinit();

    const root = parsed.value;
    try testing.expect(root == .object);
    try testing.expect(root.object.get("version") != null);
    try testing.expectEqual(@as(i64, 1), root.object.get("version").?.integer);
    try testing.expectEqualStrings("ztok", root.object.get("tool").?.string);
    try testing.expect(root.object.get("scenarios") != null);

    const scenarios = root.object.get("scenarios").?.array;
    try testing.expectEqual(all_scenarios.len, scenarios.items.len);
    for (scenarios.items) |sc| {
        try testing.expect(sc.object.get("name") != null);
        try testing.expect(sc.object.get("status") != null);
        // Every scenario should report skipped (no fixtures in
        // /nonexistent_dir_for_test) EXCEPT cl100k, which falls back
        // to /tmp/cl100k_base.tiktoken when that cached file exists.
        // Accept either "skipped" or "ok" for cl100k; require "skipped"
        // for the others.
        const name = sc.object.get("name").?.string;
        const status = sc.object.get("status").?.string;
        if (std.mem.eql(u8, name, "cl100k")) {
            try testing.expect(std.mem.eql(u8, status, "skipped") or std.mem.eql(u8, status, "ok"));
        } else {
            try testing.expectEqualStrings("skipped", status);
        }
        try testing.expect(sc.object.get("results") != null);
    }
}

test "runBench --include cl100k runs only cl100k" {
    var cap: captureOutput(8192) = .{};
    cap.init();

    var result = try runBench(testing.allocator, .{
        .iters = 1,
        .include = &[_]Scenario{.cl100k},
        .vocab_root = "/nonexistent_dir_for_test",
        .format = .json,
        .corpus_bytes = 1024,
        .single_only = true,
    }, &cap.writer);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 1), result.scenarios.len);
    try testing.expectEqual(Scenario.cl100k, result.scenarios[0].name);
}

test "runBench reports missing-vocab scenario as skipped (no crash)" {
    var cap: captureOutput(8192) = .{};
    cap.init();

    var result = try runBench(testing.allocator, .{
        .iters = 1,
        .include = &[_]Scenario{ .sp_bpe, .sp_unigram, .tm, .hf_bpe },
        .vocab_root = "/nonexistent_dir_for_test",
        .format = .text,
        .corpus_bytes = 1024,
        .single_only = true,
    }, &cap.writer);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 4), result.scenarios.len);
    for (result.scenarios) |sc| {
        try testing.expectEqual(ScenarioStatus.skipped, sc.status);
        // Skipped scenarios carry the attempted path for diagnostics.
        try testing.expect(sc.vocab_path != null);
    }

    const s = cap.slice();
    try testing.expect(std.mem.indexOf(u8, s, "SKIP") != null);
    try testing.expect(std.mem.indexOf(u8, s, "Summary:") != null);
}

test "Scenario.parse round-trips every variant" {
    inline for (all_scenarios) |sc| {
        const parsed = Scenario.parse(sc.cliName());
        try testing.expect(parsed != null);
        try testing.expectEqual(sc, parsed.?);
    }
    try testing.expect(Scenario.parse("nonexistent") == null);
    try testing.expect(Scenario.parse("") == null);
}

// ----------------------------------------------------------------------
// Post-1.17 extended-fixture tests (agent E).
//
// These cover the new Mistral-7B / Yi-6B / Phi-3 / Falcon-7B /
// DeepSeek-V2 / Qwen2 / Llama-3 scenarios added alongside the
// `bench/fetch_vocabs.py --extended` downloader. The bench harness is
// designed so a missing fixture reports `skipped` rather than aborting
// the whole run, so we exercise that path explicitly — the alternative
// (asserting throughput on a real fixture) would couple CI to a 25 MB
// vendored set we don't always carry.
// ----------------------------------------------------------------------

test "extended scenarios all parse + appear in all_scenarios" {
    const expected = [_]Scenario{
        .mistral7b, .yi6b, .phi3, .falcon7b, .deepseek_v2, .qwen2, .llama3,
    };
    for (expected) |want| {
        var seen = false;
        for (all_scenarios) |sc| if (sc == want) {
            seen = true;
            break;
        };
        try testing.expect(seen);
    }
    // CLI names must round-trip through Scenario.parse.
    try testing.expectEqual(Scenario.mistral7b, Scenario.parse("mistral7b").?);
    try testing.expectEqual(Scenario.yi6b, Scenario.parse("yi6b").?);
    try testing.expectEqual(Scenario.phi3, Scenario.parse("phi3").?);
    try testing.expectEqual(Scenario.falcon7b, Scenario.parse("falcon7b").?);
    try testing.expectEqual(Scenario.deepseek_v2, Scenario.parse("deepseek-v2").?);
    try testing.expectEqual(Scenario.qwen2, Scenario.parse("qwen2").?);
    try testing.expectEqual(Scenario.llama3, Scenario.parse("llama3").?);
}

test "extended fixtures report skipped when not vendored" {
    // Point at an empty directory so every extended scenario reports
    // skipped — independent of whether the developer has run
    // `bench/fetch_vocabs.py --extended` on this checkout.
    var cap: captureOutput(8192) = .{};
    cap.init();

    var result = try runBench(testing.allocator, .{
        .iters = 1,
        .include = &[_]Scenario{
            .mistral7b, .yi6b, .phi3, .falcon7b, .deepseek_v2, .qwen2, .llama3,
        },
        .vocab_root = "/nonexistent_dir_for_test",
        .format = .text,
        .corpus_bytes = 1024,
        .single_only = true,
    }, &cap.writer);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 7), result.scenarios.len);
    for (result.scenarios) |sc| {
        try testing.expectEqual(ScenarioStatus.skipped, sc.status);
        try testing.expect(sc.vocab_path != null);
    }
}

test "runBench --include mistral7b,phi3,qwen2 runs only those scenarios" {
    // Verifies the include-filter behavior reported in the task spec:
    // selecting a subset of extended fixtures must produce exactly those
    // scenarios (in order) regardless of fixture presence.
    var cap: captureOutput(8192) = .{};
    cap.init();

    const include = [_]Scenario{ .mistral7b, .phi3, .qwen2 };
    var result = try runBench(testing.allocator, .{
        .iters = 1,
        .include = &include,
        .vocab_root = "/nonexistent_dir_for_test",
        .format = .json,
        .corpus_bytes = 1024,
        .single_only = true,
    }, &cap.writer);
    defer result.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 3), result.scenarios.len);
    try testing.expectEqual(Scenario.mistral7b, result.scenarios[0].name);
    try testing.expectEqual(Scenario.phi3, result.scenarios[1].name);
    try testing.expectEqual(Scenario.qwen2, result.scenarios[2].name);
}

test "specFor wires extended HF-BPE scenarios to hf_byte_level pretok" {
    // The extended HF-BPE scenarios all need ByteLevel pre-tokenization
    // (regex split + byte_to_unicode) — running them through identity
    // pretok would (a) lose the GPT-2 ByteLevel mapping and (b) make
    // the BPE encoder swallow the whole 1 MB corpus as one chunk,
    // bombing throughput by ~10×. Pin the spec choice so a future
    // refactor can't silently downgrade.
    const hf_scenarios = [_]Scenario{
        .hf_bpe, .phi3, .falcon7b, .deepseek_v2, .qwen2, .llama3,
    };
    for (hf_scenarios) |sc| {
        const spec = specFor(sc);
        try testing.expectEqual(PretokChoice.hf_byte_level, spec.pretok);
        try testing.expectEqual(auto_detect.Format.hf_json, spec.format.?);
    }
    // SP-BPE scenarios stay on identity pretok.
    const sp_scenarios = [_]Scenario{ .sp_bpe, .sp_unigram, .mistral7b, .yi6b };
    for (sp_scenarios) |sc| {
        const spec = specFor(sc);
        try testing.expectEqual(PretokChoice.identity, spec.pretok);
        try testing.expectEqual(auto_detect.Format.sentencepiece, spec.format.?);
    }
}
