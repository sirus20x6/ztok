//! ztok command-line interface.
//!
//! Subcommands:
//!   ztok encode    --model PATH [--cl100k] [TEXT]
//!   ztok decode    --model PATH ID...
//!   ztok info      --model PATH
//!   ztok chunk     --model PATH [--cl100k] --max-tokens N [--overlap N] [--boundary MODE] [--format jsonl|text] [TEXT]
//!   ztok validate  --model PATH [--cl100k] [--checks CHECK1,CHECK2,...] [--format text|json] [--fixtures PATH]
//!   ztok roundtrip --model PATH [--cl100k] [--optimal] [--summary] [INPUT_FILE|--stdin]
//!   ztok diff      --a PATH --b PATH [--cl100k] [--format text|json] [INPUT_FILE|--stdin]
//!   ztok eval      --model PATH [--cl100k] [--format text|json] [--top-k N] [--bottom-k N] [--hidden-dim N] [--fairness] [INPUT_FILE|--stdin]
//!   ztok train     --kind bpe|unigram|wordpiece|monster --input PATH --vocab-size N --output PATH [--cl100k] [--threads N] [model-specific flags]
//!   ztok serve     --model PATH [--cl100k] [--host HOST] [--port N] [--workers N]
//!   ztok grpc-serve --model PATH [--cl100k] [--host HOST] [--port N] [--workers N]
//!
//! TEXT defaults to stdin when omitted on `encode` and `chunk`.

const std = @import("std");
const ztok = @import("ztok");

// Single source of truth — see ztok.VERSION (root.zig). Parsed from
// build.zig.zon at build time via addOptions.
const VERSION = ztok.VERSION;

const Cmd = enum { encode, encode_multimodal, decode, info, explain, train, tokenize_dataset, chunk, validate, roundtrip, diff, eval, transcode, serve, grpc_serve, bench, fingerprint, adapt_vocab, merge_vocab, visualize, help, version };

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var arg_iter = try std.process.Args.Iterator.initAllocator(init.minimal.args, gpa);
    defer arg_iter.deinit();

    var argv: std.ArrayList([]const u8) = .empty;
    defer {
        for (argv.items) |s| gpa.free(s);
        argv.deinit(gpa);
    }
    while (arg_iter.next()) |a| try argv.append(gpa, try gpa.dupe(u8, a));

    var stdout_buf: [4096]u8 = undefined;
    var stdout_file = std.Io.File.stdout();
    var stdout_w = stdout_file.writer(io, &stdout_buf);
    const out = &stdout_w.interface;
    defer out.flush() catch {};

    if (argv.items.len < 2) {
        try printUsage(out);
        return;
    }

    const cmd = parseCmd(argv.items[1]) orelse {
        try out.print("unknown subcommand: {s}\n\n", .{argv.items[1]});
        try printUsage(out);
        std.process.exit(2);
    };

    const rest = argv.items[2..];
    switch (cmd) {
        .help => try printUsage(out),
        .version => try out.print("ztok {s}\n", .{VERSION}),
        .encode => try cmdEncode(gpa, io, rest, out),
        .encode_multimodal => try cmdEncodeMultimodal(gpa, io, rest, out),
        .decode => try cmdDecode(gpa, rest, out),
        .info => try cmdInfo(gpa, rest, out),
        .explain => try cmdExplain(gpa, io, rest, out),
        .train => try cmdTrain(gpa, io, rest, out),
        .tokenize_dataset => try cmdTokenizeDataset(gpa, io, rest, out),
        .chunk => try cmdChunk(gpa, io, rest, out),
        .validate => try cmdValidate(gpa, io, rest, out),
        .roundtrip => try cmdRoundtrip(gpa, io, rest, out),
        .diff => try cmdDiff(gpa, io, rest, out),
        .eval => try cmdEval(gpa, io, rest, out),
        .transcode => try cmdTranscode(gpa, io, rest, out),
        .serve => try cmdServe(gpa, io, rest, out),
        .grpc_serve => try cmdGrpcServe(gpa, io, rest, out),
        .bench => try cmdBench(gpa, rest, out),
        .fingerprint => try cmdFingerprint(gpa, io, rest, out),
        .adapt_vocab => try cmdAdaptVocab(gpa, io, rest, out),
        .merge_vocab => try cmdMergeVocab(gpa, io, rest, out),
        .visualize => try cmdVisualize(gpa, io, rest, out),
    }
}

fn parseCmd(s: []const u8) ?Cmd {
    if (std.mem.eql(u8, s, "encode")) return .encode;
    if (std.mem.eql(u8, s, "encode-multimodal")) return .encode_multimodal;
    if (std.mem.eql(u8, s, "decode")) return .decode;
    if (std.mem.eql(u8, s, "info")) return .info;
    if (std.mem.eql(u8, s, "explain")) return .explain;
    if (std.mem.eql(u8, s, "train")) return .train;
    if (std.mem.eql(u8, s, "tokenize-dataset")) return .tokenize_dataset;
    if (std.mem.eql(u8, s, "chunk")) return .chunk;
    if (std.mem.eql(u8, s, "validate")) return .validate;
    if (std.mem.eql(u8, s, "roundtrip")) return .roundtrip;
    if (std.mem.eql(u8, s, "diff")) return .diff;
    if (std.mem.eql(u8, s, "eval")) return .eval;
    if (std.mem.eql(u8, s, "transcode")) return .transcode;
    if (std.mem.eql(u8, s, "serve")) return .serve;
    if (std.mem.eql(u8, s, "grpc-serve")) return .grpc_serve;
    if (std.mem.eql(u8, s, "bench")) return .bench;
    if (std.mem.eql(u8, s, "fingerprint")) return .fingerprint;
    if (std.mem.eql(u8, s, "adapt-vocab")) return .adapt_vocab;
    if (std.mem.eql(u8, s, "merge-vocab")) return .merge_vocab;
    if (std.mem.eql(u8, s, "visualize")) return .visualize;
    if (std.mem.eql(u8, s, "help") or std.mem.eql(u8, s, "-h") or std.mem.eql(u8, s, "--help")) return .help;
    if (std.mem.eql(u8, s, "version") or std.mem.eql(u8, s, "-v") or std.mem.eql(u8, s, "--version")) return .version;
    return null;
}

fn printUsage(out: *std.Io.Writer) !void {
    try out.writeAll(
        \\ztok — fast multithreaded tokenizer
        \\
        \\Usage:
        \\  ztok encode    --model PATH [--cl100k] [--optimal] [TEXT]
        \\  ztok encode-multimodal --model tekken.json --spec content.json
        \\  ztok decode    --model PATH ID [ID...]
        \\  ztok info      --model PATH
        \\  ztok explain   --model PATH [--cl100k] [--format text|json] [TEXT|--stdin]
        \\  ztok train     --kind bpe|unigram|wordpiece|monster|pathpiece --input PATH --vocab-size N
        \\                 --output PATH [--cl100k] [--threads N]
        \\                 [--em-iterations N] [--shrink-rate F]   (unigram)
        \\                 [--branches N]                          (monster)
        \\                 [--avoid PATH] [--avoid-mode penalize|exclude]
        \\  ztok tokenize-dataset --model PATH --input PATH --output PATH [--cl100k]
        \\                 [--format bin|npy] [--seq-len N] [--dtype auto|u16|u32]
        \\                 [--doc-mode lines|whole] [--add-bos --bos-id N]
        \\                 [--add-eos --eos-id N] [--pad-last --pad-id N]
        \\  ztok chunk     --model PATH [--cl100k] --max-tokens N [--overlap N]
        \\                 [--boundary token|codepoint|word|sentence|paragraph]
        \\                 [--format jsonl|text] [TEXT]
        \\  ztok validate  --model PATH [--cl100k] [--checks CHECK1,CHECK2,...]
        \\                 [--format text|json] [--fixtures PATH]
        \\  ztok roundtrip --model PATH [--cl100k] [--optimal] [--summary] [INPUT_FILE|--stdin]
        \\  ztok diff      --a PATH --b PATH [--cl100k] [--format text|json]
        \\                 [INPUT_FILE|--stdin]
        \\  ztok diff      --report --vocab-a PATH --vocab-b PATH --corpus PATH
        \\                 [--cl100k] [--out report.md]
        \\  ztok eval      --model PATH [--cl100k] [--format text|json]
        \\                 [--top-k N] [--bottom-k N] [--hidden-dim N]
        \\                 [--fairness] [INPUT_FILE|--stdin]
        \\  ztok transcode --from VOCAB_A --to VOCAB_B [--cl100k]
        \\                 [--input ids.txt|--stdin] [--format text|jsonl]
        \\  ztok serve     --model PATH [--cl100k]
        \\                 [--host HOST] [--port N] [--workers N]
        \\                 [--auth-token TOKEN | --auth-token-file PATH]
        \\                 [--auth-oidc-issuer URL --auth-oidc-audience NAME]
        \\                 [--rate-limit REQ_PER_SEC]
        \\                 [--log-format text|json] [--metrics]
        \\                 [--tls-cert PATH --tls-key PATH]
        \\  ztok grpc-serve --model PATH [--cl100k]
        \\                 [--host HOST] [--port N] [--workers N]
        \\                 (gRPC-Web over HTTP/1.1, default port 7891)
        \\  ztok bench     [--quick] [--iters N] [--include cl100k,sp-bpe,sp-unigram,tm,hf-bpe]
        \\                 [--vocab-root DIR] [--format text|json]
        \\  ztok fingerprint VOCAB_PATH
        \\  ztok adapt-vocab --base VOCAB --corpus PATH --add N --output NEW_VOCAB
        \\                 [--report report.json]
        \\  ztok merge-vocab --a VOCAB_A --b VOCAB_B --output MERGED
        \\                 [--on-conflict keep-a|keep-b|error] [--prefix-b STR]
        \\  ztok visualize VOCAB [--corpus PATH] [--out report.html] [--top-k N]
        \\  ztok version
        \\
        \\Notes:
        \\  --model auto-detects .tiktoken / HF tokenizer.json / SentencePiece .model
        \\           / ztok Monster .ztm.
        \\  --cl100k enables the cl100k_base pre-tokenizer (default: identity).
        \\  --optimal (encode, BPE only) emits the provably FEWEST tokens via a
        \\           dynamic-programming segmentation. Beats greedy/merge-order
        \\           token counts; not byte-identical to tiktoken/HF output.
        \\  If TEXT is omitted on encode/chunk, stdin is read.
        \\  train output formats by --kind:
        \\    bpe       -> .tiktoken
        \\    unigram   -> SentencePiece .model
        \\    wordpiece -> HF tokenizer.json
        \\    monster   -> ztok .ztm
        \\    pathpiece -> .tiktoken (CTC-minimizing vocab; load as bpe + run --optimal)
        \\  diff exits non-zero if any line diverges between A and B.
        \\  transcode re-maps A-ids to B-ids via text: decode_A then encode_B.
        \\           Input = lines of space-separated A-ids; output = B-ids/line.
        \\           Exact when both vocabs round-trip text losslessly
        \\           (byte-level / byte-fallback).
        \\  eval --hidden-dim enables KV-cache byte estimate (assumes 32 layers, fp16).
        \\  validate checks (BPE):       unreachable_merges, duplicate_decodings,
        \\                                roundtrip, cl100k_pathologies, whitespace,
        \\                                special_shadowing, single_byte_coverage.
        \\  validate checks (Unigram):   roundtrip, duplicate_decodings, whitespace,
        \\                                special_shadowing, score_sanity, unk_coverage.
        \\  validate checks (WordPiece): roundtrip, duplicate_decodings, whitespace,
        \\                                special_shadowing, continuation_consistency.
        \\  validate checks (Monster):   roundtrip, duplicate_decodings, whitespace,
        \\                                special_shadowing, branch_coverage,
        \\                                lilbuf_prefix_count.
        \\
    );
}

const Args = struct {
    model_path: ?[]const u8 = null,
    input_path: ?[]const u8 = null,
    output_path: ?[]const u8 = null,
    vocab_size: ?u32 = null,
    threads: ?u32 = null,
    max_tokens: ?u32 = null,
    overlap: ?u32 = null,
    boundary: ?[]const u8 = null,
    format: ?[]const u8 = null,
    checks: ?[]const u8 = null,
    fixtures: ?[]const u8 = null,
    cl100k: bool = false,
    stdin: bool = false,
    summary: bool = false,
    trace: bool = false,
    // encode: opt into provably-minimum-token DP segmentation.
    optimal: bool = false,
    // diff
    a_path: ?[]const u8 = null,
    b_path: ?[]const u8 = null,
    // diff --report
    report: bool = false,
    out_path: ?[]const u8 = null,
    vocab_a_path: ?[]const u8 = null,
    vocab_b_path: ?[]const u8 = null,
    // eval
    top_k: ?u32 = null,
    bottom_k: ?u32 = null,
    hidden_dim: ?u32 = null,
    // eval multilingual fairness (per-script proxy)
    fairness: bool = false,
    // eval cost-estimation
    corpus_path: ?[]const u8 = null,
    price: ?f64 = null,
    price_input: ?f64 = null,
    price_output: ?f64 = null,
    price_per: ?[]const u8 = null,
    prices_file: ?[]const u8 = null,
    model_name: ?[]const u8 = null,
    // train extensions
    kind: ?[]const u8 = null,
    em_iterations: ?u32 = null,
    shrink_rate: ?f32 = null,
    branches: ?u32 = null,
    avoid_path: ?[]const u8 = null,
    avoid_mode: ?[]const u8 = null,
    // serve
    host: ?[]const u8 = null,
    port: ?u16 = null,
    workers: ?u32 = null,
    auth_token: ?[]const u8 = null,
    auth_token_file: ?[]const u8 = null,
    rate_limit: ?u32 = null,
    // serve hardening (post-1.22 agent D)
    log_format: ?[]const u8 = null,
    metrics: bool = false,
    tls_cert: ?[]const u8 = null,
    tls_key: ?[]const u8 = null,
    auth_oidc_issuer: ?[]const u8 = null,
    auth_oidc_audience: ?[]const u8 = null,
    prefix_cache_dir: ?[]const u8 = null,
    // bench
    quick: bool = false,
    include: ?[]const u8 = null,
    // adapt-vocab
    base_path: ?[]const u8 = null,
    add_count: ?u32 = null,
    vocab_root: ?[]const u8 = null,
    iters: ?u32 = null,
    corpus_bytes: ?usize = null,
    // merge-vocab
    on_conflict: ?[]const u8 = null,
    prefix_b: ?[]const u8 = null,
    // transcode
    from_path: ?[]const u8 = null,
    to_path: ?[]const u8 = null,
    // encode-multimodal
    spec_path: ?[]const u8 = null,
    // tokenize-dataset
    seq_len: ?usize = null,
    dtype: ?[]const u8 = null,
    add_bos: bool = false,
    add_eos: bool = false,
    bos_id: ?u32 = null,
    eos_id: ?u32 = null,
    pad_last: bool = false,
    pad_id: ?u32 = null,
    doc_mode: ?[]const u8 = null,
    positional: std.ArrayList([]const u8) = .empty,

    fn parse(allocator: std.mem.Allocator, raw: []const []const u8) !Args {
        var a: Args = .{};
        var i: usize = 0;
        while (i < raw.len) : (i += 1) {
            const tok = raw[i];
            if (std.mem.eql(u8, tok, "--model")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.model_path = raw[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, tok, "--input")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.input_path = raw[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, tok, "--output")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.output_path = raw[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, tok, "--vocab-size")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.vocab_size = try std.fmt.parseInt(u32, raw[i + 1], 10);
                i += 1;
            } else if (std.mem.eql(u8, tok, "--threads")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.threads = try std.fmt.parseInt(u32, raw[i + 1], 10);
                i += 1;
            } else if (std.mem.eql(u8, tok, "--max-tokens")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.max_tokens = try std.fmt.parseInt(u32, raw[i + 1], 10);
                i += 1;
            } else if (std.mem.eql(u8, tok, "--overlap")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.overlap = try std.fmt.parseInt(u32, raw[i + 1], 10);
                i += 1;
            } else if (std.mem.eql(u8, tok, "--boundary")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.boundary = raw[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, tok, "--format")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.format = raw[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, tok, "--checks")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.checks = raw[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, tok, "--fixtures")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.fixtures = raw[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, tok, "--cl100k")) {
                a.cl100k = true;
            } else if (std.mem.eql(u8, tok, "--stdin")) {
                a.stdin = true;
            } else if (std.mem.eql(u8, tok, "--summary")) {
                a.summary = true;
            } else if (std.mem.eql(u8, tok, "--trace")) {
                a.trace = true;
            } else if (std.mem.eql(u8, tok, "--optimal")) {
                a.optimal = true;
            } else if (std.mem.eql(u8, tok, "--fairness")) {
                a.fairness = true;
            } else if (std.mem.eql(u8, tok, "--a")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.a_path = raw[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, tok, "--b")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.b_path = raw[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, tok, "--report")) {
                a.report = true;
            } else if (std.mem.eql(u8, tok, "--out")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.out_path = raw[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, tok, "--vocab-a")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.vocab_a_path = raw[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, tok, "--vocab-b")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.vocab_b_path = raw[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, tok, "--top-k")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.top_k = try std.fmt.parseInt(u32, raw[i + 1], 10);
                i += 1;
            } else if (std.mem.eql(u8, tok, "--bottom-k")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.bottom_k = try std.fmt.parseInt(u32, raw[i + 1], 10);
                i += 1;
            } else if (std.mem.eql(u8, tok, "--hidden-dim")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.hidden_dim = try std.fmt.parseInt(u32, raw[i + 1], 10);
                i += 1;
            } else if (std.mem.eql(u8, tok, "--vocab")) {
                // Alias for --model (the tokenizer file). Spec uses
                // `ztok eval --vocab VOCAB --corpus CORPUS ...`.
                if (i + 1 >= raw.len) return error.MissingValue;
                a.model_path = raw[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, tok, "--corpus")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.corpus_path = raw[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, tok, "--price")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.price = try std.fmt.parseFloat(f64, raw[i + 1]);
                i += 1;
            } else if (std.mem.eql(u8, tok, "--price-input")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.price_input = try std.fmt.parseFloat(f64, raw[i + 1]);
                i += 1;
            } else if (std.mem.eql(u8, tok, "--price-output")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.price_output = try std.fmt.parseFloat(f64, raw[i + 1]);
                i += 1;
            } else if (std.mem.eql(u8, tok, "--per")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.price_per = raw[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, tok, "--prices-file")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.prices_file = raw[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, tok, "--model-name")) {
                // Picks an entry out of --prices-file. Distinct from
                // --model (the tokenizer path) to avoid clobbering it.
                if (i + 1 >= raw.len) return error.MissingValue;
                a.model_name = raw[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, tok, "--kind")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.kind = raw[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, tok, "--em-iterations")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.em_iterations = try std.fmt.parseInt(u32, raw[i + 1], 10);
                i += 1;
            } else if (std.mem.eql(u8, tok, "--shrink-rate")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.shrink_rate = try std.fmt.parseFloat(f32, raw[i + 1]);
                i += 1;
            } else if (std.mem.eql(u8, tok, "--branches")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.branches = try std.fmt.parseInt(u32, raw[i + 1], 10);
                i += 1;
            } else if (std.mem.eql(u8, tok, "--avoid")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.avoid_path = raw[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, tok, "--avoid-mode")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.avoid_mode = raw[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, tok, "--host")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.host = raw[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, tok, "--port")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.port = try std.fmt.parseInt(u16, raw[i + 1], 10);
                i += 1;
            } else if (std.mem.eql(u8, tok, "--workers")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.workers = try std.fmt.parseInt(u32, raw[i + 1], 10);
                i += 1;
            } else if (std.mem.eql(u8, tok, "--auth-token")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.auth_token = raw[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, tok, "--auth-token-file")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.auth_token_file = raw[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, tok, "--rate-limit")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.rate_limit = try std.fmt.parseInt(u32, raw[i + 1], 10);
                i += 1;
            } else if (std.mem.eql(u8, tok, "--log-format")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.log_format = raw[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, tok, "--metrics")) {
                a.metrics = true;
            } else if (std.mem.eql(u8, tok, "--tls-cert")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.tls_cert = raw[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, tok, "--tls-key")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.tls_key = raw[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, tok, "--auth-oidc-issuer")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.auth_oidc_issuer = raw[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, tok, "--auth-oidc-audience")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.auth_oidc_audience = raw[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, tok, "--prefix-cache-dir")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.prefix_cache_dir = raw[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, tok, "--quick")) {
                a.quick = true;
            } else if (std.mem.eql(u8, tok, "--include")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.include = raw[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, tok, "--vocab-root")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.vocab_root = raw[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, tok, "--iters")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.iters = try std.fmt.parseInt(u32, raw[i + 1], 10);
                i += 1;
            } else if (std.mem.eql(u8, tok, "--corpus-bytes")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.corpus_bytes = try std.fmt.parseInt(usize, raw[i + 1], 10);
                i += 1;
            } else if (std.mem.eql(u8, tok, "--seq-len")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.seq_len = try std.fmt.parseInt(usize, raw[i + 1], 10);
                i += 1;
            } else if (std.mem.eql(u8, tok, "--dtype")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.dtype = raw[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, tok, "--add-bos")) {
                a.add_bos = true;
            } else if (std.mem.eql(u8, tok, "--add-eos")) {
                a.add_eos = true;
            } else if (std.mem.eql(u8, tok, "--bos-id")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.bos_id = try std.fmt.parseInt(u32, raw[i + 1], 10);
                i += 1;
            } else if (std.mem.eql(u8, tok, "--eos-id")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.eos_id = try std.fmt.parseInt(u32, raw[i + 1], 10);
                i += 1;
            } else if (std.mem.eql(u8, tok, "--pad-last")) {
                a.pad_last = true;
            } else if (std.mem.eql(u8, tok, "--pad-id")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.pad_id = try std.fmt.parseInt(u32, raw[i + 1], 10);
                i += 1;
            } else if (std.mem.eql(u8, tok, "--doc-mode")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.doc_mode = raw[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, tok, "--base")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.base_path = raw[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, tok, "--add")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.add_count = try std.fmt.parseInt(u32, raw[i + 1], 10);
                i += 1;
            } else if (std.mem.eql(u8, tok, "--on-conflict")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.on_conflict = raw[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, tok, "--prefix-b")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.prefix_b = raw[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, tok, "--from")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.from_path = raw[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, tok, "--to")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.to_path = raw[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, tok, "--spec")) {
                if (i + 1 >= raw.len) return error.MissingValue;
                a.spec_path = raw[i + 1];
                i += 1;
            } else {
                try a.positional.append(allocator, tok);
            }
        }
        return a;
    }

    fn deinit(self: *Args, allocator: std.mem.Allocator) void {
        self.positional.deinit(allocator);
    }
};

fn cmdEncode(gpa: std.mem.Allocator, io: std.Io, raw: []const []const u8, out: *std.Io.Writer) !void {
    var args = try Args.parse(gpa, raw);
    defer args.deinit(gpa);

    const model_path = args.model_path orelse {
        try out.writeAll("encode: --model PATH required\n");
        return;
    };

    // Auto-detect format so --trace works with the broader set of
    // tokenizers (tiktoken / HF / SP / .ztm), not just .tiktoken.
    var loaded = loadPipelineAutoDetect(gpa, io, model_path) catch |err| {
        try out.print("encode: failed to load {s}: {s}\n", .{ model_path, @errorName(err) });
        std.process.exit(2);
    };
    defer loaded.deinit();

    // `--optimal`: switch the BPE model to the dynamic-programming
    // minimum-token segmentation. Only meaningful for BPE models; a
    // no-op (with a notice) for other model kinds.
    if (args.optimal) {
        switch (loaded) {
            .bpe => |*b| b.encode_mode = .optimal,
            else => try out.writeAll("encode: --optimal only applies to BPE models; ignoring\n"),
        }
    }

    var v = ztok.Vocab.empty(gpa);
    defer v.deinit();

    // Trace sink (stderr). The id stream still goes to stdout via `out`.
    var stderr_buf: [4096]u8 = undefined;
    var stderr_file = std.Io.File.stderr();
    var stderr_w = stderr_file.writer(io, &stderr_buf);
    const stderr_iface = &stderr_w.interface;
    defer if (args.trace) stderr_iface.flush() catch {};
    var trace_sink: ztok.trace.Trace = .{ .writer = stderr_iface };

    const pipe: ztok.Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = if (args.cl100k) .cl100k else .identity,
        .model = loaded.modelValue(),
        .decoder = .concat,
        .vocab = &v,
        .trace = if (args.trace) &trace_sink else null,
    };

    const text = if (args.positional.items.len > 0)
        try std.mem.join(gpa, " ", args.positional.items)
    else
        try readStdin(gpa, io);
    defer gpa.free(text);

    const ids = try pipe.encode(gpa, text);
    defer gpa.free(ids);

    for (ids, 0..) |id, i| {
        if (i > 0) try out.writeAll(" ");
        try out.print("{d}", .{id});
    }
    try out.writeAll("\n");
}

/// `ztok encode-multimodal --model tekken.json --spec content.json`
///
/// content.json is a JSON array of interleaved content parts. Each part
/// is one of:
///   - a bare string                  → text part
///   - {"text": "..."}                → text part
///   - {"image": {"w": W, "h": H}}    → image part (pixel dims; "width"/
///                                       "height" also accepted)
///   - {"audio": {"samples": N}}      → audio part (raw waveform samples)
///   - {"audio": {"duration": S}}     → audio part (seconds → samples)
///   - {"audio": {"frames": F}}       → audio part (frame_rate frames →
///                                       seconds = F / frame_rate)
///
/// Emits the flattened space-separated token id stream on stdout, exactly
/// like `encode`.
fn cmdEncodeMultimodal(gpa: std.mem.Allocator, io: std.Io, raw: []const []const u8, out: *std.Io.Writer) !void {
    var args = try Args.parse(gpa, raw);
    defer args.deinit(gpa);

    const model_path = args.model_path orelse {
        try out.writeAll("encode-multimodal: --model PATH required\n");
        out.flush() catch {};
        std.process.exit(2);
    };
    const spec_path = args.spec_path orelse {
        try out.writeAll("encode-multimodal: --spec PATH required\n");
        out.flush() catch {};
        std.process.exit(2);
    };

    var tk = ztok.tekken.loadTekkenFile(gpa, model_path) catch |err| {
        try out.print("encode-multimodal: failed to load {s}: {s}\n", .{ model_path, @errorName(err) });
        out.flush() catch {};
        std.process.exit(2);
    };
    defer tk.deinit();

    const spec_bytes = std.Io.Dir.cwd().readFileAlloc(io, spec_path, gpa, .unlimited) catch |err| {
        try out.print("encode-multimodal: failed to read {s}: {s}\n", .{ spec_path, @errorName(err) });
        out.flush() catch {};
        std.process.exit(2);
    };
    defer gpa.free(spec_bytes);

    var parsed = std.json.parseFromSlice(std.json.Value, gpa, spec_bytes, .{}) catch |err| {
        try out.print("encode-multimodal: invalid JSON in {s}: {s}\n", .{ spec_path, @errorName(err) });
        out.flush() catch {};
        std.process.exit(2);
    };
    defer parsed.deinit();

    if (parsed.value != .array) {
        try out.writeAll("encode-multimodal: spec must be a JSON array of content parts\n");
        out.flush() catch {};
        std.process.exit(2);
    }

    var parts: std.ArrayList(ztok.tekken.ContentPart) = .empty;
    defer parts.deinit(gpa);

    const frame_rate: ?f32 = if (tk.audio_config) |ac| ac.frame_rate else null;

    for (parsed.value.array.items, 0..) |item, idx| {
        const part = parseContentPart(item, frame_rate) catch {
            try out.print("encode-multimodal: malformed content part at index {d}\n", .{idx});
            out.flush() catch {};
            std.process.exit(2);
        };
        try parts.append(gpa, part);
    }

    const ids = ztok.tekken.encodeMultimodal(gpa, &tk, parts.items) catch |err| {
        try out.print("encode-multimodal: encode failed: {s}\n", .{@errorName(err)});
        out.flush() catch {};
        std.process.exit(2);
    };
    defer gpa.free(ids);

    for (ids, 0..) |id, i| {
        if (i > 0) try out.writeAll(" ");
        try out.print("{d}", .{id});
    }
    try out.writeAll("\n");
}

/// Parse a single content-part JSON value into a `ContentPart`.
/// `frame_rate` is the model's audio frame rate (if it has one), used to
/// convert a `"frames"` audio spec to a duration. Errors with
/// `error.MalformedPart` on anything it can't interpret.
fn parseContentPart(v: std.json.Value, frame_rate: ?f32) !ztok.tekken.ContentPart {
    // Bare string → text.
    if (v == .string) return .{ .text = v.string };
    if (v != .object) return error.MalformedPart;
    const obj = v.object;

    if (obj.get("text")) |tv| {
        if (tv != .string) return error.MalformedPart;
        return .{ .text = tv.string };
    }

    if (obj.get("image")) |iv| {
        if (iv != .object) return error.MalformedPart;
        const w = jsonDimField(iv, "w", "width") orelse return error.MalformedPart;
        const h = jsonDimField(iv, "h", "height") orelse return error.MalformedPart;
        return .{ .image = .{ .width = w, .height = h } };
    }

    if (obj.get("audio")) |av| {
        if (av != .object) return error.MalformedPart;
        if (jsonU64Field(av, "samples")) |n| {
            return .{ .audio = .{ .num_samples = n } };
        }
        if (jsonF64Field(av, "duration")) |d| {
            return .{ .audio = .{ .duration_s = d } };
        }
        if (jsonF64Field(av, "frames")) |f| {
            // frames at the model frame_rate → seconds.
            const fr = frame_rate orelse return error.MalformedPart;
            if (fr <= 0) return error.MalformedPart;
            return .{ .audio = .{ .duration_s = f / @as(f64, fr) } };
        }
        // Empty audio object → zero-length clip (just [BEGIN_AUDIO]).
        return .{ .audio = .{} };
    }

    return error.MalformedPart;
}

fn jsonDimField(v: std.json.Value, key_a: []const u8, key_b: []const u8) ?u32 {
    const got = v.object.get(key_a) orelse v.object.get(key_b) orelse return null;
    return switch (got) {
        .integer => |i| if (i < 0 or i > std.math.maxInt(u32)) null else @intCast(i),
        else => null,
    };
}

fn jsonU64Field(v: std.json.Value, key: []const u8) ?u64 {
    const got = v.object.get(key) orelse return null;
    return switch (got) {
        .integer => |i| if (i < 0) null else @intCast(i),
        else => null,
    };
}

fn jsonF64Field(v: std.json.Value, key: []const u8) ?f64 {
    const got = v.object.get(key) orelse return null;
    return switch (got) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => null,
    };
}

fn cmdDecode(gpa: std.mem.Allocator, raw: []const []const u8, out: *std.Io.Writer) !void {
    var args = try Args.parse(gpa, raw);
    defer args.deinit(gpa);

    const model_path = args.model_path orelse {
        try out.writeAll("decode: --model PATH required\n");
        return;
    };

    var bpe = try ztok.Bpe.loadTiktokenFile(gpa, model_path);
    defer bpe.deinit();

    var v = ztok.Vocab.empty(gpa);
    defer v.deinit();

    const pipe: ztok.Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .{ .bpe = &bpe },
        .decoder = .concat,
        .vocab = &v,
    };

    const ids = try gpa.alloc(ztok.TokenId, args.positional.items.len);
    defer gpa.free(ids);
    for (args.positional.items, 0..) |s, i| {
        ids[i] = try std.fmt.parseInt(ztok.TokenId, s, 10);
    }

    const bytes = try pipe.decode(gpa, ids);
    defer gpa.free(bytes);
    try out.writeAll(bytes);
    try out.writeAll("\n");
}

fn cmdInfo(gpa: std.mem.Allocator, raw: []const []const u8, out: *std.Io.Writer) !void {
    var args = try Args.parse(gpa, raw);
    defer args.deinit(gpa);

    const model_path = args.model_path orelse {
        try out.writeAll("info: --model PATH required\n");
        return;
    };

    var bpe = try ztok.Bpe.loadTiktokenFile(gpa, model_path);
    defer bpe.deinit();

    try out.print("model:       {s}\n", .{model_path});
    try out.print("format:      .tiktoken\n", .{});
    try out.print("vocab_size:  {d}\n", .{bpe.count});
    try out.print("byte_count:  {d}\n", .{bpe.bytes.len});
}

fn cmdExplain(gpa: std.mem.Allocator, io: std.Io, raw: []const []const u8, out: *std.Io.Writer) !void {
    var args = try Args.parse(gpa, raw);
    defer args.deinit(gpa);

    const model_path = args.model_path orelse {
        try out.writeAll("explain: --model PATH required\n");
        return;
    };

    const fmt: ztok.explain.Format = blk: {
        const f = args.format orelse break :blk .text;
        if (std.mem.eql(u8, f, "text")) break :blk .text;
        if (std.mem.eql(u8, f, "json")) break :blk .json;
        try out.print("explain: unknown --format '{s}' (expected text|json)\n", .{f});
        std.process.exit(2);
    };

    var loaded = loadPipelineAutoDetect(gpa, io, model_path) catch |err| {
        try out.print("explain: failed to load {s}: {s}\n", .{ model_path, @errorName(err) });
        std.process.exit(2);
    };
    defer loaded.deinit();

    var v = ztok.Vocab.empty(gpa);
    defer v.deinit();

    const pipe: ztok.Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = if (args.cl100k) .cl100k else .identity,
        .model = loaded.modelValue(),
        .decoder = .concat,
        .vocab = &v,
    };

    const text = if (args.positional.items.len > 0)
        try std.mem.join(gpa, " ", args.positional.items)
    else
        try readStdin(gpa, io);
    defer gpa.free(text);

    var exp = try ztok.explain.explain(gpa, &pipe, text);
    defer exp.deinit();

    try ztok.explain.render(&exp, fmt, out);
    if (fmt == .json) try out.writeAll("\n");
}

fn loadBpeFromPath(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !ztok.Bpe {
    _ = io;
    return ztok.Bpe.loadTiktokenFile(gpa, path);
}

const TrainKind = enum { bpe, unigram, wordpiece, monster, pathpiece };

fn parseTrainKind(s: []const u8) ?TrainKind {
    if (std.mem.eql(u8, s, "bpe")) return .bpe;
    if (std.mem.eql(u8, s, "unigram")) return .unigram;
    if (std.mem.eql(u8, s, "wordpiece")) return .wordpiece;
    if (std.mem.eql(u8, s, "monster")) return .monster;
    if (std.mem.eql(u8, s, "pathpiece")) return .pathpiece;
    return null;
}

fn cmdTrain(gpa: std.mem.Allocator, io: std.Io, raw: []const []const u8, out: *std.Io.Writer) !void {
    var args = try Args.parse(gpa, raw);
    defer args.deinit(gpa);

    const input_path = args.input_path orelse {
        try out.writeAll("train: --input PATH required\n");
        return;
    };
    const output_path = args.output_path orelse {
        try out.writeAll("train: --output PATH required\n");
        return;
    };
    const vocab_size = args.vocab_size orelse {
        try out.writeAll("train: --vocab-size N required\n");
        return;
    };

    const kind: TrainKind = if (args.kind) |k| (parseTrainKind(k) orelse {
        try out.print("train: unknown --kind '{s}' (expected bpe|unigram|wordpiece|monster|pathpiece)\n", .{k});
        return;
    }) else .bpe;

    const min_vsz: u32 = switch (kind) {
        .bpe, .wordpiece, .monster, .pathpiece => 256,
        .unigram => 257,
    };
    if (vocab_size < min_vsz) {
        try out.print("train: --vocab-size must be >= {d} for kind={s}\n", .{ min_vsz, @tagName(kind) });
        return;
    }

    const raw_bytes = try std.Io.Dir.cwd().readFileAlloc(io, input_path, gpa, .unlimited);
    defer gpa.free(raw_bytes);

    var pool = if (args.threads) |n|
        try ztok.thread_pool.BatchPool.init(gpa, n)
    else
        try ztok.thread_pool.BatchPool.init(gpa, null);
    defer pool.deinit();

    // Optional avoid-list. Loaded once here, passed by pointer into
    // whichever trainer the user picked. Wordpiece training is not
    // wired through avoid (owned by a parallel agent); the flag is
    // silently a no-op there for now.
    var avoid_storage: ?ztok.negative_train.AvoidList = null;
    defer if (avoid_storage) |*al| al.deinit();
    var avoid_ptr: ?*const ztok.negative_train.AvoidList = null;
    var avoid_mode: ztok.negative_train.Mode = .penalize;
    if (args.avoid_path) |ap| {
        const av_bytes = std.Io.Dir.cwd().readFileAlloc(io, ap, gpa, .unlimited) catch |err| {
            try out.print("train: failed to read --avoid {s}: {s}\n", .{ ap, @errorName(err) });
            return;
        };
        defer gpa.free(av_bytes);
        avoid_storage = ztok.negative_train.parse(gpa, av_bytes) catch |err| {
            try out.print("train: failed to parse --avoid {s}: {s}\n", .{ ap, @errorName(err) });
            return;
        };
        avoid_ptr = &avoid_storage.?;
        if (args.avoid_mode) |m| {
            if (std.mem.eql(u8, m, "penalize")) {
                avoid_mode = .penalize;
            } else if (std.mem.eql(u8, m, "exclude")) {
                avoid_mode = .exclude;
            } else {
                try out.print("train: unknown --avoid-mode '{s}' (expected penalize|exclude)\n", .{m});
                return;
            }
        }
        try out.print("train: avoid={s} ({d} patterns, mode={s})\n", .{
            ap, avoid_storage.?.len(), @tagName(avoid_mode),
        });
    }

    try out.print("train: kind={s}, corpus={d} bytes, target vocab={d}, workers={d}, cl100k={}\n", .{
        @tagName(kind), raw_bytes.len, vocab_size, pool.workerCount(), args.cl100k,
    });
    try out.flush();

    const SplitFn = *const fn (std.mem.Allocator, []const u8) anyerror![]ztok.Span;
    const split_fn: SplitFn = if (args.cl100k) &ztok.cl100k.split else &identitySplit;

    switch (kind) {
        .bpe => {
            var bpe = try ztok.train_bpe.trainFromBytes(gpa, raw_bytes, split_fn, .{
                .vocab_size = vocab_size,
                .pool = &pool,
                .avoid = avoid_ptr,
                .avoid_mode = avoid_mode,
            });
            defer bpe.deinit();
            try writeTiktokenFile(gpa, io, output_path, &bpe);
            try out.print("train: wrote {s} ({d} tokens, {d} bytes total)\n", .{
                output_path, bpe.count, bpe.bytes.len,
            });
        },
        .unigram => {
            // train_unigram has no trainFromBytes today; build the
            // word/count corpus inline using the same shape the other
            // train_* modules use.
            const spans = try split_fn(gpa, raw_bytes);
            defer gpa.free(spans);
            var freq: std.StringHashMap(u32) = .init(gpa);
            defer freq.deinit();
            for (spans) |sp| {
                const w = sp.slice(raw_bytes);
                if (w.len == 0) continue;
                const gop = try freq.getOrPut(w);
                if (!gop.found_existing) gop.value_ptr.* = 0;
                gop.value_ptr.* +|= 1;
            }
            const n = freq.count();
            const words = try gpa.alloc([]const u8, n);
            defer gpa.free(words);
            const counts = try gpa.alloc(u32, n);
            defer gpa.free(counts);
            var it = freq.iterator();
            var i: usize = 0;
            while (it.next()) |e| : (i += 1) {
                words[i] = e.key_ptr.*;
                counts[i] = e.value_ptr.*;
            }
            var u_opts: ztok.train_unigram.TrainOptions = .{
                .vocab_size = vocab_size,
                .pool = &pool,
                .avoid = avoid_ptr,
                .avoid_mode = avoid_mode,
            };
            if (args.em_iterations) |it_n| u_opts.em_iters_per_round = it_n;
            if (args.shrink_rate) |sr| u_opts.prune_fraction = sr;
            var u = try ztok.train_unigram.train(gpa, .{ .words = words, .counts = counts }, u_opts);
            defer u.deinit();
            try ztok.sp_writer.writeUnigramFile(gpa, &u, output_path, .{
                .unk_id = u.unk_id,
            });
            try out.print("train: wrote {s} ({d} tokens, unigram .model)\n", .{
                output_path, u.count,
            });
        },
        .wordpiece => {
            const wp_opts: ztok.train_wordpiece.TrainOptions = .{
                .vocab_size = vocab_size,
                .pool = &pool,
            };
            // (em_iterations / shrink_rate / branches not used here.)
            var wp = try ztok.train_wordpiece.trainFromBytes(gpa, raw_bytes, split_fn, wp_opts);
            defer wp.deinit();
            try ztok.hf_writer.writeWordPieceFile(gpa, &wp, output_path, .{});
            try out.print("train: wrote {s} ({d} tokens, HF tokenizer.json)\n", .{
                output_path, wp.count,
            });
        },
        .monster => {
            const m_opts: ztok.train_monster.TrainOptions = .{
                .vocab_size = vocab_size,
                .pool = &pool,
                .avoid = avoid_ptr,
                .avoid_mode = avoid_mode,
            };
            _ = args.branches; // reserved — TrainOptions does not expose a branch knob yet
            var m = try ztok.train_monster.trainFromBytes(gpa, raw_bytes, split_fn, m_opts);
            defer m.deinit();
            try ztok.monster_io.writeFile(gpa, &m, output_path);
            try out.print("train: wrote {s} ({d} tokens, ztok .ztm)\n", .{
                output_path, m.count,
            });
        },
        .pathpiece => {
            // Same word/count harvest as unigram; PathPiece's learner
            // is corpus-driven and shares the Corpus shape.
            const spans = try split_fn(gpa, raw_bytes);
            defer gpa.free(spans);
            var freq: std.StringHashMap(u32) = .init(gpa);
            defer freq.deinit();
            for (spans) |sp| {
                const w = sp.slice(raw_bytes);
                if (w.len == 0) continue;
                const gop = try freq.getOrPut(w);
                if (!gop.found_existing) gop.value_ptr.* = 0;
                gop.value_ptr.* +|= 1;
            }
            const n = freq.count();
            const words = try gpa.alloc([]const u8, n);
            defer gpa.free(words);
            const counts = try gpa.alloc(u32, n);
            defer gpa.free(counts);
            var it = freq.iterator();
            var i: usize = 0;
            while (it.next()) |e| : (i += 1) {
                words[i] = e.key_ptr.*;
                counts[i] = e.value_ptr.*;
            }
            var r = try ztok.train_pathpiece.train(gpa, .{ .words = words, .counts = counts }, .{
                .vocab_size = vocab_size,
            });
            defer r.deinit();
            try writePathpieceFile(io, output_path, &r);
            try out.print("train: wrote {s} ({d} tokens, .tiktoken — load as bpe, run with --optimal)\n", .{
                output_path, r.count(),
            });
        },
    }
}

fn writePathpieceFile(io: std.Io, path: []const u8, r: *const ztok.train_pathpiece.Result) !void {
    var file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
    defer file.close(io);

    var buf: [4096]u8 = undefined;
    var fw = file.writer(io, &buf);
    const w = &fw.interface;
    defer w.flush() catch {};

    const b64 = std.base64.standard.Encoder;
    var enc_buf: [4096]u8 = undefined;
    var id: u32 = 0;
    while (id < r.count()) : (id += 1) {
        const bytes = r.piece(id);
        std.debug.assert(b64.calcSize(bytes.len) <= enc_buf.len);
        const encoded = b64.encode(&enc_buf, bytes);
        try w.print("{s} {d}\n", .{ encoded, id });
    }
}

fn cmdTokenizeDataset(gpa: std.mem.Allocator, io: std.Io, raw: []const []const u8, out: *std.Io.Writer) !void {
    var args = try Args.parse(gpa, raw);
    defer args.deinit(gpa);

    const model_path = args.model_path orelse {
        try out.writeAll("tokenize-dataset: --model PATH required\n");
        return;
    };
    const input_path = args.input_path orelse {
        try out.writeAll("tokenize-dataset: --input PATH required\n");
        return;
    };
    const output_path = args.output_path orelse {
        try out.writeAll("tokenize-dataset: --output PATH required\n");
        return;
    };

    const fmt: ztok.tokenize_dataset.Format = blk: {
        const f = args.format orelse "bin";
        if (std.mem.eql(u8, f, "bin")) break :blk .bin;
        if (std.mem.eql(u8, f, "npy")) break :blk .npy;
        try out.print("tokenize-dataset: unknown --format '{s}' (expected bin|npy)\n", .{f});
        return;
    };

    const doc_per_line: bool = blk: {
        const m = args.doc_mode orelse "lines";
        if (std.mem.eql(u8, m, "lines")) break :blk true;
        if (std.mem.eql(u8, m, "whole")) break :blk false;
        try out.print("tokenize-dataset: unknown --doc-mode '{s}' (expected lines|whole)\n", .{m});
        return;
    };

    var loaded = loadPipelineAutoDetect(gpa, io, model_path) catch |err| {
        try out.print("tokenize-dataset: failed to load {s}: {s}\n", .{ model_path, @errorName(err) });
        std.process.exit(2);
    };
    defer loaded.deinit();

    const dtype_bytes: u8 = blk: {
        const d = args.dtype orelse "auto";
        if (std.mem.eql(u8, d, "auto")) break :blk ztok.tokenize_dataset.pickDtypeBytes(loaded.vocabSize());
        if (std.mem.eql(u8, d, "u16")) break :blk 2;
        if (std.mem.eql(u8, d, "u32")) break :blk 4;
        try out.print("tokenize-dataset: unknown --dtype '{s}' (expected auto|u16|u32)\n", .{d});
        return;
    };

    var v = ztok.Vocab.empty(gpa);
    defer v.deinit();
    const pipe: ztok.Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = if (args.cl100k) .cl100k else .identity,
        .model = loaded.modelValue(),
        .decoder = .concat,
        .vocab = &v,
    };

    const opts: ztok.tokenize_dataset.Options = .{
        .seq_len = args.seq_len orelse 2048,
        .format = fmt,
        .dtype_bytes = dtype_bytes,
        .add_bos = args.add_bos,
        .bos_id = args.bos_id orelse 0,
        .add_eos = args.add_eos,
        .eos_id = args.eos_id orelse 0,
        .pad_last = args.pad_last,
        .pad_id = args.pad_id orelse 0,
        .doc_per_line = doc_per_line,
    };

    const input = std.Io.Dir.cwd().readFileAlloc(io, input_path, gpa, .unlimited) catch |err| {
        try out.print("tokenize-dataset: failed to read {s}: {s}\n", .{ input_path, @errorName(err) });
        std.process.exit(2);
    };
    defer gpa.free(input);

    var ids: std.ArrayList(ztok.TokenId) = .empty;
    defer ids.deinit(gpa);
    const docs = try ztok.tokenize_dataset.tokenize(gpa, &pipe, input, opts, &ids);

    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(gpa);
    var sequences: usize = 0;
    switch (fmt) {
        .bin => try ztok.tokenize_dataset.serializeBin(gpa, ids.items, dtype_bytes, &bytes),
        .npy => sequences = try ztok.tokenize_dataset.serializeNpy(gpa, ids.items, opts.seq_len, dtype_bytes, opts.pad_last, opts.pad_id, &bytes),
    }

    {
        var file = try std.Io.Dir.cwd().createFile(io, output_path, .{ .truncate = true });
        defer file.close(io);
        var wbuf: [64 * 1024]u8 = undefined;
        var fw = file.writer(io, &wbuf);
        const w = &fw.interface;
        defer w.flush() catch {};
        try w.writeAll(bytes.items);
    }

    const dtype_name = if (dtype_bytes == 2) "uint16" else "uint32";
    switch (fmt) {
        .bin => try out.print(
            "tokenize-dataset: wrote {s} — {d} docs, {d} tokens, flat {s} .bin ({d} bytes)\n",
            .{ output_path, docs, ids.items.len, dtype_name, bytes.items.len },
        ),
        .npy => try out.print(
            "tokenize-dataset: wrote {s} — {d} docs, {d} tokens -> {d} sequences × {d} ({s}) .npy\n",
            .{ output_path, docs, ids.items.len, sequences, opts.seq_len, dtype_name },
        ),
    }
}

fn identitySplit(allocator: std.mem.Allocator, input: []const u8) anyerror![]ztok.Span {
    const out = try allocator.alloc(ztok.Span, 1);
    out[0] = .{ .start = 0, .end = @intCast(input.len) };
    return out;
}

fn writeTiktokenFile(_: std.mem.Allocator, io: std.Io, path: []const u8, bpe: *const ztok.Bpe) !void {
    var file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
    defer file.close(io);

    var buf: [4096]u8 = undefined;
    var fw = file.writer(io, &buf);
    const w = &fw.interface;
    defer w.flush() catch {};

    const b64 = std.base64.standard.Encoder;
    var enc_buf: [4096]u8 = undefined;
    var id: u32 = 0;
    while (id < bpe.count) : (id += 1) {
        const bytes = bpe.idBytes(id);
        std.debug.assert(b64.calcSize(bytes.len) <= enc_buf.len);
        const encoded = b64.encode(&enc_buf, bytes);
        try w.print("{s} {d}\n", .{ encoded, id });
    }
}

fn parseBoundary(s: []const u8) ?ztok.chunk.Boundary {
    if (std.mem.eql(u8, s, "token")) return .token;
    if (std.mem.eql(u8, s, "codepoint")) return .codepoint;
    if (std.mem.eql(u8, s, "word")) return .word;
    if (std.mem.eql(u8, s, "sentence")) return .sentence;
    if (std.mem.eql(u8, s, "paragraph")) return .paragraph;
    return null;
}

fn cmdChunk(gpa: std.mem.Allocator, io: std.Io, raw: []const []const u8, out: *std.Io.Writer) !void {
    var args = try Args.parse(gpa, raw);
    defer args.deinit(gpa);

    const model_path = args.model_path orelse {
        try out.writeAll("chunk: --model PATH required\n");
        return;
    };
    const max_tokens = args.max_tokens orelse {
        try out.writeAll("chunk: --max-tokens N required\n");
        return;
    };
    const overlap = args.overlap orelse 0;
    const boundary: ztok.chunk.Boundary = if (args.boundary) |b| (parseBoundary(b) orelse {
        try out.print("chunk: unknown boundary mode '{s}'\n", .{b});
        return;
    }) else .token;
    const format = args.format orelse "jsonl";
    const is_jsonl = std.mem.eql(u8, format, "jsonl");
    if (!is_jsonl and !std.mem.eql(u8, format, "text")) {
        try out.print("chunk: unknown --format '{s}' (expected jsonl|text)\n", .{format});
        return;
    }

    var bpe = try loadBpeFromPath(gpa, io, model_path);
    defer bpe.deinit();

    var v = ztok.Vocab.empty(gpa);
    defer v.deinit();

    const pipe: ztok.Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = if (args.cl100k) .cl100k else .identity,
        .model = .{ .bpe = &bpe },
        .decoder = .concat,
        .vocab = &v,
    };

    const text = if (args.positional.items.len > 0)
        try std.mem.join(gpa, " ", args.positional.items)
    else
        try readStdin(gpa, io);
    defer gpa.free(text);

    var result = try ztok.chunk.chunkText(gpa, pipe, text, .{
        .max_tokens = max_tokens,
        .overlap_tokens = overlap,
        .boundary = boundary,
    });
    defer result.deinit();

    if (is_jsonl) {
        try emitJsonl(out, result.chunks);
    } else {
        try emitText(out, result.chunks);
    }
}

fn emitJsonl(out: *std.Io.Writer, chunks: []const ztok.chunk.Chunk) !void {
    for (chunks, 0..) |c, idx| {
        try out.print(
            "{{\"chunk\":{d},\"byte_start\":{d},\"byte_end\":{d},\"token_start\":{d},\"token_end\":{d},\"ids\":[",
            .{ idx, c.byte_start, c.byte_end, c.token_start, c.token_end },
        );
        for (c.ids, 0..) |id, j| {
            if (j > 0) try out.writeAll(",");
            try out.print("{d}", .{id});
        }
        try out.writeAll("]}\n");
    }
}

fn emitText(out: *std.Io.Writer, chunks: []const ztok.chunk.Chunk) !void {
    for (chunks, 0..) |c, idx| {
        try out.print(
            "--- chunk {d} (bytes {d}..{d}, tokens {d}..{d}) ---\n",
            .{ idx, c.byte_start, c.byte_end, c.token_start, c.token_end },
        );
        for (c.ids, 0..) |id, j| {
            if (j > 0) try out.writeAll(" ");
            try out.print("{d}", .{id});
        }
        try out.writeAll("\n");
    }
}

// --- diff ----------------------------------------------------------------

fn cmdDiff(gpa: std.mem.Allocator, io: std.Io, raw: []const []const u8, out: *std.Io.Writer) !void {
    var args = try Args.parse(gpa, raw);
    defer args.deinit(gpa);

    if (args.report) {
        try cmdDiffReport(gpa, io, &args, out);
        return;
    }

    const a_path = args.a_path orelse {
        try out.writeAll("diff: --a PATH required\n");
        std.process.exit(2);
    };
    const b_path = args.b_path orelse {
        try out.writeAll("diff: --b PATH required\n");
        std.process.exit(2);
    };

    var fmt: ztok.cli_diff.Format = .text;
    if (args.format) |f| {
        if (std.mem.eql(u8, f, "text")) {
            fmt = .text;
        } else if (std.mem.eql(u8, f, "json")) {
            fmt = .json;
        } else {
            try out.print("diff: unknown --format '{s}' (expected text|json)\n", .{f});
            std.process.exit(2);
        }
    }

    var loaded_a = loadPipelineAutoDetect(gpa, io, a_path) catch |err| {
        try out.print("diff: failed to load --a {s}: {s}\n", .{ a_path, @errorName(err) });
        std.process.exit(2);
    };
    defer loaded_a.deinit();
    var loaded_b = loadPipelineAutoDetect(gpa, io, b_path) catch |err| {
        try out.print("diff: failed to load --b {s}: {s}\n", .{ b_path, @errorName(err) });
        std.process.exit(2);
    };
    defer loaded_b.deinit();

    var vocab_a = ztok.Vocab.empty(gpa);
    defer vocab_a.deinit();
    var vocab_b = ztok.Vocab.empty(gpa);
    defer vocab_b.deinit();
    const pipe_a: ztok.Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = if (args.cl100k) .cl100k else .identity,
        .model = loaded_a.modelValue(),
        .decoder = .concat,
        .vocab = &vocab_a,
    };
    const pipe_b: ztok.Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = if (args.cl100k) .cl100k else .identity,
        .model = loaded_b.modelValue(),
        .decoder = .concat,
        .vocab = &vocab_b,
    };

    const input_owned: []u8 = if (args.stdin or args.positional.items.len == 0)
        try readStdin(gpa, io)
    else
        try std.Io.Dir.cwd().readFileAlloc(io, args.positional.items[0], gpa, .unlimited);
    defer gpa.free(input_owned);

    const result = try ztok.cli_diff.runDiff(gpa, &pipe_a, &pipe_b, input_owned, .{
        .format = fmt,
    }, out);
    try out.flush();
    if (!result.allMatch()) std.process.exit(1);
}

/// `ztok diff --report` implementation. Loads both vocabs, encodes the
/// corpus through each, and renders a Markdown report. Output destination
/// is `--out PATH` if given, otherwise the inherited `out` writer
/// (stdout). Keeps the existing `runDiff` path untouched.
fn cmdDiffReport(gpa: std.mem.Allocator, io: std.Io, args: *const Args, out: *std.Io.Writer) !void {
    const va_path = args.vocab_a_path orelse {
        try out.writeAll("diff --report: --vocab-a PATH required\n");
        std.process.exit(2);
    };
    const vb_path = args.vocab_b_path orelse {
        try out.writeAll("diff --report: --vocab-b PATH required\n");
        std.process.exit(2);
    };
    const corpus_path = args.corpus_path orelse {
        try out.writeAll("diff --report: --corpus PATH required\n");
        std.process.exit(2);
    };

    var loaded_a = loadPipelineAutoDetect(gpa, io, va_path) catch |err| {
        try out.print("diff --report: failed to load --vocab-a {s}: {s}\n", .{ va_path, @errorName(err) });
        std.process.exit(2);
    };
    defer loaded_a.deinit();
    var loaded_b = loadPipelineAutoDetect(gpa, io, vb_path) catch |err| {
        try out.print("diff --report: failed to load --vocab-b {s}: {s}\n", .{ vb_path, @errorName(err) });
        std.process.exit(2);
    };
    defer loaded_b.deinit();

    var vocab_a = ztok.Vocab.empty(gpa);
    defer vocab_a.deinit();
    var vocab_b = ztok.Vocab.empty(gpa);
    defer vocab_b.deinit();
    const pipe_a: ztok.Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = if (args.cl100k) .cl100k else .identity,
        .model = loaded_a.modelValue(),
        .decoder = .concat,
        .vocab = &vocab_a,
    };
    const pipe_b: ztok.Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = if (args.cl100k) .cl100k else .identity,
        .model = loaded_b.modelValue(),
        .decoder = .concat,
        .vocab = &vocab_b,
    };

    const corpus_bytes = std.Io.Dir.cwd().readFileAlloc(io, corpus_path, gpa, .unlimited) catch |err| {
        try out.print("diff --report: failed to read --corpus {s}: {s}\n", .{ corpus_path, @errorName(err) });
        std.process.exit(2);
    };
    defer gpa.free(corpus_bytes);

    const meta_a: ztok.diff.VocabMeta = .{
        .path = va_path,
        .kind = loaded_a.kind(),
        .vocab_size = loaded_a.vocabSize(),
    };
    const meta_b: ztok.diff.VocabMeta = .{
        .path = vb_path,
        .kind = loaded_b.kind(),
        .vocab_size = loaded_b.vocabSize(),
    };

    // Write either to --out PATH or to the inherited stdout writer.
    if (args.out_path) |path| {
        var file = std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true }) catch |err| {
            try out.print("diff --report: failed to open --out {s}: {s}\n", .{ path, @errorName(err) });
            std.process.exit(2);
        };
        defer file.close(io);

        var fbuf: [4096]u8 = undefined;
        var fw = file.writer(io, &fbuf);
        const w = &fw.interface;
        defer w.flush() catch {};

        try ztok.cli_diff.runReport(gpa, &pipe_a, &pipe_b, meta_a, meta_b, corpus_bytes, .{}, w);
    } else {
        try ztok.cli_diff.runReport(gpa, &pipe_a, &pipe_b, meta_a, meta_b, corpus_bytes, .{}, out);
        try out.flush();
    }
}

// --- eval ----------------------------------------------------------------

fn cmdEval(gpa: std.mem.Allocator, io: std.Io, raw: []const []const u8, out: *std.Io.Writer) !void {
    var args = try Args.parse(gpa, raw);
    defer args.deinit(gpa);

    const model_path = args.model_path orelse {
        try out.writeAll("eval: --model PATH required\n");
        std.process.exit(2);
    };

    var fmt: ztok.cli_eval.Format = .text;
    if (args.format) |f| {
        if (std.mem.eql(u8, f, "text")) {
            fmt = .text;
        } else if (std.mem.eql(u8, f, "json")) {
            fmt = .json;
        } else {
            try out.print("eval: unknown --format '{s}' (expected text|json)\n", .{f});
            std.process.exit(2);
        }
    }

    var loaded = loadPipelineAutoDetect(gpa, io, model_path) catch |err| {
        try out.print("eval: failed to load {s}: {s}\n", .{ model_path, @errorName(err) });
        std.process.exit(2);
    };
    defer loaded.deinit();

    var vocab = ztok.Vocab.empty(gpa);
    defer vocab.deinit();
    const pipe: ztok.Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = if (args.cl100k) .cl100k else .identity,
        .model = loaded.modelValue(),
        .decoder = .concat,
        .vocab = &vocab,
    };

    // Source priority for input text: --corpus > positional > --stdin / empty stdin.
    const input_owned: []u8 = if (args.corpus_path) |cp|
        try std.Io.Dir.cwd().readFileAlloc(io, cp, gpa, .unlimited)
    else if (args.stdin or args.positional.items.len == 0)
        try readStdin(gpa, io)
    else
        try std.Io.Dir.cwd().readFileAlloc(io, args.positional.items[0], gpa, .unlimited);
    defer gpa.free(input_owned);

    // --per: case-insensitive 1k / 1m. Default per_1m when --price is set
    // without --per.
    var price_unit: ztok.eval.PriceUnit = .per_1m;
    if (args.price_per) |p| {
        if (eqIgnoreCase(p, "1k")) {
            price_unit = .per_1k;
        } else if (eqIgnoreCase(p, "1m")) {
            price_unit = .per_1m;
        } else {
            try out.print("eval: unknown --per '{s}' (expected 1k|1M)\n", .{p});
            std.process.exit(2);
        }
    }

    // Resolve prices: --prices-file + --model-name wins, then explicit
    // --price-input/--price-output, then --price (treated as input price).
    var resolved_input: ?f64 = args.price_input orelse args.price;
    var resolved_output: ?f64 = args.price_output;
    if (args.prices_file) |pf_path| {
        const model_name = args.model_name orelse {
            try out.writeAll("eval: --prices-file requires --model-name NAME\n");
            std.process.exit(2);
        };
        const pf_bytes = std.Io.Dir.cwd().readFileAlloc(io, pf_path, gpa, .unlimited) catch |err| {
            try out.print("eval: failed to read --prices-file {s}: {s}\n", .{ pf_path, @errorName(err) });
            std.process.exit(2);
        };
        defer gpa.free(pf_bytes);
        const parsed = parsePricesFile(pf_bytes, model_name) catch |err| {
            try out.print("eval: failed to parse --prices-file: {s}\n", .{@errorName(err)});
            std.process.exit(2);
        };
        if (parsed) |p| {
            // Prices-file wins over earlier flags.
            if (p.input) |v| resolved_input = v;
            if (p.output) |v| resolved_output = v;
            price_unit = p.unit;
        } else {
            try out.print("eval: model '{s}' not found in {s}\n", .{ model_name, pf_path });
            std.process.exit(2);
        }
    }

    try ztok.cli_eval.runEval(gpa, &pipe, input_owned, .{
        .format = fmt,
        .top_k = args.top_k orelse 10,
        .bottom_k = args.bottom_k orelse 10,
        .hidden_dim = args.hidden_dim,
        .price_input = resolved_input,
        .price_output = resolved_output,
        .price_unit = price_unit,
    }, out);

    // --fairness: append a per-script multilingual fairness report after the
    // normal eval output. The CLI JSON object emitted above is already closed,
    // so in JSON mode the fairness payload is emitted as its own object line.
    if (args.fairness) {
        var report = try ztok.eval.evaluate(gpa, &pipe, input_owned, .{ .scripts = true });
        defer report.deinit();
        switch (fmt) {
            .text => try ztok.eval.writeFairnessText(gpa, out, &report),
            .json => try ztok.eval.writeFairnessJson(gpa, out, &report),
        }
    }

    try out.flush();
}

fn eqIgnoreCase(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    for (a, b) |x, y| {
        if (std.ascii.toLower(x) != std.ascii.toLower(y)) return false;
    }
    return true;
}

const PricesEntry = struct {
    input: ?f64 = null,
    output: ?f64 = null,
    unit: ztok.eval.PriceUnit = .per_1m,
};

/// Minimal TOML-ish parser tuned for the `--prices-file` schema:
///
///   [model-name]
///   input_per_1m = 2.50
///   output_per_1m = 10.00
///
/// Comments start with `#`. Section names are everything between `[` and
/// `]`. Keys are matched literally (case-sensitive). Returns the entry
/// for `wanted` or null if no matching section exists. All values in the
/// returned entry are normalized to per_1m — this keeps the rest of the
/// pipeline single-unit and matches every documented row in the spec.
fn parsePricesFile(bytes: []const u8, wanted: []const u8) !?PricesEntry {
    var entry: PricesEntry = .{};
    var found = false;
    var in_section = false;
    var it = std.mem.splitScalar(u8, bytes, '\n');
    while (it.next()) |raw_line| {
        const line = trimAscii(raw_line);
        if (line.len == 0) continue;
        if (line[0] == '#') continue;
        if (line[0] == '[') {
            const close = std.mem.indexOfScalar(u8, line, ']') orelse return error.MalformedSection;
            const name = trimAscii(line[1..close]);
            in_section = std.mem.eql(u8, name, wanted);
            if (in_section) found = true;
            continue;
        }
        if (!in_section) continue;

        const eq = std.mem.indexOfScalar(u8, line, '=') orelse return error.MissingEquals;
        const key = trimAscii(line[0..eq]);
        var val_raw = trimAscii(line[eq + 1 ..]);
        // Strip inline comment.
        if (std.mem.indexOfScalar(u8, val_raw, '#')) |hi| val_raw = trimAscii(val_raw[0..hi]);
        if (val_raw.len == 0) return error.MissingValue;

        const v = try std.fmt.parseFloat(f64, val_raw);
        if (std.mem.eql(u8, key, "input_per_1m")) {
            entry.input = v;
        } else if (std.mem.eql(u8, key, "output_per_1m")) {
            entry.output = v;
        }
        // Unknown keys silently ignored — keeps the parser forward-compatible.
    }
    if (!found) return null;
    return entry;
}

fn trimAscii(s: []const u8) []const u8 {
    return std.mem.trim(u8, s, " \t\r");
}

// --- serve --------------------------------------------------------------

fn cmdServe(gpa: std.mem.Allocator, io: std.Io, raw: []const []const u8, out: *std.Io.Writer) !void {
    var args = try Args.parse(gpa, raw);
    defer args.deinit(gpa);

    const model_path = args.model_path orelse {
        try out.writeAll("serve: --model PATH required\n");
        std.process.exit(2);
    };

    var loaded = loadPipelineAutoDetect(gpa, io, model_path) catch |err| {
        try out.print("serve: failed to load {s}: {s}\n", .{ model_path, @errorName(err) });
        std.process.exit(2);
    };
    defer loaded.deinit();

    var vocab = ztok.Vocab.empty(gpa);
    defer vocab.deinit();
    const pipeline: ztok.Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = if (args.cl100k) .cl100k else .identity,
        .model = loaded.modelValue(),
        .decoder = .concat,
        .vocab = &vocab,
    };

    var pool = if (args.workers) |n|
        try ztok.thread_pool.BatchPool.init(gpa, n)
    else
        try ztok.thread_pool.BatchPool.init(gpa, null);
    defer pool.deinit();

    const model_kind: ztok.cli_serve.ModelKind = switch (loaded) {
        .bpe => .bpe,
        .unigram => .unigram,
        .wordpiece => .wordpiece,
        .monster => .monster,
    };

    const host = args.host orelse ztok.cli_serve.default_host;
    const port = args.port orelse ztok.cli_serve.default_port;

    // Resolve --auth-token / --auth-token-file. The file path wins if
    // both are set; warn but don't error so scripts that pass both for
    // backstop reasons keep working.
    var auth_token_owned: ?[]u8 = null;
    defer if (auth_token_owned) |t| gpa.free(t);
    var auth_token: ?[]const u8 = args.auth_token;
    if (args.auth_token_file) |path| {
        if (args.auth_token != null) {
            try out.writeAll("serve: warning — both --auth-token and --auth-token-file set; using file\n");
        }
        const file_bytes = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited) catch |err| {
            try out.print("serve: failed to read auth-token file {s}: {s}\n", .{ path, @errorName(err) });
            std.process.exit(2);
        };
        // Strip trailing whitespace (newline from `echo`, etc.).
        const trimmed = std.mem.trimEnd(u8, file_bytes, " \t\r\n");
        if (trimmed.len == 0) {
            gpa.free(file_bytes);
            try out.print("serve: auth-token file {s} is empty\n", .{path});
            std.process.exit(2);
        }
        // Copy the trimmed slice so we can free `file_bytes`.
        const dup = try gpa.dupe(u8, trimmed);
        gpa.free(file_bytes);
        auth_token_owned = dup;
        auth_token = dup;
    }

    if (args.rate_limit) |rl| if (rl == 0) {
        try out.writeAll("serve: --rate-limit 0 is not allowed; omit the flag to disable\n");
        std.process.exit(2);
    };

    // Parse --log-format.
    var log_format: ztok.cli_serve.LogFormat = .text;
    if (args.log_format) |lf| {
        if (std.mem.eql(u8, lf, "text")) {
            log_format = .text;
        } else if (std.mem.eql(u8, lf, "json")) {
            log_format = .json;
        } else {
            try out.print("serve: unknown --log-format '{s}' (expected text|json)\n", .{lf});
            std.process.exit(2);
        }
    }

    // TLS requested? Routes through the build-time TLS backend
    // (-Dtls=mbedtls) inside cli_serve.zig. Default builds (-Dtls=none)
    // refuse to start with TLSServerNotAvailable rather than silently
    // serving cleartext.
    const tls_requested = args.tls_cert != null or args.tls_key != null;
    if (tls_requested) {
        try out.writeAll("serve: TLS termination requested. The default build (-Dtls=none) will refuse to start; rebuild with `zig build -Dtls=mbedtls` to enable.\n");
    }

    // OIDC startup. Best-effort: if the discovery / JWKS pull fails,
    // log a warning and proceed without OIDC (so a transient network
    // hiccup doesn't take the serve down). With both
    // --auth-oidc-issuer and --auth-oidc-audience set, we GET the
    // discovery doc + JWKS via `std.http.Client` (HTTPS-capable in
    // 0.16 stdlib) and build the validator. RS256 keys verify
    // through the new RS256 path; the JWKS refreshes lazily after a
    // 10-minute TTL or on kid miss.
    var http_client: ?std.http.Client = null;
    defer if (http_client) |*c| c.deinit();
    var oidc_validator: ?*ztok.auth_oidc.Validator = null;
    defer if (oidc_validator) |v| {
        v.deinit();
        gpa.destroy(v);
    };
    if (args.auth_oidc_issuer != null and args.auth_oidc_audience != null) {
        http_client = .{ .allocator = gpa, .io = io };
        oidc_validator = ztok.auth_oidc.initFromDiscovery(
            gpa,
            ztok.auth_oidc.httpFetchWithStdClient,
            @ptrCast(&http_client.?),
            args.auth_oidc_issuer.?,
            args.auth_oidc_audience.?,
        ) catch |err| blk: {
            try out.print("serve: warning — OIDC discovery/JWKS fetch failed: {s}; proceeding without OIDC. Use --auth-token to keep clients usable.\n", .{@errorName(err)});
            break :blk null;
        };
    }

    // Per-process metrics sink. Lives on the stack of cmdServe — the
    // server borrows a pointer for its lifetime.
    var metrics: ztok.cli_serve.Metrics = .{};

    // Persistent prefix cache (--prefix-cache-dir). When set, we
    // create the directory if missing and open the cache file inside
    // it. The server borrows the pointer for its lifetime.
    var prefix_cache: ?*ztok.prefix_cache.PrefixCache = null;
    var prefix_cache_path_owned: ?[]u8 = null;
    defer if (prefix_cache_path_owned) |p| gpa.free(p);
    defer if (prefix_cache) |pc| pc.closePersistent();
    if (args.prefix_cache_dir) |dir| {
        // mkdir -p so operators don't have to pre-create the directory.
        std.Io.Dir.cwd().createDirPath(io, dir) catch |err| switch (err) {
            error.PathAlreadyExists => {},
            else => {
                try out.print("serve: failed to create prefix-cache-dir {s}: {s}\n", .{ dir, @errorName(err) });
                std.process.exit(2);
            },
        };
        const cache_path = try std.fmt.allocPrint(gpa, "{s}/prefix_cache.dat", .{dir});
        prefix_cache_path_owned = cache_path;
        prefix_cache = ztok.prefix_cache.PrefixCache.openPersistent(gpa, .{
            .path = cache_path,
        }) catch |err| {
            try out.print("serve: failed to open prefix cache at {s}: {s}\n", .{ cache_path, @errorName(err) });
            std.process.exit(2);
        };
    }

    try out.print(
        "ztok serve: model={s} host={s} port={d} workers={d} model_kind={s} auth={s} rate_limit={s} metrics={s} log_format={s} tls={s} prefix_cache={s}\n",
        .{
            model_path,
            host,
            port,
            pool.workerCount(),
            @tagName(model_kind),
            if (auth_token != null) "bearer" else if (oidc_validator != null) "oidc" else "off",
            if (args.rate_limit) |_| "on" else "off",
            if (args.metrics) "on" else "off",
            @tagName(log_format),
            if (tls_requested) "requested-stub" else "off",
            if (args.prefix_cache_dir) |_| "on" else "off",
        },
    );
    try out.flush();

    try ztok.cli_serve.run(gpa, io, &pipeline, &pool, .{
        .host = host,
        .port = port,
        .version = VERSION,
        .model_kind = model_kind,
        .auth_token = auth_token,
        .rate_limit_rps = args.rate_limit,
        .log_format = log_format,
        .metrics = if (args.metrics) &metrics else null,
        .oidc = oidc_validator,
        .tls_enabled = tls_requested,
        .tls_cert_path = args.tls_cert,
        .tls_key_path = args.tls_key,
        .prefix_cache = prefix_cache,
    });
}

// --- grpc-serve --------------------------------------------------------

fn cmdGrpcServe(gpa: std.mem.Allocator, io: std.Io, raw: []const []const u8, out: *std.Io.Writer) !void {
    var args = try Args.parse(gpa, raw);
    defer args.deinit(gpa);

    const model_path = args.model_path orelse {
        try out.writeAll("grpc-serve: --model PATH required\n");
        std.process.exit(2);
    };

    var loaded = loadPipelineAutoDetect(gpa, io, model_path) catch |err| {
        try out.print("grpc-serve: failed to load {s}: {s}\n", .{ model_path, @errorName(err) });
        std.process.exit(2);
    };
    defer loaded.deinit();

    var vocab = ztok.Vocab.empty(gpa);
    defer vocab.deinit();
    const pipeline: ztok.Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = if (args.cl100k) .cl100k else .identity,
        .model = loaded.modelValue(),
        .decoder = .concat,
        .vocab = &vocab,
    };

    var pool = if (args.workers) |n|
        try ztok.thread_pool.BatchPool.init(gpa, n)
    else
        try ztok.thread_pool.BatchPool.init(gpa, null);
    defer pool.deinit();

    const model_kind: ztok.cli_grpc.ModelKind = switch (loaded) {
        .bpe => .bpe,
        .unigram => .unigram,
        .wordpiece => .wordpiece,
        .monster => .monster,
    };

    const host = args.host orelse ztok.cli_grpc.default_host;
    const port = args.port orelse ztok.cli_grpc.default_port;

    try out.print(
        "ztok grpc-serve: model={s} host={s} port={d} workers={d} model_kind={s}\n",
        .{ model_path, host, port, pool.workerCount(), @tagName(model_kind) },
    );
    try out.flush();

    try ztok.cli_grpc.run(gpa, io, &pipeline, &pool, .{
        .host = host,
        .port = port,
        .version = VERSION,
        .model_kind = model_kind,
    });
}

// --- bench --------------------------------------------------------------

fn cmdBench(gpa: std.mem.Allocator, raw: []const []const u8, out: *std.Io.Writer) !void {
    var args = try Args.parse(gpa, raw);
    defer args.deinit(gpa);

    var fmt: ztok.cli_bench.Format = .text;
    if (args.format) |f| {
        if (std.mem.eql(u8, f, "text")) {
            fmt = .text;
        } else if (std.mem.eql(u8, f, "json")) {
            fmt = .json;
        } else {
            try out.print("bench: unknown --format '{s}' (expected text|json)\n", .{f});
            std.process.exit(2);
        }
    }

    // Iters: --quick = 1; --iters N overrides; default 5.
    const iters: u32 = if (args.iters) |n| n else if (args.quick) @as(u32, 1) else @as(u32, 5);

    const vocab_root: []const u8 = args.vocab_root orelse "bench/vocabs";

    // Parse --include into a Scenario slice. Caller owns the slice;
    // freed at end of function.
    var include_slice: ?[]ztok.cli_bench.Scenario = null;
    defer if (include_slice) |s| gpa.free(s);
    if (args.include) |spec| {
        var list: std.ArrayList(ztok.cli_bench.Scenario) = .empty;
        errdefer list.deinit(gpa);
        var it = std.mem.splitScalar(u8, spec, ',');
        while (it.next()) |raw_name| {
            const name = std.mem.trim(u8, raw_name, " \t");
            if (name.len == 0) continue;
            const sc = ztok.cli_bench.Scenario.parse(name) orelse {
                try out.print(
                    "bench: unknown scenario '{s}' (expected one of: cl100k, sp-bpe, sp-unigram, tm, hf-bpe)\n",
                    .{name},
                );
                list.deinit(gpa);
                std.process.exit(2);
            };
            try list.append(gpa, sc);
        }
        include_slice = try list.toOwnedSlice(gpa);
    }

    var result = try ztok.cli_bench.runBench(gpa, .{
        .iters = iters,
        .include = if (include_slice) |s| s else null,
        .vocab_root = vocab_root,
        .format = fmt,
        .corpus_bytes = args.corpus_bytes orelse (1024 * 1024),
    }, out);
    defer result.deinit(gpa);
    try out.flush();
}

// --- fingerprint --------------------------------------------------------
//
// Compute the tokenizer fingerprint for a vocab and print
// `ztok:<64-hex>`. Auto-detects the on-disk format the same way every
// other model-consuming subcommand does.

fn cmdFingerprint(gpa: std.mem.Allocator, io: std.Io, raw: []const []const u8, out: *std.Io.Writer) !void {
    if (raw.len == 0) {
        try out.writeAll("fingerprint: vocab path required (usage: ztok fingerprint VOCAB)\n");
        std.process.exit(2);
    }
    const path = raw[0];

    var loaded = loadPipelineAutoDetect(gpa, io, path) catch |err| {
        try out.print("fingerprint: failed to load {s}: {s}\n", .{ path, @errorName(err) });
        std.process.exit(2);
    };
    defer loaded.deinit();

    // Synthesize a Vocab whose `count` reflects the loaded model so
    // the fingerprint's vocab_size byte is correct. The fingerprint
    // never reads `bytes` / `offsets` / `ranks`; only `count`.
    var vocab: ztok.Vocab = .{
        .allocator = gpa,
        .bytes = &.{},
        .offsets = &.{},
        .ranks = null,
        .count = loaded.vocabSize(),
    };

    const pipeline: ztok.Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = loaded.modelValue(),
        .decoder = .concat,
        .vocab = &vocab,
    };

    const fp = try ztok.fingerprint.computeFingerprint(&pipeline, gpa);
    const hex = ztok.fingerprint.formatHex(fp);
    try out.print("ztok:{s}\n", .{hex});
}

// --- visualize ----------------------------------------------------------

// Render a self-contained HTML page (token grid + byte heatmap + length
// histogram + BPE dendrogram + added-token table) for a given vocab.
// Optional `--corpus PATH` swaps in the user's corpus for the bundled
// mini-corpus; `--out PATH` overrides the default
// `<vocab-basename>-viz.html` in cwd; `--top-k N` overrides the default
// top-200 frequency grid.
fn cmdVisualize(gpa: std.mem.Allocator, io: std.Io, raw: []const []const u8, out: *std.Io.Writer) !void {
    var args = try Args.parse(gpa, raw);
    defer args.deinit(gpa);

    // Vocab path: first positional, or `--model PATH` as a convenience
    // alias to match how every other subcommand looks.
    const vocab_path: []const u8 = if (args.positional.items.len > 0)
        args.positional.items[0]
    else if (args.model_path) |m| m
    else {
        try out.writeAll("visualize: vocab path required (usage: ztok visualize VOCAB [--corpus PATH] [--out FILE] [--top-k N])\n");
        std.process.exit(2);
    };

    var loaded = loadPipelineAutoDetect(gpa, io, vocab_path) catch |err| {
        try out.print("visualize: failed to load {s}: {s}\n", .{ vocab_path, @errorName(err) });
        std.process.exit(2);
    };
    defer loaded.deinit();

    // Build a minimal pipeline so we can encode the corpus into ids.
    var vocab = ztok.Vocab.empty(gpa);
    defer vocab.deinit();
    const pipeline: ztok.Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = loaded.modelValue(),
        .decoder = .concat,
        .vocab = &vocab,
    };

    // Optional corpus override.
    var corpus_owned: ?[]u8 = null;
    defer if (corpus_owned) |c| gpa.free(c);
    const corpus_slice: ?[]const u8 = blk: {
        if (args.corpus_path) |p| {
            corpus_owned = std.Io.Dir.cwd().readFileAlloc(io, p, gpa, .unlimited) catch |err| {
                try out.print("visualize: failed to read corpus {s}: {s}\n", .{ p, @errorName(err) });
                std.process.exit(2);
            };
            break :blk corpus_owned.?;
        }
        break :blk null; // vocab_viz will fall back to the bundled MINI_CORPUS
    };

    // For HF tokenizers, surface the added_tokens table. The HF JSON
    // is re-parsed here (separately from `loadPipelineAutoDetect`)
    // because the auto-detect loader keeps only the model — the
    // added_tokens array doesn't survive the bridge call. The HF
    // holder must outlive `hf_added_owned`: every `content` slice
    // borrows into `HFTokenizer.allocator`-owned buffers freed by
    // `hf.deinit()`.
    var hf_holder: ?ztok.hf_json.HFTokenizer = null;
    defer if (hf_holder) |*h| h.deinit();
    var hf_added_owned: ?[]ztok.vocab_viz.AddedTokenView = null;
    defer if (hf_added_owned) |a| gpa.free(a);
    if (std.mem.endsWith(u8, vocab_path, ".json")) {
        if (ztok.hf_json.loadFromFile(gpa, vocab_path)) |hf_val| {
            hf_holder = hf_val;
            hf_added_owned = try ztok.vocab_viz.addedTokensFromHF(gpa, &hf_holder.?);
        } else |_| {}
    }

    const model_ref: ztok.vocab_viz.ModelRef = switch (loaded) {
        .bpe => |*b| .{ .bpe = b },
        .unigram => |*u| .{ .unigram = u },
        .wordpiece => |*w| .{ .wordpiece = w },
        .monster => |*m| .{ .monster = m },
    };

    // Determine output path: `--out` wins, else `<basename>-viz.html` in cwd.
    var out_path_buf: ?[]u8 = null;
    defer if (out_path_buf) |p| gpa.free(p);
    const out_path: []const u8 = if (args.out_path) |p| p else blk: {
        const base = std.fs.path.basename(vocab_path);
        const stem = std.fs.path.stem(base);
        out_path_buf = try std.fmt.allocPrint(gpa, "{s}-viz.html", .{stem});
        break :blk out_path_buf.?;
    };

    // Open output + render via a buffered Allocating writer so we can
    // report final size at the end (the renderer streams a few hundred
    // KB at most; keeping it all in memory is fine and lets us hand
    // either stdout or a file the same buffer).
    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();

    const top_k: u32 = args.top_k orelse 200;
    try ztok.vocab_viz.render(gpa, &aw.writer, model_ref, &pipeline, .{
        .title = vocab_path,
        .top_k = top_k,
        .corpus = corpus_slice,
        .added_tokens = if (hf_added_owned) |a| a else &.{},
    });

    const html = aw.written();
    var file = std.Io.Dir.cwd().createFile(io, out_path, .{ .truncate = true }) catch |err| {
        try out.print("visualize: failed to create {s}: {s}\n", .{ out_path, @errorName(err) });
        std.process.exit(2);
    };
    defer file.close(io);
    var file_buf: [4096]u8 = undefined;
    var file_w = file.writer(io, &file_buf);
    const fw = &file_w.interface;
    try fw.writeAll(html);
    try fw.flush();

    try out.print("visualize: wrote {s} ({d} bytes)\n", .{ out_path, html.len });
}

// --- validate -----------------------------------------------------------

fn cmdValidate(gpa: std.mem.Allocator, io: std.Io, raw: []const []const u8, out: *std.Io.Writer) !void {
    var args = try Args.parse(gpa, raw);
    defer args.deinit(gpa);

    const model_path = args.model_path orelse {
        try out.writeAll("validate: --model PATH required\n");
        std.process.exit(2);
    };

    var loaded = loadPipelineAutoDetect(gpa, io, model_path) catch |err| {
        try out.print("validate: failed to load {s}: {s}\n", .{ model_path, @errorName(err) });
        std.process.exit(2);
    };
    defer loaded.deinit();

    // Build a minimal pipeline so the roundtrip / whitespace / cl100k
    // checks have somewhere to run.
    var vocab = ztok.Vocab.empty(gpa);
    defer vocab.deinit();
    const pipeline: ztok.Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = if (args.cl100k) .cl100k else .identity,
        .model = loaded.modelValue(),
        .decoder = .concat,
        .vocab = &vocab,
    };

    // Decide which checks to run. Default: all on.
    var checks: ztok.doctor.Checks = .{};
    if (args.checks) |spec| {
        var unknown: ?[]const u8 = null;
        const parsed = ztok.cli_validate.parseChecks(spec, &unknown);
        if (parsed == null) {
            try out.print("validate: unknown check name '{s}'\n", .{unknown.?});
            std.process.exit(2);
        }
        checks = parsed.?;
    }

    // Format.
    var fmt: ztok.cli_validate.Format = .text;
    if (args.format) |f| {
        if (std.mem.eql(u8, f, "text")) {
            fmt = .text;
        } else if (std.mem.eql(u8, f, "json")) {
            fmt = .json;
        } else {
            try out.print("validate: unknown --format '{s}' (expected text|json)\n", .{f});
            std.process.exit(2);
        }
    }

    // Optional fixtures file: one input string per line.
    var fixtures_owned: ?[]u8 = null;
    defer if (fixtures_owned) |f| gpa.free(f);
    var fixtures_list: std.ArrayList([]const u8) = .empty;
    defer fixtures_list.deinit(gpa);

    if (args.fixtures) |fpath| {
        fixtures_owned = try std.Io.Dir.cwd().readFileAlloc(io, fpath, gpa, .unlimited);
        const body = fixtures_owned.?;
        var iter = std.mem.splitScalar(u8, body, '\n');
        const last_was_newline = body.len > 0 and body[body.len - 1] == '\n';
        while (iter.next()) |line| {
            if (iter.peek() == null and line.len == 0 and last_was_newline) break;
            try fixtures_list.append(gpa, line);
        }
    }

    const fixtures_slice: ?[]const []const u8 = if (args.fixtures != null)
        fixtures_list.items
    else
        null;

    if (fmt == .text) {
        try out.print("ztok validate {s}\n", .{model_path});
    }

    const result = try ztok.cli_validate.runValidateAny(gpa, loaded.loadedModel(), &pipeline, .{
        .checks = checks,
        .fixtures = fixtures_slice,
        .format = fmt,
    }, out);

    try out.flush();
    if (result.errors > 0) std.process.exit(1);
}

// --- roundtrip ----------------------------------------------------------

fn cmdRoundtrip(gpa: std.mem.Allocator, io: std.Io, raw: []const []const u8, out: *std.Io.Writer) !void {
    var args = try Args.parse(gpa, raw);
    defer args.deinit(gpa);

    const model_path = args.model_path orelse {
        try out.writeAll("roundtrip: --model PATH required\n");
        std.process.exit(2);
    };

    var loaded = loadPipelineAutoDetect(gpa, io, model_path) catch |err| {
        try out.print("roundtrip: failed to load {s}: {s}\n", .{ model_path, @errorName(err) });
        std.process.exit(2);
    };
    defer loaded.deinit();

    // `--optimal`: validate under minimum-token (shortest-path) BPE
    // segmentation — matches `encode --optimal`, so PathPiece / optimal
    // vocabs round-trip in the same mode they're meant to be used.
    if (args.optimal) {
        switch (loaded) {
            .bpe => |*b| b.encode_mode = .optimal,
            else => try out.writeAll("roundtrip: --optimal only applies to BPE models; ignoring\n"),
        }
    }

    var vocab = ztok.Vocab.empty(gpa);
    defer vocab.deinit();
    const pipeline: ztok.Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = if (args.cl100k) .cl100k else .identity,
        .model = loaded.modelValue(),
        .decoder = .concat,
        .vocab = &vocab,
    };

    // Input source: a positional file path, --stdin, or fall back to
    // stdin if neither is given.
    const input_owned: []u8 = if (args.stdin or args.positional.items.len == 0)
        try readStdin(gpa, io)
    else
        try std.Io.Dir.cwd().readFileAlloc(io, args.positional.items[0], gpa, .unlimited);
    defer gpa.free(input_owned);

    const result = try ztok.cli_validate.runRoundtrip(gpa, &pipeline, input_owned, .{
        .summary_only = args.summary,
    }, out);
    try out.flush();
    if (!result.allOk()) std.process.exit(1);
}

// --- shared helper: model loader ----------------------------------------

fn cmdAdaptVocab(gpa: std.mem.Allocator, io: std.Io, raw: []const []const u8, out: *std.Io.Writer) !void {
    var args = try Args.parse(gpa, raw);
    defer args.deinit(gpa);

    const base_path = args.base_path orelse {
        try out.writeAll("adapt-vocab: --base PATH required\n");
        return;
    };
    const corpus_path = args.corpus_path orelse {
        try out.writeAll("adapt-vocab: --corpus PATH required\n");
        return;
    };
    const output_path = args.output_path orelse {
        try out.writeAll("adapt-vocab: --output PATH required\n");
        return;
    };
    const add_n: u32 = args.add_count orelse 100;
    if (add_n == 0) {
        try out.writeAll("adapt-vocab: --add must be > 0\n");
        return;
    }

    // Detect base format up-front; we re-use the same detection result
    // when serializing so the output matches.
    const base_fmt = ztok.auto_detect.detectFile(base_path) catch ztok.auto_detect.Format.unknown;
    if (base_fmt == .ztm or base_fmt == .tekken or base_fmt == .unknown) {
        try out.print("adapt-vocab: unsupported base format ({s}). Only tiktoken / HF BPE / SP BPE are supported.\n", .{@tagName(base_fmt)});
        return ztok.vocab_continued_pretrain.Error.UnsupportedFormat;
    }

    // Load base. We only need the BPE shape so we go through
    // `loadPipelineAutoDetect` and reject non-BPE variants explicitly.
    var loaded = loadPipelineAutoDetect(gpa, io, base_path) catch |err| {
        try out.print("adapt-vocab: failed to load {s}: {s}\n", .{ base_path, @errorName(err) });
        return err;
    };
    defer loaded.deinit();

    const base_bpe_ptr: *const ztok.Bpe = switch (loaded) {
        .bpe => |*b| b,
        else => {
            try out.print("adapt-vocab: base vocab is not a BPE model (got {s}).\n", .{@tagName(loaded.kind())});
            return ztok.vocab_continued_pretrain.Error.UnsupportedFormat;
        },
    };

    const corpus = std.Io.Dir.cwd().readFileAlloc(io, corpus_path, gpa, .unlimited) catch |err| {
        try out.print("adapt-vocab: failed to read corpus {s}: {s}\n", .{ corpus_path, @errorName(err) });
        return err;
    };
    defer gpa.free(corpus);

    try out.print(
        "adapt-vocab: base={s} ({s}, {d} tokens), corpus={d} bytes, add={d}\n",
        .{ base_path, @tagName(base_fmt), base_bpe_ptr.count, corpus.len, add_n },
    );
    try out.flush();

    var result = ztok.vocab_continued_pretrain.adaptBpe(gpa, base_bpe_ptr, corpus, .{
        .add = add_n,
    }) catch |err| {
        try out.print("adapt-vocab: adaptation failed: {s}\n", .{@errorName(err)});
        return err;
    };
    defer result.deinit();

    ztok.vocab_continued_pretrain.writeAdaptedBpe(gpa, &result.adapted_bpe, base_fmt, output_path) catch |err| {
        try out.print("adapt-vocab: write failed: {s}\n", .{@errorName(err)});
        return err;
    };

    try out.print(
        "adapt-vocab: wrote {s} ({d} -> {d} tokens, {d} added)\n",
        .{ output_path, result.old_count, result.adapted_bpe.count, result.new_token_added_count },
    );

    if (args.out_path) |rpt_path| {
        // diff already uses --report+--out; for adapt-vocab the spec
        // overloads --report to take a value. Honor either: if
        // args.report is set, the user provided --report PATH (parsed
        // as flag + positional); if args.out_path is set, --out works.
        try ztok.vocab_continued_pretrain.writeJsonReport(gpa, &result, base_fmt, base_path, rpt_path);
        try out.print("adapt-vocab: wrote report to {s}\n", .{rpt_path});
    } else if (args.report) {
        // --report consumed as flag — try to pick the first positional
        // as its value (spec uses `--report report.json`).
        if (args.positional.items.len > 0) {
            const rpt_path = args.positional.items[0];
            try ztok.vocab_continued_pretrain.writeJsonReport(gpa, &result, base_fmt, base_path, rpt_path);
            try out.print("adapt-vocab: wrote report to {s}\n", .{rpt_path});
        }
    }
}

fn parseOnConflict(s: []const u8) ?ztok.vocab_merge.OnConflict {
    if (std.mem.eql(u8, s, "error")) return .error_out;
    if (std.mem.eql(u8, s, "keep-a")) return .keep_a;
    if (std.mem.eql(u8, s, "keep-b")) return .keep_b;
    return null;
}

/// `ztok merge-vocab` — union two vocabs of the same model kind. See
/// `vocab_merge.zig` for the semantic guarantees.
fn cmdMergeVocab(gpa: std.mem.Allocator, io: std.Io, raw: []const []const u8, out: *std.Io.Writer) !void {
    var args = try Args.parse(gpa, raw);
    defer args.deinit(gpa);

    const a_path = args.a_path orelse {
        try out.writeAll("merge-vocab: --a PATH required\n");
        return;
    };
    const b_path = args.b_path orelse {
        try out.writeAll("merge-vocab: --b PATH required\n");
        return;
    };
    const output_path = args.output_path orelse {
        try out.writeAll("merge-vocab: --output PATH required\n");
        return;
    };
    const on_conflict: ztok.vocab_merge.OnConflict = if (args.on_conflict) |s| (parseOnConflict(s) orelse {
        try out.print("merge-vocab: unknown --on-conflict '{s}' (expected error|keep-a|keep-b)\n", .{s});
        return;
    }) else .error_out;
    const prefix_b: []const u8 = args.prefix_b orelse "";

    // Detect format of A — the output format mirrors A's.
    const a_fmt = ztok.auto_detect.detectFile(a_path) catch ztok.auto_detect.Format.unknown;
    if (a_fmt == .ztm or a_fmt == .tekken or a_fmt == .unknown) {
        try out.print("merge-vocab: unsupported A format ({s}). Only tiktoken / HF / SP are supported.\n", .{@tagName(a_fmt)});
        return ztok.vocab_continued_pretrain.Error.UnsupportedFormat;
    }

    var loaded_a = loadPipelineAutoDetect(gpa, io, a_path) catch |err| {
        try out.print("merge-vocab: failed to load --a {s}: {s}\n", .{ a_path, @errorName(err) });
        return err;
    };
    defer loaded_a.deinit();

    var loaded_b = loadPipelineAutoDetect(gpa, io, b_path) catch |err| {
        try out.print("merge-vocab: failed to load --b {s}: {s}\n", .{ b_path, @errorName(err) });
        return err;
    };
    defer loaded_b.deinit();

    // Cross-kind merges are undefined — reject up-front with the
    // sentinel error the spec calls out.
    const a_kind = loaded_a.kind();
    const b_kind = loaded_b.kind();
    if (a_kind != b_kind) {
        try out.print("merge-vocab: incompatible kinds (--a is {s}, --b is {s})\n", .{ @tagName(a_kind), @tagName(b_kind) });
        return ztok.vocab_merge.Error.IncompatibleModelKind;
    }
    if (a_kind == .monster) {
        try out.print("merge-vocab: monster vocab merge is not supported.\n", .{});
        return ztok.vocab_merge.Error.IncompatibleModelKind;
    }

    try out.print(
        "merge-vocab: a={s} ({s}, {d} tokens, {s}), b={s} ({d} tokens), on-conflict={s}, prefix-b='{s}'\n",
        .{
            a_path,            @tagName(a_fmt),       loaded_a.vocabSize(),
            @tagName(a_kind),  b_path,                loaded_b.vocabSize(),
            @tagName(on_conflict), prefix_b,
        },
    );
    try out.flush();

    const opts: ztok.vocab_merge.Options = .{
        .on_conflict = on_conflict,
        .prefix_b = prefix_b,
    };

    switch (a_kind) {
        .bpe => {
            const a_bpe = switch (loaded_a) {
                .bpe => |*x| x,
                else => unreachable,
            };
            const b_bpe = switch (loaded_b) {
                .bpe => |*x| x,
                else => unreachable,
            };
            var res = ztok.vocab_merge.mergeBpe(gpa, a_bpe, b_bpe, opts) catch |err| {
                try out.print("merge-vocab: merge failed: {s}\n", .{@errorName(err)});
                return err;
            };
            defer res.deinit();

            ztok.vocab_continued_pretrain.writeAdaptedBpe(gpa, &res.merged, a_fmt, output_path) catch |err| {
                try out.print("merge-vocab: write failed: {s}\n", .{@errorName(err)});
                return err;
            };
            try writeMergeSidecar(gpa, output_path, .{
                .original_a_size = res.original_a_size,
                .original_b_size = res.original_b_size,
                .merged_size = res.merged_size,
                .conflicts_count = res.conflicts_count,
                .b_to_merged = res.b_to_merged,
            }, out);
            try out.print("merge-vocab: wrote {s} ({d} tokens, {d} conflicts)\n", .{
                output_path, res.merged_size, res.conflicts_count,
            });
        },
        .unigram => {
            const a_uni = switch (loaded_a) {
                .unigram => |*x| x,
                else => unreachable,
            };
            const b_uni = switch (loaded_b) {
                .unigram => |*x| x,
                else => unreachable,
            };
            var res = ztok.vocab_merge.mergeUnigram(gpa, a_uni, b_uni, opts) catch |err| {
                try out.print("merge-vocab: merge failed: {s}\n", .{@errorName(err)});
                return err;
            };
            defer res.deinit();

            switch (a_fmt) {
                .hf_json => ztok.hf_writer.writeUnigramFile(gpa, &res.merged, output_path, .{}) catch |err| {
                    try out.print("merge-vocab: write failed: {s}\n", .{@errorName(err)});
                    return err;
                },
                .sentencepiece => ztok.sp_writer.writeUnigramFile(gpa, &res.merged, output_path, .{
                    .unk_id = res.merged.unk_id,
                }) catch |err| {
                    try out.print("merge-vocab: write failed: {s}\n", .{@errorName(err)});
                    return err;
                },
                else => {
                    try out.print("merge-vocab: cannot write Unigram to {s}\n", .{@tagName(a_fmt)});
                    return ztok.vocab_continued_pretrain.Error.UnsupportedFormat;
                },
            }
            try writeMergeSidecar(gpa, output_path, .{
                .original_a_size = res.original_a_size,
                .original_b_size = res.original_b_size,
                .merged_size = res.merged_size,
                .conflicts_count = res.conflicts_count,
                .b_to_merged = res.b_to_merged,
            }, out);
            try out.print("merge-vocab: wrote {s} ({d} tokens, {d} conflicts)\n", .{
                output_path, res.merged_size, res.conflicts_count,
            });
        },
        .wordpiece => {
            const a_wp = switch (loaded_a) {
                .wordpiece => |*x| x,
                else => unreachable,
            };
            const b_wp = switch (loaded_b) {
                .wordpiece => |*x| x,
                else => unreachable,
            };
            var res = ztok.vocab_merge.mergeWordPiece(gpa, a_wp, b_wp, opts) catch |err| {
                try out.print("merge-vocab: merge failed: {s}\n", .{@errorName(err)});
                return err;
            };
            defer res.deinit();

            if (a_fmt != .hf_json) {
                try out.print("merge-vocab: cannot write WordPiece to {s}\n", .{@tagName(a_fmt)});
                return ztok.vocab_continued_pretrain.Error.UnsupportedFormat;
            }
            ztok.hf_writer.writeWordPieceFile(gpa, &res.merged, output_path, .{}) catch |err| {
                try out.print("merge-vocab: write failed: {s}\n", .{@errorName(err)});
                return err;
            };
            try writeMergeSidecar(gpa, output_path, .{
                .original_a_size = res.original_a_size,
                .original_b_size = res.original_b_size,
                .merged_size = res.merged_size,
                .conflicts_count = res.conflicts_count,
                .b_to_merged = res.b_to_merged,
            }, out);
            try out.print("merge-vocab: wrote {s} ({d} tokens, {d} conflicts)\n", .{
                output_path, res.merged_size, res.conflicts_count,
            });
        },
        .monster, .byte_id => {
            // monster guarded above; byte_id is a synthetic diff-only
            // kind that loadPipelineAutoDetect doesn't return.
            unreachable;
        },
    }
}

fn writeMergeSidecar(
    gpa: std.mem.Allocator,
    output_path: []const u8,
    meta: ztok.vocab_merge.MergeMapMeta,
    out: *std.Io.Writer,
) !void {
    const sidecar_path = try std.fmt.allocPrint(gpa, "{s}.merge-map.json", .{output_path});
    defer gpa.free(sidecar_path);
    try ztok.vocab_merge.writeMergeMap(gpa, meta, sidecar_path);
    try out.print("merge-vocab: wrote sidecar {s}\n", .{sidecar_path});
}

/// Owned model union returned by `loadPipelineAutoDetect`. The variant
/// stores the concrete model value (not a pointer) so the caller can
/// drop the union after `deinit()` without dangling pointers. Helpers
/// `modelValue()` and `loadedModel()` produce the borrow-pointer
/// references the pipeline + validator need.
const OwnedModel = union(enum) {
    bpe: ztok.Bpe,
    unigram: ztok.Unigram,
    wordpiece: ztok.WordPiece,
    monster: ztok.Monster,

    fn deinit(self: *OwnedModel) void {
        switch (self.*) {
            .bpe => |*b| b.deinit(),
            .unigram => |*u| u.deinit(),
            .wordpiece => |*w| w.deinit(),
            .monster => |*m| m.deinit(),
        }
    }

    fn modelValue(self: *const OwnedModel) ztok.Model {
        return switch (self.*) {
            .bpe => |*b| .{ .bpe = b },
            .unigram => |*u| .{ .unigram = u },
            .wordpiece => |*w| .{ .wordpiece = w },
            .monster => |*m| .{ .monster = m },
        };
    }

    fn loadedModel(self: *const OwnedModel) ztok.cli_validate.LoadedModel {
        return switch (self.*) {
            .bpe => |*b| .{ .bpe = b },
            .unigram => |*u| .{ .unigram = u },
            .wordpiece => |*w| .{ .wordpiece = w },
            .monster => |*m| .{ .monster = m },
        };
    }

    fn kind(self: *const OwnedModel) ztok.diff.ModelKind {
        return switch (self.*) {
            .bpe => .bpe,
            .unigram => .unigram,
            .wordpiece => .wordpiece,
            .monster => .monster,
        };
    }

    fn vocabSize(self: *const OwnedModel) u32 {
        return switch (self.*) {
            .bpe => |*b| b.count,
            .unigram => |*u| u.count,
            .wordpiece => |*w| w.count,
            .monster => |*m| m.count,
        };
    }
};

/// Pipeline-stage choices for a transcode endpoint. `transcode` decodes
/// A's ids to text and re-encodes with B, so each endpoint's pipeline
/// must round-trip text losslessly. The right normalizer / pre-tokenizer
/// / decoder triple depends on the model format:
///   * HF byte-level BPE (GPT-2 / Llama-3 / Qwen2 …): the `hf_byte_level`
///     pre-tokenizer maps raw bytes into the GPT-2 printable alphabet on
///     encode and the `byte_level` decoder reverses it.
///   * HF WordPiece (BERT): the `wordpiece` decoder strips `##` and
///     re-spaces — note this is *not* byte-exact (see transcode caveat).
///   * everything else (raw `.tiktoken`, SP, Monster, HF non-byte-level):
///     `identity` + `concat`, which round-trips when the model stores
///     pieces as literal bytes.
const TranscodeStages = struct {
    normalizer: ztok.Normalizer = .identity,
    pre_tokenizer: ztok.PreTokenizer = .identity,
    decoder: ztok.Decoder = .concat,
};

/// Decide the round-tripping pipeline stages for `path`. Inspects the HF
/// JSON pre-tokenizer/decoder kind for byte-level vs wordpiece; defaults
/// to identity+concat for all other formats.
fn transcodeStagesFor(io: std.Io, path: []const u8, cl100k: bool) TranscodeStages {
    _ = io;
    const fmt = ztok.auto_detect.detectFile(path) catch ztok.auto_detect.Format.unknown;
    const is_hf = fmt == .hf_json or
        (fmt == .unknown and std.mem.endsWith(u8, path, ".json"));

    if (is_hf) {
        // HF JSON tells us the right byte-level / wordpiece wiring
        // directly — that always wins over the global --cl100k flag,
        // which only makes sense for raw .tiktoken endpoints.
        var hf = ztok.hf_json.loadFromFile(std.heap.page_allocator, path) catch return .{};
        defer hf.deinit();
        if (hf.pre_tok_kind == .byte_level or hf.decoder_kind == .byte_level) {
            return .{ .pre_tokenizer = .hf_byte_level, .decoder = .byte_level };
        }
        if (hf.model_kind == .wordpiece or hf.decoder_kind == .wordpiece) {
            return .{ .decoder = .{ .wordpiece = .{} } };
        }
        return .{};
    }

    // Raw tiktoken / SP / Monster: honor --cl100k for the pre-tokenizer
    // (cl100k_base regex split); pieces are literal bytes so concat
    // decode round-trips.
    if (cl100k) return .{ .pre_tokenizer = .cl100k, .decoder = .concat };
    return .{};
}

// --- transcode -----------------------------------------------------------

/// `ztok transcode --from VOCAB_A --to VOCAB_B [--input ids.txt|--stdin]
/// [--format text|jsonl]`. Re-maps an id stream from tokenizer A to
/// tokenizer B by decoding A's ids to text and re-encoding with B (text
/// is the lossless bridge). Both vocab formats are auto-detected.
fn cmdTranscode(gpa: std.mem.Allocator, io: std.Io, raw: []const []const u8, out: *std.Io.Writer) !void {
    var args = try Args.parse(gpa, raw);
    defer args.deinit(gpa);

    const from_path = args.from_path orelse {
        try out.writeAll("transcode: --from VOCAB_A required\n");
        std.process.exit(2);
    };
    const to_path = args.to_path orelse {
        try out.writeAll("transcode: --to VOCAB_B required\n");
        std.process.exit(2);
    };

    var fmt: ztok.transcode.Format = .text;
    if (args.format) |f| {
        if (std.mem.eql(u8, f, "text")) {
            fmt = .text;
        } else if (std.mem.eql(u8, f, "jsonl")) {
            fmt = .jsonl;
        } else {
            try out.print("transcode: unknown --format '{s}' (expected text|jsonl)\n", .{f});
            std.process.exit(2);
        }
    }

    var loaded_a = loadPipelineAutoDetect(gpa, io, from_path) catch |err| {
        try out.print("transcode: failed to load --from {s}: {s}\n", .{ from_path, @errorName(err) });
        std.process.exit(2);
    };
    defer loaded_a.deinit();
    var loaded_b = loadPipelineAutoDetect(gpa, io, to_path) catch |err| {
        try out.print("transcode: failed to load --to {s}: {s}\n", .{ to_path, @errorName(err) });
        std.process.exit(2);
    };
    defer loaded_b.deinit();

    const stages_a = transcodeStagesFor(io, from_path, args.cl100k);
    const stages_b = transcodeStagesFor(io, to_path, args.cl100k);

    var vocab_a = ztok.Vocab.empty(gpa);
    defer vocab_a.deinit();
    var vocab_b = ztok.Vocab.empty(gpa);
    defer vocab_b.deinit();

    const pipe_a: ztok.Pipeline = .{
        .normalizer = stages_a.normalizer,
        .pre_tokenizer = stages_a.pre_tokenizer,
        .model = loaded_a.modelValue(),
        .decoder = stages_a.decoder,
        .vocab = &vocab_a,
    };
    const pipe_b: ztok.Pipeline = .{
        .normalizer = stages_b.normalizer,
        .pre_tokenizer = stages_b.pre_tokenizer,
        .model = loaded_b.modelValue(),
        .decoder = stages_b.decoder,
        .vocab = &vocab_b,
    };

    // Input: lines of space-separated A-ids. --input PATH > positional
    // file > --stdin / empty stdin.
    const input_owned: []u8 = if (args.input_path) |ip|
        try std.Io.Dir.cwd().readFileAlloc(io, ip, gpa, .unlimited)
    else if (args.stdin or args.positional.items.len == 0)
        try readStdin(gpa, io)
    else
        try std.Io.Dir.cwd().readFileAlloc(io, args.positional.items[0], gpa, .unlimited);
    defer gpa.free(input_owned);

    _ = ztok.transcode.transcodeCorpus(gpa, &pipe_a, &pipe_b, input_owned, out, .{
        .format = fmt,
        .max_src_id = loaded_a.vocabSize(),
    }) catch |err| switch (err) {
        error.IdOutOfRange => {
            try out.print(
                "transcode: input contains an id >= --from vocab size ({d}); " ++
                    "is the input really tokenized with {s}?\n",
                .{ loaded_a.vocabSize(), from_path },
            );
            try out.flush();
            std.process.exit(1);
        },
        error.InvalidCharacter, error.Overflow => {
            try out.writeAll("transcode: input has a non-numeric token; expected space-separated decimal ids\n");
            try out.flush();
            std.process.exit(1);
        },
        else => return err,
    };
}

/// Load a tokenizer model from `path` by auto-detecting the file
/// format. Supports all four formats `auto_detect` covers:
///   * raw `.tiktoken` (BPE)
///   * HF `tokenizer.json` (BPE, WordPiece, or Unigram — model type
///     inside the JSON determines which validator runs)
///   * SentencePiece `.model` (BPE or Unigram)
///   * ztok Monster `.ztm`
///
/// Returns an `OwnedModel` discriminated union; caller must call
/// `deinit()`.
fn loadPipelineAutoDetect(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !OwnedModel {
    _ = io; // auto_detect.detectFile + each loader uses its own io internally
    const fmt = ztok.auto_detect.detectFile(path) catch ztok.auto_detect.Format.unknown;
    const effective: ztok.auto_detect.Format = if (fmt != .unknown) fmt else blk: {
        // Extension fallback for unknown magic, e.g. tiny custom files.
        const base = std.fs.path.basename(path);
        if (std.mem.eql(u8, base, "tekken.json")) break :blk .tekken;
        if (std.mem.endsWith(u8, base, ".tekken.json")) break :blk .tekken;
        if (std.mem.endsWith(u8, path, ".tiktoken")) break :blk .tiktoken;
        if (std.mem.endsWith(u8, path, ".json")) break :blk .hf_json;
        if (std.mem.endsWith(u8, path, ".model")) break :blk .sentencepiece;
        if (std.mem.endsWith(u8, path, ".ztm")) break :blk .ztm;
        break :blk .unknown;
    };

    switch (effective) {
        .tiktoken => return .{ .bpe = try ztok.Bpe.loadTiktokenFile(gpa, path) },
        .hf_json => {
            var hf = try ztok.hf_json.loadFromFile(gpa, path);
            defer hf.deinit();
            return switch (hf.model_kind) {
                .bpe => .{ .bpe = try ztok.hf_bridge.bpeFromHF(gpa, &hf) },
                .unigram => .{ .unigram = try ztok.hf_bridge.unigramFromHF(gpa, &hf) },
                .wordpiece => blk: {
                    // Resolve unk_id by looking up the token name from
                    // the HF `unk_token` string against the vocab. Fall
                    // back to 0 (matches BERT's [UNK]=0 convention).
                    var unk_id: ztok.TokenId = 0;
                    if (hf.unk_token) |needle| {
                        var i: u32 = 0;
                        while (i < hf.vocab.count) : (i += 1) {
                            const piece = hf.vocab.bytes[hf.vocab.offsets[i]..hf.vocab.offsets[i + 1]];
                            if (std.mem.eql(u8, piece, needle)) {
                                unk_id = i;
                                break;
                            }
                        }
                    }
                    break :blk .{ .wordpiece = try ztok.hf_bridge.wordPieceFromHF(gpa, &hf, .{
                        .unk_id = unk_id,
                    }) };
                },
            };
        },
        .sentencepiece => {
            var sp = try ztok.sp_model.loadFromFile(gpa, path);
            defer sp.deinit();
            return switch (sp.model_kind) {
                .bpe => .{ .bpe = try ztok.sp_bridge.bpeFromSP(gpa, &sp) },
                .unigram => .{ .unigram = try ztok.sp_bridge.unigramFromSP(gpa, &sp) },
                else => return error.UnsupportedSpModelKind,
            };
        },
        .ztm => {
            // `readFileMeta` returns a `LoadedMonster` whose `deinit`
            // would re-free the Monster's SoA buffers. We move the
            // Monster value out of the wrapper without invoking its
            // deinit, so the union owns the buffers directly.
            const loaded = try ztok.monster_io.readFileMeta(gpa, path);
            return .{ .monster = loaded.monster };
        },
        .tekken => {
            // Tekken's BPE is shaped exactly like a tiktoken vocab once
            // the special-token id shift is applied (specials get the
            // bottom of the id space, vocab tokens are bumped up by
            // `num_special_tokens`). The loader returns a `TekkenModel`
            // wrapper that owns the `Bpe` value plus the special-token
            // list; we move the Bpe out (then drop the wrapper, freeing
            // only the specials + pattern) so the `OwnedModel` union
            // takes responsibility for the Bpe's SoA buffers.
            //
            // CLI consumers that need the special-token table or the
            // `config.pattern` regex should call `ztok.tekken.loadTekkenFile`
            // directly — `loadPipelineAutoDetect` is the lowest-common-
            // denominator path and only exposes the raw Bpe today.
            var tk = try ztok.tekken.loadTekkenFile(gpa, path);
            const bpe_out = tk.bpe;
            // Avoid double-free of the Bpe buffers by zeroing the
            // wrapper's reference before `tk.deinit()`.
            tk.bpe = .{
                .allocator = gpa,
                .bytes = &.{},
                .offsets = &.{},
                .count = 0,
                .by_bytes = std.StringHashMap(ztok.TokenId).init(gpa),
            };
            tk.deinit();
            return .{ .bpe = bpe_out };
        },
        .unknown => return error.UnknownModelFormat,
    }
}

fn readStdin(gpa: std.mem.Allocator, io: std.Io) ![]u8 {
    var buf: [4096]u8 = undefined;
    var stdin_file = std.Io.File.stdin();
    var stdin_r = stdin_file.reader(io, &buf);
    const r = &stdin_r.interface;

    var list: std.ArrayList(u8) = .empty;
    defer list.deinit(gpa);

    var tmp: [4096]u8 = undefined;
    while (true) {
        const n = try r.readSliceShort(&tmp);
        if (n == 0) break;
        try list.appendSlice(gpa, tmp[0..n]);
    }
    return list.toOwnedSlice(gpa);
}
