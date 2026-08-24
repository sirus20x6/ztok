//! Cross-tokenizer benchmark: ztok against TokenMonster and
//! SentencePiece on each tool's own vocab.
//!
//! Usage:
//!   bench_cross --kind monster     --model VOCAB.ztm  --corpus PATH [--iters N] [--batch N]
//!   bench_cross --kind unigram     --model VOCAB.model --corpus PATH [--iters N] [--batch N]
//!   bench_cross --kind sp-bpe      --model VOCAB.model --corpus PATH [--iters N] [--batch N]
//!   bench_cross --kind hf-unigram  --model TOKENIZER.json --corpus PATH [--iters N] [--batch N]
//!   bench_cross --kind hf-bpe      --model TOKENIZER.json --corpus PATH [--iters N] [--batch N]
//!
//! Reports MB/s, tokens/s, and bytes-per-token. Used in tandem with
//! `bench/bench_competitors.py --lib tokenmonster|sentencepiece` so the
//! same corpus is encoded by both implementations of the same vocab.
//!
//! Also writes the first 100 lines of the corpus through the loaded
//! vocab and prints the resulting id stream to stderr when
//! `--dump-sample` is passed — used by `bench/equivalence_check.py` to
//! verify the ztok loader produced an encoder that agrees with the
//! reference tool on the same input.

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

const Kind = enum { monster, unigram, sp_bpe, hf_unigram, hf_bpe, hf_wordpiece, tekken };

const Args = struct {
    kind: Kind,
    model_path: []const u8,
    corpus_path: []const u8,
    iters: u32 = 5,
    batch: u32 = 0,
    workers: ?u32 = null,
    cache_entries_per_worker: usize = 0,
    cache_stats: bool = false,
    disable_merge_index: bool = false,
    ragged: bool = false,
    pretok_only: bool = false,
    dump_sample: bool = false,
    sample_lines: u32 = 100,
    /// When set, instead of (or in addition to) the per-line id dump,
    /// emit the normalized BYTES for each line as
    ///   `NORM <line_idx> <hex-encoded normalized bytes>`
    /// on stderr. Used by `equivalence_check.py hf-wordpiece` to
    /// validate the HF normalizer chain (BertNormalizer + WordPiece
    /// for bert-base-uncased) against the upstream `tokenizers`
    /// library WITHOUT requiring a full pretokenizer port. Post-1.17
    /// agent D.
    dump_normalized: bool = false,
};

fn parseArgs(gpa: std.mem.Allocator, owned: *std.ArrayList([]const u8)) !?Args {
    var kind: ?Kind = null;
    var model_path: ?[]const u8 = null;
    var corpus_path: ?[]const u8 = null;
    var iters: u32 = 5;
    var batch: u32 = 0;
    var workers: ?u32 = null;
    var cache_entries_per_worker: usize = 0;
    var cache_stats = false;
    var disable_merge_index = false;
    var ragged = false;
    var pretok_only = false;
    var dump_sample = false;
    var sample_lines: u32 = 100;
    var dump_normalized = false;

    var i: usize = 1;
    while (i < owned.items.len) : (i += 1) {
        const a = owned.items[i];
        if (std.mem.eql(u8, a, "--kind")) {
            const v = owned.items[i + 1];
            kind = if (std.mem.eql(u8, v, "monster"))
                .monster
            else if (std.mem.eql(u8, v, "unigram"))
                .unigram
            else if (std.mem.eql(u8, v, "sp-bpe"))
                .sp_bpe
            else if (std.mem.eql(u8, v, "hf-unigram"))
                .hf_unigram
            else if (std.mem.eql(u8, v, "hf-bpe"))
                .hf_bpe
            else if (std.mem.eql(u8, v, "hf-wordpiece"))
                .hf_wordpiece
            else if (std.mem.eql(u8, v, "tekken"))
                .tekken
            else
                return error.UnknownKind;
            i += 1;
        } else if (std.mem.eql(u8, a, "--model")) {
            model_path = owned.items[i + 1];
            i += 1;
        } else if (std.mem.eql(u8, a, "--corpus")) {
            corpus_path = owned.items[i + 1];
            i += 1;
        } else if (std.mem.eql(u8, a, "--iters")) {
            iters = try std.fmt.parseInt(u32, owned.items[i + 1], 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--batch")) {
            batch = try std.fmt.parseInt(u32, owned.items[i + 1], 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--cache-entries")) {
            cache_entries_per_worker = try std.fmt.parseInt(usize, owned.items[i + 1], 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--workers")) {
            workers = try std.fmt.parseInt(u32, owned.items[i + 1], 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--cache-stats")) {
            cache_stats = true;
        } else if (std.mem.eql(u8, a, "--disable-merge-index")) {
            disable_merge_index = true;
        } else if (std.mem.eql(u8, a, "--ragged")) {
            ragged = true;
        } else if (std.mem.eql(u8, a, "--pretok-only")) {
            pretok_only = true;
        } else if (std.mem.eql(u8, a, "--dump-sample")) {
            dump_sample = true;
        } else if (std.mem.eql(u8, a, "--dump-normalized")) {
            dump_normalized = true;
        } else if (std.mem.eql(u8, a, "--sample-lines")) {
            sample_lines = try std.fmt.parseInt(u32, owned.items[i + 1], 10);
            i += 1;
        }
    }
    _ = gpa;

    const k = kind orelse return null;
    const m = model_path orelse return null;
    const c = corpus_path orelse return null;
    return .{
        .kind = k,
        .model_path = m,
        .corpus_path = c,
        .iters = iters,
        .batch = batch,
        .workers = workers,
        .cache_entries_per_worker = cache_entries_per_worker,
        .cache_stats = cache_stats,
        .disable_merge_index = disable_merge_index,
        .ragged = ragged,
        .pretok_only = pretok_only,
        .dump_sample = dump_sample,
        .sample_lines = sample_lines,
        .dump_normalized = dump_normalized,
    };
}

fn printUsage() void {
    std.debug.print(
        \\usage: bench_cross --kind {{monster|unigram|sp-bpe|hf-unigram|hf-bpe|hf-wordpiece|tekken}} \
        \\                   --model VOCAB --corpus PATH [--iters N] [--batch N]
        \\                   [--workers N] [--cache-entries N] [--cache-stats] [--pretok-only]
        \\                   [--disable-merge-index] [--ragged]
        \\                   [--dump-sample] [--dump-normalized] [--sample-lines N]
        \\
    , .{});
}

const Loaded = struct {
    // Exactly one of these is populated.
    monster: ?ztok.Monster = null,
    unigram: ?ztok.Unigram = null,
    bpe: ?ztok.Bpe = null,
    wordpiece: ?ztok.WordPiece = null,
    // SP model lives only long enough to bridge into bpe/unigram; we
    // keep it around because Bpe borrows into its bytes via `by_bytes`
    // map keys. Wait — that's actually not true: `bpeFromSP` copies
    // bytes. So we *can* free it. But it's tiny, so we keep for
    // post-mortem inspection if needed.
    sp_model: ?ztok.SpModel = null,
    // HF JSON loader output (only set when --kind hf-unigram). Kept
    // alive because the produced Unigram's piece bytes are arena-owned
    // by the HFTokenizer's vocab storage. (HF bridge: `unigramFromHF`
    // builds its own trie from copies, so deinit order between hf and
    // unigram is independent.)
    hf_tokenizer: ?ztok.hf_json.HFTokenizer = null,
    /// Owned Tekken model (only set when --kind tekken). The model's
    /// inner `Bpe` is borrowed via `&self.tekken.?.bpe` in `pipeline()` —
    /// we don't copy it into `Loaded.bpe` because `TekkenModel.deinit`
    /// calls `bpe.deinit` itself; a double-free would result.
    tekken: ?ztok.tekken.TekkenModel = null,
    /// Pre-encode normalizer chosen by the loader. For the Monster
    /// path we honor the .ztm header's capcode/normalizer-flag bytes
    /// (via `monster_io.readBytesMeta + recommendedNormalizer`) so the
    /// pipeline mirrors TM-Go's `normalize(data, capcode, normalizer)`
    /// pre-processing. Falls back to `monster_io.normalizerForVocab(name)`
    /// when the .ztm file pre-dates the metadata bytes (legacy headers
    /// stamped capcode=0/norm=0 even on TM-derived vocabs).
    normalizer: ztok.Normalizer = .identity,
    /// Allocator that owns the `normalizer`'s heap children (Sequence
    /// inner slice, Replace pattern/content strings). Set by the loader
    /// to the same allocator used for the rest of the load — `deinit`
    /// hands it back to `Normalizer.deinit` so HF normalizer chains
    /// don't leak. Defaulted to a no-op (Failing) allocator for variants
    /// whose normalizer never owns heap state.
    normalizer_allocator: std.mem.Allocator = std.testing.failing_allocator,
    /// Pre-tokenizer chosen by the loader. SP / TM stay on `.identity`
    /// (they tokenize raw bytes); HF GPT-2-style BPE wants
    /// `.hf_byte_level` (regex split + byte_to_unicode in one pass)
    /// to match the upstream `tokenizers` library output.
    pre_tokenizer: ztok.PreTokenizer = .identity,
    /// Decoder chosen by the loader. HF byte-level BPE needs
    /// `.byte_level` to reverse the byte_to_unicode mapping when
    /// decoding ids back to text. Other loaders use `.concat`.
    decoder: ztok.Decoder = .concat,
    /// Optional special-token scanner. Heap-allocated by the loader
    /// (hf-bpe / hf-unigram paths) when the source tokenizer.json
    /// includes an `added_tokens` array with at least one entry that
    /// must bypass model encoding (e.g. `<|im_start|>`, `<|endoftext|>`).
    /// The scanner runs ahead of normalization/BPE in `Pipeline.encode`
    /// and emits the assigned id directly. Lifetime: must outlive every
    /// encode call in `main`; freed in `deinit` below.
    ///
    /// SP / Monster / WordPiece paths leave this null — `added_tokens`
    /// on the returned Pipeline stays null and the fast path is taken.
    added_tokens_scanner: ?*ztok.added_tokens.Scanner = null,
    /// Allocator that owns the heap `added_tokens_scanner` slot itself.
    /// Defaulted to a failing allocator so accidental frees on paths
    /// that never populated the scanner are caught.
    added_tokens_allocator: std.mem.Allocator = std.testing.failing_allocator,

    fn deinit(self: *Loaded) void {
        if (self.monster) |*m| m.deinit();
        if (self.unigram) |*u| u.deinit();
        if (self.bpe) |*b| b.deinit();
        if (self.wordpiece) |*w| w.deinit();
        if (self.sp_model) |*s| s.deinit();
        if (self.hf_tokenizer) |*h| h.deinit();
        // Tekken owns its own Bpe; freeing it here would double-free with
        // self.bpe (which we deliberately leave null on the tekken path).
        if (self.tekken) |*t| t.deinit();
        // Free the HF-derived normalizer's owned children (Sequence
        // inner slice, Replace pattern/content strings, etc.). Cheap
        // no-op for identity / capcode / sp_precompiled normalizers.
        self.normalizer.deinit(self.normalizer_allocator);
        // Free the added_tokens scanner (built only on hf-bpe / hf-unigram
        // paths with a non-empty `added_tokens` array).
        if (self.added_tokens_scanner) |sc| {
            sc.deinit();
            self.added_tokens_allocator.destroy(sc);
        }
    }

    fn pipeline(self: *const Loaded, vocab: *const ztok.Vocab) ztok.Pipeline {
        if (self.monster) |*m| return .{
            .normalizer = self.normalizer,
            .pre_tokenizer = self.pre_tokenizer,
            .model = .{ .monster = m },
            .decoder = self.decoder,
            .vocab = vocab,
            .added_tokens = self.added_tokens_scanner,
        };
        // SP-derived BPE / Unigram: honor the SP normalizer recorded
        // alongside the model. For LLaMA-style models that's
        // `sp_precompiled` with add_dummy_prefix + escape_whitespaces
        // on, so every input gets a leading U+2581 and every space
        // becomes U+2581. The BPE encoder uses `.longest_match` mode
        // (set by `bpeFromSP`) so multi-byte pieces like `▁world` get
        // emitted directly instead of routing through byte-fallback.
        //
        // Unigram now honors the SP normalizer (post-1.13). The Viterbi
        // encoder iterates by codepoint and uses `min_score -
        // K_UNK_PENALTY` for the unk fallback edge (mirroring SP's
        // `kUnkPenalty = 10`), so U+2581-prefixed input encodes
        // bit-identically to sp-python on the T5 vocab. For SP models
        // with `trainer_spec.byte_fallback`, `unigramFromSP` populates
        // `byte_fallback` and Viterbi prefers the per-byte fallback over
        // the unk-merge path.
        if (self.unigram) |*u| return .{
            .normalizer = self.normalizer,
            .pre_tokenizer = self.pre_tokenizer,
            .model = .{ .unigram = u },
            .decoder = self.decoder,
            .vocab = vocab,
            .added_tokens = self.added_tokens_scanner,
        };
        if (self.bpe) |*b| return .{
            .normalizer = self.normalizer,
            .pre_tokenizer = self.pre_tokenizer,
            .model = .{ .bpe = b },
            .decoder = self.decoder,
            .vocab = vocab,
            .added_tokens = self.added_tokens_scanner,
        };
        // Tekken: identical encode shape to HF/SP BPE, but the live `Bpe`
        // lives inside `self.tekken.bpe`. Borrow at use-site so the
        // pointer reflects the actual storage of the resident `Loaded`.
        if (self.tekken) |*t| return .{
            .normalizer = self.normalizer,
            .pre_tokenizer = self.pre_tokenizer,
            .model = .{ .bpe = &t.bpe },
            .decoder = self.decoder,
            .vocab = vocab,
            .added_tokens = self.added_tokens_scanner,
        };
        if (self.wordpiece) |*w| return .{
            .normalizer = self.normalizer,
            .pre_tokenizer = self.pre_tokenizer,
            .model = .{ .wordpiece = w },
            .decoder = self.decoder,
            .vocab = vocab,
            .added_tokens = self.added_tokens_scanner,
        };
        unreachable;
    }
};

/// Convert the `added_tokens` array parsed by `hf_json` into the
/// `added_tokens.Scanner` form expected by the runtime pipeline.
///
/// We forward every entry verbatim — content / single_word / lstrip /
/// rstrip — regardless of the `special` flag. HF treats entries with
/// `special: false` (e.g. extra chat-template control strings, padding
/// tokens) as still needing fixed-id resolution at the same priority as
/// special entries; the upstream `tokenizers` library runs both through
/// the same byte-trie scan before the model. Filtering on `special`
/// would silently drop `<|im_start|>` / `<|im_end|>` on some vocabs
/// (Qwen2 ships them with `special: true`) but also drop entries the
/// HF reference does honor (Yi-6B has reserved-id pad tokens at
/// `special: false`); a uniform forward keeps us aligned with the
/// reference and lets the scanner's longest-match arbitration handle
/// any overlap.
///
/// Returns null when the source vocab has no `added_tokens` at all
/// (Gemma, T5, GPT-2 — every non-chat-templated vocab), so the
/// downstream pipeline keeps the fast `added_tokens == null` path.
/// Allocates the returned scanner on `gpa` so it can outlive the
/// borrowed `HFTokenizer.added_tokens` slice (Scanner.init copies
/// content bytes into its own arena).
fn buildAddedTokensScanner(
    gpa: std.mem.Allocator,
    hf: *const ztok.hf_json.HFTokenizer,
) !?*ztok.added_tokens.Scanner {
    if (hf.added_tokens.len == 0) return null;

    // Translate hf_json.AddedToken -> added_tokens.AddedToken. The
    // runtime form drops the `special` / `normalized` flags (the
    // scanner doesn't need them — its only job is byte-trie match +
    // single_word/lstrip/rstrip arbitration).
    var converted = try gpa.alloc(ztok.added_tokens.AddedToken, hf.added_tokens.len);
    defer gpa.free(converted);
    for (hf.added_tokens, 0..) |t, i| {
        converted[i] = .{
            .id = t.id,
            .content = t.content,
            .single_word = t.single_word,
            .lstrip = t.lstrip,
            .rstrip = t.rstrip,
        };
    }

    const scanner_ptr = try gpa.create(ztok.added_tokens.Scanner);
    errdefer gpa.destroy(scanner_ptr);
    scanner_ptr.* = try ztok.added_tokens.Scanner.init(gpa, converted);
    return scanner_ptr;
}

/// Build an `added_tokens.Scanner` from an SP `SpModel`'s `.user_defined`
/// pieces — SP's BPE encoder pre-matches these as whole strings before
/// the BPE merge loop sees their bytes. `.control` pieces are
/// deliberately EXCLUDED: SP's encoder treats them as ordinary text
/// during tokenization (`<|endoftext|>` on yi6b is type=.control with
/// id 2 and SP splits it into `<`, `|`, `end`, `of`, `text`, `|>` —
/// pre-scanning them would diverge from sp-python on every code-corpus
/// line that mentions one literally). Pieces with empty byte images,
/// or with types `.unknown` / `.byte` / `.unused` / `.normal` /
/// `.control`, are skipped. Returns `null` when the model has no
/// user_defined pieces (every piece is `.normal`).
///
/// The HF scanner's `single_word` / lstrip / rstrip arbitration knobs
/// don't have an SP analog — SP treats these as literal byte runs with
/// no boundary or whitespace policy. We forward all-false defaults so
/// the scanner's matching is plain longest-prefix.
fn buildSpecialScannerFromSP(
    gpa: std.mem.Allocator,
    sp: *const ztok.sp_model.SpModel,
) !?*ztok.added_tokens.Scanner {
    // Two-pass: count first, allocate exact, then fill — avoids a
    // resizing ArrayList. Special-piece counts are tiny (a few hundred
    // for chat-tuned vocabs) so the second pass is cheap.
    var n_special: usize = 0;
    var id: u32 = 0;
    while (id < sp.count) : (id += 1) {
        const ty = sp.types[id];
        if (ty != .user_defined) continue;
        const start = sp.offsets[id];
        const end = sp.offsets[id + 1];
        if (end == start) continue; // empty piece — defensive
        n_special += 1;
    }
    if (n_special == 0) return null;

    const converted = try gpa.alloc(ztok.added_tokens.AddedToken, n_special);
    defer gpa.free(converted);
    var w: usize = 0;
    id = 0;
    while (id < sp.count) : (id += 1) {
        const ty = sp.types[id];
        if (ty != .user_defined) continue;
        const start = sp.offsets[id];
        const end = sp.offsets[id + 1];
        if (end == start) continue;
        converted[w] = .{
            .id = id,
            .content = sp.bytes[start..end],
            .single_word = false,
            .lstrip = false,
            .rstrip = false,
        };
        w += 1;
    }

    const scanner_ptr = try gpa.create(ztok.added_tokens.Scanner);
    errdefer gpa.destroy(scanner_ptr);
    scanner_ptr.* = try ztok.added_tokens.Scanner.init(gpa, converted);
    return scanner_ptr;
}

fn loadModel(gpa: std.mem.Allocator, io: std.Io, args: Args) !Loaded {
    var out: Loaded = .{};
    errdefer out.deinit();
    switch (args.kind) {
        .monster => {
            // Use readFileMeta so we can pick a TM-equivalent normalizer
            // from the .ztm header bytes when present.
            var loaded = try ztok.monster_io.readFileMeta(gpa, args.model_path);
            out.monster = loaded.monster;
            // Prefer the metadata recommendation; fall back to a
            // vocab-name lookup if the header is legacy (capcode=0,
            // norm=0) — older .ztm files lost the TM metadata in
            // conversion.
            const meta_rec = loaded.recommendedNormalizer();
            if (meta_rec != .identity) {
                out.normalizer = meta_rec;
            } else {
                // basename = path with extension stripped, last segment.
                const path = args.model_path;
                var slash: usize = 0;
                var i: usize = path.len;
                while (i > 0) {
                    i -= 1;
                    if (path[i] == '/' or path[i] == '\\') {
                        slash = i + 1;
                        break;
                    }
                }
                const basename = path[slash..];
                out.normalizer = ztok.monster_io.normalizerForVocab(basename);
            }
        },
        .unigram => {
            // sp_model.loadFromFile uses std.fs.cwd() which is gone in
            // Zig 0.16; read the file via the threaded Io and call
            // loadFromBytes directly.
            const contents = try std.Io.Dir.cwd().readFileAlloc(io, args.model_path, gpa, .unlimited);
            defer gpa.free(contents);
            var sp = try ztok.sp_model.loadFromBytes(gpa, contents);
            errdefer sp.deinit();
            const uni = try ztok.sp_bridge.unigramFromSP(gpa, &sp);
            // Use the charsmap-aware bridge so T5 / mT5 / Japanese SP
            // models pick up their `precompiled_charsmap` rewrite rules.
            // The lazy parse cost is one-time per model load (≤ 1 ms on
            // T5's 177 KB trie); LLaMA-style models without a charsmap
            // skip the work entirely.
            out.normalizer = try ztok.sp_bridge.normalizerFromSPModel(&sp);
            out.sp_model = sp;
            out.unigram = uni;
        },
        .sp_bpe => {
            const contents = try std.Io.Dir.cwd().readFileAlloc(io, args.model_path, gpa, .unlimited);
            defer gpa.free(contents);
            var sp = try ztok.sp_model.loadFromBytes(gpa, contents);
            errdefer sp.deinit();
            const bpe = try ztok.sp_bridge.bpeFromSP(gpa, &sp);
            out.normalizer = try ztok.sp_bridge.normalizerFromSPModel(&sp);
            out.sp_model = sp;
            out.bpe = bpe;
            // SP's BPE encoder pre-scans the input for pieces of type
            // `.user_defined` or `.control`, emitting their fixed id
            // before the BPE merge loop sees the bytes. Without this
            // pre-scan the codepoint-level merge loop can't reach those
            // pieces (intermediate merges don't exist in the trained
            // vocab) and instead splits them into per-character ids —
            // see the gemma `</s>` and yi6b `<|im_start|>` code-corpus
            // regressions at stress 1.24.
            out.added_tokens_scanner = try buildSpecialScannerFromSP(gpa, &out.sp_model.?);
            out.added_tokens_allocator = gpa;
        },
        .hf_unigram => {
            // Load a real HF Unigram tokenizer.json. We deliberately
            // skip the HF Normalizer/PreTokenizer/Decoder pipeline here
            // because real-world HF Unigram exports (llm-jp, etc.) use
            // Sequence([Replace+Regex, Replace+Regex]) normalizers that
            // ztok's hf_json loader simply tags as `.sequence` /
            // `.other` — there's no general Regex normalizer in ztok.
            //
            // For the byte_fallback verification path we only need to
            // confirm the model-level Unigram encoder reproduces HF's
            // `model.tokenize(text)` output. So we run with the
            // identity normalizer and pass pre-normalized text on the
            // wire — the equivalence checker on the Python side does
            // the same (calls `tokenizer.model.tokenize(text)`).
            const contents = try std.Io.Dir.cwd().readFileAlloc(io, args.model_path, gpa, .unlimited);
            defer gpa.free(contents);
            var hf = try ztok.hf_json.loadFromBytes(gpa, contents);
            errdefer hf.deinit();
            const uni = try ztok.hf_bridge.unigramFromHF(gpa, &hf);
            out.normalizer = .identity;
            out.hf_tokenizer = hf;
            out.unigram = uni;
            // NOTE on added_tokens scanner: deliberately NOT wired here.
            // The Python equivalence check compares `tok.model.tokenize(text)`
            // — the MODEL-LEVEL encoder — which bypasses HF's added-token
            // pre-scan. Routing input through the scanner would resolve
            // literal `<unk>` / `<s>` substrings in source code to their
            // reserved ids and diverge from the reference (llmjp3 × code
            // line 178: `addToken("<unk>")` then encodes to [..., 0, ...]
            // on the ztok side while HF picks `("<` + `unk` + `>` — the
            // Viterbi path is the correct comparison target). Production
            // callers that want the chat-style scanning should configure
            // a Pipeline with `added_tokens` populated; bench_cross's
            // model-level harness intentionally exercises only the
            // Unigram lattice walk.
        },
        .hf_bpe => {
            // Load a real HF BPE tokenizer.json. The canonical GPT-2
            // pipeline is:
            //   normalizer: identity (GPT-2 doesn't normalize)
            //   pre_tokenizer: ByteLevel (regex split + byte_to_unicode)
            //   model: BPE (rank by id, `.bpe_merge` encode mode)
            //   decoder: ByteLevel (reverses byte_to_unicode)
            //
            // The HF ByteLevel pre-tokenizer maps each raw byte to a
            // printable codepoint in U+0021..U+0142 — its output bytes
            // are what the BPE model sees as "tokens" in the vocab.
            // ztok's `hf_byte_level` PreTokenizer mirrors that one-pass
            // split-and-map so the BPE encoder gets the exact same
            // input bytes the upstream `tokenizers` library hands it.
            //
            // Post-1.18 agent B: honor the HF normalizer chain too —
            // Phi-3-style models ship `Sequence[Prepend("▁"),
            // Replace(" "→"▁")]` as the normalizer and the ByteLevel
            // pre-tok still runs on its output. GPT-2 (no normalizer
            // section) degrades cleanly to `.identity` via
            // `normalizerFromHF`, so existing GPT-2 paths keep working.
            const contents = try std.Io.Dir.cwd().readFileAlloc(io, args.model_path, gpa, .unlimited);
            defer gpa.free(contents);
            var hf = try ztok.hf_json.loadFromBytes(gpa, contents);
            errdefer hf.deinit();
            const bpe = try ztok.hf_bridge.bpeFromHF(gpa, &hf);
            const norm = try ztok.hf_bridge.normalizerFromHF(gpa, &hf);
            out.normalizer = norm;
            out.normalizer_allocator = gpa;
            out.decoder = .byte_level;
            // Stash hf FIRST so the chain pointer below references the
            // stable in-place storage on `out`. If we set hf_tokenizer
            // after taking the pointer, the union value would point at a
            // stack temporary that's about to die.
            out.hf_tokenizer = hf;
            out.bpe = bpe;
            // Post-1.18 agent A: if the JSON contained a Sequence pretok
            // (Falcon-7B, Qwen2-7B, Llama-3-8B, etc.), wire up the chain
            // executor. Post-1.18 agent B: when the JSON's pre_tokenizer
            // is explicitly null (Phi-3-style SP-reshelled HF BPE),
            // run identity pretok + concat decoder — the normalizer
            // chain already produced raw U+2581-tagged UTF-8 bytes and
            // the vocab pieces are stored as raw UTF-8 too (no byte_to_
            // unicode round-trip needed). Otherwise stay on the
            // hand-coded GPT-2 ByteLevel path.
            if (out.hf_tokenizer != null and out.hf_tokenizer.?.pretok_chain != null) {
                out.pre_tokenizer = .{ .chain = &out.hf_tokenizer.?.pretok_chain.? };
            } else if (out.hf_tokenizer != null and out.hf_tokenizer.?.pre_tok_kind == .none) {
                out.pre_tokenizer = .identity;
                out.decoder = .concat;
            } else {
                out.pre_tokenizer = .hf_byte_level;
            }
            // Build the added_tokens scanner. Critical for chat-tuned BPE
            // vocabs (Qwen2-7B, Yi-6B, Phi-3-mini, Llama-3-8B): without
            // this step `<|im_start|>` / `<|im_end|>` / `<|endoftext|>`
            // get byte-level-encoded character by character instead of
            // resolving to their reserved ids, which causes equivalence
            // failures on every code-corpus line that mentions them.
            //
            // Yields `null` (and the pipeline keeps the fast path) for
            // GPT-2 / vanilla BPE exports without an `added_tokens`
            // array — those keep their existing behavior unchanged.
            if (out.hf_tokenizer) |*hf_ref| {
                out.added_tokens_scanner = try buildAddedTokensScanner(gpa, hf_ref);
                out.added_tokens_allocator = gpa;
            }
        },
        .hf_wordpiece => {
            // Load a real HF WordPiece tokenizer.json (BERT family).
            // bert-base-uncased's pipeline:
            //   normalizer: BertNormalizer { clean_text, handle_chinese_chars,
            //                                strip_accents=null, lowercase=true }
            //   pre_tokenizer: BertPreTokenizer (not yet ported — bench
            //     skips this pretok and relies on the equivalence
            //     checker to compare NORMALIZED bytes only via
            //     `--dump-normalized`)
            //   model: WordPiece (greedy longest-match w/ ## prefix)
            //   decoder: WordPiece
            //
            // For throughput numbers we run with the identity pretok
            // (so the WordPiece encoder sees the whole normalized
            // text as one "word" and falls back to UNK; the perf
            // measurement still exercises the BertNormalizer code
            // path which is what we're benchmarking). For correctness
            // the caller passes `--dump-normalized` and compares the
            // normalized-bytes stream against HF.
            const contents = try std.Io.Dir.cwd().readFileAlloc(io, args.model_path, gpa, .unlimited);
            defer gpa.free(contents);
            var hf = try ztok.hf_json.loadFromBytes(gpa, contents);
            errdefer hf.deinit();
            // Find the [UNK] id for the WordPiece encoder.
            var unk_id: ztok.TokenId = 0;
            if (hf.unk_token) |needle| {
                var idx: u32 = 0;
                while (idx < hf.vocab.count) : (idx += 1) {
                    const piece = hf.vocab.bytes[hf.vocab.offsets[idx]..hf.vocab.offsets[idx + 1]];
                    if (std.mem.eql(u8, piece, needle)) {
                        unk_id = idx;
                        break;
                    }
                }
            }
            const wp = try ztok.hf_bridge.wordPieceFromHF(gpa, &hf, .{ .unk_id = unk_id });
            const norm = try ztok.hf_bridge.normalizerFromHF(gpa, &hf);
            out.normalizer = norm;
            out.normalizer_allocator = gpa;
            out.pre_tokenizer = .identity;
            out.decoder = .{ .wordpiece = .{} };
            out.hf_tokenizer = hf;
            out.wordpiece = wp;
        },
        .tekken => {
            // Mistral Tekken: tiktoken-format BPE wrapped in JSON. The
            // loader already builds the Bpe with `.bpe_merge` encode_mode
            // and a populated 256-entry byte_fallback table, so the encode
            // path is identical to the HF / SP BPE arms — only the
            // pre-tokenizer differs. We use `.cl100k` (tiktoken's GPT-4
            // regex) as a close-enough proxy for the Tekken pattern until
            // a generic PCRE-tiktoken pretok lands; this matches what the
            // Tekken loader's docstring suggests for the bench path and is
            // bit-correct on prose corpora (where the regex variants split
            // identically). For code corpora the contractions / digit
            // segmentation may diverge — flagged in equivalence_check.py.
            var tk = try ztok.tekken.loadTekkenFile(gpa, args.model_path);
            errdefer tk.deinit();

            // Wire the named special tokens through the added-tokens
            // scanner so strings like `<unk>`, `<s>`, `[INST]`,
            // `<|begin_of_text|>` resolve to their reserved ids before
            // hitting the BPE merge loop. Without this every literal
            // special-token byte sequence in the input would be byte-
            // level-encoded instead.
            if (tk.specials.len > 0) {
                var converted = try gpa.alloc(ztok.added_tokens.AddedToken, tk.specials.len);
                defer gpa.free(converted);
                for (tk.specials, 0..) |s, i| {
                    converted[i] = .{
                        .id = s.id,
                        .content = s.content,
                        .single_word = false,
                        .lstrip = false,
                        .rstrip = false,
                    };
                }
                const scanner_ptr = try gpa.create(ztok.added_tokens.Scanner);
                errdefer gpa.destroy(scanner_ptr);
                scanner_ptr.* = try ztok.added_tokens.Scanner.init(gpa, converted);
                out.added_tokens_scanner = scanner_ptr;
                out.added_tokens_allocator = gpa;
            }

            out.normalizer = .identity;
            out.pre_tokenizer = .cl100k;
            out.decoder = .concat;
            out.tekken = tk;
        },
    }
    return out;
}

fn vocabStats(loaded: *const Loaded) struct { count: u32, bytes: usize } {
    if (loaded.monster) |m| return .{ .count = m.count, .bytes = m.bytes.len };
    if (loaded.unigram) |u| return .{ .count = u.count, .bytes = u.bytes.len };
    if (loaded.bpe) |b| return .{ .count = b.count, .bytes = b.bytes.len };
    if (loaded.tekken) |t| return .{ .count = t.bpe.count, .bytes = t.bpe.bytes.len };
    return .{ .count = 0, .bytes = 0 };
}

/// Encode one line through the Tekken pipeline using Tekken's OWN
/// pre-tokenization regex (`ztok.tekken.pretok`) rather than the cl100k
/// proxy the throughput path wires. The throughput benchmark only needs
/// representative MB/s, but the equivalence dump must match
/// `mistral_common` id-for-id, and Tekken's pattern differs from cl100k
/// (case-aware word split, single-codepoint digits, `/`-terminated
/// punctuation). We replicate `Pipeline.encodeText`'s scan -> split ->
/// encodeChunk composition here, swapping in the Tekken splitter.
///
/// `loaded.tekken` must be set. Honors the special-token scanner so
/// literal `<s>` / `[INST]` / etc. resolve to their reserved ids first.
fn tekkenEncodeLine(
    gpa: std.mem.Allocator,
    loaded: *Loaded,
    line: []const u8,
) ![]ztok.token.TokenId {
    const TokenId = ztok.token.TokenId;
    const bpe = &loaded.tekken.?.bpe;

    var ids: std.ArrayList(TokenId) = .empty;
    errdefer ids.deinit(gpa);

    const encodeSegment = struct {
        fn run(
            a: std.mem.Allocator,
            b: *const ztok.bpe.Bpe,
            acc: *std.ArrayList(TokenId),
            text: []const u8,
        ) !void {
            if (text.len == 0) return;
            const spans = try ztok.tekken.pretok.split(a, text);
            defer a.free(spans);
            // BPE writes at most one id per byte per span.
            const scratch = try a.alloc(TokenId, text.len);
            defer a.free(scratch);
            for (spans) |sp| {
                const chunk = text[sp.start..sp.end];
                const got = b.encodeChunk(chunk, scratch[0..chunk.len]);
                try acc.appendSlice(a, got);
            }
        }
    }.run;

    if (loaded.added_tokens_scanner) |scanner| {
        const segs = try ztok.added_tokens.scan(scanner, gpa, line);
        defer gpa.free(segs);
        for (segs) |seg| switch (seg) {
            .text => |t| try encodeSegment(gpa, bpe, &ids, line[t.start..t.end]),
            .special => |s| try ids.append(gpa, s.id),
        };
    } else {
        try encodeSegment(gpa, bpe, &ids, line);
    }

    return ids.toOwnedSlice(gpa);
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
        printUsage();
        std.process.exit(2);
    };

    const corpus = try std.Io.Dir.cwd().readFileAlloc(io, args.corpus_path, gpa, .unlimited);
    defer gpa.free(corpus);

    var loaded = try loadModel(gpa, io, args);
    defer loaded.deinit();
    if (args.disable_merge_index) {
        if (loaded.bpe) |*bpe| bpe.merge_index_enabled = false;
    }

    var v = ztok.Vocab.empty(gpa);
    defer v.deinit();
    const pipe = loaded.pipeline(&v);

    var stdout_buf: [4096]u8 = undefined;
    var stdout_w = std.Io.File.stdout().writer(io, &stdout_buf);
    const out = &stdout_w.interface;
    defer out.flush() catch {};

    const stats = vocabStats(&loaded);

    try out.print("kind:    {s}\n", .{@tagName(args.kind)});
    try out.print("model:   {s}\n", .{args.model_path});
    try out.print("corpus:  {d} bytes ({s})\n", .{ corpus.len, args.corpus_path });
    try out.print("vocab:   {d} tokens, {d} vocab bytes\n", .{ stats.count, stats.bytes });

    // Optional sample dump for the equivalence checker. Goes to stderr
    // so stdout stays parseable.
    if (args.dump_sample) {
        var lines_done: u32 = 0;
        var p: usize = 0;
        while (lines_done < args.sample_lines and p < corpus.len) : (lines_done += 1) {
            const start = p;
            while (p < corpus.len and corpus[p] != '\n') : (p += 1) {}
            const line = corpus[start..p];
            if (p < corpus.len) p += 1;

            // Tekken's equivalence dump must use Tekken's own
            // pre-tokenization regex, not the cl100k proxy the pipeline
            // wires for throughput. Route it through the dedicated helper.
            const ids = if (args.kind == .tekken)
                try tekkenEncodeLine(gpa, &loaded, line)
            else
                try pipe.encode(gpa, line);
            defer gpa.free(ids);

            var sbuf: [256]u8 = undefined;
            var sw = std.Io.File.stderr().writer(io, &sbuf);
            const serr = &sw.interface;
            try serr.print("LINE {d}", .{lines_done});
            for (ids) |id| try serr.print(" {d}", .{id});
            try serr.print("\n", .{});
            try serr.flush();
        }
    }

    // Optional normalized-bytes dump for the HF normalizer chain
    // equivalence checker. Emits one line per input as
    //   NORM <idx> <hex-encoded normalized bytes>
    // The hex encoding sidesteps newline/control-byte ambiguity in
    // the per-line line-oriented protocol (post-normalization output
    // can contain whitespace, NUL, or replacement chars).
    if (args.dump_normalized) {
        var lines_done: u32 = 0;
        var p: usize = 0;
        while (lines_done < args.sample_lines and p < corpus.len) : (lines_done += 1) {
            const start = p;
            while (p < corpus.len and corpus[p] != '\n') : (p += 1) {}
            const line = corpus[start..p];
            if (p < corpus.len) p += 1;

            const norm = try loaded.normalizer.normalize(gpa, line);
            defer gpa.free(norm);

            var sbuf: [256]u8 = undefined;
            var sw = std.Io.File.stderr().writer(io, &sbuf);
            const serr = &sw.interface;
            try serr.print("NORM {d} ", .{lines_done});
            for (norm) |b| try serr.print("{x:0>2}", .{b});
            try serr.print("\n", .{});
            try serr.flush();
        }
    }

    // When dumping (for the equivalence checker), skip the throughput
    // run — equivalence callers don't need the MB/s numbers and the
    // SP-BPE encoder's O(N^2) merge loop on the full corpus would hang
    // the script. Throughput callers don't pass --dump-sample so they're
    // unaffected.
    if (args.dump_sample or args.dump_normalized) return;

    if (args.pretok_only) {
        var total_spans: u64 = 0;
        const t0 = nanosNow();
        var k: u32 = 0;
        while (k < args.iters) : (k += 1) {
            var scanner = ztok.hf_bytelevel_pretok.SpanScanner.init(corpus);
            var relative_ends: [264]u32 = undefined;
            while (true) {
                const batch_ends = scanner.fillRelativeEnds(&relative_ends, 256);
                total_spans += batch_ends.count;
                if (batch_ends.count == 0) break;
            }
        }
        const elapsed_ns = nanosNow() - t0;
        const bytes_total = @as(u64, corpus.len) * args.iters;
        const mb_per_sec = @as(f64, @floatFromInt(bytes_total)) /
            (@as(f64, @floatFromInt(elapsed_ns)) / 1e9) / 1e6;
        try out.print("mode:    GPT-2 pretoken-only, iters={d}\n", .{args.iters});
        try out.print("spans/run: {d}\n", .{total_spans / args.iters});
        try out.print("time:    {d:.2} ms\n", .{@as(f64, @floatFromInt(elapsed_ns)) / 1e6});
        try out.print("MB/s:    {d:.1}\n", .{mb_per_sec});
        return;
    }

    if (args.batch == 0) {
        // Single-thread.
        var total_ids: u64 = 0;
        const t0 = nanosNow();
        var k: u32 = 0;
        while (k < args.iters) : (k += 1) {
            const ids = try pipe.encode(gpa, corpus);
            total_ids += ids.len;
            gpa.free(ids);
        }
        const elapsed_ns = nanosNow() - t0;
        const bytes_total = @as(u64, corpus.len) * args.iters;
        const mb_per_sec = @as(f64, @floatFromInt(bytes_total)) /
            (@as(f64, @floatFromInt(elapsed_ns)) / 1e9) / 1e6;
        const tokens_per_sec = @as(f64, @floatFromInt(total_ids)) /
            (@as(f64, @floatFromInt(elapsed_ns)) / 1e9);
        const bpt = @as(f64, @floatFromInt(corpus.len)) /
            @as(f64, @floatFromInt(total_ids / args.iters));
        try out.print("mode:    single-thread, iters={d}\n", .{args.iters});
        try out.print("ids/run: {d}\n", .{total_ids / args.iters});
        try out.print("time:    {d:.2} ms\n", .{@as(f64, @floatFromInt(elapsed_ns)) / 1e6});
        try out.print("MB/s:    {d:.1}\n", .{mb_per_sec});
        try out.print("tok/s:   {d:.0}\n", .{tokens_per_sec});
        try out.print("bytes/tok: {d:.2}\n", .{bpt});
    } else {
        // Batch.
        var pool = try ztok.thread_pool.BatchPool.initWithOptions(gpa, args.workers, .{
            // Keep ztok workers on distinct physical cores before using SMT
            // siblings; the cache/BPE hot path is sensitive to migration.
            .pin_to_physical_cores = true,
        });
        defer pool.deinit();
        var cache: ?ztok.ChunkedEncodeCache = if (args.cache_entries_per_worker > 0)
            try ztok.ChunkedEncodeCache.initForPool(gpa, &pool, args.cache_entries_per_worker)
        else
            null;
        defer if (cache) |*c| c.deinit();
        if (cache) |*c| {
            if (loaded.bpe) |*bpe| try c.seedByteLevelBpe(bpe);
            if (args.cache_stats) {
                for (c.workers) |*worker_cache| worker_cache.enableStats(true);
            }
        }

        // The identity pre-tokenizer has trivial findSafeCut behavior
        // (returns the desired position). `encodeChunked` falls back to
        // single-shot for pretokenizers without findSafeCut — that's
        // exactly the same path we want here. For TM and SP the encoder
        // is greedy/Viterbi over byte spans, so an arbitrary mid-corpus
        // cut may shift a few ids at the boundary, but throughput
        // numbers are still meaningful.
        var total_ids: u64 = 0;
        const t0 = nanosNow();
        var first_iter_ns: u64 = 0;
        var k: u32 = 0;
        while (k < args.iters) : (k += 1) {
            const iter_start = nanosNow();
            if (args.ragged) {
                var encoded = if (cache) |*c|
                    try pipe.encodeChunkedRaggedCached(gpa, &pool, corpus, args.batch, c)
                else
                    try pipe.encodeChunkedRagged(gpa, &pool, corpus, args.batch);
                total_ids += encoded.tokenCount();
                encoded.deinit();
            } else {
                const ids = if (cache) |*c|
                    try pipe.encodeChunkedCached(gpa, &pool, corpus, args.batch, c)
                else
                    try pipe.encodeChunked(gpa, &pool, corpus, args.batch);
                total_ids += ids.len;
                gpa.free(ids);
            }
            if (k == 0) first_iter_ns = nanosNow() - iter_start;
        }
        const elapsed_ns = nanosNow() - t0;
        const bytes_total = @as(u64, corpus.len) * args.iters;
        const mb_per_sec = @as(f64, @floatFromInt(bytes_total)) /
            (@as(f64, @floatFromInt(elapsed_ns)) / 1e9) / 1e6;
        const tokens_per_sec = @as(f64, @floatFromInt(total_ids)) /
            (@as(f64, @floatFromInt(elapsed_ns)) / 1e9);
        const bpt = @as(f64, @floatFromInt(corpus.len)) /
            @as(f64, @floatFromInt(total_ids / args.iters));
        try out.print("mode:    {s}, chunks={d}, workers={d}, iters={d}\n", .{
            if (args.ragged) "ragged" else "batch", args.batch, pool.workerCount(), args.iters,
        });
        try out.print("ids/run: {d}\n", .{total_ids / args.iters});
        try out.print("time:    {d:.2} ms\n", .{@as(f64, @floatFromInt(elapsed_ns)) / 1e6});
        try out.print("MB/s:    {d:.1}\n", .{mb_per_sec});
        if (args.iters > 1 and elapsed_ns > first_iter_ns) {
            const cold_mb_s = @as(f64, @floatFromInt(corpus.len)) /
                (@as(f64, @floatFromInt(first_iter_ns)) / 1e9) / 1e6;
            const warm_bytes = @as(u64, corpus.len) * (args.iters - 1);
            const warm_mb_s = @as(f64, @floatFromInt(warm_bytes)) /
                (@as(f64, @floatFromInt(elapsed_ns - first_iter_ns)) / 1e9) / 1e6;
            try out.print("cold/warm: {d:.1} / {d:.1} MB/s\n", .{ cold_mb_s, warm_mb_s });
        }
        try out.print("tok/s:   {d:.0}\n", .{tokens_per_sec});
        try out.print("bytes/tok: {d:.2}\n", .{bpt});
        if (cache != null and args.cache_stats) {
            const c = &cache.?;
            const stats_cache = c.stats();
            const probes = stats_cache.hits + stats_cache.misses;
            const hit_rate = if (probes == 0) 0.0 else 100.0 * @as(f64, @floatFromInt(stats_cache.hits)) /
                @as(f64, @floatFromInt(probes));
            try out.print(
                "cache:   hits={d} (home={d}, displaced={d}) misses={d} inserts={d} bypasses={d} hit-rate={d:.1}%\n",
                .{
                    stats_cache.hits,
                    stats_cache.home_hits,
                    stats_cache.displaced_hits,
                    stats_cache.misses,
                    stats_cache.inserts,
                    stats_cache.bypasses,
                    hit_rate,
                },
            );
            var min_worker_probes: u64 = std.math.maxInt(u64);
            var max_worker_probes: u64 = 0;
            var active_workers: usize = 0;
            for (c.workers) |worker_cache| {
                const worker_probes = worker_cache.stats.hits + worker_cache.stats.misses +
                    worker_cache.stats.bypasses;
                if (worker_probes != 0) active_workers += 1;
                min_worker_probes = @min(min_worker_probes, worker_probes);
                max_worker_probes = @max(max_worker_probes, worker_probes);
            }
            try out.print("workers: active={d}/{d}, pretokens/worker min={d} max={d}\n", .{
                active_workers,
                c.workers.len,
                min_worker_probes,
                max_worker_probes,
            });
        }
    }
}
