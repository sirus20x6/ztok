//! C ABI for ztok. Stable, opaque-handle wrapper around the Zig
//! `Pipeline`. Wave-C additions:
//!   * Extended normalizer/pretokenizer/decoder enum mappings expose
//!     every variant wired today (.nfc/.nfd/.nfkc/.nfkd/.byte_level on
//!     the normalizer, .cl100k on the pre-tokenizer, .wordpiece /
//!     .byte_level on the decoder).
//!   * File-based pipeline constructors for BPE (tiktoken + HF JSON),
//!     WordPiece (HF JSON) and Unigram (SentencePiece .model). All
//!     return the same opaque `*PipelineHandle`.
//!   * Persistent `BatchPoolHandle` plus `ztok_encode_batch_pooled`.
//!   * `ztok_encode` writes directly into the caller buffer, skipping
//!     the previous scratch+copy.
//!
//! The opaque pipeline struct now carries a tag and owns the heap model
//! (Bpe/Unigram/WordPiece/Monster) referenced by the inner `Pipeline`.
//! The original ABI is preserved: `ztok_pipeline_new`, `ztok_encode*`,
//! `ztok_decode`, `ztok_encode_batch`, `ztok_ids_free`,
//! `ztok_pipeline_free`, `ztok_version` keep their names, signatures,
//! and integer status / enum values.

const std = @import("std");

const Pipeline = @import("pipeline.zig").Pipeline;
const Vocab = @import("vocab.zig").Vocab;
const Normalizer = @import("normalizer.zig").Normalizer;
const PreTokenizer = @import("pretok.zig").PreTokenizer;
const Model = @import("model.zig").Model;
const Decoder = @import("decoder.zig").Decoder;
const BatchPool = @import("thread_pool.zig").BatchPool;
const TokenId = @import("token.zig").TokenId;
const OverlayKind = @import("token.zig").OverlayKind;
const Bpe = @import("bpe.zig").Bpe;
const Unigram = @import("unigram.zig").Unigram;
const WordPiece = @import("wordpiece.zig").WordPiece;
const Monster = @import("monster.zig").Monster;
const RwkvWorld = @import("rwkv_world.zig").RwkvWorld;
const TekkenModel = @import("tekken.zig").TekkenModel;
const monster_io = @import("monster_io.zig");
const sp_model = @import("sp_model.zig");
const auto_detect = @import("auto_detect.zig");
const StreamEncoder = @import("stream.zig").StreamEncoder;
const ngram = @import("ngram.zig");
const chunk = @import("chunk.zig");

// c_allocator (libc malloc) is universally safe across dlopen/.so
// contexts. smp_allocator's threadlocal `thread_index` plus its global
// `cpu_count` initialization race can panic when first touched from a
// thread that the .so didn't initialize itself; libc malloc has none of
// that history-of-process baggage.
const gpa: std.mem.Allocator = std.heap.c_allocator;

// Single source of truth: parsed from build.zig.zon at build time
// (see build.zig where addOptions is wired). Eliminates the multi-file
// VERSION drift that recurred across the 1.18-1.21 waves. Stored as a
// sentinel-terminated slice so ztok_version() can return [*:0]const u8
// without copying.
const VERSION: [:0]const u8 = blk: {
    const v = @import("build_options").version;
    var buf: [v.len:0]u8 = undefined;
    @memcpy(buf[0..v.len], v);
    const out = buf;
    break :blk &out;
};

// --- status codes -----------------------------------------------------

const ZTOK_OK: c_int = 0;
const ZTOK_ERR_OUT_OF_MEMORY: c_int = 1;
const ZTOK_ERR_INVALID_INPUT: c_int = 2;
const ZTOK_ERR_BUFFER_TOO_SMALL: c_int = 3;
const ZTOK_ERR_INTERNAL: c_int = 99;

const Config = extern struct {
    normalizer: c_uint,
    pre_tokenizer: c_uint,
    model: c_uint,
    decoder: c_uint,
};

// --- handle ----------------------------------------------------------

const HandleKind = enum(u8) { byte_id, bpe, unigram, wordpiece, monster, rwkv_world, tekken };

const ModelStorage = union(HandleKind) {
    byte_id: void,
    bpe: Bpe,
    unigram: Unigram,
    wordpiece: WordPiece,
    monster: Monster,
    rwkv_world: RwkvWorld,
    // Tekken lowers its byte vocab into a `Bpe`; we keep the whole loaded
    // model so its `specials` / image / audio config and pattern stay
    // owned for the handle's lifetime. `modelFromStorage` references the
    // inner `.bpe` for the merge loop.
    tekken: TekkenModel,
};

// Wraps the Pipeline plus the heap-owned model+vocab. The Pipeline's
// `Model` variant carries pointers into `model_storage`; the handle
// pins both so those pointers stay valid for the handle's lifetime.
// The opaque `ztok_pipeline*` C type is a `*PipelineHandle` cast to
// `*Pipeline` for ABI back-compat with the old export type.
const PipelineHandle = struct {
    kind: HandleKind,
    pipeline: Pipeline,
    vocab: Vocab,
    model_storage: ModelStorage,
};

// Per-byte_id pipeline storage is shared and never freed. The empty
// constructor doesn't allocate so the allocator field is inert.
var empty_vocab: Vocab = .{
    .allocator = std.heap.c_allocator,
    .bytes = &.{},
    .offsets = &.{},
    .ranks = null,
    .count = 0,
};

// --- enum mapping helpers (open-ended; unknown values -> null) -------

fn normalizerFromKind(k: c_uint) ?Normalizer {
    return switch (k) {
        0 => .identity,
        1 => .nfc,
        2 => .nfd,
        3 => .nfkc,
        4 => .nfkd,
        5 => .byte_level,
        else => null,
    };
}

fn pretokFromKind(k: c_uint) ?PreTokenizer {
    return switch (k) {
        0 => .identity,
        1 => .cl100k,
        2 => .tekken,
        else => null,
    };
}

fn modelKindFromConfig(k: c_uint) ?HandleKind {
    return switch (k) {
        0 => .byte_id,
        else => null,
    };
}

fn decoderFromKind(k: c_uint) ?Decoder {
    return switch (k) {
        0 => .concat,
        1 => .{ .wordpiece = .{} },
        2 => .byte_level,
        else => null,
    };
}

fn mapErr(e: anyerror) c_int {
    return switch (e) {
        error.OutOfMemory => ZTOK_ERR_OUT_OF_MEMORY,
        else => ZTOK_ERR_INTERNAL,
    };
}

fn setStatus(out: ?*c_int, v: c_int) void {
    if (out) |p| p.* = v;
}

// Cast helpers between the exported opaque `*Pipeline` type and our
// internal handle layout.
inline fn handleFromPtr(p: *Pipeline) *PipelineHandle {
    return @ptrCast(@alignCast(p));
}
inline fn handleFromConstPtr(p: *const Pipeline) *const PipelineHandle {
    return @ptrCast(@alignCast(p));
}
inline fn ptrFromHandle(h: *PipelineHandle) *Pipeline {
    return @ptrCast(@alignCast(h));
}

// --- handle construction shared by every constructor -----------------

const default_byte_id_config: Config = .{
    .normalizer = 0,
    .pre_tokenizer = 0,
    .model = 0,
    .decoder = 0,
};

fn newHandle(
    kind: HandleKind,
    norm: Normalizer,
    pre: PreTokenizer,
    dec: Decoder,
    vocab: Vocab,
    storage: ModelStorage,
) !*PipelineHandle {
    const h = try gpa.create(PipelineHandle);
    h.* = .{
        .kind = kind,
        .pipeline = undefined,
        .vocab = vocab,
        .model_storage = storage,
    };
    // The Pipeline.model carries a pointer into h.model_storage; we
    // must set it AFTER the storage is in its final memory location.
    h.pipeline = .{
        .normalizer = norm,
        .pre_tokenizer = pre,
        .model = modelFromStorage(&h.model_storage),
        .decoder = dec,
        // byte_id never reads vocab; loaded models embed their own.
        // Point the field at the handle's vocab (or the empty static
        // for byte_id) so Decoder code that consults it stays safe.
        .vocab = if (kind == .byte_id) &empty_vocab else &h.vocab,
    };
    return h;
}

fn modelFromStorage(s: *ModelStorage) Model {
    return switch (s.*) {
        .byte_id => .byte_id,
        .bpe => .{ .bpe = &s.bpe },
        .unigram => .{ .unigram = &s.unigram },
        .wordpiece => .{ .wordpiece = &s.wordpiece },
        .monster => .{ .monster = &s.monster },
        .rwkv_world => .{ .rwkv_world = &s.rwkv_world },
        .tekken => .{ .bpe = &s.tekken.bpe },
    };
}

fn freeHandle(h: *PipelineHandle) void {
    switch (h.model_storage) {
        .byte_id => {},
        .bpe => |*b| b.deinit(),
        .unigram => |*u| u.deinit(),
        .wordpiece => |*w| w.deinit(),
        .monster => |*m| m.deinit(),
        .rwkv_world => |*r| r.deinit(),
        .tekken => |*t| t.deinit(),
    }
    h.vocab.deinit();
    gpa.destroy(h);
}

// --- exported lifecycle ----------------------------------------------

export fn ztok_pipeline_new(cfg: ?*const Config, out_status: ?*c_int) ?*Pipeline {
    const c = cfg orelse {
        setStatus(out_status, ZTOK_ERR_INVALID_INPUT);
        return null;
    };

    const n = normalizerFromKind(c.normalizer) orelse {
        setStatus(out_status, ZTOK_ERR_INVALID_INPUT);
        return null;
    };
    const pt = pretokFromKind(c.pre_tokenizer) orelse {
        setStatus(out_status, ZTOK_ERR_INVALID_INPUT);
        return null;
    };
    const mk = modelKindFromConfig(c.model) orelse {
        setStatus(out_status, ZTOK_ERR_INVALID_INPUT);
        return null;
    };
    const d = decoderFromKind(c.decoder) orelse {
        setStatus(out_status, ZTOK_ERR_INVALID_INPUT);
        return null;
    };

    // Only byte_id is selectable from the cfg today; concrete model
    // variants come from the file-based constructors.
    std.debug.assert(mk == .byte_id);

    const h = newHandle(.byte_id, n, pt, d, Vocab.empty(gpa), .byte_id) catch |e| {
        setStatus(out_status, mapErr(e));
        return null;
    };
    setStatus(out_status, ZTOK_OK);
    return ptrFromHandle(h);
}

export fn ztok_pipeline_free(p: ?*Pipeline) void {
    if (p) |pp| freeHandle(handleFromPtr(pp));
}

// --- file-based constructors -----------------------------------------

fn effectiveConfig(cfg: ?*const Config, default_pretok: c_uint) Config {
    if (cfg) |c| return c.*;
    return .{
        .normalizer = 0,
        .pre_tokenizer = default_pretok,
        .model = 0,
        .decoder = 0,
    };
}

export fn ztok_pipeline_new_bpe_from_tiktoken(
    tiktoken_path: ?[*:0]const u8,
    cfg_or_null: ?*const Config,
    out_status: ?*c_int,
) ?*Pipeline {
    const path_z = tiktoken_path orelse {
        setStatus(out_status, ZTOK_ERR_INVALID_INPUT);
        return null;
    };
    const path = std.mem.span(path_z);

    // Defaults for tiktoken: identity normalizer, cl100k pre-tokenizer,
    // concat decoder. Caller overrides via cfg.
    const cfg = effectiveConfig(cfg_or_null, 1);
    const n = normalizerFromKind(cfg.normalizer) orelse {
        setStatus(out_status, ZTOK_ERR_INVALID_INPUT);
        return null;
    };
    const pt = pretokFromKind(cfg.pre_tokenizer) orelse {
        setStatus(out_status, ZTOK_ERR_INVALID_INPUT);
        return null;
    };
    const d = decoderFromKind(cfg.decoder) orelse {
        setStatus(out_status, ZTOK_ERR_INVALID_INPUT);
        return null;
    };

    var bpe = Bpe.loadTiktokenFile(gpa, path) catch |e| {
        setStatus(out_status, mapErr(e));
        return null;
    };

    const h = newHandle(.bpe, n, pt, d, Vocab.empty(gpa), .{ .bpe = bpe }) catch |e| {
        bpe.deinit();
        setStatus(out_status, mapErr(e));
        return null;
    };
    setStatus(out_status, ZTOK_OK);
    return ptrFromHandle(h);
}

export fn ztok_pipeline_new_bpe_from_hf_json(
    tokenizer_json_path: ?[*:0]const u8,
    cfg_or_null: ?*const Config,
    out_status: ?*c_int,
) ?*Pipeline {
    const path_z = tokenizer_json_path orelse {
        setStatus(out_status, ZTOK_ERR_INVALID_INPUT);
        return null;
    };
    const path = std.mem.span(path_z);

    const cfg = effectiveConfig(cfg_or_null, 0);
    const n = normalizerFromKind(cfg.normalizer) orelse {
        setStatus(out_status, ZTOK_ERR_INVALID_INPUT);
        return null;
    };
    const pt = pretokFromKind(cfg.pre_tokenizer) orelse {
        setStatus(out_status, ZTOK_ERR_INVALID_INPUT);
        return null;
    };
    const d = decoderFromKind(cfg.decoder) orelse {
        setStatus(out_status, ZTOK_ERR_INVALID_INPUT);
        return null;
    };

    const hf_json = @import("hf_json.zig");
    var hf = hf_json.loadFromFile(gpa, path) catch |e| {
        setStatus(out_status, mapErr(e));
        return null;
    };
    defer hf.deinit();

    var bpe = @import("hf_bridge.zig").bpeFromHF(gpa, &hf) catch |e| {
        setStatus(out_status, mapErr(e));
        return null;
    };

    const h = newHandle(.bpe, n, pt, d, Vocab.empty(gpa), .{ .bpe = bpe }) catch |e| {
        bpe.deinit();
        setStatus(out_status, mapErr(e));
        return null;
    };
    setStatus(out_status, ZTOK_OK);
    return ptrFromHandle(h);
}

// The HF JSON loader now supports WordPiece natively (since 1.3),
// so we route through hf_bridge instead of inlining a parser here.
fn loadWordPieceFromHFJsonPath(path: []const u8, unk_id: u32) !WordPiece {
    const hf_json = @import("hf_json.zig");
    const hf_bridge = @import("hf_bridge.zig");
    var hf = try hf_json.loadFromFile(gpa, path);
    defer hf.deinit();
    return hf_bridge.wordPieceFromHF(gpa, &hf, .{ .unk_id = unk_id });
}

export fn ztok_pipeline_new_wordpiece_from_hf_json(
    tokenizer_json_path: ?[*:0]const u8,
    unk_id: u32,
    cfg_or_null: ?*const Config,
    out_status: ?*c_int,
) ?*Pipeline {
    const path_z = tokenizer_json_path orelse {
        setStatus(out_status, ZTOK_ERR_INVALID_INPUT);
        return null;
    };
    const path = std.mem.span(path_z);

    const cfg = effectiveConfig(cfg_or_null, 0);
    const n = normalizerFromKind(cfg.normalizer) orelse {
        setStatus(out_status, ZTOK_ERR_INVALID_INPUT);
        return null;
    };
    const pt = pretokFromKind(cfg.pre_tokenizer) orelse {
        setStatus(out_status, ZTOK_ERR_INVALID_INPUT);
        return null;
    };
    const d = decoderFromKind(cfg.decoder) orelse {
        setStatus(out_status, ZTOK_ERR_INVALID_INPUT);
        return null;
    };

    var wp = loadWordPieceFromHFJsonPath(path, unk_id) catch |e| {
        setStatus(out_status, mapErr(e));
        return null;
    };

    const h = newHandle(.wordpiece, n, pt, d, Vocab.empty(gpa), .{ .wordpiece = wp }) catch |e| {
        wp.deinit();
        setStatus(out_status, mapErr(e));
        return null;
    };
    setStatus(out_status, ZTOK_OK);
    return ptrFromHandle(h);
}

// Build a Unigram from an SpModel by re-using its piece bytes and
// scores. The SpModel deinits at the end; the Unigram's Builder copies
// piece bytes through its own allocator so lifetime is independent.
fn unigramFromSpModelPath(path: []const u8, unk_id: u32) !Unigram {
    var sp = try sp_model.loadFromFile(gpa, path);
    defer sp.deinit();

    if (sp.count == 0) return error.MalformedJson;
    if (unk_id >= sp.count) return error.InvalidInput;

    var bld = Unigram.Builder.init(gpa);
    defer bld.deinit();
    var id: u32 = 0;
    while (id < sp.count) : (id += 1) {
        _ = try bld.addToken(sp.pieceBytes(id), sp.scores[id]);
    }
    return bld.finalize(unk_id);
}

export fn ztok_pipeline_new_unigram_from_sp_model(
    sp_model_path: ?[*:0]const u8,
    unk_id: u32,
    cfg_or_null: ?*const Config,
    out_status: ?*c_int,
) ?*Pipeline {
    const path_z = sp_model_path orelse {
        setStatus(out_status, ZTOK_ERR_INVALID_INPUT);
        return null;
    };
    const path = std.mem.span(path_z);

    const cfg = effectiveConfig(cfg_or_null, 0);
    const n = normalizerFromKind(cfg.normalizer) orelse {
        setStatus(out_status, ZTOK_ERR_INVALID_INPUT);
        return null;
    };
    const pt = pretokFromKind(cfg.pre_tokenizer) orelse {
        setStatus(out_status, ZTOK_ERR_INVALID_INPUT);
        return null;
    };
    const d = decoderFromKind(cfg.decoder) orelse {
        setStatus(out_status, ZTOK_ERR_INVALID_INPUT);
        return null;
    };

    var u = unigramFromSpModelPath(path, unk_id) catch |e| {
        setStatus(out_status, mapErr(e));
        return null;
    };

    const h = newHandle(.unigram, n, pt, d, Vocab.empty(gpa), .{ .unigram = u }) catch |e| {
        u.deinit();
        setStatus(out_status, mapErr(e));
        return null;
    };
    setStatus(out_status, ZTOK_OK);
    return ptrFromHandle(h);
}

export fn ztok_pipeline_new_monster_from_file(
    path_c: ?[*:0]const u8,
    cfg_or_null: ?*const Config,
    out_status: ?*c_int,
) ?*Pipeline {
    const path_z = path_c orelse {
        setStatus(out_status, ZTOK_ERR_INVALID_INPUT);
        return null;
    };
    const path = std.mem.span(path_z);

    const cfg = effectiveConfig(cfg_or_null, 0);
    const n = normalizerFromKind(cfg.normalizer) orelse {
        setStatus(out_status, ZTOK_ERR_INVALID_INPUT);
        return null;
    };
    const pt = pretokFromKind(cfg.pre_tokenizer) orelse {
        setStatus(out_status, ZTOK_ERR_INVALID_INPUT);
        return null;
    };
    const d = decoderFromKind(cfg.decoder) orelse {
        setStatus(out_status, ZTOK_ERR_INVALID_INPUT);
        return null;
    };

    var m = monster_io.readFile(gpa, path) catch |e| {
        setStatus(out_status, mapErr(e));
        return null;
    };

    const h = newHandle(.monster, n, pt, d, Vocab.empty(gpa), .{ .monster = m }) catch |e| {
        m.deinit();
        setStatus(out_status, mapErr(e));
        return null;
    };
    setStatus(out_status, ZTOK_OK);
    return ptrFromHandle(h);
}

// RWKV "World" tokenizer from a `rwkv_vocab_v20230424.txt`-style file.
// Defaults: identity normalizer, identity pre-tokenizer (greedy match
// runs over the whole input), concat decoder. Caller overrides via cfg.
export fn ztok_pipeline_new_rwkv_from_file(
    path_c: ?[*:0]const u8,
    cfg_or_null: ?*const Config,
    out_status: ?*c_int,
) ?*Pipeline {
    const path_z = path_c orelse {
        setStatus(out_status, ZTOK_ERR_INVALID_INPUT);
        return null;
    };
    const path = std.mem.span(path_z);

    const cfg = effectiveConfig(cfg_or_null, 0);
    const n = normalizerFromKind(cfg.normalizer) orelse {
        setStatus(out_status, ZTOK_ERR_INVALID_INPUT);
        return null;
    };
    const pt = pretokFromKind(cfg.pre_tokenizer) orelse {
        setStatus(out_status, ZTOK_ERR_INVALID_INPUT);
        return null;
    };
    const d = decoderFromKind(cfg.decoder) orelse {
        setStatus(out_status, ZTOK_ERR_INVALID_INPUT);
        return null;
    };

    var r = RwkvWorld.loadFromFile(gpa, path) catch |e| {
        setStatus(out_status, mapErr(e));
        return null;
    };

    const h = newHandle(.rwkv_world, n, pt, d, Vocab.empty(gpa), .{ .rwkv_world = r }) catch |e| {
        r.deinit();
        setStatus(out_status, mapErr(e));
        return null;
    };
    setStatus(out_status, ZTOK_OK);
    return ptrFromHandle(h);
}

// Mistral Tekken tokenizer from a `tekken.json` file (Nemo / Pixtral /
// Devstral / Magistral, etc.). The loader lowers Tekken's base64 byte
// vocab into a `Bpe` with special tokens packed into the bottom of the id
// space (see tekken.zig). Defaults: identity normalizer, Tekken pre-
// tokenizer (kind 2 — the hand-written `tekken_pretok` pattern, NOT
// cl100k), concat decoder (pieces are raw bytes). Caller overrides via cfg.
export fn ztok_pipeline_new_tekken_from_file(
    path_c: ?[*:0]const u8,
    cfg_or_null: ?*const Config,
    out_status: ?*c_int,
) ?*Pipeline {
    const path_z = path_c orelse {
        setStatus(out_status, ZTOK_ERR_INVALID_INPUT);
        return null;
    };
    const path = std.mem.span(path_z);

    const cfg = effectiveConfig(cfg_or_null, 2);
    const n = normalizerFromKind(cfg.normalizer) orelse {
        setStatus(out_status, ZTOK_ERR_INVALID_INPUT);
        return null;
    };
    const pt = pretokFromKind(cfg.pre_tokenizer) orelse {
        setStatus(out_status, ZTOK_ERR_INVALID_INPUT);
        return null;
    };
    const d = decoderFromKind(cfg.decoder) orelse {
        setStatus(out_status, ZTOK_ERR_INVALID_INPUT);
        return null;
    };

    var t = @import("tekken.zig").loadTekkenFile(gpa, path) catch |e| {
        setStatus(out_status, mapErr(e));
        return null;
    };

    const h = newHandle(.tekken, n, pt, d, Vocab.empty(gpa), .{ .tekken = t }) catch |e| {
        t.deinit();
        setStatus(out_status, mapErr(e));
        return null;
    };
    setStatus(out_status, ZTOK_OK);
    return ptrFromHandle(h);
}

// --- encode / decode -------------------------------------------------
//
// `ztok_encode` writes directly into the caller buffer. When the buffer
// is too small or absent, we still need to know the exact id count to
// report through `*out_len`; the simplest correct answer is to run the
// pipeline into a discardable arena, then return the count.

fn encodeIntoBuffer(
    h: *const PipelineHandle,
    input: []const u8,
    out: []TokenId,
) !usize {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const scratch = arena.allocator();

    const pipe = &h.pipeline;
    const normalized = try pipe.normalizer.normalize(scratch, input);
    const pr = try pipe.pre_tokenizer.split(scratch, normalized);
    // Arena will reclaim pr's owned buffer; explicit free is fine but
    // unnecessary here since the arena's about to deinit.

    var n: usize = 0;
    for (pr.spans) |s| {
        const need = pipe.model.maxTokensFor(s.len());
        if (n + need > out.len) return error.BufferTooSmall;
        const written = try pipe.model.encode(scratch, s.slice(pr.data), out[n..]);
        n += written.len;
    }
    return n;
}

fn encodedLen(h: *const PipelineHandle, input: []const u8) !usize {
    const ids = try h.pipeline.encode(gpa, input);
    defer gpa.free(ids);
    return ids.len;
}

export fn ztok_encode(
    p: ?*const Pipeline,
    input: [*]const u8,
    input_len: usize,
    out: ?[*]TokenId,
    out_cap: usize,
    out_len: ?*usize,
) c_int {
    const pp = p orelse return ZTOK_ERR_INVALID_INPUT;
    const olp = out_len orelse return ZTOK_ERR_INVALID_INPUT;
    const h = handleFromConstPtr(pp);

    if (out) |buf| {
        // Fast path: try to write directly into the caller buffer. If
        // it turns out to be too small mid-way we fall back to the
        // sizing path (the spec promises *out_len with the required
        // size on BUFFER_TOO_SMALL).
        if (encodeIntoBuffer(h, input[0..input_len], buf[0..out_cap])) |n| {
            olp.* = n;
            return ZTOK_OK;
        } else |e| switch (e) {
            error.BufferTooSmall => {
                const need = encodedLen(h, input[0..input_len]) catch |e2| return mapErr(e2);
                olp.* = need;
                return ZTOK_ERR_BUFFER_TOO_SMALL;
            },
            else => return mapErr(e),
        }
    }

    // No buffer at all — just report the required size.
    const need = encodedLen(h, input[0..input_len]) catch |e| return mapErr(e);
    olp.* = need;
    return ZTOK_ERR_BUFFER_TOO_SMALL;
}

// One requested annotation channel. `kind` is a `ztok_overlay_kind`
// value; `out` is a caller-owned buffer of `out_cap` u32 entries that
// the call fills with one value per emitted token. See OverlayKind for
// per-kind semantics.
const OverlayChannel = extern struct {
    kind: c_uint,
    out: ?[*]u32,
    out_cap: usize,
};

// Encode `input`, writing ids into `out_ids` and, for each entry in
// `channels`, one aligned u32 value per token into that channel's `out`
// buffer. All buffers share the token count reported via `out_len`.
//
// Sizing protocol mirrors `ztok_encode`: pass `out_ids == null` to query
// the token count (returns BUFFER_TOO_SMALL, sets *out_len). When
// `out_ids` is non-null, every listed channel's `out` must be non-null
// and hold at least `*out_len` entries; if any buffer (ids or a channel)
// is too small, nothing is copied, *out_len is set to the required count,
// and BUFFER_TOO_SMALL is returned.
export fn ztok_encode_with_overlays(
    p: ?*const Pipeline,
    input: [*]const u8,
    input_len: usize,
    out_ids: ?[*]TokenId,
    out_ids_cap: usize,
    channels: ?[*]OverlayChannel,
    n_channels: usize,
    out_len: ?*usize,
) c_int {
    const pp = p orelse return ZTOK_ERR_INVALID_INPUT;
    const olp = out_len orelse return ZTOK_ERR_INVALID_INPUT;
    if (n_channels > 0 and channels == null) return ZTOK_ERR_INVALID_INPUT;
    const h = handleFromConstPtr(pp);
    const chans: []OverlayChannel = if (channels) |c| c[0..n_channels] else &.{};

    // Translate the requested channel kinds into the Zig `want` list.
    // OverlayKind is a non-exhaustive enum(u16), so any value maps.
    const want = gpa.alloc(OverlayKind, n_channels) catch return ZTOK_ERR_OUT_OF_MEMORY;
    defer gpa.free(want);
    for (chans, 0..) |c, i| want[i] = @enumFromInt(@as(u16, @truncate(c.kind)));

    var enc = h.pipeline.encodeWithOverlays(gpa, input[0..input_len], want) catch |e| return mapErr(e);
    defer enc.deinit(gpa);

    const n = enc.ids.len;
    olp.* = n;

    // Does every destination buffer have room?
    var fits = out_ids != null and out_ids_cap >= n;
    if (fits) {
        for (chans) |c| {
            if (c.out == null or c.out_cap < n) {
                fits = false;
                break;
            }
        }
    }
    if (!fits) return ZTOK_ERR_BUFFER_TOO_SMALL;

    @memcpy(out_ids.?[0..n], enc.ids[0..n]);
    for (chans, 0..) |c, i| @memcpy(c.out.?[0..n], enc.overlays[i].values[0..n]);
    return ZTOK_OK;
}

// Select which domain overlay channels (e.g. x86-64 opcode_class /
// operand_class) `ztok_encode_with_overlays` will populate. `domain` is a
// `ztok_overlay_domain` value; an unrecognized value leaves the pipeline
// unchanged and returns ZTOK_ERR_INVALID_INPUT.
export fn ztok_pipeline_set_overlay_domain(p: ?*Pipeline, domain: c_uint) c_int {
    const pp = p orelse return ZTOK_ERR_INVALID_INPUT;
    const h = handleFromPtr(pp);
    const dom: Pipeline.OverlayDomain = switch (domain) {
        0 => .none,
        1 => .x86_64,
        else => return ZTOK_ERR_INVALID_INPUT,
    };
    h.pipeline.overlay_domain = dom;
    return ZTOK_OK;
}

export fn ztok_decode(
    p: ?*const Pipeline,
    ids: [*]const TokenId,
    ids_len: usize,
    out: ?[*]u8,
    out_cap: usize,
    out_len: ?*usize,
) c_int {
    const pp = p orelse return ZTOK_ERR_INVALID_INPUT;
    const olp = out_len orelse return ZTOK_ERR_INVALID_INPUT;
    const h = handleFromConstPtr(pp);

    const bytes = h.pipeline.decode(gpa, ids[0..ids_len]) catch |e| return mapErr(e);
    defer gpa.free(bytes);

    olp.* = bytes.len;
    if (out == null or out_cap < bytes.len) return ZTOK_ERR_BUFFER_TOO_SMALL;

    @memcpy(out.?[0..bytes.len], bytes);
    return ZTOK_OK;
}

// --- batch encode (per-call pool, preserved for back-compat) ---------
//
// Each returned id-buffer is prefixed with a `Header` storing the
// allocation length so `ztok_ids_free` can call `Allocator.free`
// correctly without the C caller round-tripping the length.

// Distinct magic tags so a mismatched free (e.g. passing an ngram-batch
// u64 buffer to ztok_ids_free, or vice versa) is caught instead of
// silently corrupting the heap. The `magic` field only exists in builds
// with runtime safety on (debug / ReleaseSafe); in ReleaseFast/Small it
// compiles to a zero-sized field so the ABI and allocation size are
// unchanged. See `MagicTag` below.
const id_buf_magic: u32 = 0x5A54_4944; // "ZTID"
const u64_buf_magic: u32 = 0x5A54_5536; // "ZTU6"

// Zero-sized in unsafe builds, a u32 tag in safe builds. Kept first in
// each header so the layout past it (byte_len) is identical to the
// pre-magic layout in release.
const MagicTag = if (std.debug.runtime_safety) u32 else void;

inline fn setMagic(slot: *MagicTag, comptime tag: u32) void {
    if (std.debug.runtime_safety) slot.* = tag;
}
inline fn checkMagic(slot: *MagicTag, comptime tag: u32) void {
    if (std.debug.runtime_safety) {
        std.debug.assert(slot.* == tag); // mismatched/wrong-typed free
    }
}

const Header = extern struct { magic: MagicTag, byte_len: usize };
const header_size = std.mem.alignForward(usize, @sizeOf(Header), @alignOf(TokenId));
const buf_align: std.mem.Alignment = .fromByteUnits(@max(@alignOf(Header), @alignOf(TokenId)));

fn payloadFromHeader(h: *Header) [*]TokenId {
    const raw: [*]u8 = @ptrCast(h);
    return @ptrCast(@alignCast(raw + header_size));
}

fn headerFromPayload(p: [*]TokenId) *Header {
    const raw: [*]u8 = @ptrCast(p);
    return @ptrCast(@alignCast(raw - header_size));
}

fn allocIdBuf(n: usize) ?[*]TokenId {
    if (n == 0) return null;
    const total_bytes = header_size + n * @sizeOf(TokenId);
    const raw = gpa.alignedAlloc(u8, buf_align, total_bytes) catch return null;
    const hdr: *Header = @ptrCast(@alignCast(raw.ptr));
    setMagic(&hdr.magic, id_buf_magic);
    hdr.byte_len = total_bytes;
    return payloadFromHeader(hdr);
}

fn freeIdBuf(p: [*]TokenId) void {
    const hdr = headerFromPayload(p);
    checkMagic(&hdr.magic, id_buf_magic);
    const raw_ptr: [*]u8 = @ptrCast(hdr);
    const aligned: [*]align(buf_align.toByteUnits()) u8 = @alignCast(raw_ptr);
    gpa.free(aligned[0..hdr.byte_len]);
}

// Worker-side context for the parallel encode-direct-into-C-buffer
// path. Each worker reuses its arena scratch (from the BatchPool) AND
// allocates its own output buffer in-thread — moving the per-input
// malloc into the worker fan-out so it no longer runs serially before
// the encode kicks off. This shaves several ms off the batch tail on
// 10K-input batches where a serial pre-allocation loop dominated.
const BatchCtx = struct {
    pipe: *const Pipeline,
    inputs_ptr: [*]const [*]const u8,
    input_lens: [*]const usize,
    pool: *BatchPool,
    out_ids: [*]?[*]TokenId,
    out_lens: [*]usize,
    expansion: usize,
    errored: std.atomic.Value(u32),

    pub fn run(c: *BatchCtx, idx: usize, widx: usize) void {
        const input = c.inputs_ptr[idx][0..c.input_lens[idx]];
        const cap = c.pipe.model.maxTokensFor(input.len * c.expansion);
        if (cap == 0) {
            c.out_ids[idx] = null;
            c.out_lens[idx] = 0;
            return;
        }
        const buf_ptr = allocIdBuf(cap) orelse {
            _ = c.errored.fetchAdd(1, .acq_rel);
            c.out_ids[idx] = null;
            c.out_lens[idx] = 0;
            return;
        };
        const scratch = c.pool.resetArena(widx);

        const written = encodeOneInto(c.pipe, scratch, input, buf_ptr) catch {
            _ = c.errored.fetchAdd(1, .acq_rel);
            freeIdBuf(buf_ptr);
            c.out_ids[idx] = null;
            c.out_lens[idx] = 0;
            return;
        };
        if (written == 0) {
            freeIdBuf(buf_ptr);
            c.out_ids[idx] = null;
            c.out_lens[idx] = 0;
            return;
        }
        c.out_ids[idx] = buf_ptr;
        c.out_lens[idx] = written;
    }

    // Same logic as Pipeline.encodeText but writes directly into the
    // caller-owned id buffer. Returns ids-written count.
    fn encodeOneInto(
        pipe: *const Pipeline,
        scratch: std.mem.Allocator,
        input: []const u8,
        out_buf: [*]TokenId,
    ) !usize {
        const normalized = try pipe.normalizer.normalize(scratch, input);
        var pr = try pipe.pre_tokenizer.split(scratch, normalized);
        defer pr.deinit(scratch);

        var n: usize = 0;
        for (pr.spans) |s| {
            const w = try pipe.model.encode(scratch, s.slice(pr.data), out_buf[n .. n + pipe.model.maxTokensFor(s.len())]);
            n += w.len;
        }
        return n;
    }
};

fn runBatch(
    h: *const PipelineHandle,
    pool: *BatchPool,
    inputs_ptr: [*]const [*]const u8,
    input_lens: [*]const usize,
    n: usize,
    out_ids: [*]?[*]TokenId,
    out_lens: [*]usize,
) c_int {
    if (n == 0) return ZTOK_OK;

    // If the pipeline uses added_tokens (specials scanned PRE-normalize),
    // the encode path can interleave specials with normalize/pretok output
    // and the maxTokensFor upper bound is harder to pre-allocate per
    // input without running the scanner. Fall through to the legacy
    // alloc-then-memcpy path for safety.
    if (h.pipeline.added_tokens != null) {
        return runBatchLegacy(h, pool, inputs_ptr, input_lens, n, out_ids, out_lens);
    }

    // Pre-zero slots so a worker that exits early on cap==0 leaves a
    // clean state, and an OOM rollback below has well-defined memory.
    for (0..n) |i| {
        out_ids[i] = null;
        out_lens[i] = 0;
    }

    var ctx: BatchCtx = .{
        .pipe = &h.pipeline,
        .inputs_ptr = inputs_ptr,
        .input_lens = input_lens,
        .pool = pool,
        .out_ids = out_ids,
        .out_lens = out_lens,
        .expansion = h.pipeline.normalizer.maxByteExpansion() * h.pipeline.pre_tokenizer.maxByteExpansion(),
        .errored = .init(0),
    };

    pool.runBatch(BatchCtx, &ctx, n) catch |e| {
        for (out_ids[0..n]) |maybe| if (maybe) |q| freeIdBuf(q);
        for (out_ids[0..n]) |*slot| slot.* = null;
        for (out_lens[0..n]) |*slot| slot.* = 0;
        return mapErr(e);
    };

    if (ctx.errored.load(.acquire) != 0) {
        for (out_ids[0..n]) |maybe| if (maybe) |q| freeIdBuf(q);
        for (out_ids[0..n]) |*slot| slot.* = null;
        for (out_lens[0..n]) |*slot| slot.* = 0;
        return ZTOK_ERR_INTERNAL;
    }

    return ZTOK_OK;
}

// Fallback for the added-tokens path: stays on the legacy
// alloc-encode-memcpy path so we don't double the maintenance burden
// for a niche feature.
fn runBatchLegacy(
    h: *const PipelineHandle,
    pool: *BatchPool,
    inputs_ptr: [*]const [*]const u8,
    input_lens: [*]const usize,
    n: usize,
    out_ids: [*]?[*]TokenId,
    out_lens: [*]usize,
) c_int {
    const input_slices = gpa.alloc([]const u8, n) catch return ZTOK_ERR_OUT_OF_MEMORY;
    defer gpa.free(input_slices);
    for (input_slices, 0..) |*s, i| s.* = inputs_ptr[i][0..input_lens[i]];

    const results = gpa.alloc([]TokenId, n) catch return ZTOK_ERR_OUT_OF_MEMORY;
    defer gpa.free(results);
    for (results) |*r| r.* = &.{};

    h.pipeline.encodeBatch(gpa, pool, input_slices, results) catch |e| {
        for (results) |r| if (r.len > 0) gpa.free(r);
        return mapErr(e);
    };

    for (results, 0..) |r, i| {
        if (r.len == 0) {
            out_ids[i] = null;
            out_lens[i] = 0;
            continue;
        }
        const c_buf = allocIdBuf(r.len) orelse {
            for (results[i..]) |rr| if (rr.len > 0) gpa.free(rr);
            for (out_ids[0..i]) |maybe| if (maybe) |q| freeIdBuf(q);
            for (out_ids[0..i]) |*slot| slot.* = null;
            for (out_lens[0..i]) |*slot| slot.* = 0;
            return ZTOK_ERR_OUT_OF_MEMORY;
        };
        @memcpy(c_buf[0..r.len], r);
        gpa.free(r);
        out_ids[i] = c_buf;
        out_lens[i] = r.len;
    }
    return ZTOK_OK;
}

export fn ztok_encode_batch(
    p: ?*const Pipeline,
    inputs: [*]const [*]const u8,
    input_lens: [*]const usize,
    n: usize,
    out_ids: [*]?[*]TokenId,
    out_lens: [*]usize,
    n_workers: u32,
) c_int {
    const pp = p orelse return ZTOK_ERR_INVALID_INPUT;
    const h = handleFromConstPtr(pp);

    const workers: ?u32 = if (n_workers == 0) null else n_workers;
    var pool = BatchPool.init(gpa, workers) catch |e| return mapErr(e);
    defer pool.deinit();

    return runBatch(h, &pool, inputs, input_lens, n, out_ids, out_lens);
}

// --- persistent BatchPool handle ------------------------------------

const BatchPoolHandle = struct {
    pool: BatchPool,
};

export fn ztok_batch_pool_new(n_workers: u32, out_status: ?*c_int) ?*BatchPoolHandle {
    const workers: ?u32 = if (n_workers == 0) null else n_workers;
    const h = gpa.create(BatchPoolHandle) catch {
        setStatus(out_status, ZTOK_ERR_OUT_OF_MEMORY);
        return null;
    };
    h.pool = BatchPool.init(gpa, workers) catch |e| {
        gpa.destroy(h);
        setStatus(out_status, mapErr(e));
        return null;
    };
    setStatus(out_status, ZTOK_OK);
    return h;
}

export fn ztok_batch_pool_free(pool: ?*BatchPoolHandle) void {
    if (pool) |h| {
        h.pool.deinit();
        gpa.destroy(h);
    }
}

export fn ztok_batch_pool_worker_count(pool: ?*const BatchPoolHandle) usize {
    if (pool) |h| return h.pool.workerCount();
    return 0;
}

export fn ztok_encode_batch_pooled(
    p: ?*const Pipeline,
    pool_handle: ?*BatchPoolHandle,
    inputs: [*]const [*]const u8,
    input_lens: [*]const usize,
    n: usize,
    out_ids: [*]?[*]TokenId,
    out_lens: [*]usize,
) c_int {
    const pp = p orelse return ZTOK_ERR_INVALID_INPUT;
    const ph = pool_handle orelse return ZTOK_ERR_INVALID_INPUT;
    return runBatch(handleFromConstPtr(pp), &ph.pool, inputs, input_lens, n, out_ids, out_lens);
}

// --- misc exports ---------------------------------------------------

export fn ztok_ids_free(ids: ?[*]TokenId) void {
    if (ids) |p| freeIdBuf(p);
}

export fn ztok_version() [*:0]const u8 {
    return VERSION;
}

// --- Engram n-gram hashing ------------------------------------------
//
// Deterministic multi-head token-n-gram hashing (see ngram.zig). These
// operate on raw token ids and need no Pipeline handle. Output is
// row-major [position][head] raw u64 hashes; the caller masks each hash
// to its own table width.

// Hash every length-`n` window of `ids` under `heads` hash functions.
// `out` is a CALLER-OWNED buffer of `out_cap` u64 entries — ztok never
// owns it, so there is nothing to free on the ztok side (contrast
// ztok_ngram_hash_batch, whose returned buffers are ztok-owned and must
// go through ztok_u64s_free). On success writes positions*heads hashes
// and sets *out_len to that count.
//
// Sizing/NULL convention differs from ztok_encode: when the required
// count is 0 (stream shorter than one window, or n/heads == 0) we set
// *out_len=0 and return ZTOK_OK even if `out` is null — an empty result
// is success, not a sizing error. Only when need>0 and `out` is null or
// too small do we set *out_len to the required count and return
// BUFFER_TOO_SMALL (nothing written). ztok_encode instead treats a null
// `out` as ALWAYS BUFFER_TOO_SMALL.
export fn ztok_ngram_hash(
    ids: ?[*]const TokenId,
    n_ids: usize,
    n: u32,
    heads: u32,
    out: ?[*]u64,
    out_cap: usize,
    out_len: ?*usize,
) c_int {
    const olp = out_len orelse return ZTOK_ERR_INVALID_INPUT;
    if (n_ids > 0 and ids == null) return ZTOK_ERR_INVALID_INPUT;

    const need = ngram.hashNGramsOutLen(n_ids, n, heads);
    olp.* = need;
    if (need == 0) return ZTOK_OK;

    const buf = out orelse return ZTOK_ERR_BUFFER_TOO_SMALL;
    if (out_cap < need) return ZTOK_ERR_BUFFER_TOO_SMALL;

    _ = ngram.hashNGrams(ids.?[0..n_ids], n, heads, buf[0..out_cap]);
    return ZTOK_OK;
}

// u64 buffers returned by ztok_ngram_hash_batch are prefixed with a
// length header (parallel to the TokenId Header above) so the matching
// free can recover the allocation size.
const U64Header = extern struct { magic: MagicTag, byte_len: usize };
const u64_header_size = std.mem.alignForward(usize, @sizeOf(U64Header), @alignOf(u64));
const u64_buf_align: std.mem.Alignment = .fromByteUnits(@max(@alignOf(U64Header), @alignOf(u64)));

fn u64PayloadFromHeader(h: *U64Header) [*]u64 {
    const raw: [*]u8 = @ptrCast(h);
    return @ptrCast(@alignCast(raw + u64_header_size));
}

fn u64HeaderFromPayload(p: [*]u64) *U64Header {
    const raw: [*]u8 = @ptrCast(p);
    return @ptrCast(@alignCast(raw - u64_header_size));
}

fn allocU64Buf(n: usize) ?[*]u64 {
    if (n == 0) return null;
    const total_bytes = u64_header_size + n * @sizeOf(u64);
    const raw = gpa.alignedAlloc(u8, u64_buf_align, total_bytes) catch return null;
    const hdr: *U64Header = @ptrCast(@alignCast(raw.ptr));
    setMagic(&hdr.magic, u64_buf_magic);
    hdr.byte_len = total_bytes;
    return u64PayloadFromHeader(hdr);
}

fn freeU64Buf(p: [*]u64) void {
    const hdr = u64HeaderFromPayload(p);
    checkMagic(&hdr.magic, u64_buf_magic);
    const raw_ptr: [*]u8 = @ptrCast(hdr);
    const aligned: [*]align(u64_buf_align.toByteUnits()) u8 = @alignCast(raw_ptr);
    gpa.free(aligned[0..hdr.byte_len]);
}

// Worker context for the parallel n-gram hash fan-out: each worker
// hashes one id stream into its own freshly allocated, header-prefixed
// u64 buffer.
const NGramBatchCtx = struct {
    id_arrays: [*]const [*]const TokenId,
    id_lens: [*]const usize,
    out_hashes: [*]?[*]u64,
    out_lens: [*]usize,
    n: u32,
    heads: u32,
    errored: std.atomic.Value(u32),

    pub fn run(c: *NGramBatchCtx, idx: usize, widx: usize) void {
        _ = widx;
        const out_len = ngram.hashNGramsOutLen(c.id_lens[idx], c.n, c.heads);
        if (out_len == 0) {
            c.out_hashes[idx] = null;
            c.out_lens[idx] = 0;
            return;
        }
        const buf = allocU64Buf(out_len) orelse {
            _ = c.errored.fetchAdd(1, .acq_rel);
            c.out_hashes[idx] = null;
            c.out_lens[idx] = 0;
            return;
        };
        _ = ngram.hashNGrams(c.id_arrays[idx][0..c.id_lens[idx]], c.n, c.heads, buf[0..out_len]);
        c.out_hashes[idx] = buf;
        c.out_lens[idx] = out_len;
    }
};

// Hash `n_docs` id streams in parallel across `pool`. On ZTOK_OK each
// `out_hashes[i]` is set to a ZTOK-OWNED, header-prefixed u64 buffer
// holding the row-major hashes for doc i, and `out_lens[i]` to its u64
// count. Each non-null buffer MUST be freed with ztok_u64s_free (NOT
// free()/ztok_ids_free — those expect a different/absent header and would
// corrupt the heap). A stream shorter than one window yields a null
// buffer and length 0.
//
// On any non-OK return every `out_hashes[i]` is null and `out_lens[i]` 0:
// ztok freed everything it allocated, so the caller frees nothing. And
// n_docs==0 returns ZTOK_OK leaving the out arrays UNTOUCHED (the early
// return below runs before the pre-zeroing loop).
export fn ztok_ngram_hash_batch(
    pool_handle: ?*BatchPoolHandle,
    id_arrays: ?[*]const [*]const TokenId,
    id_lens: ?[*]const usize,
    n_docs: usize,
    n: u32,
    heads: u32,
    out_hashes: ?[*]?[*]u64,
    out_lens: ?[*]usize,
) c_int {
    const ph = pool_handle orelse return ZTOK_ERR_INVALID_INPUT;
    const arrays = id_arrays orelse return ZTOK_ERR_INVALID_INPUT;
    const lens = id_lens orelse return ZTOK_ERR_INVALID_INPUT;
    const oh = out_hashes orelse return ZTOK_ERR_INVALID_INPUT;
    const ol = out_lens orelse return ZTOK_ERR_INVALID_INPUT;
    if (n_docs == 0) return ZTOK_OK;

    // Clean initial state so an early-exit worker or an OOM rollback
    // leaves well-defined slots.
    for (0..n_docs) |i| {
        oh[i] = null;
        ol[i] = 0;
    }

    var ctx: NGramBatchCtx = .{
        .id_arrays = arrays,
        .id_lens = lens,
        .out_hashes = oh,
        .out_lens = ol,
        .n = n,
        .heads = heads,
        .errored = .init(0),
    };

    ph.pool.runBatch(NGramBatchCtx, &ctx, n_docs) catch |e| {
        for (oh[0..n_docs]) |maybe| if (maybe) |q| freeU64Buf(q);
        for (oh[0..n_docs]) |*slot| slot.* = null;
        for (ol[0..n_docs]) |*slot| slot.* = 0;
        return mapErr(e);
    };

    if (ctx.errored.load(.acquire) != 0) {
        for (oh[0..n_docs]) |maybe| if (maybe) |q| freeU64Buf(q);
        for (oh[0..n_docs]) |*slot| slot.* = null;
        for (ol[0..n_docs]) |*slot| slot.* = 0;
        return ZTOK_ERR_OUT_OF_MEMORY;
    }

    return ZTOK_OK;
}

// Free a u64 buffer returned by ztok_ngram_hash_batch (only those — never
// a caller's ztok_ngram_hash buffer, and never via free()/ztok_ids_free).
// Null is a no-op.
export fn ztok_u64s_free(hashes: ?[*]u64) void {
    if (hashes) |p| freeU64Buf(p);
}

// --- chunking --------------------------------------------------------
//
// Expose chunk.chunkText to C: split text into overlapping token
// windows for embedding / late-chunking pipelines. Each output record
// carries the chunk's token ids plus its byte- and token-index ranges
// in the original input. `ids` points at a ztok-allocated,
// header-prefixed buffer (parallel to ztok_ids_free's layout); free the
// whole array — and every record's `ids` — with ztok_chunks_free.
//
// boundary mirrors chunk.Boundary: 0=token, 1=codepoint, 2=word,
// 3=word_dict, 4=sentence, 5=paragraph.
const CChunk = extern struct {
    ids: ?[*]TokenId,
    ids_len: usize,
    byte_start: u32,
    byte_end: u32,
    token_start: u32,
    token_end: u32,
};

fn boundaryFromKind(k: u32) ?chunk.Boundary {
    return switch (k) {
        0 => .token,
        1 => .codepoint,
        2 => .word,
        3 => .word_dict,
        4 => .sentence,
        5 => .paragraph,
        else => null,
    };
}

// Chunk `text` into windows of at most `max_tokens` tokens with
// `overlap` tokens shared between neighbors. `out_chunks` is a
// caller-owned buffer of `out_cap` records. On success writes one
// record per chunk and sets *out_len to the chunk count. If `out_chunks`
// is null or too small, sets *out_len to the required count and returns
// BUFFER_TOO_SMALL (nothing is written, no ids allocated). Empty input
// (or a zero-token encoding) yields zero chunks and ZTOK_OK.
export fn ztok_chunk(
    p: ?*const Pipeline,
    text: [*]const u8,
    text_len: usize,
    max_tokens: u32,
    overlap: u32,
    boundary: u32,
    out_chunks: ?[*]CChunk,
    out_cap: usize,
    out_len: ?*usize,
) c_int {
    const pp = p orelse return ZTOK_ERR_INVALID_INPUT;
    const olp = out_len orelse return ZTOK_ERR_INVALID_INPUT;
    const bnd = boundaryFromKind(boundary) orelse return ZTOK_ERR_INVALID_INPUT;
    if (max_tokens == 0 or overlap >= max_tokens) return ZTOK_ERR_INVALID_INPUT;
    const h = handleFromConstPtr(pp);

    var result = chunk.chunkText(gpa, h.pipeline, text[0..text_len], .{
        .max_tokens = max_tokens,
        .overlap_tokens = overlap,
        .boundary = bnd,
    }) catch |e| switch (e) {
        error.OverlapTooLarge, error.InvalidMaxTokens, error.InvalidStride => return ZTOK_ERR_INVALID_INPUT,
        else => return mapErr(e),
    };
    defer result.deinit();

    const n = result.chunks.len;
    olp.* = n;
    if (n == 0) return ZTOK_OK;

    const out = out_chunks orelse return ZTOK_ERR_BUFFER_TOO_SMALL;
    if (out_cap < n) return ZTOK_ERR_BUFFER_TOO_SMALL;

    // Copy each chunk's ids into its own header-prefixed buffer. On any
    // OOM, roll back every buffer allocated so far and clear all slots.
    for (result.chunks, 0..) |ch, i| {
        var buf: ?[*]TokenId = null;
        if (ch.ids.len > 0) {
            buf = allocIdBuf(ch.ids.len) orelse {
                for (out[0..i]) |*rec| if (rec.ids) |q| freeIdBuf(q);
                for (out[0..n]) |*rec| {
                    rec.ids = null;
                    rec.ids_len = 0;
                }
                return ZTOK_ERR_OUT_OF_MEMORY;
            };
            @memcpy(buf.?[0..ch.ids.len], ch.ids);
        }
        out[i] = .{
            .ids = buf,
            .ids_len = ch.ids.len,
            .byte_start = ch.byte_start,
            .byte_end = ch.byte_end,
            .token_start = ch.token_start,
            .token_end = ch.token_end,
        };
    }
    return ZTOK_OK;
}

// Free an array of `n` chunk records written by ztok_chunk, releasing
// every record's `ids` buffer. The `chunks` array itself is caller-owned
// (we never allocated it), so only the id buffers are freed.
export fn ztok_chunks_free(chunks: ?[*]CChunk, n: usize) void {
    const c = chunks orelse return;
    for (c[0..n]) |*rec| {
        if (rec.ids) |p| freeIdBuf(p);
        rec.ids = null;
        rec.ids_len = 0;
    }
}

// --- fingerprint -----------------------------------------------------
//
// Compute the tokenizer fingerprint — a 32-byte SHA-256 digest over
// the pipeline's behavior on a fixed canonical input set. Two pipelines
// with the same fingerprint encode any input to bit-identical ids.
//
// Returns ZTOK_OK on success and writes 32 bytes into `out_32`. Returns
// ZTOK_ERR_INVALID_INPUT if the handle or output pointer is NULL.
// Internal errors (OOM, etc.) collapse to ZTOK_ERR_INTERNAL.

const fingerprint_mod = @import("fingerprint.zig");

export fn ztok_fingerprint(handle: ?*anyopaque, out_32: ?*[32]u8) c_int {
    const hp = handle orelse return ZTOK_ERR_INVALID_INPUT;
    const op = out_32 orelse return ZTOK_ERR_INVALID_INPUT;
    const pipe_ptr: *const Pipeline = @ptrCast(@alignCast(hp));
    const h = handleFromConstPtr(pipe_ptr);
    const fp = fingerprint_mod.computeFingerprint(&h.pipeline, gpa) catch |e| return mapErr(e);
    op.* = fp;
    return ZTOK_OK;
}

// --- auto-detect -----------------------------------------------------
//
// Best-effort format sniffer. We map the Zig `auto_detect.Format` enum
// to the stable C `ztok_format` integer codes. Errors (missing file,
// unreadable bytes, etc.) collapse to ZTOK_FORMAT_UNKNOWN — auto-detect
// is purely a "tell me what you think this is" call, never the place to
// surface a real I/O error.

const ZTOK_FORMAT_UNKNOWN: c_uint = 0;
const ZTOK_FORMAT_TIKTOKEN: c_uint = 1;
const ZTOK_FORMAT_HF_JSON: c_uint = 2;
const ZTOK_FORMAT_SP_MODEL: c_uint = 3;
const ZTOK_FORMAT_ZTM: c_uint = 4;
const ZTOK_FORMAT_TEKKEN: c_uint = 5;
const ZTOK_FORMAT_RWKV: c_uint = 6;

fn formatToC(f: auto_detect.Format) c_uint {
    return switch (f) {
        .unknown => ZTOK_FORMAT_UNKNOWN,
        .tiktoken => ZTOK_FORMAT_TIKTOKEN,
        .hf_json => ZTOK_FORMAT_HF_JSON,
        .sentencepiece => ZTOK_FORMAT_SP_MODEL,
        .ztm => ZTOK_FORMAT_ZTM,
        .tekken => ZTOK_FORMAT_TEKKEN,
        .rwkv => ZTOK_FORMAT_RWKV,
    };
}

export fn ztok_auto_detect(path_c: ?[*:0]const u8) c_uint {
    const path_z = path_c orelse return ZTOK_FORMAT_UNKNOWN;
    const path = std.mem.span(path_z);
    const fmt = auto_detect.detectFile(path) catch return ZTOK_FORMAT_UNKNOWN;
    return formatToC(fmt);
}

// --- streaming encode -----------------------------------------------
//
// Wraps `stream.StreamEncoder` behind an opaque handle. Each `feed` /
// `finish` call materializes any newly-emitted ids into a malloc'd
// buffer using the same Header-prefix convention as `ztok_encode_batch`
// (so callers free with `ztok_ids_free`).
//
// Carry semantics carry over from `StreamEncoder` verbatim: the encoder
// defers a trailing partial codepoint or pre-tokenizer span up to a
// 1 MiB soft cap; past that, it force-cuts at a codepoint boundary.

const StreamHandle = struct {
    encoder: StreamEncoder,
    ids: std.ArrayList(TokenId),
};

fn drainStreamIds(
    h: *StreamHandle,
    out_ids: ?*?[*]TokenId,
    out_n_ids: ?*usize,
) c_int {
    const p_ids = out_ids orelse return ZTOK_ERR_INVALID_INPUT;
    const p_n = out_n_ids orelse return ZTOK_ERR_INVALID_INPUT;

    const n = h.ids.items.len;
    if (n == 0) {
        p_ids.* = null;
        p_n.* = 0;
        return ZTOK_OK;
    }
    const buf_ptr = allocIdBuf(n) orelse {
        p_ids.* = null;
        p_n.* = 0;
        return ZTOK_ERR_OUT_OF_MEMORY;
    };
    @memcpy(buf_ptr[0..n], h.ids.items);
    h.ids.clearRetainingCapacity();
    p_ids.* = buf_ptr;
    p_n.* = n;
    return ZTOK_OK;
}

export fn ztok_stream_new(
    p: ?*const Pipeline,
    out_status: ?*c_int,
) ?*StreamHandle {
    const pp = p orelse {
        setStatus(out_status, ZTOK_ERR_INVALID_INPUT);
        return null;
    };
    const h_in = handleFromConstPtr(pp);

    const sh = gpa.create(StreamHandle) catch {
        setStatus(out_status, ZTOK_ERR_OUT_OF_MEMORY);
        return null;
    };
    sh.* = .{
        .encoder = StreamEncoder.init(gpa, &h_in.pipeline),
        .ids = .empty,
    };
    setStatus(out_status, ZTOK_OK);
    return sh;
}

export fn ztok_stream_free(s: ?*StreamHandle) void {
    if (s) |sh| {
        sh.encoder.deinit();
        sh.ids.deinit(gpa);
        gpa.destroy(sh);
    }
}

export fn ztok_stream_feed(
    s: ?*StreamHandle,
    bytes: ?[*]const u8,
    n_bytes: usize,
    out_ids: ?*?[*]TokenId,
    out_n_ids: ?*usize,
) c_int {
    const sh = s orelse return ZTOK_ERR_INVALID_INPUT;
    // Reset the id accumulator so each feed reports only what was newly
    // emitted by THIS call.
    sh.ids.clearRetainingCapacity();
    if (n_bytes > 0) {
        const bp = bytes orelse return ZTOK_ERR_INVALID_INPUT;
        sh.encoder.feed(bp[0..n_bytes], &sh.ids) catch |e| return mapErr(e);
    }
    return drainStreamIds(sh, out_ids, out_n_ids);
}

export fn ztok_stream_finish(
    s: ?*StreamHandle,
    out_ids: ?*?[*]TokenId,
    out_n_ids: ?*usize,
) c_int {
    const sh = s orelse return ZTOK_ERR_INVALID_INPUT;
    sh.ids.clearRetainingCapacity();
    sh.encoder.finish(&sh.ids) catch |e| return mapErr(e);
    return drainStreamIds(sh, out_ids, out_n_ids);
}

// =====================================================================
// Tests
// =====================================================================

const testing = std.testing;

test {
    _ = monster_io;
}

test "pipeline_new + free identity/byte_id" {
    var status: c_int = -1;
    const cfg: Config = .{ .normalizer = 0, .pre_tokenizer = 0, .model = 0, .decoder = 0 };
    const p = ztok_pipeline_new(&cfg, &status);
    try testing.expect(p != null);
    try testing.expectEqual(ZTOK_OK, status);
    ztok_pipeline_free(p);
}

test "pipeline_new rejects unknown kind" {
    var status: c_int = ZTOK_OK;
    const cfg: Config = .{ .normalizer = 0, .pre_tokenizer = 0, .model = 99, .decoder = 0 };
    const p = ztok_pipeline_new(&cfg, &status);
    try testing.expect(p == null);
    try testing.expectEqual(ZTOK_ERR_INVALID_INPUT, status);
}

test "encode buffer too small reports required size" {
    const cfg: Config = .{ .normalizer = 0, .pre_tokenizer = 0, .model = 0, .decoder = 0 };
    const p = ztok_pipeline_new(&cfg, null);
    defer ztok_pipeline_free(p);

    const input = "hello";
    var out_len: usize = 0;
    const rc = ztok_encode(p, input.ptr, input.len, null, 0, &out_len);
    try testing.expectEqual(ZTOK_ERR_BUFFER_TOO_SMALL, rc);
    try testing.expectEqual(@as(usize, 5), out_len);
}

test "encode round-trip" {
    const cfg: Config = .{ .normalizer = 0, .pre_tokenizer = 0, .model = 0, .decoder = 0 };
    const p = ztok_pipeline_new(&cfg, null);
    defer ztok_pipeline_free(p);

    const input = "hi";
    var ids: [8]TokenId = undefined;
    var n: usize = 0;
    try testing.expectEqual(ZTOK_OK, ztok_encode(p, input.ptr, input.len, &ids, ids.len, &n));
    try testing.expectEqual(@as(usize, 2), n);
    try testing.expectEqualSlices(TokenId, &.{ 0x68, 0x69 }, ids[0..n]);

    var dec: [8]u8 = undefined;
    var dn: usize = 0;
    try testing.expectEqual(ZTOK_OK, ztok_decode(p, ids[0..n].ptr, n, &dec, dec.len, &dn));
    try testing.expectEqualStrings("hi", dec[0..dn]);
}

test "encode_batch parallel" {
    const cfg: Config = .{ .normalizer = 0, .pre_tokenizer = 0, .model = 0, .decoder = 0 };
    const p = ztok_pipeline_new(&cfg, null);
    defer ztok_pipeline_free(p);

    const a = "ab";
    const b = "cde";
    const c = "f";
    const ptrs = [_][*]const u8{ a.ptr, b.ptr, c.ptr };
    const lens = [_]usize{ a.len, b.len, c.len };

    var out_ids: [3]?[*]TokenId = .{ null, null, null };
    var out_lens: [3]usize = .{ 0, 0, 0 };

    try testing.expectEqual(
        ZTOK_OK,
        ztok_encode_batch(p, &ptrs, &lens, 3, &out_ids, &out_lens, 2),
    );

    try testing.expectEqual(@as(usize, 2), out_lens[0]);
    try testing.expectEqual(@as(usize, 3), out_lens[1]);
    try testing.expectEqual(@as(usize, 1), out_lens[2]);
    try testing.expectEqualSlices(TokenId, &.{ 'a', 'b' }, out_ids[0].?[0..out_lens[0]]);
    try testing.expectEqualSlices(TokenId, &.{ 'c', 'd', 'e' }, out_ids[1].?[0..out_lens[1]]);
    try testing.expectEqualSlices(TokenId, &.{'f'}, out_ids[2].?[0..out_lens[2]]);

    for (out_ids) |maybe| ztok_ids_free(maybe);
}

test "version is non-empty C string" {
    const v = ztok_version();
    const slice = std.mem.span(v);
    try testing.expect(slice.len > 0);
}

// --- new tests (Wave C) ---------------------------------------------

test "batch_pool create/free/worker_count" {
    var status: c_int = -1;
    const pool = ztok_batch_pool_new(3, &status);
    try testing.expect(pool != null);
    try testing.expectEqual(ZTOK_OK, status);
    try testing.expectEqual(@as(usize, 3), ztok_batch_pool_worker_count(pool));
    ztok_batch_pool_free(pool);

    // 0 = auto -> at least 1 worker.
    const auto_pool = ztok_batch_pool_new(0, null);
    try testing.expect(auto_pool != null);
    try testing.expect(ztok_batch_pool_worker_count(auto_pool) >= 1);
    ztok_batch_pool_free(auto_pool);
}

test "encode_batch_pooled equals encode_batch" {
    const cfg: Config = .{ .normalizer = 0, .pre_tokenizer = 0, .model = 0, .decoder = 0 };
    const p = ztok_pipeline_new(&cfg, null);
    defer ztok_pipeline_free(p);

    const a = "alpha";
    const b = "bravo";
    const c = "charlie";
    const ptrs = [_][*]const u8{ a.ptr, b.ptr, c.ptr };
    const lens = [_]usize{ a.len, b.len, c.len };

    var ids_a: [3]?[*]TokenId = .{ null, null, null };
    var lens_a: [3]usize = .{ 0, 0, 0 };
    try testing.expectEqual(
        ZTOK_OK,
        ztok_encode_batch(p, &ptrs, &lens, 3, &ids_a, &lens_a, 2),
    );
    defer for (ids_a) |m| ztok_ids_free(m);

    const pool = ztok_batch_pool_new(2, null);
    defer ztok_batch_pool_free(pool);

    var ids_b: [3]?[*]TokenId = .{ null, null, null };
    var lens_b: [3]usize = .{ 0, 0, 0 };
    try testing.expectEqual(
        ZTOK_OK,
        ztok_encode_batch_pooled(p, pool, &ptrs, &lens, 3, &ids_b, &lens_b),
    );
    defer for (ids_b) |m| ztok_ids_free(m);

    for (0..3) |i| {
        try testing.expectEqual(lens_a[i], lens_b[i]);
        try testing.expectEqualSlices(
            TokenId,
            ids_a[i].?[0..lens_a[i]],
            ids_b[i].?[0..lens_b[i]],
        );
    }
}

test "extended normalizer kind dispatch (NFC vs NFD)" {
    // Composed "é" (U+00E9) vs combining form "e" + U+0301.
    const combining = "e\xCC\x81";

    const cfg_nfc: Config = .{ .normalizer = 1, .pre_tokenizer = 0, .model = 0, .decoder = 0 };
    const p_nfc = ztok_pipeline_new(&cfg_nfc, null);
    defer ztok_pipeline_free(p_nfc);
    var nfc_ids: [16]TokenId = undefined;
    var nfc_n: usize = 0;
    try testing.expectEqual(
        ZTOK_OK,
        ztok_encode(p_nfc, combining.ptr, combining.len, &nfc_ids, nfc_ids.len, &nfc_n),
    );
    try testing.expect(nfc_n > 0);

    const cfg_nfd: Config = .{ .normalizer = 2, .pre_tokenizer = 0, .model = 0, .decoder = 0 };
    const p_nfd = ztok_pipeline_new(&cfg_nfd, null);
    defer ztok_pipeline_free(p_nfd);
    var nfd_ids: [16]TokenId = undefined;
    var nfd_n: usize = 0;
    try testing.expectEqual(
        ZTOK_OK,
        ztok_encode(p_nfd, combining.ptr, combining.len, &nfd_ids, nfd_ids.len, &nfd_n),
    );
    try testing.expect(nfd_n > 0);

    // NFC composes e+0301 -> é (2 bytes); NFD keeps 3 bytes. So the
    // two byte_id encodings must differ.
    try testing.expect(nfc_n != nfd_n or !std.mem.eql(TokenId, nfc_ids[0..nfc_n], nfd_ids[0..nfd_n]));
}

test "pipeline_new_bpe_from_tiktoken round-trips" {
    // Build a minimal tiktoken vocab covering all 256 single bytes plus
    // a few merges for "hello" and " world", then load via the new
    // file-based constructor.
    const a = testing.allocator;
    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(a);

    const b64 = std.base64.standard.Encoder;
    var enc_buf: [16]u8 = undefined;
    var rank: u32 = 0;
    var b: u32 = 0;
    while (b < 256) : (b += 1) {
        const byte: [1]u8 = .{@intCast(b)};
        const encoded = b64.encode(&enc_buf, &byte);
        try src.print(a, "{s} {d}\n", .{ encoded, rank });
        rank += 1;
    }
    const extras = [_][]const u8{
        "he",     "hel", "hell", "hello",
        " w",     " wo", " wor", " worl",
        " world",
    };
    for (extras) |bytes| {
        const encoded = b64.encode(&enc_buf, bytes);
        try src.print(a, "{s} {d}\n", .{ encoded, rank });
        rank += 1;
    }

    const path = "/tmp/ztok_c_api_tiktoken_test.txt";
    const io = std.Io.Threaded.global_single_threaded.io();
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = src.items });
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    var status: c_int = -1;
    const p = ztok_pipeline_new_bpe_from_tiktoken(path, null, &status);
    try testing.expect(p != null);
    try testing.expectEqual(ZTOK_OK, status);
    defer ztok_pipeline_free(p);

    const input = "hello world";
    var ids: [32]TokenId = undefined;
    var n: usize = 0;
    try testing.expectEqual(
        ZTOK_OK,
        ztok_encode(p, input.ptr, input.len, &ids, ids.len, &n),
    );
    // cl100k pre-tokenizer splits "hello world" into ["hello", " world"],
    // each merges to its single longest vocab entry.
    try testing.expectEqual(@as(usize, 2), n);

    var out: [32]u8 = undefined;
    var dn: usize = 0;
    try testing.expectEqual(ZTOK_OK, ztok_decode(p, ids[0..n].ptr, n, &out, out.len, &dn));
    try testing.expectEqualStrings("hello world", out[0..dn]);

    // Now exercise encode_batch_pooled with a real BPE model.
    const pool = ztok_batch_pool_new(2, null);
    defer ztok_batch_pool_free(pool);

    const s0 = "hello";
    const s1 = " world";
    const s2 = "hello world";
    const ptrs = [_][*]const u8{ s0.ptr, s1.ptr, s2.ptr };
    const ilens = [_]usize{ s0.len, s1.len, s2.len };
    var out_ids: [3]?[*]TokenId = .{ null, null, null };
    var out_lens: [3]usize = .{ 0, 0, 0 };
    try testing.expectEqual(
        ZTOK_OK,
        ztok_encode_batch_pooled(p, pool, &ptrs, &ilens, 3, &out_ids, &out_lens),
    );
    defer for (out_ids) |m| ztok_ids_free(m);
}

// --- Wave-perf 4 tests: direct-write batch path -----------------------

test "encode_batch_pooled empty inputs return null + zero len" {
    // n=0 should be ZTOK_OK with no work. Also covers the all-zero-len
    // branch where pre-alloc skips slots.
    const cfg: Config = .{ .normalizer = 0, .pre_tokenizer = 0, .model = 0, .decoder = 0 };
    const p = ztok_pipeline_new(&cfg, null);
    defer ztok_pipeline_free(p);
    const pool = ztok_batch_pool_new(2, null);
    defer ztok_batch_pool_free(pool);

    try testing.expectEqual(
        ZTOK_OK,
        ztok_encode_batch_pooled(p, pool, @ptrFromInt(@alignOf([*]const u8)), @ptrFromInt(@alignOf(usize)), 0, @ptrFromInt(@alignOf(?[*]TokenId)), @ptrFromInt(@alignOf(usize))),
    );

    // Now batch of 3 with one zero-length input in the middle.
    const a = "hi";
    const empty = "";
    const c = "lo";
    const ptrs = [_][*]const u8{ a.ptr, empty.ptr, c.ptr };
    const lens = [_]usize{ a.len, empty.len, c.len };
    var out_ids: [3]?[*]TokenId = .{ null, null, null };
    var out_lens: [3]usize = .{ 0, 0, 0 };
    try testing.expectEqual(
        ZTOK_OK,
        ztok_encode_batch_pooled(p, pool, &ptrs, &lens, 3, &out_ids, &out_lens),
    );
    defer for (out_ids) |m| ztok_ids_free(m);

    try testing.expectEqual(@as(usize, 2), out_lens[0]);
    try testing.expectEqual(@as(usize, 0), out_lens[1]);
    try testing.expect(out_ids[1] == null);
    try testing.expectEqual(@as(usize, 2), out_lens[2]);
}

test "encode_batch_pooled bit-identical to pipeline.encodeBatch" {
    // The direct-write fast path must produce the same ids as the
    // Zig-side encodeBatch on a realistic input mix.
    const a_alloc = testing.allocator;
    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(a_alloc);
    const b64 = std.base64.standard.Encoder;
    var enc_buf: [16]u8 = undefined;
    var rank: u32 = 0;
    var b: u32 = 0;
    while (b < 256) : (b += 1) {
        const byte: [1]u8 = .{@intCast(b)};
        try src.print(a_alloc, "{s} {d}\n", .{ b64.encode(&enc_buf, &byte), rank });
        rank += 1;
    }
    const extras = [_][]const u8{ "he", "hel", "hello", " w", " wo", " wor", " world", "th", "the", " th", " the" };
    for (extras) |bytes| {
        try src.print(a_alloc, "{s} {d}\n", .{ b64.encode(&enc_buf, bytes), rank });
        rank += 1;
    }
    const path = "/tmp/ztok_c_api_perf4_test.txt";
    const io = std.Io.Threaded.global_single_threaded.io();
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = src.items });
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    const p = ztok_pipeline_new_bpe_from_tiktoken(path, null, null);
    try testing.expect(p != null);
    defer ztok_pipeline_free(p);

    const pool = ztok_batch_pool_new(4, null);
    defer ztok_batch_pool_free(pool);

    const inputs_z = [_][]const u8{
        "hello world",
        "the quick brown fox",
        "hello there hello world",
        "  the  ",
    };
    const ptrs = [_][*]const u8{ inputs_z[0].ptr, inputs_z[1].ptr, inputs_z[2].ptr, inputs_z[3].ptr };
    const lens = [_]usize{ inputs_z[0].len, inputs_z[1].len, inputs_z[2].len, inputs_z[3].len };

    // C path: direct-write into pre-allocated buffers.
    var c_ids: [4]?[*]TokenId = .{ null, null, null, null };
    var c_lens: [4]usize = .{ 0, 0, 0, 0 };
    try testing.expectEqual(
        ZTOK_OK,
        ztok_encode_batch_pooled(p, pool, &ptrs, &lens, 4, &c_ids, &c_lens),
    );
    defer for (c_ids) |m| ztok_ids_free(m);

    // Zig path: pull the inner pipeline out of the handle and encode
    // through its Pipeline.encodeBatch directly.
    const handle = handleFromConstPtr(@ptrCast(p.?));
    var z_pool = try BatchPool.init(testing.allocator, 4);
    defer z_pool.deinit();
    var z_results: [4][]TokenId = undefined;
    try handle.pipeline.encodeBatch(testing.allocator, &z_pool, &inputs_z, &z_results);
    defer for (z_results) |r| testing.allocator.free(r);

    for (0..4) |i| {
        try testing.expectEqual(z_results[i].len, c_lens[i]);
        if (c_lens[i] == 0) continue;
        try testing.expectEqualSlices(TokenId, z_results[i], c_ids[i].?[0..c_lens[i]]);
    }
}

test "ids_free handles per-input buffer with header intact after batch" {
    // Direct-write path stores Header just before the returned pointer
    // and ztok_ids_free walks back through it. Verify several frees in
    // any order plus interspersed re-encodes don't corrupt state.
    const cfg: Config = .{ .normalizer = 0, .pre_tokenizer = 0, .model = 0, .decoder = 0 };
    const p = ztok_pipeline_new(&cfg, null);
    defer ztok_pipeline_free(p);
    const pool = ztok_batch_pool_new(2, null);
    defer ztok_batch_pool_free(pool);

    const inputs = [_][]const u8{ "aaa", "bbbb", "ccccc", "dd" };
    const ptrs = [_][*]const u8{ inputs[0].ptr, inputs[1].ptr, inputs[2].ptr, inputs[3].ptr };
    const lens = [_]usize{ inputs[0].len, inputs[1].len, inputs[2].len, inputs[3].len };

    var k: usize = 0;
    while (k < 4) : (k += 1) {
        var out_ids: [4]?[*]TokenId = .{ null, null, null, null };
        var out_lens: [4]usize = .{ 0, 0, 0, 0 };
        try testing.expectEqual(
            ZTOK_OK,
            ztok_encode_batch_pooled(p, pool, &ptrs, &lens, 4, &out_ids, &out_lens),
        );
        try testing.expectEqual(@as(usize, 3), out_lens[0]);
        try testing.expectEqual(@as(usize, 4), out_lens[1]);
        try testing.expectEqual(@as(usize, 5), out_lens[2]);
        try testing.expectEqual(@as(usize, 2), out_lens[3]);
        // Free in reverse + middle-first orders alternately.
        if (k % 2 == 0) {
            ztok_ids_free(out_ids[3]);
            ztok_ids_free(out_ids[0]);
            ztok_ids_free(out_ids[2]);
            ztok_ids_free(out_ids[1]);
        } else {
            ztok_ids_free(out_ids[2]);
            ztok_ids_free(out_ids[1]);
            ztok_ids_free(out_ids[3]);
            ztok_ids_free(out_ids[0]);
        }
    }
}

// --- post-1.18 agent C: auto_detect + streaming C ABI -----------------

test "ztok_auto_detect returns UNKNOWN for null path" {
    try testing.expectEqual(ZTOK_FORMAT_UNKNOWN, ztok_auto_detect(null));
}

test "ztok_auto_detect returns UNKNOWN for a bogus path" {
    try testing.expectEqual(
        ZTOK_FORMAT_UNKNOWN,
        ztok_auto_detect("/tmp/this_file_definitely_does_not_exist_ztok_test.bin"),
    );
}

test "ztok_auto_detect recognises every supported format" {
    const io = std.Io.Threaded.global_single_threaded.io();
    const cwd = std.Io.Dir.cwd();

    // tiktoken
    const tt_path = "/tmp/ztok_c_api_auto_detect_test.tiktoken";
    try cwd.writeFile(io, .{ .sub_path = tt_path, .data = "aGVsbG8= 0\n" });
    defer cwd.deleteFile(io, tt_path) catch {};
    try testing.expectEqual(ZTOK_FORMAT_TIKTOKEN, ztok_auto_detect(tt_path));

    // hf_json
    const hf_path = "/tmp/ztok_c_api_auto_detect_test.json";
    try cwd.writeFile(io, .{ .sub_path = hf_path, .data = "{\"model\":{}}" });
    defer cwd.deleteFile(io, hf_path) catch {};
    try testing.expectEqual(ZTOK_FORMAT_HF_JSON, ztok_auto_detect(hf_path));

    // sentencepiece (wire-tag + tiny payload)
    const sp_path = "/tmp/ztok_c_api_auto_detect_test.spmodel";
    try cwd.writeFile(io, .{ .sub_path = sp_path, .data = "\x0A\x05<unk>" });
    defer cwd.deleteFile(io, sp_path) catch {};
    try testing.expectEqual(ZTOK_FORMAT_SP_MODEL, ztok_auto_detect(sp_path));

    // ztm (ZTM\x01 magic plus zero-padding the rest of the header)
    const zm_path = "/tmp/ztok_c_api_auto_detect_test.ztm";
    var zm_buf: [monster_io.HEADER_SIZE]u8 = @splat(0);
    @memcpy(zm_buf[0..monster_io.MAGIC.len], &monster_io.MAGIC);
    try cwd.writeFile(io, .{ .sub_path = zm_path, .data = &zm_buf });
    defer cwd.deleteFile(io, zm_path) catch {};
    try testing.expectEqual(ZTOK_FORMAT_ZTM, ztok_auto_detect(zm_path));
}

test "ztok_stream feed+finish equals single-shot ztok_encode" {
    const cfg: Config = .{ .normalizer = 0, .pre_tokenizer = 0, .model = 0, .decoder = 0 };
    const p = ztok_pipeline_new(&cfg, null);
    defer ztok_pipeline_free(p);

    const input = "hello world";

    // Single-shot reference.
    var ref_buf: [32]TokenId = undefined;
    var ref_n: usize = 0;
    try testing.expectEqual(
        ZTOK_OK,
        ztok_encode(p, input.ptr, input.len, &ref_buf, ref_buf.len, &ref_n),
    );

    // Streaming.
    var status: c_int = -1;
    const sh = ztok_stream_new(p, &status);
    try testing.expect(sh != null);
    try testing.expectEqual(ZTOK_OK, status);
    defer ztok_stream_free(sh);

    var got_ids: std.ArrayList(TokenId) = .empty;
    defer got_ids.deinit(testing.allocator);

    var feed_ids: ?[*]TokenId = null;
    var feed_n: usize = 0;
    try testing.expectEqual(
        ZTOK_OK,
        ztok_stream_feed(sh, input.ptr, input.len, &feed_ids, &feed_n),
    );
    if (feed_n > 0) {
        try got_ids.appendSlice(testing.allocator, feed_ids.?[0..feed_n]);
        ztok_ids_free(feed_ids);
    }

    var fin_ids: ?[*]TokenId = null;
    var fin_n: usize = 0;
    try testing.expectEqual(
        ZTOK_OK,
        ztok_stream_finish(sh, &fin_ids, &fin_n),
    );
    if (fin_n > 0) {
        try got_ids.appendSlice(testing.allocator, fin_ids.?[0..fin_n]);
        ztok_ids_free(fin_ids);
    }

    try testing.expectEqualSlices(TokenId, ref_buf[0..ref_n], got_ids.items);
}

test "ztok_stream 3-chunk feed matches single-shot encode" {
    const cfg: Config = .{ .normalizer = 0, .pre_tokenizer = 0, .model = 0, .decoder = 0 };
    const p = ztok_pipeline_new(&cfg, null);
    defer ztok_pipeline_free(p);

    const input = "the quick brown fox jumps over the lazy dog";

    var ref_buf: [128]TokenId = undefined;
    var ref_n: usize = 0;
    try testing.expectEqual(
        ZTOK_OK,
        ztok_encode(p, input.ptr, input.len, &ref_buf, ref_buf.len, &ref_n),
    );

    const sh = ztok_stream_new(p, null);
    defer ztok_stream_free(sh);

    var got_ids: std.ArrayList(TokenId) = .empty;
    defer got_ids.deinit(testing.allocator);

    const cuts = [_]usize{ 15, 30, input.len };
    var prev: usize = 0;
    for (cuts) |c| {
        var fids: ?[*]TokenId = null;
        var fn_: usize = 0;
        try testing.expectEqual(
            ZTOK_OK,
            ztok_stream_feed(sh, input[prev..].ptr, c - prev, &fids, &fn_),
        );
        if (fn_ > 0) {
            try got_ids.appendSlice(testing.allocator, fids.?[0..fn_]);
            ztok_ids_free(fids);
        }
        prev = c;
    }
    var fin_ids: ?[*]TokenId = null;
    var fin_n: usize = 0;
    try testing.expectEqual(ZTOK_OK, ztok_stream_finish(sh, &fin_ids, &fin_n));
    if (fin_n > 0) {
        try got_ids.appendSlice(testing.allocator, fin_ids.?[0..fin_n]);
        ztok_ids_free(fin_ids);
    }
    try testing.expectEqualSlices(TokenId, ref_buf[0..ref_n], got_ids.items);
}

test "ztok_stream_free cleans up cleanly (smoke)" {
    // The handle's StreamEncoder owns an ArrayList carry and a
    // ScratchArena; the StreamHandle owns a second ArrayList for ids.
    // Round-trip a few feeds, then free — leaks would surface as
    // testing-allocator failures in adjacent tests, but the c_allocator
    // path used by the C ABI doesn't track them. The point of this
    // test is to exercise the lifecycle as a whole and assert no panic.
    const cfg: Config = .{ .normalizer = 0, .pre_tokenizer = 0, .model = 0, .decoder = 0 };
    const p = ztok_pipeline_new(&cfg, null);
    defer ztok_pipeline_free(p);

    var k: usize = 0;
    while (k < 4) : (k += 1) {
        const sh = ztok_stream_new(p, null);
        try testing.expect(sh != null);
        const data = "abcdefg";
        var fids: ?[*]TokenId = null;
        var fn_: usize = 0;
        try testing.expectEqual(
            ZTOK_OK,
            ztok_stream_feed(sh, data.ptr, data.len, &fids, &fn_),
        );
        if (fids) |q| ztok_ids_free(q);

        var fin_ids: ?[*]TokenId = null;
        var fin_n: usize = 0;
        try testing.expectEqual(ZTOK_OK, ztok_stream_finish(sh, &fin_ids, &fin_n));
        if (fin_ids) |q| ztok_ids_free(q);

        ztok_stream_free(sh);
    }
}
