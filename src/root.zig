//! ztok — a multithreaded, data-oriented tokenizer toolkit in Zig.
//!
//! Public surface is intentionally small. Build a `Pipeline` from a
//! `Normalizer`, `PreTokenizer`, `Model`, and `Decoder`, then call
//! `encode`, `encodeBatch`, or `decode`.

const std = @import("std");

/// Project version, parsed from `build.zig.zon` at build time and
/// exposed via `addOptions`. Single source of truth — c_api.zig and
/// main.zig both read this so we never drift again.
pub const VERSION: []const u8 = @import("build_options").version;

// Pipeline stages
pub const token = @import("token.zig");
pub const vocab = @import("vocab.zig");
pub const normalizer = @import("normalizer.zig");
pub const pretok = @import("pretok.zig");
pub const model = @import("model.zig");
pub const decoder = @import("decoder.zig");
pub const pipeline = @import("pipeline.zig");
pub const pretoken_cache = @import("pretoken_cache.zig");
pub const fingerprint = @import("fingerprint.zig");
pub const thread_pool = @import("thread_pool.zig");
pub const connection_pool = @import("connection_pool.zig");
pub const trace = @import("trace.zig");
pub const ngram = @import("ngram.zig");
pub const merge_graph = @import("merge_graph.zig");
pub const superposition = @import("superposition.zig");
pub const semantic_superposition = @import("semantic_superposition.zig");
pub const semantic_exchange = @import("semantic_exchange.zig");

// Models
pub const bpe = @import("bpe.zig");
pub const simd_min = @import("simd_min.zig");
pub const simd_bytes = @import("simd_bytes.zig");
pub const asm_normalizer = @import("asm_normalizer.zig");
pub const bpe_heap = @import("bpe_heap.zig");
pub const unigram = @import("unigram.zig");
pub const wordpiece = @import("wordpiece.zig");
pub const monster = @import("monster.zig");
pub const rwkv_world = @import("rwkv_world.zig");

// Training
pub const train_bpe = @import("train_bpe.zig");
pub const train_unigram = @import("train_unigram.zig");
pub const train_wordpiece = @import("train_wordpiece.zig");
pub const train_monster = @import("train_monster.zig");
pub const train_pathpiece = @import("train_pathpiece.zig");
pub const negative_train = @import("negative_train.zig");

// Post-processing + chat templates
pub const post_processor = @import("post_processor.zig");
pub const chat_template = @import("chat_template.zig");

// Writers
pub const hf_writer = @import("hf_writer.zig");
pub const sp_writer = @import("sp_writer.zig");

// Diagnostics + evaluation + serving helpers
pub const doctor = @import("doctor.zig");
pub const cli_validate = @import("cli_validate.zig");
pub const cli_diff = @import("cli_diff.zig");
pub const cli_eval = @import("cli_eval.zig");
pub const cli_bench = @import("cli_bench.zig");
pub const cli_serve = @import("cli_serve.zig");
pub const cli_grpc = @import("cli_grpc.zig");
pub const proto_min = @import("proto_min.zig");
pub const metrics = @import("metrics.zig");
pub const websocket = @import("websocket.zig");
pub const auth_oidc = @import("auth_oidc.zig");
pub const mbedtls = @import("mbedtls.zig");
pub const eval = @import("eval.zig");
pub const fairness = @import("fairness.zig");
pub const explain = @import("explain.zig");
pub const diff = @import("diff.zig");
pub const prefix_cache = @import("prefix_cache.zig");
pub const constrained = @import("constrained.zig");

// Streaming encode API (for `ztok serve` and external streaming clients)
pub const stream = @import("stream.zig");

// Chunking
pub const chunk = @import("chunk.zig");

// Vocab manipulation
pub const vocab_extend = @import("vocab_extend.zig");
pub const vocab_prune = @import("vocab_prune.zig");
pub const vocab_merge = @import("vocab_merge.zig");
pub const transcode = @import("transcode.zig");
pub const tokenize_dataset = @import("tokenize_dataset.zig");
pub const vocab_continued_pretrain = @import("vocab_continued_pretrain.zig");
pub const vocab_viz = @import("vocab_viz.zig");

// Pre-tokenizers
pub const cl100k = @import("cl100k.zig");
pub const o200k = @import("o200k.zig");
pub const hf_bytelevel_pretok = @import("hf_bytelevel_pretok.zig");
pub const hf_regex = @import("hf_regex.zig");

// Tokenizer loaders
pub const o200k_harmony = @import("o200k_harmony.zig");

// Capcode
pub const capcode = @import("capcode.zig");
pub const tm_norm = @import("tm_norm.zig");

// Unicode
pub const unicode_props = @import("unicode_props.zig");
pub const unicode_norm = @import("unicode_norm.zig");
pub const byte_level = @import("byte_level.zig");

// Loaders + bridges
pub const hf_json = @import("hf_json.zig");
pub const hf_bridge = @import("hf_bridge.zig");
pub const sp_model = @import("sp_model.zig");
pub const sp_bridge = @import("sp_bridge.zig");
pub const sp_charsmap = @import("sp_charsmap.zig");
pub const proto = @import("proto.zig");
pub const tokenizer_config = @import("tokenizer_config.zig");
pub const auto_detect = @import("auto_detect.zig");
pub const monster_io = @import("monster_io.zig");
pub const tekken = @import("tekken.zig");

// Added-token resolution
pub const added_tokens = @import("added_tokens.zig");

// Token healing — trim mid-token prompt tails so generation resumes on a
// natural token boundary.
pub const token_healing = @import("token_healing.zig");

// C ABI — comptime ref forces analysis of the `export fn` decls so
// they appear in the static and shared libraries.
pub const c_api = @import("c_api.zig");
comptime {
    _ = c_api;
}

// Convenience re-exports of the most-used types
pub const TokenId = token.TokenId;
pub const Span = token.Span;
pub const OverlayKind = token.OverlayKind;
pub const Boundary = token.Boundary;
pub const Provenance = token.Provenance;
pub const Pipeline = pipeline.Pipeline;
pub const Overlay = pipeline.Overlay;
pub const EncodingWithOverlays = pipeline.EncodingWithOverlays;
pub const ChunkedEncodeCache = pipeline.ChunkedEncodeCache;
pub const EncodedChunk = pipeline.EncodedChunk;
pub const ChunkedEncoding = pipeline.ChunkedEncoding;
pub const Normalizer = normalizer.Normalizer;
pub const PreTokenizer = pretok.PreTokenizer;
pub const Model = model.Model;
pub const Decoder = decoder.Decoder;
pub const Vocab = vocab.Vocab;
pub const Bpe = bpe.Bpe;
pub const Unigram = unigram.Unigram;
pub const WordPiece = wordpiece.WordPiece;
pub const Monster = monster.Monster;
pub const RwkvWorld = rwkv_world.RwkvWorld;
pub const HFTokenizer = hf_json.HFTokenizer;
pub const SpModel = sp_model.SpModel;
pub const Constraint = constrained.Constraint;
pub const hashNGrams = ngram.hashNGrams;
pub const hashNGramsLen = ngram.hashNGramsLen;
pub const hashNGramsBatch = ngram.hashNGramsBatch;

// Tekken multimodal (text + image + audio) public API.
pub const TekkenModel = tekken.TekkenModel;
pub const ContentPart = tekken.ContentPart;
pub const ImageDims = tekken.ImageDims;
pub const AudioSpec = tekken.AudioSpec;
pub const encodeMultimodal = tekken.encodeMultimodal;
pub const encodeImage = tekken.encodeImage;
pub const encodeAudio = tekken.encodeAudio;

test {
    _ = token;
    _ = vocab;
    _ = normalizer;
    _ = pretok;
    _ = model;
    _ = decoder;
    _ = thread_pool;
    _ = connection_pool;
    _ = ngram;
    _ = pipeline;
    _ = fingerprint;
    _ = trace;
    _ = superposition;
    _ = semantic_superposition;
    _ = semantic_exchange;
    _ = bpe;
    _ = simd_min;
    _ = simd_bytes;
    _ = asm_normalizer;
    _ = bpe_heap;
    _ = unigram;
    _ = wordpiece;
    _ = monster;
    _ = rwkv_world;
    _ = train_bpe;
    _ = train_unigram;
    _ = train_wordpiece;
    _ = train_monster;
    _ = train_pathpiece;
    _ = negative_train;
    _ = post_processor;
    _ = chat_template;
    _ = hf_writer;
    _ = sp_writer;
    _ = doctor;
    _ = cli_validate;
    _ = cli_diff;
    _ = cli_eval;
    _ = cli_bench;
    _ = cli_serve;
    _ = cli_grpc;
    _ = proto_min;
    _ = stream;
    _ = eval;
    _ = fairness;
    _ = explain;
    _ = diff;
    _ = prefix_cache;
    _ = constrained;
    _ = chunk;
    _ = vocab_extend;
    _ = vocab_prune;
    _ = vocab_merge;
    _ = transcode;
    _ = tokenize_dataset;
    _ = vocab_continued_pretrain;
    _ = vocab_viz;
    _ = cl100k;
    _ = o200k;
    _ = o200k_harmony;
    _ = hf_bytelevel_pretok;
    _ = hf_regex;
    _ = capcode;
    _ = unicode_props;
    _ = unicode_norm;
    _ = byte_level;
    _ = hf_json;
    _ = hf_bridge;
    _ = sp_model;
    _ = sp_bridge;
    _ = sp_charsmap;
    _ = proto;
    _ = tokenizer_config;
    _ = auto_detect;
    _ = added_tokens;
    _ = token_healing;
    _ = monster_io;
    _ = tekken;
    _ = metrics;
    _ = websocket;
    _ = auth_oidc;
    _ = mbedtls;
    _ = c_api;
}
