//! Mistral Tekken tokenizer loader.
//!
//! Tekken is the tiktoken-based BPE tokenizer Mistral introduced with
//! Nemo (2024). It's also used by Pixtral and Codestral. The on-disk
//! format is a single `tekken.json` file with this top-level shape:
//!
//! ```json
//! {
//!   "version": 7,
//!   "type": "Tekkenizer",
//!   "config": {
//!     "pattern": "(?i:...)...",
//!     "num_vocab_tokens": 130000,
//!     "default_vocab_size": 131072,
//!     "default_num_special_tokens": 1000,
//!     "version": "v7"
//!   },
//!   "vocab": [
//!     { "rank": 0,   "token_bytes": "AA==",      "token_str": "<0x00>" },
//!     { "rank": 1,   "token_bytes": "AQ==",      "token_str": "<0x01>" },
//!     ...
//!     { "rank": 256, "token_bytes": "IGFuZA==",  "token_str": " and"  }
//!   ],
//!   "special_tokens": [
//!     { "rank": 0, "token_str": "<unk>",        "is_control": true },
//!     { "rank": 1, "token_str": "<s>",          "is_control": true },
//!     { "rank": 2, "token_str": "</s>",         "is_control": true },
//!     ...
//!   ]
//! }
//! ```
//!
//! ## ID layout
//!
//! Mistral packs special tokens into the **bottom** of the id space, then
//! shifts the regular vocab up by `num_special_tokens`:
//!
//!   final_id(special)   = special.rank                       (in [0, N_sp))
//!   final_id(vocab)     = vocab.rank + num_special_tokens    (in [N_sp, N_sp + N_voc))
//!
//! Reference: `mistral_common/tokens/tokenizers/tekken.py` (Tekkenizer.encode /
//! _decode_all). Numbers cross-checked against Mistral-Nemo-Base-2407's
//! `tekken.json` (vocab_size=131072, num_special_tokens=1000).
//!
//! ## Construction strategy
//!
//! The underlying byte vocab is identical in shape to a tiktoken file —
//! base64 byte sequences indexed by rank — so we build a `Bpe` with the
//! special tokens occupying ids 0..N_sp-1 (their literal `token_str`
//! bytes are the piece bytes) and the regular vocab occupying ids
//! N_sp..N_sp+N_voc-1 (their decoded base64 bytes are the piece bytes).
//! The first 256 vocab ranks are the raw bytes 0..255, so byte fallback
//! comes for free via the `byte_fallback` table.
//!
//! The `pattern` field on `config` is a tiktoken-style PCRE regex. For
//! the BPE merge loop itself the pattern only matters at the pre-tokenize
//! stage — the pipeline picks a pre-tokenizer separately. Until ztok has
//! a generic PCRE-tiktoken pre-tokenizer, callers can wire `cl100k` (the
//! closest existing variant) or `identity` themselves. We expose the raw
//! pattern string on `TekkenModel.pattern` so callers can dispatch.
//!
//! ## What this loader does today
//!
//! - Parses the JSON via `std.json` (mirrors `hf_json.zig`).
//! - Decodes the base64 vocab and builds a `Bpe` with the id-shift
//!   convention above.
//! - Returns the list of special tokens (id, content, is_control) so
//!   callers can wire them into a `Pipeline.added_tokens` scanner.
//! - Validates the v7-and-later "special_tokens required" rule. Older
//!   versions (≤7) that omit the array are rejected with
//!   `error.MissingSpecialTokens` for now — adding the deprecated
//!   defaults is a TODO.
//! - Validates that ranks 0..255 decode to the raw byte values (Tekken
//!   invariant; reject otherwise so we don't silently load a broken file).
//!
//! ## What this loader does NOT do today
//!
//! - Apply the `pattern` regex. Encoded results will be byte-correct only
//!   when callers wire a matching pre-tokenizer (or accept the
//!   `.identity` pre-tokenizer's longest-token-per-input behaviour on
//!   short inputs).
//! - Parse `model_settings_builder`. We ignore that section.
//! - Run the actual image preprocessing pipeline (resize / normalize /
//!   patchify). We only expose the *config* and a placeholder-token
//!   generator that produces the correct number of `[IMG]` ids for a
//!   given image size — the pixel pipeline is out of scope for the
//!   tokenizer crate.
//! - Run the actual audio spectrogram / frame pipeline. Same rationale:
//!   we expose the parsed `AudioConfig` so callers can know the expected
//!   sampling rate / frame rate, but ztok does not perform STFT.
//! - Fill in placeholder `<SPECIAL_{id}>` slots up to
//!   `default_num_special_tokens`. The spec reserves the gap; we only
//!   register the actually-named specials.
//!
//! ## Image / audio config
//!
//! Pixtral (image) and some Codestral builds (audio) ship extra
//! top-level sections alongside `config`:
//!
//! ```json
//! {
//!   "image": {
//!     "image_patch_size": 16,
//!     "max_image_size": 1024,
//!     "spatial_merge_size": 1
//!   },
//!   "audio": {
//!     "sampling_rate": 16000,
//!     "frame_rate": 12.5,
//!     "chunk_length_s": 30.0,
//!     "audio_encoding_config": {
//!       "num_mel_bins": 128,
//!       "hop_length": 160,
//!       "window_size": 400
//!     }
//!   }
//! }
//! ```
//!
//! Older Pixtral builds put the image config under the key `multimodal`
//! instead — Mistral's loader accepts either name pre-v11, then required
//! `image` post-v11. We mirror that: try `image` first, fall back to
//! `multimodal` for older files.
//!
//! Audio carries a nested `audio_encoding_config` (the spectrogram
//! settings) which we lift into a dedicated `AudioSpectrogramConfig`
//! struct so callers can inspect either piece independently.

const std = @import("std");
const TokenId = @import("token.zig").TokenId;
const Bpe = @import("bpe.zig").Bpe;

/// Tekken's own pre-tokenization regex, implemented by hand (it is NOT
/// cl100k — see the file docstring). Exposed here so callers wire the
/// correct splitter for the Tekken BPE.
pub const pretok = @import("tekken_pretok.zig");

pub const Error = error{
    MalformedJson,
    MissingField,
    UnsupportedTekkenVersion,
    UnsupportedTekkenType,
    MissingSpecialTokens,
    InvalidBase64,
    ByteRankMismatch,
    DuplicateRank,
    DuplicateSpecial,
    SpecialOutOfRange,
    InvalidConfig,
    InvalidImageConfig,
    InvalidAudioConfig,
} || std.mem.Allocator.Error || std.json.ParseError(std.json.Scanner);

/// One special token as stored in `tekken.json`.
pub const SpecialToken = struct {
    id: TokenId,
    content: []u8,
    is_control: bool,
};

/// Pixtral-style image config. Maps directly onto Mistral's
/// `ImageConfig` dataclass.
///
/// - `image_patch_size`: edge of the square patch the vision encoder
///   consumes (e.g. 16 px).
/// - `max_image_size`: longest-side cap before tiling; the image is
///   downscaled to fit inside `max_image_size × max_image_size` while
///   preserving aspect ratio.
/// - `spatial_merge_size`: how many `image_patch_size`-sized patches get
///   collapsed into a single token along each spatial axis. Default 1
///   (one patch = one token). Larger values shrink the token grid
///   quadratically.
pub const ImageConfig = struct {
    image_patch_size: u32,
    max_image_size: u32,
    spatial_merge_size: u32 = 1,
};

/// Image dimensions in pixels, used as input to `placeholderImageTokens`.
pub const ImageDims = struct {
    width: u32,
    height: u32,
};

/// Mistral audio spectrogram settings (nested inside `AudioConfig` on
/// disk under the key `audio_encoding_config`).
pub const AudioSpectrogramConfig = struct {
    num_mel_bins: u32,
    hop_length: u32,
    window_size: u32,
};

/// Mistral audio config. Mirrors the `AudioConfig` dataclass minus the
/// streaming-only TTS extras (we expose the optionals as nullable
/// floats; callers only need them for downstream STFT, which is out of
/// scope for ztok).
pub const AudioConfig = struct {
    sampling_rate: u32,
    frame_rate: f32,
    encoding_config: AudioSpectrogramConfig,
    chunk_length_s: ?f32 = null,
};

/// The three image-special token IDs Pixtral models reserve. Resolved
/// from `specials` by matching well-known content names. `null` means
/// the special wasn't found — typical for text-only tokenizers.
pub const SpecialImageIds = struct {
    img: ?TokenId = null,
    img_break: ?TokenId = null,
    img_end: ?TokenId = null,
};

/// The two audio-special token IDs Voxtral-style models reserve.
/// Resolved from `specials` by matching the well-known content names
/// `[BEGIN_AUDIO]` and `[AUDIO]` (mistral_common's `SpecialTokens.begin_audio`
/// = "[BEGIN_AUDIO]" and `.audio` = "[AUDIO]"). `null` means the special
/// wasn't found — typical for text-only / image-only tokenizers.
///
/// Note: unlike images (which are framed by `[IMG]…[IMG_BREAK]…[IMG_END]`),
/// mistral_common's non-streaming audio path emits a single leading
/// `[BEGIN_AUDIO]` followed by `n` repeated `[AUDIO]` placeholders with no
/// trailing terminator — see `AudioEncoder._encode_audio_tokens` in
/// mistral_common/tokens/tokenizers/audio.py.
pub const SpecialAudioIds = struct {
    begin_audio: ?TokenId = null,
    audio: ?TokenId = null,
};

/// Input descriptor for `encodeAudio` / a `.audio` content part. The
/// caller specifies the clip length in exactly one of two ways:
///   - `num_samples`: raw waveform sample count (after resampling to the
///     model's `sampling_rate`). This maps directly onto mistral_common's
///     `audio.audio_array.shape[0]`.
///   - `duration_s`: clip duration in seconds; converted to samples via
///     `round(duration_s * sampling_rate)`.
/// If both are set, `num_samples` wins. If neither is set the clip is
/// treated as zero-length (yields just `[BEGIN_AUDIO]`).
pub const AudioSpec = struct {
    num_samples: ?u64 = null,
    duration_s: ?f64 = null,
};

/// Loaded Tekken tokenizer. Owns its allocations; call `deinit` to free.
pub const TekkenModel = struct {
    allocator: std.mem.Allocator,
    bpe: Bpe,
    specials: []SpecialToken,
    /// `default_num_special_tokens` from the config — the reserved
    /// special-id range size. The actual `specials.len` may be smaller
    /// (unused slots are placeholders we don't materialize today).
    num_special_tokens: u32,
    /// Raw `config.pattern` regex string. Owned. Callers that need to
    /// pre-tokenize per the Tekken pattern can read this; the loader
    /// itself does not apply it.
    pattern: []u8,
    /// Format version (integer from the top-level `version` field).
    version: u32,
    /// Optional Pixtral-style image config. `null` for text-only files
    /// (the vast majority of Tekken tokenizers in the wild).
    image_config: ?ImageConfig = null,
    /// Optional audio config for Tekken variants that ship audio
    /// support. `null` otherwise.
    audio_config: ?AudioConfig = null,

    pub fn deinit(self: *TekkenModel) void {
        self.bpe.deinit();
        for (self.specials) |s| {
            if (s.content.len > 0) self.allocator.free(s.content);
        }
        if (self.specials.len > 0) self.allocator.free(self.specials);
        if (self.pattern.len > 0) self.allocator.free(self.pattern);
        self.* = .{
            .allocator = self.allocator,
            .bpe = .{
                .allocator = self.allocator,
                .bytes = &.{},
                .offsets = &.{},
                .count = 0,
                .by_bytes = std.StringHashMap(TokenId).init(self.allocator),
            },
            .specials = &.{},
            .num_special_tokens = 0,
            .pattern = &.{},
            .version = 0,
            .image_config = null,
            .audio_config = null,
        };
    }

    /// Locate the three image special-token IDs (`[IMG]`, `[IMG_BREAK]`,
    /// `[IMG_END]`) by scanning the special-token table. Returns nulls
    /// for any that aren't present, which is the common case on
    /// text-only models.
    pub fn specialImageIds(self: *const TekkenModel) SpecialImageIds {
        var out: SpecialImageIds = .{};
        for (self.specials) |s| {
            if (std.mem.eql(u8, s.content, "[IMG]")) {
                out.img = s.id;
            } else if (std.mem.eql(u8, s.content, "[IMG_BREAK]")) {
                out.img_break = s.id;
            } else if (std.mem.eql(u8, s.content, "[IMG_END]")) {
                out.img_end = s.id;
            }
        }
        return out;
    }

    /// Compute the number of placeholder image tokens for an image of
    /// the given pixel dimensions. Mirrors Mistral's
    /// `_image_to_num_tokens` exactly:
    ///
    ///   ratio = max(h / max_image_size, w / max_image_size)
    ///   if ratio > 1: w, h = round(w / ratio), round(h / ratio)
    ///   width_tokens  = (w - 1) / (image_patch_size * spatial_merge_size) + 1
    ///   height_tokens = (h - 1) / (image_patch_size * spatial_merge_size) + 1
    ///
    /// Returns the `(width_tokens, height_tokens)` grid. The final flat
    /// sequence length is `(width_tokens + 1) * height_tokens` because
    /// each row ends with an `[IMG_BREAK]` (the very last break is
    /// rewritten to `[IMG_END]`).
    pub fn imageTokenGrid(self: *const TekkenModel, dims: ImageDims) !struct { width: u32, height: u32 } {
        const cfg = self.image_config orelse return error.InvalidImageConfig;
        if (cfg.image_patch_size == 0 or cfg.spatial_merge_size == 0)
            return error.InvalidImageConfig;

        // Floating-point downscale to match Mistral's `round(...)`
        // semantics. Banker's rounding vs. away-from-zero doesn't matter
        // here since w/h are positive and Mistral uses Python's
        // `round()` which is banker's-round but in practice never hits
        // a .5 because of the integer division below.
        var w_f: f64 = @floatFromInt(dims.width);
        var h_f: f64 = @floatFromInt(dims.height);
        const max_f: f64 = @floatFromInt(cfg.max_image_size);
        const ratio = @max(h_f / max_f, w_f / max_f);
        if (ratio > 1.0) {
            w_f = @round(w_f / ratio);
            h_f = @round(h_f / ratio);
        }
        const w: u32 = @intFromFloat(w_f);
        const h: u32 = @intFromFloat(h_f);
        if (w == 0 or h == 0) return error.InvalidImageConfig;

        const block = cfg.image_patch_size * cfg.spatial_merge_size;
        const width_tokens: u32 = (w - 1) / block + 1;
        const height_tokens: u32 = (h - 1) / block + 1;
        return .{ .width = width_tokens, .height = height_tokens };
    }

    /// Materialize the full placeholder token sequence for an image of
    /// the given dimensions, mirroring `mistral_common`'s assembly:
    ///
    ///   image_tokens = ([img] * w + [img_break]) * h
    ///   image_tokens[-1] = img_end
    ///
    /// The returned slice is owned by the caller and must be freed with
    /// the allocator that was passed in. Returns `error.InvalidImageConfig`
    /// if no image config was loaded, or `error.MissingSpecialTokens`
    /// if any of the three required image specials are missing from the
    /// tokenizer.
    pub fn placeholderImageTokens(
        self: *const TekkenModel,
        allocator: std.mem.Allocator,
        dims: ImageDims,
    ) ![]TokenId {
        const grid = try self.imageTokenGrid(dims);
        const ids = self.specialImageIds();
        const img = ids.img orelse return error.MissingSpecialTokens;
        const brk = ids.img_break orelse return error.MissingSpecialTokens;
        const end = ids.img_end orelse return error.MissingSpecialTokens;

        const row_len: u32 = grid.width + 1; // one [IMG_BREAK] terminator per row
        const total: usize = @as(usize, row_len) * @as(usize, grid.height);
        const out = try allocator.alloc(TokenId, total);
        errdefer allocator.free(out);

        var row: u32 = 0;
        while (row < grid.height) : (row += 1) {
            const base: usize = @as(usize, row) * @as(usize, row_len);
            var i: u32 = 0;
            while (i < grid.width) : (i += 1) {
                out[base + i] = img;
            }
            out[base + grid.width] = brk;
        }
        out[total - 1] = end;
        return out;
    }

    /// Locate the two audio special-token IDs (`[BEGIN_AUDIO]`, `[AUDIO]`)
    /// by scanning the special-token table. Returns nulls for any that
    /// aren't present — the common case on text-only / image-only models.
    pub fn specialAudioIds(self: *const TekkenModel) SpecialAudioIds {
        var out: SpecialAudioIds = .{};
        for (self.specials) |s| {
            if (std.mem.eql(u8, s.content, "[BEGIN_AUDIO]")) {
                out.begin_audio = s.id;
            } else if (std.mem.eql(u8, s.content, "[AUDIO]")) {
                out.audio = s.id;
            }
        }
        return out;
    }

    /// Compute the number of `[AUDIO]` placeholder tokens for an audio
    /// clip of `num_samples` waveform samples (already resampled to the
    /// config `sampling_rate`). Mirrors mistral_common's
    /// `AudioEncoder._encode_audio_tokens` count arithmetic exactly:
    ///
    ///   hop = encoding_config.hop_length
    ///   if signal_len % hop != 0:
    ///       signal_len = ceil(signal_len / hop - 1)
    ///   else:
    ///       signal_len = signal_len / hop
    ///   audio_length_per_tok = int( (sampling_rate // frame_rate) / hop )
    ///   num_audio_tokens = ceil(signal_len / audio_length_per_tok)
    ///
    /// This is the *placeholder* token count only (the leading
    /// `[BEGIN_AUDIO]` is added by `placeholderAudioTokens`). Returns
    /// `error.InvalidAudioConfig` if no audio config was loaded or the
    /// derived `audio_length_per_tok` is zero (degenerate config).
    pub fn audioTokenCount(self: *const TekkenModel, num_samples: u64) !u64 {
        const cfg = self.audio_config orelse return error.InvalidAudioConfig;
        const hop: u64 = cfg.encoding_config.hop_length;
        if (hop == 0) return error.InvalidAudioConfig;

        // Spectrogram downsample by hop_length (matches the log-mel STFT
        // frame count the reference computes). Python uses
        // `math.ceil(x/hop - 1)` for the non-divisible case.
        var signal_len: u64 = undefined;
        if (num_samples % hop != 0) {
            // ceil(num_samples/hop - 1). With integer math:
            //   ceil(a/hop) - 1 == (a + hop - 1)/hop - 1 for a%hop!=0,
            // and ceil(a/hop - 1) == ceil(a/hop) - 1 exactly because the
            // subtracted 1 is an integer. Guard against underflow when
            // num_samples < hop (ceil(<1) == 1, minus 1 == 0).
            const ceil_div: u64 = (num_samples + hop - 1) / hop;
            signal_len = if (ceil_div > 0) ceil_div - 1 else 0;
        } else {
            signal_len = num_samples / hop;
        }

        // audio_length_per_tok = int( (sampling_rate // frame_rate) / hop )
        // `raw_audio_length_per_tok` is an integer floor division in the
        // reference; we replicate the // then the float→int truncation.
        if (cfg.frame_rate <= 0) return error.InvalidAudioConfig;
        const raw_per_tok: u64 = @intFromFloat(@floor(
            @as(f64, @floatFromInt(cfg.sampling_rate)) / @as(f64, cfg.frame_rate),
        ));
        const per_tok: u64 = raw_per_tok / hop; // int(float/hop) truncates
        if (per_tok == 0) return error.InvalidAudioConfig;

        // ceil(signal_len / per_tok)
        return (signal_len + per_tok - 1) / per_tok;
    }

    /// Materialize the full placeholder audio token sequence for a clip
    /// of `num_samples` samples, mirroring mistral_common's
    /// `[BEGIN_AUDIO] + [AUDIO] * num_audio_tokens` assembly. The returned
    /// slice is caller-owned (free with `allocator`). Returns
    /// `error.InvalidAudioConfig` if no audio config was loaded, or
    /// `error.MissingSpecialTokens` if either of `[BEGIN_AUDIO]` /
    /// `[AUDIO]` is missing from the tokenizer's special table.
    pub fn placeholderAudioTokens(
        self: *const TekkenModel,
        allocator: std.mem.Allocator,
        num_samples: u64,
    ) ![]TokenId {
        const n = try self.audioTokenCount(num_samples);
        const ids = self.specialAudioIds();
        const begin = ids.begin_audio orelse return error.MissingSpecialTokens;
        const audio = ids.audio orelse return error.MissingSpecialTokens;

        const total: usize = @intCast(n + 1); // +1 for [BEGIN_AUDIO]
        const out = try allocator.alloc(TokenId, total);
        errdefer allocator.free(out);
        out[0] = begin;
        var i: usize = 1;
        while (i < total) : (i += 1) out[i] = audio;
        return out;
    }
};

/// Encode-time entry for a single image. Resolves the three image
/// special-token IDs via `specialImageIds`, computes the placeholder
/// grid via `imageTokenGrid`, then materializes the flat sequence via
/// `placeholderImageTokens`. The returned slice is caller-owned (free
/// with `allocator`).
///
/// Errors:
/// - `error.InvalidImageConfig` if `model` was loaded from a text-only
///   tekken.json (no `image_config`) or the dims downscale to zero.
/// - `error.MissingSpecialTokens` if any of `[IMG]` / `[IMG_BREAK]` /
///   `[IMG_END]` is absent from the tokenizer's special table.
///
/// Mirrors the encode-side half of Pixtral's `MultiModalEncoder`
/// (mistral_common): the actual pixel preprocessing is out of scope,
/// but the placeholder ID sequence we hand back is byte-identical to
/// what the reference path would emit for the same `(w, h)` input.
pub fn encodeImage(
    allocator: std.mem.Allocator,
    model: *const TekkenModel,
    image_dims: ImageDims,
) ![]TokenId {
    // Pre-check the special-id table so callers get the clearer
    // `MissingSpecialTokens` error rather than discovering at the
    // bottom of `placeholderImageTokens` that one of three is null.
    const ids = model.specialImageIds();
    if (ids.img == null or ids.img_break == null or ids.img_end == null)
        return error.MissingSpecialTokens;
    // `placeholderImageTokens` internally calls `imageTokenGrid` and
    // `specialImageIds` again — that's the documented composition.
    return model.placeholderImageTokens(allocator, image_dims);
}

/// Resolve the clip's sample count from an `AudioSpec`: `num_samples`
/// takes priority; otherwise `duration_s * sampling_rate` rounded to the
/// nearest integer; otherwise zero.
fn audioSampleCount(cfg: AudioConfig, spec: AudioSpec) u64 {
    if (spec.num_samples) |ns| return ns;
    if (spec.duration_s) |d| {
        if (d <= 0) return 0;
        const samples = @round(d * @as(f64, @floatFromInt(cfg.sampling_rate)));
        if (samples <= 0) return 0;
        return @intFromFloat(samples);
    }
    return 0;
}

/// Encode-time entry for a single audio clip. Resolves the two audio
/// special-token IDs via `specialAudioIds`, computes the placeholder
/// count via `audioTokenCount`, then materializes the flat sequence
/// `[BEGIN_AUDIO] + [AUDIO]*n` via `placeholderAudioTokens`. The returned
/// slice is caller-owned (free with `allocator`).
///
/// Errors:
/// - `error.InvalidAudioConfig` if `model` was loaded without an
///   `audio` section (text-only / image-only tekken.json), or the config
///   is degenerate (zero hop / frame_rate).
/// - `error.MissingSpecialTokens` if either `[BEGIN_AUDIO]` / `[AUDIO]`
///   is absent from the tokenizer's special table.
///
/// Mirrors the encode-side half of Voxtral's audio encoder
/// (mistral_common `AudioEncoder._encode_audio_tokens`): real feature
/// extraction (resampling + log-mel STFT) is OUT OF SCOPE, but the
/// placeholder ID sequence we hand back is byte-identical to what the
/// reference path would emit for the same sample count.
pub fn encodeAudio(
    allocator: std.mem.Allocator,
    model: *const TekkenModel,
    spec: AudioSpec,
) ![]TokenId {
    const cfg = model.audio_config orelse return error.InvalidAudioConfig;
    // Pre-check the special-id table so callers get the clearer
    // `MissingSpecialTokens` error before any allocation, matching the
    // `encodeImage` contract.
    const ids = model.specialAudioIds();
    if (ids.begin_audio == null or ids.audio == null)
        return error.MissingSpecialTokens;
    const num_samples = audioSampleCount(cfg, spec);
    return model.placeholderAudioTokens(allocator, num_samples);
}

/// One slice of a mixed-content message. Text segments are encoded
/// through the BPE merge loop; image segments expand into their
/// grid-shaped placeholder sequence; audio segments expand into the
/// `[BEGIN_AUDIO] + [AUDIO]*n` placeholder sequence.
pub const ContentPart = union(enum) {
    text: []const u8,
    image: ImageDims,
    audio: AudioSpec,
};

/// Encode an interleaved sequence of text, image, and audio parts into a
/// flat token stream, mimicking Pixtral / Voxtral / mistral_common's
/// multimodal message encoder. Text is fed straight through `model.bpe.encodeChunk`
/// (no pretokenizer applied — callers wanting Tekken's `pattern` regex
/// segmentation should pre-split and feed the pieces in as separate
/// `text` parts, same constraint as the rest of the Tekken loader).
/// Each image expands inline via `encodeImage`, so the produced stream
/// matches the order callers passed in.
///
/// The returned slice is caller-owned (free with `allocator`). On any
/// per-part failure the partial output is freed before returning.
pub fn encodeMultimodal(
    allocator: std.mem.Allocator,
    model: *const TekkenModel,
    content: []const ContentPart,
) ![]TokenId {
    var out: std.ArrayList(TokenId) = .empty;
    errdefer out.deinit(allocator);

    for (content) |part| switch (part) {
        .text => |s| {
            if (s.len == 0) continue;
            // Worst case: one id per byte (no merges). Reserve up front
            // so the bpe encode writes into a stable slice and we then
            // truncate to the real length.
            const base = out.items.len;
            try out.resize(allocator, base + s.len);
            const dst = out.items[base..];
            const written = model.bpe.encodeChunk(s, dst);
            try out.resize(allocator, base + written.len);
        },
        .image => |dims| {
            const ids = try encodeImage(allocator, model, dims);
            defer allocator.free(ids);
            try out.appendSlice(allocator, ids);
        },
        .audio => |spec| {
            const ids = try encodeAudio(allocator, model, spec);
            defer allocator.free(ids);
            try out.appendSlice(allocator, ids);
        },
    };

    return out.toOwnedSlice(allocator);
}

/// Read `path` and parse it as a Tekken tokenizer. Uses the global
/// single-threaded `Io` since loading is a synchronous one-shot.
pub fn loadTekkenFile(allocator: std.mem.Allocator, path: []const u8) !TekkenModel {
    const io = std.Io.Threaded.global_single_threaded.io();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(
        io,
        path,
        allocator,
        .limited(std.math.maxInt(usize)),
    );
    defer allocator.free(bytes);
    return loadTekkenBytes(allocator, bytes);
}

/// Parse Tekken JSON from an in-memory buffer.
pub fn loadTekkenBytes(allocator: std.mem.Allocator, json_bytes: []const u8) !TekkenModel {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_bytes, .{});
    defer parsed.deinit();

    const root = parsed.value;
    if (root != .object) return error.MalformedJson;

    // Type guard. v7+ files set "Tekkenizer"; older internal builds may
    // omit the field entirely, which we accept.
    if (root.object.get("type")) |t| {
        if (t != .string) return error.MalformedJson;
        if (!std.mem.eql(u8, t.string, "Tekkenizer")) return error.UnsupportedTekkenType;
    }

    const config_v = root.object.get("config") orelse return error.MissingField;
    if (config_v != .object) return error.MalformedJson;

    // Version resolution. Prefer the top-level `version` integer (v7+
    // tekken.json). Older / internal Mistral builds (e.g. the real
    // mistral-nemo `tekken.json` shipped on HuggingFace) omit the
    // top-level field entirely and only store the version inside the
    // nested `config` object as a string like `"v3"` / `"v11"`. Accept
    // that as a fallback so those files load instead of strict-rejecting
    // with MissingField.
    const version: u32 = if (root.object.get("version")) |vv| switch (vv) {
        .integer => |i| if (i < 0 or i > std.math.maxInt(u32))
            return error.UnsupportedTekkenVersion
        else
            @intCast(i),
        else => return error.MalformedJson,
    } else if (config_v.object.get("version")) |cv| switch (cv) {
        .string => |s| parseConfigVersionString(s) catch return error.UnsupportedTekkenVersion,
        .integer => |i| if (i < 0 or i > std.math.maxInt(u32))
            return error.UnsupportedTekkenVersion
        else
            @intCast(i),
        else => return error.MalformedJson,
    } else return error.MissingField;

    // Parse optional image/audio config NOW, before any of the large BPE
    // allocations below. These are value types (no owned heap pointers)
    // so they don't need cleanup on a later error path. Doing them up
    // front means a malformed image/audio section can fail-fast without
    // leaking the BPE arena.
    //
    // Pre-v11 Mistral wrote the image config under `multimodal`; v11+
    // requires `image`. We accept either, preferring `image`.
    var image_cfg: ?ImageConfig = null;
    if (root.object.get("image")) |iv| {
        image_cfg = try parseImageConfig(iv);
    } else if (root.object.get("multimodal")) |iv| {
        image_cfg = try parseImageConfig(iv);
    }
    var audio_cfg: ?AudioConfig = null;
    if (root.object.get("audio")) |av| {
        audio_cfg = try parseAudioConfig(av);
    }

    const pattern_v = config_v.object.get("pattern") orelse return error.MissingField;
    if (pattern_v != .string) return error.MalformedJson;
    const pattern_owned = try allocator.dupe(u8, pattern_v.string);
    errdefer allocator.free(pattern_owned);

    const num_special_v = config_v.object.get("default_num_special_tokens") orelse
        return error.MissingField;
    const num_special_tokens: u32 = switch (num_special_v) {
        .integer => |i| if (i < 0 or i > std.math.maxInt(u32))
            return error.InvalidConfig
        else
            @intCast(i),
        else => return error.MalformedJson,
    };

    // `default_vocab_size` caps the MERGEABLE vocabulary. Tekken files
    // ship MORE `vocab` entries than the model actually uses (the Nemo
    // fixture lists 150000 ranks but `default_vocab_size` is 131072).
    // mistral_common loads only the first `inner_vocab_size = vocab_size -
    // num_special_tokens` ranks as mergeable ranks
    // (`_reload_mergeable_ranks(vocab, max_vocab=inner_vocab_size)`), so
    // higher-rank pieces like ` Gutenberg` never form during BPE — the
    // merge stops at ` Guten`+`berg`. We mirror that: parse the cap and,
    // below, register only ranks < `inner_vocab_size` in the merge
    // lookup table. `default_vocab_size` is optional (older internal
    // files omit it); when absent we fall back to loading every rank
    // (`inner_vocab_size = vocab_count`), preserving prior behaviour.
    const default_vocab_size: ?u32 = if (config_v.object.get("default_vocab_size")) |dv| switch (dv) {
        .integer => |i| if (i < 0 or i > std.math.maxInt(u32))
            return error.InvalidConfig
        else
            @intCast(i),
        else => return error.MalformedJson,
    } else null;

    // Parse the vocab — array of {rank, token_bytes, token_str}.
    const vocab_v = root.object.get("vocab") orelse return error.MissingField;
    if (vocab_v != .array) return error.MalformedJson;

    // Decode base64 entries into a side table indexed by rank, then emit
    // them into the final SoA buffers shifted by num_special_tokens.
    const decoder = std.base64.standard.Decoder;
    const vocab_count: u32 = @intCast(vocab_v.array.items.len);

    // Side table: per-rank bytes. We collect into a single staging arena
    // and remember (start, len) per rank so we can stream out in rank order
    // after the parse pass.
    const Entry = struct { start: u32, len: u32 };
    const entries = try allocator.alloc(Entry, vocab_count);
    defer allocator.free(entries);
    @memset(entries, .{ .start = 0, .len = std.math.maxInt(u32) });

    // First pass: tally decoded bytes so we can size the staging arena.
    var total_vocab_bytes: usize = 0;
    for (vocab_v.array.items) |it| {
        if (it != .object) return error.MalformedJson;
        const tb = it.object.get("token_bytes") orelse return error.MissingField;
        if (tb != .string) return error.MalformedJson;
        const n = decoder.calcSizeForSlice(tb.string) catch return error.InvalidBase64;
        total_vocab_bytes += n;
    }

    var staging: []u8 = &.{};
    if (total_vocab_bytes > 0) staging = try allocator.alloc(u8, total_vocab_bytes);
    defer if (staging.len > 0) allocator.free(staging);

    var staging_off: u32 = 0;
    for (vocab_v.array.items) |it| {
        const rank_v = it.object.get("rank") orelse return error.MissingField;
        const rank: u32 = switch (rank_v) {
            .integer => |i| if (i < 0 or i >= vocab_count)
                return error.DuplicateRank
            else
                @intCast(i),
            else => return error.MalformedJson,
        };
        if (entries[rank].len != std.math.maxInt(u32)) return error.DuplicateRank;

        const tb = it.object.get("token_bytes").?;
        const n: u32 = @intCast(decoder.calcSizeForSlice(tb.string) catch return error.InvalidBase64);
        decoder.decode(staging[staging_off .. staging_off + n], tb.string) catch
            return error.InvalidBase64;
        entries[rank] = .{ .start = staging_off, .len = n };
        staging_off += n;
    }

    // Sanity: ranks 0..255 must equal the raw byte 0..255.
    {
        const cap = @min(256, vocab_count);
        var i: u32 = 0;
        while (i < cap) : (i += 1) {
            const e = entries[i];
            if (e.len != 1 or staging[e.start] != @as(u8, @intCast(i)))
                return error.ByteRankMismatch;
        }
    }

    // Parse special tokens. v7+ requires them; older files may omit and
    // expect the deprecated defaults. We reject the deprecated path here
    // (better a loud failure than silently wrong ids).
    const specials_v_opt = root.object.get("special_tokens");
    if (version > 7 and specials_v_opt == null) return error.MissingSpecialTokens;
    if (specials_v_opt) |v| if (v != .array and v != .null) return error.MalformedJson;

    var specials_list: std.ArrayList(SpecialToken) = .empty;
    errdefer {
        for (specials_list.items) |s| {
            if (s.content.len > 0) allocator.free(s.content);
        }
        specials_list.deinit(allocator);
    }

    // Tally bytes for the specials portion of the final BPE arena too.
    var total_special_bytes: usize = 0;
    var max_special_rank: u32 = 0;
    var have_specials: bool = false;
    if (specials_v_opt) |sv| if (sv == .array) {
        have_specials = true;
        for (sv.array.items) |it| {
            if (it != .object) return error.MalformedJson;
            const ts = it.object.get("token_str") orelse return error.MissingField;
            if (ts != .string) return error.MalformedJson;
            const rk_v = it.object.get("rank") orelse return error.MissingField;
            const rk: u32 = switch (rk_v) {
                .integer => |i| if (i < 0 or i > std.math.maxInt(u32))
                    return error.MalformedJson
                else
                    @intCast(i),
                else => return error.MalformedJson,
            };
            if (rk >= num_special_tokens) return error.SpecialOutOfRange;
            if (rk > max_special_rank) max_special_rank = rk;
            total_special_bytes += ts.string.len;
        }
    };

    // Compose the final BPE vocab. Total id count = num_special_tokens +
    // vocab_count. Special slots that weren't named in the file are filled
    // with synthetic `<SPECIAL_{rank}>` placeholders so the SoA stays
    // contiguous (the placeholder bytes are visible if the encoder ever
    // emits that id, which it won't unless an explicit special is wired
    // through `added_tokens`).
    const placeholder_max_len: usize = 32; // "<SPECIAL_4294967295>" fits easily
    const total_bytes = num_special_tokens * placeholder_max_len + total_vocab_bytes;
    var bpe_bytes_list: std.ArrayList(u8) = try .initCapacity(allocator, total_bytes);
    errdefer bpe_bytes_list.deinit(allocator);
    const total_count: u32 = num_special_tokens + vocab_count;
    var bpe_offsets = try allocator.alloc(u32, @as(usize, total_count) + 1);
    errdefer allocator.free(bpe_offsets);
    bpe_offsets[0] = 0;

    // Per-rank pointer back into specials_list so we know which slot got
    // a real special and which got a placeholder. -1 = placeholder.
    const special_at_rank = try allocator.alloc(i32, num_special_tokens);
    defer allocator.free(special_at_rank);
    @memset(special_at_rank, -1);

    if (have_specials) {
        for (specials_v_opt.?.array.items) |it| {
            const ts = it.object.get("token_str").?.string;
            const rk_v = it.object.get("rank").?;
            const rk: u32 = @intCast(rk_v.integer);
            if (special_at_rank[rk] != -1) return error.DuplicateSpecial;
            const owned = try allocator.dupe(u8, ts);
            errdefer allocator.free(owned);
            const ic_v = it.object.get("is_control");
            const is_control: bool = if (ic_v) |v| switch (v) {
                .bool => |b| b,
                else => true,
            } else true;
            special_at_rank[rk] = @intCast(specials_list.items.len);
            try specials_list.append(allocator, .{
                .id = rk,
                .content = owned,
                .is_control = is_control,
            });
        }
    }

    // Write special slots (0..num_special_tokens) into the SoA.
    var write: u32 = 0;
    var ph_buf: [placeholder_max_len]u8 = undefined;
    {
        var rank: u32 = 0;
        while (rank < num_special_tokens) : (rank += 1) {
            const piece: []const u8 = if (special_at_rank[rank] != -1)
                specials_list.items[@intCast(special_at_rank[rank])].content
            else blk: {
                const fb = try std.fmt.bufPrint(&ph_buf, "<SPECIAL_{d}>", .{rank});
                break :blk fb;
            };
            try bpe_bytes_list.appendSlice(allocator, piece);
            write += @intCast(piece.len);
            bpe_offsets[rank + 1] = write;
        }
    }

    // Write vocab slots (num_special_tokens..total_count).
    {
        var rank: u32 = 0;
        while (rank < vocab_count) : (rank += 1) {
            const e = entries[rank];
            if (e.len == std.math.maxInt(u32)) return error.MissingField;
            try bpe_bytes_list.appendSlice(allocator, staging[e.start .. e.start + e.len]);
            write += e.len;
            bpe_offsets[num_special_tokens + rank + 1] = write;
        }
    }

    const bpe_bytes = try bpe_bytes_list.toOwnedSlice(allocator);
    errdefer allocator.free(bpe_bytes);

    // Build the reverse-lookup hashmap. We DON'T register special-token
    // pieces in `by_bytes` because the BPE merge loop must not be allowed
    // to merge into a literal `<s>` if that string appears in the input;
    // the canonical path for that is the `added_tokens` scanner running
    // before pre-tokenization. Vocab pieces (including the 256 raw bytes)
    // do go in.
    var by_bytes = std.StringHashMap(TokenId).init(allocator);
    errdefer by_bytes.deinit();

    // Number of regular ranks that are MERGEABLE. mistral_common keeps
    // only `vocab_size - num_special_tokens` of them; the rest of the
    // `vocab` array is ignored for merging (decode still maps them, but
    // BPE never produces them). Clamp to what the file actually provides.
    const inner_vocab_size: u32 = if (default_vocab_size) |vs|
        @min(if (vs > num_special_tokens) vs - num_special_tokens else 0, vocab_count)
    else
        vocab_count;
    try by_bytes.ensureTotalCapacity(inner_vocab_size);

    var byte_fallback: [256]TokenId = undefined;
    var byte_fallback_filled: u32 = 0;
    var max_piece_len: u32 = 1;

    {
        var rank: u32 = 0;
        while (rank < inner_vocab_size) : (rank += 1) {
            const final_id: TokenId = num_special_tokens + rank;
            const start = bpe_offsets[final_id];
            const end = bpe_offsets[final_id + 1];
            const key = bpe_bytes[start..end];

            // Ranks 0..255 are the raw byte tokens; route them to
            // byte_fallback so `<0x41>` literals don't accidentally
            // collapse to the byte-A id during pre-tokenization. Match
            // tiktoken's own behaviour: those base ids are still in
            // `by_bytes` since `bytes([0x41])` is also a legal lookup
            // path during the merge loop (a single 'A' byte must map to
            // the 'A'-byte id whether reached as raw byte or as a fallback).
            if (rank < 256) {
                byte_fallback[rank] = final_id;
                byte_fallback_filled += 1;
            }
            try by_bytes.put(key, final_id);
            if (key.len > max_piece_len) max_piece_len = @intCast(key.len);
        }
    }

    const bpe: Bpe = .{
        .allocator = allocator,
        .bytes = bpe_bytes,
        .offsets = bpe_offsets,
        .count = total_count,
        .by_bytes = by_bytes,
        .max_piece_len = max_piece_len,
        .byte_fallback = if (byte_fallback_filled == 256) byte_fallback else null,
    };

    const specials_owned = try specials_list.toOwnedSlice(allocator);

    return .{
        .allocator = allocator,
        .bpe = bpe,
        .specials = specials_owned,
        .num_special_tokens = num_special_tokens,
        .pattern = pattern_owned,
        .version = version,
        .image_config = image_cfg,
        .audio_config = audio_cfg,
    };
}

/// Parse the `config.version` string used by older Mistral tekken files
/// that lack a top-level `version` integer. Accepts the canonical
/// `"v<N>"` form (e.g. `"v3"`, `"v7"`, `"v11"`) as well as a bare
/// decimal integer. Returns the integer version. Errors on empty input,
/// a missing `v` prefix combined with non-numeric characters, or any
/// overflow past u32.
fn parseConfigVersionString(s: []const u8) !u32 {
    if (s.len == 0) return error.UnsupportedTekkenVersion;
    const digits = if (s[0] == 'v' or s[0] == 'V') s[1..] else s;
    if (digits.len == 0) return error.UnsupportedTekkenVersion;
    return std.fmt.parseInt(u32, digits, 10) catch error.UnsupportedTekkenVersion;
}

/// Pull a non-negative integer field out of a JSON object, with a typed
/// error so callers can distinguish "missing" from "wrong type".
fn jsonU32Field(obj: std.json.Value, name: []const u8) !u32 {
    const v = obj.object.get(name) orelse return error.MissingField;
    return switch (v) {
        .integer => |i| if (i < 0 or i > std.math.maxInt(u32))
            error.InvalidConfig
        else
            @intCast(i),
        else => error.MalformedJson,
    };
}

/// Same shape as `jsonU32Field` but for `f32` — accepts JSON integers
/// or floats since `frame_rate` is often written as an int (12) when
/// it happens to be whole.
fn jsonF32Field(obj: std.json.Value, name: []const u8) !f32 {
    const v = obj.object.get(name) orelse return error.MissingField;
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| @floatCast(f),
        else => error.MalformedJson,
    };
}

fn jsonOptF32Field(obj: std.json.Value, name: []const u8) !?f32 {
    const v = obj.object.get(name) orelse return null;
    return switch (v) {
        .null => null,
        .integer => |i| @as(f32, @floatFromInt(i)),
        .float => |f| @as(f32, @floatCast(f)),
        else => error.MalformedJson,
    };
}

fn parseImageConfig(v: std.json.Value) !ImageConfig {
    if (v != .object) return error.InvalidImageConfig;
    const patch = jsonU32Field(v, "image_patch_size") catch |e| switch (e) {
        error.MissingField, error.MalformedJson, error.InvalidConfig => return error.InvalidImageConfig,
    };
    const max_sz = jsonU32Field(v, "max_image_size") catch |e| switch (e) {
        error.MissingField, error.MalformedJson, error.InvalidConfig => return error.InvalidImageConfig,
    };
    if (patch == 0 or max_sz == 0) return error.InvalidImageConfig;

    // `spatial_merge_size` is optional with a default of 1.
    var merge: u32 = 1;
    if (v.object.get("spatial_merge_size")) |sv| switch (sv) {
        .integer => |i| if (i <= 0 or i > std.math.maxInt(u32))
            return error.InvalidImageConfig
        else {
            merge = @intCast(i);
        },
        .null => {},
        else => return error.InvalidImageConfig,
    };

    return .{
        .image_patch_size = patch,
        .max_image_size = max_sz,
        .spatial_merge_size = merge,
    };
}

fn parseAudioConfig(v: std.json.Value) !AudioConfig {
    if (v != .object) return error.InvalidAudioConfig;
    const enc_v = v.object.get("audio_encoding_config") orelse return error.InvalidAudioConfig;
    if (enc_v != .object) return error.InvalidAudioConfig;

    const num_mel = jsonU32Field(enc_v, "num_mel_bins") catch return error.InvalidAudioConfig;
    const hop = jsonU32Field(enc_v, "hop_length") catch return error.InvalidAudioConfig;
    const win = jsonU32Field(enc_v, "window_size") catch return error.InvalidAudioConfig;
    if (num_mel == 0 or hop == 0 or win == 0) return error.InvalidAudioConfig;

    const sr = jsonU32Field(v, "sampling_rate") catch return error.InvalidAudioConfig;
    const fr = jsonF32Field(v, "frame_rate") catch return error.InvalidAudioConfig;
    if (sr == 0 or fr <= 0) return error.InvalidAudioConfig;

    const chunk = jsonOptF32Field(v, "chunk_length_s") catch return error.InvalidAudioConfig;

    return .{
        .sampling_rate = sr,
        .frame_rate = fr,
        .encoding_config = .{
            .num_mel_bins = num_mel,
            .hop_length = hop,
            .window_size = win,
        },
        .chunk_length_s = chunk,
    };
}

// --- tests ----------------------------------------------------------------

const testing = std.testing;

/// Build a minimal Tekken-shape JSON suitable for parser smoke tests.
/// Vocab is the first 256 raw bytes plus a handful of merged pieces.
fn buildMinimalTekkenJson(allocator: std.mem.Allocator) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);

    try buf.appendSlice(allocator,
        \\{
        \\  "version": 7,
        \\  "type": "Tekkenizer",
        \\  "config": {
        \\    "pattern": "[^\\s\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+",
        \\    "num_vocab_tokens": 260,
        \\    "default_vocab_size": 270,
        \\    "default_num_special_tokens": 10,
        \\    "version": "v7"
        \\  },
        \\  "vocab": [
    );

    const b64 = std.base64.standard.Encoder;
    var enc_buf: [16]u8 = undefined;

    // First 256 vocab entries: raw bytes 0..255.
    var byte_v: u32 = 0;
    while (byte_v < 256) : (byte_v += 1) {
        const b: u8 = @intCast(byte_v);
        const encoded = b64.encode(&enc_buf, &[_]u8{b});
        if (byte_v > 0) try buf.appendSlice(allocator, ",");
        try buf.print(allocator,
            \\
            \\    {{ "rank": {d}, "token_bytes": "{s}", "token_str": "<0x{X:0>2}>" }}
        , .{ byte_v, encoded, b });
    }

    // A few merged pieces.
    const merges = [_][]const u8{ "he", "lo", "hello" };
    for (merges, 0..) |piece, i| {
        const rank: u32 = 256 + @as(u32, @intCast(i));
        const encoded = b64.encode(&enc_buf, piece);
        try buf.appendSlice(allocator, ",");
        try buf.print(allocator,
            \\
            \\    {{ "rank": {d}, "token_bytes": "{s}", "token_str": "{s}" }}
        , .{ rank, encoded, piece });
    }

    try buf.appendSlice(allocator,
        \\
        \\  ],
        \\  "special_tokens": [
        \\    { "rank": 0, "token_str": "<unk>", "is_control": true },
        \\    { "rank": 1, "token_str": "<s>",   "is_control": true },
        \\    { "rank": 2, "token_str": "</s>",  "is_control": true }
        \\  ]
        \\}
    );

    return buf.toOwnedSlice(allocator);
}

test "loadTekkenBytes parses a minimal hand-crafted file" {
    const json = try buildMinimalTekkenJson(testing.allocator);
    defer testing.allocator.free(json);

    var tk = try loadTekkenBytes(testing.allocator, json);
    defer tk.deinit();

    try testing.expectEqual(@as(u32, 7), tk.version);
    try testing.expectEqual(@as(u32, 10), tk.num_special_tokens);
    // 10 specials + 256 raw bytes + 3 merged = 269 total ids.
    try testing.expectEqual(@as(u32, 10 + 256 + 3), tk.bpe.count);
    // The pattern string survived.
    try testing.expect(tk.pattern.len > 0);
}

test "loadTekkenBytes shifts vocab ids by num_special_tokens" {
    const json = try buildMinimalTekkenJson(testing.allocator);
    defer testing.allocator.free(json);

    var tk = try loadTekkenBytes(testing.allocator, json);
    defer tk.deinit();

    // Byte 'A' (0x41) has tekken rank 0x41 -> final id 10 + 0x41 = 0x4B.
    const a_id = tk.bpe.by_bytes.get("A") orelse return error.TestExpectedEqual;
    try testing.expectEqual(@as(TokenId, 10 + 0x41), a_id);

    // The merged piece "hello" was rank 258 in vocab -> final id 268.
    const hello_id = tk.bpe.by_bytes.get("hello") orelse return error.TestExpectedEqual;
    try testing.expectEqual(@as(TokenId, 10 + 258), hello_id);

    // byte_fallback is populated for all 256 bytes (offset by 10).
    try testing.expect(tk.bpe.byte_fallback != null);
    try testing.expectEqual(@as(TokenId, 10 + 0), tk.bpe.byte_fallback.?[0]);
    try testing.expectEqual(@as(TokenId, 10 + 255), tk.bpe.byte_fallback.?[255]);
}

test "loadTekkenBytes captures all named special tokens" {
    const json = try buildMinimalTekkenJson(testing.allocator);
    defer testing.allocator.free(json);

    var tk = try loadTekkenBytes(testing.allocator, json);
    defer tk.deinit();

    try testing.expectEqual(@as(usize, 3), tk.specials.len);
    // Ordering follows file order, not rank order — but for this fixture
    // they happen to coincide.
    try testing.expectEqualStrings("<unk>", tk.specials[0].content);
    try testing.expectEqual(@as(TokenId, 0), tk.specials[0].id);
    try testing.expectEqualStrings("<s>", tk.specials[1].content);
    try testing.expectEqual(@as(TokenId, 1), tk.specials[1].id);
    try testing.expectEqualStrings("</s>", tk.specials[2].content);
    try testing.expectEqual(@as(TokenId, 2), tk.specials[2].id);
}

test "loadTekkenBytes encodes a known input through the BPE" {
    const json = try buildMinimalTekkenJson(testing.allocator);
    defer testing.allocator.free(json);

    var tk = try loadTekkenBytes(testing.allocator, json);
    defer tk.deinit();

    // "hello" with the merges {he, lo, hello} encodes to [he, l, lo]
    // under the bpe_merge mode: the merge loop walks pair-by-pair and
    // can't reach the literal "hello" piece without intermediate "hel"
    // / "hell" merges (which a tiny test vocab omits). The result is
    // still bit-correct for what the vocab can represent.
    //
    // Real tekken.json files ship the full merge chain, so the
    // production path collapses straight to the longest piece. We
    // assert the predictable shape here: three tokens, and the boundary
    // pieces match the merges we did register.
    var out_buf: [16]TokenId = undefined;
    const ids = tk.bpe.encodeChunk("hello", &out_buf);
    try testing.expectEqual(@as(usize, 3), ids.len);
    // First token = "he" (rank 256 + 10)
    try testing.expectEqual(@as(TokenId, 10 + 256), ids[0]);
    // Middle token = raw byte 'l' (0x6C) shifted by num_special_tokens
    try testing.expectEqual(@as(TokenId, 10 + 0x6C), ids[1]);
    // Last token = "lo" (rank 257 + 10)
    try testing.expectEqual(@as(TokenId, 10 + 257), ids[2]);
}

test "loadTekkenBytes rejects v8 file without special_tokens" {
    const bad =
        \\{
        \\  "version": 8,
        \\  "type": "Tekkenizer",
        \\  "config": {
        \\    "pattern": ".",
        \\    "num_vocab_tokens": 1,
        \\    "default_vocab_size": 1,
        \\    "default_num_special_tokens": 1,
        \\    "version": "v8"
        \\  },
        \\  "vocab": [
        \\    { "rank": 0, "token_bytes": "AA==", "token_str": "<0x00>" }
        \\  ]
        \\}
    ;
    try testing.expectError(error.MissingSpecialTokens, loadTekkenBytes(testing.allocator, bad));
}

test "loadTekkenBytes rejects when ranks 0..255 aren't the raw bytes" {
    // Rank 0 holds 'A' instead of byte 0x00 — should reject.
    const bad =
        \\{
        \\  "version": 7,
        \\  "type": "Tekkenizer",
        \\  "config": {
        \\    "pattern": ".",
        \\    "num_vocab_tokens": 1,
        \\    "default_vocab_size": 1,
        \\    "default_num_special_tokens": 1,
        \\    "version": "v7"
        \\  },
        \\  "vocab": [
        \\    { "rank": 0, "token_bytes": "QQ==", "token_str": "A" }
        \\  ],
        \\  "special_tokens": [
        \\    { "rank": 0, "token_str": "<unk>", "is_control": true }
        \\  ]
        \\}
    ;
    try testing.expectError(error.ByteRankMismatch, loadTekkenBytes(testing.allocator, bad));
}

test "loadTekkenFile reads a real file" {
    const json = try buildMinimalTekkenJson(testing.allocator);
    defer testing.allocator.free(json);

    const io = std.Io.Threaded.global_single_threaded.io();
    const path = "/tmp/ztok_tekken_test.json";
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = json });
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    var tk = try loadTekkenFile(testing.allocator, path);
    defer tk.deinit();
    try testing.expectEqual(@as(u32, 269), tk.bpe.count);
}

// --- image / audio config tests ------------------------------------------

/// Build a Pixtral-shaped tekken.json: minimal vocab plus `image` and
/// `audio` config sections, and the three image specials so
/// `placeholderImageTokens` has IDs to emit.
fn buildMultimodalTekkenJson(allocator: std.mem.Allocator) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);

    try buf.appendSlice(allocator,
        \\{
        \\  "version": 11,
        \\  "type": "Tekkenizer",
        \\  "config": {
        \\    "pattern": ".",
        \\    "num_vocab_tokens": 256,
        \\    "default_vocab_size": 266,
        \\    "default_num_special_tokens": 10,
        \\    "version": "v11"
        \\  },
        \\  "vocab": [
    );

    const b64 = std.base64.standard.Encoder;
    var enc_buf: [16]u8 = undefined;
    var byte_v: u32 = 0;
    while (byte_v < 256) : (byte_v += 1) {
        const b: u8 = @intCast(byte_v);
        const encoded = b64.encode(&enc_buf, &[_]u8{b});
        if (byte_v > 0) try buf.appendSlice(allocator, ",");
        try buf.print(allocator,
            \\
            \\    {{ "rank": {d}, "token_bytes": "{s}", "token_str": "<0x{X:0>2}>" }}
        , .{ byte_v, encoded, b });
    }

    try buf.appendSlice(allocator,
        \\
        \\  ],
        \\  "special_tokens": [
        \\    { "rank": 0, "token_str": "<unk>",      "is_control": true },
        \\    { "rank": 1, "token_str": "<s>",        "is_control": true },
        \\    { "rank": 2, "token_str": "</s>",       "is_control": true },
        \\    { "rank": 3, "token_str": "[IMG]",      "is_control": true },
        \\    { "rank": 4, "token_str": "[IMG_BREAK]","is_control": true },
        \\    { "rank": 5, "token_str": "[IMG_END]",  "is_control": true }
        \\  ],
        \\  "image": {
        \\    "image_patch_size": 16,
        \\    "max_image_size": 1024,
        \\    "spatial_merge_size": 2
        \\  },
        \\  "audio": {
        \\    "sampling_rate": 16000,
        \\    "frame_rate": 12.5,
        \\    "chunk_length_s": 30.0,
        \\    "audio_encoding_config": {
        \\      "num_mel_bins": 128,
        \\      "hop_length": 160,
        \\      "window_size": 400
        \\    }
        \\  }
        \\}
    );

    return buf.toOwnedSlice(allocator);
}

test "loadTekkenBytes parses image and audio config" {
    const json = try buildMultimodalTekkenJson(testing.allocator);
    defer testing.allocator.free(json);

    var tk = try loadTekkenBytes(testing.allocator, json);
    defer tk.deinit();

    try testing.expect(tk.image_config != null);
    const img = tk.image_config.?;
    try testing.expectEqual(@as(u32, 16), img.image_patch_size);
    try testing.expectEqual(@as(u32, 1024), img.max_image_size);
    try testing.expectEqual(@as(u32, 2), img.spatial_merge_size);

    try testing.expect(tk.audio_config != null);
    const aud = tk.audio_config.?;
    try testing.expectEqual(@as(u32, 16000), aud.sampling_rate);
    try testing.expectApproxEqAbs(@as(f32, 12.5), aud.frame_rate, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 30.0), aud.chunk_length_s.?, 1e-6);
    try testing.expectEqual(@as(u32, 128), aud.encoding_config.num_mel_bins);
    try testing.expectEqual(@as(u32, 160), aud.encoding_config.hop_length);
    try testing.expectEqual(@as(u32, 400), aud.encoding_config.window_size);

    // Image specials surfaced through the helper.
    const ids = tk.specialImageIds();
    try testing.expectEqual(@as(?TokenId, 3), ids.img);
    try testing.expectEqual(@as(?TokenId, 4), ids.img_break);
    try testing.expectEqual(@as(?TokenId, 5), ids.img_end);
}

test "loadTekkenBytes accepts legacy 'multimodal' key for image config" {
    // Pre-v11 Pixtral files used 'multimodal' instead of 'image'.
    const json =
        \\{
        \\  "version": 7,
        \\  "type": "Tekkenizer",
        \\  "config": {
        \\    "pattern": ".",
        \\    "num_vocab_tokens": 1,
        \\    "default_vocab_size": 2,
        \\    "default_num_special_tokens": 1,
        \\    "version": "v7"
        \\  },
        \\  "vocab": [
        \\    { "rank": 0, "token_bytes": "AA==", "token_str": "<0x00>" }
        \\  ],
        \\  "special_tokens": [
        \\    { "rank": 0, "token_str": "<unk>", "is_control": true }
        \\  ],
        \\  "multimodal": {
        \\    "image_patch_size": 14,
        \\    "max_image_size": 448
        \\  }
        \\}
    ;
    var tk = try loadTekkenBytes(testing.allocator, json);
    defer tk.deinit();
    try testing.expect(tk.image_config != null);
    try testing.expectEqual(@as(u32, 14), tk.image_config.?.image_patch_size);
    try testing.expectEqual(@as(u32, 448), tk.image_config.?.max_image_size);
    // Default fills in when omitted.
    try testing.expectEqual(@as(u32, 1), tk.image_config.?.spatial_merge_size);
    try testing.expect(tk.audio_config == null);
}

test "placeholderImageTokens builds the expected grid" {
    const json = try buildMultimodalTekkenJson(testing.allocator);
    defer testing.allocator.free(json);

    var tk = try loadTekkenBytes(testing.allocator, json);
    defer tk.deinit();

    // image_patch_size=16, spatial_merge_size=2 → block = 32.
    // 32x32 input → grid 1x1 → tokens: [IMG] + [IMG_END] then the
    // trailing-break-becomes-end rule rewrites position (-1).
    // Sequence = ([IMG] * 1 + [IMG_BREAK]) * 1, then [-1] = [IMG_END].
    // Result: [IMG, IMG_END]  (length 2)
    {
        const ids = try tk.placeholderImageTokens(testing.allocator, .{ .width = 32, .height = 32 });
        defer testing.allocator.free(ids);
        try testing.expectEqual(@as(usize, 2), ids.len);
        try testing.expectEqual(@as(TokenId, 3), ids[0]); // [IMG]
        try testing.expectEqual(@as(TokenId, 5), ids[1]); // [IMG_END]
    }

    // 64x32 input → grid 2x1 → ([IMG, IMG, IMG_BREAK]) then [-1]=IMG_END
    // length = (2 + 1) * 1 = 3
    {
        const ids = try tk.placeholderImageTokens(testing.allocator, .{ .width = 64, .height = 32 });
        defer testing.allocator.free(ids);
        try testing.expectEqual(@as(usize, 3), ids.len);
        try testing.expectEqual(@as(TokenId, 3), ids[0]); // [IMG]
        try testing.expectEqual(@as(TokenId, 3), ids[1]); // [IMG]
        try testing.expectEqual(@as(TokenId, 5), ids[2]); // [IMG_END]
    }

    // 64x64 → grid 2x2 → length = 3 * 2 = 6
    // rows: [IMG IMG IMG_BREAK | IMG IMG IMG_BREAK], then [-1]=IMG_END.
    {
        const ids = try tk.placeholderImageTokens(testing.allocator, .{ .width = 64, .height = 64 });
        defer testing.allocator.free(ids);
        try testing.expectEqual(@as(usize, 6), ids.len);
        try testing.expectEqual(@as(TokenId, 3), ids[0]);
        try testing.expectEqual(@as(TokenId, 3), ids[1]);
        try testing.expectEqual(@as(TokenId, 4), ids[2]); // [IMG_BREAK]
        try testing.expectEqual(@as(TokenId, 3), ids[3]);
        try testing.expectEqual(@as(TokenId, 3), ids[4]);
        try testing.expectEqual(@as(TokenId, 5), ids[5]); // [IMG_END]
    }

    // Oversized image: 4096x2048 with max_image_size=1024 → ratio = 4
    // → downscale to (1024, 512) → grid = (1024/32, 512/32) = (32, 16)
    // → length = (32 + 1) * 16 = 528.
    {
        const ids = try tk.placeholderImageTokens(testing.allocator, .{ .width = 4096, .height = 2048 });
        defer testing.allocator.free(ids);
        try testing.expectEqual(@as(usize, 528), ids.len);
        try testing.expectEqual(@as(TokenId, 5), ids[527]); // last token = [IMG_END]
    }
}

test "imageTokenGrid matches Mistral's (w-1)/block + 1 formula on non-multiples" {
    const json = try buildMultimodalTekkenJson(testing.allocator);
    defer testing.allocator.free(json);

    var tk = try loadTekkenBytes(testing.allocator, json);
    defer tk.deinit();

    // 33x33 with block=32 → ceil(33/32) = 2 in each dim per the
    // (w-1)/block+1 formula: (33-1)/32+1 = 32/32+1 = 2. So grid is 2x2.
    const g = try tk.imageTokenGrid(.{ .width = 33, .height = 33 });
    try testing.expectEqual(@as(u32, 2), g.width);
    try testing.expectEqual(@as(u32, 2), g.height);

    // 1x1 → (0)/32+1 = 1 in each dim → 1x1 grid.
    const g_tiny = try tk.imageTokenGrid(.{ .width = 1, .height = 1 });
    try testing.expectEqual(@as(u32, 1), g_tiny.width);
    try testing.expectEqual(@as(u32, 1), g_tiny.height);
}

test "v6 text-only Tekken file still loads with null image/audio config" {
    // Regression: the original (text-only) buildMinimalTekkenJson path
    // must continue to work and produce nulls for the new optionals.
    const json = try buildMinimalTekkenJson(testing.allocator);
    defer testing.allocator.free(json);

    var tk = try loadTekkenBytes(testing.allocator, json);
    defer tk.deinit();

    try testing.expectEqual(@as(?ImageConfig, null), tk.image_config);
    try testing.expectEqual(@as(?AudioConfig, null), tk.audio_config);

    // And `placeholderImageTokens` errors out cleanly rather than
    // panicking when no image_config was loaded.
    try testing.expectError(
        error.InvalidImageConfig,
        tk.placeholderImageTokens(testing.allocator, .{ .width = 32, .height = 32 }),
    );

    // The image-special lookup returns all-nulls on a text-only file.
    const ids = tk.specialImageIds();
    try testing.expectEqual(@as(?TokenId, null), ids.img);
    try testing.expectEqual(@as(?TokenId, null), ids.img_break);
    try testing.expectEqual(@as(?TokenId, null), ids.img_end);
}

test "image config with missing required field is rejected" {
    const bad =
        \\{
        \\  "version": 7,
        \\  "type": "Tekkenizer",
        \\  "config": {
        \\    "pattern": ".",
        \\    "num_vocab_tokens": 1,
        \\    "default_vocab_size": 2,
        \\    "default_num_special_tokens": 1,
        \\    "version": "v7"
        \\  },
        \\  "vocab": [
        \\    { "rank": 0, "token_bytes": "AA==", "token_str": "<0x00>" }
        \\  ],
        \\  "special_tokens": [
        \\    { "rank": 0, "token_str": "<unk>", "is_control": true }
        \\  ],
        \\  "image": { "image_patch_size": 16 }
        \\}
    ;
    try testing.expectError(error.InvalidImageConfig, loadTekkenBytes(testing.allocator, bad));
}

// --- config.version fallback tests ---------------------------------------

test "loadTekkenBytes uses top-level version when present (config.version ignored)" {
    // Top-level version=7 wins over config.version="v99" — the latter
    // is purely fallback for older files lacking the integer field.
    // Use a tiny inline JSON so we can pin both fields explicitly.
    const json =
        \\{
        \\  "version": 7,
        \\  "type": "Tekkenizer",
        \\  "config": {
        \\    "pattern": ".",
        \\    "num_vocab_tokens": 1,
        \\    "default_vocab_size": 2,
        \\    "default_num_special_tokens": 1,
        \\    "version": "v99"
        \\  },
        \\  "vocab": [
        \\    { "rank": 0, "token_bytes": "AA==", "token_str": "<0x00>" }
        \\  ],
        \\  "special_tokens": [
        \\    { "rank": 0, "token_str": "<unk>", "is_control": true }
        \\  ]
        \\}
    ;
    var tk = try loadTekkenBytes(testing.allocator, json);
    defer tk.deinit();
    try testing.expectEqual(@as(u32, 7), tk.version);
}

test "loadTekkenBytes falls back to config.version when top-level absent" {
    // Mirrors the real Mistral-Nemo tekken.json: no top-level `version`,
    // version is only inside config as `"v3"`. Pre-fallback the loader
    // would strict-reject this with `error.MissingField`. v3 is also
    // <= 7 so `special_tokens` can be omitted.
    const json =
        \\{
        \\  "config": {
        \\    "pattern": ".",
        \\    "num_vocab_tokens": 1,
        \\    "default_vocab_size": 2,
        \\    "default_num_special_tokens": 1,
        \\    "version": "v3"
        \\  },
        \\  "vocab": [
        \\    { "rank": 0, "token_bytes": "AA==", "token_str": "<0x00>" }
        \\  ]
        \\}
    ;
    var tk = try loadTekkenBytes(testing.allocator, json);
    defer tk.deinit();
    try testing.expectEqual(@as(u32, 3), tk.version);
}

// --- encodeImage / encodeMultimodal tests --------------------------------

test "encodeImage materializes the same grid as placeholderImageTokens" {
    const json = try buildMultimodalTekkenJson(testing.allocator);
    defer testing.allocator.free(json);

    var tk = try loadTekkenBytes(testing.allocator, json);
    defer tk.deinit();

    // 64x64 input with patch=16 / merge=2 → block 32 → grid 2x2 →
    // sequence length (2+1) * 2 = 6, last token = [IMG_END] (id 5).
    const ids = try encodeImage(testing.allocator, &tk, .{ .width = 64, .height = 64 });
    defer testing.allocator.free(ids);

    try testing.expectEqual(@as(usize, 6), ids.len);
    try testing.expectEqual(@as(TokenId, 3), ids[0]); // [IMG]
    try testing.expectEqual(@as(TokenId, 3), ids[1]); // [IMG]
    try testing.expectEqual(@as(TokenId, 4), ids[2]); // [IMG_BREAK]
    try testing.expectEqual(@as(TokenId, 3), ids[3]); // [IMG]
    try testing.expectEqual(@as(TokenId, 3), ids[4]); // [IMG]
    try testing.expectEqual(@as(TokenId, 5), ids[5]); // [IMG_END]
}

test "encodeImage errors cleanly on text-only model" {
    // Text-only minimal fixture has no image config — encodeImage should
    // surface `InvalidImageConfig` rather than panic, AND must do so
    // before allocating an output buffer (no leak).
    const json = try buildMinimalTekkenJson(testing.allocator);
    defer testing.allocator.free(json);

    var tk = try loadTekkenBytes(testing.allocator, json);
    defer tk.deinit();

    // Three image specials are also absent here, so the early-out kicks
    // in first → MissingSpecialTokens.
    try testing.expectError(
        error.MissingSpecialTokens,
        encodeImage(testing.allocator, &tk, .{ .width = 32, .height = 32 }),
    );
}

test "encodeMultimodal interleaves text and image segments in order" {
    const json = try buildMultimodalTekkenJson(testing.allocator);
    defer testing.allocator.free(json);

    var tk = try loadTekkenBytes(testing.allocator, json);
    defer tk.deinit();

    // text "AB" → two bytes through the BPE (vocab has no merges in the
    // multimodal fixture beyond the 256 raw bytes), so 2 ids.
    // image 32x32 → 2-id placeholder ([IMG], [IMG_END]).
    // text "C"  → 1 id.
    // Total: 2 + 2 + 1 = 5 ids in the exact (text, img, text) order.
    const parts = [_]ContentPart{
        .{ .text = "AB" },
        .{ .image = .{ .width = 32, .height = 32 } },
        .{ .text = "C" },
    };
    const ids = try encodeMultimodal(testing.allocator, &tk, &parts);
    defer testing.allocator.free(ids);

    try testing.expectEqual(@as(usize, 5), ids.len);
    // Bytes A=0x41 / B=0x42 / C=0x43, shifted by 10 (num_special_tokens).
    try testing.expectEqual(@as(TokenId, 10 + 0x41), ids[0]);
    try testing.expectEqual(@as(TokenId, 10 + 0x42), ids[1]);
    try testing.expectEqual(@as(TokenId, 3), ids[2]); // [IMG]
    try testing.expectEqual(@as(TokenId, 5), ids[3]); // [IMG_END]
    try testing.expectEqual(@as(TokenId, 10 + 0x43), ids[4]);
}

test "encodeMultimodal handles multiple consecutive images" {
    const json = try buildMultimodalTekkenJson(testing.allocator);
    defer testing.allocator.free(json);

    var tk = try loadTekkenBytes(testing.allocator, json);
    defer tk.deinit();

    // Three back-to-back 32x32 images. Each expands to [IMG, IMG_END]
    // (length 2), so the concatenated stream is 6 ids and every even
    // index is [IMG] (3), every odd index is [IMG_END] (5).
    const parts = [_]ContentPart{
        .{ .image = .{ .width = 32, .height = 32 } },
        .{ .image = .{ .width = 32, .height = 32 } },
        .{ .image = .{ .width = 32, .height = 32 } },
    };
    const ids = try encodeMultimodal(testing.allocator, &tk, &parts);
    defer testing.allocator.free(ids);

    try testing.expectEqual(@as(usize, 6), ids.len);
    var i: usize = 0;
    while (i < ids.len) : (i += 2) {
        try testing.expectEqual(@as(TokenId, 3), ids[i]);
        try testing.expectEqual(@as(TokenId, 5), ids[i + 1]);
    }
}

test "loadTekkenFile smoke-loads real mistral-nemo tekken.json (config.version fallback)" {
    // Skip if the bench vocab isn't available. The fixture is too big
    // to ship in a normal test corpus, but when present this catches
    // regressions in the config.version fallback against a real
    // 130k-token Tekken file.
    const path = "bench/vocabs/mistral_nemo_tekken.json";
    const io = std.Io.Threaded.global_single_threaded.io();
    std.Io.Dir.cwd().access(io, path, .{}) catch return error.SkipZigTest;

    var tk = try loadTekkenFile(testing.allocator, path);
    defer tk.deinit();

    // Real file is `config.version = "v3"`, no top-level version.
    try testing.expectEqual(@as(u32, 3), tk.version);
    try testing.expect(tk.num_special_tokens > 0);
    try testing.expect(tk.bpe.count >= tk.num_special_tokens + 256);
}

test "encodeMultimodal on empty content yields empty output" {
    const json = try buildMultimodalTekkenJson(testing.allocator);
    defer testing.allocator.free(json);

    var tk = try loadTekkenBytes(testing.allocator, json);
    defer tk.deinit();

    // Both empty-text segments and an empty content slice must produce
    // a zero-length, freeable result without touching the allocator
    // beyond the toOwnedSlice() call.
    const parts: []const ContentPart = &.{};
    const ids = try encodeMultimodal(testing.allocator, &tk, parts);
    defer testing.allocator.free(ids);
    try testing.expectEqual(@as(usize, 0), ids.len);

    const empty_text = [_]ContentPart{ .{ .text = "" }, .{ .text = "" } };
    const ids2 = try encodeMultimodal(testing.allocator, &tk, &empty_text);
    defer testing.allocator.free(ids2);
    try testing.expectEqual(@as(usize, 0), ids2.len);
}

// --- audio config tests --------------------------------------------------

/// Build a Voxtral-shaped tekken.json: minimal vocab plus an `audio`
/// config section and the two audio specials (`[BEGIN_AUDIO]`, `[AUDIO]`)
/// so `placeholderAudioTokens` has IDs to emit. Mirrors
/// `buildMultimodalTekkenJson` but for the audio path; also keeps the
/// image specials + image config so the unified interleave test can use
/// text + image + audio in one file.
fn buildAudioTekkenJson(allocator: std.mem.Allocator) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);

    try buf.appendSlice(allocator,
        \\{
        \\  "version": 11,
        \\  "type": "Tekkenizer",
        \\  "config": {
        \\    "pattern": ".",
        \\    "num_vocab_tokens": 256,
        \\    "default_vocab_size": 266,
        \\    "default_num_special_tokens": 10,
        \\    "version": "v11"
        \\  },
        \\  "vocab": [
    );

    const b64 = std.base64.standard.Encoder;
    var enc_buf: [16]u8 = undefined;
    var byte_v: u32 = 0;
    while (byte_v < 256) : (byte_v += 1) {
        const b: u8 = @intCast(byte_v);
        const encoded = b64.encode(&enc_buf, &[_]u8{b});
        if (byte_v > 0) try buf.appendSlice(allocator, ",");
        try buf.print(allocator,
            \\
            \\    {{ "rank": {d}, "token_bytes": "{s}", "token_str": "<0x{X:0>2}>" }}
        , .{ byte_v, encoded, b });
    }

    // Specials: text controls + image trio + audio pair.
    // [BEGIN_AUDIO] = id 6, [AUDIO] = id 7.
    try buf.appendSlice(allocator,
        \\
        \\  ],
        \\  "special_tokens": [
        \\    { "rank": 0, "token_str": "<unk>",        "is_control": true },
        \\    { "rank": 1, "token_str": "<s>",          "is_control": true },
        \\    { "rank": 2, "token_str": "</s>",         "is_control": true },
        \\    { "rank": 3, "token_str": "[IMG]",        "is_control": true },
        \\    { "rank": 4, "token_str": "[IMG_BREAK]",  "is_control": true },
        \\    { "rank": 5, "token_str": "[IMG_END]",    "is_control": true },
        \\    { "rank": 6, "token_str": "[BEGIN_AUDIO]","is_control": true },
        \\    { "rank": 7, "token_str": "[AUDIO]",      "is_control": true }
        \\  ],
        \\  "image": {
        \\    "image_patch_size": 16,
        \\    "max_image_size": 1024,
        \\    "spatial_merge_size": 2
        \\  },
        \\  "audio": {
        \\    "sampling_rate": 16000,
        \\    "frame_rate": 12.5,
        \\    "chunk_length_s": 30.0,
        \\    "audio_encoding_config": {
        \\      "num_mel_bins": 128,
        \\      "hop_length": 160,
        \\      "window_size": 400
        \\    }
        \\  }
        \\}
    );

    return buf.toOwnedSlice(allocator);
}

test "audioTokenCount matches mistral_common arithmetic" {
    const json = try buildAudioTekkenJson(testing.allocator);
    defer testing.allocator.free(json);

    var tk = try loadTekkenBytes(testing.allocator, json);
    defer tk.deinit();

    // sampling_rate=16000, frame_rate=12.5, hop_length=160.
    //   raw_audio_length_per_tok = floor(16000/12.5) = 1280
    //   audio_length_per_tok     = int(1280/160)     = 8
    //
    // 1s clip = 16000 samples (divisible by hop): signal_len = 16000/160
    // = 100 → ceil(100/8) = 13 placeholder [AUDIO] tokens.
    try testing.expectEqual(@as(u64, 13), try tk.audioTokenCount(16000));

    // Non-divisible: 16161 samples (16161 % 160 = 1):
    //   signal_len = ceil(16161/160 - 1) = ceil(101.00625 - 1) = 101
    //   ceil(101/8) = 13.
    try testing.expectEqual(@as(u64, 13), try tk.audioTokenCount(16161));

    // Zero-length clip → 0 placeholder tokens.
    try testing.expectEqual(@as(u64, 0), try tk.audioTokenCount(0));

    // Exactly one token's worth: 8 frames * 160 hop = 1280 samples →
    // signal_len = 8 → ceil(8/8) = 1.
    try testing.expectEqual(@as(u64, 1), try tk.audioTokenCount(1280));
}

test "audio specials surface through specialAudioIds" {
    const json = try buildAudioTekkenJson(testing.allocator);
    defer testing.allocator.free(json);

    var tk = try loadTekkenBytes(testing.allocator, json);
    defer tk.deinit();

    const ids = tk.specialAudioIds();
    try testing.expectEqual(@as(?TokenId, 6), ids.begin_audio);
    try testing.expectEqual(@as(?TokenId, 7), ids.audio);
}

test "encodeAudio materializes [BEGIN_AUDIO] + [AUDIO]*n" {
    const json = try buildAudioTekkenJson(testing.allocator);
    defer testing.allocator.free(json);

    var tk = try loadTekkenBytes(testing.allocator, json);
    defer tk.deinit();

    // 1s (16000 samples) → 13 [AUDIO] + 1 [BEGIN_AUDIO] = 14 ids.
    const ids = try encodeAudio(testing.allocator, &tk, .{ .num_samples = 16000 });
    defer testing.allocator.free(ids);

    try testing.expectEqual(@as(usize, 14), ids.len);
    try testing.expectEqual(@as(TokenId, 6), ids[0]); // [BEGIN_AUDIO]
    var i: usize = 1;
    while (i < ids.len) : (i += 1) {
        try testing.expectEqual(@as(TokenId, 7), ids[i]); // [AUDIO]
    }

    // duration_s path: 1.0s * 16000 = 16000 samples → identical length.
    const ids_dur = try encodeAudio(testing.allocator, &tk, .{ .duration_s = 1.0 });
    defer testing.allocator.free(ids_dur);
    try testing.expectEqual(@as(usize, 14), ids_dur.len);

    // num_samples wins when both supplied (16000 samples, duration ignored).
    const ids_both = try encodeAudio(
        testing.allocator,
        &tk,
        .{ .num_samples = 16000, .duration_s = 999.0 },
    );
    defer testing.allocator.free(ids_both);
    try testing.expectEqual(@as(usize, 14), ids_both.len);

    // Empty clip → just [BEGIN_AUDIO].
    const ids_empty = try encodeAudio(testing.allocator, &tk, .{});
    defer testing.allocator.free(ids_empty);
    try testing.expectEqual(@as(usize, 1), ids_empty.len);
    try testing.expectEqual(@as(TokenId, 6), ids_empty[0]);
}

test "encodeMultimodal interleaves text, image, and audio in order" {
    const json = try buildAudioTekkenJson(testing.allocator);
    defer testing.allocator.free(json);

    var tk = try loadTekkenBytes(testing.allocator, json);
    defer tk.deinit();

    // text "AB"          → 2 ids (raw bytes 0x41,0x42 + 10).
    // image 32x32        → [IMG, IMG_END] = 2 ids (3, 5).
    // audio 1280 samples → [BEGIN_AUDIO] + [AUDIO]*1 = 2 ids (6, 7).
    // text "C"           → 1 id (0x43 + 10).
    // Total: 2 + 2 + 2 + 1 = 7 ids in exact order.
    const parts = [_]ContentPart{
        .{ .text = "AB" },
        .{ .image = .{ .width = 32, .height = 32 } },
        .{ .audio = .{ .num_samples = 1280 } },
        .{ .text = "C" },
    };
    const ids = try encodeMultimodal(testing.allocator, &tk, &parts);
    defer testing.allocator.free(ids);

    try testing.expectEqual(@as(usize, 7), ids.len);
    try testing.expectEqual(@as(TokenId, 10 + 0x41), ids[0]); // A
    try testing.expectEqual(@as(TokenId, 10 + 0x42), ids[1]); // B
    try testing.expectEqual(@as(TokenId, 3), ids[2]); // [IMG]
    try testing.expectEqual(@as(TokenId, 5), ids[3]); // [IMG_END]
    try testing.expectEqual(@as(TokenId, 6), ids[4]); // [BEGIN_AUDIO]
    try testing.expectEqual(@as(TokenId, 7), ids[5]); // [AUDIO]
    try testing.expectEqual(@as(TokenId, 10 + 0x43), ids[6]); // C
}

test "encodeMultimodal handles consecutive audio parts" {
    const json = try buildAudioTekkenJson(testing.allocator);
    defer testing.allocator.free(json);

    var tk = try loadTekkenBytes(testing.allocator, json);
    defer tk.deinit();

    // Two back-to-back 1280-sample clips, each [BEGIN_AUDIO, AUDIO].
    const parts = [_]ContentPart{
        .{ .audio = .{ .num_samples = 1280 } },
        .{ .audio = .{ .num_samples = 1280 } },
    };
    const ids = try encodeMultimodal(testing.allocator, &tk, &parts);
    defer testing.allocator.free(ids);

    try testing.expectEqual(@as(usize, 4), ids.len);
    try testing.expectEqual(@as(TokenId, 6), ids[0]);
    try testing.expectEqual(@as(TokenId, 7), ids[1]);
    try testing.expectEqual(@as(TokenId, 6), ids[2]);
    try testing.expectEqual(@as(TokenId, 7), ids[3]);
}

test "encodeAudio errors cleanly when audio config is missing" {
    // Text-only minimal fixture has no `audio` section at all. encodeAudio
    // must surface InvalidAudioConfig (no audio_config) without allocating.
    const json = try buildMinimalTekkenJson(testing.allocator);
    defer testing.allocator.free(json);

    var tk = try loadTekkenBytes(testing.allocator, json);
    defer tk.deinit();

    try testing.expect(tk.audio_config == null);
    try testing.expectError(
        error.InvalidAudioConfig,
        encodeAudio(testing.allocator, &tk, .{ .num_samples = 16000 }),
    );

    // And via the unified API: audio content against an audio-less model.
    const parts = [_]ContentPart{
        .{ .text = "A" },
        .{ .audio = .{ .num_samples = 16000 } },
    };
    try testing.expectError(
        error.InvalidAudioConfig,
        encodeMultimodal(testing.allocator, &tk, &parts),
    );
}

test "encodeAudio errors when audio config present but specials missing" {
    // Audio config but the [BEGIN_AUDIO]/[AUDIO] specials aren't in the
    // table → MissingSpecialTokens, raised before allocating output.
    const json =
        \\{
        \\  "version": 11,
        \\  "type": "Tekkenizer",
        \\  "config": {
        \\    "pattern": ".",
        \\    "num_vocab_tokens": 1,
        \\    "default_vocab_size": 2,
        \\    "default_num_special_tokens": 1,
        \\    "version": "v11"
        \\  },
        \\  "vocab": [
        \\    { "rank": 0, "token_bytes": "AA==", "token_str": "<0x00>" }
        \\  ],
        \\  "special_tokens": [
        \\    { "rank": 0, "token_str": "<unk>", "is_control": true }
        \\  ],
        \\  "audio": {
        \\    "sampling_rate": 16000,
        \\    "frame_rate": 12.5,
        \\    "audio_encoding_config": {
        \\      "num_mel_bins": 128,
        \\      "hop_length": 160,
        \\      "window_size": 400
        \\    }
        \\  }
        \\}
    ;
    var tk = try loadTekkenBytes(testing.allocator, json);
    defer tk.deinit();

    try testing.expect(tk.audio_config != null);
    try testing.expectError(
        error.MissingSpecialTokens,
        encodeAudio(testing.allocator, &tk, .{ .num_samples = 16000 }),
    );
}

test {
    // Pull in the Tekken pre-tokenizer's unit tests.
    _ = pretok;
}
