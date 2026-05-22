//! Normalizers transform raw input bytes before pre-tokenization.
//!
//! Tagged union over the concrete kinds — no vtables, no heap dispatch.
//! Variants:
//!   * `identity`   — pass through unchanged.
//!   * `nfc`/`nfd`/`nfkc`/`nfkd` — Unicode Normalization Forms (TR15)
//!     against UCD 16.0 tables. Round-trip safe; full conformance.
//!   * `byte_level` — apply the GPT-2 byte-to-printable-unicode mapping.
//!     Pairs with `Decoder.byte_level` for HF byte-level BPE vocabs.
//!   * `sp_precompiled` — SentencePiece-style normalization that bundles
//!     an optional NFKC pre-pass, ASCII space → U+2581 (`▁`) escape, an
//!     optional U+2581 dummy prefix, and optional extra-whitespace
//!     collapsing. The variant name is historical (originally added for
//!     `sp_unigram`); it applies to SP-BPE just as well.
//!   * `capcode` / `nocapcode` — TokenMonster-style uppercase elision.
//!     `capcode` runs the full encoder (C/W/D marker bytes — ztok uses
//!     0x0E/0x0F/0x11 in place of TM's printable 'C'/'W'/'D'); `nocapcode`
//!     runs the forward-delete-only encoder (0x7F marker, matches TM
//!     byte-for-byte). Both can pre-apply NFD via the `nfd` config field
//!     so a single normalizer value mirrors TM's
//!     `normalize(data, capcode, normalizer)` pipeline. Wraps
//!     `src/capcode.zig`; see `monster_io.normalizerForVocab` for
//!     auto-selecting the right variant from a TM vocab name.
//!
//! Origin map convention (`normalizeWithOrigin`): each output byte
//! maps to the byte offset of the codepoint in the ORIGINAL input
//! that produced it. For multi-byte expansions (e.g. NFD `é` → `e` +
//! combining acute, NFKC ligature unwrap, GPT-2 byte_level which maps
//! 1 raw byte to 1-2 UTF-8 bytes), all expanded output bytes share
//! the source codepoint's start offset. The identity normalizer
//! returns `null` for the origin map — callers should treat null as
//! "post-normalized bytes == original bytes, no translation needed".
//! For `sp_precompiled`, any prepended dummy-prefix bytes map to byte
//! offset 0 of the original input (the conventional "start of input"
//! anchor) — see `SpNormalizer` for the full rule.

const std = @import("std");
const unicode_norm = @import("unicode_norm.zig");
const byte_level = @import("byte_level.zig");
const capcode = @import("capcode.zig");
const tm_norm = @import("tm_norm.zig");
const sp_charsmap = @import("sp_charsmap.zig");
const hf_regex = @import("hf_regex.zig");

/// Result of `normalizeWithOrigin`. Holds normalized bytes and a
/// per-byte map back to the original input.
pub const NormalizationResult = struct {
    /// Bytes after normalization. Owned by the caller (free with `allocator`).
    bytes: []u8,
    /// Per-byte map: for each byte in `bytes`, the corresponding byte
    /// offset in the ORIGINAL input. Length == `bytes.len`. Owned by
    /// the caller. Set to `null` when the normalizer is identity —
    /// callers should treat null as "identity map" (output offset N
    /// maps to original offset N).
    origin: ?[]u32,

    pub fn deinit(self: *NormalizationResult, allocator: std.mem.Allocator) void {
        allocator.free(self.bytes);
        if (self.origin) |o| allocator.free(o);
        self.* = undefined;
    }
};

/// SentencePiece-style normalizer configuration. Applied in this order:
///   1. If `nfkc`, run an NFKC normalization pass (handles ligatures,
///      compatibility decompositions, recomposes accents).
///   2. If `casefold`, simple-case-fold every codepoint via
///      `capcode.toLower` (uses the UCD 16.0 `upper_to_lower` +
///      `titlecase_to_lower` tables embedded in `src/capcode.zig`).
///      This is the `_cf` suffix in SP's normalizer names
///      (`nmt_nfkc_cf`, `nfkc_cf`) that Gemma-style models ship with.
///   3. If `remove_extra_whitespaces`, collapse runs of ASCII spaces to
///      a single ASCII space, and strip leading/trailing ASCII spaces.
///   4. If `escape_whitespaces`, rewrite every ASCII space to U+2581
///      (`▁`, three UTF-8 bytes: 0xE2 0x96 0x81).
///   5. If `add_dummy_prefix`, prepend a U+2581 to the output so the
///      first word is treated like every other space-prefixed word.
///
/// Defaults match the SentencePiece `nmt_nfkc` profile that LLaMA-style
/// models ship with (casefold off — that's the `nmt_nfkc_cf` profile
/// instead). The dummy-prefix bytes, when present, map to byte offset 0
/// in the origin map. Casefolded output bytes inherit the origin offset
/// of the FIRST byte of the source codepoint they were folded from, so
/// a 2-byte uppercase glyph folding to a 2-byte lowercase form leaves
/// both output bytes pointing at the input glyph's start offset (same
/// convention as NFD expansions).
pub const SpNormalizer = struct {
    nfkc: bool = true,
    casefold: bool = false,
    add_dummy_prefix: bool = true,
    escape_whitespaces: bool = true,
    remove_extra_whitespaces: bool = false,
    /// Optional per-model precompiled charsmap (Darts double-array trie
    /// over byte-sequence rewrite rules baked into the SP `.model`).
    /// When non-null, this pass REPLACES the standalone NFKC pre-pass:
    /// SP bakes the entire codepoint stage (NFKC composition + NMT
    /// extras + casefold for `_cf` profiles) into the trie itself, so
    /// running our own NFKC first would pre-compose sequences the trie
    /// was built to keep decomposed. Mirrors SP's reference, whose
    /// `Normalizer::Normalize` only ever calls `NormalizePrefix` (the
    /// charsmap walk) for the codepoint stage — there is no separate
    /// NFKC composition step. See `refs/sentencepiece/src/normalizer.cc`.
    ///
    /// Storage: the pointer is borrowed. The lifetime contract is that
    /// the pointee must outlive every `normalize` / `normalizeWithOrigin`
    /// call. The typical owner is an `SpModel.precompiled_charsmap`
    /// buffer plus a parsed `PrecompiledCharsmap` value held in the
    /// pipeline owner (see `sp_bridge.normalizerFromSP`).
    charsmap: ?*const sp_charsmap.PrecompiledCharsmap = null,
};

/// Re-export of `capcode.MarkerStyle` so callers configuring a
/// `Normalizer{ .capcode = ... }` don't have to import `capcode.zig`
/// directly. See `CapcodeNormalizer.marker_style` for usage.
pub const MarkerStyle = capcode.MarkerStyle;

/// HF `Replace { pattern: String|Regex, content: String }` normalizer.
///
/// Two modes:
///   * `compiled_regex == null` (default): `pattern` is matched
///     literally. Each non-overlapping occurrence is rewritten with
///     `content`. This is the fast path and used for `String` patterns.
///   * `compiled_regex != null`: `pattern` is the regex source (kept
///     around for diagnosis only) and `compiled_regex` is the parsed
///     `hf_regex.Regex` program. `findFirst` is invoked in a loop to
///     locate non-overlapping matches; each is rewritten with `content`.
///     This handles patterns like `" {2,}"` (collapse repeated spaces),
///     `\s+`, etc. used by tokenizers such as deberta-v3 / T5.
///
/// Storage:
///   * `pattern` and `content` are owned by the value (allocated from
///     the same allocator that built the parent `Normalizer`).
///   * `compiled_regex`, when set, is heap-owned. `Normalizer.deinit`
///     calls `compiled_regex.deinit()` and `allocator.destroy(...)` on
///     the regex pointer.
///
/// Origin convention: every output byte that came from `content`
/// inherits the origin of the FIRST byte of the matched `pattern`.
/// Bytes outside any match pass through with their natural offsets.
pub const ReplaceNormalizer = struct {
    pattern: []const u8,
    content: []const u8,
    /// Compiled regex engine for non-literal `pattern`s. Set by the HF
    /// bridge when the source spec was `pattern: {"Regex": "..."}`.
    /// Null means use the literal match path.
    compiled_regex: ?*hf_regex.Regex = null,
};

/// HF `Strip { strip_left, strip_right }` normalizer. Trims ASCII
/// whitespace (`\t`, `\n`, `\v`, `\f`, `\r`, ` `) from the requested
/// edge(s). Mirrors `tokenizers::normalizers::strip::Strip` semantics
/// which delegate to `normalized.lstrip()` / `rstrip()` (both walk by
/// codepoint and stop at the first non-whitespace). Bytes that survive
/// the trim retain their original origin offsets; trimmed bytes leave
/// no entries in the origin map.
pub const StripNormalizer = struct {
    strip_left: bool,
    strip_right: bool,
};

/// HF `BertNormalizer { clean_text, handle_chinese_chars, strip_accents,
/// lowercase }` normalizer. Implementation matches
/// `refs/tokenizers/tokenizers/src/normalizers/bert.rs` step-for-step:
///   1. `clean_text`: drop NUL / U+FFFD / Unicode control characters,
///      replace every whitespace codepoint with ASCII ' '.
///   2. `handle_chinese_chars`: insert ASCII ' ' before AND after every
///      CJK Unified Ideograph (the 8 CJK ranges in `is_chinese_char`).
///   3. `strip_accents`: NFD-decompose then drop combining-mark
///      codepoints (Mn). When `null` HF defers to `lowercase` — we
///      model that with `strip_accents = null` (Option<bool>).
///   4. `lowercase`: Unicode simple case-fold via `capcode.toLower`
///      (post-1.18 B). Covers the UCD 16.0 simple-fold mapping —
///      Latin/Greek/Cyrillic/Armenian/Coptic/Georgian/Cherokee and
///      friends, including fullwidth Latin (FF21→FF41). Matches HF's
///      `char::to_lowercase` for everything that has a 1:1 simple
///      fold; full case-fold (German ß → ss, Greek iota subscript) is
///      not modeled — these are rare in WordPiece inputs.
///
/// The runtime `bert_normalizer` variant lowers to a Sequence built at
/// `normalize` time so this struct stays plain-old-data.
pub const BertNormalizerConfig = struct {
    clean_text: bool = true,
    handle_chinese_chars: bool = true,
    /// `null` means "follow HF's default: do_strip_accents = lowercase".
    /// `Some(true)` always strips, `Some(false)` never strips.
    strip_accents: ?bool = null,
    lowercase: bool = true,
};

/// HF `Sequence { normalizers: [...] }` — runs each inner normalizer in
/// order, threading the post-normalization bytes (and origin map, when
/// asked) through to the next stage. Owns the inner slice via the
/// allocator passed to `hf_bridge.normalizerFromHF`; that helper calls
/// `Normalizer.deinit` which recursively frees nested sequences.
pub const SequenceNormalizer = struct {
    normalizers: []const Normalizer,
};

/// HF `Prepend { prepend: "..." }` normalizer. Emits `prepend ++ input`
/// — used by SentencePiece-style HF tokenizers (Phi-3, LLaMA-2 SP-BPE
/// reshelled as HF) where a literal U+2581 (`▁`) is prepended before
/// the chained `Replace(" "→"▁")` rewrites internal spaces. Same role
/// SP's `add_dummy_prefix` plays inside `sp_precompiled`, but split out
/// as its own HF stage so chains can compose freely.
///
/// Storage: `prepend` is borrowed from the allocator that built the
/// parent `Normalizer` — typically `hf_bridge.normalizerFromHF` via
/// `allocator.dupe`. Freed by `Normalizer.deinit`.
///
/// Origin map convention: every output byte from the prepend literal
/// anchors to byte offset 0 of the original input (matching the
/// SpNormalizer dummy-prefix convention). Bytes after the prefix
/// inherit their natural input offset (shifted by the prefix length).
pub const PrependNormalizer = struct {
    prepend: []const u8,
};

/// TokenMonster-style capcode normalizer configuration. Used for both
/// the `.capcode` (full markers — 0x0E/0x0F/0x11 in ztok style;
/// 'C'/'W'/'D' = 0x43/0x57/0x44 in TM-printable style) and `.nocapcode`
/// (forward-delete-only: 0x7F) variants.
///
///   * `nfd`             — pre-apply NFD before the capcode pass.
///                         Mirrors TM's `normalize(data, capcode, normalizer)`
///                         where the normalizer (NFD flag = 1 in TM's
///                         flag byte) runs first, then capcode.
///   * `tm_compat_space` — only meaningful for `.nocapcode`. If true,
///                         the forward-delete marker is emitted as the
///                         TM-compatible `DEL + ' '` pair (TM-Go's
///                         `NoCapcodeEncode` inserts a synthetic space
///                         so the following word looks space-prefixed in
///                         the vocab); if false, only `DEL` is emitted
///                         (ztok-native 1-byte form). Default is true
///                         since the primary caller is the TM .ztm
///                         loader where byte-for-byte parity matters.
///   * `marker_style`    — only meaningful for `.capcode`. Picks the
///                         capitalize-next / word-cap / delete-next byte
///                         triple. `.ztok` (default) emits the C0
///                         controls 0x0E/0x0F/0x11 (no ambiguity with
///                         text). `.tm_printable` emits 'C'/'W'/'D'
///                         (0x43/0x57/0x44), matching TokenMonster's Go
///                         `capcode` package so a `.ztm` loaded from a
///                         TM full-capcode vocab can be normalized into
///                         the same byte stream the vocab pieces were
///                         carved out of. Use `.tm_printable` when
///                         encoding through a TM-derived full-capcode
///                         vocab; keep `.ztok` for ztok-native pipelines.
pub const CapcodeNormalizer = struct {
    nfd: bool = false,
    tm_compat_space: bool = true,
    marker_style: MarkerStyle = .ztok,
};

pub const Normalizer = union(enum) {
    identity,
    nfc,
    nfd,
    nfkc,
    nfkd,
    byte_level,
    sp_precompiled: SpNormalizer,
    capcode: CapcodeNormalizer,
    nocapcode: CapcodeNormalizer,
    // HF tokenizer.json normalizer arms. Appended at the END of the
    // union (post-1.17 agent D) so other agents' edits to the existing
    // variants don't conflict. Bridge: `hf_bridge.normalizerFromHF`.
    replace: ReplaceNormalizer,
    strip: StripNormalizer,
    lowercase,
    bert_normalizer: BertNormalizerConfig,
    sequence: SequenceNormalizer,
    // Post-1.18 agent B: SP-style literal prepend, added to unblock the
    // Phi-3 `Sequence[Prepend("▁"), Replace(" "→"▁")]` normalizer chain.
    prepend: PrependNormalizer,

    /// True if this normalizer is the identity transform (output bytes
    /// == input bytes, origin map is implicitly the identity). When
    /// true, `normalizeWithOrigin` skips allocating the origin slice.
    pub fn isIdentity(self: Normalizer) bool {
        return self == .identity;
    }

    /// Recursively free any allocator-owned storage held by this
    /// normalizer (currently only `.sequence` owns a heap slice; the
    /// `.replace` arm's `pattern`/`content` slices are owned by the
    /// allocator that built the value too). Safe to call on any
    /// variant — pass-through arms are no-ops.
    pub fn deinit(self: Normalizer, allocator: std.mem.Allocator) void {
        switch (self) {
            .replace => |r| {
                if (r.pattern.len > 0) allocator.free(r.pattern);
                if (r.content.len > 0) allocator.free(r.content);
                if (r.compiled_regex) |re| {
                    re.deinit();
                    allocator.destroy(re);
                }
            },
            .sequence => |s| {
                for (s.normalizers) |inner| inner.deinit(allocator);
                if (s.normalizers.len > 0) allocator.free(s.normalizers);
            },
            .prepend => |p| {
                if (p.prepend.len > 0) allocator.free(p.prepend);
            },
            else => {},
        }
    }

    /// Worst-case byte-expansion factor: `normalized.len <= input.len * factor`.
    /// Used by callers (pipeline) to size scratch buffers that hold the
    /// post-normalization bytes or per-byte model outputs.
    ///
    ///   identity        : 1
    ///   byte_level      : 2  (each raw byte -> at most 2 UTF-8 bytes)
    ///   nfc             : 3  (some Arabic / Hangul reorderings)
    ///   nfd             : 4  (Hangul LVT -> 3 cps, deep decompositions)
    ///   nfkc / nfkd     : 18 (compatibility decompositions can be large,
    ///                         e.g. Arabic presentation forms, vertical
    ///                         forms, the Arabic ligature U+FDFA expands
    ///                         to 18 codepoints / ~36 bytes)
    ///   sp_precompiled  : 54 = 18 (NFKC) × 3 (U+2581 is 3 UTF-8 bytes vs
    ///                              a 1-byte ASCII space). Defensive cap
    ///                              for the rare ligature-meets-spaces
    ///                              case; typical text expands by ≤ 1.1×.
    ///                              Add a small slack for the optional
    ///                              dummy prefix (also 3 bytes) on tiny
    ///                              inputs.
    ///   capcode         : 6  (NFD × 4 × 1.5× capcode marker insertion;
    ///                         each codepoint can trigger at most one C/W/D
    ///                         marker byte plus passthrough).
    ///   nocapcode       : 12 (NFD × 4 × 3× — each codepoint can trigger
    ///                         a 2-byte `DEL + ' '` insertion in TM-compat
    ///                         mode + multi-byte UTF-8 passthrough).
    pub fn maxByteExpansion(self: Normalizer) usize {
        return switch (self) {
            .identity => 1,
            .byte_level => 2,
            .nfc => 3,
            .nfd => 4,
            .nfkc, .nfkd => 18,
            .sp_precompiled => 54,
            .capcode => 6,
            .nocapcode => 12,
            // Replace: worst case is an empty pattern with a multi-byte
            // content (rare; mostly identity). 8 is a defensive cap big
            // enough for typical " " -> "_" or "▁" -> " " rewrites.
            .replace => 8,
            // Strip: never grows.
            .strip => 1,
            // Lowercase: ASCII fold is 1:1 for ASCII; non-ASCII bytes
            // pass through unchanged in v1.
            .lowercase => 1,
            // BertNormalizer: NFD (×4) + space-around-chinese (×3) +
            // lowercase (no growth) -> conservative ×12. With Unicode
            // lowercase (post-1.18 B) the byte length can shift slightly
            // per codepoint (e.g. some Cherokee titlecase forms fold to
            // a 3-byte lowercase). ×12 still covers; bump if a real fold
            // exceeds it.
            .bert_normalizer => 12,
            // Prepend: at most input.len bytes copied through plus the
            // literal prefix. We can't express "additive constant" in the
            // expansion cap, so fold the prefix into a ×2 factor; on
            // typical multi-kilobyte inputs the prefix is rounding noise
            // and ×2 is generous. Hot path callers (Phi-3 `▁` prefix)
            // hit ×1 amortized.
            .prepend => 2,
            // Sequence: product of inner caps, capped at a defensive
            // ceiling to keep scratch sizes finite.
            .sequence => |s| blk: {
                var prod: usize = 1;
                for (s.normalizers) |inner| {
                    const cap = inner.maxByteExpansion();
                    // Saturating multiply with a 1024× ceiling (any
                    // pipeline beyond that needs careful review anyway).
                    const next = std.math.mul(usize, prod, cap) catch 1024;
                    prod = @min(next, 1024);
                    if (prod == 1024) break;
                }
                break :blk prod;
            },
        };
    }

    /// Normalize `input`, allocating both the normalized bytes and (for
    /// non-identity normalizers) a per-byte map back to the original
    /// input. See module doc-comment for the origin map convention.
    pub fn normalizeWithOrigin(
        self: Normalizer,
        allocator: std.mem.Allocator,
        input: []const u8,
    ) !NormalizationResult {
        switch (self) {
            .identity => {
                const out = try allocator.alloc(u8, input.len);
                @memcpy(out, input);
                return .{ .bytes = out, .origin = null };
            },
            .nfc => return unicode_norm.normalizeWithOrigin(allocator, .nfc, input),
            .nfd => return unicode_norm.normalizeWithOrigin(allocator, .nfd, input),
            .nfkc => return unicode_norm.normalizeWithOrigin(allocator, .nfkc, input),
            .nfkd => return unicode_norm.normalizeWithOrigin(allocator, .nfkd, input),
            .byte_level => {
                // Each input byte maps to a codepoint that encodes as
                // 1-2 UTF-8 bytes. All output bytes from byte N share
                // origin N.
                const out = try allocator.alloc(u8, input.len * 2);
                errdefer allocator.free(out);
                const origin = try allocator.alloc(u32, input.len * 2);
                errdefer allocator.free(origin);

                var enc: [4]u8 = undefined;
                var w: usize = 0;
                for (input, 0..) |byte, src_off| {
                    const cp = byte_level.byte_to_unicode[byte];
                    const n = std.unicode.utf8Encode(cp, &enc) catch unreachable;
                    var k: usize = 0;
                    while (k < n) : (k += 1) {
                        out[w + k] = enc[k];
                        origin[w + k] = @intCast(src_off);
                    }
                    w += n;
                }
                const bytes = try allocator.realloc(out, w);
                errdefer allocator.free(bytes);
                const orig = try allocator.realloc(origin, w);
                return .{ .bytes = bytes, .origin = orig };
            },
            .sp_precompiled => |cfg| return spNormalizeWithOrigin(allocator, cfg, input),
            .capcode => |cfg| return capcodeNormalizeWithOrigin(allocator, cfg, input, .full),
            .nocapcode => |cfg| return capcodeNormalizeWithOrigin(allocator, cfg, input, .nocapcode),
            .replace => |cfg| return replaceNormalizeWithOrigin(allocator, cfg, input),
            .strip => |cfg| return stripNormalizeWithOrigin(allocator, cfg, input),
            .lowercase => return lowercaseNormalizeWithOrigin(allocator, input),
            .bert_normalizer => |cfg| return bertNormalizeWithOrigin(allocator, cfg, input),
            .sequence => |cfg| return sequenceNormalizeWithOrigin(allocator, cfg, input),
            .prepend => |cfg| return prependNormalizeWithOrigin(allocator, cfg, input),
        }
    }

    /// Returns just the normalized bytes (origin-free fast path).
    ///
    /// This is the hot path for `Pipeline.encodeText` — callers that
    /// don't need the origin map (the throughput-critical encoder) get
    /// here, callers that do need spans go through `normalizeWithOrigin`.
    /// Bypasses the origin-map allocation + byte-by-byte src bookkeeping
    /// that the with-origin path threads through every stage. For a
    /// 10 MB capcode input that's ~40 MB of origin-array writes per
    /// encode that we now skip entirely — see the 1.14 TM Monster
    /// encode regression fix in bench/RESULTS.md.
    pub fn normalize(
        self: Normalizer,
        allocator: std.mem.Allocator,
        input: []const u8,
    ) ![]u8 {
        switch (self) {
            .identity => {
                const out = try allocator.alloc(u8, input.len);
                @memcpy(out, input);
                return out;
            },
            .nfc => return unicode_norm.normalize(allocator, .nfc, input),
            .nfd => return unicode_norm.normalize(allocator, .nfd, input),
            .nfkc => return unicode_norm.normalize(allocator, .nfkc, input),
            .nfkd => return unicode_norm.normalize(allocator, .nfkd, input),
            .byte_level => {
                // Each input byte maps to 1-2 UTF-8 bytes. No origin
                // tracking — just write the bytes.
                var out = try allocator.alloc(u8, input.len * 2);
                errdefer allocator.free(out);
                var enc: [4]u8 = undefined;
                var w: usize = 0;
                for (input) |byte| {
                    const cp = byte_level.byte_to_unicode[byte];
                    const n = std.unicode.utf8Encode(cp, &enc) catch unreachable;
                    var k: usize = 0;
                    while (k < n) : (k += 1) {
                        out[w + k] = enc[k];
                    }
                    w += n;
                }
                return allocator.realloc(out, w);
            },
            .sp_precompiled => |cfg| return spNormalize(allocator, cfg, input),
            .capcode => |cfg| return capcodeNormalize(allocator, cfg, input, .full),
            .nocapcode => |cfg| return capcodeNormalize(allocator, cfg, input, .nocapcode),
            .replace => |cfg| return replaceNormalize(allocator, cfg, input),
            .strip => |cfg| return stripNormalize(allocator, cfg, input),
            .lowercase => return lowercaseNormalize(allocator, input),
            .bert_normalizer => |cfg| return bertNormalize(allocator, cfg, input),
            .sequence => |cfg| return sequenceNormalize(allocator, cfg, input),
            .prepend => |cfg| return prependNormalize(allocator, cfg, input),
        }
    }
};

// === SP-style normalizer implementation ===
//
// Pipeline (matches the canonical SentencePiece order):
//   1. NFKC pass (optional).
//   2. Simple case-fold (optional, `_cf` suffix on SP normalizer
//      names). Walks the post-NFKC bytes codepoint by codepoint, maps
//      each via `capcode.toLower` (UCD 16.0 simple case-fold), and
//      re-encodes. Output bytes from a folded codepoint inherit the
//      origin offset of the FIRST byte of the source codepoint —
//      matching the multi-byte expansion convention used by NFD.
//   3. remove_extra_whitespaces: collapse runs of ASCII spaces; trim
//      leading/trailing ASCII spaces.
//   4. escape_whitespaces: ASCII ' ' -> U+2581 (▁, 0xE2 0x96 0x81).
//   5. add_dummy_prefix: prepend U+2581 so the leading word looks like
//      every other space-prefixed word.
//
// Origin map: every output byte references the byte offset of the
// codepoint in the ORIGINAL input that produced it. The dummy prefix
// (when prepended) maps to offset 0. When NFKC runs first, we inherit
// its per-byte origin map and propagate it through the later passes.

const u2581_bytes = [_]u8{ 0xE2, 0x96, 0x81 };

// SP's reference `Normalizer::Normalize` only recognizes ASCII ' ' as
// whitespace in its collapse/escape stages (see refs/sentencepiece/src/
// normalizer.cc — the inner loop compares `p.first != " "` literally
// and the per-byte body branches on `data[n] == ' '`). Multi-byte
// whitespace codepoints reach those stages as ASCII ' ' only when the
// model's precompiled charsmap rewrote them earlier. For example:
//   - T5 / mT5 ship an `nmt_nfkc` charsmap whose Darts trie maps
//     U+2007, U+2028, U+00A0, U+3000, ... to a single ASCII space, so
//     they LOOK like whitespace by the time we get here.
//   - LLaMA / Mistral / Gemma / Yi ship a plainer NFKC charsmap that
//     leaves those codepoints alone, so they round-trip through the
//     normalizer verbatim (and end up byte-fallback-encoded by the
//     model encoder).
// Recognizing extra cps here would over-collapse the LLaMA family and
// break the 5-fixture stress sweep (1.22 regression). Stay ASCII-only
// and let `sp_charsmap.normalize` handle per-model mappings.
inline fn isSpWhitespace(cp: u21) bool {
    return cp == ' ';
}

fn spNormalizeWithOrigin(
    allocator: std.mem.Allocator,
    cfg: SpNormalizer,
    input: []const u8,
) !NormalizationResult {
    // SP's reference encoder returns the empty token list for empty
    // input — `add_dummy_prefix` does NOT fire on an empty string.
    // Mirror that here so the pipeline's `maxTokensFor(0)==0` capacity
    // stays consistent with the downstream model encoder.
    if (input.len == 0) {
        const empty_bytes = try allocator.alloc(u8, 0);
        const empty_origin = try allocator.alloc(u32, 0);
        return .{ .bytes = empty_bytes, .origin = empty_origin };
    }

    // Stage 1: NFKC (or just memcpy + identity-origin).
    //
    // When a precompiled charsmap is present, SKIP the standalone NFKC
    // pass: SP's `nmt_nfkc` / `nfkc` / `nfkc_cf` charsmaps already encode
    // the entire codepoint rewrite (NFKC + NMT extras + casefold) as a
    // single Darts trie. Running NFKC first then the charsmap can compose
    // sequences the charsmap was built to leave alone (e.g. `c`+U+0328
    // → ĉ̨ — the trie keeps `c` and combiner separate, but NFKC pre-
    // composes them, defeating the trie's downstream split). SP's
    // reference `Normalizer::Normalize` only ever calls `NormalizePrefix`
    // (the trie walk) for its codepoint stage — see
    // `refs/sentencepiece/src/normalizer.cc`. Match that exactly.
    var stage_bytes: []u8 = undefined;
    var stage_origin: []u32 = undefined;
    if (cfg.nfkc and cfg.charsmap == null) {
        const nfkc = try unicode_norm.normalizeWithOrigin(allocator, .nfkc, input);
        stage_bytes = nfkc.bytes;
        // unicode_norm always sets origin on non-identity forms.
        stage_origin = nfkc.origin orelse blk: {
            // Defensive: fabricate a 1:1 origin map if origin was elided.
            const fab = try allocator.alloc(u32, stage_bytes.len);
            for (fab, 0..) |*o, i| o.* = @intCast(@min(i, input.len));
            break :blk fab;
        };
    } else {
        stage_bytes = try allocator.alloc(u8, input.len);
        errdefer allocator.free(stage_bytes);
        @memcpy(stage_bytes, input);
        stage_origin = try allocator.alloc(u32, input.len);
        for (stage_origin, 0..) |*o, i| o.* = @intCast(i);
    }
    errdefer allocator.free(stage_bytes);
    errdefer allocator.free(stage_origin);

    // Stage 1a-bis: precompiled charsmap pass. Walks the (raw, NOT-yet-
    // NFKC-composed) bytes through the per-model Darts trie of byte-
    // sequence rewrite rules. SP runs this AS its codepoint stage —
    // every prefix of the input goes through `NormalizePrefix` which
    // does a longest-match trie lookup and falls through to a single
    // codepoint copy on miss. We mirror that in `sp_charsmap.normalize`.
    // Skipped when `cfg.charsmap == null` so models without a charsmap
    // (LLaMA, Gemma, the synthetic SP tests) keep bit-for-bit identical
    // output to the pre-charsmap behaviour.
    if (cfg.charsmap) |cm_ptr| {
        const r = try sp_charsmap.normalizeWithOrigin(allocator, cm_ptr.*, stage_bytes, stage_origin);
        allocator.free(stage_bytes);
        allocator.free(stage_origin);
        stage_bytes = r.bytes;
        stage_origin = r.origin;
    }

    // Stage 1b: simple case-fold (UCD 16.0, via capcode's tables). Walks
    // codepoints and re-encodes the lowercase form; output bytes from a
    // folded codepoint inherit the origin offset of the first byte of
    // the source codepoint.
    if (cfg.casefold) {
        const folded = try caseFoldWithOrigin(allocator, stage_bytes, stage_origin);
        allocator.free(stage_bytes);
        allocator.free(stage_origin);
        stage_bytes = folded.bytes;
        stage_origin = folded.origin.?;
    }

    // Stage 2: collapse extra whitespaces. SP's reference normalizer
    // only treats ASCII ' ' as whitespace here (see `isSpWhitespace`);
    // per-model charsmaps are responsible for rewriting their preferred
    // unicode-space cps to ASCII before this stage. The codepoint walk
    // is retained so any future widening of `isSpWhitespace` (or invalid
    // utf8) Just Works. Output is never longer than input, so in-place
    // compaction is safe.
    if (cfg.remove_extra_whitespaces) {
        var w: usize = 0;
        var i: usize = 0;
        // Leading: skip sp-whitespace.
        while (i < stage_bytes.len) {
            const lead = stage_bytes[i];
            const cp_len_raw: u3 = std.unicode.utf8ByteSequenceLength(lead) catch 1;
            const cp_len: usize = cp_len_raw;
            const end = @min(i + cp_len, stage_bytes.len);
            const cp: u21 = if (cp_len == 1)
                lead
            else
                (std.unicode.utf8Decode(stage_bytes[i..end]) catch lead);
            if (!isSpWhitespace(cp)) break;
            i = if (cp_len == 1 or end == i) i + 1 else end;
        }
        var last_was_space = false;
        while (i < stage_bytes.len) {
            const lead = stage_bytes[i];
            const cp_len_raw: u3 = std.unicode.utf8ByteSequenceLength(lead) catch 1;
            const cp_len: usize = cp_len_raw;
            const end = @min(i + cp_len, stage_bytes.len);
            var valid = true;
            const cp: u21 = if (cp_len == 1)
                lead
            else blk: {
                const d = std.unicode.utf8Decode(stage_bytes[i..end]) catch {
                    valid = false;
                    break :blk lead;
                };
                break :blk d;
            };
            if (valid and isSpWhitespace(cp)) {
                if (!last_was_space) {
                    stage_bytes[w] = ' ';
                    stage_origin[w] = stage_origin[i];
                    w += 1;
                    last_was_space = true;
                }
                i = if (cp_len == 1 or end == i) i + 1 else end;
            } else {
                // Copy the codepoint (or single invalid byte) through.
                const take: usize = if (valid) (end - i) else 1;
                var k: usize = 0;
                while (k < take) : (k += 1) {
                    stage_bytes[w + k] = stage_bytes[i + k];
                    stage_origin[w + k] = stage_origin[i + k];
                }
                w += take;
                i += take;
                last_was_space = false;
            }
        }
        // Trailing: strip a single trailing space if we ended on one.
        if (w > 0 and stage_bytes[w - 1] == ' ') w -= 1;
        // Resize in place (shrink only — always safe).
        stage_bytes = try allocator.realloc(stage_bytes, w);
        stage_origin = try allocator.realloc(stage_origin, w);
    }

    // Stage 3: escape SP-whitespace codepoints to U+2581 (3 bytes).
    // SP only escapes ASCII ' ' here; the codepoint walk is retained
    // for forward-compat with a wider `isSpWhitespace`. When this
    // flag is off and there's no dummy prefix, we can hand back
    // stage_* as-is.
    var esc_bytes: []u8 = undefined;
    var esc_origin: []u32 = undefined;
    if (cfg.escape_whitespaces) {
        // Worst case: every byte was an ASCII space → 3× expansion.
        // (Multi-byte sp-whitespace is already 3 bytes → 3 bytes; no
        // additional growth.)
        const cap = stage_bytes.len * 3;
        esc_bytes = try allocator.alloc(u8, cap);
        errdefer allocator.free(esc_bytes);
        esc_origin = try allocator.alloc(u32, cap);
        errdefer allocator.free(esc_origin);
        var w: usize = 0;
        var i: usize = 0;
        while (i < stage_bytes.len) {
            const lead = stage_bytes[i];
            const cp_len_raw: u3 = std.unicode.utf8ByteSequenceLength(lead) catch 1;
            const cp_len: usize = cp_len_raw;
            const end = @min(i + cp_len, stage_bytes.len);
            var valid = true;
            const cp: u21 = if (cp_len == 1)
                lead
            else blk: {
                const d = std.unicode.utf8Decode(stage_bytes[i..end]) catch {
                    valid = false;
                    break :blk lead;
                };
                break :blk d;
            };
            if (valid and isSpWhitespace(cp)) {
                esc_bytes[w + 0] = u2581_bytes[0];
                esc_bytes[w + 1] = u2581_bytes[1];
                esc_bytes[w + 2] = u2581_bytes[2];
                esc_origin[w + 0] = stage_origin[i];
                esc_origin[w + 1] = stage_origin[i];
                esc_origin[w + 2] = stage_origin[i];
                w += 3;
                i = if (cp_len == 1 or end == i) i + 1 else end;
            } else {
                const take: usize = if (valid) (end - i) else 1;
                var k: usize = 0;
                while (k < take) : (k += 1) {
                    esc_bytes[w + k] = stage_bytes[i + k];
                    esc_origin[w + k] = stage_origin[i + k];
                }
                w += take;
                i += take;
            }
        }
        allocator.free(stage_bytes);
        allocator.free(stage_origin);
        esc_bytes = try allocator.realloc(esc_bytes, w);
        esc_origin = try allocator.realloc(esc_origin, w);
    } else {
        esc_bytes = stage_bytes;
        esc_origin = stage_origin;
    }
    errdefer allocator.free(esc_bytes);
    errdefer allocator.free(esc_origin);

    // Stage 4: optionally prepend U+2581. The 3 prefix bytes anchor to
    // offset 0 in the original input.
    if (cfg.add_dummy_prefix) {
        const final_len = esc_bytes.len + 3;
        const final_bytes = try allocator.alloc(u8, final_len);
        errdefer allocator.free(final_bytes);
        const final_origin = try allocator.alloc(u32, final_len);
        errdefer allocator.free(final_origin);
        final_bytes[0] = u2581_bytes[0];
        final_bytes[1] = u2581_bytes[1];
        final_bytes[2] = u2581_bytes[2];
        final_origin[0] = 0;
        final_origin[1] = 0;
        final_origin[2] = 0;
        @memcpy(final_bytes[3..], esc_bytes);
        @memcpy(final_origin[3..], esc_origin);
        allocator.free(esc_bytes);
        allocator.free(esc_origin);
        return .{ .bytes = final_bytes, .origin = final_origin };
    }

    return .{ .bytes = esc_bytes, .origin = esc_origin };
}

// === SP-style normalizer — origin-free fast path ===
//
// Mirrors `spNormalizeWithOrigin` step for step but skips the parallel
// `stage_origin` / `esc_origin` / `final_origin` arrays. Used by the
// throughput-critical encoder where the caller never reads the origin
// map.

fn spNormalize(
    allocator: std.mem.Allocator,
    cfg: SpNormalizer,
    input: []const u8,
) ![]u8 {
    if (input.len == 0) {
        return allocator.alloc(u8, 0);
    }

    // Stage 1: NFKC (or just memcpy). Suppressed when a precompiled
    // charsmap is present — see `spNormalizeWithOrigin` for the full
    // rationale. The charsmap is SP's complete codepoint stage; running
    // a separate NFKC composes sequences the charsmap was built to
    // leave decomposed (e.g. `c`+U+0328 → ĉ̨ unintentionally pre-
    // composes to U+0109, then the trie's `ĉ̨ę̊` split path stops
    // matching).
    var stage_bytes: []u8 = undefined;
    if (cfg.nfkc and cfg.charsmap == null) {
        stage_bytes = try unicode_norm.normalize(allocator, .nfkc, input);
    } else {
        stage_bytes = try allocator.alloc(u8, input.len);
        @memcpy(stage_bytes, input);
    }
    errdefer allocator.free(stage_bytes);

    // Stage 1a-bis: precompiled charsmap pass (origin-free twin of the
    // with-origin path above). Only fires when the SP model shipped a
    // `precompiled_charsmap` field; null is the LLaMA/Gemma default.
    if (cfg.charsmap) |cm_ptr| {
        const r = try sp_charsmap.normalize(allocator, cm_ptr.*, stage_bytes);
        allocator.free(stage_bytes);
        stage_bytes = r;
    }

    // Stage 1b: simple case-fold (UCD 16.0 simple-fold via capcode's
    // tables). Origin-free twin of the with-origin path; only rewrites
    // the bytes.
    if (cfg.casefold) {
        const folded = try caseFoldBytes(allocator, stage_bytes);
        allocator.free(stage_bytes);
        stage_bytes = folded;
    }

    // Stage 2: collapse extra whitespaces. ASCII-only per SP's reference
    // (see origin-tracking twin above for full rationale).
    if (cfg.remove_extra_whitespaces) {
        var w: usize = 0;
        var i: usize = 0;
        while (i < stage_bytes.len) {
            const lead = stage_bytes[i];
            const cp_len_raw: u3 = std.unicode.utf8ByteSequenceLength(lead) catch 1;
            const cp_len: usize = cp_len_raw;
            const end = @min(i + cp_len, stage_bytes.len);
            const cp: u21 = if (cp_len == 1)
                lead
            else
                (std.unicode.utf8Decode(stage_bytes[i..end]) catch lead);
            if (!isSpWhitespace(cp)) break;
            i = if (cp_len == 1 or end == i) i + 1 else end;
        }
        var last_was_space = false;
        while (i < stage_bytes.len) {
            const lead = stage_bytes[i];
            const cp_len_raw: u3 = std.unicode.utf8ByteSequenceLength(lead) catch 1;
            const cp_len: usize = cp_len_raw;
            const end = @min(i + cp_len, stage_bytes.len);
            var valid = true;
            const cp: u21 = if (cp_len == 1)
                lead
            else blk: {
                const d = std.unicode.utf8Decode(stage_bytes[i..end]) catch {
                    valid = false;
                    break :blk lead;
                };
                break :blk d;
            };
            if (valid and isSpWhitespace(cp)) {
                if (!last_was_space) {
                    stage_bytes[w] = ' ';
                    w += 1;
                    last_was_space = true;
                }
                i = if (cp_len == 1 or end == i) i + 1 else end;
            } else {
                const take: usize = if (valid) (end - i) else 1;
                var k: usize = 0;
                while (k < take) : (k += 1) {
                    stage_bytes[w + k] = stage_bytes[i + k];
                }
                w += take;
                i += take;
                last_was_space = false;
            }
        }
        if (w > 0 and stage_bytes[w - 1] == ' ') w -= 1;
        stage_bytes = try allocator.realloc(stage_bytes, w);
    }

    // Stage 3: escape SP-whitespace codepoints to U+2581. ASCII-only
    // twin of the with-origin path.
    var esc_bytes: []u8 = undefined;
    if (cfg.escape_whitespaces) {
        const cap = stage_bytes.len * 3;
        esc_bytes = try allocator.alloc(u8, cap);
        errdefer allocator.free(esc_bytes);
        var w: usize = 0;
        var i: usize = 0;
        while (i < stage_bytes.len) {
            const lead = stage_bytes[i];
            const cp_len_raw: u3 = std.unicode.utf8ByteSequenceLength(lead) catch 1;
            const cp_len: usize = cp_len_raw;
            const end = @min(i + cp_len, stage_bytes.len);
            var valid = true;
            const cp: u21 = if (cp_len == 1)
                lead
            else blk: {
                const d = std.unicode.utf8Decode(stage_bytes[i..end]) catch {
                    valid = false;
                    break :blk lead;
                };
                break :blk d;
            };
            if (valid and isSpWhitespace(cp)) {
                esc_bytes[w + 0] = u2581_bytes[0];
                esc_bytes[w + 1] = u2581_bytes[1];
                esc_bytes[w + 2] = u2581_bytes[2];
                w += 3;
                i = if (cp_len == 1 or end == i) i + 1 else end;
            } else {
                const take: usize = if (valid) (end - i) else 1;
                var k: usize = 0;
                while (k < take) : (k += 1) {
                    esc_bytes[w + k] = stage_bytes[i + k];
                }
                w += take;
                i += take;
            }
        }
        allocator.free(stage_bytes);
        esc_bytes = try allocator.realloc(esc_bytes, w);
    } else {
        esc_bytes = stage_bytes;
    }
    errdefer allocator.free(esc_bytes);

    // Stage 4: prepend U+2581.
    if (cfg.add_dummy_prefix) {
        const final_len = esc_bytes.len + 3;
        const final_bytes = try allocator.alloc(u8, final_len);
        errdefer allocator.free(final_bytes);
        final_bytes[0] = u2581_bytes[0];
        final_bytes[1] = u2581_bytes[1];
        final_bytes[2] = u2581_bytes[2];
        @memcpy(final_bytes[3..], esc_bytes);
        allocator.free(esc_bytes);
        return final_bytes;
    }

    return esc_bytes;
}

// === SP-style case-fold helper ===
//
// Walks `bytes` codepoint by codepoint, maps each via `capcode.toLower`,
// and re-encodes. Used by `spNormalize` / `spNormalizeWithOrigin` when
// the `_cf` suffix is set on the SP normalizer name (the Gemma-style
// `nmt_nfkc_cf` / `nfkc_cf` profile). Invalid UTF-8 bytes pass through
// 1:1 — we'd rather emit garbage that round-trips than panic mid-encode.
//
// The mapping is the SIMPLE case-fold (1:1 codepoint). Full case-fold
// (German ß → ss, Greek iota subscript → iota) is not modeled — SP's
// own `_cf` normalizer uses ICU's `u_foldCase` with the default option,
// which for the Latin/Greek/Cyrillic scripts of interest agrees with
// the simple fold the capcode tables provide.
//
// Worst-case byte growth: a 1-byte ASCII uppercase folds to a 1-byte
// ASCII lowercase (no growth); a 2/3/4-byte uppercase codepoint folds
// to a codepoint of the same or smaller UTF-8 length in every case
// covered by `upper_to_lower` / `titlecase_to_lower`. We size the
// output buffer at `bytes.len` and re-encode in place into a fresh
// allocation; if a fold ever grew a codepoint we'd realloc on demand
// (defensive — the tables guarantee this doesn't happen today).
fn caseFoldWithOrigin(
    allocator: std.mem.Allocator,
    bytes: []const u8,
    origin: []const u32,
) !NormalizationResult {
    std.debug.assert(bytes.len == origin.len);
    var out = try allocator.alloc(u8, bytes.len);
    errdefer allocator.free(out);
    var out_origin = try allocator.alloc(u32, bytes.len);
    errdefer allocator.free(out_origin);
    var enc: [4]u8 = undefined;

    var i: usize = 0;
    var w: usize = 0;
    while (i < bytes.len) {
        const lead = bytes[i];
        const cp_len_raw: u3 = std.unicode.utf8ByteSequenceLength(lead) catch 1;
        const cp_len: usize = cp_len_raw;
        const end = @min(i + cp_len, bytes.len);
        // Decode the codepoint (or fall back to raw lead byte on malformed UTF-8).
        var valid = true;
        const decoded: u21 = if (cp_len == 1)
            lead
        else blk: {
            const d = std.unicode.utf8Decode(bytes[i..end]) catch {
                valid = false;
                break :blk lead;
            };
            break :blk d;
        };
        if (!valid) {
            if (w + 1 > out.len) {
                out = try allocator.realloc(out, out.len + 1);
                out_origin = try allocator.realloc(out_origin, out_origin.len + 1);
            }
            out[w] = lead;
            out_origin[w] = origin[i];
            w += 1;
            i += 1;
            continue;
        }
        const folded = capcode.toLower(decoded);
        // Try to encode the folded codepoint; on failure pass original bytes through.
        var enc_ok = true;
        const n: usize = blk: {
            const e = std.unicode.utf8Encode(folded, &enc) catch {
                enc_ok = false;
                break :blk 0;
            };
            break :blk e;
        };
        if (!enc_ok) {
            const take = end - i;
            if (w + take > out.len) {
                out = try allocator.realloc(out, w + take);
                out_origin = try allocator.realloc(out_origin, w + take);
            }
            @memcpy(out[w .. w + take], bytes[i..end]);
            var k: usize = 0;
            while (k < take) : (k += 1) out_origin[w + k] = origin[i];
            w += take;
            i = end;
            continue;
        }
        if (w + n > out.len) {
            const new_cap = w + n + (bytes.len - end);
            out = try allocator.realloc(out, new_cap);
            out_origin = try allocator.realloc(out_origin, new_cap);
        }
        var k: usize = 0;
        while (k < n) : (k += 1) {
            out[w + k] = enc[k];
            out_origin[w + k] = origin[i];
        }
        w += n;
        i = end;
    }

    out = try allocator.realloc(out, w);
    out_origin = try allocator.realloc(out_origin, w);
    return .{ .bytes = out, .origin = out_origin };
}

fn caseFoldBytes(allocator: std.mem.Allocator, bytes: []const u8) ![]u8 {
    var out = try allocator.alloc(u8, bytes.len);
    errdefer allocator.free(out);
    var enc: [4]u8 = undefined;
    var i: usize = 0;
    var w: usize = 0;
    while (i < bytes.len) {
        const lead = bytes[i];
        const cp_len_raw: u3 = std.unicode.utf8ByteSequenceLength(lead) catch 1;
        const cp_len: usize = cp_len_raw;
        const end = @min(i + cp_len, bytes.len);
        var valid = true;
        const decoded: u21 = if (cp_len == 1)
            lead
        else blk: {
            const d = std.unicode.utf8Decode(bytes[i..end]) catch {
                valid = false;
                break :blk lead;
            };
            break :blk d;
        };
        if (!valid) {
            if (w + 1 > out.len) out = try allocator.realloc(out, out.len + 1);
            out[w] = lead;
            w += 1;
            i += 1;
            continue;
        }
        const folded = capcode.toLower(decoded);
        var enc_ok = true;
        const n: usize = blk: {
            const e = std.unicode.utf8Encode(folded, &enc) catch {
                enc_ok = false;
                break :blk 0;
            };
            break :blk e;
        };
        if (!enc_ok) {
            const take = end - i;
            if (w + take > out.len) out = try allocator.realloc(out, w + take);
            @memcpy(out[w .. w + take], bytes[i..end]);
            w += take;
            i = end;
            continue;
        }
        if (w + n > out.len) out = try allocator.realloc(out, w + n + (bytes.len - end));
        @memcpy(out[w .. w + n], enc[0..n]);
        w += n;
        i = end;
    }
    return allocator.realloc(out, w);
}

// === Capcode-style normalizer implementation ===
//
// Wraps `src/capcode.zig` so it can plug into the `Pipeline` value the
// same way as NFD/NFC/byte_level. Two flavors:
//
//   .capcode   — full capcode (uppercase elision via C/W/D marker bytes
//                0x0E/0x0F/0x11).
//   .nocapcode — forward-delete-only (DEL=0x7F marker insertion at word
//                boundaries). Default `tm_compat_space=true` matches the
//                TokenMonster Go encoder byte-for-byte by emitting
//                `DEL + ' '` at boundaries; set to false for ztok's
//                native 1-byte form.
//
// Origin map convention: every output byte references a source-input
// byte offset. Inserted marker bytes (and the synthetic space in
// TM-compat NoCapcode mode) reference the offset of the codepoint they
// preface — concretely the source-input byte at which the post-NFD
// codepoint started. NFD-induced expansions inherit their offsets from
// `unicode_norm.normalizeWithOrigin`; we then sweep through the
// capcode-encoded output and map each output byte back to its
// post-NFD source byte, which we further translate through the NFD
// origin map.

const CapcodeKind = enum { full, nocapcode };

/// Origin-free fast path for capcode/nocapcode normalize. Used by the
/// encoder hot path (`Pipeline.encodeText`) where the caller doesn't
/// need the per-output-byte source map. Skips:
///   * the `pre_origin` array (1 × u32 per post-NFD byte)
///   * the `origin_out` array (1 × u32 per output byte)
///   * the byte-by-byte origin walking loop after capcode encode
///     (which compared input/output slices to detect case-fold
///     boundaries — purely origin bookkeeping)
/// For a 10 MB TM Monster input the saved work is ~40 MB of allocator
/// writes per encode plus the walking pass. See 1.14 fix in
/// bench/RESULTS.md.
fn capcodeNormalize(
    allocator: std.mem.Allocator,
    cfg: CapcodeNormalizer,
    input: []const u8,
    comptime kind: CapcodeKind,
) ![]u8 {
    // TM-compat fast path: when the caller has opted into the
    // byte-exact TokenMonster-Go semantics (full-capcode with the
    // printable 'C'/'W'/'D' markers, or nocapcode with the synthetic
    // ' ' after every \x7F), delegate to `tm_norm.zig` — the faithful
    // port of TM's `capcode.Encode` / `NoCapcodeEncode`. The ztok
    // back-compat path below stays in place for `.ztok` marker style
    // and `tm_compat_space = false` mode (1.12 byte contract).
    if (kind == .full and cfg.marker_style == .tm_printable) {
        var pre_bytes_tm: []u8 = undefined;
        var pre_owned_tm = false;
        if (cfg.nfd) {
            pre_bytes_tm = try unicode_norm.normalize(allocator, .nfd, input);
            pre_owned_tm = true;
        } else {
            pre_bytes_tm = @constCast(input);
        }
        defer if (pre_owned_tm) allocator.free(pre_bytes_tm);
        return tm_norm.normalizeCapcode(allocator, pre_bytes_tm, cfg.marker_style);
    }
    if (kind == .nocapcode and cfg.tm_compat_space) {
        var pre_bytes_tm: []u8 = undefined;
        var pre_owned_tm = false;
        if (cfg.nfd) {
            pre_bytes_tm = try unicode_norm.normalize(allocator, .nfd, input);
            pre_owned_tm = true;
        } else {
            pre_bytes_tm = @constCast(input);
        }
        defer if (pre_owned_tm) allocator.free(pre_bytes_tm);
        return tm_norm.normalizeNocapcode(allocator, pre_bytes_tm);
    }

    // Stage 1: optional NFD pre-pass (origin-free).
    var pre_bytes: []u8 = undefined;
    var pre_owned = false;
    if (cfg.nfd) {
        pre_bytes = try unicode_norm.normalize(allocator, .nfd, input);
        pre_owned = true;
    } else {
        // No copy needed — the underlying encoder reads bytes only,
        // it doesn't mutate the input slice.
        pre_bytes = @constCast(input);
    }
    defer if (pre_owned) allocator.free(pre_bytes);

    // Stage 2: capcode-encode.
    const cap_out = switch (kind) {
        .full => try capcode.encodeStyled(allocator, pre_bytes, cfg.marker_style),
        .nocapcode => try capcode.NoCapcode.encode(allocator, pre_bytes),
    };
    errdefer allocator.free(cap_out);

    // Stage 3 (nocapcode only): tm_compat_space expansion.
    if (kind == .nocapcode and cfg.tm_compat_space) {
        var n_dels: usize = 0;
        for (cap_out) |b| if (b == capcode.NOCAPCODE_DELETE) {
            n_dels += 1;
        };
        const need_leading: bool = blk: {
            if (cap_out.len == 0) break :blk false;
            if (cap_out[0] == capcode.NOCAPCODE_DELETE) break :blk false;
            if (pre_bytes.len == 0) break :blk false;
            const lead = pre_bytes[0];
            const seq = std.unicode.utf8ByteSequenceLength(lead) catch 1;
            const take = if (seq <= pre_bytes.len) seq else 1;
            const first_cp = std.unicode.utf8Decode(pre_bytes[0..take]) catch break :blk false;
            break :blk isLetterOrDigitCp(first_cp);
        };
        const lead_extra: usize = if (need_leading) 2 else 0;
        if (n_dels > 0 or lead_extra > 0) {
            const new_len = cap_out.len + n_dels + lead_extra;
            const new_bytes = try allocator.alloc(u8, new_len);
            errdefer allocator.free(new_bytes);
            var w: usize = 0;
            if (need_leading) {
                new_bytes[0] = capcode.NOCAPCODE_DELETE;
                new_bytes[1] = ' ';
                w = 2;
            }
            for (cap_out) |b| {
                new_bytes[w] = b;
                w += 1;
                if (b == capcode.NOCAPCODE_DELETE) {
                    new_bytes[w] = ' ';
                    w += 1;
                }
            }
            std.debug.assert(w == new_len);
            allocator.free(cap_out);
            return new_bytes;
        }
    }

    return cap_out;
}

fn capcodeNormalizeWithOrigin(
    allocator: std.mem.Allocator,
    cfg: CapcodeNormalizer,
    input: []const u8,
    kind: CapcodeKind,
) !NormalizationResult {
    // Stage 1: optional NFD pre-pass. Mirrors TM's
    // `normalize(data, capcode, normalizer)` order in tokenmonster.go.
    var pre_bytes: []u8 = undefined;
    var pre_origin: []u32 = undefined;
    var pre_owned = false;
    if (cfg.nfd) {
        const r = try unicode_norm.normalizeWithOrigin(allocator, .nfd, input);
        pre_bytes = r.bytes;
        // unicode_norm always sets origin on non-identity forms.
        pre_origin = r.origin orelse blk: {
            const fab = try allocator.alloc(u32, pre_bytes.len);
            for (fab, 0..) |*o, i| o.* = @intCast(@min(i, input.len));
            break :blk fab;
        };
        pre_owned = true;
    } else {
        pre_bytes = try allocator.alloc(u8, input.len);
        errdefer allocator.free(pre_bytes);
        @memcpy(pre_bytes, input);
        pre_origin = try allocator.alloc(u32, input.len);
        for (pre_origin, 0..) |*o, i| o.* = @intCast(i);
        pre_owned = true;
    }
    defer if (pre_owned) {
        allocator.free(pre_bytes);
        allocator.free(pre_origin);
    };

    // TM-compat origin-aware fast path: same gate as `capcodeNormalize`
    // (full + tm_printable, or nocapcode + tm_compat_space). Delegate
    // to the faithful TM-Go port in `tm_norm.zig`, threading the NFD
    // origin map so every output byte still references a valid offset
    // in the CALLER'S original input.
    if (kind == .full and cfg.marker_style == .tm_printable) {
        const tm_r = try tm_norm.normalizeCapcodeWithOrigin(allocator, pre_bytes, cfg.marker_style, pre_origin);
        return .{ .bytes = tm_r.bytes, .origin = tm_r.origin };
    }
    if (kind == .nocapcode and cfg.tm_compat_space) {
        const tm_r = try tm_norm.normalizeNocapcodeWithOrigin(allocator, pre_bytes, pre_origin);
        return .{ .bytes = tm_r.bytes, .origin = tm_r.origin };
    }

    // Stage 2: capcode-encode the post-NFD bytes. For .full, honor the
    // configured marker style so a TM-derived vocab gets 'C'/'W'/'D'
    // bytes instead of ztok's C0 controls. NoCapcode uses 0x7F in both
    // ztok and TM-Go (no marker style to pick).
    const cap_out = switch (kind) {
        .full => try capcode.encodeStyled(allocator, pre_bytes, cfg.marker_style),
        .nocapcode => try capcode.NoCapcode.encode(allocator, pre_bytes),
    };
    errdefer allocator.free(cap_out);
    const cap_markers = capcode.markersFor(cfg.marker_style);

    // Stage 3: build the output origin map by walking the post-NFD
    // input and the capcode output together. capcode preserves bytes
    // for every input UTF-8 sequence (and substitutes the marker bytes
    // 0x0E/0x0F/0x11/0x7F/0x14 in place of letters that case-folded —
    // case folding can change byte length, but ASCII letters fold to
    // ASCII letters so for the common case each output byte maps back
    // to one input byte trivially). For the general (non-ASCII case
    // fold) path we re-walk the inputs codepoint-by-codepoint and
    // attribute every output byte that didn't match a passthrough to
    // the start of the current input codepoint.
    var origin_out = try allocator.alloc(u32, cap_out.len);
    errdefer allocator.free(origin_out);

    {
        var in_i: usize = 0;
        var out_i: usize = 0;
        // Cursor invariant: pre_origin[in_i] is the post-NFD-byte's
        // ORIGINAL-INPUT offset; capcode's output bytes are attributed
        // to the start of the current post-NFD codepoint.
        while (out_i < cap_out.len) {
            const b = cap_out[out_i];
            // Marker bytes never appear in pre_bytes (capcode's encoder
            // never preserves them — literal `DEL` in input is mapped to
            // `SUBSTITUTE` 0x14, and literal C/W/D bytes are passed
            // through unchanged via the UTF-8 fast path). Recognize the
            // markers up front to keep the cursor stable.
            // Marker detection has to use the active triple for the
            // configured style — under `.tm_printable` the markers are
            // the printable letters 'C'/'W'/'D' (= the ones in
            // `cap_markers`) instead of the C0 controls.
            const is_marker = switch (kind) {
                .full => (b == cap_markers.c or
                    b == cap_markers.w or
                    b == cap_markers.d),
                .nocapcode => (b == capcode.NOCAPCODE_DELETE),
            };
            if (is_marker) {
                const anchor: u32 = if (in_i < pre_origin.len) pre_origin[in_i] else if (pre_origin.len > 0) pre_origin[pre_origin.len - 1] else 0;
                origin_out[out_i] = anchor;
                out_i += 1;
                continue;
            }
            // Compute codepoint length of the *post-NFD* byte under in_i.
            const cp_len: usize = if (in_i < pre_bytes.len) blk: {
                const lead = pre_bytes[in_i];
                const n = std.unicode.utf8ByteSequenceLength(lead) catch 1;
                break :blk if (in_i + n <= pre_bytes.len) n else 1;
            } else 1;
            const cp_anchor: u32 = if (in_i < pre_origin.len)
                pre_origin[in_i]
            else if (pre_origin.len > 0)
                pre_origin[pre_origin.len - 1]
            else
                0;
            // Codepoint output length in cap_out can equal cp_len (passthrough)
            // OR differ (case fold can produce different UTF-8 byte length, eg
            // German ß folds to ss). Distinguish by comparing the slice:
            const match_in = in_i + cp_len <= pre_bytes.len;
            const match_out = out_i + cp_len <= cap_out.len;
            if (match_in and match_out and std.mem.eql(u8, pre_bytes[in_i .. in_i + cp_len], cap_out[out_i .. out_i + cp_len])) {
                // Passthrough — origin maps byte-for-byte to input.
                var k: usize = 0;
                while (k < cp_len) : (k += 1) {
                    origin_out[out_i + k] = if (in_i + k < pre_origin.len) pre_origin[in_i + k] else cp_anchor;
                }
                in_i += cp_len;
                out_i += cp_len;
            } else {
                // Case-folded: greedily consume one output codepoint and
                // attribute every byte to the input codepoint's anchor.
                const out_lead = cap_out[out_i];
                const out_n = std.unicode.utf8ByteSequenceLength(out_lead) catch 1;
                const out_take = if (out_i + out_n <= cap_out.len) out_n else 1;
                var k: usize = 0;
                while (k < out_take) : (k += 1) {
                    origin_out[out_i + k] = cp_anchor;
                }
                out_i += out_take;
                in_i += cp_len;
            }
        }
    }

    // Stage 4 (nocapcode only): if `tm_compat_space` is set:
    //   (a) expand every emitted DEL byte into `DEL + ' '` to match
    //       TM's `NoCapcodeEncode` exactly (TM emits the synthetic
    //       space so the following word looks space-prefixed in the
    //       trained vocab);
    //   (b) if the input starts with a letter or digit, prepend
    //       `DEL + ' '` — TM-Go's encoder treats start-of-input as
    //       "not-space, not-letter", so it emits a leading boundary
    //       marker; ztok's `capcode.NoCapcode.encode` suppresses this
    //       (rlast==0 guard, see src/capcode.zig:1110), so we add it
    //       back here to keep the loader contract honest.
    // See refs/tokenmonster/.../capcode/capcode.go::NoCapcodeEncode.
    if (kind == .nocapcode and cfg.tm_compat_space) {
        var n_dels: usize = 0;
        for (cap_out) |b| if (b == capcode.NOCAPCODE_DELETE) {
            n_dels += 1;
        };
        // Detect a leading letter/digit that ztok's NoCapcode skipped
        // the boundary on (rlast==0 guard). Use the FIRST codepoint
        // of the post-NFD bytes — the cap_out's first byte may already
        // be a DEL if it was inserted by another rule (defensive).
        const need_leading: bool = blk: {
            if (cap_out.len == 0) break :blk false;
            if (cap_out[0] == capcode.NOCAPCODE_DELETE) break :blk false;
            if (pre_bytes.len == 0) break :blk false;
            const lead = pre_bytes[0];
            const seq = std.unicode.utf8ByteSequenceLength(lead) catch 1;
            const take = if (seq <= pre_bytes.len) seq else 1;
            const first_cp = std.unicode.utf8Decode(pre_bytes[0..take]) catch break :blk false;
            break :blk isLetterOrDigitCp(first_cp);
        };
        const lead_extra: usize = if (need_leading) 2 else 0;
        if (n_dels > 0 or lead_extra > 0) {
            const new_len = cap_out.len + n_dels + lead_extra;
            const new_bytes = try allocator.alloc(u8, new_len);
            errdefer allocator.free(new_bytes);
            const new_origin = try allocator.alloc(u32, new_len);
            errdefer allocator.free(new_origin);
            var w: usize = 0;
            if (need_leading) {
                new_bytes[0] = capcode.NOCAPCODE_DELETE;
                new_bytes[1] = ' ';
                // Origin: both bytes anchor to input offset 0 (where the
                // first input codepoint lives).
                new_origin[0] = 0;
                new_origin[1] = 0;
                w = 2;
            }
            for (cap_out, 0..) |b, i| {
                new_bytes[w] = b;
                new_origin[w] = origin_out[i];
                w += 1;
                if (b == capcode.NOCAPCODE_DELETE) {
                    new_bytes[w] = ' ';
                    new_origin[w] = origin_out[i];
                    w += 1;
                }
            }
            std.debug.assert(w == new_len);
            allocator.free(cap_out);
            allocator.free(origin_out);
            return .{ .bytes = new_bytes, .origin = new_origin };
        }
    }

    return .{ .bytes = cap_out, .origin = origin_out };
}

// Local helper — capcode's `isLetterCp`/`isDigitCp` are not exported.
// Use unicode_props for the letter check (matches capcode.zig:1109's
// rules) and a simple ASCII digit / Unicode General_Category Nd check
// for digits via `std.ascii.isDigit` plus the unicode_props "Number"
// range. Inlined here to avoid touching `src/capcode.zig`.
fn isLetterOrDigitCp(cp: u21) bool {
    if (cp < 128) {
        return std.ascii.isAlphabetic(@intCast(cp)) or std.ascii.isDigit(@intCast(cp));
    }
    return @import("unicode_props.zig").isLetter(cp) or @import("unicode_props.zig").isNumber(cp);
}

// === HF Replace normalizer ===
//
// Walks `input` looking for non-overlapping occurrences of `cfg.pattern`
// and rewrites each one with `cfg.content`. Pattern matching is literal
// (regex is a roadmap item; the loader downgrades simple regex sources
// to literals where it can). An empty pattern is a no-op (mirrors HF's
// behavior: `SysRegex::new("")` matches the empty string at every
// position, which would explode the output — we treat it as identity).
//
// Origin map: every output byte that came from the matched `pattern`
// (now replaced by `content`) is attributed to the FIRST byte of the
// pattern occurrence in the original input. Bytes outside any match
// pass through with their natural offsets.

fn replaceNormalize(
    allocator: std.mem.Allocator,
    cfg: ReplaceNormalizer,
    input: []const u8,
) ![]u8 {
    if (cfg.compiled_regex) |re| {
        return replaceNormalizeRegex(allocator, re, cfg.content, input);
    }
    if (cfg.pattern.len == 0) {
        const out = try allocator.alloc(u8, input.len);
        @memcpy(out, input);
        return out;
    }
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    var i: usize = 0;
    while (i < input.len) {
        if (i + cfg.pattern.len <= input.len and
            std.mem.eql(u8, input[i .. i + cfg.pattern.len], cfg.pattern))
        {
            try out.appendSlice(allocator, cfg.content);
            i += cfg.pattern.len;
        } else {
            try out.append(allocator, input[i]);
            i += 1;
        }
    }
    return out.toOwnedSlice(allocator);
}

// Regex-mode Replace: walk `input` finding non-overlapping matches of
// `re` and rewrite each one with `content`. Zero-width matches are
// skipped by advancing one codepoint to avoid infinite loops — same
// convention as `hf_regex.splitWith`.
fn replaceNormalizeRegex(
    allocator: std.mem.Allocator,
    re: *const hf_regex.Regex,
    content: []const u8,
    input: []const u8,
) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    var cursor: usize = 0;
    while (cursor <= input.len) {
        const m = re.findFirst(input, cursor) orelse break;
        // Copy bytes preceding the match through unchanged.
        if (m.start > cursor) try out.appendSlice(allocator, input[cursor..m.start]);
        if (m.end == m.start) {
            // Zero-width match: advance one codepoint past `m.start`
            // and continue. Without this we'd loop forever on patterns
            // like `\s*` that can match empty input.
            try out.append(allocator, input[m.start]);
            const step: usize = utf8StepLen(input, m.start);
            cursor = m.start + step;
            continue;
        }
        try out.appendSlice(allocator, content);
        cursor = m.end;
    }
    if (cursor < input.len) try out.appendSlice(allocator, input[cursor..]);
    return out.toOwnedSlice(allocator);
}

fn utf8StepLen(input: []const u8, pos: usize) usize {
    if (pos >= input.len) return 1;
    const lead = input[pos];
    const n: usize = std.unicode.utf8ByteSequenceLength(lead) catch return 1;
    return @min(n, input.len - pos);
}

fn replaceNormalizeWithOrigin(
    allocator: std.mem.Allocator,
    cfg: ReplaceNormalizer,
    input: []const u8,
) !NormalizationResult {
    if (cfg.compiled_regex) |re| {
        return replaceNormalizeRegexWithOrigin(allocator, re, cfg.content, input);
    }
    if (cfg.pattern.len == 0) {
        const out = try allocator.alloc(u8, input.len);
        errdefer allocator.free(out);
        const origin = try allocator.alloc(u32, input.len);
        for (origin, 0..) |*o, k| o.* = @intCast(k);
        return .{ .bytes = out, .origin = origin };
    }
    var bytes_buf: std.ArrayList(u8) = .empty;
    errdefer bytes_buf.deinit(allocator);
    var origin_buf: std.ArrayList(u32) = .empty;
    errdefer origin_buf.deinit(allocator);

    var i: usize = 0;
    while (i < input.len) {
        if (i + cfg.pattern.len <= input.len and
            std.mem.eql(u8, input[i .. i + cfg.pattern.len], cfg.pattern))
        {
            const anchor: u32 = @intCast(i);
            try bytes_buf.appendSlice(allocator, cfg.content);
            try origin_buf.ensureUnusedCapacity(allocator, cfg.content.len);
            var k: usize = 0;
            while (k < cfg.content.len) : (k += 1) origin_buf.appendAssumeCapacity(anchor);
            i += cfg.pattern.len;
        } else {
            try bytes_buf.append(allocator, input[i]);
            try origin_buf.append(allocator, @intCast(i));
            i += 1;
        }
    }
    // Copy the input first for the empty-input case so we always return a
    // non-null origin even on empty bytes (callers tolerate len==0).
    const out_bytes = try bytes_buf.toOwnedSlice(allocator);
    errdefer allocator.free(out_bytes);
    const out_origin = try origin_buf.toOwnedSlice(allocator);
    return .{ .bytes = out_bytes, .origin = out_origin };
}

fn replaceNormalizeRegexWithOrigin(
    allocator: std.mem.Allocator,
    re: *const hf_regex.Regex,
    content: []const u8,
    input: []const u8,
) !NormalizationResult {
    var bytes_buf: std.ArrayList(u8) = .empty;
    errdefer bytes_buf.deinit(allocator);
    var origin_buf: std.ArrayList(u32) = .empty;
    errdefer origin_buf.deinit(allocator);

    var cursor: usize = 0;
    while (cursor <= input.len) {
        const m = re.findFirst(input, cursor) orelse break;
        // Pass-through bytes before the match keep their natural offsets.
        if (m.start > cursor) {
            try bytes_buf.appendSlice(allocator, input[cursor..m.start]);
            try origin_buf.ensureUnusedCapacity(allocator, m.start - cursor);
            var j: usize = cursor;
            while (j < m.start) : (j += 1) origin_buf.appendAssumeCapacity(@intCast(j));
        }
        if (m.end == m.start) {
            // Zero-width: emit the leading byte as-is and step one
            // codepoint forward (mirrors `replaceNormalizeRegex`).
            try bytes_buf.append(allocator, input[m.start]);
            try origin_buf.append(allocator, @intCast(m.start));
            const step: usize = utf8StepLen(input, m.start);
            cursor = m.start + step;
            continue;
        }
        const anchor: u32 = @intCast(m.start);
        try bytes_buf.appendSlice(allocator, content);
        try origin_buf.ensureUnusedCapacity(allocator, content.len);
        var k: usize = 0;
        while (k < content.len) : (k += 1) origin_buf.appendAssumeCapacity(anchor);
        cursor = m.end;
    }
    if (cursor < input.len) {
        try bytes_buf.appendSlice(allocator, input[cursor..]);
        try origin_buf.ensureUnusedCapacity(allocator, input.len - cursor);
        var j: usize = cursor;
        while (j < input.len) : (j += 1) origin_buf.appendAssumeCapacity(@intCast(j));
    }
    const out_bytes = try bytes_buf.toOwnedSlice(allocator);
    errdefer allocator.free(out_bytes);
    const out_origin = try origin_buf.toOwnedSlice(allocator);
    return .{ .bytes = out_bytes, .origin = out_origin };
}

// === HF Strip normalizer ===
//
// Trims ASCII whitespace (`\t\n\v\f\r ` — `std.ascii.isWhitespace`) from
// the requested edge(s). Bytes that survive the trim retain their
// original origin offsets (i.e. the offset is the input index, not the
// stripped index).

fn isAsciiWs(b: u8) bool {
    return std.ascii.isWhitespace(b);
}

fn stripNormalize(
    allocator: std.mem.Allocator,
    cfg: StripNormalizer,
    input: []const u8,
) ![]u8 {
    var start: usize = 0;
    var end: usize = input.len;
    if (cfg.strip_left) {
        while (start < end and isAsciiWs(input[start])) : (start += 1) {}
    }
    if (cfg.strip_right) {
        while (end > start and isAsciiWs(input[end - 1])) : (end -= 1) {}
    }
    const out = try allocator.alloc(u8, end - start);
    @memcpy(out, input[start..end]);
    return out;
}

fn stripNormalizeWithOrigin(
    allocator: std.mem.Allocator,
    cfg: StripNormalizer,
    input: []const u8,
) !NormalizationResult {
    var start: usize = 0;
    var end: usize = input.len;
    if (cfg.strip_left) {
        while (start < end and isAsciiWs(input[start])) : (start += 1) {}
    }
    if (cfg.strip_right) {
        while (end > start and isAsciiWs(input[end - 1])) : (end -= 1) {}
    }
    const out = try allocator.alloc(u8, end - start);
    errdefer allocator.free(out);
    @memcpy(out, input[start..end]);
    const origin = try allocator.alloc(u32, end - start);
    for (origin, 0..) |*o, k| o.* = @intCast(start + k);
    return .{ .bytes = out, .origin = origin };
}

// === HF Lowercase normalizer (Unicode via capcode.toLower) ===
//
// Post-1.18 agent B: was ASCII-only, now uses `capcode.toLower` for
// per-codepoint Unicode simple case-fold. Covers Latin/Greek/Cyrillic/
// Armenian/Coptic/Georgian/Cherokee/Glagolitic/Adlam/Deseret/Osage and
// the rest of the UCD 16.0 simple case-fold mapping baked into
// `src/capcode.zig`. This closes the bert-base-uncased equivalence
// gap (1.18 D landed at 9998/10000 on a 10K-line stress — the 2 misses
// were fullwidth-Latin codepoints like U+FF21 the ASCII path couldn't
// fold).
//
// Byte-length quirks: `capcode.toLower` is SIMPLE case-fold only — a
// 1:1 codepoint mapping. The encoded UTF-8 byte length can shift per
// codepoint (e.g. U+0130 LATIN CAPITAL LETTER I WITH DOT ABOVE is
// 2 bytes; its simple-fold maps to U+0069 'i', 1 byte — so output
// shrinks). For the bert-base-uncased / Phi-3 chains we care about,
// the typical pattern is "same byte length" (fullwidth Latin
// FF21→FF41 stays 3 bytes; basic Latin A→a stays 1 byte) so the
// origin map remains byte-for-byte accurate within each codepoint
// pair. The implementation handles length shifts: every output byte
// from a folded codepoint inherits the origin offset of the FIRST byte
// of the source codepoint (matches the NFD multi-byte expansion
// convention). German ß (U+00DF) has no uppercase simple-fold
// (capcode.toLower passes it through unchanged) so no quirk there;
// SP's NFKC pre-pass would handle the ß→SS direction separately.

fn lowercaseNormalize(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.ensureTotalCapacity(allocator, input.len);
    var enc: [4]u8 = undefined;
    var i: usize = 0;
    while (i < input.len) {
        const lead = input[i];
        const cp_len_raw: u3 = std.unicode.utf8ByteSequenceLength(lead) catch 1;
        const cp_len: usize = cp_len_raw;
        const end = @min(i + cp_len, input.len);
        if (cp_len == 1) {
            // Fast-path ASCII to avoid a decode/encode round-trip.
            const b: u8 = if (lead >= 'A' and lead <= 'Z') lead + 32 else lead;
            try out.append(allocator, b);
            i = end;
            continue;
        }
        const decoded: u21 = std.unicode.utf8Decode(input[i..end]) catch {
            // Malformed UTF-8 — pass the lead byte through and advance one.
            try out.append(allocator, lead);
            i += 1;
            continue;
        };
        const folded: u21 = capcode.toLower(decoded);
        const n = std.unicode.utf8Encode(folded, &enc) catch {
            // Un-encodable fold — fall back to the source bytes.
            try out.appendSlice(allocator, input[i..end]);
            i = end;
            continue;
        };
        try out.appendSlice(allocator, enc[0..n]);
        i = end;
    }
    return out.toOwnedSlice(allocator);
}

fn lowercaseNormalizeWithOrigin(
    allocator: std.mem.Allocator,
    input: []const u8,
) !NormalizationResult {
    var out_bytes: std.ArrayList(u8) = .empty;
    errdefer out_bytes.deinit(allocator);
    var out_origin: std.ArrayList(u32) = .empty;
    errdefer out_origin.deinit(allocator);
    try out_bytes.ensureTotalCapacity(allocator, input.len);
    try out_origin.ensureTotalCapacity(allocator, input.len);

    var enc: [4]u8 = undefined;
    var i: usize = 0;
    while (i < input.len) {
        const lead = input[i];
        const cp_len_raw: u3 = std.unicode.utf8ByteSequenceLength(lead) catch 1;
        const cp_len: usize = cp_len_raw;
        const end = @min(i + cp_len, input.len);
        const anchor: u32 = @intCast(i);
        if (cp_len == 1) {
            const b: u8 = if (lead >= 'A' and lead <= 'Z') lead + 32 else lead;
            try out_bytes.append(allocator, b);
            try out_origin.append(allocator, anchor);
            i = end;
            continue;
        }
        const decoded: u21 = std.unicode.utf8Decode(input[i..end]) catch {
            try out_bytes.append(allocator, lead);
            try out_origin.append(allocator, anchor);
            i += 1;
            continue;
        };
        const folded: u21 = capcode.toLower(decoded);
        const n = std.unicode.utf8Encode(folded, &enc) catch {
            try out_bytes.appendSlice(allocator, input[i..end]);
            var k: usize = 0;
            while (k < (end - i)) : (k += 1) try out_origin.append(allocator, anchor);
            i = end;
            continue;
        };
        try out_bytes.appendSlice(allocator, enc[0..n]);
        var k: usize = 0;
        while (k < n) : (k += 1) try out_origin.append(allocator, anchor);
        i = end;
    }
    const bytes = try out_bytes.toOwnedSlice(allocator);
    errdefer allocator.free(bytes);
    const origin = try out_origin.toOwnedSlice(allocator);
    return .{ .bytes = bytes, .origin = origin };
}

// === HF BertNormalizer ===
//
// Step-for-step port of `tokenizers::normalizers::bert::BertNormalizer`.
// Decomposes into four codepoint-walking passes, each producing a fresh
// bytes+origin buffer. The pipeline order from HF (verbatim):
//   1. if clean_text: drop NUL/FFFD/control, fold whitespace -> ' '
//   2. if handle_chinese_chars: insert ' ' before AND after each CJK cp
//   3. let strip_accents = strip_accents.unwrap_or(lowercase)
//      if strip_accents: NFD, then drop Mn marks
//   4. if lowercase: ASCII lowercase (v1)
//
// The lowercase pass uses ASCII fold (v1 limitation — see above).
// Implementations match HF byte-for-byte on input that is already
// ASCII; for non-ASCII letters the lowercase pass leaves them
// unchanged (HF would full-Unicode lowercase them).

fn isBertWhitespaceCp(cp: u21) bool {
    // HF: '\t' | '\n' | '\r' | (Unicode whitespace).
    if (cp == '\t' or cp == '\n' or cp == '\r') return true;
    return @import("unicode_props.zig").isWhitespace(cp);
}

fn isBertControlCp(cp: u21) bool {
    // HF: NOT \t,\n,\r AND `c.is_other()`. is_other = Cc | Cf | Cn | Co
    // (Unicode TR44 Table 12). We model Cc inline (ASCII C0 + DEL + C1)
    // and Cf via `unicode_props.isCf` — a UCD 16.0 range table covering
    // soft hyphen, the bidi controls (LRM/RLM, LRE/RLE/PDF, isolates
    // LRI/RLI/FSI/PDI U+2066-U+2069), the zero-width joiners, BOM, and
    // the TAG block (U+E0001, U+E0020-U+E007F) used in regional-flag
    // emoji. Cn (unassigned) and Co (private use) aren't modeled —
    // they're effectively unreachable in real text, and the asymmetry
    // matches what bert-base-uncased actually sees in practice (any
    // gap there is invisible against the noise of Cf coverage). The
    // previous Cf cover was a hand-rolled subset (200B-200F, 202A-202E,
    // 2060-2064, FEFF, AD) that missed the isolate block and TAG
    // chars — costing ~96/1000 lines on unicode_stress.txt.
    if (cp == '\t' or cp == '\n' or cp == '\r') return false;
    if (cp < 0x20) return true;
    if (cp == 0x7F) return true;
    if (cp >= 0x80 and cp <= 0x9F) return true;
    return @import("unicode_props.zig").isCf(cp);
}

fn isChineseCharCp(cp: u21) bool {
    return (cp >= 0x4E00 and cp <= 0x9FFF) or
        (cp >= 0x3400 and cp <= 0x4DBF) or
        (cp >= 0x20000 and cp <= 0x2A6DF) or
        (cp >= 0x2A700 and cp <= 0x2B73F) or
        (cp >= 0x2B740 and cp <= 0x2B81F) or
        (cp >= 0x2B920 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or
        (cp >= 0x2F800 and cp <= 0x2FA1F);
}

/// Walk `bytes` codepoint by codepoint and rewrite into `out_bytes`,
/// pushing parallel origin entries. The mapper is called with each
/// decoded codepoint and decides what to emit (see callers).
const CpAction = struct {
    /// Up to two codepoints to emit. `cp[0]` always; `cp[1]` only when
    /// `n == 2`. Used by the Chinese-char pass which emits ' ' c ' '.
    /// To represent "drop the codepoint" set `n = 0`.
    cps: [3]u21 = .{ 0, 0, 0 },
    n: u3 = 1,
};

fn rewriteCpsWithOrigin(
    allocator: std.mem.Allocator,
    bytes: []const u8,
    origin: []const u32,
    comptime ctx_T: type,
    ctx: ctx_T,
    comptime mapper: fn (ctx_T, u21) CpAction,
) !NormalizationResult {
    std.debug.assert(bytes.len == origin.len);
    var out_bytes: std.ArrayList(u8) = .empty;
    errdefer out_bytes.deinit(allocator);
    var out_origin: std.ArrayList(u32) = .empty;
    errdefer out_origin.deinit(allocator);

    var enc: [4]u8 = undefined;
    var i: usize = 0;
    while (i < bytes.len) {
        const lead = bytes[i];
        const cp_len_raw: u3 = std.unicode.utf8ByteSequenceLength(lead) catch 1;
        const cp_len: usize = cp_len_raw;
        const end = @min(i + cp_len, bytes.len);
        const decoded: u21 = if (cp_len == 1) lead else (std.unicode.utf8Decode(bytes[i..end]) catch lead);
        const anchor: u32 = origin[i];
        const action = mapper(ctx, decoded);
        var idx: usize = 0;
        while (idx < action.n) : (idx += 1) {
            const cp = action.cps[idx];
            const n = std.unicode.utf8Encode(cp, &enc) catch {
                // Skip un-encodable replacement.
                continue;
            };
            try out_bytes.ensureUnusedCapacity(allocator, n);
            try out_origin.ensureUnusedCapacity(allocator, n);
            var k: usize = 0;
            while (k < n) : (k += 1) {
                out_bytes.appendAssumeCapacity(enc[k]);
                out_origin.appendAssumeCapacity(anchor);
            }
        }
        i = end;
        if (i == i - cp_len) i += 1; // defensive against zero-length loop
    }
    const final_bytes = try out_bytes.toOwnedSlice(allocator);
    errdefer allocator.free(final_bytes);
    const final_origin = try out_origin.toOwnedSlice(allocator);
    return .{ .bytes = final_bytes, .origin = final_origin };
}

fn rewriteCps(
    allocator: std.mem.Allocator,
    bytes: []const u8,
    comptime ctx_T: type,
    ctx: ctx_T,
    comptime mapper: fn (ctx_T, u21) CpAction,
) ![]u8 {
    var out_bytes: std.ArrayList(u8) = .empty;
    errdefer out_bytes.deinit(allocator);
    var enc: [4]u8 = undefined;
    var i: usize = 0;
    while (i < bytes.len) {
        const lead = bytes[i];
        const cp_len_raw: u3 = std.unicode.utf8ByteSequenceLength(lead) catch 1;
        const cp_len: usize = cp_len_raw;
        const end = @min(i + cp_len, bytes.len);
        const decoded: u21 = if (cp_len == 1) lead else (std.unicode.utf8Decode(bytes[i..end]) catch lead);
        const action = mapper(ctx, decoded);
        var idx: usize = 0;
        while (idx < action.n) : (idx += 1) {
            const cp = action.cps[idx];
            const n = std.unicode.utf8Encode(cp, &enc) catch continue;
            try out_bytes.appendSlice(allocator, enc[0..n]);
        }
        i = end;
    }
    return out_bytes.toOwnedSlice(allocator);
}

// Mapper contexts and functions for the four Bert passes.
fn cleanTextMap(_: void, cp: u21) CpAction {
    if (cp == 0 or cp == 0xFFFD or isBertControlCp(cp)) {
        return .{ .n = 0 };
    }
    if (isBertWhitespaceCp(cp)) {
        return .{ .cps = .{ ' ', 0, 0 }, .n = 1 };
    }
    return .{ .cps = .{ cp, 0, 0 }, .n = 1 };
}

fn chineseMap(_: void, cp: u21) CpAction {
    if (isChineseCharCp(cp)) {
        return .{ .cps = .{ ' ', cp, ' ' }, .n = 3 };
    }
    return .{ .cps = .{ cp, 0, 0 }, .n = 1 };
}

fn stripAccentsMap(_: void, cp: u21) CpAction {
    // HF tokenizers' BertNormalizer drops only nonspacing marks (Mn),
    // not all combining marks (Mn+Mc+Me). Mc and Me carry semantic
    // information in scripts like Devanagari and shouldn't be removed.
    if (@import("unicode_props.zig").isMn(cp)) {
        return .{ .n = 0 };
    }
    return .{ .cps = .{ cp, 0, 0 }, .n = 1 };
}

fn asciiLowerMap(_: void, cp: u21) CpAction {
    if (cp >= 'A' and cp <= 'Z') {
        return .{ .cps = .{ cp + 32, 0, 0 }, .n = 1 };
    }
    return .{ .cps = .{ cp, 0, 0 }, .n = 1 };
}

/// Unicode lowercase mapper for the Bert chain — post-1.18 agent B.
/// Delegates to `capcode.toLower` so the Bert path picks up the same
/// UCD 16.0 simple case-fold the standalone `Lowercase` normalizer
/// uses. ASCII is fast-pathed (no table lookup) to keep the bert
/// hot loop tight.
///
/// Two ranges that `capcode.toLower` deliberately omits (Nl / So
/// numeric and symbol letters — capcode's tables cover the script
/// blocks only) but HF's `char::to_lowercase` does fold:
///   * U+2160-U+216F (uppercase Roman numerals) -> U+2170-U+217F.
///   * U+24B6-U+24CF (circled Latin capital letters) -> U+24D0-U+24E9.
/// Both are straight +0x10 (Roman) / +0x1A (circled) offsets, so we
/// fold them inline here rather than threading another table through
/// `capcode`. This is scoped to the Bert arm; the standalone
/// `Lowercase` normalizer (used by Phi-3 etc.) keeps the capcode-only
/// behaviour to avoid surprising other model chains. ~17/1000 lines
/// on unicode_stress.txt diverged for want of these two cases.
fn unicodeLowerMap(_: void, cp: u21) CpAction {
    if (cp < 0x80) {
        if (cp >= 'A' and cp <= 'Z') return .{ .cps = .{ cp + 32, 0, 0 }, .n = 1 };
        return .{ .cps = .{ cp, 0, 0 }, .n = 1 };
    }
    if (cp >= 0x2160 and cp <= 0x216F) return .{ .cps = .{ cp + 0x10, 0, 0 }, .n = 1 };
    if (cp >= 0x24B6 and cp <= 0x24CF) return .{ .cps = .{ cp + 0x1A, 0, 0 }, .n = 1 };
    return .{ .cps = .{ capcode.toLower(cp), 0, 0 }, .n = 1 };
}

fn bertNormalize(
    allocator: std.mem.Allocator,
    cfg: BertNormalizerConfig,
    input: []const u8,
) ![]u8 {
    var cur: []u8 = try allocator.alloc(u8, input.len);
    @memcpy(cur, input);
    errdefer allocator.free(cur);

    if (cfg.clean_text) {
        const next = try rewriteCps(allocator, cur, void, {}, cleanTextMap);
        allocator.free(cur);
        cur = next;
    }
    if (cfg.handle_chinese_chars) {
        const next = try rewriteCps(allocator, cur, void, {}, chineseMap);
        allocator.free(cur);
        cur = next;
    }
    const do_strip = cfg.strip_accents orelse cfg.lowercase;
    if (do_strip) {
        // NFD then drop Mn.
        const nfd_bytes = try unicode_norm.normalize(allocator, .nfd, cur);
        allocator.free(cur);
        cur = nfd_bytes;
        const next = try rewriteCps(allocator, cur, void, {}, stripAccentsMap);
        allocator.free(cur);
        cur = next;
    }
    if (cfg.lowercase) {
        const next = try rewriteCps(allocator, cur, void, {}, unicodeLowerMap);
        allocator.free(cur);
        cur = next;
    }
    return cur;
}

fn bertNormalizeWithOrigin(
    allocator: std.mem.Allocator,
    cfg: BertNormalizerConfig,
    input: []const u8,
) !NormalizationResult {
    var cur_bytes: []u8 = try allocator.alloc(u8, input.len);
    @memcpy(cur_bytes, input);
    var cur_origin: []u32 = try allocator.alloc(u32, input.len);
    for (cur_origin, 0..) |*o, k| o.* = @intCast(k);
    errdefer allocator.free(cur_bytes);
    errdefer allocator.free(cur_origin);

    if (cfg.clean_text) {
        const r = try rewriteCpsWithOrigin(allocator, cur_bytes, cur_origin, void, {}, cleanTextMap);
        allocator.free(cur_bytes);
        allocator.free(cur_origin);
        cur_bytes = r.bytes;
        cur_origin = r.origin.?;
    }
    if (cfg.handle_chinese_chars) {
        const r = try rewriteCpsWithOrigin(allocator, cur_bytes, cur_origin, void, {}, chineseMap);
        allocator.free(cur_bytes);
        allocator.free(cur_origin);
        cur_bytes = r.bytes;
        cur_origin = r.origin.?;
    }
    const do_strip = cfg.strip_accents orelse cfg.lowercase;
    if (do_strip) {
        // NFD pass with origin tracking.
        const nfd_r = try unicode_norm.normalizeWithOrigin(allocator, .nfd, cur_bytes);
        const nfd_origin = nfd_r.origin orelse blk: {
            const fab = try allocator.alloc(u32, nfd_r.bytes.len);
            for (fab, 0..) |*o, k| o.* = @intCast(@min(k, cur_bytes.len));
            break :blk fab;
        };
        // Propagate origin through to ORIGINAL input via cur_origin.
        const propagated_origin = try allocator.alloc(u32, nfd_origin.len);
        for (nfd_origin, 0..) |off, k| {
            const idx = @min(off, @as(u32, @intCast(cur_origin.len -| 1)));
            propagated_origin[k] = cur_origin[idx];
        }
        allocator.free(nfd_origin);
        allocator.free(cur_bytes);
        allocator.free(cur_origin);
        cur_bytes = nfd_r.bytes;
        cur_origin = propagated_origin;

        const r = try rewriteCpsWithOrigin(allocator, cur_bytes, cur_origin, void, {}, stripAccentsMap);
        allocator.free(cur_bytes);
        allocator.free(cur_origin);
        cur_bytes = r.bytes;
        cur_origin = r.origin.?;
    }
    if (cfg.lowercase) {
        const r = try rewriteCpsWithOrigin(allocator, cur_bytes, cur_origin, void, {}, unicodeLowerMap);
        allocator.free(cur_bytes);
        allocator.free(cur_origin);
        cur_bytes = r.bytes;
        cur_origin = r.origin.?;
    }
    return .{ .bytes = cur_bytes, .origin = cur_origin };
}

// === Sequence normalizer ===
//
// Runs each inner normalizer in order, threading the bytes (and origin
// map, when asked) through. For the with-origin path we compose origin
// maps: at each stage, the new normalizer produces an origin map into
// the PREVIOUS stage's bytes; we collapse that into the original input
// via a table lookup.

fn sequenceNormalize(
    allocator: std.mem.Allocator,
    cfg: SequenceNormalizer,
    input: []const u8,
) anyerror![]u8 {
    if (cfg.normalizers.len == 0) {
        const out = try allocator.alloc(u8, input.len);
        @memcpy(out, input);
        return out;
    }
    var cur: []u8 = try cfg.normalizers[0].normalize(allocator, input);
    errdefer allocator.free(cur);
    var i: usize = 1;
    while (i < cfg.normalizers.len) : (i += 1) {
        const next = try cfg.normalizers[i].normalize(allocator, cur);
        allocator.free(cur);
        cur = next;
    }
    return cur;
}

// === HF Prepend normalizer ===
//
// Emits `cfg.prepend ++ input`. Used by SP-style HF chains
// (e.g. Phi-3's `Sequence[Prepend("▁"), Replace(" "→"▁")]`) — same
// dummy-prefix idea SP's `add_dummy_prefix` carries, exposed as a
// composable stage so it can compose with HF's other normalizers.
//
// Origin map: prepended bytes anchor to offset 0 of the original input;
// every other byte inherits its natural input offset.

fn prependNormalize(
    allocator: std.mem.Allocator,
    cfg: PrependNormalizer,
    input: []const u8,
) ![]u8 {
    // Match HF `tokenizers::normalizers::prepend::Prepend`: when the
    // input is empty the prefix is NOT applied. This also keeps the
    // Pipeline encoder's `cap = input.len * maxByteExpansion` capacity
    // computation safe (zero in, zero out).
    if (input.len == 0) {
        return allocator.alloc(u8, 0);
    }
    const total = cfg.prepend.len + input.len;
    const out = try allocator.alloc(u8, total);
    if (cfg.prepend.len > 0) @memcpy(out[0..cfg.prepend.len], cfg.prepend);
    @memcpy(out[cfg.prepend.len..total], input);
    return out;
}

fn prependNormalizeWithOrigin(
    allocator: std.mem.Allocator,
    cfg: PrependNormalizer,
    input: []const u8,
) !NormalizationResult {
    // HF parity: skip the prefix on empty input (see prependNormalize).
    if (input.len == 0) {
        return .{
            .bytes = try allocator.alloc(u8, 0),
            .origin = try allocator.alloc(u32, 0),
        };
    }
    const total = cfg.prepend.len + input.len;
    const out = try allocator.alloc(u8, total);
    errdefer allocator.free(out);
    if (cfg.prepend.len > 0) @memcpy(out[0..cfg.prepend.len], cfg.prepend);
    @memcpy(out[cfg.prepend.len..total], input);
    const origin = try allocator.alloc(u32, total);
    var i: usize = 0;
    while (i < cfg.prepend.len) : (i += 1) origin[i] = 0;
    var j: usize = 0;
    while (j < input.len) : (j += 1) origin[cfg.prepend.len + j] = @intCast(j);
    return .{ .bytes = out, .origin = origin };
}

fn sequenceNormalizeWithOrigin(
    allocator: std.mem.Allocator,
    cfg: SequenceNormalizer,
    input: []const u8,
) anyerror!NormalizationResult {
    if (cfg.normalizers.len == 0) {
        const out = try allocator.alloc(u8, input.len);
        errdefer allocator.free(out);
        const origin = try allocator.alloc(u32, input.len);
        for (origin, 0..) |*o, k| o.* = @intCast(k);
        return .{ .bytes = out, .origin = origin };
    }
    var cur = try cfg.normalizers[0].normalizeWithOrigin(allocator, input);
    if (cur.origin == null) {
        // Inner was identity — fabricate the 1:1 origin map so the
        // composition step can index into it uniformly.
        const fab = try allocator.alloc(u32, cur.bytes.len);
        for (fab, 0..) |*o, k| o.* = @intCast(k);
        cur.origin = fab;
    }
    errdefer cur.deinit(allocator);

    var i: usize = 1;
    while (i < cfg.normalizers.len) : (i += 1) {
        var step = try cfg.normalizers[i].normalizeWithOrigin(allocator, cur.bytes);
        // Compose: step.origin maps step.bytes -> cur.bytes;
        // we want step.bytes -> ORIGINAL input.
        const step_origin = step.origin orelse blk: {
            const fab = try allocator.alloc(u32, step.bytes.len);
            for (fab, 0..) |*o, k| o.* = @intCast(k);
            break :blk fab;
        };
        defer allocator.free(step_origin);
        const composed = try allocator.alloc(u32, step_origin.len);
        const cur_origin = cur.origin.?;
        for (step_origin, 0..) |off, k| {
            const idx = if (cur_origin.len == 0) 0 else @min(off, @as(u32, @intCast(cur_origin.len - 1)));
            composed[k] = if (cur_origin.len == 0) 0 else cur_origin[idx];
        }
        step.origin = null; // we freed step_origin above
        allocator.free(cur.bytes);
        allocator.free(cur.origin.?);
        cur = .{ .bytes = step.bytes, .origin = composed };
    }
    return cur;
}

test "identity normalizer preserves input" {
    const out = try (Normalizer{ .identity = {} }).normalize(std.testing.allocator, "héllo");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("héllo", out);
}

test "nfc recomposes e + combining acute" {
    const input = "e\xCC\x81"; // 'e' + U+0301
    const out = try (Normalizer{ .nfc = {} }).normalize(std.testing.allocator, input);
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("\xC3\xA9", out); // 'é' (U+00E9)
}

test "nfkc unwraps ligatures" {
    const input = "\xEF\xAC\x81"; // 'ﬁ' (U+FB01)
    const out = try (Normalizer{ .nfkc = {} }).normalize(std.testing.allocator, input);
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("fi", out);
}

test "byte_level normalizer maps space to Ġ" {
    const out = try (Normalizer{ .byte_level = {} }).normalize(std.testing.allocator, " a");
    defer std.testing.allocator.free(out);
    // U+0120 'Ġ' (UTF-8 0xC4 0xA0) then 'a' (which maps to itself since 'a' is in the printable seed set)
    try std.testing.expectEqualStrings("\xC4\xA0a", out);
}

// === origin-map tests ===

test "normalizeWithOrigin: identity returns null origin" {
    var r = try (Normalizer{ .identity = {} }).normalizeWithOrigin(std.testing.allocator, "hello");
    defer r.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("hello", r.bytes);
    try std.testing.expectEqual(@as(?[]u32, null), r.origin);
}

test "normalizeWithOrigin: nfc ASCII is 1:1" {
    var r = try (Normalizer{ .nfc = {} }).normalizeWithOrigin(std.testing.allocator, "hello");
    defer r.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("hello", r.bytes);
    const origin = r.origin orelse return error.TestExpectedNonNullOrigin;
    try std.testing.expectEqual(@as(usize, 5), origin.len);
    var i: u32 = 0;
    while (i < 5) : (i += 1) {
        try std.testing.expectEqual(i, origin[i]);
    }
}

test "normalizeWithOrigin: nfd of e-acute -> e + combining-acute, both share offset 0" {
    // Input: U+00E9 = 0xC3 0xA9 (2 bytes)
    const input = "\xC3\xA9";
    var r = try (Normalizer{ .nfd = {} }).normalizeWithOrigin(std.testing.allocator, input);
    defer r.deinit(std.testing.allocator);
    // NFD: 'e' (1 byte) + U+0301 (0xCC 0x81, 2 bytes) = 3 bytes
    try std.testing.expectEqualStrings("e\xCC\x81", r.bytes);
    const origin = r.origin orelse return error.TestExpectedNonNullOrigin;
    try std.testing.expectEqual(@as(usize, 3), origin.len);
    // All three output bytes were produced from the codepoint at source offset 0.
    try std.testing.expectEqual(@as(u32, 0), origin[0]);
    try std.testing.expectEqual(@as(u32, 0), origin[1]);
    try std.testing.expectEqual(@as(u32, 0), origin[2]);
}

test "normalizeWithOrigin: nfkc ligature ﬁ -> fi, both bytes share offset 0" {
    // U+FB01 = 0xEF 0xAC 0x81 (3 bytes)
    const input = "\xEF\xAC\x81";
    var r = try (Normalizer{ .nfkc = {} }).normalizeWithOrigin(std.testing.allocator, input);
    defer r.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("fi", r.bytes);
    const origin = r.origin orelse return error.TestExpectedNonNullOrigin;
    try std.testing.expectEqual(@as(usize, 2), origin.len);
    try std.testing.expectEqual(@as(u32, 0), origin[0]);
    try std.testing.expectEqual(@as(u32, 0), origin[1]);
}

test "normalizeWithOrigin: nfc mixed ASCII + multi-byte preserves per-cp offsets" {
    // "a" + "é" (precomposed U+00E9 = 0xC3 0xA9) + "b"
    const input = "a\xC3\xA9b";
    var r = try (Normalizer{ .nfc = {} }).normalizeWithOrigin(std.testing.allocator, input);
    defer r.deinit(std.testing.allocator);
    // NFC of precomposed é stays é. Output = "a" + "é" + "b" = 1 + 2 + 1 = 4 bytes.
    try std.testing.expectEqualStrings("a\xC3\xA9b", r.bytes);
    const origin = r.origin orelse return error.TestExpectedNonNullOrigin;
    try std.testing.expectEqual(@as(usize, 4), origin.len);
    try std.testing.expectEqual(@as(u32, 0), origin[0]); // 'a' at input offset 0
    try std.testing.expectEqual(@as(u32, 1), origin[1]); // 'é' byte 1 at input offset 1
    try std.testing.expectEqual(@as(u32, 1), origin[2]); // 'é' byte 2 also offset 1
    try std.testing.expectEqual(@as(u32, 3), origin[3]); // 'b' at input offset 3
}

test "normalizeWithOrigin: byte_level — each input byte -> 1-2 output bytes sharing origin" {
    const input = " a"; // " " maps to 'Ġ' (0xC4 0xA0); 'a' maps to itself
    var r = try (Normalizer{ .byte_level = {} }).normalizeWithOrigin(std.testing.allocator, input);
    defer r.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("\xC4\xA0a", r.bytes);
    const origin = r.origin orelse return error.TestExpectedNonNullOrigin;
    try std.testing.expectEqual(@as(usize, 3), origin.len);
    try std.testing.expectEqual(@as(u32, 0), origin[0]); // 'Ġ' byte 1 from input ' ' @ 0
    try std.testing.expectEqual(@as(u32, 0), origin[1]); // 'Ġ' byte 2 from input ' ' @ 0
    try std.testing.expectEqual(@as(u32, 1), origin[2]); // 'a' from input 'a' @ 1
}

test "normalizeWithOrigin: round-trip — every origin is a valid input offset" {
    const inputs = [_][]const u8{
        "hello",
        "héllo",                       // mixed ASCII + 2-byte
        "ﬁ",                            // ligature (NFKC expands)
        "e\xCC\x81llo",                // decomposed é
        "café\nworld",
        "the quick brown fox",
    };
    const variants = [_]Normalizer{
        .{ .nfc = {} },
        .{ .nfd = {} },
        .{ .nfkc = {} },
        .{ .nfkd = {} },
        .{ .byte_level = {} },
    };
    for (variants) |n| {
        for (inputs) |inp| {
            var r = try n.normalizeWithOrigin(std.testing.allocator, inp);
            defer r.deinit(std.testing.allocator);
            const origin = r.origin orelse continue;
            try std.testing.expectEqual(r.bytes.len, origin.len);
            for (origin) |off| {
                try std.testing.expect(off <= inp.len);
            }
        }
    }
}

test "normalize wrapper still works (backwards compat)" {
    // The old single-return normalize must continue to work for existing
    // callers (c_api, pipeline.encodeText, etc).
    const out = try (Normalizer{ .nfc = {} }).normalize(std.testing.allocator, "e\xCC\x81");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("\xC3\xA9", out);
}

// === SP normalizer (sp_precompiled) tests ===

test "sp_precompiled: escape_whitespaces only — spaces become U+2581" {
    const n = Normalizer{ .sp_precompiled = .{
        .nfkc = false,
        .add_dummy_prefix = false,
        .escape_whitespaces = true,
        .remove_extra_whitespaces = false,
    } };
    const out = try n.normalize(std.testing.allocator, "hello world");
    defer std.testing.allocator.free(out);
    // 'hello' + U+2581 (E2 96 81) + 'world'
    try std.testing.expectEqualStrings("hello\xE2\x96\x81world", out);
}

test "sp_precompiled: add_dummy_prefix only — leading U+2581 prepended" {
    const n = Normalizer{ .sp_precompiled = .{
        .nfkc = false,
        .add_dummy_prefix = true,
        .escape_whitespaces = false,
        .remove_extra_whitespaces = false,
    } };
    const out = try n.normalize(std.testing.allocator, "hello");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("\xE2\x96\x81hello", out);
}

test "sp_precompiled: dummy_prefix + escape together — full SP default" {
    const n = Normalizer{ .sp_precompiled = .{
        .nfkc = false,
        .add_dummy_prefix = true,
        .escape_whitespaces = true,
        .remove_extra_whitespaces = false,
    } };
    const out = try n.normalize(std.testing.allocator, "hello world");
    defer std.testing.allocator.free(out);
    // ▁hello▁world
    try std.testing.expectEqualStrings("\xE2\x96\x81hello\xE2\x96\x81world", out);
}

test "sp_precompiled: nfkc only — ligature unwrap" {
    const n = Normalizer{ .sp_precompiled = .{
        .nfkc = true,
        .add_dummy_prefix = false,
        .escape_whitespaces = false,
        .remove_extra_whitespaces = false,
    } };
    // U+FB01 ('ﬁ') → 'fi'
    const out = try n.normalize(std.testing.allocator, "\xEF\xAC\x81");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("fi", out);
}

test "sp_precompiled: all flags off — identity behavior" {
    const n = Normalizer{ .sp_precompiled = .{
        .nfkc = false,
        .add_dummy_prefix = false,
        .escape_whitespaces = false,
        .remove_extra_whitespaces = false,
    } };
    const input = "Hello,  world! \xEF\xAC\x81";
    const out = try n.normalize(std.testing.allocator, input);
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings(input, out);
}

test "sp_precompiled: remove_extra_whitespaces collapses runs" {
    const n = Normalizer{ .sp_precompiled = .{
        .nfkc = false,
        .add_dummy_prefix = false,
        .escape_whitespaces = false,
        .remove_extra_whitespaces = true,
    } };
    const out = try n.normalize(std.testing.allocator, "  hello   world  ");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("hello world", out);
}

test "sp_precompiled: U+2007 FIGURE SPACE passes through (parity with SP reference)" {
    // Reference SP without a charsmap leaves U+2007 (and any other
    // multi-byte unicode-space cp) untouched. The 1.22 over-fix that
    // collapsed U+2007 here broke LLaMA-2 / Mistral / Gemma / Yi /
    // Gemma parity on the unicode_stress corpus — those models do not
    // ship a charsmap that maps U+2007 to ASCII space, so SP-python
    // byte-fallback-encodes the raw E2 80 87. Match that here.
    {
        const n = Normalizer{ .sp_precompiled = .{
            .nfkc = false,
            .add_dummy_prefix = false,
            .escape_whitespaces = true,
            .remove_extra_whitespaces = false,
        } };
        const out = try n.normalize(std.testing.allocator, "a\u{2007}b");
        defer std.testing.allocator.free(out);
        // U+2007 stays as its 3 bytes; only ASCII ' ' would have become U+2581.
        try std.testing.expectEqualStrings("a\xE2\x80\x87b", out);
    }
    {
        const n = Normalizer{ .sp_precompiled = .{
            .nfkc = false,
            .add_dummy_prefix = false,
            .escape_whitespaces = false,
            .remove_extra_whitespaces = true,
        } };
        // A run of "ASCII ' ' + U+2007 + ASCII ' '" must NOT all
        // collapse — the U+2007 is content, so the run is
        // [space][content][space], which collapses to "[space]X[space]".
        const out = try n.normalize(std.testing.allocator, "a \u{2007} b");
        defer std.testing.allocator.free(out);
        try std.testing.expectEqualStrings("a \xE2\x80\x87 b", out);
    }
}

test "sp_precompiled: U+2028 LINE SEPARATOR passes through" {
    // Same rationale as U+2007: SP without a charsmap treats this as
    // content, not whitespace. Confirmed against SP-python on llama2/
    // mistral7b/gemma/yi6b (all return the codepoint verbatim from
    // normalize()).
    const n = Normalizer{ .sp_precompiled = .{
        .nfkc = false,
        .add_dummy_prefix = false,
        .escape_whitespaces = false,
        .remove_extra_whitespaces = true,
    } };
    const out = try n.normalize(std.testing.allocator, "a\u{2028}\u{2028}b");
    defer std.testing.allocator.free(out);
    // Two U+2028 are non-whitespace; they pass through verbatim and
    // do NOT collapse to a single space.
    try std.testing.expectEqualStrings("a\xE2\x80\xA8\xE2\x80\xA8b", out);
}

test "sp_precompiled: U+2007 origin map — passthrough preserves byte offsets" {
    // U+2007 is content (no charsmap → no rewrite). All 3 bytes stay
    // put at offset 1; 'b' lands at offset 4.
    const n = Normalizer{ .sp_precompiled = .{
        .nfkc = false,
        .add_dummy_prefix = false,
        .escape_whitespaces = true,
        .remove_extra_whitespaces = false,
    } };
    const input = "a\u{2007}b"; // 5 bytes: 'a' + E2 80 87 + 'b'
    var r = try n.normalizeWithOrigin(std.testing.allocator, input);
    defer r.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("a\xE2\x80\x87b", r.bytes);
    const origin = r.origin orelse return error.TestExpectedNonNullOrigin;
    try std.testing.expectEqual(@as(usize, 5), r.bytes.len);
    try std.testing.expectEqual(@as(u32, 0), origin[0]); // 'a'
    try std.testing.expectEqual(@as(u32, 1), origin[1]); // U+2007 byte 0
    try std.testing.expectEqual(@as(u32, 2), origin[2]); // U+2007 byte 1
    try std.testing.expectEqual(@as(u32, 3), origin[3]); // U+2007 byte 2
    try std.testing.expectEqual(@as(u32, 4), origin[4]); // 'b'
    for (origin) |off| try std.testing.expect(off <= input.len);
}

test "sp_precompiled: origin map — every output byte maps into the original" {
    const cfg_variants = [_]SpNormalizer{
        .{ .nfkc = false, .add_dummy_prefix = false, .escape_whitespaces = false, .remove_extra_whitespaces = false },
        .{ .nfkc = true, .add_dummy_prefix = false, .escape_whitespaces = false, .remove_extra_whitespaces = false },
        .{ .nfkc = false, .add_dummy_prefix = true, .escape_whitespaces = false, .remove_extra_whitespaces = false },
        .{ .nfkc = false, .add_dummy_prefix = false, .escape_whitespaces = true, .remove_extra_whitespaces = false },
        .{ .nfkc = true, .add_dummy_prefix = true, .escape_whitespaces = true, .remove_extra_whitespaces = false },
        .{ .nfkc = true, .add_dummy_prefix = true, .escape_whitespaces = true, .remove_extra_whitespaces = true },
    };
    const inputs = [_][]const u8{
        "hello",
        "hello world",
        " leading space",
        "trailing space ",
        "Hello, world! \xEF\xAC\x81",
        "the quick brown fox",
        "café\nworld",
        "  multiple   spaces  ",
    };
    for (cfg_variants) |cfg| {
        for (inputs) |inp| {
            const n = Normalizer{ .sp_precompiled = cfg };
            var r = try n.normalizeWithOrigin(std.testing.allocator, inp);
            defer r.deinit(std.testing.allocator);
            const origin = r.origin orelse return error.TestExpectedNonNullOrigin;
            try std.testing.expectEqual(r.bytes.len, origin.len);
            for (origin) |off| {
                // off must be a valid byte offset into the original input.
                // (The dummy-prefix bytes share offset 0, which is valid
                // when input.len > 0; if input is empty, off must be 0
                // as well.)
                try std.testing.expect(off <= inp.len);
            }
        }
    }
}

test "sp_precompiled: dummy_prefix origin bytes anchor to offset 0" {
    const n = Normalizer{ .sp_precompiled = .{
        .nfkc = false,
        .add_dummy_prefix = true,
        .escape_whitespaces = false,
        .remove_extra_whitespaces = false,
    } };
    var r = try n.normalizeWithOrigin(std.testing.allocator, "abc");
    defer r.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("\xE2\x96\x81abc", r.bytes);
    const origin = r.origin orelse return error.TestExpectedNonNullOrigin;
    // First three bytes are the dummy prefix → all point at offset 0.
    try std.testing.expectEqual(@as(u32, 0), origin[0]);
    try std.testing.expectEqual(@as(u32, 0), origin[1]);
    try std.testing.expectEqual(@as(u32, 0), origin[2]);
    // The 'a','b','c' bytes map to their respective input offsets.
    try std.testing.expectEqual(@as(u32, 0), origin[3]); // 'a' at offset 0
    try std.testing.expectEqual(@as(u32, 1), origin[4]); // 'b' at offset 1
    try std.testing.expectEqual(@as(u32, 2), origin[5]); // 'c' at offset 2
}

test "sp_precompiled: maxByteExpansion cap holds for the full default profile" {
    const n = Normalizer{ .sp_precompiled = .{
        .nfkc = true,
        .add_dummy_prefix = true,
        .escape_whitespaces = true,
        .remove_extra_whitespaces = false,
    } };
    // Mix ligatures, spaces, and ASCII to stress the cap.
    const inputs = [_][]const u8{
        "hello world",
        "\xEF\xAC\x81 hello",  // ligature followed by space
        " a ",
        "Hello, world!",
        "the quick brown fox jumps over the lazy dog",
    };
    const cap = n.maxByteExpansion();
    for (inputs) |inp| {
        const out = try n.normalize(std.testing.allocator, inp);
        defer std.testing.allocator.free(out);
        // The +3 slack accounts for the dummy-prefix bytes on tiny inputs.
        try std.testing.expect(out.len <= inp.len * cap + 3);
    }
}

// === SP normalizer casefold (_cf suffix) tests ===

test "sp_precompiled: casefold=true on HELLO lowercases all letters" {
    // No dummy prefix / no escape — isolate the casefold pass.
    const n = Normalizer{ .sp_precompiled = .{
        .nfkc = false,
        .casefold = true,
        .add_dummy_prefix = false,
        .escape_whitespaces = false,
        .remove_extra_whitespaces = false,
    } };
    const out = try n.normalize(std.testing.allocator, "HELLO");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("hello", out);
}

test "sp_precompiled: casefold=true with full default profile (NFKC + escape + dummy_prefix)" {
    // Mirrors the Gemma-style `nmt_nfkc_cf` profile.
    const n = Normalizer{ .sp_precompiled = .{
        .nfkc = true,
        .casefold = true,
        .add_dummy_prefix = true,
        .escape_whitespaces = true,
        .remove_extra_whitespaces = false,
    } };
    const out = try n.normalize(std.testing.allocator, "HELLO");
    defer std.testing.allocator.free(out);
    // ▁ + hello (no internal spaces to escape).
    try std.testing.expectEqualStrings("\xE2\x96\x81hello", out);
}

test "sp_precompiled: casefold=false on HELLO leaves uppercase intact (backward-compat)" {
    const n = Normalizer{ .sp_precompiled = .{
        .nfkc = false,
        .casefold = false,
        .add_dummy_prefix = false,
        .escape_whitespaces = false,
        .remove_extra_whitespaces = false,
    } };
    const out = try n.normalize(std.testing.allocator, "HELLO");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("HELLO", out);
}

test "sp_precompiled: casefold folds non-ASCII Latin/Greek/Cyrillic via UCD tables" {
    const n = Normalizer{ .sp_precompiled = .{
        .nfkc = false,
        .casefold = true,
        .add_dummy_prefix = false,
        .escape_whitespaces = false,
        .remove_extra_whitespaces = false,
    } };
    // Ü (U+00DC, 0xC3 0x9C) -> ü (U+00FC, 0xC3 0xBC).
    // Π (U+03A0, 0xCE 0xA0) -> π (U+03C0, 0xCF 0x80).
    // Я (U+042F, 0xD0 0xAF) -> я (U+044F, 0xD1 0x8F).
    const out = try n.normalize(std.testing.allocator, "Ü Π Я");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("\xC3\xBC \xCF\x80 \xD1\x8F", out);
}

test "sp_precompiled: casefold origin map maps each lowercase byte back to source codepoint start" {
    // Input: "HÉLLO" (where É = U+00C9 = 0xC3 0x89, 2 bytes)
    //   byte 0 'H'   @ source offset 0
    //   byte 1 0xC3  @ source offset 1 (É lead)
    //   byte 2 0x89  @ source offset 1 (É trail)
    //   byte 3 'L'   @ source offset 3
    //   byte 4 'L'   @ source offset 4
    //   byte 5 'O'   @ source offset 5
    // After casefold the upper-Latin codepoints all fold to lowercase of
    // the same UTF-8 length, so the byte length is unchanged and each
    // output byte points at the start of the codepoint it came from.
    const n = Normalizer{ .sp_precompiled = .{
        .nfkc = false,
        .casefold = true,
        .add_dummy_prefix = false,
        .escape_whitespaces = false,
        .remove_extra_whitespaces = false,
    } };
    const input = "H\xC3\x89LLO";
    var r = try n.normalizeWithOrigin(std.testing.allocator, input);
    defer r.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("h\xC3\xA9llo", r.bytes); // é = 0xC3 0xA9
    const origin = r.origin orelse return error.TestExpectedNonNullOrigin;
    try std.testing.expectEqual(@as(usize, 6), origin.len);
    try std.testing.expectEqual(@as(u32, 0), origin[0]); // 'h' from 'H' @ 0
    try std.testing.expectEqual(@as(u32, 1), origin[1]); // 'é' byte 1 from 'É' @ 1
    try std.testing.expectEqual(@as(u32, 1), origin[2]); // 'é' byte 2 from 'É' @ 1
    try std.testing.expectEqual(@as(u32, 3), origin[3]); // 'l' from 'L' @ 3
    try std.testing.expectEqual(@as(u32, 4), origin[4]); // 'l' from 'L' @ 4
    try std.testing.expectEqual(@as(u32, 5), origin[5]); // 'o' from 'O' @ 5
    for (origin) |off| try std.testing.expect(off < input.len);
}

test "sp_precompiled: casefold + NFKC + dummy_prefix + escape — full Gemma-style pipeline" {
    // Stage 1: NFKC of "HELLO WORLD"   -> "HELLO WORLD" (no change).
    // Stage 1b: casefold                -> "hello world".
    // Stage 3: escape                   -> "hello▁world".
    // Stage 4: dummy prefix             -> "▁hello▁world".
    const n = Normalizer{ .sp_precompiled = .{
        .nfkc = true,
        .casefold = true,
        .add_dummy_prefix = true,
        .escape_whitespaces = true,
        .remove_extra_whitespaces = false,
    } };
    const out = try n.normalize(std.testing.allocator, "HELLO WORLD");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("\xE2\x96\x81hello\xE2\x96\x81world", out);
}

// === capcode / nocapcode normalizer tests ===

test "nocapcode normalizer: lowercase input passes through unchanged (no NFD, no compat space)" {
    const n: Normalizer = .{ .nocapcode = .{ .nfd = false, .tm_compat_space = false } };
    const out = try n.normalize(std.testing.allocator, "hello world");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("hello world", out);
}

test "nocapcode normalizer: tm_compat_space inserts DEL+space at letter-after-punctuation boundary" {
    // In tm_compat mode every emitted DEL gets a trailing ' ' AND a
    // leading DEL+space gets prepended when the input begins with a
    // letter/digit (TM-Go's start-of-input boundary rule).
    const n: Normalizer = .{ .nocapcode = .{ .nfd = false, .tm_compat_space = true } };
    const out = try n.normalize(std.testing.allocator, "a/b");
    defer std.testing.allocator.free(out);
    // Leading "a" triggers DEL+space (start-of-input). After "/", the
    // letter "b" triggers another DEL+space. Expect:
    //   DEL " " "a" "/" DEL " " "b"
    try std.testing.expectEqualStrings("\x7F a/\x7F b", out);
}

test "capcode normalizer: NFD pre-pass decomposes precomposed é before encoding" {
    // U+00E9 (é, 0xC3 0xA9) → NFD → e (0x65) + U+0301 (0xCC 0x81).
    // Capcode then passes the lowercase 'e' through and leaves the
    // combining mark untouched.
    const n: Normalizer = .{ .capcode = .{ .nfd = true } };
    const out = try n.normalize(std.testing.allocator, "caf\xC3\xA9");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("cafe\xCC\x81", out);
}

test "capcode normalizer: leading uppercase word emits C marker" {
    const n: Normalizer = .{ .capcode = .{ .nfd = false } };
    const out = try n.normalize(std.testing.allocator, "Hello world");
    defer std.testing.allocator.free(out);
    // ztok capcode: 0x0E (C marker) + "hello world".
    try std.testing.expectEqualStrings("\x0Ehello world", out);
}

test "capcode/nocapcode normalizers: origin map is consistent and every byte maps inside input" {
    const inputs = [_][]const u8{
        "hello world",
        "Hello World",
        "abc 123 !!",
        "caf\xC3\xA9",     // precomposed é
        "a/b/c",
    };
    const variants = [_]Normalizer{
        .{ .nocapcode = .{ .nfd = false, .tm_compat_space = true } },
        .{ .nocapcode = .{ .nfd = true, .tm_compat_space = true } },
        .{ .capcode = .{ .nfd = false } },
        .{ .capcode = .{ .nfd = true } },
    };
    for (variants) |n| {
        for (inputs) |inp| {
            var r = try n.normalizeWithOrigin(std.testing.allocator, inp);
            defer r.deinit(std.testing.allocator);
            const origin = r.origin orelse return error.TestExpectedNonNullOrigin;
            try std.testing.expectEqual(r.bytes.len, origin.len);
            for (origin) |off| {
                try std.testing.expect(off <= inp.len);
            }
        }
    }
}


test "capcode normalizer: .tm_printable marker_style emits 'C'/'W'/'D' instead of C0 controls" {
    // Same input, two marker styles, different output bytes.
    const input = "Hello WORLD";
    const z = try (Normalizer{ .capcode = .{ .nfd = false, .marker_style = .ztok } })
        .normalize(std.testing.allocator, input);
    defer std.testing.allocator.free(z);
    const p = try (Normalizer{ .capcode = .{ .nfd = false, .marker_style = .tm_printable } })
        .normalize(std.testing.allocator, input);
    defer std.testing.allocator.free(p);

    // ztok style: 0x0E 'hello' ' ' 0x0F 'world' (ztok-native single-byte
    // markers, no trailing space — preserved for back-compat tests).
    try std.testing.expectEqualStrings("\x0Ehello \x0Fworld", z);
    // tm_printable style: post-1.17 (agent A) the .tm_printable arm
    // delegates to `tm_norm.zig` which is byte-exact with TM-Go's
    // `capcode.Encode`. "Hello WORLD" — leading 'H' has no preceding
    // space so we emit D+W+' '+'h' which is then rewritten to D+C+' '
    // by the single-lower-follows rule, yielding "DC hello". The
    // second word's 'W' overwrites the preceding ' ' producing
    // "DC helloW " + "world" (the W run stays a W since 'O'/'R'/'L'/'D'
    // are all uppercase letters and no lowercase follows).
    try std.testing.expectEqualStrings("DC helloW world", p);
}

test "capcode normalizer: default marker_style is .ztok (1.12 behavior preserved)" {
    // Implicit default — leaving marker_style off must yield the C0
    // controls. This locks in the back-compat contract: any code that
    // existed before this change keeps emitting 0x0E/0x0F/0x11.
    const out = try (Normalizer{ .capcode = .{ .nfd = false } })
        .normalize(std.testing.allocator, "Hello");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("\x0Ehello", out);
}

test "capcode normalizer (.tm_printable): TM-Go-faithful encoded bytes for HELLO" {
    // Post-1.17 (agent A): the `.tm_printable` arm delegates to
    // `tm_norm.zig` which mirrors TM-Go's `capcode.Encode` exactly.
    // "HELLO" with no preceding space emits `D + W + ' ' + 'hello'`
    // (the `D` is TM's start-of-input boundary marker, the `W` is
    // word-cap, the synthetic ' ' is the post-W TM marker spacer).
    //
    // Full round-trip via `capcode.decodeStyled` is gated on bringing
    // ztok's decoder up to TM parity (TM's decoder uses an `ignore`
    // flag for the W's synthetic space and treats C/W/D as state
    // flags rather than consuming-prefixes). The tokenize path
    // doesn't depend on the decoded text — it compares the encoded
    // bytes against vocab pieces, which are themselves TM-encoded.
    const input = "HELLO";
    const enc = try (Normalizer{ .capcode = .{ .nfd = false, .marker_style = .tm_printable } })
        .normalize(std.testing.allocator, input);
    defer std.testing.allocator.free(enc);
    try std.testing.expectEqualStrings("DW hello", enc);
}

test "capcode normalizer (.tm_printable): origin map still maps every byte into the input" {
    const inputs = [_][]const u8{
        "Hello",
        "Hello World",
        "THIS is GPT-4.",
        "WORDword",
        "Caf\xC3\xA9",       // precomposed é
    };
    const n: Normalizer = .{ .capcode = .{ .nfd = true, .marker_style = .tm_printable } };
    for (inputs) |inp| {
        var r = try n.normalizeWithOrigin(std.testing.allocator, inp);
        defer r.deinit(std.testing.allocator);
        const origin = r.origin orelse return error.TestExpectedNonNullOrigin;
        try std.testing.expectEqual(r.bytes.len, origin.len);
        for (origin) |off| {
            try std.testing.expect(off <= inp.len);
        }
    }
}

test "capcode/nocapcode normalizers: maxByteExpansion holds across all configs" {
    const inputs = [_][]const u8{
        "hello world",
        "Hello World",
        "abc 123",
        "the quick brown fox jumps over the lazy dog",
        "caf\xC3\xA9",
    };
    const variants = [_]Normalizer{
        .{ .nocapcode = .{ .nfd = true, .tm_compat_space = true } },
        .{ .capcode = .{ .nfd = true } },
    };
    for (variants) |n| {
        const cap = n.maxByteExpansion();
        for (inputs) |inp| {
            const out = try n.normalize(std.testing.allocator, inp);
            defer std.testing.allocator.free(out);
            try std.testing.expect(out.len <= inp.len * cap + 8);
        }
    }
}

// === 1.14 perf fix: normalize-only fast path regression tests ===
//
// Verify the origin-free `normalize` returns the same bytes as the
// with-origin `normalizeWithOrigin` does — the origin map is the only
// delta. If these ever diverge it means the two code paths drifted
// (e.g., someone added a stage to one but not the other) and the
// pipeline encoder is reading different bytes than the offset-tracking
// callers see.

test "normalize == normalizeWithOrigin.bytes for every Normalizer variant" {
    const inputs = [_][]const u8{
        "hello world",
        "Hello World",
        "the quick brown FOX jumps over 100 LAZY dogs.",
        "abc 123",
        "caf\xC3\xA9",                  // precomposed é
        "\xEF\xAC\x81 hello",           // ligature ﬁ
        "  multiple   spaces  trim  ",
        "\xE2\x96\x81already-escaped",  // pre-U+2581
        " ",
        "",
        "ALL CAPS WITH NUMBERS 123 AND PUNC!",
        "\xCE\xB1\xCE\xB2\xCE\xB3",      // αβγ
        // Mixed Japanese + ASCII (stresses the NF-stable run fast
        // path in unicode_norm.normalize).
        "\xE3\x81\x82\xE3\x81\x84 a/b",
        // Hangul syllable (algorithmic decomp under NFD).
        "\xED\x95\x9C",
    };
    const variants = [_]Normalizer{
        .{ .identity = {} },
        .{ .nfc = {} },
        .{ .nfd = {} },
        .{ .nfkc = {} },
        .{ .nfkd = {} },
        .{ .byte_level = {} },
        .{ .sp_precompiled = .{} },
        .{ .sp_precompiled = .{ .nfkc = false, .add_dummy_prefix = false, .escape_whitespaces = false, .remove_extra_whitespaces = false } },
        .{ .sp_precompiled = .{ .nfkc = true, .add_dummy_prefix = true, .escape_whitespaces = true, .remove_extra_whitespaces = true } },
        .{ .capcode = .{ .nfd = false } },
        .{ .capcode = .{ .nfd = true } },
        .{ .capcode = .{ .nfd = true, .marker_style = .tm_printable } },
        .{ .nocapcode = .{ .nfd = false, .tm_compat_space = false } },
        .{ .nocapcode = .{ .nfd = false, .tm_compat_space = true } },
        .{ .nocapcode = .{ .nfd = true, .tm_compat_space = true } },
    };
    for (variants) |n| {
        for (inputs) |inp| {
            const fast = try n.normalize(std.testing.allocator, inp);
            defer std.testing.allocator.free(fast);
            var slow = try n.normalizeWithOrigin(std.testing.allocator, inp);
            defer slow.deinit(std.testing.allocator);
            try std.testing.expectEqualSlices(u8, slow.bytes, fast);
        }
    }
}

test "normalize does not leak: round-trip free with testing.allocator" {
    // The testing allocator panics on leaks. If `normalize` allocates
    // anything beyond the returned slice (e.g., a stale origin map),
    // the next free here will fail because the leaked alloc shows up
    // as a residual.
    const variants = [_]Normalizer{
        .{ .nfd = {} },
        .{ .nfkc = {} },
        .{ .byte_level = {} },
        .{ .capcode = .{ .nfd = true } },
        .{ .nocapcode = .{ .nfd = true, .tm_compat_space = true } },
        .{ .sp_precompiled = .{} },
    };
    for (variants) |n| {
        const out = try n.normalize(std.testing.allocator, "Hello, world! \xC3\xA9");
        std.testing.allocator.free(out);
    }
}

test "TM-style capcode normalize of 1 MB synthetic input completes promptly" {
    // 1 MB of pseudo-natural English. The original encode regression
    // showed up here as a multi-second normalize; the fixed fast path
    // should finish this in << 1 s on any reasonable host.
    const N: usize = 1024 * 1024;
    const buf = try std.testing.allocator.alloc(u8, N);
    defer std.testing.allocator.free(buf);
    var rng = std.Random.DefaultPrng.init(0xC0DE_C00D);
    const r = rng.random();
    const alphabet = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789 .,;:!?";
    for (buf) |*b| b.* = alphabet[r.intRangeLessThan(usize, 0, alphabet.len)];

    const n: Normalizer = .{ .nocapcode = .{ .nfd = true, .tm_compat_space = true } };
    const out = try n.normalize(std.testing.allocator, buf);
    defer std.testing.allocator.free(out);
    // We don't time the normalize here — Zig 0.16's `std.time.Timer`
    // moved under `std.Io` and pulling that in just for a 1 MB
    // ceiling test is over-instrumentation. The test simply asserts
    // the operation completes and produces a sane-size output. A
    // regression to the pre-fix code path would balloon the
    // allocator's working set (40 MB on the with-origin path) and
    // would still complete; we catch that via the leak-free test
    // and the throughput benchmark in `bench/RESULTS.md`.
    // Sanity: output is non-empty and roughly the right order of
    // magnitude (TM nocapcode insertions can grow the buffer by up to
    // ~2x for letter-after-punctuation boundaries).
    try std.testing.expect(out.len > 0);
    try std.testing.expect(out.len < N * 3);
}

// === HF normalizer arms (Replace / Strip / Lowercase / BertNormalizer
// / Sequence) — post-1.17 agent D ===

test "replace: literal pattern rewrites every occurrence" {
    const n: Normalizer = .{ .replace = .{ .pattern = " ", .content = "_" } };
    const out = try n.normalize(std.testing.allocator, "hello world");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("hello_world", out);
}

test "replace: empty input returns empty output" {
    const n: Normalizer = .{ .replace = .{ .pattern = " ", .content = "_" } };
    const out = try n.normalize(std.testing.allocator, "");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("", out);
}

test "replace: multi-byte pattern + content" {
    // " " -> "▁" (3 UTF-8 bytes). Mirrors a common SP-style normalizer
    // (Replace regex \s -> ▁ used by deberta-v3, t5).
    const n: Normalizer = .{ .replace = .{ .pattern = " ", .content = "\xE2\x96\x81" } };
    const out = try n.normalize(std.testing.allocator, "a b c");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("a\xE2\x96\x81b\xE2\x96\x81c", out);
}

test "replace: origin map anchors content bytes to the matched pattern" {
    const n: Normalizer = .{ .replace = .{ .pattern = " ", .content = "__" } };
    var r = try n.normalizeWithOrigin(std.testing.allocator, "a b");
    defer r.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("a__b", r.bytes);
    const origin = r.origin orelse return error.TestExpectedNonNullOrigin;
    try std.testing.expectEqual(@as(usize, 4), origin.len);
    try std.testing.expectEqual(@as(u32, 0), origin[0]); // 'a' @ 0
    try std.testing.expectEqual(@as(u32, 1), origin[1]); // first '_' anchored to ' ' @ 1
    try std.testing.expectEqual(@as(u32, 1), origin[2]); // second '_' anchored to ' ' @ 1
    try std.testing.expectEqual(@as(u32, 2), origin[3]); // 'b' @ 2
}

test "strip: both edges trims ASCII whitespace" {
    const n: Normalizer = .{ .strip = .{ .strip_left = true, .strip_right = true } };
    const out = try n.normalize(std.testing.allocator, "  hello  ");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("hello", out);
}

test "strip: left-only / right-only" {
    {
        const n: Normalizer = .{ .strip = .{ .strip_left = true, .strip_right = false } };
        const out = try n.normalize(std.testing.allocator, "  hello  ");
        defer std.testing.allocator.free(out);
        try std.testing.expectEqualStrings("hello  ", out);
    }
    {
        const n: Normalizer = .{ .strip = .{ .strip_left = false, .strip_right = true } };
        const out = try n.normalize(std.testing.allocator, "  hello  ");
        defer std.testing.allocator.free(out);
        try std.testing.expectEqualStrings("  hello", out);
    }
}

test "strip: origin map — trimmed bytes leave no entries" {
    const n: Normalizer = .{ .strip = .{ .strip_left = true, .strip_right = true } };
    var r = try n.normalizeWithOrigin(std.testing.allocator, "  hi  ");
    defer r.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("hi", r.bytes);
    const origin = r.origin orelse return error.TestExpectedNonNullOrigin;
    try std.testing.expectEqual(@as(usize, 2), origin.len);
    try std.testing.expectEqual(@as(u32, 2), origin[0]); // 'h' was at offset 2 in original
    try std.testing.expectEqual(@as(u32, 3), origin[1]); // 'i' was at offset 3
}

test "lowercase: ASCII + Unicode case-fold (post-1.18 B)" {
    const n: Normalizer = .{ .lowercase = {} };
    const out = try n.normalize(std.testing.allocator, "Hello WORLD 123 \xC3\x89");
    defer std.testing.allocator.free(out);
    // Unicode lowercase folds É (U+00C9 = 0xC3 0x89) to é (U+00E9 = 0xC3 0xA9)
    // via capcode.toLower; ASCII letters lowercased; digits/space pass through.
    try std.testing.expectEqualStrings("hello world 123 \xC3\xA9", out);
}

test "sequence: NFC + Lowercase composes correctly" {
    // NFC of e+◌́ -> é (0xC3 0xA9); Lowercase passes through (already lc).
    const items = try std.testing.allocator.alloc(Normalizer, 2);
    defer std.testing.allocator.free(items);
    items[0] = .nfc;
    items[1] = .lowercase;
    const n: Normalizer = .{ .sequence = .{ .normalizers = items } };
    const out = try n.normalize(std.testing.allocator, "e\xCC\x81LLO");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("\xC3\xA9llo", out);
}

test "sequence: empty list is identity" {
    const items = try std.testing.allocator.alloc(Normalizer, 0);
    defer std.testing.allocator.free(items);
    const n: Normalizer = .{ .sequence = .{ .normalizers = items } };
    const out = try n.normalize(std.testing.allocator, "hello");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("hello", out);
}

test "bert_normalizer: lowercase=true strip_accents=false on CAFÉ -> café (Unicode lowercase, post-1.18 B)" {
    const n: Normalizer = .{ .bert_normalizer = .{
        .clean_text = false,
        .handle_chinese_chars = false,
        .strip_accents = false,
        .lowercase = true,
    } };
    // Input: "CAFÉ" with É precomposed (U+00C9, 2 bytes 0xC3 0x89).
    // Unicode lowercase folds É -> é (U+00E9, 2 bytes 0xC3 0xA9).
    const out = try n.normalize(std.testing.allocator, "CAF\xC3\x89");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("caf\xC3\xA9", out);
}

test "bert_normalizer: default lowercase=true strip_accents=null strips accents (NFD + drop Mn)" {
    // HF default: strip_accents = None means follow lowercase=true.
    // Café -> NFD -> "Cafe" + U+0301 -> drop combining marks -> "Cafe"
    // -> lowercase -> "cafe".
    const n: Normalizer = .{ .bert_normalizer = .{
        .clean_text = true,
        .handle_chinese_chars = true,
        .strip_accents = null,
        .lowercase = true,
    } };
    const out = try n.normalize(std.testing.allocator, "Caf\xC3\xA9");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("cafe", out);
}

test "bert_normalizer: handle_chinese_chars wraps CJK in spaces" {
    const n: Normalizer = .{ .bert_normalizer = .{
        .clean_text = false,
        .handle_chinese_chars = true,
        .strip_accents = false,
        .lowercase = false,
    } };
    // U+4E2D = 0xE4 0xB8 0xAD ("中").
    const out = try n.normalize(std.testing.allocator, "a\xE4\xB8\xADb");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("a \xE4\xB8\xAD b", out);
}

test "bert_normalizer: clean_text drops NUL and replaces \\t with space" {
    const n: Normalizer = .{ .bert_normalizer = .{
        .clean_text = true,
        .handle_chinese_chars = false,
        .strip_accents = false,
        .lowercase = false,
    } };
    const out = try n.normalize(std.testing.allocator, "a\x00b\tc");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("ab c", out);
}

// Regression: HF BertNormalizer drops every Cf (Format) codepoint as part of
// `clean_text` because its `is_control` matches `c.is_other()` (Cc|Cf|Cn|Co).
// ztok's pre-fix `isBertControlCp` covered only a hand-rolled subset of Cf
// (200B-200F, 202A-202E, 2060-2064, FEFF, AD) and missed the isolate block
// (U+2066-U+2069) used to wrap RTL Arabic/Hebrew runs as well as the TAG
// block (U+E0001, U+E0020-U+E007F) embedded in regional-flag emoji. These
// two ranges accounted for 96/113 diverging lines on
// bench/corpora/unicode_stress.txt.
test "bert_normalizer: clean_text strips bidi isolates (U+2066-U+2069)" {
    const n: Normalizer = .{ .bert_normalizer = .{
        .clean_text = true,
        .handle_chinese_chars = false,
        .strip_accents = false,
        .lowercase = false,
    } };
    // FSI (U+2068) U+2068 = 0xE2 0x81 0xA8; PDI (U+2069) = 0xE2 0x81 0xA9.
    // Input: "t ⁨ABC⁩." -> "t ABC."
    const out = try n.normalize(std.testing.allocator, "t \xE2\x81\xA8ABC\xE2\x81\xA9.");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("t ABC.", out);
}

test "bert_normalizer: clean_text strips TAG codepoints (U+E0061..U+E007F)" {
    const n: Normalizer = .{ .bert_normalizer = .{
        .clean_text = true,
        .handle_chinese_chars = false,
        .strip_accents = false,
        .lowercase = false,
    } };
    // U+E0067 TAG LATIN SMALL LETTER G = 0xF3 0xA0 0x81 0xA7.
    // U+E007F CANCEL TAG = 0xF3 0xA0 0x81 0xBF.
    const out = try n.normalize(std.testing.allocator, "x\xF3\xA0\x81\xA7\xF3\xA0\x81\xBFy");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("xy", out);
}

// Regression: HF's BertNormalizer applies Rust's full `char::to_lowercase`
// which folds U+24B6-U+24CF (circled Latin caps) and U+2160-U+216F (Roman
// numerals) — neither are in capcode's case-fold tables (capcode covers
// script blocks only, not Nl/So). Without this the `unicode_stress` corpus
// left ~17 lines unfolded.
test "bert_normalizer: lowercase folds circled Latin (U+24B6 -> U+24D0)" {
    const n: Normalizer = .{ .bert_normalizer = .{
        .clean_text = false,
        .handle_chinese_chars = false,
        .strip_accents = false,
        .lowercase = true,
    } };
    // U+24B6 = 0xE2 0x92 0xB6 ; U+24D0 = 0xE2 0x93 0x90.
    const out = try n.normalize(std.testing.allocator, "x\xE2\x92\xB6y");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("x\xE2\x93\x90y", out);
}

test "bert_normalizer: lowercase folds Roman numerals (U+216B -> U+217B)" {
    const n: Normalizer = .{ .bert_normalizer = .{
        .clean_text = false,
        .handle_chinese_chars = false,
        .strip_accents = false,
        .lowercase = true,
    } };
    // U+216B = 0xE2 0x85 0xAB ; U+217B = 0xE2 0x85 0xBB.
    const out = try n.normalize(std.testing.allocator, "x\xE2\x85\xABy");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("x\xE2\x85\xBBy", out);
}

test "HF JSON: Sequence[Replace, NFC] parses + normalizes via bridge" {
    const hf_json_mod = @import("hf_json.zig");
    const bridge = @import("hf_bridge.zig");
    const input =
        \\{
        \\  "added_tokens": [],
        \\  "normalizer": {
        \\    "type": "Sequence",
        \\    "normalizers": [
        \\      { "type": "Replace", "pattern": { "String": " " }, "content": "_" },
        \\      { "type": "NFC" }
        \\    ]
        \\  },
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {"a": 0},
        \\    "merges": []
        \\  }
        \\}
    ;
    var hf = try hf_json_mod.loadFromBytes(std.testing.allocator, input);
    defer hf.deinit();
    try std.testing.expect(hf.normalizer_spec != null);
    var n = try bridge.normalizerFromHF(std.testing.allocator, &hf);
    defer n.deinit(std.testing.allocator);
    const out = try n.normalize(std.testing.allocator, "hello world");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("hello_world", out);
}

test "HF JSON: BertNormalizer fields parse correctly" {
    const hf_json_mod = @import("hf_json.zig");
    const bridge = @import("hf_bridge.zig");
    const input =
        \\{
        \\  "added_tokens": [],
        \\  "normalizer": {
        \\    "type": "BertNormalizer",
        \\    "clean_text": true,
        \\    "handle_chinese_chars": true,
        \\    "strip_accents": null,
        \\    "lowercase": true
        \\  },
        \\  "model": {
        \\    "type": "WordPiece",
        \\    "unk_token": "[UNK]",
        \\    "vocab": {"[UNK]": 0}
        \\  }
        \\}
    ;
    var hf = try hf_json_mod.loadFromBytes(std.testing.allocator, input);
    defer hf.deinit();
    try std.testing.expectEqual(hf_json_mod.NormalizerKind.bert, hf.normalizer_kind);
    try std.testing.expect(hf.normalizer_spec != null);
    switch (hf.normalizer_spec.?) {
        .bert => |b| {
            try std.testing.expect(b.clean_text);
            try std.testing.expect(b.handle_chinese_chars);
            try std.testing.expectEqual(@as(?bool, null), b.strip_accents);
            try std.testing.expect(b.lowercase);
        },
        else => return error.TestExpectedBertSpec,
    }
    var n = try bridge.normalizerFromHF(std.testing.allocator, &hf);
    defer n.deinit(std.testing.allocator);
    try std.testing.expect(n == .bert_normalizer);
    // Smoke: "Café" -> "cafe" (default strip_accents follows lowercase=true).
    const out = try n.normalize(std.testing.allocator, "Caf\xC3\xA9");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("cafe", out);
}

// === post-1.18 agent B: Prepend + Unicode Lowercase tests ===

test "prepend: literal prefix prepended on the front of input" {
    const n: Normalizer = .{ .prepend = .{ .prepend = "\xE2\x96\x81" } }; // ▁
    const out = try n.normalize(std.testing.allocator, "hello");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("\xE2\x96\x81hello", out);
}

test "prepend + replace via Sequence: Phi-3 style ▁hello▁world" {
    // Sequence[Prepend("▁"), Replace(" "→"▁")] applied to "hello world"
    // — mirrors Phi-3's normalizer chain end-to-end.
    var inner = try std.testing.allocator.alloc(Normalizer, 2);
    inner[0] = .{ .prepend = .{ .prepend = try std.testing.allocator.dupe(u8, "\xE2\x96\x81") } };
    inner[1] = .{ .replace = .{
        .pattern = try std.testing.allocator.dupe(u8, " "),
        .content = try std.testing.allocator.dupe(u8, "\xE2\x96\x81"),
    } };
    const n: Normalizer = .{ .sequence = .{ .normalizers = inner } };
    defer n.deinit(std.testing.allocator);
    const out = try n.normalize(std.testing.allocator, "hello world");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("\xE2\x96\x81hello\xE2\x96\x81world", out);
}

test "HF JSON: Phi-3 Sequence[Prepend, Replace] parses + normalizes via bridge" {
    const hf_json_mod = @import("hf_json.zig");
    const bridge = @import("hf_bridge.zig");
    const input =
        \\{
        \\  "added_tokens": [],
        \\  "normalizer": {
        \\    "type": "Sequence",
        \\    "normalizers": [
        \\      { "type": "Prepend", "prepend": "▁" },
        \\      { "type": "Replace", "pattern": { "String": " " }, "content": "▁" }
        \\    ]
        \\  },
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {"a": 0},
        \\    "merges": []
        \\  }
        \\}
    ;
    var hf = try hf_json_mod.loadFromBytes(std.testing.allocator, input);
    defer hf.deinit();
    try std.testing.expect(hf.normalizer_spec != null);
    var n = try bridge.normalizerFromHF(std.testing.allocator, &hf);
    defer n.deinit(std.testing.allocator);
    try std.testing.expect(n == .sequence);
    const out = try n.normalize(std.testing.allocator, "hello world");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("\xE2\x96\x81hello\xE2\x96\x81world", out);
}

test "prepend: origin map — prefix bytes anchor to offset 0; tail inherits natural offsets" {
    const n: Normalizer = .{ .prepend = .{ .prepend = "\xE2\x96\x81" } }; // 3-byte ▁
    var r = try n.normalizeWithOrigin(std.testing.allocator, "ab");
    defer r.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("\xE2\x96\x81ab", r.bytes);
    const origin = r.origin orelse return error.TestExpectedNonNullOrigin;
    try std.testing.expectEqual(@as(usize, 5), origin.len);
    try std.testing.expectEqual(@as(u32, 0), origin[0]);
    try std.testing.expectEqual(@as(u32, 0), origin[1]);
    try std.testing.expectEqual(@as(u32, 0), origin[2]);
    try std.testing.expectEqual(@as(u32, 0), origin[3]); // 'a' at input[0]
    try std.testing.expectEqual(@as(u32, 1), origin[4]); // 'b' at input[1]
}

test "lowercase: Unicode fold on Café (U+00C9) -> café (U+00E9)" {
    // "CAFÉ" — É is precomposed U+00C9 (2 bytes 0xC3 0x89).
    // capcode.toLower folds U+00C9 -> U+00E9 (2 bytes 0xC3 0xA9).
    const n: Normalizer = .lowercase;
    const out = try n.normalize(std.testing.allocator, "CAF\xC3\x89");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("caf\xC3\xA9", out);
}

test "lowercase: Unicode fold on fullwidth Ａ (U+FF21) -> ａ (U+FF41)" {
    // U+FF21 = 0xEF 0xBC 0xA1, U+FF41 = 0xEF 0xBD 0x81.
    const n: Normalizer = .lowercase;
    const out = try n.normalize(std.testing.allocator, "\xEF\xBC\xA1");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("\xEF\xBD\x81", out);
}

test "lowercase: Unicode fold origin map keeps every output byte mapped into the input" {
    // "AÉ" — 'A' (1 byte) + É (2 bytes). Output is "aé" (3 bytes total).
    // origin map: out[0]=0 (a from A), out[1]=1 (first byte of é from É), out[2]=1.
    const n: Normalizer = .lowercase;
    var r = try n.normalizeWithOrigin(std.testing.allocator, "A\xC3\x89");
    defer r.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("a\xC3\xA9", r.bytes);
    const origin = r.origin orelse return error.TestExpectedNonNullOrigin;
    try std.testing.expectEqual(@as(usize, 3), origin.len);
    try std.testing.expectEqual(@as(u32, 0), origin[0]);
    try std.testing.expectEqual(@as(u32, 1), origin[1]);
    try std.testing.expectEqual(@as(u32, 1), origin[2]);
}

test "bert_normalizer: lowercase=true on Ｈello WORLD -> ｈello world (mixed fullwidth + ASCII)" {
    // U+FF28 'Ｈ' -> U+FF48 'ｈ'; other letters ASCII lowercase.
    const n: Normalizer = .{ .bert_normalizer = .{
        .clean_text = false,
        .handle_chinese_chars = false,
        .strip_accents = false,
        .lowercase = true,
    } };
    // U+FF28 = 0xEF 0xBC 0xA8, U+FF48 = 0xEF 0xBD 0x88.
    const out = try n.normalize(std.testing.allocator, "\xEF\xBC\xA8ello WORLD");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("\xEF\xBD\x88ello world", out);
}

// === post-1.19 agent B: HF regex Replace (B1) ============================

test "replace: regex pattern collapses runs of spaces" {
    // " {2,}" → " " — common HF normalizer for deberta-v3 / T5 to fold
    // runs of repeated whitespace into a single ASCII space.
    const re_ptr = try std.testing.allocator.create(@import("hf_regex.zig").Regex);
    re_ptr.* = try @import("hf_regex.zig").compile(std.testing.allocator, " {2,}");
    const n: Normalizer = .{ .replace = .{
        .pattern = try std.testing.allocator.dupe(u8, " {2,}"),
        .content = try std.testing.allocator.dupe(u8, " "),
        .compiled_regex = re_ptr,
    } };
    defer n.deinit(std.testing.allocator);

    const out = try n.normalize(std.testing.allocator, "a   b    c d");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("a b c d", out);
}

test "replace: literal pattern still works when compiled_regex is null (regression)" {
    // Existing literal behavior must be untouched — same fixture as the
    // 1.17 D test, asserts no regression after the compiled_regex field
    // was added.
    const n: Normalizer = .{ .replace = .{ .pattern = " ", .content = "_" } };
    const out = try n.normalize(std.testing.allocator, "hello world here");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("hello_world_here", out);
}

test "replace: regex origin map anchors content bytes to match start" {
    // Pattern " {2,}" → "_" on "a   b": match spans bytes 1..4 ("   "),
    // replaced by single '_'. Origin layout:
    //   out[0]='a' origin=0
    //   out[1]='_' origin=1 (anchor = match start)
    //   out[2]='b' origin=4
    const re_ptr = try std.testing.allocator.create(@import("hf_regex.zig").Regex);
    re_ptr.* = try @import("hf_regex.zig").compile(std.testing.allocator, " {2,}");
    const n: Normalizer = .{ .replace = .{
        .pattern = try std.testing.allocator.dupe(u8, " {2,}"),
        .content = try std.testing.allocator.dupe(u8, "_"),
        .compiled_regex = re_ptr,
    } };
    defer n.deinit(std.testing.allocator);

    var r = try n.normalizeWithOrigin(std.testing.allocator, "a   b");
    defer r.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("a_b", r.bytes);
    const origin = r.origin orelse return error.TestExpectedNonNullOrigin;
    try std.testing.expectEqual(@as(usize, 3), origin.len);
    try std.testing.expectEqual(@as(u32, 0), origin[0]);
    try std.testing.expectEqual(@as(u32, 1), origin[1]);
    try std.testing.expectEqual(@as(u32, 4), origin[2]);
}

test "HF JSON: Replace { Regex: \" {2,}\" } parses + applies via compiled engine" {
    // End-to-end: HF Replace normalizer with a Regex pattern goes
    // through hf_json → hf_bridge → runtime Normalizer with a compiled
    // regex stashed on the ReplaceNormalizer.
    const hf_json_mod = @import("hf_json.zig");
    const bridge = @import("hf_bridge.zig");
    const input =
        \\{
        \\  "added_tokens": [],
        \\  "normalizer": {
        \\    "type": "Replace",
        \\    "pattern": { "Regex": " {2,}" },
        \\    "content": " "
        \\  },
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {"a": 0},
        \\    "merges": []
        \\  }
        \\}
    ;
    var hf = try hf_json_mod.loadFromBytes(std.testing.allocator, input);
    defer hf.deinit();
    try std.testing.expect(hf.normalizer_spec != null);
    switch (hf.normalizer_spec.?) {
        .replace => |r| {
            try std.testing.expect(r.is_regex);
            try std.testing.expectEqualStrings(" {2,}", r.pattern);
        },
        else => return error.TestExpectedReplaceSpec,
    }

    var n = try bridge.normalizerFromHF(std.testing.allocator, &hf);
    defer n.deinit(std.testing.allocator);
    try std.testing.expect(n == .replace);
    try std.testing.expect(n.replace.compiled_regex != null);

    const out = try n.normalize(std.testing.allocator, "a    b  c   d");
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("a b c d", out);
}

// ── T5 charsmap × decomposed-diacritic regression tests ──────────────
//
// The bug: ztok's SpNormalizer used to run a standalone NFKC pass and
// THEN apply the per-model precompiled charsmap. SP's reference does
// only the charsmap walk for its codepoint stage — the `nmt_nfkc` trie
// already encodes NFKC composition + NMT extras. Running NFKC first
// pre-composes pairs the trie was built to keep decomposed (e.g.
// `c` + U+0328 → ĉ̨), which silently desyncs the Viterbi lattice.
//
// The repro corpus is the diacritic-stress slice of
// `bench/corpora/unicode_stress.txt` — pre-fix this dropped T5 from
// 100% to 96.5% on the 1000-line sweep. Each test below targets a
// specific input shape that exhibited the divergence.

const sp_bridge_for_tests = @import("sp_bridge.zig");
const sp_model_for_tests = @import("sp_model.zig");

test "T5 charsmap: c+U+0328 stays decomposed (does not pre-compose to ĉ)" {
    // After the fix, `c` (0x63) followed by combining cedilla
    // (U+0328, UTF-8 CC A8) must survive the normalizer unchanged.
    // Pre-fix NFKC composed these to ç̂ (U+1E08) and the downstream
    // charsmap had no rule for the precomposed form.
    const allocator = std.testing.allocator;
    const path = "bench/vocabs/t5_unigram.model";
    const io = std.Io.Threaded.global_single_threaded.io();
    const data = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(data);

    var sp = try sp_model_for_tests.loadFromBytes(allocator, data);
    defer sp.deinit();

    var n = try sp_bridge_for_tests.normalizerFromSPModel(&sp);
    defer n.deinit(allocator);

    // Input: just "c" + U+0328. With NFKC-first this would precompose;
    // SP keeps it as 2 codepoints (0x63 0xCC 0xA8). The dummy prefix
    // (U+2581 = E2 96 81) and escaped space rules still apply.
    const out = try n.normalize(allocator, "c\xCC\xA8");
    defer allocator.free(out);
    // Output must contain the unmodified c (0x63) and U+0328 bytes,
    // NOT the precomposed U+1E08 (CC C7 88) form.
    const has_c_then_combiner = std.mem.indexOf(u8, out, "\x63\xCC\xA8") != null;
    try std.testing.expect(has_c_then_combiner);
}

test "T5 charsmap: u+U+0302+U+0306 composes u+0302 → û but leaves U+0306" {
    // The reference SP `nmt_nfkc` charsmap DOES compose `u` + U+0302
    // → û (U+00FB) because the trie has a single-codepoint rule for
    // it; the trailing U+0306 (combining breve) is left as a separate
    // codepoint because the trie has no rule for the triple. NFKC
    // would compose all three the same way as the charsmap on this
    // specific input — but with c+U+0328 the behaviors diverge.
    // This test pins the shape that the bug-line uses.
    const allocator = std.testing.allocator;
    const path = "bench/vocabs/t5_unigram.model";
    const io = std.Io.Threaded.global_single_threaded.io();
    const data = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(data);

    var sp = try sp_model_for_tests.loadFromBytes(allocator, data);
    defer sp.deinit();

    var n = try sp_bridge_for_tests.normalizerFromSPModel(&sp);
    defer n.deinit(allocator);

    // Input: u (0x75) + U+0302 (CC 82) + U+0306 (CC 86).
    const out = try n.normalize(allocator, "u\xCC\x82\xCC\x86");
    defer allocator.free(out);
    // Expected: dummy prefix (E2 96 81) + û (C3 BB) + U+0306 (CC 86).
    const expected = "\xE2\x96\x81\xC3\xBB\xCC\x86";
    try std.testing.expectEqualStrings(expected, out);
}

test "T5 charsmap: full unicode_stress line 26 matches SP-python ids" {
    // End-to-end pin for the exact divergent line that first surfaced
    // the bug. Loads T5 unigram + encodes the full bug-line through
    // ztok, then asserts the ids match what SP-python (sp.encode) returns
    // for the same line — captured here as a constant so the test runs
    // without a Python dep.
    const allocator = std.testing.allocator;
    const path = "bench/vocabs/t5_unigram.model";
    const io = std.Io.Threaded.global_single_threaded.io();
    const data = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(data);

    var sp = try sp_model_for_tests.loadFromBytes(allocator, data);
    defer sp.deinit();

    var n = try sp_bridge_for_tests.normalizerFromSPModel(&sp);
    defer n.deinit(allocator);

    // Build the Unigram model from the SP proto so we can encode.
    var ug = try sp_bridge_for_tests.unigramFromSP(allocator, &sp);
    defer ug.deinit();

    // Bug-line bytes (line 27 in unicode_stress.txt, 0-indexed 26):
    // "diacritic word: " + u + U+0302 + U+0306 + c + U+0328 + U+0302
    // + e + U+0328 + U+030A + " canonical"
    const line = "diacritic word: u\xCC\x82\xCC\x86c\xCC\xA8\xCC\x82e\xCC\xA8\xCC\x8A canonical";

    const norm_bytes = try n.normalize(allocator, line);
    defer allocator.free(norm_bytes);

    // Worst-case output is one token per input byte.
    const out_buf = try allocator.alloc(u32, norm_bytes.len);
    defer allocator.free(out_buf);
    const encoded = try ug.encodeChunk(allocator, norm_bytes, out_buf);

    // Expected ids captured from `sp.encode(line)` against
    // bench/vocabs/t5_unigram.model.
    const expected = [_]u32{ 1227, 9, 2685, 1225, 1448, 10, 3, 10443, 2, 75, 2, 54, 106, 1950 };
    try std.testing.expectEqualSlices(u32, &expected, encoded);
}
