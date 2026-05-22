//! RAG-aware chunking on top of `Pipeline.encodeWithOffsets`.
//!
//! Given an input text and a Pipeline, split the token stream into
//! overlapping windows of at most `max_tokens` tokens, where each
//! window also carries the byte range it covers in the original
//! input. The chunker is a thin layer on top of `encodeWithOffsets`
//! and never reaches into Pipeline internals.
//!
//! Boundary modes determine where windows snap. `.token` is the
//! default: strictly N tokens per window. `.codepoint`, `.word`,
//! `.sentence`, and `.paragraph` snap window boundaries to natural
//! splits in the input, sliding the window end LEFT to the nearest
//! such boundary. If snapping cannot find a boundary inside the
//! window, the window is emitted at its max-token cap so progress is
//! guaranteed.
//!
//! Memory: every `Chunk` borrows its `ids` slice from a single arena
//! owned by the returned `ChunkResult`. There are no per-chunk
//! allocations on the hot path. Call `ChunkResult.deinit` when done.
//!
//! Concurrency: stateless — safe to call from multiple threads with
//! independent Pipelines (and the same Pipeline value, since it's
//! value-typed and `encodeWithOffsets` only touches its inputs).
//!
//! Sentence segmenter: a small heuristic state machine over raw bytes
//! that recognizes English abbreviations (Mr./Dr./U.S.A./...), three
//! ellipsis dialects (`...`, multi-dot runs, mid-sentence vs
//! sentence-end use), decimal numbers / URLs / file extensions (a `.`
//! sandwiched between two alphanumeric chars is never a boundary),
//! quoted dialog (`'I left.' She waved.` → two sentences), CJK
//! punctuation (`。`, `！`, `？` at U+3002 / U+FF01 / U+FF1F),
//! period-final acronyms (`We visited the U.S.` — recognized by the
//! `([A-Z]\.){2,}$` shape, no locale list needed), nested quote depth
//! (`"He said 'wait. then go.' I left."` → 2 sentences, not 4 — the
//! inner closing `'` doesn't fire because the outer `"` is still open;
//! depth resets at paragraph breaks to bound state), other-script
//! terminators (Arabic `؟` U+061F, Armenian `։` U+0589, Tibetan `།`
//! U+0F0D, Ethiopic `።` U+1362), and an opt-in
//! `aggressive_lowercase` flag that fires a boundary on lowercase
//! sentence-opener words like `then`/`but`/`and`/pronouns. Includes a
//! handful of common multi-language abbreviation lists for French,
//! German, Spanish, and Italian opt-in via `ChunkOptions.sentence_locale`.
//! It is still a HEURISTIC: it has no model of clause structure, no
//! capitalization-aware NER, and trades correctness on adversarial
//! input for zero dependencies and predictable runtime. Production RAG
//! pipelines at scale should hook in ICU's `BreakIterator` or a real
//! sentence segmenter; ztok's segmenter is intended for "good enough"
//! chunking of well-formed prose.
//!
//! Scriptio-continua segmentation (Thai / Lao / Chinese / Japanese):
//! these scripts are written without inter-word spaces, so the
//! `.word` boundary mode (which keys on whitespace) is a no-op for
//! them. `.word_dict` ships dictionary-based longest-match word
//! segmentation backed by tiny built-in dicts:
//!
//!   * Thai (U+0E00-U+0E7F)  — `chunk_dict_th.zig`, ~465 entries
//!   * Lao (U+0E80-U+0EFF)   — `chunk_dict_lo.zig`, ~465 entries
//!   * Chinese (CJK Unified) — `chunk_dict_zh.zig`, ~880 entries
//!   * Japanese (kana+kanji) — `chunk_dict_ja.zig`, ~840 entries
//!
//! Dict-detected words snap; unknown spans fall back to single-
//! codepoint boundaries. This is APPROXIMATION-GRADE, intended for
//! "good enough" RAG chunking — NOT for linguistic analysis. The
//! shipped dicts are deliberately tiny (full production dicts are
//! 50K-700K entries) to keep binary size sane; users who need higher
//! quality should hook in ICU's `BreakIterator` or a real morphological
//! analyzer (MeCab / jieba / PyThaiNLP).
//!
//! For Thai/Lao sentence boundaries (no built-in terminator
//! punctuation in those scripts), the `.sentence` mode with
//! `Locale.th` or `Locale.lo` treats `\n\n` paragraph breaks as
//! sentence boundaries when the surrounding window is dominantly Thai
//! or Lao. True linguistic Thai sentence segmentation requires word-
//! segmentation first; this is an approximation.

const std = @import("std");

const ztok = @import("root.zig");
const TokenId = ztok.TokenId;
const Span = ztok.Span;
const Pipeline = ztok.Pipeline;
const unicode_props = @import("unicode_props.zig");

/// Tiny bundled word dictionaries for scriptio-continua segmentation.
/// Re-exported as `chunk.dict_th` / `chunk.dict_lo` / `chunk.dict_zh`
/// / `chunk.dict_ja` for callers that want to inspect or replace the
/// shipped lists.
pub const dict_th = @import("chunk_dict_th.zig");
pub const dict_lo = @import("chunk_dict_lo.zig");
pub const dict_zh = @import("chunk_dict_zh.zig");
pub const dict_ja = @import("chunk_dict_ja.zig");

pub const Boundary = enum {
    /// Pure token-count windows.
    token,
    /// Snap to the nearest token boundary that is ALSO a UTF-8
    /// codepoint boundary in the original input. For identity-style
    /// pipelines (where every token is already codepoint-aligned —
    /// the byte-level BPE case as well as the `byte_id` baseline when
    /// the input is pure ASCII) this is a no-op vs `.token`. The
    /// difference shows up under byte-fallback BPE / `byte_id` on
    /// inputs containing multi-byte codepoints: a single 4-byte
    /// emoji split across 4 byte-tokens can otherwise leave a chunk
    /// whose final token is the first byte of an emoji and whose
    /// neighbor opens with a UTF-8 continuation byte. This mode
    /// detects that case and rolls the boundary back to the previous
    /// token whose `end` lands on a UTF-8 leading byte (or on
    /// end-of-input).
    codepoint,
    /// Snap to a token whose preceding byte is ASCII or Unicode
    /// whitespace. Inspects the byte before `byte_start` of the
    /// candidate token in the ORIGINAL input. Works correctly under
    /// any normalizer (NFC/NFD/NFKC/NFKD/byte_level) — origin
    /// translation in `encodeWithOffsets` ensures offsets index the
    /// original bytes.
    word,
    /// Snap to a dictionary-detected word boundary in the original
    /// input. Designed for scriptio-continua scripts (Thai, Lao,
    /// Chinese, Japanese) where `.word` is a no-op because there are
    /// no spaces between words. The segmenter walks the input doing
    /// longest-prefix lookup against the bundled tiny per-script
    /// dictionaries (see `chunk_dict_*.zig`); spans that match the
    /// dict become word boundaries. Unknown spans fall back to a
    /// per-codepoint boundary so progress is always guaranteed.
    ///
    /// ASCII whitespace (and Unicode whitespace, via the same
    /// `isWordBoundaryByte` predicate as `.word`) is ALSO treated as
    /// a word boundary on this path — so mixed-script input (English
    /// interleaved with Chinese, hiragana mixed with kanji + ASCII
    /// numerals) gets sensible boundaries everywhere. The dict
    /// lookups only fire on bytes whose lead codepoint sits in one of
    /// the supported script blocks; pure ASCII / Cyrillic / Devanagari
    /// input behaves identically to `.word`.
    ///
    /// Approximation-grade: the bundled dicts ship ~500-1000 entries
    /// per language to keep binary size sane. Production RAG should
    /// hook in ICU's `BreakIterator` or a full morphological analyzer
    /// (jieba / MeCab / PyThaiNLP).
    word_dict,
    /// Snap to a sentence boundary in the original input. Recognizes
    /// `?` `!` and `.` followed by whitespace + capital letter (or
    /// end of text), with carve-outs for: known abbreviations
    /// (`default_abbreviations` for English, `multilang_abbreviations`
    /// for the European-locale union; pick one with
    /// `ChunkOptions.sentence_locale`); ellipsis (`...`, mid-sentence
    /// use suppressed); decimals / URLs / file extensions (`.`
    /// sandwiched between two alphanumeric chars); quoted dialog
    /// (closing quote after `.`/`!`/`?`); CJK terminators
    /// (`。` U+3002, `！` U+FF01, `？` U+FF1F — no abbreviation
    /// interaction); other-script terminators (`؟` Arabic U+061F,
    /// `։` Armenian U+0589, `།` Tibetan U+0F0D, `።` Ethiopic
    /// U+1362); period-final ALL-CAPS acronyms (`U.S.`, `U.S.A.` —
    /// detected by the dotted-capital shape, not a locale list); and
    /// nested-quote depth tracking (a closing quote inside a still-open
    /// outer quote does NOT close the sentence — depth resets at
    /// paragraph breaks). Opt-in to lowercase-opener recognition via
    /// `ChunkOptions.aggressive_lowercase`. Override the abbreviation
    /// list via `ChunkOptions.sentence_abbreviations` (REPLACES the
    /// locale default rather than extending it). Still a heuristic —
    /// production-grade segmentation should use ICU.
    sentence,
    /// Snap to a token whose preceding two bytes are `\n\n`.
    paragraph,
};

/// Built-in list of English abbreviations the sentence segmenter
/// treats as non-terminal when they appear immediately before a `.`.
/// Stored WITHOUT the trailing dot — the segmenter does `prev word ==
/// abbrev && current byte == '.'`. Intentionally small and
/// English-only; pass a custom list via
/// `ChunkOptions.sentence_abbreviations` if you need more, or pick a
/// non-English locale via `ChunkOptions.sentence_locale`.
///
/// Entries with internal dots (e.g. `Ph.D`, `M.D`, `B.A`, `R.N`) match
/// the word-before-final-dot scan in `isAbbreviationBefore`, which
/// walks back across letters / digits / internal dots. The segmenter
/// already treats the EARLIER internal dot as non-terminal via the
/// alphanumeric-sandwich rule (`isAsciiAlnum(before) and
/// isAsciiAlnum(after)` in `isDotBoundaryAt`), so only the TRAILING
/// dot needs an abbreviation entry.
pub const default_abbreviations: []const []const u8 = &.{
    // 1.13 set.
    "Mr",   "Mrs",   "Ms",   "Dr",   "Prof", "St",
    "Sr",   "Jr",    "Mt",   "Ave",  "Blvd", "Capt",
    "Col",  "Gen",   "Hon",  "Inc",  "Ltd",  "Maj",
    "Rev",  "Sgt",   "vs",   "etc",  "i.e",  "e.g",
    "cf",   "approx", "no",
    // 1.15 additions: military ranks, additional titles, academic
    // degrees (both with and without internal dots — written prose
    // alternates between `Ph.D.` and `PhD`).
    "Lt",   "Cpl",   "Cmdr", "Adm",  "Atty",
    "PhD",  "MD",    "BA",   "RN",   "MA",  "MS",
    "MSc",  "BSc",   "DDS",  "DVM",  "CPA",
    "Ph.D", "M.D",   "B.A",  "R.N",
};

/// French abbreviations common in journalistic prose. Match is
/// case-sensitive on the immediately-preceding alphanumeric run.
pub const french_abbreviations: []const []const u8 = &.{
    "M",   "Mme", "Mlle", "Dr",  "Pr",  "St",  "Ste",
    "app", "av",  "ch",   "fig", "p",   "pp",  "réf",
    "vol", "cf",  "etc",
};

/// German abbreviations.
pub const german_abbreviations: []const []const u8 = &.{
    "Hr",   "Fr",   "Dr",  "Prof", "Sg",  "Kfm",
    "Mag",  "Dipl", "Ing", "usw",  "bzw", "bzgl",
    "ca",   "etc",  "gem", "geg",  "ggf", "z",
    "B",    "u",    "a",   "d",    "h",   "Nr",
    "Str",
};

/// Spanish abbreviations.
pub const spanish_abbreviations: []const []const u8 = &.{
    "Sr",   "Sra",  "Srta", "Lic",  "Dr",   "Dra",
    "Ing",  "Prof", "p",    "ej",   "Av",   "Cía",
    "Dpto", "Ud",   "Uds",  "etc",  "cf",
};

/// Italian abbreviations. Notes: "Sig.ra" is a two-token abbrev (a
/// dot followed by "ra"), the segmenter's preceding-word scan looks
/// back across internal dots so "Sig.ra" is recognized as a single
/// abbreviation when the trailing dot is the one being inspected.
pub const italian_abbreviations: []const []const u8 = &.{
    "Sig", "Sigra", "Dott", "Prof", "Avv", "ecc",
    "cc",  "cm",    "km",   "mg",   "p",   "pp",
};

/// Hindi abbreviations. Includes both Devanagari-script honorifics
/// (`डॉ` Dr., `श्री` Mr., `श्रीमती` Mrs., `कुमार` Mr. (unmarried),
/// `आदि` etc., `अर्थात` i.e.) and the Romanized variants common in
/// English-script Indian news prose. Devanagari entries are
/// multi-codepoint UTF-8; the segmenter's `isAbbreviationBefore` walks
/// back over BOTH ASCII and Unicode letter codepoints to match.
pub const hindi_abbreviations: []const []const u8 = &.{
    // Devanagari.
    "डॉ",        // Doctor (डॉ.)
    "श्री",       // Shri / Mr.
    "श्रीमती",     // Shrimati / Mrs.
    "कुमार",      // Kumar (unmarried male)
    "कुमारी",     // Kumari (unmarried female)
    "आदि",       // adi / etc.
    "अर्थात",     // arthat / i.e.
    "पृ",         // pri / page
    "सं",         // san / number/edition
    // Romanized.
    "Sh",   "Sri",  "Smt",  "Km",    "Kr",
    "adi",  "arthat", "Pt", "Dr",   "Shri",
};

/// Vietnamese abbreviations. Honorifics (Ô, Bà, Cô, Anh, Chị, Ông),
/// academic / professional titles (TS doctor, GS professor, PGS
/// associate professor, BS medical doctor, KTS architect, KS engineer,
/// HS student), and administrative-subdivision short forms (TP city,
/// TX town, TT town center, P. ward, Q. district, H. district/county).
/// All match via the ASCII letter walk-back; Vietnamese diacritic
/// marks are decoded as UTF-8 multi-byte sequences and matched bytewise.
pub const vietnamese_abbreviations: []const []const u8 = &.{
    // Honorifics.
    "Ô",    "Ông",  "Bà",   "Cô",   "Anh",  "Chị",
    // Academic / professional.
    "TS",   "GS",   "PGS",  "BS",   "KTS",  "KS",
    "HS",   "SV",   "ThS",  "CN",
    // Administrative subdivisions.
    "TP",   "TX",   "TT",   "P",    "Q",    "H",
    "X",    "TỉNH", "Q.",
};

/// Polish abbreviations. Academic / professional titles (dr, prof,
/// mgr, inż, arch), military ranks (płk colonel, ppłk lt-colonel, gen
/// general, mjr major, kpt captain), religious (ks priest, św saint,
/// bp bishop), and common written-prose abbreviations (np for
/// example, tj that is, tzn that means, ds for, nr number).
pub const polish_abbreviations: []const []const u8 = &.{
    // Academic — both lowercase (Polish convention) and capitalized
    // (sentence-initial / proper-name usage).
    "dr",   "prof", "mgr",   "inż",  "arch", "hab",
    "doc",  "dypl",
    "Dr",   "Prof", "Mgr",   "Inż",
    // Military.
    "płk",  "ppłk", "gen",   "mjr",  "kpt",  "por",
    "ppor", "sierż", "kpr",
    // Religious.
    "ks",   "św",   "bp",    "abp",  "o",
    // Common written-prose.
    "np",   "tj",   "tzn",   "ds",   "nr",   "ul",
    "al",   "tzw",  "itd",   "itp",  "tys",
};

/// Union of all multi-language lists above plus the English defaults.
/// Use when you don't know the input locale up front. This is the
/// list `abbrevsForLocale(.multilang_union)` returns and is the
/// recommended default for mixed-language journalistic corpora.
pub const multilang_abbreviations: []const []const u8 = blk: {
    const total: usize = default_abbreviations.len +
        french_abbreviations.len +
        german_abbreviations.len +
        spanish_abbreviations.len +
        italian_abbreviations.len +
        hindi_abbreviations.len +
        vietnamese_abbreviations.len +
        polish_abbreviations.len;
    var arr: [total][]const u8 = undefined;
    var i: usize = 0;
    for (default_abbreviations) |s| {
        arr[i] = s;
        i += 1;
    }
    for (french_abbreviations) |s| {
        arr[i] = s;
        i += 1;
    }
    for (german_abbreviations) |s| {
        arr[i] = s;
        i += 1;
    }
    for (spanish_abbreviations) |s| {
        arr[i] = s;
        i += 1;
    }
    for (italian_abbreviations) |s| {
        arr[i] = s;
        i += 1;
    }
    for (hindi_abbreviations) |s| {
        arr[i] = s;
        i += 1;
    }
    for (vietnamese_abbreviations) |s| {
        arr[i] = s;
        i += 1;
    }
    for (polish_abbreviations) |s| {
        arr[i] = s;
        i += 1;
    }
    const out = arr;
    break :blk &out;
};

/// Sentence-segmenter locale. Controls which built-in abbreviation
/// list is consulted when `ChunkOptions.sentence_abbreviations` is
/// null. `.en` is the default and uses `default_abbreviations`.
///
/// 1.15 adds Hindi (`.hi`), Vietnamese (`.vi`), and Polish (`.pl`) to
/// the original 5-locale set. Hindi uses Devanagari-script
/// abbreviations like `डॉ` — the segmenter walks back over multi-byte
/// UTF-8 letter codepoints to match them, not just ASCII.
pub const Locale = enum {
    en,
    fr,
    de,
    es,
    it,
    hi,
    vi,
    pl,
    /// Thai (U+0E00-U+0E7F). Has no traditional sentence terminator
    /// punctuation; the segmenter treats `\n\n` paragraph breaks
    /// (and any of the Latin/CJK terminators it already recognizes,
    /// for mixed input) as sentence boundaries. The abbreviation list
    /// is empty because Thai prose rarely uses period-abbreviations.
    /// True linguistic Thai sentence segmentation requires word-
    /// segmentation first — pair `.sentence` + `.th` with `.word_dict`
    /// for higher-quality results.
    th,
    /// Lao (U+0E80-U+0EFF). Same approximation-grade approach as
    /// `.th`: no traditional sentence-terminator punctuation, so
    /// `\n\n` paragraph breaks (plus any cross-script terminators in
    /// mixed text) drive sentence boundaries. Empty abbreviation list.
    lo,
    /// Union of every locale's abbreviation list. Recommended for
    /// mixed-language corpora when the per-document locale is
    /// unknown — false-negative-biased (treats more dots as
    /// abbreviations than any single locale would on its own).
    multilang_union,
};

/// Empty list — used by Thai/Lao locales, which have no period-
/// abbreviation pattern in normal prose. Separate const so the
/// pointer is stable across calls.
pub const empty_abbreviations: []const []const u8 = &.{};

/// Resolve a `Locale` to its built-in abbreviation list. Mirror of
/// the per-locale `pub const`s above; provided so callers can pass
/// the list around (e.g. into `countSentences` in tests) without
/// touching internal layout.
pub fn abbrevsForLocale(locale: Locale) []const []const u8 {
    return switch (locale) {
        .en => default_abbreviations,
        .fr => french_abbreviations,
        .de => german_abbreviations,
        .es => spanish_abbreviations,
        .it => italian_abbreviations,
        .hi => hindi_abbreviations,
        .vi => vietnamese_abbreviations,
        .pl => polish_abbreviations,
        .th => empty_abbreviations,
        .lo => empty_abbreviations,
        .multilang_union => multilang_abbreviations,
    };
}

/// Returns the bundled word dictionary for a locale, or null when no
/// dict is shipped (most locales — only Thai/Lao/Chinese/Japanese
/// have dicts here). When `.word_dict` is used with a no-dict locale
/// the segmenter still falls back to per-codepoint boundaries (and
/// whitespace), which matches `.word` exactly for whitespace-separated
/// languages.
pub fn dictForLocale(locale: Locale) ?[]const []const u8 {
    return switch (locale) {
        .th => dict_th.words,
        .lo => dict_lo.words,
        else => null,
    };
}

pub const ChunkOptions = struct {
    max_tokens: u32,
    overlap_tokens: u32 = 0,
    boundary: Boundary = .token,
    include_offsets: bool = true,
    /// REPLACES the sentence segmenter's built-in abbreviation list
    /// when non-null. Strings are matched case-sensitively against
    /// the word preceding a `.` (without the dot). Set to `&.{}` to
    /// disable abbreviation handling entirely. Note: this REPLACES
    /// rather than extends — if you want the locale defaults plus
    /// your domain terms, concatenate them yourself.
    sentence_abbreviations: ?[]const []const u8 = null,
    /// Picks which built-in abbreviation list the sentence segmenter
    /// uses when `sentence_abbreviations` is null. Defaults to `.en`
    /// for backward compatibility. `.multilang_union` is the
    /// recommended setting for mixed-language journalistic corpora.
    sentence_locale: Locale = .en,
    /// HEURISTIC (off by default): allow `.`/`!`/`?` followed by
    /// whitespace + a lowercase letter to count as a sentence boundary
    /// when that lowercase word looks like a "sentence opener" — i.e.
    /// it matches a short curated list of words people use to start
    /// sentences in informal prose (`the`, `a`, `an`, `and`, `but`,
    /// `or`, `so`, `then`, `for`, `nor`, `yet`, `however`, `because`,
    /// plus the pronouns `i`, `it`, `he`, `she`, `they`, `we`, `you`).
    /// Catches things like `"He left. then she arrived."` where the
    /// writer didn't capitalize. Off by default to preserve back-compat
    /// with the strict capital-after-period rule. The list is fixed
    /// and English-only; production use should layer a real segmenter
    /// (ICU's `BreakIterator`) over ztok for this case.
    aggressive_lowercase: bool = false,
};

pub const Chunk = struct {
    /// Token ids in this chunk; borrowed from the parent slab inside
    /// `ChunkResult.arena`.
    ids: []const TokenId,
    /// Byte range in the ORIGINAL input that this chunk covers.
    byte_start: u32,
    byte_end: u32,
    /// Token-index range in the parent encoding (half-open).
    token_start: u32,
    token_end: u32,
};

pub const ChunkResult = struct {
    chunks: []Chunk,
    arena: std.heap.ArenaAllocator,

    pub fn deinit(self: *ChunkResult) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

pub const Error = error{
    /// `overlap_tokens >= max_tokens` would never make progress.
    OverlapTooLarge,
    /// `max_tokens == 0`.
    InvalidMaxTokens,
} || std.mem.Allocator.Error;

/// Chunk `text` according to `opts`. The returned `ChunkResult` owns
/// an arena that backs every chunk's `ids` slice.
pub fn chunkText(
    allocator: std.mem.Allocator,
    pipeline: Pipeline,
    text: []const u8,
    opts: ChunkOptions,
) !ChunkResult {
    if (opts.max_tokens == 0) return error.InvalidMaxTokens;
    if (opts.overlap_tokens >= opts.max_tokens) return error.OverlapTooLarge;

    var arena: std.heap.ArenaAllocator = .init(allocator);
    errdefer arena.deinit();
    const aa = arena.allocator();

    // We need the offsets to do RAG-aware chunking, so always encode
    // with offsets. include_offsets is honored at output assembly:
    // when false we still need them internally but won't re-expose
    // them beyond what callers need.
    _ = opts.include_offsets;

    var enc = try pipeline.encodeWithOffsets(allocator, text);
    defer enc.deinit(allocator);

    // Empty input -> no chunks. (Offset-less zero-token encodings
    // also fall through here.)
    if (enc.ids.len == 0) {
        return .{
            .chunks = try aa.alloc(Chunk, 0),
            .arena = arena,
        };
    }

    // Slab-copy the ids once into the arena, then every chunk's
    // `ids` is a subslice of `slab`. No per-chunk allocation.
    const slab = try aa.alloc(TokenId, enc.ids.len);
    @memcpy(slab, enc.ids);

    // Conservative cap on chunk count: stride = max - overlap, so
    // at most ceil(n / stride) + 1 chunks.
    const stride: u32 = opts.max_tokens - opts.overlap_tokens;
    const n_tokens: u32 = @intCast(enc.ids.len);
    const cap: usize = (@as(usize, n_tokens) / stride) + 2;

    var chunks: std.ArrayList(Chunk) = .empty;
    defer chunks.deinit(aa);
    try chunks.ensureTotalCapacity(aa, cap);

    var start: u32 = 0;
    while (start < n_tokens) {
        const hard_end: u32 = @min(start + opts.max_tokens, n_tokens);

        // Snap hard_end LEFT to the requested boundary. The snapped
        // end MUST stay strictly greater than `start` (i.e. emit at
        // least one token) so we always make progress.
        const abbrevs = opts.sentence_abbreviations orelse abbrevsForLocale(opts.sentence_locale);
        const seg_ctx: SegmenterCtx = .{
            .abbreviations = abbrevs,
            .aggressive_lowercase = opts.aggressive_lowercase,
            .require_double_newline = switch (opts.sentence_locale) {
                .th, .lo => true,
                else => false,
            },
        };
        const snapped_end = switch (opts.boundary) {
            .token => hard_end,
            .codepoint => snapLeftCodepoint(text, enc.offsets, start, hard_end, n_tokens),
            .word => snapLeft(text, enc.offsets, start, hard_end, isWordBoundaryByte),
            .word_dict => snapLeftWordDict(text, enc.offsets, start, hard_end),
            .sentence => snapLeftSentence(text, enc.offsets, start, hard_end, seg_ctx),
            .paragraph => snapLeftParagraph(text, enc.offsets, start, hard_end),
        };

        // If snapping erased the whole window (e.g. no boundary inside
        // it), fall back to the hard cap so we don't stall.
        const end: u32 = if (snapped_end > start) snapped_end else hard_end;

        const byte_start = enc.offsets[start].start;
        const byte_end = enc.offsets[end - 1].end;

        try chunks.append(aa, .{
            .ids = slab[start..end],
            .byte_start = byte_start,
            .byte_end = byte_end,
            .token_start = start,
            .token_end = end,
        });

        if (end >= n_tokens) break;

        // Advance: drop `stride` tokens, keep the last `overlap` for
        // the next chunk. If we snapped short of the hard cap, the
        // "natural" advance is from `end`, not from `hard_end` — this
        // means the next chunk starts a tad earlier than a pure-stride
        // walk would, which is the right behavior for boundary modes.
        const advance = if (opts.boundary == .token)
            stride
        else
            // For boundary-aware modes, advance by the snapped span
            // minus overlap. Guards against zero/negative advance.
            blk: {
                const span = end - start;
                if (span <= opts.overlap_tokens) break :blk @as(u32, 1);
                break :blk span - opts.overlap_tokens;
            };

        start += advance;
    }

    const out_chunks = try aa.dupe(Chunk, chunks.items);
    return .{
        .chunks = out_chunks,
        .arena = arena,
    };
}

/// Generic LEFT-snap: scan candidate chunk-end tokens in (start, hard_end]
/// from the right. For each candidate i, we treat the boundary as
/// lying between token i-1 and token i; the chunk would be `[start..i)`
/// (half-open token range), ending at byte `offsets[i-1].end`. We snap
/// when that boundary byte satisfies `pred`.
///
/// Returns the snapped `end` (exclusive); chunk = `tokens[start..end]`.
/// Returns `hard_end` if no suitable boundary is found.
fn snapLeft(
    text: []const u8,
    offsets: []const Span,
    start: u32,
    hard_end: u32,
    pred: *const fn ([]const u8, u32) bool,
) u32 {
    if (hard_end <= start + 1) return hard_end;
    var i: u32 = hard_end;
    while (i > start + 1) : (i -= 1) {
        // Boundary byte = offsets[i-1].end (first byte AFTER the chunk).
        const boundary = offsets[i - 1].end;
        if (boundary == 0 or boundary > text.len) continue;
        if (pred(text, boundary)) return i;
    }
    return hard_end;
}

fn snapLeftParagraph(
    text: []const u8,
    offsets: []const Span,
    start: u32,
    hard_end: u32,
) u32 {
    if (hard_end <= start + 1) return hard_end;
    var i: u32 = hard_end;
    while (i > start + 1) : (i -= 1) {
        const boundary = offsets[i - 1].end;
        if (boundary < 2 or boundary > text.len) continue;
        // Chunk ends at `boundary`; want text[boundary-2..boundary]
        // == "\n\n" so the paragraph break sits INSIDE the chunk.
        if (text[boundary - 1] == '\n' and text[boundary - 2] == '\n') return i;
    }
    return hard_end;
}

/// Snap so the chunk never ends mid-UTF-8-codepoint.
///
/// Walks candidate ends from `hard_end` downward. The chunk is valid
/// at candidate `i` if the byte AT `offsets[i-1].end` (the first byte
/// after the chunk in the original input) is a UTF-8 leading byte —
/// i.e. NOT a continuation byte `10xxxxxx`. End-of-input also counts.
///
/// Under identity / byte-level BPE on UTF-8-clean input this is a
/// no-op: tokens are already codepoint-aligned. The mode earns its
/// keep on byte-fallback BPE / `byte_id` over multi-byte codepoints,
/// where a single emoji might be 4 byte-tokens — if `hard_end` cuts
/// after the 2nd, the boundary byte is a continuation and we roll
/// back to the 4th (or, if that's outside the window, to whichever
/// earlier token closes a codepoint).
///
/// Falls back to `hard_end` if no valid boundary exists in
/// `(start, hard_end]` — `chunkText` will then defer to the hard cap
/// so progress is preserved. `n_tokens` is the parent encoding's
/// total token count; used only to short-circuit when `hard_end ==
/// n_tokens` (the final chunk always closes at end-of-input, which
/// is by definition a codepoint boundary).
fn snapLeftCodepoint(
    text: []const u8,
    offsets: []const Span,
    start: u32,
    hard_end: u32,
    n_tokens: u32,
) u32 {
    if (hard_end <= start + 1) return hard_end;
    _ = n_tokens; // reserved for future use (e.g. early-exit at EOT)
    var i: u32 = hard_end;
    while (i > start) : (i -= 1) {
        const boundary = offsets[i - 1].end;
        // boundary is an index into `text`. The chunk would be
        // [start..i); boundary == first byte AFTER the chunk.
        // boundary == text.len: trivially a codepoint boundary.
        // boundary > text.len: malformed; skip.
        if (boundary > text.len) continue;
        if (boundary == text.len) return i;
        const b = text[boundary];
        // Continuation byte = 10xxxxxx. Anything else (ASCII 0xxxxxxx
        // or leading 11xxxxxx) opens a fresh codepoint.
        if ((b & 0xC0) != 0x80) return i;
    }
    return hard_end;
}

/// Dict-based word boundary snapper for scriptio-continua scripts.
///
/// Strategy: for each candidate boundary position p (in [start+1, hard_end])
/// inspect whether p falls on:
///   (a) A whitespace boundary (same as `.word` — handles mixed-script
///       input where Thai/Chinese/Japanese is interleaved with English).
///   (b) An end-of-text boundary (trivially yes).
///   (c) A script-run boundary: the byte before p is in a supported
///       script and the byte at p is in a DIFFERENT script. This
///       catches transitions like Latin→Chinese.
///   (d) A dict-detected word boundary INSIDE a same-script run. We
///       locate the start of the script run containing p, then walk
///       forward from there doing longest-prefix matches against the
///       script's bundled dict. p is a boundary iff it lands on the
///       end of some matched word or on a single-codepoint fallback
///       boundary inside the run.
///
/// Falls back to `hard_end` when no boundary is found in the window —
/// `chunkText` handles the fallback-progress case.
fn snapLeftWordDict(
    text: []const u8,
    offsets: []const Span,
    start: u32,
    hard_end: u32,
) u32 {
    if (hard_end <= start + 1) return hard_end;
    var i: u32 = hard_end;
    while (i > start + 1) : (i -= 1) {
        const boundary = offsets[i - 1].end;
        if (boundary == 0 or boundary > text.len) continue;
        if (isWordDictBoundaryByte(text, boundary)) return i;
    }
    return hard_end;
}

/// Per-position word-boundary check for `.word_dict`. The `byte_pos`
/// is the candidate boundary (first byte AFTER the chunk-end). True
/// when the boundary aligns with (a) a whitespace transition, (b) a
/// script transition into/out of a supported scriptio-continua block,
/// or (c) a dict-detected word end inside a same-script run.
fn isWordDictBoundaryByte(text: []const u8, byte_pos: u32) bool {
    if (byte_pos == 0 or byte_pos > text.len) return false;
    if (byte_pos == text.len) return true;
    // Whitespace path inherits the `.word` semantics: any whitespace
    // (ASCII or Unicode) immediately before `byte_pos` is a boundary.
    if (isWordBoundaryByte(text, byte_pos)) return true;
    // Identify the script of the codepoint that ENDS at byte_pos and
    // the script of the codepoint that STARTS at byte_pos.
    const prev_script = scriptOfCodepointEndingAt(text, byte_pos);
    const next_script = scriptOfCodepointStartingAt(text, byte_pos);
    // Script transition: always a boundary (e.g. Latin → Thai, Chinese → Latin).
    if (prev_script != next_script) return true;
    // Same-script run: consult the relevant dict.
    const dict = dictForScript(prev_script) orelse return false;
    // Find the start of the same-script run containing byte_pos.
    const run_start = findScriptRunStart(text, byte_pos, prev_script);
    const run_end = findScriptRunEnd(text, byte_pos, prev_script);
    return isDictWordBoundary(text[run_start..run_end], byte_pos - run_start, dict);
}

/// Script enum for the scriptio-continua languages we handle plus
/// "other" (everything else — Latin, Cyrillic, Devanagari, ASCII
/// digits, punctuation, control codes, etc.).
const Script = enum { other, thai, lao, han, hiragana, katakana };

/// Returns the dict associated with a script, or null when the script
/// has no bundled dict (the `other` case, plus scripts we don't yet
/// segment).
fn dictForScript(s: Script) ?[]const []const u8 {
    return switch (s) {
        .thai => dict_th.words,
        .lao => dict_lo.words,
        .han => dict_zh.words,
        // Japanese: hiragana and katakana share the ja dict. Pure
        // kanji runs use the zh dict (han); the ja dict adds kana
        // particles + hiragana words.
        .hiragana, .katakana => dict_ja.words,
        else => null,
    };
}

/// Decode the codepoint at byte_pos in `text` (starting position) and
/// classify its script.
fn scriptOfCodepointStartingAt(text: []const u8, byte_pos: u32) Script {
    if (byte_pos >= text.len) return .other;
    const b = text[byte_pos];
    if (b < 0x80) return .other; // ASCII never maps to a CJK/SEA script
    const cp_len = std.unicode.utf8ByteSequenceLength(b) catch return .other;
    if (byte_pos + cp_len > text.len) return .other;
    const cp = std.unicode.utf8Decode(text[byte_pos .. byte_pos + cp_len]) catch return .other;
    return scriptOfCodepoint(cp);
}

/// Decode the codepoint whose final byte is at byte_pos-1 (i.e. the
/// codepoint immediately before `byte_pos`) and classify its script.
fn scriptOfCodepointEndingAt(text: []const u8, byte_pos: u32) Script {
    if (byte_pos == 0) return .other;
    const prev = text[byte_pos - 1];
    if (prev < 0x80) return .other;
    // Walk back up to 4 bytes looking for the UTF-8 leading byte.
    var s: u32 = byte_pos;
    var back: u32 = 0;
    while (back < 4 and s > 0) : (back += 1) {
        s -= 1;
        const b = text[s];
        if (b < 0x80) return .other;
        if ((b & 0xC0) != 0x80) {
            const cp_len = std.unicode.utf8ByteSequenceLength(b) catch return .other;
            if (s + cp_len != byte_pos) return .other;
            const cp = std.unicode.utf8Decode(text[s..byte_pos]) catch return .other;
            return scriptOfCodepoint(cp);
        }
    }
    return .other;
}

/// Classify a codepoint into one of the scripts we segment, or `.other`.
/// Block boundaries follow the Unicode 15 BlockProperties tables; we
/// keep this small (5 blocks) and biased toward "the obvious cases."
/// CJK extension blocks are folded into .han.
fn scriptOfCodepoint(cp: u21) Script {
    // Thai (U+0E00..U+0E7F).
    if (cp >= 0x0E00 and cp <= 0x0E7F) return .thai;
    // Lao (U+0E80..U+0EFF).
    if (cp >= 0x0E80 and cp <= 0x0EFF) return .lao;
    // Hiragana (U+3040..U+309F).
    if (cp >= 0x3040 and cp <= 0x309F) return .hiragana;
    // Katakana (U+30A0..U+30FF) + Katakana phonetic extensions (U+31F0..U+31FF).
    if ((cp >= 0x30A0 and cp <= 0x30FF) or (cp >= 0x31F0 and cp <= 0x31FF)) return .katakana;
    // CJK Unified Ideographs (U+4E00..U+9FFF) + Extension A (U+3400..U+4DBF).
    if (cp >= 0x4E00 and cp <= 0x9FFF) return .han;
    if (cp >= 0x3400 and cp <= 0x4DBF) return .han;
    // CJK Compatibility Ideographs (U+F900..U+FAFF).
    if (cp >= 0xF900 and cp <= 0xFAFF) return .han;
    // CJK Symbols and Punctuation (U+3000..U+303F) — keep these in
    // .other so `。` (full stop) and friends act as natural breaks.
    return .other;
}

/// Walk backwards from byte_pos through codepoints that classify into
/// the same script `s`. Returns the byte index of the first byte of
/// the run.
fn findScriptRunStart(text: []const u8, byte_pos: u32, s: Script) u32 {
    var p: u32 = byte_pos;
    while (p > 0) {
        const cp_script = scriptOfCodepointEndingAt(text, p);
        if (cp_script != s) break;
        // Step back one codepoint.
        const new_p = stepBackOneCodepoint(text, p);
        if (new_p == p) break; // safety
        p = new_p;
    }
    return p;
}

/// Walk forward from byte_pos through codepoints in script `s`.
/// Returns the byte index just past the run end.
fn findScriptRunEnd(text: []const u8, byte_pos: u32, s: Script) u32 {
    var p: u32 = byte_pos;
    while (p < text.len) {
        const cp_script = scriptOfCodepointStartingAt(text, p);
        if (cp_script != s) break;
        const cp_len = std.unicode.utf8ByteSequenceLength(text[p]) catch break;
        if (p + cp_len > text.len) break;
        p += cp_len;
    }
    return p;
}

/// Step back one codepoint from `pos`. Returns the new position. On
/// malformed input returns `pos` unchanged (caller should treat as
/// terminating condition).
fn stepBackOneCodepoint(text: []const u8, pos: u32) u32 {
    if (pos == 0) return 0;
    var s: u32 = pos;
    var back: u32 = 0;
    while (back < 4 and s > 0) : (back += 1) {
        s -= 1;
        const b = text[s];
        if (b < 0x80) return s; // ASCII = 1-byte cp, found leading byte
        if ((b & 0xC0) != 0x80) {
            // leading byte
            const cp_len = std.unicode.utf8ByteSequenceLength(b) catch return pos;
            if (s + cp_len == pos) return s;
            return pos; // malformed
        }
    }
    return pos;
}

/// Given a same-script run `run` and a target position `target` inside
/// it (byte index relative to run start), return true if `target` is
/// the END of some segmentation step under longest-prefix-first dict
/// matching. Walks forward from 0, at each cursor doing the longest
/// dict match (or single-codepoint fallback when no match), and yields
/// segmentation points at cursor advances.
///
/// O(run_len * max_dict_word_len * dict_size) worst case; for the
/// tiny shipped dicts (hundreds of entries, average word length 6-12
/// bytes) this is essentially linear in run_len.
fn isDictWordBoundary(run: []const u8, target: u32, dict: []const []const u8) bool {
    if (target == 0) return true; // start-of-run is always a boundary
    if (target >= run.len) return target == run.len;
    var cursor: u32 = 0;
    while (cursor < run.len) {
        if (cursor == target) return true;
        // Find the longest dict entry that matches at `cursor`.
        var best_len: u32 = 0;
        for (dict) |entry| {
            const elen = entry.len;
            if (elen == 0) continue;
            if (cursor + elen > run.len) continue;
            if (elen <= best_len) continue;
            if (std.mem.eql(u8, run[cursor .. cursor + elen], entry)) {
                best_len = @intCast(elen);
            }
        }
        if (best_len > 0) {
            cursor += best_len;
            continue;
        }
        // Fallback: advance by one codepoint.
        if (cursor >= run.len) break;
        const cp_len = std.unicode.utf8ByteSequenceLength(run[cursor]) catch 1;
        const advance: u32 = @intCast(@min(cp_len, run.len - cursor));
        cursor += if (advance == 0) 1 else advance;
    }
    return cursor == target;
}

/// Treat any ASCII whitespace OR Unicode whitespace as a word boundary.
/// `byte_pos` is the start of the candidate token; we inspect the byte
/// immediately before it.
fn isWordBoundaryByte(text: []const u8, byte_pos: u32) bool {
    if (byte_pos == 0 or byte_pos > text.len) return false;
    const prev = text[byte_pos - 1];
    // ASCII fast path.
    if (prev == ' ' or prev == '\t' or prev == '\n' or prev == '\r') return true;
    if (prev < 0x80) return false;
    // Non-ASCII: decode the codepoint that ENDS at byte_pos. We walk
    // back up to 4 bytes looking for a UTF-8 leading byte. If we find
    // one and it decodes to a whitespace codepoint, snap here.
    var s: usize = byte_pos;
    var back: usize = 0;
    while (back < 4 and s > 0) : (back += 1) {
        s -= 1;
        const b = text[s];
        if (b < 0x80) break; // ascii — not a leading byte of a multi-byte cp
        if ((b & 0xC0) != 0x80) {
            // Leading byte.
            const cp_len = std.unicode.utf8ByteSequenceLength(b) catch return false;
            if (s + cp_len != byte_pos) return false;
            const cp = std.unicode.utf8Decode(text[s..byte_pos]) catch return false;
            return unicode_props.isWhitespace(cp);
        }
    }
    return false;
}

/// Context bundle threaded through the sentence segmenter. Keeps the
/// helper signatures from ballooning every time we add a new knob.
const SegmenterCtx = struct {
    abbreviations: []const []const u8,
    aggressive_lowercase: bool,
    /// When true, a single `\n` is NOT enough to fire a sentence
    /// boundary — only `\n\n` (paragraph break) counts. Set
    /// automatically for `Locale.th` / `Locale.lo` since Thai and Lao
    /// prose conventionally line-wraps mid-sentence and only uses
    /// blank lines for true paragraph breaks. Other-script
    /// terminators (CJK `。`, Latin `.!?`, etc.) still fire normally
    /// in mixed-script input.
    require_double_newline: bool = false,
};

/// Curated, fixed, English-only list of words that frequently start a
/// sentence in informal prose. Used only when
/// `aggressive_lowercase` is set: a `.`/`!`/`?` followed by whitespace
/// + a word from this list fires a boundary even though the next
/// letter is lowercase. Match is case-sensitive against the
/// canonical lowercase form so we don't double-fire on already-capital
/// `Then`/`But`/... (that case is already a boundary under the strict
/// rule). The list is INTENTIONALLY small — every entry risks
/// over-splitting (e.g. `The` after `St.` if abbreviation handling
/// elsewhere fails), and adding entries scales the false-positive
/// rate proportionally.
const lowercase_sentence_openers: []const []const u8 = &.{
    "the",     "a",       "an",      "and",     "but",
    "or",      "nor",     "so",      "yet",     "for",
    "then",    "however", "because", "i",       "it",
    "he",      "she",     "they",    "we",      "you",
    "this",    "that",    "these",   "those",
};

/// Snap to a sentence boundary. Walks candidate ends right-to-left,
/// delegating per-position acceptance to `isSentenceBoundary`. See
/// that function's doc-comment for the full ruleset (abbreviations,
/// ellipsis, decimals/URLs/files, quoted dialog, CJK).
fn snapLeftSentence(
    text: []const u8,
    offsets: []const Span,
    start: u32,
    hard_end: u32,
    ctx: SegmenterCtx,
) u32 {
    if (hard_end <= start + 1) return hard_end;
    var i: u32 = hard_end;
    while (i > start + 1) : (i -= 1) {
        const boundary = offsets[i - 1].end;
        if (boundary == 0 or boundary > text.len) continue;
        if (isSentenceBoundary(text, boundary, ctx)) return i;
    }
    return hard_end;
}

/// Returns true if `pos` is a sentence boundary — i.e. the chunk
/// `text[..pos]` would end a sentence. Cases handled:
///
///   * `\n` — always a sentence boundary (line break).
///   * `?` / `!` — boundary if (after skipping any closing quote run)
///     the next non-space byte is a capital letter, or EOT.
///   * `.` — see `isDotBoundary`. Handles single-dot abbreviations,
///     decimals / URLs / files (dot between alphanumerics),
///     ellipsis (2+ consecutive dots), and U.S.A.-style runs.
///   * Closing quote (`"` `'` `”` `’`) — boundary if the prior
///     non-quote byte is `.`/`!`/`?` AND the post-quote position
///     satisfies the next-context rule AND closing the quote at this
///     position brings nested-quote depth to 0 (an inner `'` inside a
///     still-open outer `"` does NOT close the sentence). This catches
///     quoted dialog like `'I left.' She waved.`.
///   * `。` (U+3002), `！` (U+FF01), `？` (U+FF1F) — CJK sentence
///     terminators, unconditional boundary (CJK doesn't use the
///     abbreviation pattern).
///   * `؟` (U+061F Arabic), `։` (U+0589 Armenian), `།` (U+0F0D
///     Tibetan), `።` (U+1362 Ethiopic), `।` `॥` (U+0964 / U+0965
///     Devanagari danda / double-danda), `᠃` (U+1803 Mongolian),
///     `။` (U+104B Burmese / Myanmar), `។` (U+17D4 Khmer) —
///     other-script sentence terminators, same
///     unconditional-boundary treatment as CJK.
fn isSentenceBoundary(
    text: []const u8,
    pos: u32,
    ctx: SegmenterCtx,
) bool {
    if (pos == 0 or pos > text.len) return false;
    const prev = text[pos - 1];

    // ----- ASCII fast paths -----

    if (prev == '\n') {
        // Thai/Lao approximation: only `\n\n` (paragraph break) counts.
        // Single newlines are part of mid-sentence line-wrapping in
        // those scripts (which lack sentence-terminator punctuation).
        if (ctx.require_double_newline) {
            return pos >= 2 and text[pos - 2] == '\n';
        }
        return true;
    }

    if (prev == '?' or prev == '!') {
        // `He said "What?" then left.` — boundary is after the quote.
        // Treat the closing-quote run as part of the terminator and
        // anchor the next-context check past it.
        const after_quotes = skipClosingQuotes(text, pos);
        // Nested-quote suppression: if the terminator itself sits
        // INSIDE an open quote (depth at `pos` is > 0, meaning the
        // `?`/`!` is internal to a still-open quote), don't fire here
        // — wait for the closing quote that actually balances the
        // outer-most opener.
        if (after_quotes == pos and quoteDepthAt(text, pos) > 0) return false;
        return followedByCapitalOrEnd(text, after_quotes, ctx);
    }

    if (prev == '.') return isDotBoundary(text, pos, ctx);

    // ----- Closing-quote terminator (quoted dialog) -----
    //
    // Patterns like `'I left.' She waved.` — boundary lies after the
    // closing quote. We're at pos with text[pos-1] == quote; walk
    // back over the quote run, check the byte before that for a
    // sentence terminator (., !, ?), then apply the standard
    // next-context test from `pos`.
    if (isAsciiClosingQuote(prev)) {
        const before_quotes = skipClosingQuotesBack(text, pos);
        if (before_quotes == 0) return false;
        const inner = text[before_quotes - 1];
        if (inner == '!' or inner == '?') return followedByCapitalOrEnd(text, pos, ctx);
        if (inner == '.') return isDotBoundaryAt(text, before_quotes, pos, ctx);
        return false;
    }

    // ----- Multi-byte UTF-8 terminator -----
    if (prev >= 0x80) {
        // 2-byte terminators (Arabic + Armenian).
        // ؟ U+061F ARABIC QUESTION MARK — D8 9F
        // ։ U+0589 ARMENIAN FULL STOP   — D6 89
        if (pos >= 2) {
            const last2 = text[pos - 2 .. pos];
            if (std.mem.eql(u8, last2, "\xD8\x9F") or
                std.mem.eql(u8, last2, "\xD6\x89"))
            {
                return true;
            }
        }
        if (pos >= 3) {
            const last3 = text[pos - 3 .. pos];
            // CJK sentence terminators: no abbreviation interaction.
            // 。 U+3002 IDEOGRAPHIC FULL STOP         — E3 80 82
            // ！ U+FF01 FULLWIDTH EXCLAMATION MARK    — EF BC 81
            // ？ U+FF1F FULLWIDTH QUESTION MARK       — EF BC 9F
            // ። U+1362 ETHIOPIC FULL STOP             — E1 8D A2
            // ། U+0F0D TIBETAN MARK SHAD              — E0 BC 8D
            // 1.15 additions (same unconditional rule):
            // । U+0964 DEVANAGARI DANDA              — E0 A5 A4
            // ॥ U+0965 DEVANAGARI DOUBLE DANDA       — E0 A5 A5
            // ᠃ U+1803 MONGOLIAN FULL STOP           — E1 A0 83
            // ။ U+104B MYANMAR SIGN SECTION (Burmese) — E1 81 8B
            // ។ U+17D4 KHMER SIGN KHAN               — E1 9F 94
            if (std.mem.eql(u8, last3, "\xE3\x80\x82") or
                std.mem.eql(u8, last3, "\xEF\xBC\x81") or
                std.mem.eql(u8, last3, "\xEF\xBC\x9F") or
                std.mem.eql(u8, last3, "\xE1\x8D\xA2") or
                std.mem.eql(u8, last3, "\xE0\xBC\x8D") or
                std.mem.eql(u8, last3, "\xE0\xA5\xA4") or
                std.mem.eql(u8, last3, "\xE0\xA5\xA5") or
                std.mem.eql(u8, last3, "\xE1\xA0\x83") or
                std.mem.eql(u8, last3, "\xE1\x81\x8B") or
                std.mem.eql(u8, last3, "\xE1\x9F\x94"))
            {
                return true;
            }
            // Closing curly quotes — same quoted-dialog handling as ASCII.
            // ” U+201D RIGHT DOUBLE QUOTATION MARK  — E2 80 9D
            // ’ U+2019 RIGHT SINGLE QUOTATION MARK  — E2 80 99
            if (std.mem.eql(u8, last3, "\xE2\x80\x9D") or
                std.mem.eql(u8, last3, "\xE2\x80\x99"))
            {
                const before_quotes = skipClosingQuotesBack(text, pos);
                if (before_quotes == 0) return false;
                const inner = text[before_quotes - 1];
                if (inner == '!' or inner == '?') return followedByCapitalOrEnd(text, pos, ctx);
                if (inner == '.') return isDotBoundaryAt(text, before_quotes, pos, ctx);
                return false;
            }
        }
    }

    return false;
}

/// Core dot-terminator decision. Factored out so the quoted-dialog
/// branch can reuse it with a different "post-context" anchor: when
/// the dot sits inside `."`, abbreviation/ellipsis lookback is
/// anchored at the dot but the whitespace-and-capital lookforward is
/// anchored AFTER the closing quote.
fn isDotBoundary(
    text: []const u8,
    pos: u32,
    ctx: SegmenterCtx,
) bool {
    return isDotBoundaryAt(text, pos, pos, ctx);
}

fn isDotBoundaryAt(
    text: []const u8,
    /// Position immediately after the dot itself (i.e. `text[dot_after_pos - 1] == '.'`).
    dot_after_pos: u32,
    /// Position to start scanning for whitespace + capital. Same as
    /// dot_after_pos unless closing quotes intervene.
    next_anchor: u32,
    ctx: SegmenterCtx,
) bool {
    std.debug.assert(dot_after_pos > 0);
    std.debug.assert(text[dot_after_pos - 1] == '.');

    const dot_idx = dot_after_pos - 1;
    const in_run = dot_idx >= 1 and text[dot_idx - 1] == '.';
    const more_dots_after = dot_after_pos < text.len and text[dot_after_pos] == '.';

    // --- Ellipsis: 2+ consecutive dots. ---
    //
    // At an internal dot of a multi-dot run, never a boundary — the
    // segmenter will re-test at the closing dot of the run.
    if (more_dots_after) return false;

    if (in_run) {
        // Closing dot of a 2+ dot run = ellipsis. Boundary only if
        // next non-space is capital or EOT. `and then... she left` is
        // NOT a boundary (lowercase `s`); `He paused... Then continued.`
        // IS a boundary (capital `T`).
        return next_anchor == text.len or followedByCapitalOrEnd(text, next_anchor, ctx);
    }

    // --- Single dot. ---

    // Decimal / URL / file extension: a single `.` flanked by
    // alphanumeric on BOTH sides is never a sentence boundary. Catches
    // `3.14`, `192.168.1.1`, `notes.txt`, `example.com/path`. We check
    // this before the "next must be whitespace" guard so we can
    // short-circuit even when the dot is internal to a word.
    if (dot_idx >= 1 and dot_after_pos < text.len) {
        const before = text[dot_idx - 1];
        const after = text[dot_after_pos];
        if (isAsciiAlnum(before) and isAsciiAlnum(after)) return false;
    }

    if (dot_after_pos < text.len) {
        const next = text[dot_after_pos];
        // Inside-word dot ("Dr.Smith" with no space): require
        // whitespace, line end, or a closing quote after the dot.
        if (!isAsciiSpaceOrLineEnd(next) and !isAsciiClosingQuote(next)) {
            // Closing curly quote also OK (multi-byte; cheap probe).
            const is_curly_quote = next == 0xE2 and
                dot_after_pos + 3 <= text.len and
                (std.mem.eql(u8, text[dot_after_pos .. dot_after_pos + 3], "\xE2\x80\x9D") or
                    std.mem.eql(u8, text[dot_after_pos .. dot_after_pos + 3], "\xE2\x80\x99"));
            if (!is_curly_quote) return false;
        }
    }

    // Period-final ALL-CAPS acronym: a dot immediately preceded by an
    // `([A-Z]\.){2,}` shape (think `U.S.` or `U.S.A.`) is treated as
    // an abbreviation regardless of the locale list. Walks back only
    // through the local word — O(word-len), not O(text). Must run
    // BEFORE the explicit abbreviation list so locale-agnostic
    // acronyms always win.
    //
    // CAVEAT: only suppress when the next non-space byte is NOT a
    // capital letter. The capital-after-acronym case (`U.S. It was
    // great.`) is genuinely ambiguous between "acronym ends a sentence"
    // and "acronym used mid-sentence followed by a proper noun"; we
    // defer to the standard `capital-follows` rule there (treat as
    // sentence boundary). Without this caveat, every `U.S. <Capital>`
    // would be glued into one sentence — wrong for the common case
    // where the acronym IS the sentence terminator.
    if (next_anchor < text.len) {
        if (!nextNonSpaceIsCapital(text, next_anchor) and
            isAcronymBefore(text, dot_idx))
        {
            return false;
        }
    }

    // Dot followed by whitespace, closing quote, or EOT. Apply
    // abbreviation lookback.
    if (isAbbreviationBefore(text, dot_idx, ctx.abbreviations)) return false;

    // Nested-quote suppression: a dot that sits INSIDE an unclosed
    // quote (depth > 0 at the dot's position) is NOT a sentence
    // boundary — wait for the outermost closing quote. Only suppress
    // when the dot is NOT immediately followed by a closing quote
    // (in which case the quoted-dialog branch will handle firing at
    // the post-quote position).
    if (next_anchor == dot_after_pos and quoteDepthAt(text, dot_after_pos) > 0) return false;

    return next_anchor == text.len or followedByCapitalOrEnd(text, next_anchor, ctx);
}

inline fn isAsciiAlnum(b: u8) bool {
    return (b >= '0' and b <= '9') or
        (b >= 'A' and b <= 'Z') or
        (b >= 'a' and b <= 'z');
}

/// ASCII closing quotes that may follow a sentence terminator inside
/// dialog. Curly-quote variants (U+201D, U+2019) are handled
/// alongside in the multi-byte branch of `isSentenceBoundary` and in
/// `skipClosingQuotes` / `skipClosingQuotesBack`.
inline fn isAsciiClosingQuote(b: u8) bool {
    return b == '"' or b == '\'';
}

/// Forward-skip a run of closing quotes (ASCII `"` `'` plus U+201D
/// U+2019). Returns the position just past the run. Lets us walk
/// `He said "Stop!" then left.` from the `!` to the byte AFTER the
/// closing `"` so we can apply the next-context check at the right
/// anchor.
fn skipClosingQuotes(text: []const u8, pos: u32) u32 {
    var p: u32 = pos;
    while (p < text.len) {
        const b = text[p];
        if (isAsciiClosingQuote(b)) {
            p += 1;
            continue;
        }
        if (b == 0xE2 and p + 3 <= text.len) {
            const seq = text[p .. p + 3];
            if (std.mem.eql(u8, seq, "\xE2\x80\x9D") or
                std.mem.eql(u8, seq, "\xE2\x80\x99"))
            {
                p += 3;
                continue;
            }
        }
        break;
    }
    return p;
}

/// Backward-skip a run of closing quotes ending at `pos` (exclusive).
/// Returns the position of the first byte of the run.
fn skipClosingQuotesBack(text: []const u8, pos: u32) u32 {
    var p: u32 = pos;
    while (p > 0) {
        const b = text[p - 1];
        if (isAsciiClosingQuote(b)) {
            p -= 1;
            continue;
        }
        if (p >= 3) {
            const seq = text[p - 3 .. p];
            if (std.mem.eql(u8, seq, "\xE2\x80\x9D") or
                std.mem.eql(u8, seq, "\xE2\x80\x99"))
            {
                p -= 3;
                continue;
            }
        }
        break;
    }
    return p;
}

/// True if `text[pos..]` starts with optional ASCII whitespace then
/// an ASCII uppercase letter (A..Z) or a UTF-8 leading byte that
/// decodes to an uppercase / titlecase letter — or if there is
/// nothing left (end of text). When `ctx.aggressive_lowercase` is
/// true, also returns true if the post-whitespace word is a known
/// sentence-opener (case-sensitive lowercase match against
/// `lowercase_sentence_openers`).
fn followedByCapitalOrEnd(text: []const u8, pos: u32, ctx: SegmenterCtx) bool {
    var p: u32 = pos;
    while (p < text.len and isAsciiSpaceOrLineEnd(text[p])) p += 1;
    if (p >= text.len) return true;
    const b = text[p];
    if (b >= 'A' and b <= 'Z') return true;
    if (b >= 'a' and b <= 'z') {
        if (!ctx.aggressive_lowercase) return false;
        // Slice off the next ASCII-letter run and compare against the
        // opener list. Conservative: we deliberately don't include
        // digits or apostrophes so `it's`/`don't` don't match.
        var end: u32 = p;
        while (end < text.len) {
            const c = text[end];
            if ((c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z')) {
                end += 1;
            } else break;
        }
        const word = text[p..end];
        for (lowercase_sentence_openers) |opener| {
            if (std.mem.eql(u8, opener, word)) return true;
        }
        return false;
    }
    if (b < 0x80) return false;
    // Multi-byte UTF-8: decode and check Lu/Lt.
    const cp_len = std.unicode.utf8ByteSequenceLength(b) catch return false;
    if (p + cp_len > text.len) return false;
    const cp = std.unicode.utf8Decode(text[p .. p + cp_len]) catch return false;
    return unicode_props.isLu(cp) or unicode_props.isLt(cp);
}

/// Look back from `dot_pos` (the index of the `.`) to find the
/// alphanumeric/dotted "word" immediately before it. Compare against
/// the abbreviation list. Match is case-sensitive — the canonical
/// abbreviations are capitalized as written ("Dr", "Mr"), and a
/// lowercase "dr." in the middle of a sentence almost certainly IS
/// a sentence end anyway.
///
/// The walk-back accepts ASCII letters / digits / internal dots
/// (Latin fast path) AND multi-byte UTF-8 sequences that decode to
/// any letter category (Lu/Ll/Lt/Lm/Lo) or a combining mark
/// (Mn/Mc/Me). The mark inclusion is critical for Indic scripts:
/// Hindi `डॉ` is `ड` (Lo) + `ॉ` (Mc, vowel sign), and treating only
/// letters would split the cluster mid-word and miss the abbrev.
/// Bytewise comparison against the abbreviation list still works
/// because UTF-8 is a self-synchronizing prefix code — distinct
/// codepoints have distinct byte sequences.
fn isAbbreviationBefore(
    text: []const u8,
    dot_pos: u32,
    abbreviations: []const []const u8,
) bool {
    if (dot_pos == 0) return false;
    // Walk back over letters / digits / internal dots / non-ASCII
    // letter or mark codepoints. The non-ASCII branch decodes one
    // codepoint at a time by walking up to 4 bytes looking for a
    // leading byte, then classifying with unicode_props.
    var s: u32 = dot_pos;
    walk: while (s > 0) {
        const c = text[s - 1];
        if ((c >= 'a' and c <= 'z') or
            (c >= 'A' and c <= 'Z') or
            (c >= '0' and c <= '9') or
            c == '.')
        {
            s -= 1;
            continue :walk;
        }
        if (c < 0x80) break; // ASCII non-letter — stop.
        // Multi-byte UTF-8: find the leading byte of the codepoint
        // ending at `s` (`s` exclusive of the codepoint).
        var lead: u32 = s;
        var back: u32 = 0;
        while (back < 4 and lead > 0) : (back += 1) {
            lead -= 1;
            const b = text[lead];
            if (b < 0x80) break :walk; // malformed; stop walk
            if ((b & 0xC0) != 0x80) {
                const cp_len = std.unicode.utf8ByteSequenceLength(b) catch break :walk;
                if (lead + cp_len != s) break :walk;
                const cp = std.unicode.utf8Decode(text[lead..s]) catch break :walk;
                if (unicode_props.isLetter(cp) or unicode_props.isMark(cp)) {
                    s = lead;
                    continue :walk;
                }
                break :walk;
            }
        }
        break :walk;
    }
    if (s >= dot_pos) return false;
    // Strip a leading dot if any (so "...U.S" still looks like "U.S").
    var word_start: u32 = s;
    while (word_start < dot_pos and text[word_start] == '.') word_start += 1;
    if (word_start >= dot_pos) return false;
    const word = text[word_start..dot_pos];
    for (abbreviations) |abbr| {
        if (std.mem.eql(u8, abbr, word)) return true;
    }
    return false;
}

inline fn isAsciiSpaceOrLineEnd(b: u8) bool {
    return b == ' ' or b == '\t' or b == '\n' or b == '\r';
}

/// Period-final ALL-CAPS acronym detector. Returns true if the word
/// immediately preceding `dot_pos` (with `text[dot_pos] == '.'`) has
/// the shape `([A-Z]\.){1,}[A-Z]` — i.e. one or more capital-then-dot
/// pairs followed by a final capital that's about to be terminated by
/// `dot_pos` itself. Combined with the surrounding `.`, that's the
/// `([A-Z]\.){2,}` pattern from the spec. Examples that match:
///
///   * `U.S.` — at the trailing `.`, looks back through `S`, sees the
///     `.` before it, sees `U`, sees the boundary. Shape: `U.S` =>
///     `([A-Z]\.){1}[A-Z]` => YES.
///   * `U.S.A.` — same, with one more `A.` segment.
///
/// Examples that don't match:
///
///   * `A.` (single letter) — only one capital, no internal dot.
///   * `e.g.` — leading lowercase breaks the pattern.
///   * `U.s.a.` — mixed case breaks the pattern.
///
/// Strictly O(local-word-len): walks back only through the connected
/// alpha-dot run, stops at the first non-matching byte.
fn isAcronymBefore(text: []const u8, dot_pos: u32) bool {
    if (dot_pos == 0) return false;
    // The byte immediately before the dot must be an ASCII capital.
    const last = text[dot_pos - 1];
    if (!(last >= 'A' and last <= 'Z')) return false;
    // Walk back through (`.` + capital) pairs.
    var caps_with_dot: u32 = 0;
    var s: u32 = dot_pos - 1; // points at the trailing capital
    while (s >= 2) {
        const dot = text[s - 1];
        const cap = text[s - 2];
        if (dot != '.') break;
        if (!(cap >= 'A' and cap <= 'Z')) break;
        caps_with_dot += 1;
        s -= 2;
        if (s == 0) break;
    }
    if (caps_with_dot == 0) return false;
    // The byte before the start of the run (if any) must NOT be an
    // alpha or digit — otherwise it's a fragment of a longer word like
    // `xU.S` which isn't an acronym.
    if (s > 0) {
        const c = text[s - 1];
        if (isAsciiAlnum(c)) return false;
    }
    return true;
}

/// Running quote depth at byte position `pos`, where depth increases
/// on an opening quote and decreases on a closing quote. ASCII `"` and
/// `'` are ambiguous (open vs close); we disambiguate with the classic
/// smart-quote heuristic: a quote is an OPEN when it sits between a
/// non-alphanumeric byte on the left (or start-of-paragraph) and an
/// alphanumeric byte on the right; it's a CLOSE when between an
/// alphanumeric byte on the left and a non-alphanumeric on the right
/// (or end). When neither pattern holds we fall back to a parity
/// toggle within the paragraph. Apostrophes inside contractions
/// (`it's`, `don't` — alpha on BOTH sides) are skipped entirely so
/// they don't pollute the quote count. Unicode curly quotes are
/// unambiguous: `"` `'` open (U+201C U+2018), `"` `'` close (U+201D
/// U+2019).
///
/// Depth resets at every `\n\n` paragraph break, bounding state to a
/// single paragraph. Negative depths (which arise when a paragraph
/// opens mid-quote — e.g. the dialog snippet `And then..." She left.`)
/// are clamped to 0 so callers see "we're not currently inside a quote
/// that we know about."
///
/// O(distance-from-last-paragraph-break). For typical prose with
/// paragraphs every few hundred bytes this is fine. Adversarial
/// typographers' nightmares (mixed straight + curly quotes,
/// apostrophe-heavy contractions adjacent to dialog quotes) will
/// still confuse the heuristic — layer ICU's `BreakIterator` for that.
fn quoteDepthAt(text: []const u8, pos: u32) i32 {
    if (pos == 0) return 0;
    // Find the start of the current paragraph (byte after the most
    // recent `\n\n`, or 0).
    var para_start: u32 = 0;
    if (pos >= 2) {
        var k: u32 = pos - 1;
        while (k >= 1) : (k -= 1) {
            if (text[k] == '\n' and text[k - 1] == '\n') {
                para_start = k + 1;
                break;
            }
            if (k == 1) break;
        }
    }
    var dquote_open: bool = false; // parity fallback for ASCII "
    var squote_open: bool = false;
    var depth: i32 = 0;
    var i: u32 = para_start;
    while (i < pos) {
        const b = text[i];
        if (b == '"' or b == '\'') {
            const left_alpha = i > para_start and isAsciiAlnum(text[i - 1]);
            const right_alpha = i + 1 < text.len and isAsciiAlnum(text[i + 1]);
            // Apostrophe-in-contraction: alpha both sides. Skip
            // entirely so it doesn't pollute the count.
            if (b == '\'' and left_alpha and right_alpha) {
                i += 1;
                continue;
            }
            // Smart-quote disambiguation.
            const kind: enum { open, close, ambiguous } = blk: {
                if (!left_alpha and right_alpha) break :blk .open;
                if (left_alpha and !right_alpha) break :blk .close;
                break :blk .ambiguous;
            };
            switch (kind) {
                .open => depth += 1,
                .close => depth -= 1,
                .ambiguous => {
                    // Fall back to parity toggle within the paragraph.
                    if (b == '"') {
                        if (dquote_open) depth -= 1 else depth += 1;
                        dquote_open = !dquote_open;
                    } else {
                        if (squote_open) depth -= 1 else depth += 1;
                        squote_open = !squote_open;
                    }
                },
            }
            // Track parity even when smart-quote classified the role
            // — keeps the fallback consistent across runs.
            if (b == '"' and kind != .ambiguous) dquote_open = (kind == .open);
            if (b == '\'' and kind != .ambiguous) squote_open = (kind == .open);
            i += 1;
            continue;
        }
        if (b == 0xE2 and i + 3 <= pos) {
            const seq = text[i .. i + 3];
            // Opens
            if (std.mem.eql(u8, seq, "\xE2\x80\x9C") or // " U+201C
                std.mem.eql(u8, seq, "\xE2\x80\x98")) // ' U+2018
            {
                depth += 1;
                i += 3;
                continue;
            }
            // Closes
            if (std.mem.eql(u8, seq, "\xE2\x80\x9D") or // " U+201D
                std.mem.eql(u8, seq, "\xE2\x80\x99")) // ' U+2019
            {
                depth -= 1;
                i += 3;
                continue;
            }
        }
        i += 1;
    }
    if (depth < 0) depth = 0; // unbalanced; treat as outside any quote
    return depth;
}

inline fn isAsciiAlpha(b: u8) bool {
    return (b >= 'A' and b <= 'Z') or (b >= 'a' and b <= 'z');
}

/// Forward-scan from `pos` past ASCII whitespace and return true if
/// the next non-space byte is an ASCII capital (`A`..`Z`) or a UTF-8
/// leading byte that decodes to an uppercase / titlecase codepoint.
/// Distinct from `followedByCapitalOrEnd`: this returns FALSE on
/// end-of-text rather than true. Used by the acronym detector to gate
/// suppression strictly on lowercase / non-letter follows.
fn nextNonSpaceIsCapital(text: []const u8, pos: u32) bool {
    var p: u32 = pos;
    while (p < text.len and isAsciiSpaceOrLineEnd(text[p])) p += 1;
    if (p >= text.len) return false;
    const b = text[p];
    if (b >= 'A' and b <= 'Z') return true;
    if (b < 0x80) return false;
    const cp_len = std.unicode.utf8ByteSequenceLength(b) catch return false;
    if (p + cp_len > text.len) return false;
    const cp = std.unicode.utf8Decode(text[p .. p + cp_len]) catch return false;
    return unicode_props.isLu(cp) or unicode_props.isLt(cp);
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

const testing = std.testing;
const Vocab = ztok.Vocab;

fn bytePipeline(v: *const Vocab) Pipeline {
    return .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = v,
    };
}

test "chunk: basic non-overlapping byte_id" {
    var v = Vocab.empty(testing.allocator);
    defer v.deinit();
    const pipe = bytePipeline(&v);

    var res = try chunkText(testing.allocator, pipe, "abcdefghij", .{
        .max_tokens = 4,
        .overlap_tokens = 0,
    });
    defer res.deinit();

    try testing.expectEqual(@as(usize, 3), res.chunks.len);
    try testing.expectEqual(@as(usize, 4), res.chunks[0].ids.len);
    try testing.expectEqual(@as(usize, 4), res.chunks[1].ids.len);
    try testing.expectEqual(@as(usize, 2), res.chunks[2].ids.len);
    try testing.expectEqual(@as(u32, 0), res.chunks[0].byte_start);
    try testing.expectEqual(@as(u32, 4), res.chunks[0].byte_end);
    try testing.expectEqual(@as(u32, 4), res.chunks[1].byte_start);
    try testing.expectEqual(@as(u32, 8), res.chunks[1].byte_end);
    try testing.expectEqual(@as(u32, 8), res.chunks[2].byte_start);
    try testing.expectEqual(@as(u32, 10), res.chunks[2].byte_end);
}

test "chunk: overlapping stride correctness" {
    var v = Vocab.empty(testing.allocator);
    defer v.deinit();
    const pipe = bytePipeline(&v);

    // 10 tokens, max=4, overlap=2 -> stride=2.
    // Chunks: [0..4], [2..6], [4..8], [6..10].
    var res = try chunkText(testing.allocator, pipe, "abcdefghij", .{
        .max_tokens = 4,
        .overlap_tokens = 2,
    });
    defer res.deinit();

    try testing.expectEqual(@as(usize, 4), res.chunks.len);
    try testing.expectEqual(@as(u32, 0), res.chunks[0].token_start);
    try testing.expectEqual(@as(u32, 4), res.chunks[0].token_end);
    try testing.expectEqual(@as(u32, 2), res.chunks[1].token_start);
    try testing.expectEqual(@as(u32, 6), res.chunks[1].token_end);
    try testing.expectEqual(@as(u32, 4), res.chunks[2].token_start);
    try testing.expectEqual(@as(u32, 8), res.chunks[2].token_end);
    try testing.expectEqual(@as(u32, 6), res.chunks[3].token_start);
    try testing.expectEqual(@as(u32, 10), res.chunks[3].token_end);

    // Verify overlap: chunk[i]'s last 2 ids match chunk[i+1]'s first 2.
    var i: usize = 0;
    while (i + 1 < res.chunks.len) : (i += 1) {
        const a = res.chunks[i].ids;
        const b = res.chunks[i + 1].ids;
        try testing.expectEqualSlices(TokenId, a[a.len - 2 ..], b[0..2]);
    }
}

test "chunk: word boundary snaps to space" {
    var v = Vocab.empty(testing.allocator);
    defer v.deinit();
    const pipe = bytePipeline(&v);

    // "foo bar baz" -> 11 byte tokens.
    // max=6, boundary=word: hard end at token 6 (byte_pos=6, char 'a').
    // Snap left: token 4 has byte_pos=4, prev byte=' ' -> snap there.
    var res = try chunkText(testing.allocator, pipe, "foo bar baz", .{
        .max_tokens = 6,
        .overlap_tokens = 0,
        .boundary = .word,
    });
    defer res.deinit();

    try testing.expect(res.chunks.len >= 2);
    // First chunk should end on a space boundary (byte_end == 4 or 8).
    const first_end = res.chunks[0].byte_end;
    try testing.expect(first_end == 4 or first_end == 8);
}

test "chunk: sentence boundary" {
    var v = Vocab.empty(testing.allocator);
    defer v.deinit();
    const pipe = bytePipeline(&v);

    // "Hi. There. World."
    // max=10, boundary=sentence: should snap at '.' boundaries.
    var res = try chunkText(testing.allocator, pipe, "Hi. There. World.", .{
        .max_tokens = 10,
        .overlap_tokens = 0,
        .boundary = .sentence,
    });
    defer res.deinit();

    try testing.expect(res.chunks.len >= 1);
    // The byte preceding the first chunk's end should be '.', '!', '?',
    // or '\n'.
    for (res.chunks) |c| {
        if (c.byte_end == 17) continue; // last chunk reaches input end
        const prev = "Hi. There. World."[c.byte_end - 1];
        try testing.expect(prev == '.' or prev == '!' or prev == '?' or prev == '\n');
    }
}

test "chunk: paragraph boundary" {
    var v = Vocab.empty(testing.allocator);
    defer v.deinit();
    const pipe = bytePipeline(&v);

    // "para1\n\npara2\n\npara3"
    const text = "para1\n\npara2\n\npara3";
    var res = try chunkText(testing.allocator, pipe, text, .{
        .max_tokens = 10,
        .overlap_tokens = 0,
        .boundary = .paragraph,
    });
    defer res.deinit();

    try testing.expect(res.chunks.len >= 1);
    // Any non-final chunk's byte_end should be preceded by "\n\n".
    for (res.chunks) |c| {
        if (c.byte_end == text.len) continue;
        try testing.expectEqual(@as(u8, '\n'), text[c.byte_end - 1]);
        try testing.expectEqual(@as(u8, '\n'), text[c.byte_end - 2]);
    }
}

test "chunk: empty input -> zero chunks" {
    var v = Vocab.empty(testing.allocator);
    defer v.deinit();
    const pipe = bytePipeline(&v);

    var res = try chunkText(testing.allocator, pipe, "", .{
        .max_tokens = 16,
    });
    defer res.deinit();
    try testing.expectEqual(@as(usize, 0), res.chunks.len);
}

test "chunk: input smaller than max_tokens -> one chunk" {
    var v = Vocab.empty(testing.allocator);
    defer v.deinit();
    const pipe = bytePipeline(&v);

    var res = try chunkText(testing.allocator, pipe, "abc", .{
        .max_tokens = 32,
    });
    defer res.deinit();
    try testing.expectEqual(@as(usize, 1), res.chunks.len);
    try testing.expectEqual(@as(usize, 3), res.chunks[0].ids.len);
    try testing.expectEqual(@as(u32, 0), res.chunks[0].byte_start);
    try testing.expectEqual(@as(u32, 3), res.chunks[0].byte_end);
}

test "chunk: overlap >= max_tokens is an error" {
    var v = Vocab.empty(testing.allocator);
    defer v.deinit();
    const pipe = bytePipeline(&v);

    try testing.expectError(error.OverlapTooLarge, chunkText(
        testing.allocator,
        pipe,
        "abcdef",
        .{ .max_tokens = 4, .overlap_tokens = 4 },
    ));
    try testing.expectError(error.OverlapTooLarge, chunkText(
        testing.allocator,
        pipe,
        "abcdef",
        .{ .max_tokens = 4, .overlap_tokens = 5 },
    ));
    try testing.expectError(error.InvalidMaxTokens, chunkText(
        testing.allocator,
        pipe,
        "abcdef",
        .{ .max_tokens = 0 },
    ));
}

test "chunk: round-trip non-overlapping reconstructs input" {
    var v = Vocab.empty(testing.allocator);
    defer v.deinit();
    const pipe = bytePipeline(&v);

    const text = "the quick brown fox jumps over the lazy dog";
    var res = try chunkText(testing.allocator, pipe, text, .{
        .max_tokens = 7,
        .overlap_tokens = 0,
    });
    defer res.deinit();

    // Concatenating non-overlapping byte ranges should give back input.
    var rebuilt: std.ArrayList(u8) = .empty;
    defer rebuilt.deinit(testing.allocator);
    var cursor: u32 = 0;
    for (res.chunks) |c| {
        try testing.expectEqual(cursor, c.byte_start);
        try rebuilt.appendSlice(testing.allocator, text[c.byte_start..c.byte_end]);
        cursor = c.byte_end;
    }
    try testing.expectEqual(@as(u32, @intCast(text.len)), cursor);
    try testing.expectEqualStrings(text, rebuilt.items);
}

test "chunk: jsonl-shaped output round-trips" {
    var v = Vocab.empty(testing.allocator);
    defer v.deinit();
    const pipe = bytePipeline(&v);

    const text = "the quick brown fox";
    var res = try chunkText(testing.allocator, pipe, text, .{
        .max_tokens = 5,
        .overlap_tokens = 1,
    });
    defer res.deinit();

    // Render JSONL the same way the CLI does, then parse with std.json
    // and assert (a) parseability, (b) byte-range round-trip,
    // (c) ids match the chunk's ids slice.
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(testing.allocator);
    const ta = testing.allocator;

    for (res.chunks, 0..) |c, idx| {
        try buf.print(
            ta,
            "{{\"chunk\":{d},\"byte_start\":{d},\"byte_end\":{d},\"token_start\":{d},\"token_end\":{d},\"ids\":[",
            .{ idx, c.byte_start, c.byte_end, c.token_start, c.token_end },
        );
        for (c.ids, 0..) |id, j| {
            if (j > 0) try buf.appendSlice(ta, ",");
            try buf.print(ta, "{d}", .{id});
        }
        try buf.appendSlice(ta, "]}\n");
    }

    // Parse line-by-line.
    var line_iter = std.mem.splitScalar(u8, buf.items, '\n');
    var line_count: usize = 0;
    while (line_iter.next()) |line| {
        if (line.len == 0) continue;
        var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, line, .{});
        defer parsed.deinit();
        const obj = parsed.value.object;

        const c = res.chunks[line_count];
        try testing.expectEqual(@as(i64, c.byte_start), obj.get("byte_start").?.integer);
        try testing.expectEqual(@as(i64, c.byte_end), obj.get("byte_end").?.integer);
        try testing.expectEqual(@as(i64, c.token_start), obj.get("token_start").?.integer);
        try testing.expectEqual(@as(i64, c.token_end), obj.get("token_end").?.integer);
        const ids_arr = obj.get("ids").?.array;
        try testing.expectEqual(c.ids.len, ids_arr.items.len);
        for (c.ids, ids_arr.items) |expected, got| {
            try testing.expectEqual(@as(i64, expected), got.integer);
        }
        line_count += 1;
    }
    try testing.expectEqual(res.chunks.len, line_count);
}

test "chunk: chunks never exceed max_tokens" {
    var v = Vocab.empty(testing.allocator);
    defer v.deinit();
    const pipe = bytePipeline(&v);

    const text = "abcdefghijklmnopqrstuvwxyz0123456789";
    inline for (.{ Boundary.token, Boundary.word, Boundary.sentence, Boundary.paragraph }) |b| {
        var res = try chunkText(testing.allocator, pipe, text, .{
            .max_tokens = 5,
            .overlap_tokens = 1,
            .boundary = b,
        });
        defer res.deinit();
        for (res.chunks) |c| try testing.expect(c.ids.len <= 5);
    }
}

test "chunk: word boundary snaps correctly under nfc normalizer" {
    // Regression test for 1.10: normalizer origin maps make
    // encodeWithOffsets return original-input spans, so word-boundary
    // snapping (which inspects bytes in the original input via
    // offsets) lands on the right character even when the corpus
    // contains multi-byte composing characters that the normalizer
    // would have moved around in the post-norm buffer.
    var v = Vocab.empty(testing.allocator);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .nfc,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };

    // "café world" — café has 'é' precomposed (U+00E9, 0xC3 0xA9).
    // 4 chars "café" -> 5 bytes ('c','a','f',0xC3,0xA9), then ' ',
    // then 5 bytes "world" = 11 bytes total = 11 byte_id tokens.
    const text = "caf\xC3\xA9 world";
    try testing.expectEqual(@as(usize, 11), text.len);

    var res = try chunkText(testing.allocator, pipe, text, .{
        .max_tokens = 7,
        .overlap_tokens = 0,
        .boundary = .word,
    });
    defer res.deinit();

    // We expect a snap that lands on a word boundary. The snap-left
    // algorithm finds the rightmost candidate token whose preceding
    // byte is whitespace. With max_tokens=7 over the original 11
    // bytes "caf\xC3\xA9 world", the rightmost such position inside
    // the window is byte 6 (the byte before is ' '). The first
    // chunk's byte_end must be either 5 (just before the space) or 6
    // (just after) — both indicate a successful snap. Without origin
    // translation under .nfc this would fail because offsets would
    // index a different buffer than `text`.
    try testing.expect(res.chunks.len >= 2);
    try testing.expectEqual(@as(u32, 0), res.chunks[0].byte_start);
    const first_end = res.chunks[0].byte_end;
    try testing.expect(first_end <= text.len);
    try testing.expect(first_end == 5 or first_end == 6);
    // Whichever it is, the byte immediately before or at it must be a
    // space — confirming the snap inspected the ORIGINAL bytes.
    const has_space_adjacent =
        (first_end > 0 and text[first_end - 1] == ' ') or
        (first_end < text.len and text[first_end] == ' ');
    try testing.expect(has_space_adjacent);

    // The concatenation of all chunk byte ranges should round-trip the
    // original input (since overlap = 0).
    var rebuilt: std.ArrayList(u8) = .empty;
    defer rebuilt.deinit(testing.allocator);
    var cursor: u32 = 0;
    for (res.chunks) |c| {
        try testing.expectEqual(cursor, c.byte_start);
        try rebuilt.appendSlice(testing.allocator, text[c.byte_start..c.byte_end]);
        cursor = c.byte_end;
    }
    try testing.expectEqual(@as(u32, @intCast(text.len)), cursor);
    try testing.expectEqualStrings(text, rebuilt.items);
}

// ---------------------------------------------------------------------
// .codepoint snapping (1.12)
// ---------------------------------------------------------------------

test "chunk: .codepoint snaps back from mid-codepoint cut" {
    // byte_id: every input byte is one token. The 4-byte emoji
    // U+1F600 ("\xF0\x9F\x98\x80") therefore takes 4 tokens. With
    // max_tokens=4 on text "ab" + emoji + "cd" (8 bytes / 8 tokens),
    // a pure-token cut at hard_end=4 puts the chunk boundary 2 bytes
    // into the emoji — the next byte (offsets[3].end == 4) is
    // 0x98, a UTF-8 continuation byte. `.codepoint` must roll back
    // to token 2 where the boundary lands on the leading byte 0xF0.
    var v = Vocab.empty(testing.allocator);
    defer v.deinit();
    const pipe = bytePipeline(&v);

    const text = "ab\xF0\x9F\x98\x80cd";
    try testing.expectEqual(@as(usize, 8), text.len);

    var res = try chunkText(testing.allocator, pipe, text, .{
        .max_tokens = 4,
        .overlap_tokens = 0,
        .boundary = .codepoint,
    });
    defer res.deinit();

    // First chunk must end on a codepoint boundary — either byte 2
    // (before the emoji starts) or byte 6 (after the emoji ends).
    // Cutting at byte 4 (inside the emoji) is a violation.
    try testing.expect(res.chunks.len >= 1);
    const first_end = res.chunks[0].byte_end;
    try testing.expect(first_end == 2 or first_end == 6);
    // Confirm it's actually on a UTF-8 leading byte (or EOT).
    if (first_end < text.len) {
        const b = text[first_end];
        try testing.expect((b & 0xC0) != 0x80);
    }

    // Cross-check that `.token` would have cut inside the emoji.
    var ref = try chunkText(testing.allocator, pipe, text, .{
        .max_tokens = 4,
        .overlap_tokens = 0,
        .boundary = .token,
    });
    defer ref.deinit();
    try testing.expectEqual(@as(u32, 4), ref.chunks[0].byte_end);
    // text[4] is 0x98, a continuation byte: proves .token splits cp.
    try testing.expectEqual(@as(u8, 0x98), text[4]);
}

test "chunk: .codepoint on ASCII corpus matches .token (regression)" {
    var v = Vocab.empty(testing.allocator);
    defer v.deinit();
    const pipe = bytePipeline(&v);

    const text = "the quick brown fox jumps over the lazy dog";
    var cp = try chunkText(testing.allocator, pipe, text, .{
        .max_tokens = 7,
        .overlap_tokens = 2,
        .boundary = .codepoint,
    });
    defer cp.deinit();
    var tk = try chunkText(testing.allocator, pipe, text, .{
        .max_tokens = 7,
        .overlap_tokens = 2,
        .boundary = .token,
    });
    defer tk.deinit();

    try testing.expectEqual(tk.chunks.len, cp.chunks.len);
    for (tk.chunks, cp.chunks) |a, b| {
        try testing.expectEqual(a.byte_start, b.byte_start);
        try testing.expectEqual(a.byte_end, b.byte_end);
        try testing.expectEqual(a.token_start, b.token_start);
        try testing.expectEqual(a.token_end, b.token_end);
    }
}

// ---------------------------------------------------------------------
// Smarter sentence segmenter (1.12)
// ---------------------------------------------------------------------

/// Count the sentence boundaries the segmenter would emit if asked to
/// split `text` into as many pieces as possible. The initial `n=1`
/// turns a boundary count into a sentence count for non-empty text:
/// the trailing region after the last boundary is the final sentence.
fn countSentences(text: []const u8, abbrevs: []const []const u8) usize {
    return countSentencesCtx(text, .{ .abbreviations = abbrevs, .aggressive_lowercase = false });
}

fn countSentencesCtx(text: []const u8, ctx: SegmenterCtx) usize {
    if (text.len == 0) return 0;
    var n: usize = 1;
    var pos: u32 = 1;
    while (pos < text.len) : (pos += 1) {
        if (isSentenceBoundary(text, pos, ctx)) {
            n += 1;
            // Skip forward past the whitespace run so the next
            // sentence starts at the first non-space. If there is no
            // whitespace (e.g. CJK), pos stays at the boundary; we
            // still must advance at least one byte to avoid
            // re-evaluating the same boundary forever. The `pos -= 1`
            // is the cancellation for the `: pos += 1` step.
            const before_ws = pos;
            while (pos < text.len and isAsciiSpaceOrLineEnd(text[pos])) pos += 1;
            if (pos == before_ws) {
                // No whitespace — just let the for-loop's pos += 1
                // step do its job (we're already at the byte after
                // the terminator).
            } else if (pos > 0) {
                pos -= 1;
            }
        }
    }
    return n;
}

test "chunk: sentence segmenter handles 'Dr.' abbreviation" {
    const text = "Dr. Smith went home. He arrived at 5pm.";
    // Old behavior: 3 boundaries (after every '.') -> 3 sentences.
    // New: "Dr." is abbreviation, NOT a boundary. "home." (followed
    // by capital "He") and the final "5pm." are boundaries. 2
    // sentences.
    try testing.expectEqual(@as(usize, 2), countSentences(text, default_abbreviations));
}

test "chunk: sentence segmenter handles 'U.S.A.' internal dots" {
    const text = "I went to the U.S.A. last summer. It was great!";
    // Internal dots in U.S.A. are followed by letters (no whitespace),
    // so they don't trigger boundaries. The final dot in "U.S.A." is
    // followed by space + lowercase 'l' -> NOT a boundary either
    // (no capital). "summer." is followed by space + capital "It" ->
    // boundary. The exclamation at the end closes sentence 2.
    try testing.expectEqual(@as(usize, 2), countSentences(text, default_abbreviations));
}

test "chunk: sentence segmenter handles bang/question/period mix" {
    const text = "Stop! Wait. Go.";
    // ! after "Stop" + capital "W" -> boundary.
    // . after "Wait" + capital "G" -> boundary.
    // . after "Go" + EOT -> boundary (closes the 3rd sentence).
    try testing.expectEqual(@as(usize, 3), countSentences(text, default_abbreviations));
}

test "chunk: paragraph boundary still wins under sentence-heavy text" {
    // \n\n always wins for `.paragraph`, regardless of sentence
    // segmentation.
    var v = Vocab.empty(testing.allocator);
    defer v.deinit();
    const pipe = bytePipeline(&v);

    const text = "Dr. Smith arrived.\n\nHe was late.";
    var res = try chunkText(testing.allocator, pipe, text, .{
        .max_tokens = 25,
        .overlap_tokens = 0,
        .boundary = .paragraph,
    });
    defer res.deinit();
    try testing.expect(res.chunks.len >= 1);
    for (res.chunks) |c| {
        if (c.byte_end == text.len) continue;
        try testing.expectEqual(@as(u8, '\n'), text[c.byte_end - 1]);
        try testing.expectEqual(@as(u8, '\n'), text[c.byte_end - 2]);
    }
}

test "chunk: custom sentence_abbreviations overrides defaults" {
    // The decisive case: capitalized "Bar" after "Foo.". Default
    // segmenter (no "Foo") calls this a boundary. With ["Foo"]
    // override, it does not.
    const text2 = "Foo. Bar quux.";
    try testing.expectEqual(
        @as(usize, 2),
        countSentences(text2, default_abbreviations),
    );
    try testing.expectEqual(
        @as(usize, 1),
        countSentences(text2, &.{"Foo"}),
    );

    // chunkText must propagate the override.
    var v = Vocab.empty(testing.allocator);
    defer v.deinit();
    const pipe = bytePipeline(&v);
    var res = try chunkText(testing.allocator, pipe, text2, .{
        .max_tokens = 6,
        .overlap_tokens = 0,
        .boundary = .sentence,
        .sentence_abbreviations = &.{"Foo"},
    });
    defer res.deinit();
    // With "Foo" abbreviation, the segmenter won't snap at byte 4
    // (after "Foo."). First chunk falls back to hard_end=6.
    try testing.expect(res.chunks[0].byte_end >= 6);
}

// ---------------------------------------------------------------------
// Post-1.12 sentence-segmenter improvements: ellipsis, decimals/URLs/
// files, quoted dialog, multi-language abbreviations, CJK
// ---------------------------------------------------------------------

test "chunk: ellipsis mid-sentence is not a boundary" {
    // `...` followed by lowercase is mid-sentence — sentence continues.
    const text = "He paused... then continued.";
    try testing.expectEqual(
        @as(usize, 1),
        countSentences(text, default_abbreviations),
    );
}

test "chunk: ellipsis at end of quoted dialog snaps after the quote" {
    // The closing quote AFTER `...` plus space + capital is a
    // sentence boundary; the boundary lies after the `"`, not at the
    // last dot of the ellipsis itself.
    const text = "And then...\" She left.";
    try testing.expectEqual(
        @as(usize, 2),
        countSentences(text, default_abbreviations),
    );
}

test "chunk: decimal numbers do not split sentences" {
    // `3.14` has a `.` flanked by digits — the alphanumeric-sandwich
    // rule suppresses the boundary. The second `.` after `14` IS a
    // boundary (space + capital).
    const text = "Pi is 3.14. So that's that.";
    try testing.expectEqual(
        @as(usize, 2),
        countSentences(text, default_abbreviations),
    );
}

test "chunk: URLs do not split sentences" {
    const text = "Visit example.com. It's great.";
    try testing.expectEqual(
        @as(usize, 2),
        countSentences(text, default_abbreviations),
    );
}

test "chunk: file extensions do not split sentences" {
    const text = "Save as notes.txt. Then close.";
    try testing.expectEqual(
        @as(usize, 2),
        countSentences(text, default_abbreviations),
    );
}

test "chunk: quoted dialog snaps after closing quote" {
    // `'I left.' She waved.` — boundary after the closing `'`, NOT at
    // the `.` inside the quote (which would dangle the `'`).
    const text = "'I left.' She waved.";
    try testing.expectEqual(
        @as(usize, 2),
        countSentences(text, default_abbreviations),
    );
}

test "chunk: French abbreviation 'M.' under .fr locale" {
    // Without French abbreviations, `M.` would split the sentence at
    // the capital `D` — English default produces 3 sentences. With
    // `.fr`, `M` is recognized and suppressed, yielding 2.
    const text = "M. Dupont est là. Il a dit oui.";
    try testing.expectEqual(
        @as(usize, 3),
        countSentences(text, default_abbreviations),
    );
    try testing.expectEqual(
        @as(usize, 2),
        countSentences(text, abbrevsForLocale(.fr)),
    );
}

test "chunk: German abbreviation 'Dr.' under .de locale" {
    const text = "Herr Dr. Schmidt kommt. Er ist müde.";
    try testing.expectEqual(
        @as(usize, 2),
        countSentences(text, abbrevsForLocale(.de)),
    );
}

test "chunk: CJK ideographic full stop is a sentence boundary" {
    const text = "これは文章です。次の文です。";
    try testing.expectEqual(
        @as(usize, 2),
        countSentences(text, default_abbreviations),
    );
}

test "chunk: mixed quoted-dot + ellipsis -> one narrative sentence" {
    // `He said 'Hi.' then waved...` — the dot inside the quote is
    // followed by lowercase `t` so it's NOT a boundary; the trailing
    // `...` runs to the end of the text. countSentences only counts
    // in-text boundaries (the EOT case is captured by the initial
    // n=1). Total: 1 sentence.
    const text = "He said 'Hi.' then waved...";
    try testing.expectEqual(
        @as(usize, 1),
        countSentences(text, default_abbreviations),
    );
}

test "chunk: 192.168.1.1 IP address does not split sentences" {
    // Triple-check the alphanumeric-sandwich rule on a chained-dot
    // construct. Every internal dot has digits on both sides; the
    // final `.` after `1` is followed by space + capital, which is
    // the real sentence boundary.
    const text = "Connect to 192.168.1.1. Then ping it.";
    try testing.expectEqual(
        @as(usize, 2),
        countSentences(text, default_abbreviations),
    );
}

test "chunk: multilang_union recognizes all locales' abbreviations" {
    // M. (French) + Dr. (English / multiple) in one sentence stream.
    const text = "M. Dupont rencontre Dr. Schmidt aujourd'hui. Voilà.";
    // With .multilang_union, both M. and Dr. are abbreviations.
    // Only the `aujourd'hui.` -> Voilà boundary fires. n=2.
    try testing.expectEqual(
        @as(usize, 2),
        countSentences(text, abbrevsForLocale(.multilang_union)),
    );
}

test "chunk: sentence_locale plumbs through chunkText" {
    var v = Vocab.empty(testing.allocator);
    defer v.deinit();
    const pipe = bytePipeline(&v);

    // `M. Dupont` with English default would snap at byte 3 (after
    // `M. `). With `.fr` locale, `M.` is an abbreviation and the
    // snap shifts later.
    const text = "M. Dupont rencontre Pierre. Ils discutent.";

    var res_en = try chunkText(testing.allocator, pipe, text, .{
        .max_tokens = 15,
        .overlap_tokens = 0,
        .boundary = .sentence,
        .sentence_locale = .en,
    });
    defer res_en.deinit();

    var res_fr = try chunkText(testing.allocator, pipe, text, .{
        .max_tokens = 15,
        .overlap_tokens = 0,
        .boundary = .sentence,
        .sentence_locale = .fr,
    });
    defer res_fr.deinit();

    // English locale: first chunk ends at the `.` after `M` (byte 2)
    // since `M` isn't in the English abbreviation list, falling back
    // to hard_end if no later boundary fits the window. We just
    // assert the two locales produce DIFFERENT chunk layouts to
    // confirm the locale wiring is live.
    const en_first = res_en.chunks[0].byte_end;
    const fr_first = res_fr.chunks[0].byte_end;
    try testing.expect(en_first != fr_first);
}

// ---------------------------------------------------------------------
// Post-1.13 v3 residuals: lowercase starts, nested quotes, period-final
// acronyms, other-script terminators
// ---------------------------------------------------------------------

test "chunk: aggressive_lowercase opt-in fires on sentence-opener words" {
    // Strict rule: `.`/`!`/`?` + space + lowercase = NOT a boundary.
    // Aggressive: + space + sentence-opener-word (lowercase) = boundary.
    const text = "He left. then she arrived.";
    const default_ctx: SegmenterCtx = .{
        .abbreviations = default_abbreviations,
        .aggressive_lowercase = false,
    };
    const aggro_ctx: SegmenterCtx = .{
        .abbreviations = default_abbreviations,
        .aggressive_lowercase = true,
    };
    // Default: lowercase `t` after `.` blocks the boundary. n=1.
    try testing.expectEqual(@as(usize, 1), countSentencesCtx(text, default_ctx));
    // Aggressive: `then` is in the opener list. n=2.
    try testing.expectEqual(@as(usize, 2), countSentencesCtx(text, aggro_ctx));
}

test "chunk: aggressive_lowercase ignores non-opener lowercase" {
    // `mango` isn't a sentence-opener — aggressive mode must STILL not
    // fire here (heuristic deliberately conservative).
    const text = "I ate it. mango is yellow.";
    const aggro_ctx: SegmenterCtx = .{
        .abbreviations = default_abbreviations,
        .aggressive_lowercase = true,
    };
    try testing.expectEqual(@as(usize, 1), countSentencesCtx(text, aggro_ctx));
}

test "chunk: nested quotes do not split mid-dialog" {
    // The inner `'wait. Then go.'` quote has a CAPITAL after the inner
    // `.` — under strict rules the segmenter would over-split (3
    // sentences). With nested-quote tracking, depth at the inner `.`
    // is 2 (both `"` and `'` open), so the boundary is suppressed; the
    // boundary fires after the inner closing `'` (which still snaps at
    // `' I`). n=2.
    const text = "\"He said 'wait. Then go.' I left.\"";
    try testing.expectEqual(
        @as(usize, 2),
        countSentences(text, default_abbreviations),
    );
}

test "chunk: acronym at end of sentence — U.S. mid-clause" {
    // `We visited the U.S. It was great.` — at the `.` after `U.S.`,
    // the acronym detector recognizes `U.S.` as an abbreviation
    // regardless of locale list, so no false split. Boundary fires at
    // `It` after `great.`. n=2 (not 3+).
    const text = "We visited the U.S. It was great.";
    try testing.expectEqual(
        @as(usize, 2),
        countSentences(text, default_abbreviations),
    );
}

test "chunk: acronym U.S.A. mid-text" {
    // `The U.S.A. is large. Russia is larger.` — `U.S.A.` is an
    // acronym, suppressed. `large. Russia` fires. n=2.
    const text = "The U.S.A. is large. Russia is larger.";
    try testing.expectEqual(
        @as(usize, 2),
        countSentences(text, default_abbreviations),
    );
}

test "chunk: Arabic question mark terminates sentence" {
    // U+061F ARABIC QUESTION MARK — 2 bytes (D8 9F). Two Arabic
    // sentences separated by the `؟`.
    const text = "هل تتحدث العربية؟ نعم، قليلاً.";
    try testing.expectEqual(
        @as(usize, 2),
        countSentences(text, default_abbreviations),
    );
}

test "chunk: Armenian full stop terminates sentence" {
    // U+0589 ARMENIAN FULL STOP — 2 bytes (D6 89). Two clauses.
    const text = "Բարև։ Ինչպես ես։";
    try testing.expectEqual(
        @as(usize, 2),
        countSentences(text, default_abbreviations),
    );
}

test "chunk: Tibetan shad terminates sentence" {
    // U+0F0D TIBETAN MARK SHAD — 3 bytes (E0 BC 8D). Two clauses.
    const text = "སུ་ལ་ཡིན། སུ་ལ་མིན།";
    try testing.expectEqual(
        @as(usize, 2),
        countSentences(text, default_abbreviations),
    );
}

test "chunk: Ethiopic full stop terminates sentence" {
    // U+1362 ETHIOPIC FULL STOP — 3 bytes (E1 8D A2). Two clauses.
    const text = "ሰላም። እንዴት ነህ።";
    try testing.expectEqual(
        @as(usize, 2),
        countSentences(text, default_abbreviations),
    );
}

test "chunk: aggressive_lowercase plumbs through chunkText" {
    // End-to-end test: `aggressive_lowercase = true` changes the chunk
    // layout for text with lowercase sentence starts.
    var v = Vocab.empty(testing.allocator);
    defer v.deinit();
    const pipe = bytePipeline(&v);

    const text = "He left. then she arrived. but he came back.";

    var res_strict = try chunkText(testing.allocator, pipe, text, .{
        .max_tokens = 20,
        .overlap_tokens = 0,
        .boundary = .sentence,
        .aggressive_lowercase = false,
    });
    defer res_strict.deinit();

    var res_aggro = try chunkText(testing.allocator, pipe, text, .{
        .max_tokens = 20,
        .overlap_tokens = 0,
        .boundary = .sentence,
        .aggressive_lowercase = true,
    });
    defer res_aggro.deinit();

    // With aggressive mode, more sentence boundaries become available,
    // so the snap-left chunker tends to produce more, smaller chunks
    // (or at least different cuts). Assert layouts differ.
    var same: bool = res_strict.chunks.len == res_aggro.chunks.len;
    if (same) {
        for (res_strict.chunks, res_aggro.chunks) |a, b| {
            if (a.byte_end != b.byte_end) {
                same = false;
                break;
            }
        }
    }
    try testing.expect(!same);
}

// ---------------------------------------------------------------------
// Post-1.15 v4: Indic + SE Asian terminators, Hindi/Vietnamese/Polish
// abbreviations, multi-dot abbreviation polish (Ph.D., Lt. Col.)
// ---------------------------------------------------------------------

test "chunk: Devanagari danda terminates sentence" {
    // U+0964 DEVANAGARI DANDA — 3 bytes (E0 A5 A4). Two Hindi clauses.
    const text = "यह वाक्य है। यह दूसरा वाक्य है।";
    try testing.expectEqual(
        @as(usize, 2),
        countSentences(text, default_abbreviations),
    );
}

test "chunk: Devanagari double-danda terminates sentence" {
    // U+0965 DEVANAGARI DOUBLE DANDA — 3 bytes (E0 A5 A5). Verse
    // separator in Sanskrit; same unconditional rule.
    const text = "श्लोकः एकः॥ श्लोकः द्वौ॥";
    try testing.expectEqual(
        @as(usize, 2),
        countSentences(text, default_abbreviations),
    );
}

test "chunk: Mongolian full stop terminates sentence" {
    // U+1803 MONGOLIAN FULL STOP — 3 bytes (E1 A0 83). Two clauses.
    const text = "Сайн байна уу᠃ Сайн᠃";
    try testing.expectEqual(
        @as(usize, 2),
        countSentences(text, default_abbreviations),
    );
}

test "chunk: Burmese section sign terminates sentence" {
    // U+104B MYANMAR SIGN SECTION — 3 bytes (E1 81 8B). Two clauses.
    const text = "မင်္ဂလာပါ။ နေကောင်းပါသလား။";
    try testing.expectEqual(
        @as(usize, 2),
        countSentences(text, default_abbreviations),
    );
}

test "chunk: Khmer khan terminates sentence" {
    // U+17D4 KHMER SIGN KHAN — 3 bytes (E1 9F 94). Two clauses.
    const text = "សួស្តី។ តើអ្នកសុខសប្បាយទេ។";
    try testing.expectEqual(
        @as(usize, 2),
        countSentences(text, default_abbreviations),
    );
}

test "chunk: Hindi 'डॉ.' Devanagari abbreviation (.hi locale)" {
    // `डॉ. शर्मा` — at the `.` after `डॉ`, the Unicode-aware abbrev
    // walk-back identifies the cluster `डॉ` (= ड Lo + ॉ Mc) and
    // matches the `.hi` abbreviation list. No false split there. The
    // two real boundaries are the two trailing `।` dandas.
    const text = "डॉ. शर्मा आज आए। उन्होंने कहा अच्छा है।";
    try testing.expectEqual(
        @as(usize, 2),
        countSentences(text, abbrevsForLocale(.hi)),
    );
    // English locale also yields 2 here only because Devanagari `श`
    // isn't Lu/Lt — `followedByCapitalOrEnd` fails after the dot, so
    // no boundary fires at `डॉ.` either way. The abbreviation list is
    // a safety net for the case where the next word starts with a
    // proper noun in Roman script (which DOES capitalize), e.g.
    // a Hindi sentence with an English place name. The Hindi-locale
    // path is the one we contract on; this assertion documents it.
    try testing.expectEqual(
        @as(usize, 2),
        countSentences(text, default_abbreviations),
    );
}

test "chunk: Vietnamese 'TS.' abbreviation (.vi locale)" {
    // `TS.` (Tiến sĩ / Doctor) is a Vietnamese academic title. Under
    // English defaults the dot + space + capital `N` (`Nguyễn`) would
    // fire a boundary. Under `.vi` it's suppressed.
    const text = "TS. Nguyễn đã đến. Anh ấy nói tốt.";
    // English: 3 sentences (TS. splits).
    try testing.expectEqual(
        @as(usize, 3),
        countSentences(text, default_abbreviations),
    );
    // Vietnamese: 2 sentences.
    try testing.expectEqual(
        @as(usize, 2),
        countSentences(text, abbrevsForLocale(.vi)),
    );
}

test "chunk: Polish 'Dr.' abbreviation (.pl locale)" {
    // `Dr.` is already an English abbreviation, so the English count is
    // 2 here too. The point of this test is that the `.pl` locale list
    // ALSO recognizes it (via the `dr` lowercase entry — Polish
    // convention writes academic titles lowercase). We use the
    // capitalized `Dr.` form because `default_abbreviations` includes
    // it; both paths must produce the same answer.
    const text = "Dr. Kowalski przyszedł. Powiedział tak.";
    try testing.expectEqual(
        @as(usize, 2),
        countSentences(text, abbrevsForLocale(.pl)),
    );
    // Also verify a lowercase-Polish-style abbreviation: `prof.`.
    const text2 = "prof. Nowak wykładał. Studenci słuchali.";
    // English default doesn't have lowercase `prof` -> 3 sentences.
    try testing.expectEqual(
        @as(usize, 3),
        countSentences(text2, default_abbreviations),
    );
    // Polish locale: `prof` -> 2 sentences.
    try testing.expectEqual(
        @as(usize, 2),
        countSentences(text2, abbrevsForLocale(.pl)),
    );
}

test "chunk: extended English 'Lt. Col.' two-token abbreviation" {
    // `Lt. Col. Smith reported. He was tired.`
    // - At `Lt.`: `Lt` is now in `default_abbreviations` -> suppress.
    // - At `Col.`: `Col` was already there -> suppress.
    // - At `reported.`: space + capital `He` -> boundary.
    // - Trailing `tired.` closes the second sentence (EOT).
    // Total: 2 sentences.
    const text = "Lt. Col. Smith reported. He was tired.";
    try testing.expectEqual(
        @as(usize, 2),
        countSentences(text, default_abbreviations),
    );
}

test "chunk: 'Ph.D.' multi-dot academic abbreviation" {
    // `Ph.D. is a long road. It's worth it.`
    // - At the FIRST dot (`Ph.D`): alphanumeric on both sides -> not
    //   a boundary (decimal/URL rule).
    // - At the SECOND dot (`Ph.D.`): the abbreviation walk-back reads
    //   `Ph.D` (letters + internal dot), which is in the abbreviation
    //   list -> not a boundary.
    // - At `road.`: space + capital `It` -> boundary.
    // - Trailing `it.` closes sentence 2 (EOT).
    // Total: 2 sentences.
    const text = "Ph.D. is a long road. It's worth it.";
    try testing.expectEqual(
        @as(usize, 2),
        countSentences(text, default_abbreviations),
    );
}

test "chunk: multilang_union includes Hindi/Vietnamese/Polish entries" {
    // The union list must accept all three new locales' abbreviations.
    // We exercise one entry per locale to confirm the merge wiring.

    // Hindi: डॉ.
    const text_hi = "डॉ. शर्मा आज आए। बाद में।";
    try testing.expectEqual(
        @as(usize, 2),
        countSentences(text_hi, abbrevsForLocale(.multilang_union)),
    );

    // Vietnamese: TS.
    const text_vi = "TS. Nguyễn đã đến. Anh ấy vui.";
    try testing.expectEqual(
        @as(usize, 2),
        countSentences(text_vi, abbrevsForLocale(.multilang_union)),
    );

    // Polish: prof.
    const text_pl = "prof. Nowak wykładał. Studenci słuchali.";
    try testing.expectEqual(
        @as(usize, 2),
        countSentences(text_pl, abbrevsForLocale(.multilang_union)),
    );
}

test "chunk: extended English abbreviation list count regression" {
    // Document the 1.13 -> 1.15 grow-out: the English list went from
    // 27 to ~50 entries. If a future refactor accidentally drops or
    // duplicates entries this catches it. The exact number is
    // load-bearing for the README; bump both together when adding.
    try testing.expect(default_abbreviations.len >= 45);
}

test "chunk: Hindi locale list count sanity" {
    // ~15-20 entries per the agent-C spec; assert >= 15.
    try testing.expect(hindi_abbreviations.len >= 15);
}

test "chunk: Vietnamese locale list count sanity" {
    try testing.expect(vietnamese_abbreviations.len >= 15);
}

test "chunk: Polish locale list count sanity" {
    try testing.expect(polish_abbreviations.len >= 15);
}

// ---------------------------------------------------------------------
// Dict-based word segmentation for scriptio-continua scripts (1.16)
// ---------------------------------------------------------------------

/// Run the dict-based segmenter over an entire text and collect the
/// byte offsets at which words begin. Mirrors the per-position check
/// the chunker does, but pre-computes the boundary set for testability.
fn collectDictWordStarts(allocator: std.mem.Allocator, text: []const u8) ![]u32 {
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(allocator);
    if (text.len == 0) return out.toOwnedSlice(allocator);
    try out.append(allocator, 0);
    var p: u32 = 1;
    while (p <= text.len) : (p += 1) {
        if (isWordDictBoundaryByte(text, p)) {
            // Avoid duplicates when scanning past the end of a word.
            const last = out.items[out.items.len - 1];
            if (p != last) try out.append(allocator, p);
        }
    }
    return out.toOwnedSlice(allocator);
}

test "chunk: Thai dict segments a known phrase" {
    // "ฉันรักคุณ" = "I love you". Thai is scriptio-continua: the dict
    // segmenter must split this into ["ฉัน", "รัก", "คุณ"] — each
    // entry is in the bundled Thai dict, each 9 bytes (3 codepoints
    // * 3 bytes/cp). Total: 27 bytes.
    const text = "ฉันรักคุณ";
    try testing.expectEqual(@as(usize, 27), text.len);
    const starts = try collectDictWordStarts(testing.allocator, text);
    defer testing.allocator.free(starts);
    // Expect word boundaries at byte offsets 0, 9 (end of ฉัน),
    // 18 (end of รัก), 27 (end of คุณ / end of input).
    var has_9 = false;
    var has_18 = false;
    for (starts) |s| {
        if (s == 9) has_9 = true;
        if (s == 18) has_18 = true;
    }
    try testing.expect(has_9);
    try testing.expect(has_18);
}

test "chunk: Lao dict segments a known phrase" {
    // "ຂ້ອຍຮັກເຈົ້າ" = "I love you" in Lao. Dict entries: ຂ້ອຍ, ຮັກ, ເຈົ້າ.
    const text = "ຂ້ອຍຮັກເຈົ້າ";
    const starts = try collectDictWordStarts(testing.allocator, text);
    defer testing.allocator.free(starts);
    // ຂ້ອຍ = ຂ (3) + ້ (3) + ອ (3) + ຍ (3) = 12 bytes
    // ຮັກ = ຮ (3) + ັ (3) + ກ (3) = 9 bytes, end at 21
    // ເຈົ້າ = ເ (3) + ຈ (3) + ົ (3) + ້ (3) + າ (3) = 15 bytes, end at 36
    try testing.expect(starts.len >= 4);
    var has_12 = false;
    var has_21 = false;
    for (starts) |s| {
        if (s == 12) has_12 = true;
        if (s == 21) has_21 = true;
    }
    try testing.expect(has_12);
    try testing.expect(has_21);
}

test "chunk: Chinese HSK words segment correctly" {
    // "我爱你" = "I love you". All three are HSK 1 single-character
    // words in the bundled dict. Each Chinese character is 3 bytes.
    const text = "我爱你";
    const starts = try collectDictWordStarts(testing.allocator, text);
    defer testing.allocator.free(starts);
    // Expect starts at 0, 3, 6 (each character is its own word).
    try testing.expect(starts.len >= 3);
    var has_3 = false;
    var has_6 = false;
    for (starts) |s| {
        if (s == 3) has_3 = true;
        if (s == 6) has_6 = true;
    }
    try testing.expect(has_3);
    try testing.expect(has_6);
}

test "chunk: Chinese longest-match prefers multi-char compound" {
    // "我们是学生" = "We are students". Tokens: 我们 (we, 2 chars=6 bytes),
    // 是 (1 char=3 bytes), 学生 (students, 2 chars=6 bytes). All in dict.
    // Longest-match-first must pick 我们 (in dict) over 我 (also in dict)
    // — confirms the longest-prefix rule.
    const text = "我们是学生";
    const starts = try collectDictWordStarts(testing.allocator, text);
    defer testing.allocator.free(starts);
    // Expect boundaries at 0, 6 (end of 我们), 9 (end of 是), 15 (end of 学生).
    var has_6 = false;
    var has_9 = false;
    var has_15 = false;
    for (starts) |s| {
        if (s == 6) has_6 = true;
        if (s == 9) has_9 = true;
        if (s == 15) has_15 = true;
    }
    try testing.expect(has_6);
    try testing.expect(has_9);
    try testing.expect(has_15);
}

test "chunk: Japanese hiragana words segment correctly" {
    // "これはペンです" = "This is a pen". Tokens: これ (this, 6 bytes hiragana),
    // は (particle, 3 bytes), ペン (pen, 6 bytes katakana), です (copula, 6 bytes).
    // The hiragana run これ + は uses the ja dict (longest match picks これ).
    const text = "これはペンです";
    const starts = try collectDictWordStarts(testing.allocator, text);
    defer testing.allocator.free(starts);
    // Expect boundaries at 0, 6 (end of これ), 9 (end of は — and start
    // of katakana ペン), 15 (end of ペン), 21 (end of です).
    var has_6 = false;
    var has_9 = false;
    var has_15 = false;
    var has_21 = false;
    for (starts) |s| {
        if (s == 6) has_6 = true;
        if (s == 9) has_9 = true;
        if (s == 15) has_15 = true;
        if (s == 21) has_21 = true;
    }
    try testing.expect(has_6);
    try testing.expect(has_9);
    // 15 lands on a script transition (katakana → hiragana), so it
    // should be a boundary regardless of dict membership.
    try testing.expect(has_15);
    try testing.expect(has_21);
}

test "chunk: fallback to per-codepoint when no dict match" {
    // A Thai string of mostly-unknown content. The fallback should
    // still emit boundaries at each codepoint so progress is preserved.
    const text = "ฌญฎฏฐฑฒ"; // rare Thai consonants, none in our tiny dict
    const starts = try collectDictWordStarts(testing.allocator, text);
    defer testing.allocator.free(starts);
    // 7 codepoints * 3 bytes each = 21 bytes total. Each codepoint
    // boundary should be a word boundary in fallback mode.
    try testing.expect(starts.len >= 7);
}

test "chunk: round-trip — dict-chunked output reassembles to input" {
    var v = Vocab.empty(testing.allocator);
    defer v.deinit();
    const pipe = bytePipeline(&v);

    const text = "我爱你我们是学生";
    var res = try chunkText(testing.allocator, pipe, text, .{
        .max_tokens = 6,
        .overlap_tokens = 0,
        .boundary = .word_dict,
    });
    defer res.deinit();

    // Non-overlapping chunks must concatenate back to the input.
    var rebuilt: std.ArrayList(u8) = .empty;
    defer rebuilt.deinit(testing.allocator);
    var cursor: u32 = 0;
    for (res.chunks) |c| {
        try testing.expectEqual(cursor, c.byte_start);
        try rebuilt.appendSlice(testing.allocator, text[c.byte_start..c.byte_end]);
        cursor = c.byte_end;
    }
    try testing.expectEqual(@as(u32, @intCast(text.len)), cursor);
    try testing.expectEqualStrings(text, rebuilt.items);
}

test "chunk: word_dict on ASCII corpus matches .word behavior" {
    // Mixed-script handling: ASCII whitespace boundaries should still
    // fire under .word_dict. The whole input is below 0x80, no script
    // run matches a dict — but the whitespace path is the same as
    // .word, so the output layout matches.
    var v = Vocab.empty(testing.allocator);
    defer v.deinit();
    const pipe = bytePipeline(&v);

    const text = "the quick brown fox";
    var res = try chunkText(testing.allocator, pipe, text, .{
        .max_tokens = 6,
        .overlap_tokens = 0,
        .boundary = .word_dict,
    });
    defer res.deinit();
    try testing.expect(res.chunks.len >= 2);
    // Every non-final chunk's byte_end must be at a space.
    for (res.chunks) |c| {
        if (c.byte_end == text.len) continue;
        const prev = text[c.byte_end - 1];
        try testing.expect(prev == ' ' or prev == '\t' or prev == '\n');
    }
}

test "chunk: Thai/Lao sentence locale requires double-newline" {
    // Thai text with a single `\n` mid-text + a `\n\n` paragraph break.
    // Under `Locale.en` every `\n` is a sentence boundary (the existing
    // segmenter rule). Under `Locale.th` only the `\n\n` paragraph
    // break fires.
    const text_th = "ฉันรักคุณ\nและฉันชอบเธอ\n\nนี่คือย่อหน้าใหม่";
    const en_ctx: SegmenterCtx = .{
        .abbreviations = default_abbreviations,
        .aggressive_lowercase = false,
        .require_double_newline = false,
    };
    const th_ctx: SegmenterCtx = .{
        .abbreviations = empty_abbreviations,
        .aggressive_lowercase = false,
        .require_double_newline = true,
    };
    // English locale: every `\n` boundary fires. 1 single-newline +
    // 1 double-newline (the second `\n` also fires) = 2 boundaries
    // -> 3 sentences. (The double-newline fires both \n's, so each
    // counts; isSentenceBoundary doesn't deduplicate them.)
    const en_count = countSentencesCtx(text_th, en_ctx);
    // Thai locale: only `\n\n` counts. The double-newline at the
    // single break fires once (at the second `\n`). The single `\n`
    // is suppressed. 1 boundary -> 2 sentences.
    const th_count = countSentencesCtx(text_th, th_ctx);
    try testing.expect(en_count > th_count);
    try testing.expectEqual(@as(usize, 2), th_count);
}

test "chunk: Locale.th plumbs through chunkText to sentence mode" {
    var v = Vocab.empty(testing.allocator);
    defer v.deinit();
    const pipe = bytePipeline(&v);

    // Thai paragraph break in the middle. With Locale.th the
    // segmenter only fires on `\n\n`, so the chunker should produce
    // at least one chunk that ends precisely at the paragraph break.
    const text = "ฉันรักคุณ\n\nนี่คือย่อหน้าใหม่";
    var res = try chunkText(testing.allocator, pipe, text, .{
        .max_tokens = 40,
        .overlap_tokens = 0,
        .boundary = .sentence,
        .sentence_locale = .th,
    });
    defer res.deinit();
    try testing.expect(res.chunks.len >= 1);
    // At least one non-final chunk should end on `\n\n` (the only
    // valid sentence boundary in this pure-Thai text). Other chunks
    // may fall back to hard_end since there are no further boundaries.
    var saw_paragraph_break = false;
    for (res.chunks) |c| {
        if (c.byte_end >= 2 and c.byte_end <= text.len and
            text[c.byte_end - 1] == '\n' and text[c.byte_end - 2] == '\n')
        {
            saw_paragraph_break = true;
        }
    }
    try testing.expect(saw_paragraph_break);
}

test "chunk: dict word counts are within budgeted ranges" {
    // Sanity-check the bundled dict sizes against the spec. Thai/Lao
    // ~500, Chinese ~1000, Japanese ~500. Bump these when intentionally
    // growing a dict; they protect against a typo dropping half the
    // entries.
    try testing.expect(dict_th.words.len >= 400);
    try testing.expect(dict_th.words.len <= 700);
    try testing.expect(dict_lo.words.len >= 400);
    try testing.expect(dict_lo.words.len <= 700);
    try testing.expect(dict_zh.words.len >= 700);
    try testing.expect(dict_zh.words.len <= 1500);
    try testing.expect(dict_ja.words.len >= 400);
    try testing.expect(dict_ja.words.len <= 1200);
}
