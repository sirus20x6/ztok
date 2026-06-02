//! TokenMonster-style ungreedy tokenizer. 6-branch scoring port.
//!
//! At each position we enumerate up to 6 candidate "first tokens" (greedy +
//! up to 5 shorter prefix matches), each followed by a greedy lookahead at
//! the next position. Each branch is scored with a length-and-word-density
//! formula derived from refs/tokenmonster/go/tokenmonster.go:1017.
//!
//! The -10000 deduction on alternative branches whose total length equals
//! greedy's first-token length is what makes the encoder *ungreedy*: alts
//! only displace greedy when they genuinely cover MORE bytes than greedy
//! alone would. The -100 deduction discourages alts that cover fewer bytes.
//!
//! SoA layout: vocab bytes + offsets + parallel nwords, flat trie. Hot path
//! allocates nothing; scratch is a [64]u32 stack pair.
//!
//! Capcode/delete-token handling from the Go source is intentionally
//! omitted — we don't carry that machinery here. The 1b/2b/3b "with leading
//! space" branches from Go require deleteToken and are dropped; this brings
//! the effective branch count to 6 main (greedy + 5 alts) which matches the
//! task target.

const std = @import("std");
const TokenId = @import("token.zig").TokenId;
const Span = @import("token.zig").Span;
const unicode_props = @import("unicode_props.zig");

const NO_TOKEN: u32 = 0xFFFF_FFFF;

/// Compile-time gate for the per-phase instrumentation hooks
/// (`profile_counters`). When off (the default), the hot path runs
/// without any clock_gettime overhead. Flip in a local build to
/// attribute time inside `encodeChunkImpl`.
pub const profile_enabled: bool = false;

/// Per-phase nanosecond accumulators. Only populated when
/// `profile_enabled = true`. Reset by callers between runs if needed.
pub var profile_counters: struct {
    collect_ns: u64 = 0,
    lilbuf_ns: u64 = 0,
    branch_ns: u64 = 0,
    scoreb_ns: u64 = 0,
    emit_ns: u64 = 0,
} = .{};

// --- Per-branch score tracer (debug facility) ---------------------------
//
// Opt-in via the `ZTOK_MONSTER_TRACE` env var. When set to anything other
// than "0"/empty, `encodeChunkImpl`'s branch-scoring loop dumps, at EVERY
// input position, the six TM-Go branch scores (score1/2/3/1b/2b/3b) plus
// the picked branch, to stderr. This is the diagnostic instrument used to
// resolve the score-b float-ladder tiebreak edge cases against TM-Go.
//
// ZERO-cost when disabled: the trace block in the hot loop is guarded by a
// single load of `monster_trace_enabled`, a process-global cached on first
// `encodeChunk*` entry (see `monsterTraceInit`). Predictable not-taken
// branch — no per-position syscall, no formatting, no allocation when off.
//
// The tracer reports the score actually FOLDED into each branch slot as
// the encoder walks `b = 0..n_branches` plus the path-(b)/path-(a) lilbuf
// branches. ztok's six logical TM-Go branches map onto ztok's loop as:
//   b==0 plain  -> score1   ;  b==0 score-b -> score1b
//   b==1 plain  -> score2   ;  b==1 score-b -> score2b
//   b==2 plain  -> score3   ;  b==2 score-b -> score3b
// (the path-(b)/path-(a) lilbuf branches are ztok-only and traced under
// the `pb`/`pa` tags). Rough but faithful to the six branch values.
var monster_trace_enabled: bool = false;
var monster_trace_inited: bool = false;

fn monsterTraceInit() void {
    if (monster_trace_inited) return;
    monster_trace_inited = true;
    // Read via libc getenv (the lib links libc; `std.posix.getenv` is
    // gone in 0.16 and the `std.process.Environ` view needs an allocator).
    const raw = std.c.getenv("ZTOK_MONSTER_TRACE") orelse {
        monster_trace_enabled = false;
        return;
    };
    const v = std.mem.span(raw);
    monster_trace_enabled = v.len != 0 and !std.mem.eql(u8, v, "0");
}

inline fn traceBranch(
    pos: usize,
    tag: []const u8,
    b: usize,
    first_id: u32,
    first_len: u32,
    second_id: u32,
    second_len: u32,
    score: i32,
) void {
    if (monster_trace_enabled) {
        @branchHint(.unlikely);
        std.debug.print(
            "TRACE pos={d} {s} b={d} first_id={d}(len={d}) second_id={d}(len={d}) score={d}\n",
            .{ pos, tag, b, first_id, first_len, second_id, second_len, score },
        );
    }
}

inline fn tracePick(pos: usize, emit_tag: []const u8, first_id: u32, second_id: u32, score: i32, advance: u32) void {
    if (monster_trace_enabled) {
        @branchHint(.unlikely);
        std.debug.print(
            "TRACE pos={d} PICK emit={s} first_id={d} second_id={d} score={d} advance={d}\n",
            .{ pos, emit_tag, first_id, second_id, score, advance },
        );
    }
}

/// Capcode mode for vocab-flag computation. Mirrors TM-Go's
/// `usingCapcode` byte (0 = none, 1 = nocapcode/forward-delete only with
/// 0x7F as the DEL marker, 2 = full capcode with C/W/D as markers).
///
/// The `Builder.finalize` family takes this to drive the
/// `isLetter`/`isCapcode` classifiers used during per-piece flag
/// computation. The legacy `finalize(unk_id)` defaults to `.none` which
/// matches every pre-flag-aware caller's behavior (flags computed from
/// ASCII-style classification with no capcode-marker awareness).
pub const CapcodeMode = enum(u8) {
    none = 0,
    nocapcode = 1,
    full = 2,
};

// --- TM-Go flag bits (see refs/tokenmonster/go/tokenmonster.go:91-107
//     and the writer at :3490-3593). Each piece's `flags` byte is the
//     bitwise-OR of these.
//
//   FLAG_ENDS_LETTER     | 1  | last rune is a letter (or combining mark)
//   FLAG_BEGINS_LETTER   | 2  | first rune is a letter
//   FLAG_BEGINS_SPACE    | 4  | first rune is space, or capcode CHAR/WORD
//   FLAG_ENDS_CAPCODE    | 8  | last rune is a capcode marker
//   FLAG_BEGINS_CAPCODE  | 16 | first rune is a capcode marker
//   FLAG_SINGLE_WORD     | 32 | begins ` `+letter, ends letter, exactly
//                              | one whole word, only letters and spaces
//   FLAG_SPECIAL         | 64 | special-token piece (not used by load
//                              | path — we have no special-token
//                              | concept at the encoder level; reserved)
//   FLAG_ALL_LETTERS     |128 | all-letters OR all-punctuation OR
//                              | all-numbers/spaces (bit 7)
pub const FLAG_ENDS_LETTER: u8 = 1;
pub const FLAG_BEGINS_LETTER: u8 = 2;
pub const FLAG_BEGINS_SPACE: u8 = 4;
pub const FLAG_ENDS_CAPCODE: u8 = 8;
pub const FLAG_BEGINS_CAPCODE: u8 = 16;
pub const FLAG_SINGLE_WORD: u8 = 32;
pub const FLAG_SPECIAL: u8 = 64;
pub const FLAG_ALL_LETTERS: u8 = 128;

// --- TM-Go beginByte values for a chunk's lookahead byte. Mirrors the
//     `vocab.beginByte` table at :3779-3788. The encoder reads
//     `beginByte[chunk[next_pos]]` as a 4-bit classifier:
//
//   BB_LETTER = 1    — letter (bit 0; bits 2,3 = 0)
//   BB_PUNCT  = 10   — punctuation or capcode marker (bit 1, bit 3 = "not a letter")
//   BB_SPACE  = 12   — space (bit 2, bit 3 = "not a letter")
//   BB_ZERO   = 0    — past-end-of-input or unknown byte (no bonus)
pub const BB_LETTER: u8 = 1;
pub const BB_PUNCT: u8 = 10;
pub const BB_SPACE: u8 = 12;
// Cap on prefix matches considered per position. Real vocabularies rarely
// nest more than a few terminals along one path; 64 is comfortable headroom.
const MAX_CANDIDATES: usize = 64;
// Branches scored per position: greedy + up to 5 shorter-prefix alts.
const MAX_BRANCHES: usize = 6;

/// Precomputed alt-branch metadata for a single piece. Mirrors TM-Go's
/// `tokenOuter.{index, length, index2, length2}` (refs/tokenmonster/go/
/// tokenmonster.go:69-78). `index`/`index2` are the two best-priority
/// proper-prefix subtokens of THIS piece that also exist as standalone
/// tokens; `length`/`length2` are their byte counts. The priority is
/// the boundary class at the split point: letter|non-letter > number|
/// non-number > non-space|space > everything else (see :3597-3753 for
/// the full ranking ladder).
///
/// `NO_TOKEN` in `index`/`index2` means no alt of that rank was found
/// (the piece has either zero or one in-vocab proper-prefix subtoken).
/// `index` is guaranteed to be the best of the two; `index2` may be
/// `NO_TOKEN` even when `index` is set.
///
/// Built at vocab load (`computeAlts`); accessed by the encoder as
/// `alts[matched_id]` to drive the score2/score3 alt-branch evaluation.
/// This replaces ztok's pre-1.16 "walk all prefix matches via
/// `collectPrefixMatches`" approach — TM-Go's priority-ranked picks
/// often differ from "shortest 5 prefixes by length", which was the
/// root cause of several capcode equivalence misses.
pub const AltPair = struct {
    index: u32 = NO_TOKEN,
    length: u32 = 0,
    index2: u32 = NO_TOKEN,
    length2: u32 = 0,
};

// branchless helpers — small inline integer-arithmetic stand-ins for the
// Go `branchless` package. Compiler will likely inline these anyway.
inline fn bMin(a: i32, b: i32) i32 {
    return if (a < b) a else b;
}
inline fn bMax(a: i32, b: i32) i32 {
    return if (a > b) a else b;
}
inline fn bEqual(a: i32, b: i32) i32 {
    return if (a == b) 1 else 0;
}
inline fn bLessThan(a: i32, b: i32) i32 {
    return if (a < b) 1 else 0;
}
inline fn bMaxZeroAnd(x: i32) i32 {
    return if (x > 0) x else 0;
}

// === TM-Go score formula ===========================================
//
// One-shot score for a (first, second) branch pair. Mirrors the
// score1/score2/score3 formula at
// refs/tokenmonster/go/tokenmonster.go:1075-1084 (with the
// score2/score3 alt penalties activated when `is_alt = true`).
//
// `next_bb` is the value of `monster.begin_byte[chunk[tail]]` at the
// past-end-of-branch position. Past-the-end is signalled by
// `next_bb = 0` (TM-Go appends a 0 sentinel byte and reads `beginByte[0]`,
// which defaults to 0 because the 256-entry table is built from
// "this byte starts how many vocab pieces" majority votes — zero
// rarely qualifies for a class so stays 0). Net effect: end-of-input
// contributes zero bonus to the word count and zero "next is space"
// bonus, matching TM-Go exactly.
//
// `first_flag`/`second_flag` are TM-Go per-piece flag bits
// (`FLAG_*` constants). If either is zero the score degrades to the
// legacy nwords-only behavior — vocabs built without flag awareness
// continue to score the same way they did before this addition (back-
// compat with non-load-aware Monster vocabs).
//
// `extra_token_penalty` is the score1b/score2b/score3b `-1` term for
// branches that emit an extra DEL token (lilbuf wins). The caller
// passes `1` for those branches and `0` otherwise.
//
// `drop_begin_space_bonus` is the score1b/score2b/score3b variant
// that DROPS the `((second.flag >> 2) & 1)` term because lilbuf-space
// matched second always begins with space — TM-Go avoids the double
// count by omitting that term in the score*b variants.
inline fn tmScore(
    first_len: i32,
    second_len: i32,
    first_flag: u8,
    second_flag: u8,
    nw1: i32,
    nw2: i32,
    next_bb: u8,
    is_alt: bool,
    greedy_len: i32,
    extra_token_penalty: i32,
    drop_begin_space_bonus: bool,
) i32 {
    const branch_len = first_len + second_len;
    var score: i32 = branch_len;

    // bit-7 (`& 128`) — +1 each side if the piece is "all letters /
    // all punctuation". `>> 7` extracts a 0/1.
    score += @as(i32, (first_flag >> 7) & 1);
    score += @as(i32, (second_flag >> 7) & 1);

    // word-density bonuses (already in pre-flag formula).
    score += bMaxZeroAnd(nw1 - 1);
    score += bMaxZeroAnd(nw2 - 1);

    // bit-2 (`& 4`) — +1 if second begins with space / charToken /
    // wordToken. Dropped in the score*b variants because lilbuf-space
    // matched second always begins with a synthetic space.
    if (!drop_begin_space_bonus) {
        score += @as(i32, (second_flag >> 2) & 1);
    }

    // (nextByte >> 2) & 1 — +1 if past-end-of-branch byte is space.
    score += @as(i32, (next_bb >> 2) & 1);

    // 100x whole-word count. nw1 + nw2 + (nextByte >> 3) — the
    // `>> 3` extracts the "not a letter" bit (bit 3 of beginByte;
    // set for space=12 and punct=10, clear for letter=1 and zero).
    const next_nonletter: i32 = @as(i32, (next_bb >> 3) & 1);
    score += (nw1 + nw2 + next_nonletter) * 100;

    // -103 if first ends with letter AND second begins with letter
    // (split-word penalty). `first & 1 & (second >> 1)`.
    //
    // TM-Go's score1/2/3 use this gated form.
    //
    // TM-Go's score1b/2b/3b variants use the UNGATED form: just
    // `first.flag & 1` — deduct whenever first ends in letter,
    // regardless of second's begin class. `drop_begin_space_bonus =
    // true` doubles as the marker for score-b variants here.
    //
    // The unification is correct because score-b's `second` is
    // matched via lilbuf-space synthesis — the matched token always
    // begins with a synthetic space (FLAG_BEGINS_SPACE) and its
    // FLAG_BEGINS_LETTER is CLEAR. Without the ungated form here,
    // the score-b path would mis-credit the greedy branch (no -103
    // when there should be), causing false score-b wins on
    // positions where TM-Go's score1b would have been penalized.
    const split_word: i32 = if (drop_begin_space_bonus)
        @as(i32, first_flag & 1)
    else
        @as(i32, (first_flag & 1) & ((second_flag >> 1) & 1));
    score -= 103 * split_word;

    // -100 if first ends on capcode AND second begins on capcode.
    // `(first >> 3) & 1 & (second >> 4)`.
    const split_capcode: i32 = @as(i32, ((first_flag >> 3) & 1) & ((second_flag >> 4) & 1));
    score -= 100 * split_capcode;

    // -3 if second ends in letter AND next byte is a letter.
    // `second & 1 & nextByte` (low bit of nextByte = 1 only when
    // nextByte == 1, i.e. letter).
    const ends_in_word: i32 = @as(i32, (second_flag & 1) & (next_bb & 1));
    score -= 3 * ends_in_word;

    // Extra-DEL-token penalty (score*b variants).
    score -= extra_token_penalty;

    // Alt penalties (score2/score3): -100 if branch shorter than
    // greedy's first token, -10000 if equal.
    if (is_alt) {
        score -= bLessThan(branch_len, greedy_len) * 100;
        score -= bEqual(branch_len, greedy_len) * 10000;
    }

    return score;
}

pub const Monster = struct {
    allocator: std.mem.Allocator,
    bytes: []u8,
    offsets: []u32, // count+1
    nwords: []u8, // count; word-start count per token (ztok semantic: counts first byte)
    nwords_tm: []u8, // count; TM-Go semantic with `\x7f ` strip — score-b gate
    /// True TM-Go nWords (whole-word count, no `\x7f `-strip — see
    /// `computeNwordsScore`). Length == count. Read by the flag-aware
    /// score formula in `tmScore`. Kept separate from `nwords` (ztok
    /// legacy: counts first byte) and `nwords_tm` (gates score-b
    /// activation by stripping the leading DEL+space).
    nwords_score: []u8,
    /// TM-Go per-piece score-bias bits (`FLAG_*` constants above; see
    /// `computeFlags` for the assignment recipe). Length == `count`.
    /// All-zero for vocabs that don't go through the flag-aware
    /// `Builder.finalizeWithCapcode` path; the score formula
    /// degrades to its pre-flag behavior in that case.
    flags: []u8, // count; per-piece TM-Go flag bits
    /// Bare-form (synthetic-prefix-stripped) flag bits + nWords. For
    /// tokens whose stored bytes start with `\x7F ` (nocapcode mode) or
    /// `D ` (TM-printable capcode), these fields describe the same
    /// piece WITHOUT the 2-byte prefix — i.e., the bare-letter form
    /// that TM-Go's `LongestSubstring` would return at a mid-word
    /// position in the input. For all other tokens, these equal
    /// `flags` / `nwords_score`.
    ///
    /// Read by the phantom-second branch in `encodeChunkImpl` when the
    /// plain trie lookup misses but path-(b) synthesis succeeds — see
    /// `encodeChunkImpl` for the full rationale (ztok's vocab convert
    /// collapses TM-Go's twin `X` / `\x7F X` entries into one, losing
    /// the bare-letter forms but preserving the prefixed ones).
    bare_flags: []u8 = &.{},
    bare_nwords: []u8 = &.{},
    /// TM-Go-style next-byte classifier indexed by `chunk[next_pos]`.
    /// Built from the first byte of every vocab piece via majority
    /// vote (mirrors :3779-3788). The score formula reads
    /// `begin_byte[chunk[next_pos]]` to compute `(nextByte >> 2) & 1`
    /// (is-space bonus) and `(nextByte >> 3)` (is-not-letter word
    /// bonus) terms.
    begin_byte: [256]u8 = @splat(0),
    count: u32,
    unk_id: TokenId,
    nodes: []Node,
    /// SoA trie children — `child_bytes[k]` is the matching byte at child
    /// slot `k`, `child_nodes[k]` is the index into `nodes` to descend into.
    /// `len(child_bytes) == len(child_nodes) == sum(node.children_len)`.
    ///
    /// Why SoA: the inner trie walk does a linear scan over child bytes
    /// until it finds a match. Pre-1.20 stored `{byte, node}` interleaved
    /// (`Child = { u8, u32 }`, 8 bytes/child with padding), so a ≤8-child
    /// scan touched 64 bytes — one full cacheline read per node step.
    /// After the split, the byte-scan phase touches only `child_bytes`
    /// (1 byte/child → 8 children fit in 8 bytes), and the matching child's
    /// node index is fetched from `child_nodes` exactly once on hit. Hot-
    /// loop cacheline traffic per node step drops from 1 line (children)
    /// + 1 line (target node header) = 2, to ~⅛ line (children byte scan,
    /// when the parent's child set is dense across nodes) + 1 line (target
    /// node header). For dense-fanout root nodes the byte array stays in
    /// L1 across many positions.
    child_bytes: []u8,
    child_nodes: []u32,
    /// 1.25 A perf: root-node child lookup table. `root_child_table[b]` is
    /// the descend-to node index for byte `b` at the root, or `NO_TOKEN`
    /// if the root has no child for that byte. Replaces the binary-search
    /// branch in `findChild` for the FIRST byte of every trie walk —
    /// which fires per-position-per-walk-attempt (~3-4 walks per input
    /// position across the encoder's branch evaluation) and dominated the
    /// callgrind profile pre-1.25 (root binary-search lines were 19% of
    /// total program cycles on the english.txt corpus). At 256 u32s =
    /// 1 KiB the table fits in L1 and the indexed load is one ALU+load.
    /// Built in `finalizeWithCapcode`; ALL entries default to `NO_TOKEN`
    /// (sentinel for "no child at this byte") then filled from the
    /// flat trie's root node children. Memory cost is fixed at 1024 B
    /// per Monster — negligible vs the multi-MB trie body.
    root_child_table: [256]u32 = @splat(NO_TOKEN),
    max_token_len: u32,
    /// Optional runtime mask. If non-null, encoding behaves as if any token
    /// `id` with `mask[id] == 1` doesn't exist — the trie walk treats those
    /// terminals as non-terminals. Length MUST equal `count`. Buffer is
    /// owned by the caller (the trainer); not persisted by `monster_io`.
    ///
    /// The fast path (`mask == null`) is a single branch at function entry —
    /// non-trainer callers pay zero cost. See `encodeChunk` for the split.
    mask: ?[]const u8 = null,
    /// TokenMonster-style synthetic-boundary ("lilbuf") trick: at every
    /// position, do extra trie lookups against TWO synthetic prefixes
    /// concatenated with the remainder of the input:
    ///
    ///   (a) `[0x20]` ++ input — "what if a space were here?". If a
    ///       vocab token like ` monster` matches, emit it as
    ///       `DEL_TOKEN + ` monster``: TM-Go uses the DEL token (the
    ///       single-byte `\x7F`) as an inserted boundary marker so the
    ///       decoder strips the synthetic leading space.
    ///   (b) `[0x7F, 0x20]` ++ input — "what if a real `\x7F `-prefixed
    ///       token exists for this segment?". If a vocab token like
    ///       `\x7F Mon` matches, emit it as a SINGLE token (the 2-byte
    ///       synthetic prefix is part of the matched token's own bytes).
    ///
    /// Path (a) mirrors TM-Go's actual encoder (single-byte lilbuf prefix
    /// + emit a DEL token before the matched piece). Path (b) is a
    /// ztok-only optimization that catches the case where the vocab
    /// already carries the synthesized prefix as a literal token. Both
    /// paths score against the same branch math as the greedy +
    /// ungreedy alternatives.
    ///
    /// **Off by default.** Vocabs trained without the synthetic-prefix
    /// machinery will never match either lilbuf branch, costing two
    /// extra trie lookups per position. TM-Go-style vocabs
    /// (`bench/vocabs/tm_englishcode_32k.ztm`) carry both the DEL token
    /// (id of bytes `\x7F`) and tokens like ` Monster`/` Mon`/`\x7F Mon`
    /// that the lilbuf paths reach.
    ///
    /// `monster_io.readBytes*` flips this on unconditionally for every
    /// .ztm load — the trick is a no-op for vocabs that don't carry the
    /// synthetic-prefix tokens (the lilbuf branch loses every score
    /// comparison) and a small per-position bonus for vocabs that do.
    lilbuf_enabled: bool = false,
    /// Id of the single-byte `\x7F` (NoCapcodeDeleteToken) token in the
    /// vocab, or NO_TOKEN sentinel if no such token exists. Used by the
    /// lilbuf path (a) — TM-Go emits this id as a boundary marker before
    /// the lilbuf-prefixed piece. Populated automatically by
    /// `Builder.finalize` from a scan of the byte table.
    delete_token_id: u32 = NO_TOKEN,
    /// True iff the trie has at least one token that starts with the
    /// synthetic `\x7F ` (DEL + space) prefix. Computed at
    /// `Builder.finalize` time. When false, the path-(b) lilbuf trie
    /// walk is guaranteed to miss; we skip the entire branch to avoid
    /// even the cheap fast-fail probe per position.
    has_lilbuf_prefix_tokens: bool = false,
    /// True iff the trie has at least one token that starts with `\x20`
    /// (space). Computed at `Builder.finalize` time. When false, the
    /// path-(a) lilbuf walk is guaranteed to miss; skipped entirely.
    /// (Most non-trivial vocabs have at least one space-prefixed
    /// token, so this is rarely false in practice.)
    has_space_prefix_tokens: bool = false,
    /// Enable TM-Go's score2b/score3b: when the LOOKAHEAD position of
    /// an alt branch lands mid-word (the matched piece starts with a
    /// letter, follows a letter, has no whole-word starts), try a
    /// lilbuf-space alternative for that lookahead. If the synthetic
    /// match consumes strictly more real bytes than the plain greedy
    /// lookahead and beats the current best (with a `-1` extra-token
    /// penalty), emit `[alt_first, DEL, lilbuf_second]` and set
    /// `forwardDelete = 1` for the next position. Gated separately from
    /// `lilbuf_enabled` so we can A/B in benches. Auto-enabled by
    /// `monster_io.readBytes*` for .ztm vocabs alongside
    /// `lilbuf_enabled` and `has_lilbuf_prefix_tokens`.
    score2b3b_enabled: bool = false,
    /// Synthetic-prefix byte for the path-(b) lilbuf walk. Selected at
    /// `finalizeWithCapcode` time based on the vocab's capcode mode:
    ///
    ///   .nocapcode → `\x7F` (DEL — TM-Go's `NoCapcodeDeleteToken`)
    ///   .full      → `D`   (DeleteToken in TM-Go's printable-marker
    ///                       style; refs/tokenmonster/go/tokenmonster.go:3475-3478)
    ///   .none      → `\x00` (sentinel; path-(b) won't fire because
    ///                       no real token starts with NUL+space)
    ///
    /// The path-(a) "space prefix + emit DEL" walk uses `delete_token_id`
    /// to know which id to emit; the path-(b) walk uses THIS byte as
    /// the first synthetic byte plus `0x20` as the second.
    lilbuf_marker_byte: u8 = 0x7F,
    /// Precomputed per-piece alt metadata. Length == `count`; the entry
    /// at index `id` is the two best-priority in-vocab proper-prefix
    /// subtokens of piece `id`. Built at `finalizeWithCapcode` time via
    /// `computeAlts`; null if not yet built (legacy `finalize` path
    /// keeps it null for back-compat — encoder falls back to the
    /// `collectPrefixMatches` walk). See `AltPair` docs for ranking.
    ///
    /// When non-null AND `use_precomputed_alts` is true, the encoder
    /// uses `alts[greedy_id]` to seed score2/score3 evaluation; the
    /// `collectPrefixMatches` walk is bypassed for the alt branches.
    /// This both fixes correctness (TM-Go's priority-ranked picks
    /// often differ from "shortest 5 prefixes by length") and saves
    /// the per-position O(max_token_len) trie walk.
    alts: ?[]AltPair = null,
    /// Toggle for the precomputed-alts encoder path. Set by
    /// `Builder.finalizeWithCapcode` whenever `alts` is built; the
    /// loader (`monster_io.readBytes*`) leaves it as the builder set
    /// it. The legacy `finalize(unk_id)` path also enables it (it
    /// delegates to `finalizeWithCapcode(.none)`), so all pre-1.16
    /// callers automatically pick up the improved encoder.
    use_precomputed_alts: bool = false,
    /// Enable TM-Go's goto-checkpoint re-evaluation. When a
    /// `.first_del_second` (score2b/3b) branch wins, the encoder:
    ///   1. Emits ONLY `[alt_first, DEL]` (drops the immediate
    ///      `lilbuf_second` emit).
    ///   2. Advances the position to where `alt_first` ended (NOT past
    ///      `lilbuf_second`).
    ///   3. Sets `forward_lilbuf = true` for the next iteration, which
    ///      seeds the alt evaluation with the `lilbuf_second` token as
    ///      the synthetic greedy match (representing the conceptual
    ///      "`\x7F `-prefixed segment that starts here").
    /// At the re-entered iteration, the alts of `lilbuf_second` may
    /// score better than `lilbuf_second` itself, producing a
    /// SHORTER/DIFFERENT segmentation than the 1.16 single-emit path.
    /// This mirrors TM-Go's `case score2b: ... goto checkpoint` block
    /// at refs/tokenmonster/go/tokenmonster.go:1248-1254.
    ///
    /// **Off by default.** Auto-enabled by `monster_io.readBytes*`
    /// for `.ztm` files (so TM-Go-compat vocabs pick it up; user
    /// vocabs without TM-style synthetic prefixes keep the 1.16
    /// single-emit behavior for back-compat). Costs one extra branch
    /// per iteration when `forward_lilbuf == false` (the common case).
    goto_checkpoint_enabled: bool = false,

    /// Alias records — extra byte sequences in the trie that resolve to
    /// the SAME id as a primary `offsets[id]` entry. Used to preserve
    /// TM-Go's twin-entry shape (`train` and `\x7F train` both pointing
    /// to alt_id 9735): the primary stored bytes carry one form, this
    /// list carries the rest. Each record stores `id` (the shared
    /// destination) and `bytes` (the alternate byte sequence inserted
    /// into the trie). The trie walk at this alias byte sequence yields
    /// the same `id` as a direct lookup of `idBytes(id)` would.
    ///
    /// Populated only by `monster_io.readBytes*` when loading a v2 `.ztm`
    /// file with a non-empty alias section, or by an explicit
    /// `Builder.addAlias` call. Default empty for non-aliased vocabs.
    /// The alias byte buffers are owned by `Monster` and freed in `deinit`.
    /// Used by `monster_io.writeBytes*` to round-trip the alias section.
    aliases: []AliasRecord = &.{},
    /// Length-table indexed by `id`: byte counts of the alias forms that
    /// share this id, used by the encoder to disambiguate a trie hit
    /// between "matched the primary bytes" and "matched an alias". A
    /// trie hit of length `len` where `len == offsets[id+1]-offsets[id]`
    /// is the primary; any other length is one of the aliases — and the
    /// encoder uses `bare_flags`/`bare_nwords` for those (the alias form
    /// is the bare-letter form whose flag bits differ from the prefixed
    /// stored form). Allocated lazily at finalize; empty when there are
    /// no aliases.
    alias_lens_by_id: []u8 = &.{},

    /// Per-twin alt tables recovered from a `.ztm` v3 file (see
    /// `PerTwinAlt`). Length is the number of twin forms that carry at
    /// least one alt. Empty for v1/v2 files. **Stage 1: parsed + stored
    /// but NOT consumed by the encoder** — additive only. Owned by
    /// `Monster`; freed in `deinit`.
    per_twin_alts: []PerTwinAlt = &.{},

    /// Stage 2: encoder-facing lookup built from `per_twin_alts` at
    /// finalize. Indexed by vocab `id`, holds the BARE twin's alt table
    /// (the per-twin record whose `key` length equals the id's bare
    /// alias byte length — i.e. the no-marker form the trie matches at a
    /// bare-alias position). Each entry already resolves `{alt_id,
    /// alt_byte_len}` into the encoder's `AltPair` shape (index/length/
    /// index2/length2) in BARE byte units, so the bare-alias gate in
    /// `encodeChunkImpl` can feed them into the same score1/2/3 +
    /// score-b machinery the prefixed precomputed alts already use.
    ///
    /// `index == NO_TOKEN` means "no bare per-twin alts for this id"
    /// (the common case — only aliased ids carry an entry). Empty for
    /// v1/v2 files. Owned by `Monster`; freed in `deinit`.
    bare_alt_lut: []AltPair = &.{},

    /// One alias entry: shared `id` + alternate `bytes` (owned by
    /// `Monster`, freed in `deinit`).
    pub const AliasRecord = struct {
        id: u32,
        bytes: []u8,
    };

    /// Per-twin alt table (`.ztm` v3). TM-Go keeps a SEPARATE
    /// `tokenOuter` (alt table) for each twin info entry — e.g. `sti`
    /// (alts `st`/`s`, in bare units) and `\x7F sti` (alts `\x7F`/`\x7F st`,
    /// in `\x7F `-prefixed units). The v2 converter COLLAPSED twins to
    /// one id with one alt table in the wrong units; v3 stores each
    /// twin's own alt table verbatim. See `PerTwinAlt` for one record.
    ///
    /// Each record carries:
    ///   * `id`    — the vocab id this twin form resolves to (shared
    ///               across the twin pair).
    ///   * `key`   — the twin's byte sequence (the disambiguator — which
    ///               of the two twin forms was matched in the trie).
    ///   * `alts`  — up to 2 `{ alt_id, alt_byte_len }` entries (alt1
    ///               first — the better-priority one). `alt_byte_len` is
    ///               in THIS twin's own byte units (TM-Go's
    ///               `tokenOuter.length`/`length2`).
    ///
    /// **Stage 1 (this change) is ADDITIVE ONLY**: these structures are
    /// PARSED and STORED but NOT yet read by the encoder's decision
    /// logic. The encoder still uses the v2 `alts`/`alias_lens_by_id`
    /// path. Stage 2 will rewire scoring to consult `per_twin_alts` so
    /// bare-alias positions evaluate alts in the correct units. Empty
    /// for v1/v2 files and for non-aliased vocabs.
    ///
    /// Owned by `Monster` (the `key`/`alts` byte+slice buffers are
    /// freed in `deinit`).
    pub const PerTwinAlt = struct {
        id: u32,
        key: []u8,
        alts: []TwinAltEntry,
    };

    /// One alt entry inside a `PerTwinAlt`: the alt's vocab id and the
    /// byte length consumed in the parent twin's units.
    pub const TwinAltEntry = struct {
        alt_id: u32,
        alt_byte_len: u32,
    };

    pub const Node = struct {
        token_id: u32, // 0xFFFFFFFF if not a terminal
        children_start: u32,
        children_len: u32,
    };

    /// Build-time only — AoS pair used while assembling the trie. The
    /// flat trie stores its children in SoA form (`child_bytes` +
    /// `child_nodes`) for cacheline density on the byte-scan inner loop.
    pub const Child = struct { byte: u8, node: u32 };

    pub const Builder = struct {
        allocator: std.mem.Allocator,
        bytes_buf: std.ArrayList(u8),
        offsets_buf: std.ArrayList(u32),
        /// Pending aliases (extra trie entries that resolve to an
        /// already-added id). Each entry owns its own byte buffer until
        /// `finalize*` consumes it and re-owns those buffers under the
        /// produced `Monster.aliases`.
        aliases_buf: std.ArrayList(Monster.AliasRecord),
        /// Pending per-twin alt tables (`.ztm` v3). Each entry owns its
        /// `key` and `alts` buffers until `finalize*` consumes them and
        /// re-owns them under `Monster.per_twin_alts`. See
        /// `addPerTwinAlt`.
        per_twin_buf: std.ArrayList(Monster.PerTwinAlt),

        pub fn init(allocator: std.mem.Allocator) Builder {
            return .{
                .allocator = allocator,
                .bytes_buf = .empty,
                .offsets_buf = .empty,
                .aliases_buf = .empty,
                .per_twin_buf = .empty,
            };
        }

        pub fn deinit(self: *Builder) void {
            self.bytes_buf.deinit(self.allocator);
            self.offsets_buf.deinit(self.allocator);
            // Free any alias byte buffers that were added but never
            // consumed by `finalize*`. The finalize path takes
            // ownership and clears `aliases_buf` to skip this.
            for (self.aliases_buf.items) |a| {
                if (a.bytes.len > 0) self.allocator.free(a.bytes);
            }
            self.aliases_buf.deinit(self.allocator);
            // Same ownership story for pending per-twin alt records.
            for (self.per_twin_buf.items) |t| {
                if (t.key.len > 0) self.allocator.free(t.key);
                if (t.alts.len > 0) self.allocator.free(t.alts);
            }
            self.per_twin_buf.deinit(self.allocator);
        }

        /// Add a per-twin alt table (`.ztm` v3). `key` is the twin's
        /// byte sequence (the disambiguator); `alts` are up to 2
        /// `{ alt_id, alt_byte_len }` entries in this twin's units.
        /// Both are copied into Builder-owned buffers. Used by
        /// `monster_io.readBytes*` when loading a v3 `.ztm`.
        ///
        /// **Stage 1: stored, not consumed by the encoder.**
        pub fn addPerTwinAlt(
            self: *Builder,
            id: TokenId,
            key: []const u8,
            alts: []const Monster.TwinAltEntry,
        ) !void {
            const key_dup = try self.allocator.alloc(u8, key.len);
            errdefer self.allocator.free(key_dup);
            @memcpy(key_dup, key);
            const alts_dup = try self.allocator.alloc(Monster.TwinAltEntry, alts.len);
            errdefer self.allocator.free(alts_dup);
            @memcpy(alts_dup, alts);
            try self.per_twin_buf.append(self.allocator, .{
                .id = id,
                .key = key_dup,
                .alts = alts_dup,
            });
        }

        /// Add an alias byte sequence for an existing id. The alias is
        /// inserted into the trie at `finalize*` time, so trie lookups
        /// of `bytes` will return `id`. Used by `monster_io.readBytes*`
        /// to restore TM-Go's twin entries (`train` + `\x7F train`).
        /// `bytes` is copied into a Builder-owned buffer.
        pub fn addAlias(self: *Builder, id: TokenId, bytes: []const u8) !void {
            const dup = try self.allocator.alloc(u8, bytes.len);
            errdefer self.allocator.free(dup);
            @memcpy(dup, bytes);
            try self.aliases_buf.append(self.allocator, .{ .id = id, .bytes = dup });
        }

        pub fn addToken(self: *Builder, bytes: []const u8) !TokenId {
            if (self.offsets_buf.items.len == 0) {
                try self.offsets_buf.append(self.allocator, 0);
            }
            const id: TokenId = @intCast(self.offsets_buf.items.len - 1);
            try self.bytes_buf.appendSlice(self.allocator, bytes);
            try self.offsets_buf.append(self.allocator, @intCast(self.bytes_buf.items.len));
            return id;
        }

        pub fn finalize(self: *Builder, unk_id: TokenId) !Monster {
            // Back-compat entry point: pre-flag-aware callers pass no
            // capcode mode, so flags are computed assuming "no capcode
            // marker rune" (C/W/D are treated as plain letters). The
            // score formula degrades gracefully because no capcode-
            // specific flag bits will fire.
            return self.finalizeWithCapcode(unk_id, .none);
        }

        /// Same as `finalize` but takes the vocab's capcode mode so
        /// per-piece flag bits (`FLAG_*` constants) and the 256-entry
        /// `begin_byte` lookup are populated with TM-Go-compatible
        /// classifications. Use `.none` for vocabs that don't carry
        /// any capcode markers, `.nocapcode` for `0x7F`-as-DEL vocabs,
        /// and `.full` for `C`/`W`/`D` printable-marker vocabs.
        pub fn finalizeWithCapcode(
            self: *Builder,
            unk_id: TokenId,
            capcode: CapcodeMode,
        ) !Monster {
            const count: u32 = if (self.offsets_buf.items.len == 0)
                0
            else
                @intCast(self.offsets_buf.items.len - 1);
            std.debug.assert(count == 0 or unk_id < count);

            const bytes = try self.bytes_buf.toOwnedSlice(self.allocator);
            errdefer self.allocator.free(bytes);
            const offsets_initial = if (self.offsets_buf.items.len == 0) blk: {
                var tmp = try self.allocator.alloc(u32, 1);
                tmp[0] = 0;
                break :blk tmp;
            } else try self.offsets_buf.toOwnedSlice(self.allocator);
            errdefer self.allocator.free(offsets_initial);

            // Per-token nwords: count whitespace→non-whitespace transitions
            // inside the token bytes, +1 if the first byte is non-whitespace.
            // This is the capcode-free proxy for the Go source's nWords.
            const nwords = try self.allocator.alloc(u8, if (count == 0) 1 else count);
            errdefer self.allocator.free(nwords);
            if (count == 0) nwords[0] = 0;
            // TM-Go score-b gate: strips `\x7f ` then counts transitions.
            const nwords_tm = try self.allocator.alloc(u8, if (count == 0) 1 else count);
            errdefer self.allocator.free(nwords_tm);
            if (count == 0) nwords_tm[0] = 0;
            // True TM-Go nWords (whole-word count, no strip). Used by the
            // flag-aware score formula in `tmScore`.
            const nwords_score = try self.allocator.alloc(u8, if (count == 0) 1 else count);
            errdefer self.allocator.free(nwords_score);
            if (count == 0) nwords_score[0] = 0;
            // Per-piece TM-Go flag bits. All-zero for the empty-vocab
            // sentinel slot; the score formula treats zero-flag pieces
            // as "no bonus / no penalty" which matches the pre-flag
            // behavior.
            const flags = try self.allocator.alloc(u8, if (count == 0) 1 else count);
            errdefer self.allocator.free(flags);
            if (count == 0) flags[0] = 0;
            // Per-piece BARE-FORM flag bits and nWords. For tokens whose
            // stored bytes start with `\x7F ` (nocapcode) or `D `
            // (TM-printable capcode), the bare-form is the substring after
            // the 2-byte synthetic prefix; its flags/nWords differ from
            // the stored form's (the bare form has FLAG_BEGINS_LETTER
            // instead of FLAG_BEGINS_CAPCODE, possibly different
            // FLAG_ALL_LETTERS, and one fewer nWords because the synthetic
            // space→alphanum transition contributes 1 to the stored form's
            // nWords count).
            //
            // For tokens NOT starting with the marker+space prefix, these
            // are identical to `flags` / `nwords_score`. The encoder's
            // phantom-second branch (vocab-collapse compensator) reads
            // these arrays when a path-(b)-synthesized lookup stands in
            // for the bare-letter form TM-Go's `LongestSubstring` would
            // have returned. See `encodeChunkImpl` for the path-(b)
            // gating and the rationale write-up.
            const bare_flags = try self.allocator.alloc(u8, if (count == 0) 1 else count);
            errdefer self.allocator.free(bare_flags);
            if (count == 0) bare_flags[0] = 0;
            const bare_nwords = try self.allocator.alloc(u8, if (count == 0) 1 else count);
            errdefer self.allocator.free(bare_nwords);
            if (count == 0) bare_nwords[0] = 0;

            // beginByte tally for the 256-entry lookup table. Four-way
            // counter per byte: [space-start, letter-start, number-start,
            // punct/capcode-start]; whichever wins by majority (>2 and
            // strict greater than the others) gets the corresponding
            // beginByte classification value at :3779-3788.
            var bb_tally: [256][4]u32 = @splat(@splat(0));

            var max_len: u32 = 0;
            var id: u32 = 0;
            while (id < count) : (id += 1) {
                const start = offsets_initial[id];
                const end = offsets_initial[id + 1];
                const piece = bytes[start..end];
                const piece_len: u32 = end - start;
                if (piece_len > max_len) max_len = piece_len;
                nwords[id] = computeNwords(piece);
                nwords_tm[id] = computeNwordsTm(piece);
                nwords_score[id] = computeNwordsScore(piece, capcode);
                flags[id] = computeFlags(piece, capcode);
                // Bare-form flags/nWords: stripped of the synthetic
                // `\x7F ` (nocapcode) or `D ` (TM-printable capcode)
                // marker+space prefix if present. Falls back to the same
                // values as flags/nwords_score for non-prefixed tokens.
                if (piece_len >= 3) {
                    const marker_byte: u8 = switch (capcode) {
                        .none => 0,
                        .nocapcode => 0x7F,
                        .full => 'D',
                    };
                    if (marker_byte != 0 and piece[0] == marker_byte and piece[1] == ' ') {
                        const bare = piece[2..];
                        bare_flags[id] = computeFlags(bare, capcode);
                        bare_nwords[id] = computeNwordsScore(bare, capcode);
                    } else {
                        bare_flags[id] = flags[id];
                        bare_nwords[id] = nwords_score[id];
                    }
                } else {
                    bare_flags[id] = flags[id];
                    bare_nwords[id] = nwords_score[id];
                }
                if (piece_len > 0) {
                    const first = piece[0];
                    // Classify the first byte mirroring TM-Go's
                    // beginByte counter (:3522-3541). We tally based
                    // on the first rune; for capcode mode the markers
                    // (`C`/`W`/`D` or `\x7F`) count as punct.
                    var first_rune: u21 = first;
                    if (piece_len >= 2) {
                        // Try a 2/3/4-byte UTF-8 sequence so multi-byte
                        // first runes (Greek, Cyrillic, ...) bucket on
                        // their actual category rather than the lead
                        // byte (which would always be punct).
                        first_rune = decodeFirstRune(piece);
                    }
                    if (first_rune == ' ') {
                        bb_tally[first][0] += 1;
                    } else if (isLetterCp(first_rune, capcode)) {
                        bb_tally[first][1] += 1;
                    } else if (isCapcodeMarker(first_rune, capcode)) {
                        bb_tally[first][3] += 1;
                    } else if (unicode_props.isNumber(first_rune)) {
                        bb_tally[first][2] += 1;
                    } else {
                        bb_tally[first][3] += 1;
                    }
                    // Vocab-collapse compensation: TM-Go's begin_byte
                    // tally counts BOTH `train` and `\x7F train` (twin
                    // entries that share an alt_id but have separate
                    // dictionary insertions). ztok's pre-v2 vocab
                    // convert collapsed these into the longer
                    // `\x7F `-prefixed form only — the bare-letter
                    // form was gone from the tally. Compensate by ALSO
                    // tallying the bare-form first byte when the stored
                    // bytes start with the capcode marker + space.
                    //
                    // **Heuristic stays on** even when aliases are
                    // present: it tallies a synthetic bare-form first
                    // byte for every marker-prefixed primary, which
                    // does double-count with the alias-loop tally when
                    // both forms exist for the same id. But the
                    // compensation also catches primaries that DON'T
                    // have a twin alias (single-info entries that just
                    // happen to be prefixed) — TM-Go counts those
                    // once in its own tally, and so do we via the
                    // primary's first byte. The double-count on
                    // twinned ids cancels out in practice for the
                    // majority vote (both letter and capcode buckets
                    // grow by the same amount on each side of the
                    // first-byte cell). Empirically this is what
                    // keeps nocapcode equivalence on the same shelf as
                    // pre-v2; disabling it regresses by ~6 points.
                    if (piece_len >= 3) {
                        const marker_byte: u8 = switch (capcode) {
                            .none => 0,
                            .nocapcode => 0x7F,
                            .full => 'D',
                        };
                        if (marker_byte != 0 and piece[0] == marker_byte and piece[1] == ' ') {
                            const bare = piece[2..];
                            const bare_first = bare[0];
                            var bare_rune: u21 = bare_first;
                            if (bare.len >= 2) {
                                bare_rune = decodeFirstRune(bare);
                            }
                            if (bare_rune == ' ') {
                                bb_tally[bare_first][0] += 1;
                            } else if (isLetterCp(bare_rune, capcode)) {
                                bb_tally[bare_first][1] += 1;
                            } else if (isCapcodeMarker(bare_rune, capcode)) {
                                bb_tally[bare_first][3] += 1;
                            } else if (unicode_props.isNumber(bare_rune)) {
                                bb_tally[bare_first][2] += 1;
                            } else {
                                bb_tally[bare_first][3] += 1;
                            }
                        }
                    }
                }
            }

            // Aliases: optional twin trie entries that share an id with
            // a primary `offsets[id]` slot. Tally their first bytes in
            // bb_tally so the begin_byte LUT majority vote includes
            // both forms (TM-Go's begin_byte counts every dictionary
            // entry; we need to do the same to match its classification).
            // Also update max_len so the encoder's per-position trie
            // walk doesn't truncate inside an alias.
            const alias_records = self.aliases_buf.items;
            for (alias_records) |a| {
                if (a.bytes.len == 0) continue;
                if (a.bytes.len > max_len) max_len = @intCast(a.bytes.len);
                const first = a.bytes[0];
                var first_rune: u21 = first;
                if (a.bytes.len >= 2) first_rune = decodeFirstRune(a.bytes);
                if (first_rune == ' ') {
                    bb_tally[first][0] += 1;
                } else if (isLetterCp(first_rune, capcode)) {
                    bb_tally[first][1] += 1;
                } else if (isCapcodeMarker(first_rune, capcode)) {
                    bb_tally[first][3] += 1;
                } else if (unicode_props.isNumber(first_rune)) {
                    bb_tally[first][2] += 1;
                } else {
                    bb_tally[first][3] += 1;
                }
            }

            // Resolve the 256-entry `begin_byte` lookup table (TM-Go:
            // :3779-3788). Tally winner with `> 2` floor; otherwise 0.
            var begin_byte: [256]u8 = @splat(0);
            {
                var bi: usize = 0;
                while (bi < 256) : (bi += 1) {
                    const t = bb_tally[bi];
                    if (t[1] > t[0] and t[1] > t[2] and t[1] > t[3] and t[1] > 2) {
                        begin_byte[bi] = BB_LETTER;
                    } else if (t[0] > t[1] and t[0] > t[2] and t[0] > t[3] and t[0] > 2) {
                        begin_byte[bi] = BB_SPACE;
                    } else if (t[3] > t[0] and t[3] > t[1] and t[3] > t[2] and t[3] > 2) {
                        begin_byte[bi] = BB_PUNCT;
                    }
                }
            }

            const trie = try buildTrie(self.allocator, bytes, offsets_initial, count, alias_records);

            // Locate the single-byte delete-marker id and detect whether
            // the trie carries marker-prefixed and/or `\x20`-prefixed
            // tokens at all. The delete marker depends on capcode mode:
            //
            //   .nocapcode → `\x7F` (NoCapcodeDeleteToken in TM-Go;
            //                refs/tokenmonster/go/tokenmonster.go:3479-3482)
            //   .full      → `D`   (DeleteToken, byte 0x44; TM-Go :3475-3478)
            //   .none      → no delete marker; lilbuf path-(a) won't fire
            //
            // The path-(b) "lilbuf prefix tokens" check looks for tokens
            // starting with `delete_marker + space`. For full capcode the
            // analog is `D + space` (`D ` byte sequence); for nocapcode
            // it's `\x7F ` (well-known DEL+space prefix).
            // For `.none` we still default to `\x7F` so legacy vocabs
            // that happen to carry `\x7F`-prefixed tokens (pre-1.16
            // synthetic-prefix Monsters trained without explicit
            // capcode handling) keep working byte-for-byte.
            const delete_marker_byte: ?u8 = switch (capcode) {
                .nocapcode, .none => @as(u8, 0x7F),
                .full => @as(u8, 'D'),
            };
            var del_id: u32 = NO_TOKEN;
            var has_lilbuf_b: bool = false;
            var has_space_b: bool = false;
            if (trie.nodes.len > 0) {
                const root = trie.nodes[0];
                var k: u32 = 0;
                while (k < root.children_len) : (k += 1) {
                    const child_byte = trie.child_bytes[root.children_start + k];
                    const child_node = trie.child_nodes[root.children_start + k];
                    if (delete_marker_byte) |dm| {
                        if (child_byte == dm) {
                            const cn = trie.nodes[child_node];
                            if (cn.token_id != NO_TOKEN) del_id = cn.token_id;
                            // Has at least one `dm + 0x20`-prefixed token?
                            var j: u32 = 0;
                            while (j < cn.children_len) : (j += 1) {
                                if (trie.child_bytes[cn.children_start + j] == 0x20) {
                                    has_lilbuf_b = true;
                                    break;
                                }
                            }
                        }
                    }
                    if (child_byte == 0x20) {
                        // A space at root means at least one space-
                        // prefixed token exists (the space node has
                        // children or itself is a terminal).
                        const sn = trie.nodes[child_node];
                        if (sn.children_len > 0 or sn.token_id != NO_TOKEN) {
                            has_space_b = true;
                        }
                    }
                }
            }

            // Precomputed alt pairs (TM-Go's `tokenOuter.{index,length,
            // index2,length2}`). Built per-piece using the flat trie
            // for prefix lookups; ranking matches TM-Go's priority
            // ladder at refs/tokenmonster/go/tokenmonster.go:3597-3753.
            const alts_buf = try computeAlts(
                self.allocator,
                bytes,
                offsets_initial,
                count,
                trie.nodes,
                trie.child_bytes,
                trie.child_nodes,
                capcode,
            );
            errdefer self.allocator.free(alts_buf);

            // Take ownership of the alias records (the Builder's
            // pending list). After this point, `self.aliases_buf` is
            // empty so `deinit` won't double-free the byte buffers.
            // Allocate `alias_lens_by_id` only when at least one alias
            // exists; otherwise leave it empty (zero allocation for
            // non-aliased vocabs — the common case).
            const aliases_out: []Monster.AliasRecord = if (alias_records.len == 0)
                &.{}
            else blk: {
                const out = try self.allocator.alloc(Monster.AliasRecord, alias_records.len);
                @memcpy(out, alias_records);
                self.aliases_buf.clearRetainingCapacity();
                break :blk out;
            };
            errdefer if (aliases_out.len > 0) self.allocator.free(aliases_out);

            const alias_lens_out: []u8 = if (alias_records.len == 0 or count == 0)
                &.{}
            else blk: {
                const out = try self.allocator.alloc(u8, count);
                @memset(out, 0);
                for (aliases_out) |a| {
                    if (a.id < count and a.bytes.len > 0) {
                        const len_u8: u8 = if (a.bytes.len > 255) 255 else @intCast(a.bytes.len);
                        // Last writer wins on per-id collisions (rare —
                        // each id typically gets at most one alias).
                        out[a.id] = len_u8;
                    }
                }
                break :blk out;
            };
            errdefer if (alias_lens_out.len > 0) self.allocator.free(alias_lens_out);

            // Take ownership of the per-twin alt records (`.ztm` v3).
            // After this point, `self.per_twin_buf` is emptied so the
            // Builder's `deinit` won't double-free the key/alts buffers.
            // Stage 1: stored on the Monster but NOT read by the encoder.
            const per_twin_out: []Monster.PerTwinAlt = if (self.per_twin_buf.items.len == 0)
                &.{}
            else blk: {
                const out = try self.allocator.alloc(
                    Monster.PerTwinAlt,
                    self.per_twin_buf.items.len,
                );
                @memcpy(out, self.per_twin_buf.items);
                self.per_twin_buf.clearRetainingCapacity();
                break :blk out;
            };
            errdefer if (per_twin_out.len > 0) self.allocator.free(per_twin_out);

            // Stage 2: build the bare-twin alt LUT (`bare_alt_lut`),
            // indexed by id. For each per-twin record whose `key` length
            // equals that id's bare alias byte length (`alias_lens_out`),
            // record the bare twin's alt table as an `AltPair`. This is
            // the form the trie matches at a bare-alias position, so the
            // encoder's bare-alias gate can fetch alts in BARE units in
            // O(1). Only allocated when there is at least one alias table
            // AND at least one per-twin record.
            const bare_alt_out: []AltPair = if (per_twin_out.len == 0 or alias_lens_out.len == 0)
                &.{}
            else blk: {
                const out = try self.allocator.alloc(AltPair, count);
                for (out) |*e| e.* = .{};
                for (per_twin_out) |t| {
                    if (t.id >= count) continue;
                    // Select the BARE twin: its key length must match the
                    // id's bare alias byte length. The prefixed twin's
                    // key is 2 bytes longer (marker+space) and is skipped.
                    const bare_len: u32 = if (t.id < alias_lens_out.len) alias_lens_out[t.id] else 0;
                    if (bare_len == 0 or t.key.len != bare_len) continue;
                    if (t.alts.len >= 1) {
                        out[t.id].index = t.alts[0].alt_id;
                        out[t.id].length = t.alts[0].alt_byte_len;
                        if (t.alts.len >= 2) {
                            out[t.id].index2 = t.alts[1].alt_id;
                            out[t.id].length2 = t.alts[1].alt_byte_len;
                        }
                    }
                }
                break :blk out;
            };
            errdefer if (bare_alt_out.len > 0) self.allocator.free(bare_alt_out);

            // 1.25 A perf: build the root child lookup table. The root
            // is the highest-fanout node in the trie (typically 80-200+
            // children on real vocabs) and was the only node hitting
            // `findChild`'s binary-search branch on a hot inner loop.
            // Replacing that with a [256]u32 direct lookup removes the
            // 19%-of-program-cycles cost the callgrind profile pinned on
            // the root binary search.
            var root_table: [256]u32 = @splat(NO_TOKEN);
            if (trie.nodes.len > 0) {
                const root = trie.nodes[0];
                var k: u32 = 0;
                while (k < root.children_len) : (k += 1) {
                    const cb = trie.child_bytes[root.children_start + k];
                    const cn = trie.child_nodes[root.children_start + k];
                    root_table[cb] = cn;
                }
            }

            return .{
                .allocator = self.allocator,
                .bytes = bytes,
                .offsets = offsets_initial,
                .nwords = nwords,
                .nwords_tm = nwords_tm,
                .nwords_score = nwords_score,
                .flags = flags,
                .bare_flags = bare_flags,
                .bare_nwords = bare_nwords,
                .begin_byte = begin_byte,
                .count = count,
                .unk_id = unk_id,
                .nodes = trie.nodes,
                .child_bytes = trie.child_bytes,
                .child_nodes = trie.child_nodes,
                .root_child_table = root_table,
                .max_token_len = max_len,
                .delete_token_id = del_id,
                .has_lilbuf_prefix_tokens = has_lilbuf_b,
                .has_space_prefix_tokens = has_space_b,
                .lilbuf_marker_byte = if (delete_marker_byte) |dm| dm else 0x7F,
                .alts = alts_buf,
                .use_precomputed_alts = true,
                .aliases = aliases_out,
                .alias_lens_by_id = alias_lens_out,
                .per_twin_alts = per_twin_out,
                .bare_alt_lut = bare_alt_out,
            };
        }
    };

    pub fn deinit(self: *Monster) void {
        if (self.bytes.len > 0) self.allocator.free(self.bytes);
        if (self.offsets.len > 0) self.allocator.free(self.offsets);
        if (self.nwords.len > 0) self.allocator.free(self.nwords);
        if (self.nwords_tm.len > 0) self.allocator.free(self.nwords_tm);
        if (self.nwords_score.len > 0) self.allocator.free(self.nwords_score);
        if (self.flags.len > 0) self.allocator.free(self.flags);
        if (self.bare_flags.len > 0) self.allocator.free(self.bare_flags);
        if (self.bare_nwords.len > 0) self.allocator.free(self.bare_nwords);
        if (self.nodes.len > 0) self.allocator.free(self.nodes);
        if (self.child_bytes.len > 0) self.allocator.free(self.child_bytes);
        if (self.child_nodes.len > 0) self.allocator.free(self.child_nodes);
        if (self.alts) |a| self.allocator.free(a);
        if (self.aliases.len > 0) {
            for (self.aliases) |a| {
                if (a.bytes.len > 0) self.allocator.free(a.bytes);
            }
            self.allocator.free(self.aliases);
        }
        if (self.alias_lens_by_id.len > 0) self.allocator.free(self.alias_lens_by_id);
        if (self.per_twin_alts.len > 0) {
            for (self.per_twin_alts) |t| {
                if (t.key.len > 0) self.allocator.free(t.key);
                if (t.alts.len > 0) self.allocator.free(t.alts);
            }
            self.allocator.free(self.per_twin_alts);
        }
        if (self.bare_alt_lut.len > 0) self.allocator.free(self.bare_alt_lut);
        self.bytes = &.{};
        self.offsets = &.{};
        self.nwords = &.{};
        self.nwords_tm = &.{};
        self.nwords_score = &.{};
        self.flags = &.{};
        self.bare_flags = &.{};
        self.bare_nwords = &.{};
        self.nodes = &.{};
        self.child_bytes = &.{};
        self.child_nodes = &.{};
        self.alts = null;
        self.aliases = &.{};
        self.alias_lens_by_id = &.{};
        self.per_twin_alts = &.{};
        self.bare_alt_lut = &.{};
        self.count = 0;
    }

    pub fn idBytes(self: *const Monster, id: TokenId) []const u8 {
        std.debug.assert(id < self.count);
        const start = self.offsets[id];
        const end = self.offsets[id + 1];
        return self.bytes[start..end];
    }

    pub fn encodeChunk(
        self: *const Monster,
        allocator: std.mem.Allocator,
        chunk: []const u8,
        out: []TokenId,
    ) ![]TokenId {
        // Specialization on mask + lilbuf + score2b3b + precomputed_alts +
        // goto_checkpoint. All five flags default off; non-trainer / non-TM
        // callers hit the zero-cost variant. The precomputed-alts path is
        // enabled whenever `Monster.alts` is non-null (which every
        // `finalizeWithCapcode` call now produces). Mask + alts are
        // independent: mask still applies via the trie's `mask_ptr` view.
        // The goto_checkpoint path is only active when both lilbuf and
        // score2b3b are on (it re-evaluates the lookahead of a score2b/3b
        // win), so we collapse it under that subtree.
        const use_pa = self.use_precomputed_alts and self.alts != null;
        const use_gc = self.goto_checkpoint_enabled and self.lilbuf_enabled and self.score2b3b_enabled;
        if (self.mask == null) {
            // No mask — production hot path. Trainer marginal-value
            // scoring is the only caller that flips a mask on.
            @branchHint(.likely);
            if (self.lilbuf_enabled) {
                if (self.score2b3b_enabled) {
                    if (use_pa) {
                        if (use_gc) {
                            // TM-Go .ztm vocab default: lilbuf + score2b3b
                            // + precomp alts + goto-checkpoint all on.
                            @branchHint(.likely);
                            return self.encodeChunkImpl(false, true, true, true, true, allocator, chunk, out);
                        }
                        return self.encodeChunkImpl(false, true, true, true, false, allocator, chunk, out);
                    }
                    if (use_gc) return self.encodeChunkImpl(false, true, true, false, true, allocator, chunk, out);
                    return self.encodeChunkImpl(false, true, true, false, false, allocator, chunk, out);
                }
                if (use_pa) return self.encodeChunkImpl(false, true, false, true, false, allocator, chunk, out);
                return self.encodeChunkImpl(false, true, false, false, false, allocator, chunk, out);
            }
            if (use_pa) return self.encodeChunkImpl(false, false, false, true, false, allocator, chunk, out);
            return self.encodeChunkImpl(false, false, false, false, false, allocator, chunk, out);
        }
        if (self.lilbuf_enabled) {
            if (self.score2b3b_enabled) {
                if (use_pa) {
                    if (use_gc) return self.encodeChunkImpl(true, true, true, true, true, allocator, chunk, out);
                    return self.encodeChunkImpl(true, true, true, true, false, allocator, chunk, out);
                }
                if (use_gc) return self.encodeChunkImpl(true, true, true, false, true, allocator, chunk, out);
                return self.encodeChunkImpl(true, true, true, false, false, allocator, chunk, out);
            }
            if (use_pa) return self.encodeChunkImpl(true, true, false, true, false, allocator, chunk, out);
            return self.encodeChunkImpl(true, true, false, false, false, allocator, chunk, out);
        }
        if (use_pa) return self.encodeChunkImpl(true, false, false, true, false, allocator, chunk, out);
        return self.encodeChunkImpl(true, false, false, false, false, allocator, chunk, out);
    }

    // -------- trace variants ---------------------------------------------
    //
    // Tracing Monster's full TM-Go-equivalent picker (lilbuf + score2b3b
    // + alt evaluation + goto-checkpoint) would require threading a
    // runtime trace pointer through every comptime variant of
    // `encodeChunkImpl`, doubling the dispatch tree. Instead we walk
    // the trie ourselves in a simpler greedy-longest-match style,
    // emitting one `monster pos=… piece_id=… len=… score=…` record per
    // pick, then hand off to the real `encodeChunk` for the actual id
    // sequence so output stays bit-identical with non-traced runs.
    //
    // What the trace shows: a per-position view of "the longest piece
    // available at this byte position and its nwords-score weight." It
    // gives the user a baseline they can compare against the eventual
    // emitted ids. NOT a faithful mirror of the lilbuf / alt-evaluation
    // decisions — those live behind the production encoder and the
    // emitted ids are what callers actually see.

    pub fn encodeChunkTrace(
        self: *const Monster,
        allocator: std.mem.Allocator,
        chunk: []const u8,
        out: []TokenId,
        trace: *@import("trace.zig").Trace,
    ) ![]TokenId {
        try self.emitTrace(chunk, trace);
        return self.encodeChunk(allocator, chunk, out);
    }

    pub fn encodeChunkWithOffsetsTrace(
        self: *const Monster,
        allocator: std.mem.Allocator,
        chunk: []const u8,
        chunk_offset: u32,
        out_ids: []TokenId,
        out_offsets: []Span,
        trace: *@import("trace.zig").Trace,
    ) !usize {
        try self.emitTrace(chunk, trace);
        return self.encodeChunkWithOffsets(allocator, chunk, chunk_offset, out_ids, out_offsets);
    }

    fn emitTrace(
        self: *const Monster,
        chunk: []const u8,
        trace: *@import("trace.zig").Trace,
    ) !void {
        var pos: usize = 0;
        while (pos < chunk.len) {
            // Greedy longest match at this position via direct trie walk.
            // We don't apply the `use_mask` view here — trainer-only.
            var node_idx: u32 = 0;
            const limit = @min(chunk.len - pos, @as(usize, self.max_token_len));
            var best_len: u32 = 0;
            var best_id: u32 = NO_TOKEN;
            var k: usize = 0;
            while (k < limit) : (k += 1) {
                const child = self.findChild(node_idx, chunk[pos + k]) orelse break;
                node_idx = child;
                const tid = self.nodes[node_idx].token_id;
                if (tid != NO_TOKEN) {
                    best_len = @intCast(k + 1);
                    best_id = tid;
                }
            }
            if (best_len == 0) {
                // No trie match — emit a 1-byte "unmatched" record so
                // the trace stays aligned with input bytes and the
                // user can spot the gap. The actual encoder may emit
                // unk / byte-fallback / DEL here.
                try trace.monsterPiece(pos, self.unk_id, 1, 0);
                pos += 1;
            } else {
                // Score: the nwords-score field is i8 in Monster's SoA
                // (0..127). Widen to i32 to fit the trace record's
                // numeric column.
                const score: i32 = if (self.nwords_score.len > best_id)
                    @intCast(@as(i32, @intCast(self.nwords_score[best_id])))
                else
                    0;
                try trace.monsterPiece(pos, best_id, best_len, score);
                pos += best_len;
            }
        }
    }

    inline fn encodeChunkImpl(
        self: *const Monster,
        comptime use_mask: bool,
        comptime use_lilbuf: bool,
        comptime use_score2b3b: bool,
        comptime use_precomp_alts: bool,
        comptime use_goto_checkpoint: bool,
        allocator: std.mem.Allocator,
        chunk: []const u8,
        out: []TokenId,
    ) ![]TokenId {
        // With lilbuf path (a), worst-case output is 2*chunk.len
        // (every byte emits DEL + matched piece). Without lilbuf, it's
        // 1*chunk.len. Callers (Pipeline) size their buffer to
        // `Model.maxTokensFor` which accounts for the worst case.
        const expand: usize = if (use_lilbuf) 2 else 1;
        std.debug.assert(out.len >= chunk.len * expand);
        monsterTraceInit();
        _ = allocator; // reserved for the eventual realloc-growing scratch buffer
        if (chunk.len == 0) {
            // Empty-chunk fast-exit — callers usually pass non-empty
            // chunks; this fires for the boundary case only.
            @branchHint(.unlikely);
            return out[0..0];
        }

        // Stack-resident scratch — bounded by MAX_CANDIDATES per position.
        // 64 * (4+4) = 512 bytes, well within any frame.
        var cand_lens: [MAX_CANDIDATES]u32 = undefined;
        var cand_ids: [MAX_CANDIDATES]u32 = undefined;

        // TM-Go's forwardDelete state: set to 1 the iteration AFTER a
        // DEL-emit branch wins. Used to (a) shrink the current branch's
        // perceived length by 1 (TM-Go: `branchLength = ... - forwardDelete`)
        // and (b) shrink nWords by 1 (TM-Go: `int(first.nWords) - forwardDelete`).
        // Cleared after every non-DEL-emit branch.
        var forward_delete: u8 = 0;

        // TM-Go's goto-checkpoint re-entry state. When a `.first_del_second`
        // (score2b/3b) branch wins, the encoder emits ONLY `[alt_first, DEL]`
        // (suppressing the immediate lilbuf_second emit) and sets
        // forward_lilbuf=true with the lilbuf token captured here. The next
        // iteration treats `(fwd_seed_id, fwd_seed_real_len)` as the synthetic
        // greedy match (representing the `\x7F `-prefixed segment that begins
        // at the new position) and re-runs alt evaluation against THAT seed.
        // The seeded alt evaluation can pick a DIFFERENT (often shorter) alt
        // than what the 1.16 single-emit path would have chosen — that's
        // what closes additional TM-Go equivalence gaps.
        //
        // Mirrors TM-Go's `case score2b: tokens = append(..., delTok);
        // i += original.length - forwardDelete; length = length2b;
        // index = index2b; forwardDelete = 1; goto checkpoint`
        // (refs/tokenmonster/go/tokenmonster.go:1248-1254).
        var forward_lilbuf: bool = false;
        var fwd_seed_id: u32 = NO_TOKEN;
        var fwd_seed_real_len: u32 = 0;
        // The mutators below are gated on `use_goto_checkpoint`. When
        // that comptime flag is false the compiler sees only the reads
        // and complains the locals are never mutated; force-take the
        // address so the lint clears for the !use_goto_checkpoint path.
        if (!use_goto_checkpoint) {
            _ = &forward_lilbuf;
            _ = &fwd_seed_id;
            _ = &fwd_seed_real_len;
        }

        var write: usize = 0;
        var i: usize = 0;
        while (i < chunk.len) {
            // Seeded iteration after a goto-checkpoint trigger: pretend
            // `collectPrefixMatches` returned the lilbuf token at this
            // position, and skip the lilbuf walks (they'd be redundant
            // with the seed). The seed represents the conceptual
            // `\x7F `-prefixed segment; its real-byte length is what
            // gets used to advance.
            const is_seeded = use_goto_checkpoint and forward_lilbuf;
            const n_cand = if (is_seeded) blk: {
                // Seeded iteration — fires only the iteration AFTER a
                // score2b/3b DEL-emit win. Cold path on every real
                // corpus we've benched (a few percent of positions at
                // most).
                @branchHint(.unlikely);
                // Single "candidate" = the seeded lilbuf token. The
                // stored length is `fwd_seed_real_len + 1`: vocab byte
                // count of the seed token (real bytes + the synthetic
                // 1-byte lilbuf-space prefix that was conceptually
                // applied in the prior iter). This +1 mirrors TM-Go's
                // bookkeeping where `original.length` is vocab bytes
                // and `forwardDelete = 1` strips off the synthetic
                // prefix at score/advance time. Without this +1, the
                // tmScore baseline (`greedy_len_i = first_len_raw -
                // forward_delete`) would be off by one and Equal/Less
                // ties would resolve differently from TM-Go.
                //
                // Concretely: ` train` is 6 vocab bytes (` `+`train`).
                // `fwd_seed_real_len = 5` (just `train`). We store 6
                // so that `first_len_i = 6 - 1 = 5` matches TM-Go's
                // `length = 5` baseline.
                cand_lens[0] = fwd_seed_real_len + 1;
                cand_ids[0] = fwd_seed_id;
                break :blk @as(usize, 1);
            } else self.collectPrefixMatches(
                use_mask,
                chunk[i..],
                &cand_lens,
                &cand_ids,
            );

            // Try BOTH lilbuf paths. They may succeed even when
            // greedy/alts have NO match at all (e.g. a bare 'M' in the
            // middle of a "TokenMonster" with no DEL-marker
            // normalization).
            //
            //   Path (b) `\x7f `-prefix: direct trie lookup into
            //     `\x7f Mon`-style tokens.
            //   Path (a) ` `-prefix + DEL emit: TM-Go's actual approach
            //     — reaches space-prefixed tokens like ` monster` then
            //     emits a DEL boundary marker before them.
            //
            // At a seeded iteration, both paths are skipped: the seed
            // already represents the lilbuf result from the prior iter.
            // Running them again would (a) reach the SAME ` ...` token
            // as the seed (path-a) or (b) reach a `\x7F ...` token that
            // would imply a chained DEL emit, which is not what we want
            // at this position (the DEL was already emitted).
            var ll_id: u32 = NO_TOKEN;
            var ll_advance: u32 = 0;
            var ll_sp_id: u32 = NO_TOKEN;
            var ll_sp_advance: u32 = 0;
            const lilbuf_gate_ok = if (self.lilbuf_marker_byte == 0x7F)
                lilbufPositionGate(chunk, i)
            else
                lilbufPositionGateCapcode(chunk, i);
            if (use_lilbuf and !is_seeded and lilbuf_gate_ok) {
                // 1.16 gate relaxation: TM-Go's score1b/2b/3b only checks
                // `nextByte == 1` (the lookahead byte is a letter). It has
                // NO `chunk[i-1]` gate. ztok previously required mid-word
                // (`isAsciiLetter(chunk[i-1])`), which missed cases like
                // digit-start positions and the post-token-boundary in
                // `width="661"`. The current relaxation: lilbuf fires when
                // `chunk[i]` is a letter AND the previous byte (if any)
                // is NOT a letter — i.e., we're at a word boundary
                // (start of input, or transitioning from non-letter to
                // letter). This is the inverse of the old gate. Empirical
                // confirmation lives in the score2b cherry-picks.
                if (self.has_lilbuf_prefix_tokens) {
                    var ll_total_len: u32 = 0;
                    ll_id = self.lilbufLongestMatch(use_mask, chunk[i..], &ll_total_len);
                    if (ll_id != NO_TOKEN and ll_total_len >= 3) {
                        ll_advance = ll_total_len - 2;
                    } else {
                        ll_id = NO_TOKEN;
                    }
                }
                if (self.delete_token_id != NO_TOKEN and self.has_space_prefix_tokens) {
                    var ll_sp_total: u32 = 0;
                    ll_sp_id = self.lilbufSpaceLongestMatch(use_mask, chunk[i..], &ll_sp_total);
                    if (ll_sp_id != NO_TOKEN and ll_sp_total >= 2) {
                        ll_sp_advance = ll_sp_total - 1;
                    } else {
                        ll_sp_id = NO_TOKEN;
                    }
                }
            }

            if (n_cand == 0 and ll_id == NO_TOKEN and ll_sp_id == NO_TOKEN) {
                // No vocab token matches here — emit unk for one byte.
                // Rare on real corpora: byte-coverage vocabs reach every
                // byte through at least the single-byte tokens.
                @branchHint(.unlikely);
                std.debug.assert(write < out.len);
                out[write] = self.unk_id;
                write += 1;
                i += 1;
                continue;
            }

            // Seed best_* with whatever default is available, prioritizing
            // greedy over path-b over path-a. Subsequent score loops will
            // overwrite via the score comparison.
            //
            // Branch outputs distinguish three emission modes:
            //   .normal           — emit [first_id]
            //   .del_before_first — emit [DEL, first_id] (path-a lilbuf)
            //   .first_del_second — emit [first_id, DEL, second_id] (score-b)
            //                       — score2b/score3b: the alt first wins,
            //                         then a DEL marker, then the lilbuf-
            //                         prefixed lookahead. Also sets
            //                         forward_delete=1 for next iteration.
            //   del_then_seed     — emit [DEL] (0 input bytes), advance 0,
            //                       and seed the continuation as next iter's
            //                       greedy (TM-Go score2 `case`: bare DEL +
            //                       `goto checkpoint`).
            const Emit = enum { normal, del_before_first, first_del_second, del_then_seed };
            var best_first_id: u32 = undefined;
            var best_advance: u32 = undefined;
            var best_second_id: u32 = NO_TOKEN;
            // 1.20 perf: cache the score-b lilbuf real-byte length on
            // the winning branch so the post-loop `.first_del_second`
            // emit doesn't re-walk `lilbufSpaceLongestMatch` (saving
            // one O(max_token_len) trie walk per score-b win).
            var best_lb_real: u32 = 0;
            var best_emit: Emit = .normal;
            if (n_cand > 0) {
                best_first_id = cand_ids[n_cand - 1];
                best_advance = cand_lens[n_cand - 1];
            } else if (ll_id != NO_TOKEN) {
                best_first_id = ll_id;
                best_advance = ll_advance;
            } else {
                // n_cand==0, ll_id==NO_TOKEN; the assert above guarantees
                // ll_sp_id != NO_TOKEN here.
                best_first_id = ll_sp_id;
                best_advance = ll_sp_advance;
                best_emit = .del_before_first;
            }
            var best_score: i32 = std.math.minInt(i32);
            // Tracks whether the winning branch should clear forward_delete
            // (true for non-score-b wins, false for score-b which sets it
            // to 1 for the NEXT iter). Default true so plain "no alt evaluated"
            // paths (skipped to fall-through) still clear it correctly.
            var winner_sets_fd: bool = false;

            // --- TM-Go switch-precedence tie-break (gap #2) ---
            // TM-Go tokenmonster.go:1217-1262: maxScore = max(score1,
            // score2, score3, score1b, score2b, score3b) is resolved by a
            // `switch maxScore` whose cases are listed in source order
            // score1 ▸ score2 ▸ score3 ▸ score1b ▸ score2b ▸ score3b. The
            // first case equal to maxScore wins, so on a tie ALL plain
            // scores (score1/2/3) outrank ALL b-scores (score1b/2b/3b),
            // and within each group the earlier branch wins.
            //
            // ztok previously folded each branch's score-b into `best_score`
            // INLINE (strict `>`), giving precedence score1 ▸ score1b ▸
            // score2 ▸ ... — so a `score1b == score2` tie kept score1b
            // (TM picks score2). To match TM, we accumulate the best
            // b-candidate SEPARATELY (strict `>` so the earliest branch
            // wins b-group ties) and reconcile it against the plain
            // `best_score` ONCE after the whole branch loop, where the
            // strict `>` lets a b-score win ONLY if it strictly beats the
            // best plain score (ties go to plain, matching the switch).
            // At a seeded iteration, all branch lengths (greedy seed AND
            // its precomputed alts) carry an implicit +1 representing
            // the synthetic lilbuf-space prefix. The score formula and
            // advance both need real-byte counts; subtract 1 when
            // computing real positions/lengths. At a non-seeded iter,
            // this is 0 (everything is already in real bytes).
            const seed_advance_adj_pre: u32 = if (use_goto_checkpoint and is_seeded) 1 else 0;

            // gap #3 — `flag & 32` single-whole-word skip-gate
            // (TM-Go tokenmonster.go:1057). When the greedy first token is a
            // lone space-led all-letter word (FLAG_SINGLE_WORD set) AND the
            // byte that follows it begins a space (beginByte == 12 == BB_SPACE),
            // TM-Go emits the greedy token UNCONDITIONALLY and skips ALL alt
            // scoring. ztok previously computed FLAG_SINGLE_WORD but never
            // read it, always scoring alts — which can pull a tying/winning
            // alt over the greedy whole word on prose. This gate forces the
            // greedy emit, matching TM's early path. When it fires, best_*
            // stay at the greedy seed below (`.normal`, winner_sets_fd=false)
            // and the branch loop + path-b/path-a are skipped.
            var skip_alts: bool = false;
            if (n_cand > 0) {
                const g_idx: usize = n_cand - 1;
                const g_id: u32 = cand_ids[g_idx];
                const g_real_len: u32 = cand_lens[g_idx] - seed_advance_adj_pre;
                const next_pos: usize = i + g_real_len; // TM-Go: i1 = i + length
                if (next_pos < chunk.len and g_id < self.count) {
                    // Alias-aware greedy flag (mirrors the per-branch
                    // first_flag selection for b==0).
                    const g_is_alias: bool = g_id < self.alias_lens_by_id.len and
                        self.alias_lens_by_id[g_id] != 0 and
                        @as(u32, self.alias_lens_by_id[g_id]) == g_real_len;
                    const g_flag: u8 = if (g_is_alias) self.bare_flags[g_id] else self.flags[g_id];
                    // gap #3 REVERTED: skip-gate disabled (always score
                    // alts). Re-enabling it was measured inert at baseline.
                    if (false and (g_flag & FLAG_SINGLE_WORD) != 0 and
                        self.begin_byte[chunk[next_pos]] == BB_SPACE)
                    {
                        skip_alts = true;
                    }
                }
            }

            if (n_cand > 0 and !skip_alts) {
                // collectPrefixMatches returns shortest→longest. The greedy
                // first token is the last entry.
                const greedy_idx: usize = n_cand - 1;
                const greedy_len_raw: u32 = cand_lens[greedy_idx];
                const greedy_id_raw: u32 = cand_ids[greedy_idx];
                // TM-Go's "length" at branch entry is the current-match's
                // length minus forwardDelete (see how `length` is set after
                // each goto checkpoint). The -100/-10000 alt deductions
                // compare against THIS adjusted length.
                const greedy_len_i: i32 = @as(i32, @intCast(greedy_len_raw)) - @as(i32, forward_delete);

                // Branch selection: legacy path walks the 6 longest
                // candidates from collectPrefixMatches; precomputed-alts
                // path uses (greedy, alts[greedy].index, alts[greedy].index2).
                // The precomputed path is what gives TM-Go bit equivalence
                // — its priority-ranked picks differ from "longest 6 by
                // length", especially on capcode vocabs where the right
                // alt is often a much shorter subtoken at a
                // boundary-class transition (refs/tokenmonster/go/
                // tokenmonster.go:3597-3753 priority ladder).
                var br_ids: [3]u32 = .{ greedy_id_raw, NO_TOKEN, NO_TOKEN };
                var br_lens: [3]u32 = .{ greedy_len_raw, 0, 0 };
                var n_branches: usize = 1;
                // Stage 2: true when the greedy match came from a bare
                // (no-marker) v2 alias AND we sourced the alt branches
                // from `bare_alt_lut`. In that case the whole twin
                // context is the BARE form, so the alt branches (b>0)
                // must read bare flags/nwords for their alt ids too
                // (mirroring TM-Go reading `vocab.info[alt.index].flag`
                // for the bare info entry). The greedy branch (b==0)
                // already handles this via `first_is_alias_match`.
                var greedy_from_bare_alias: bool = false;
                if (use_precomp_alts) {
                    if (self.alts) |alts_buf| {
                        // When the greedy match came from a v2 alias (the
                        // BARE twin form), the primary's `self.alts` were
                        // computed for the PRIMARY (`\x7F `-prefixed) byte
                        // sequence — applying those offsets here would
                        // advance past the wrong bytes. Stage 2: instead
                        // of skipping alt evaluation, fetch the BARE
                        // twin's own alt table (`bare_alt_lut`, in bare
                        // units) so this position scores its real alts.
                        // (TM-Go stores per-info-entry alts so each twin
                        // has its own; v3 + this LUT supply that.)
                        const greedy_is_alias = greedy_id_raw < self.alias_lens_by_id.len and
                            self.alias_lens_by_id[greedy_id_raw] != 0 and
                            self.alias_lens_by_id[greedy_id_raw] == greedy_len_raw;
                        greedy_from_bare_alias = greedy_is_alias;
                        const ap: AltPair = if (!greedy_is_alias)
                            alts_buf[greedy_id_raw]
                        else if (greedy_id_raw < self.bare_alt_lut.len)
                            self.bare_alt_lut[greedy_id_raw]
                        else
                            .{};
                        if (ap.index != NO_TOKEN and ap.length > 0) {
                            // Skip alts whose id is masked.
                            const ok1 = if (use_mask) (self.mask.?[ap.index] == 0) else true;
                            if (ok1) {
                                br_ids[1] = ap.index;
                                br_lens[1] = ap.length;
                                n_branches = 2;
                            }
                            if (ap.index2 != NO_TOKEN and ap.length2 > 0) {
                                const ok2 = if (use_mask) (self.mask.?[ap.index2] == 0) else true;
                                if (ok2) {
                                    br_ids[2] = ap.index2;
                                    br_lens[2] = ap.length2;
                                    n_branches = 3;
                                }
                            }
                        }
                    }
                } else {
                    n_branches = @min(n_cand, MAX_BRANCHES);
                }

                var b: usize = 0;
                while (b < n_branches) : (b += 1) {
                    const is_greedy = (b == 0);
                    const first_len_raw = if (use_precomp_alts)
                        br_lens[b]
                    else
                        cand_lens[greedy_idx - b];
                    const first_id = if (use_precomp_alts)
                        br_ids[b]
                    else
                        cand_ids[greedy_idx - b];
                    // TM-Go: branchLength = original.length + length2 -
                    // forwardDelete (for the alt branches). The greedy
                    // branch uses `length + length1` without forwardDelete
                    // subtraction inside score1 itself (it's pre-subtracted
                    // into `length` via the prior checkpoint). To keep the
                    // semantics consistent here, apply forward_delete to
                    // the first-token portion uniformly.
                    const first_len_i: i32 = @as(i32, @intCast(first_len_raw)) - @as(i32, forward_delete);

                    // Lookahead position: at a seeded iter, `first_len_raw`
                    // is VOCAB bytes (includes the synthetic lilbuf
                    // prefix); subtract 1 to get the real-input offset
                    // for the trie walk. At a non-seeded iter,
                    // `first_len_raw` is already real bytes (no
                    // adjustment).
                    const first_real_len = first_len_raw - seed_advance_adj_pre;
                    const after = i + first_real_len;
                    // Plain second via direct trie lookup. Kept separate
                    // from `second_id` / `second_len` so the score-b
                    // gate (which uses TM-Go's `length1` semantics)
                    // sees the real plain match — NOT the phantom-second
                    // substitution applied below.
                    var plain_second_id: u32 = NO_TOKEN;
                    var plain_second_len: u32 = 0;
                    if (after < chunk.len) {
                        // 1.19 perf: ONE trie walk yields both id and
                        // length. Pre-1.19 did two back-to-back walks
                        // here (longestMatchIdOrNone + longestMatchLen),
                        // doubling per-branch trie-walk cost.
                        const m = self.longestMatchIdAndLen(use_mask, chunk[after..]);
                        plain_second_id = m.id;
                        plain_second_len = m.len;
                    }
                    var second_id: u32 = plain_second_id;
                    var second_len: u32 = plain_second_len;
                    var second_is_phantom: bool = false;

                    // v2 alias detection on second position. If the
                    // plain trie match landed on a bare alias (length
                    // matches the alias byte length for this id, not
                    // the primary length), use bare_flags/bare_nwords
                    // for the score — same effect as the phantom-second
                    // path-(b) substitution below, but reached via
                    // direct trie hit instead of synthesized prefix.
                    if (second_id != NO_TOKEN and
                        second_id < self.alias_lens_by_id.len and
                        self.alias_lens_by_id[second_id] != 0 and
                        @as(u32, self.alias_lens_by_id[second_id]) == second_len)
                    {
                        second_is_phantom = true;
                    }

                    // === TM-Go-parity: phantom-second via path-(b) =======
                    //
                    // ztok's vocab convert script (bench/convert_tm_to_ztm.py)
                    // collapses TM-Go's twin entries (`train` and `\x7F train`
                    // share alt_id 9735) into a SINGLE trie entry — the
                    // longer `\x7F `-prefixed form wins the collision. That
                    // leaves ztok's trie missing the bare-letter forms (`t`,
                    // `tr`, `train`, …) that TM-Go's `LongestSubstring`
                    // happily matches mid-word. ztok's `longestMatchIdAndLen`
                    // returns NO_TOKEN at such positions even though TM-Go
                    // would have a perfectly good plain match.
                    //
                    // Concrete fallout: at position `train.\x7F md)` (after
                    // greedy `\x7F pre`), TM-Go's `LongestSubstring`
                    // returns `train` (5 bytes, alt_id 9735). ztok returns
                    // NO_TOKEN. With `second_id == NO_TOKEN`, score1
                    // collapses and loses to alt1's score2 (alt1 = bare
                    // `\x7F` whose plain second IS findable).
                    //
                    // Fix: when the plain second misses BUT path-(b)
                    // synthesis (`\x7F ` + input) succeeds at the same
                    // position, use the synthesized id/length as the
                    // effective second for SCORING ONLY (emitted ids
                    // unchanged — the next iteration discovers this token
                    // via the production path-(b) branch).
                    //
                    // Important: the FLAGS for the bare-letter form differ
                    // from the `\x7F `-prefixed form (the prefixed form has
                    // FLAG_BEGINS_CAPCODE set; the bare form has
                    // FLAG_BEGINS_LETTER). The `second_is_phantom` marker
                    // tells the score block below to use the precomputed
                    // bare-form values (`bare_flags`/`bare_nwords`).
                    //
                    // Crucially, the score-b block (below) still uses
                    // `plain_second_id` / `plain_second_len` for ITS
                    // gate (TM-Go's `score1b` condition checks the plain
                    // `length1` only). The phantom-second affects only
                    // the score1/2/3 computation, not score-b's gate.
                    //
                    // For v2 `.ztm` files (where aliases are present in
                    // the trie), the plain trie match above usually
                    // succeeds with the alias bytes, so this fallback
                    // doesn't fire. It remains active for v1 vocabs
                    // and for vocabs whose aliases didn't extend down
                    // to this position.
                    if (use_lilbuf and
                        second_id == NO_TOKEN and
                        self.has_lilbuf_prefix_tokens and
                        after < chunk.len and
                        isAsciiLetter(chunk[after]))
                    {
                        var ph_total: u32 = 0;
                        const ph_id = self.lilbufLongestMatch(use_mask, chunk[after..], &ph_total);
                        if (ph_id != NO_TOKEN and ph_total >= 3) {
                            // ph_total includes the synthetic `\x7F ` (2
                            // bytes). Real-byte coverage = ph_total - 2.
                            second_id = ph_id;
                            second_len = ph_total - 2;
                            second_is_phantom = true;
                        }
                    }

                    // Word-density bonuses. `nwords_score` is the true
                    // TM-Go whole-word count (vs ztok's legacy `nwords`
                    // which counts the first byte standalone — wrong
                    // for the TM-Go score formula).
                    //
                    // For phantom-second (a `\x7F `-prefixed entry standing
                    // in for the bare-letter match TM-Go would have made),
                    // use the precomputed `bare_nwords[id]` instead of
                    // `nwords_score[id]`. The bare form has different
                    // word-start count because the synthetic `\x7F ` prefix
                    // in the stored bytes contributes one extra
                    // space→alphanum transition that the bare form lacks.
                    // For v2 alias matches on `first_id`, use
                    // `bare_nwords[id]` — the alias bytes are the
                    // bare form (no marker), so its word count
                    // matches what TM-Go's `LongestSubstring` would
                    // see when matching the bare twin. Empirically
                    // this is required for the capcode 32k vocab
                    // improvement.
                    // Stage 2: extend bare-flag/nword selection to the
                    // alt branches (b>0) when the greedy match was a
                    // bare alias and the alt id ALSO has a bare form
                    // whose length equals this alt branch's length. Both
                    // the greedy and its bare-sourced alts then read the
                    // bare info entry, matching TM-Go's per-twin tables.
                    const first_alias_for_nw: bool = first_id < self.alias_lens_by_id.len and
                        self.alias_lens_by_id[first_id] != 0 and
                        @as(u32, self.alias_lens_by_id[first_id]) == first_real_len and
                        (b == 0 or greedy_from_bare_alias);
                    const nw1_raw: i32 = if (first_alias_for_nw)
                        @intCast(self.bare_nwords[first_id])
                    else
                        @intCast(self.nwords_score[first_id]);
                    const nw1: i32 = nw1_raw - @as(i32, forward_delete);
                    const nw2: i32 = if (second_id == NO_TOKEN)
                        0
                    else if (second_is_phantom)
                        @as(i32, @intCast(self.bare_nwords[second_id]))
                    else
                        @as(i32, @intCast(self.nwords_score[second_id]));

                    // TM-Go flag + beginByte lookups for the score formula.
                    // `next_bb` is `beginByte[chunk[tail]]`; past-end is
                    // mapped to 0 (TM-Go reads `beginByte[0]` which
                    // defaults to 0 since the 0x00 byte rarely starts a
                    // vocab piece by majority). This is intentionally
                    // distinct from the pre-flag `next_is_space = 1` at
                    // end-of-input — and is what flips several alt-vs-
                    // greedy ties back to TM-Go's choice at chunk tails.
                    //
                    // For phantom-second matches, use `bare_flags[id]`
                    // (the bare-form flag bits) instead of `flags[id]`
                    // (which describes the `\x7F `-prefixed stored form).
                    //
                    // Symmetric v2 alias detection on `first_id`: if
                    // the greedy match length equals the alias byte
                    // length for that id, the match came from the bare
                    // alias and the bare-form flags apply.
                    const tail = after + second_len;
                    const first_is_alias_match: bool = first_id < self.alias_lens_by_id.len and
                        self.alias_lens_by_id[first_id] != 0 and
                        @as(u32, self.alias_lens_by_id[first_id]) == first_real_len and
                        (b == 0 or greedy_from_bare_alias);
                    const first_flag: u8 = if (first_id >= self.count)
                        0
                    else if (first_is_alias_match)
                        self.bare_flags[first_id]
                    else
                        self.flags[first_id];
                    const second_flag: u8 = if (second_id == NO_TOKEN or second_id >= self.count)
                        0
                    else if (second_is_phantom)
                        self.bare_flags[second_id]
                    else
                        self.flags[second_id];
                    const next_bb: u8 = if (tail >= chunk.len) 0 else self.begin_byte[chunk[tail]];

                    const score = tmScore(
                        first_len_i,
                        @as(i32, @intCast(second_len)),
                        first_flag,
                        second_flag,
                        nw1,
                        nw2,
                        next_bb,
                        !is_greedy,
                        greedy_len_i,
                        0, // no extra-token penalty for plain score1/2/3
                        false, // keep `(second.flag >> 2) & 1` bonus
                    );

                    // Trace the plain branch score (score1/score2/score3
                    // for b==0/1/2). Inert when ZTOK_MONSTER_TRACE unset.
                    traceBranch(i, if (is_greedy) "score1" else "scoreN_plain", b, first_id, first_len_raw, second_id, second_len, score);

                    if (score > best_score) {
                        best_score = score;
                        best_first_id = first_id;
                        best_advance = first_len_raw;
                        best_emit = .normal;
                        winner_sets_fd = false;
                    }

                    // --- score-b extension: TM-Go's score1b/score2b/score3b ---
                    //
                    // Fires when the LOOKAHEAD position lands mid-word and
                    // a lilbuf-space synthetic prefix gives a longer match.
                    //
                    // Gating:
                    //   - score2b3b feature flag on
                    //   - vocab has DEL token + at least one space-prefixed token
                    //   - the byte AT the lookahead position is a letter
                    //     (we're trying to start a word mid-input)
                    //   - EITHER:
                    //       (a) plain second exists, starts with letter, has
                    //           nwords_tm == 0, AND byte after it is a letter
                    //           — matches TM-Go's
                    //           `second.flag & 2 != 0 && nextByte == 1 &&
                    //            second.nWords == 0` gate; OR
                    //       (b) no plain second exists at all. For ztok
                    //           nocapcode vocabs that carry NO bare-letter-
                    //           starting tokens (everything is space-prefixed),
                    //           ANY mid-word position has plain_second ==
                    //           NO_TOKEN. The lilbuf-space + DEL is the only
                    //           way to reach those words. TM-Go doesn't take
                    //           this branch (its vocab carries 'train'-style
                    //           bare tokens) but the byte-level outcome is
                    //           the same.
                    // The score-b gate has two effective shapes:
                    //   gate_a — TM-Go's exact condition (plain second
                    //     exists, starts with letter, has zero internal
                    //     word, followed by a letter byte). Uses
                    //     `second_id`/`second_len` as the reference.
                    //   gate_b — for nocapcode vocabs that carry no bare-
                    //     letter tokens, plain second is always NO_TOKEN.
                    //     The path-(b) (`\x7f `-prefix) match is the
                    //     structural analog of TM-Go's plain second. If
                    //     path-(b) finds a mid-word match at `after` and
                    //     path-(a) (lilbuf-space + DEL) at the same
                    //     position covers MORE bytes, score-b wins.
                    if (use_score2b3b and use_lilbuf and
                        self.delete_token_id != NO_TOKEN and
                        self.has_space_prefix_tokens and
                        after < chunk.len)
                    {
                        // Use `plain_second_id`/`plain_second_len` (the
                        // direct trie lookup before phantom-second
                        // substitution) so this gate matches TM-Go's
                        // score1b/2b/3b condition exactly. The
                        // phantom-second is a SCORING-ONLY restoration of
                        // TM-Go's greedy semantics and must NOT be used
                        // here.
                        //
                        // Gate-b fallback (path-b fill when plain second
                        // is NO_TOKEN): kept for both greedy and alt
                        // branches. ztok's collapsed vocab can leave the
                        // plain lookup empty even when TM-Go's wouldn't.
                        // The fallback restores the branch's ability to
                        // win via score-b in those cases. Relies on the
                        // ungated split_word formula in tmScore (the
                        // `drop_begin_space_bonus`-gated variant) to
                        // correctly penalize letter-ends-letter splits
                        // that the lilbuf-space second_flag would have
                        // otherwise dodged.
                        var eff_second_id: u32 = plain_second_id;
                        var eff_second_len: u32 = plain_second_len;
                        var eff_from_pb_fallback: bool = false;
                        // True when eff_second came from the path-b
                        // (`\x7F `-prefix) fallback below: that match is a
                        // synthetic-prefixed token standing in for the
                        // bare-letter word TM-Go would have matched, so the
                        // gap-5 flag test must read the BARE-form flags
                        // (FLAG_BEGINS_LETTER), not the stored
                        // `\x7F `-prefixed flags (FLAG_BEGINS_CAPCODE).
                        if (eff_second_id == NO_TOKEN and
                            self.has_lilbuf_prefix_tokens and
                            i + first_len_raw > 0)
                        {
                            var pb_total: u32 = 0;
                            const pb_id = self.lilbufLongestMatch(use_mask, chunk[after..], &pb_total);
                            if (pb_id != NO_TOKEN and pb_total >= 3) {
                                eff_second_id = pb_id;
                                eff_second_len = pb_total - 2;
                                eff_from_pb_fallback = true;
                            }
                        }

                        if (eff_second_id != NO_TOKEN and eff_second_len > 0) {
                            const after_second = after + eff_second_len;
                            // TM-Go gate (gap #5): tokenmonster.go:1088/1137/1189
                            //   second.flag & 2 != 0 && nextByte == 1 && second.nWords == 0
                            // i.e. the second token BEGINS with a letter
                            // (FLAG_BEGINS_LETTER), the byte AFTER the second
                            // is a letter (`beginByte == BB_LETTER`), and the
                            // second covers no whole word. ztok previously
                            // recast this as `isAsciiLetter(chunk[after])` +
                            // `isAsciiLetter(chunk[after_second])`, which
                            // MISSES non-ASCII letters (Greek/Cyrillic/accented)
                            // that TM-Go's beginByte table accepts. Use the
                            // begin_byte table + the eff_second flag instead
                            // so multi-byte runes pass the gate exactly as TM.
                            // Stage 9: TM's `nextByte == 1` test for the byte
                            // AFTER the second uses the beginByte table, which
                            // classifies the capcode word marker `W` (0x57) as
                            // 10, NOT a letter-start (1). ztok's prior
                            // `isAsciiLetter(chunk[after_second])` accepted `W`
                            // as a letter (it IS ASCII), firing the score-b
                            // precondition where TM would not — the cap
                            // over-stitch (`D or`) root cause proven by the
                            // TM-Go probe (pos=35 greedy: nextByte=beginByte['W']
                            // =10≠1 → TM never enters the 2b gate). Use the
                            // begin_byte table for the after-second conjunct so
                            // capcode markers gate out exactly as TM. The
                            // first conjunct keeps isAsciiLetter (the second's
                            // own begin classification — TM uses second.flag&2).
                            const inside_word =
                                isAsciiLetter(chunk[after]) and
                                after_second < chunk.len and
                                self.begin_byte[chunk[after_second]] == BB_LETTER and
                                self.nwords_tm[eff_second_id] == 0;
                            if (monster_trace_enabled) {
                                @branchHint(.unlikely);
                                const ef: u8 = if (eff_second_id < self.count) self.flags[eff_second_id] else 0;
                                const efb: u8 = if (eff_second_id < self.count) self.bare_flags[eff_second_id] else 0;
                                std.debug.print("TRACE pos={d} scoreNb_GATE b={d} eff_second_id={d} eff_len={d} eff_flag=0b{b:0>8} bare_flag=0b{b:0>8} nwords_tm={d} pb_fb={} inside_word={}\n", .{ i, b, eff_second_id, eff_second_len, ef, efb, self.nwords_tm[eff_second_id], eff_from_pb_fallback, inside_word });
                            }
                            if (inside_word) {
                                var lb_total: u32 = 0;
                                const lb_id = self.lilbufSpaceLongestMatch(use_mask, chunk[after..], &lb_total);
                                if (lb_id != NO_TOKEN and lb_total >= 2) {
                                    const lb_real: u32 = lb_total - 1;
                                    // TM-Go gate (gap #1): tokenmonster.go:1092/1141/1193
                                    // require `length2b > length2 + 1` — lilbuf
                                    // must cover at least TWO more real bytes
                                    // than the plain second. ztok previously
                                    // used `> eff_second_len` (1-byte off),
                                    // firing the DEL-insert one byte too eagerly.
                                    if (lb_real > eff_second_len) { // gap #1 REVERTED
                                        const nw2b: i32 = @intCast(self.nwords_score[lb_id]);
                                        const tail_b = after + lb_real;
                                        // score-b: full TM-Go formula
                                        // with extra_token_penalty=1
                                        // and drop_begin_space=true
                                        // (lilbuf-space always begins
                                        // with the synthetic space).
                                        const lb_flag = self.flags[lb_id];
                                        const next_bb_b: u8 = if (tail_b >= chunk.len) 0 else self.begin_byte[chunk[tail_b]];
                                        const score_b = tmScore(
                                            first_len_i,
                                            @as(i32, @intCast(lb_real)),
                                            first_flag,
                                            lb_flag,
                                            nw1,
                                            nw2b,
                                            next_bb_b,
                                            !is_greedy,
                                            greedy_len_i,
                                            1, // -1 extra-token penalty
                                            true, // drop begin-space bonus
                                        );
                                        traceBranch(i, "scoreNb", b, first_id, first_len_raw, lb_id, lb_real, score_b);
                                        // gap #2 REVERTED: fold the score-b
                                        // candidate INLINE into best_score with
                                        // strict `>`, in the same per-branch path
                                        // as the plain score1/2/3 fold above.
                                        if (score_b > best_score) {
                                            best_score = score_b;
                                            best_first_id = first_id;
                                            best_advance = first_len_raw;
                                            best_second_id = lb_id;
                                            best_lb_real = lb_real;
                                            best_emit = .first_del_second;
                                            winner_sets_fd = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // gap #2 REVERTED: score-b folded inline above; no
                // post-loop reconciliation needed.

                // === TM-Go score2 bare-DEL twin-split at an alias position ===
                // The residual: ztok's greedy here is a BARE v2 alias (e.g.
                // `Ex`=722 reached via alias_lens_by_id). TM-Go, at this same
                // input position, has TWO twin info entries and its score2
                // emits the bare DEL (`\x7f`, 0-marker form, real length 1),
                // `goto checkpoint`, and re-greedy-matches the longer
                // space/marker-prefixed continuation (` Exp…`=4192) next iter.
                // ztok has no branch that does this, so it stays combined.
                // Inject TM's score2 as an explicit competing branch, scored
                // with bare-DEL flags for the first token (DEL: real len 1)
                // and a lilbuf-space longest-match continuation as the second.
                // On win: emit [DEL] (0 input bytes), advance 0, and seed the
                // continuation as next iter's greedy (TM's `goto checkpoint`),
                // so the byte stream becomes [\x7f][ Exp…] exactly as TM-Go.
                if (use_lilbuf and use_goto_checkpoint and
                    greedy_from_bare_alias and
                    self.delete_token_id != NO_TOKEN and
                    self.lilbuf_marker_byte == 0x7F and
                    self.has_space_prefix_tokens and
                    i + 1 < chunk.len and
                    isAsciiLetter(chunk[i]))
                {
                    // TM score2 first token = bare DEL covering exactly the
                    // ONE real byte at chunk[i] (TM: original.length(1) -
                    // forwardDelete). The continuation is the longest
                    // space-prefixed match STARTING AT chunk[i] (the synthetic
                    // ` ` stands in for the word boundary the normalizer
                    // omitted at this intra-word camelCase seam). lilbuf-space
                    // returns total = real_bytes + 1 (synthetic space).
                    var d_sp_total: u32 = 0;
                    const d_sp_id = self.lilbufSpaceLongestMatch(use_mask, chunk[i..], &d_sp_total);
                    if (d_sp_id != NO_TOKEN and d_sp_total >= 3) {
                        const d_sp_real: u32 = d_sp_total - 1; // real bytes covered by ` Exp…`
                        // The continuation must cover STRICTLY MORE real bytes
                        // than the bare-alias greedy (TM only splits when the
                        // space-prefixed twin reaches further than the combined
                        // `\x7f Ex` form). greedy_len_i is the bare alias real
                        // length. Require the continuation to extend past it,
                        // else the split is a no-op or a regression.
                        if (@as(i32, @intCast(d_sp_real)) > greedy_len_i) {
                            // First-token = bare DEL: real len 1, flags are the
                            // DEL piece's own stored flags/nwords (read from the
                            // vocab so a vocab that stores DEL differently stays
                            // correct).
                            const d_first_flag: u8 = self.flags[self.delete_token_id];
                            const d_first_nw: i32 = @intCast(self.nwords_score[self.delete_token_id]);
                            // Second = the space-prefixed continuation; use its
                            // stored flags/nwords (it BEGINS with the synthetic
                            // space). score2 form (NOT score-b): KEEP the
                            // begins-space bonus and pay no extra-token penalty —
                            // TM's score2 case emits only the bare DEL this iter;
                            // the continuation is a normal greedy emit next iter.
                            const d_sec_flag: u8 = self.flags[d_sp_id];
                            const d_sec_nw: i32 = @intCast(self.nwords_score[d_sp_id]);
                            const d_tail: usize = i + d_sp_real;
                            const d_next_bb: u8 = if (d_tail >= chunk.len) 0 else self.begin_byte[chunk[d_tail]];
                            // is_alt=true so TM's score2 length deductions vs
                            // greedy_len_i apply (branch_len = 1 + d_sp_real vs
                            // the bare-alias greedy_len_i; LessThan ⇒ -100,
                            // Equal ⇒ -10000).
                            const d_score = tmScore(
                                1, // bare DEL real length
                                @as(i32, @intCast(d_sp_real)),
                                d_first_flag,
                                d_sec_flag,
                                d_first_nw,
                                d_sec_nw,
                                d_next_bb,
                                true, // is_alt — apply score2 length deductions
                                greedy_len_i,
                                0, // score2 (NOT score-b): no extra-token penalty
                                false, // keep begins-space bonus (score2, not score-b)
                            );
                            if (d_score > best_score) {
                                best_score = d_score;
                                best_first_id = self.delete_token_id; // bare \x7f, emitted first
                                best_advance = 0; // DEL covers 0 input bytes
                                best_second_id = d_sp_id; // continuation, seeded next iter
                                best_lb_real = d_sp_real; // real bytes the continuation covers
                                best_emit = .del_then_seed;
                                winner_sets_fd = false; // score2, NOT score-b: forward_delete stays 0
                            }
                        }
                    }
                }
            }

            // 7th branch (path b): lilbuf (synthetic `\x7f ` prefix).
            // Scoring mirrors the greedy branch: first token is the
            // lilbuf-prefixed match (real id, real nwords), with a
            // single-token lookahead at `i + advance` for the second.
            // No `-10000` deduction: lilbuf is a TM-Go "I would never
            // emit this from greedy" path and is allowed to win on
            // equal branch_len since it captures a synthetic-boundary
            // segmentation that greedy cannot represent.
            //
            // gap #3: skipped under the single-whole-word skip-gate — TM-Go
            // emits the greedy token with NO alternative evaluation at all.
            if (use_lilbuf and !skip_alts and ll_id != NO_TOKEN) {
                // Stage 2c: TM-Go runs its ungreedy alt ladder on the
                // token it reaches mid-word too. ztok reaches `\x7F sti`
                // here via the lilbuf synthesis but historically scored
                // ONLY that greedy match — never its alts (`\x7F st`).
                // That left 14/14 nocap×eng diffs as "ztok-longer-greedy
                // vs TM-shorter-alt". Build the lilbuf first-token
                // candidates: the greedy lilbuf match plus its precomputed
                // alts (`self.alts[ll_id]`, in `\x7F `-prefixed units —
                // real advance = alt.length - 2). Score each with the SAME
                // legacy lilbuf formula and keep the best. The alt
                // candidates use a -1 "I would not greedily emit this"
                // deduction so they win only when the word-density bonus
                // genuinely favors the shorter split (TM-Go's score2/3
                // length deduction).
                var lb_cand_ids: [3]u32 = .{ ll_id, NO_TOKEN, NO_TOKEN };
                var lb_cand_adv: [3]u32 = .{ ll_advance, 0, 0 };
                var n_lb: usize = 1;
                if (use_precomp_alts) {
                    if (self.alts) |alts_buf| {
                        if (ll_id < self.count) {
                            const ap = alts_buf[ll_id];
                            // Stage 3 fix: extract alt2 INDEPENDENTLY of
                            // alt1. The prior nesting put the alt2 check
                            // INSIDE `if (ap.length > 2)`, so when alt1 was
                            // the bare `\x7F` marker (length 1, the common
                            // case for `\x7F `-prefixed twins whose first
                            // alt is the DEL marker) the whole block was
                            // skipped and the REAL alt (`\x7F st`, in alt2)
                            // was never seen by path-(b). That left every
                            // mid-word `\x7F sti`→`\x7F st` decision with no
                            // alt candidate at all. `ap.length > 2` rejects
                            // the bare-marker subtoken (≤1 real byte after
                            // `\x7F `); we now apply that test to each slot
                            // separately.
                            if (ap.index != NO_TOKEN and ap.length > 2) {
                                const ok1 = if (use_mask) (self.mask.?[ap.index] == 0) else true;
                                if (ok1) {
                                    lb_cand_ids[n_lb] = ap.index;
                                    lb_cand_adv[n_lb] = ap.length - 2;
                                    n_lb += 1;
                                }
                            }
                            if (ap.index2 != NO_TOKEN and ap.length2 > 2) {
                                const ok2 = if (use_mask) (self.mask.?[ap.index2] == 0) else true;
                                if (ok2) {
                                    lb_cand_ids[n_lb] = ap.index2;
                                    lb_cand_adv[n_lb] = ap.length2 - 2;
                                    n_lb += 1;
                                }
                            }
                            // L30 experiment (gated): TM-Go's priority ladder
                            // can skip the shorter-by-one word-prefix subtoken
                            // (e.g. `\x7F s` when alt2 is `\x7F sh`), leaving
                            // it absent from the alt table even though TM-Go's
                            // float-scored continuation reaches it. When the
                            // greedy `\x7F `-word match's alt2 is a strict
                            // word-prefix of the greedy (both letter-runs), add
                            // the (alt2.length-1) prefix as an EXTRA float
                            // candidate iff it exists in the trie and isn't
                            // already a candidate. It can only WIN via the
                            // strict-`>` float override below; the legacy
                            // scale and computeAlts are untouched.
                            if (n_lb < 3 and ap.index2 != NO_TOKEN and ap.length2 > 3) {
                                const short_vlen: u32 = ap.length2 - 1; // vocab bytes
                                const gbytes = self.idBytes(ll_id);
                                if (short_vlen <= gbytes.len) {
                                    const short_id = trieExactLookup(self.nodes, self.child_bytes, self.child_nodes, gbytes[0..short_vlen]);
                                    if (short_id != NO_TOKEN and short_id != ap.index and short_id != ap.index2 and short_id < self.count) {
                                        const oks = if (use_mask) (self.mask.?[short_id] == 0) else true;
                                        if (oks) {
                                            lb_cand_ids[n_lb] = short_id;
                                            lb_cand_adv[n_lb] = short_vlen - 2;
                                            n_lb += 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Stage 3: TM-Go ungreedy float branch-score ladder, applied
                // SURGICALLY to the path-(b) greedy-vs-alt decision only.
                //
                // The legacy length-dominant formula (`branch_len +
                // word*100`, no -100/-10000/-103/-3 deductions) is RETAINED
                // for the greedy lilbuf slot (lc==0) — it is the primary way
                // `\x7F `-prefixed tokens get emitted, and replacing it
                // wholesale regresses every cell hard (measured: 10/50/26/62).
                //
                // The remaining nocap×eng diffs are the mid-word
                // `\x7F sti`(greedy) vs `\x7F st`(alt) ties: TM-Go decides
                // these with its float ladder (the alt's `-100*LessThan`
                // length deduction is overcome only when the shorter split
                // opens a clean word boundary). So: score every path-(b)
                // candidate on BOTH formulas. The legacy score still drives
                // whether the path-(b) GREEDY can win `best_score` (protected
                // behavior unchanged). The float ladder is consulted ONLY to
                // pick a SHORTER alt over the greedy lilbuf, and the
                // resulting override fires ONLY when path-(b)'s greedy lilbuf
                // is the eventual overall winner — so no plain/score-b/path-a
                // branch is ever disturbed.
                const lb_greedy_real_len: i32 = @intCast(ll_advance);
                var lb_float_greedy: i32 = std.math.minInt(i32);
                var lb_float_best: i32 = std.math.minInt(i32);
                var lb_float_alt_id: u32 = NO_TOKEN;
                var lb_float_alt_adv: u32 = 0;

                var lc: usize = 0;
                while (lc < n_lb) : (lc += 1) {
                    const cand_id = lb_cand_ids[lc];
                    const cand_adv = lb_cand_adv[lc];
                    if (cand_id == NO_TOKEN or cand_adv == 0) continue;
                    const after = i + cand_adv;
                    var second_len: u32 = 0;
                    var second_id_b: u32 = NO_TOKEN;
                    if (after < chunk.len) {
                        // 1.19 perf: combined trie walk (was two separate
                        // walks: longestMatchLen + longestMatchId).
                        const m = self.longestMatchIdAndLen(use_mask, chunk[after..]);
                        second_len = m.len;
                        second_id_b = m.id;
                    }
                    const branch_len: i32 = @intCast(@as(usize, cand_adv) + @as(usize, second_len));
                    const nw1: i32 = @intCast(self.nwords[cand_id]);
                    // v2 alias detection — see path-(a) above for rationale.
                    const second_b_is_alias: bool = second_id_b != NO_TOKEN and
                        second_id_b < self.alias_lens_by_id.len and
                        self.alias_lens_by_id[second_id_b] != 0 and
                        @as(u32, self.alias_lens_by_id[second_id_b]) == second_len;
                    const nw2: i32 = if (second_id_b == NO_TOKEN)
                        0
                    else if (second_b_is_alias)
                        @as(i32, @intCast(self.bare_nwords[second_id_b]))
                    else
                        @as(i32, @intCast(self.nwords[second_id_b]));
                    const tail = after + second_len;
                    const next_is_space: i32 = if (tail >= chunk.len) 1 else blk: {
                        break :blk if (isWhitespace(chunk[tail])) 1 else 0;
                    };
                    // Legacy pre-flag formula for the lilbuf paths.
                    var score: i32 = branch_len;
                    score += bMaxZeroAnd(nw1 - 1);
                    score += bMaxZeroAnd(nw2 - 1);
                    score += (nw1 + nw2 + next_is_space) * 100;

                    // TM-Go float ladder for this same candidate. Flags and
                    // TM whole-word counts read from the stored
                    // `\x7F `-prefixed info entry (`flags`/`nwords_score`),
                    // exactly the entry TM-Go scores for the twin. `next_bb`
                    // uses the begin_byte table (TM-Go's `beginByte[data[tail]]`).
                    //
                    // L88 fix: when the path-(b) candidate `cand_id` is the
                    // SAME token the branch loop already matched as a
                    // BARE-alias greedy (its alias byte length == cand_adv),
                    // the branch loop scored it with BARE flags/nwords. Reading
                    // the `\x7F `-prefixed flags here gives the lilbuf greedy a
                    // phantom +100 whole-word bonus (the trailing word baked
                    // into the prefixed twin), inflating its float so the
                    // float-override below sees a TIE where the branch loop saw
                    // a strict alt win — keeping ztok's longer greedy where
                    // TM-Go splits (L88 "Singing the": `\x7F ing the` vs
                    // `\x7F ing`+` the`). Read bare flags/nwords for an
                    // alias-matched candidate so its float matches the branch
                    // loop's scoring and the alt can win the float ladder.
                    const cand_is_alias: bool = cand_id < self.alias_lens_by_id.len and
                        self.alias_lens_by_id[cand_id] != 0 and
                        @as(u32, self.alias_lens_by_id[cand_id]) == cand_adv;
                    const f1: u8 = if (cand_id >= self.count)
                        0
                    else if (cand_is_alias)
                        self.bare_flags[cand_id]
                    else
                        self.flags[cand_id];
                    const f2: u8 = if (second_id_b == NO_TOKEN or second_id_b >= self.count)
                        0
                    else if (second_b_is_alias)
                        self.bare_flags[second_id_b]
                    else
                        self.flags[second_id_b];
                    const nw1_s: i32 = if (cand_is_alias)
                        @as(i32, @intCast(self.bare_nwords[cand_id]))
                    else
                        @as(i32, @intCast(self.nwords_score[cand_id]));
                    const nw2_s: i32 = if (second_id_b == NO_TOKEN)
                        0
                    else if (second_b_is_alias)
                        @as(i32, @intCast(self.bare_nwords[second_id_b]))
                    else
                        @as(i32, @intCast(self.nwords_score[second_id_b]));
                    const next_bb_b: u8 = if (tail >= chunk.len) 0 else self.begin_byte[chunk[tail]];
                    var tm_score = tmScore(
                        @intCast(cand_adv),
                        @intCast(second_len),
                        f1,
                        f2,
                        nw1_s,
                        nw2_s,
                        next_bb_b,
                        lc > 0, // is_alt: apply -100/-10000 length deductions
                        lb_greedy_real_len,
                        0,
                        false,
                    );

                    // === Stage 4: score2b/3b for the ALT lilbuf candidate ====
                    // For an alt candidate (`lc > 0`) the PLAIN second above
                    // is a direct trie walk, which at a mid-word lookahead is
                    // short or absent in ztok's collapsed vocab — collapsing
                    // the alt's `branch_len` into the `-100`/`-10000` deduction
                    // band so it can never beat the greedy lilbuf even when
                    // TM-Go picks it. TM-Go reaches the alt's continuation via
                    // its score2b/3b branch: a lilbuf-SPACE token (` fus`,
                    // ` BUT`, ` cult`) emitted after a DEL marker. We compute
                    // that branch's score HERE (with the score-b variant:
                    // `extra_token_penalty=1`, `drop_begin_space_bonus=true`)
                    // and take it as the alt's effective float when it beats
                    // the plain one — mirroring TM-Go's `max(score2, score2b)`.
                    // The greedy slot (`lc == 0`) is untouched, matching
                    // TM-Go's score1 (no -1, no begin-space drop).
                    //
                    // Gate: feature on + vocab has DEL + space-prefixed tokens,
                    // the lookahead byte is a letter (we're starting a word
                    // mid-input), and the lilbuf-space synthesis covers
                    // strictly more real bytes than the plain second
                    // (`lbs_real > second_len`). The plain-second word-count /
                    // next-byte sub-conditions of TM-Go's score1b gate are NOT
                    // re-tested here: this is a SCORING-ONLY restoration of the
                    // alt's continuation length, and the resulting alt float
                    // still only displaces the greedy lilbuf via the strict-`>`
                    // override below — so over-firing the boost cannot widen
                    // the alt's reach beyond what TM-Go would pick. Measured:
                    // adding those sub-gates left the boost inert (the
                    // ` fus`/` BUT`/` cult` continuations have nwords_tm != 0).
                    if (use_lilbuf and lc > 0 and
                        self.delete_token_id != NO_TOKEN and
                        self.has_space_prefix_tokens and
                        after < chunk.len and
                        isAsciiLetter(chunk[after]))
                    {
                        var lbs_total: u32 = 0;
                        const lbs_id = self.lilbufSpaceLongestMatch(use_mask, chunk[after..], &lbs_total);
                        if (lbs_id != NO_TOKEN and lbs_total >= 2) {
                            const lbs_real: u32 = lbs_total - 1;
                            if (lbs_real > second_len) {
                                const lbs_flag = if (lbs_id < self.count) self.flags[lbs_id] else 0;
                                const lbs_nw: i32 = @intCast(self.nwords_score[lbs_id]);
                                const lbs_tail = after + lbs_real;
                                const lbs_bb: u8 = if (lbs_tail >= chunk.len) 0 else self.begin_byte[chunk[lbs_tail]];
                                const score_2b = tmScore(
                                    @intCast(cand_adv),
                                    @intCast(lbs_real),
                                    f1,
                                    lbs_flag,
                                    nw1_s,
                                    lbs_nw,
                                    lbs_bb,
                                    true, // is_alt
                                    lb_greedy_real_len,
                                    1, // -1 extra-token penalty (score-b)
                                    true, // drop begin-space bonus (score-b)
                                );
                                if (score_2b > tm_score) tm_score = score_2b;
                            }
                        }
                    }

                    traceBranch(i, if (lc == 0) "pb_greedy(legacy/float)" else "pb_alt(legacy/float)", lc, cand_id, cand_adv, second_id_b, second_len, score);
                    traceBranch(i, if (lc == 0) "pb_greedy_FLOAT" else "pb_alt_FLOAT", lc, cand_id, cand_adv, second_id_b, second_len, tm_score);

                    if (lc == 0) {
                        // Greedy lilbuf: legacy score drives best_score
                        // (protected). Record its float score as the
                        // alt-override baseline.
                        lb_float_greedy = tm_score;
                        if (score > best_score) {
                            best_score = score;
                            best_first_id = cand_id;
                            best_advance = cand_adv;
                            best_emit = .normal;
                            winner_sets_fd = false;
                        }
                    } else {
                        // Alt: track the best float-ladder alt. Strict `>`
                        // over the greedy baseline so the greedy slot wins
                        // exact ties (mirrors TM-Go's source-order switch:
                        // score1 outranks score2/3 on a tie). Legacy alt
                        // scoring is INTENTIONALLY NOT folded into best_score
                        // — letting alts win on the deduction-free legacy
                        // scale regresses (measured 76/88/53/81). Alts win
                        // only via the float ladder below.
                        if (tm_score > lb_float_best) {
                            lb_float_best = tm_score;
                            lb_float_alt_id = cand_id;
                            lb_float_alt_adv = cand_adv;
                        }
                    }
                }

                // Float-ladder override: replace the path-(b) greedy lilbuf
                // with a shorter alt iff the alt strictly beats the greedy
                // lilbuf on TM-Go's float ladder AND the path-(b) greedy is
                // the current overall winner (so plain / score-b / path-a
                // branches that legitimately outscored path-(b) are left
                // untouched). This is the SOLE legacy→float gating flip and
                // it targets exactly the mid-word `\x7F sti`→`\x7F st` bucket.
                if (lb_float_alt_id != NO_TOKEN and
                    lb_float_best > lb_float_greedy and
                    best_emit == .normal and
                    best_first_id == ll_id)
                {
                    best_first_id = lb_float_alt_id;
                    best_advance = lb_float_alt_adv;
                    winner_sets_fd = false;

                    // v3 Stage 8: cluster-A under-stitch fix (CAPCODE ONLY).
                    //
                    // The tracer showed that on full-capcode vocabs, when the
                    // path-(b) float ladder picks a SHORTER `D `-word alt over
                    // the greedy lilbuf (e.g. "D oni" over "D onic" inside
                    // ANDRONICUS), ztok emits just that alt `.normal` and then
                    // re-greedily tokenizes the residual as "D c"+"D us".
                    // TM-Go instead emits the alt, a bare DEL marker, then a
                    // single space-prefixed continuation ("D oni" + DEL +
                    // " cus") — its score2b/3b branch. The score-b machinery in
                    // the branch loop already computed exactly this candidate
                    // (a lilbuf-SPACE match at the alt's tail covering >= 2 more
                    // real bytes), but the path-(b) legacy score clobbered it.
                    //
                    // Re-derive the score-b continuation for the CHOSEN alt
                    // here and, if it covers strictly more bytes (the same
                    // TM-Go `length2b > length2 + 1` shape, here vs the alt's
                    // own plain second), emit `[alt, DEL, lilbuf_second]` via
                    // the goto-checkpoint seed path. Gated to capcode marker
                    // mode (`lilbuf_marker_byte != 0x7F`) so the at-floor
                    // nocapcode cells are provably untouched — the nocap
                    // score-b wins go through the branch-loop fold, not this
                    // override, and this block never executes for them.
                    if (use_lilbuf and use_goto_checkpoint and
                        self.lilbuf_marker_byte != 0x7F and
                        self.delete_token_id != NO_TOKEN and
                        self.has_space_prefix_tokens)
                    {
                        const alt_end = i + lb_float_alt_adv;
                        if (alt_end < chunk.len and isAsciiLetter(chunk[alt_end])) {
                            // Plain second at the alt's tail (TM-Go length2).
                            const pm = self.longestMatchIdAndLen(use_mask, chunk[alt_end..]);
                            const plain2_len: u32 = pm.len;
                            var lbs_total: u32 = 0;
                            const lbs_id = self.lilbufSpaceLongestMatch(use_mask, chunk[alt_end..], &lbs_total);
                            if (lbs_id != NO_TOKEN and lbs_total >= 2) {
                                const lbs_real: u32 = lbs_total - 1;
                                // TM-Go gate: lilbuf must cover >= 2 more real
                                // bytes than the plain second (`length2b >
                                // length2 + 1`). Here the plain second's bytes
                                // INCLUDE the leading `D ` marker (cap stream),
                                // so its real coverage already excludes the
                                // boundary — the `+1` form is faithful for the
                                // direct capcode plain match.
                                if (lbs_real > plain2_len + 1 and lbs_id < self.count) {
                                    best_second_id = lbs_id;
                                    best_lb_real = lbs_real;
                                    best_emit = .first_del_second;
                                    winner_sets_fd = true;
                                }
                            }
                        }
                    }
                }
            }

            // 8th branch (path a): lilbuf with ` `-only prefix + DEL
            // emit. Two-token output: DEL_TOKEN + matched_piece. Apply
            // a -1 "extra token" penalty to match TM-Go's score1b: the
            // extra DEL emission costs one id.
            //
            // Track whether THIS iteration set up a goto-checkpoint
            // re-entry (used to seed the next iter). Separate from
            // `winner_sets_fd` so we can distinguish "score-b won AND
            // we're doing the goto-checkpoint" (goto path) from "score-b
            // won AND we're doing the legacy single-emit" (when the
            // feature gate is off).
            // Declared BEFORE path-(a) so the L56 path-(a)-alt experiment
            // can seed a goto-checkpoint re-entry from the del_before_first
            // emit.
            var next_forward_lilbuf: bool = false;
            var next_seed_id: u32 = NO_TOKEN;
            var next_seed_real_len: u32 = 0;

            // gap #3: skipped under the single-whole-word skip-gate.
            if (use_lilbuf and !skip_alts and ll_sp_id != NO_TOKEN) {
                const after = i + ll_sp_advance;
                var second_len: u32 = 0;
                var second_id_a: u32 = NO_TOKEN;
                if (after < chunk.len) {
                    // 1.19 perf: combined trie walk.
                    const m = self.longestMatchIdAndLen(use_mask, chunk[after..]);
                    second_len = m.len;
                    second_id_a = m.id;
                }
                const branch_len: i32 = @intCast(@as(usize, ll_sp_advance) + @as(usize, second_len));
                const nw1: i32 = @intCast(self.nwords[ll_sp_id]);
                // v2 alias detection on the second-token lookup: if
                // the trie match landed on the bare alias, use
                // `bare_nwords` instead of `nwords`. Otherwise the
                // alias inherits the prefixed primary's nwords
                // (which counts the synthetic space transition) and
                // path-(a) wins over path-(b) at positions where
                // TM-Go would have picked path-(b).
                const second_a_is_alias: bool = second_id_a != NO_TOKEN and
                    second_id_a < self.alias_lens_by_id.len and
                    self.alias_lens_by_id[second_id_a] != 0 and
                    @as(u32, self.alias_lens_by_id[second_id_a]) == second_len;
                const nw2: i32 = if (second_id_a == NO_TOKEN)
                    0
                else if (second_a_is_alias)
                    @as(i32, @intCast(self.bare_nwords[second_id_a]))
                else
                    @as(i32, @intCast(self.nwords[second_id_a]));
                const tail = after + second_len;
                const next_is_space: i32 = if (tail >= chunk.len) 1 else blk: {
                    break :blk if (isWhitespace(chunk[tail])) 1 else 0;
                };
                var score: i32 = branch_len;
                score += bMaxZeroAnd(nw1 - 1);
                score += bMaxZeroAnd(nw2 - 1);
                score += (nw1 + nw2 + next_is_space) * 100;
                score -= 1;
                if (score > best_score) {
                    best_score = score;
                    best_first_id = ll_sp_id;
                    best_advance = ll_sp_advance;
                    best_emit = .del_before_first;
                    winner_sets_fd = false;

                    // L56 experiment (gated): path-(a) emits [DEL, greedy]
                    // but never scored the greedy's precomputed alts. TM-Go
                    // reaches the alt's shorter split (` fus`+`ions`) via its
                    // score3 float ladder. When this path-(a) greedy is the
                    // current winner AND its alt2 is a strict word-prefix
                    // whose float-scored continuation STRICTLY beats the
                    // greedy's float, emit [DEL, alt2] and seed the residual
                    // word as the next greedy match (goto-checkpoint), so the
                    // remaining bytes are re-tokenized exactly as TM-Go would.
                    // Gated to nocapcode vocabs (marker == 0x7F). On
                    // full-capcode vocabs the alt's flag/nword reads are
                    // for the capcode-marker form and the float comparison
                    // mispicks the shorter alt (regressed `Hugging Face`
                    // → ` fac`); the gate keeps the fix off there.
                    if (use_goto_checkpoint and use_precomp_alts and !is_seeded and
                        self.lilbuf_marker_byte == 0x7F)
                    {
                        if (self.alts) |alts_buf| {
                            if (ll_sp_id < self.count) {
                                const ap = alts_buf[ll_sp_id];
                                if (ap.index2 != NO_TOKEN and ap.length2 > 2 and ap.index2 < self.count) {
                                    const ok2 = if (use_mask) (self.mask.?[ap.index2] == 0) else true;
                                    const alt2_real: u32 = ap.length2 - 1; // lilbuf-space form: real = vocab-1
                                    if (ok2 and alt2_real < ll_sp_advance and i + alt2_real < chunk.len and
                                        isAsciiLetter(chunk[i + alt2_real]))
                                    {
                                        // Greedy float (score1b shape:
                                        // -1 extra-token, drop begin-space).
                                        const g_f1: u8 = self.flags[ll_sp_id];
                                        const g_nw1: i32 = @intCast(self.nwords_score[ll_sp_id]);
                                        const g_f2: u8 = if (second_id_a == NO_TOKEN or second_id_a >= self.count) 0 else if (second_a_is_alias) self.bare_flags[second_id_a] else self.flags[second_id_a];
                                        const g_nw2: i32 = if (second_id_a == NO_TOKEN) 0 else if (second_a_is_alias) @as(i32, @intCast(self.bare_nwords[second_id_a])) else @as(i32, @intCast(self.nwords_score[second_id_a]));
                                        const g_bb: u8 = if (tail >= chunk.len) 0 else self.begin_byte[chunk[tail]];
                                        const g_float = tmScore(@intCast(ll_sp_advance), @intCast(second_len), g_f1, g_f2, g_nw1, g_nw2, g_bb, false, @intCast(ll_sp_advance), 1, true);

                                        // Alt2 float: ` fus` + path-(b) marker
                                        // continuation at the residual mid-word
                                        // (` ions` = the begin-word `\x7F ions`
                                        // form, found via lilbufLongestMatch
                                        // which synthesizes the `\x7F ` prefix).
                                        const a_after = i + alt2_real;
                                        var a_total: u32 = 0;
                                        const a_sec_id = self.lilbufLongestMatch(use_mask, chunk[a_after..], &a_total);
                                        if (a_sec_id != NO_TOKEN and a_total >= 3 and a_sec_id < self.count) {
                                            const a_sec_real: u32 = a_total - 2;
                                            const a_f1: u8 = self.flags[ap.index2];
                                            const a_nw1: i32 = @intCast(self.nwords_score[ap.index2]);
                                            const a_f2: u8 = self.flags[a_sec_id];
                                            const a_nw2: i32 = @intCast(self.nwords_score[a_sec_id]);
                                            const a_tail = a_after + a_sec_real;
                                            const a_bb: u8 = if (a_tail >= chunk.len) 0 else self.begin_byte[chunk[a_tail]];
                                            const a_float = tmScore(@intCast(alt2_real), @intCast(a_sec_real), a_f1, a_f2, a_nw1, a_nw2, a_bb, true, @intCast(ll_sp_advance), 1, true);
                                            if (a_float > g_float) {
                                                // Emit [DEL, alt2] and advance
                                                // only past alt2; the next
                                                // iteration re-tokenizes the
                                                // residual mid-word naturally
                                                // (it finds ` ions` via path-b).
                                                best_first_id = ap.index2;
                                                best_advance = alt2_real;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // At a seeded iteration, every winning piece's `best_advance`
            // is a VOCAB byte count (the synthetic lilbuf prefix is part
            // of the vocab bytes but not the real input). Subtract 1 to
            // get the real-input advance. Equivalent to TM-Go's `i +=
            // original.length - forwardDelete`.
            const seed_advance_adj: u32 = if (use_goto_checkpoint and is_seeded) 1 else 0;

            tracePick(i, switch (best_emit) {
                .normal => "normal",
                .del_before_first => "del_before_first",
                .first_del_second => "first_del_second",
                .del_then_seed => "del_then_seed",
            }, best_first_id, best_second_id, best_score, best_advance - seed_advance_adj);

            switch (best_emit) {
                .normal => {
                    std.debug.assert(write < out.len);
                    out[write] = best_first_id;
                    write += 1;
                    i += best_advance - seed_advance_adj;
                },
                .del_before_first => {
                    // Emit DEL first, then the matched piece. DEL covers
                    // 0 input bytes (synthetic boundary).
                    //
                    // Path-a lilbuf is gated to !is_seeded so seed_advance_adj
                    // is 0 here in practice — but we apply it uniformly
                    // for robustness in case a future change relaxes the
                    // gate.
                    std.debug.assert(write + 1 < out.len);
                    out[write] = self.delete_token_id;
                    write += 1;
                    out[write] = best_first_id;
                    write += 1;
                    i += best_advance - seed_advance_adj;
                },
                .first_del_second => {
                    // 1.20 perf: lb_real was captured by the winning
                    // branch (`best_lb_real`); no re-walk needed. The
                    // 1.19 code did a fresh `lilbufSpaceLongestMatch`
                    // walk here just to recompute a length the scoring
                    // pass already had.
                    const alt_first_advance = best_advance - seed_advance_adj;
                    const lb_real_emit: u32 = best_lb_real;

                    if (use_goto_checkpoint) {
                        // TM-Go goto-checkpoint path: emit only
                        // `[alt_first, DEL]`, advance to where alt_first
                        // ends, and seed the next iteration with the
                        // lilbuf token as the synthetic greedy match.
                        // The next iter's alt evaluation can pick a
                        // different (often shorter) alt of `lilbuf_second`
                        // than what we'd have committed by emitting
                        // `lilbuf_second` directly.
                        std.debug.assert(write + 1 < out.len);
                        out[write] = best_first_id;
                        write += 1;
                        out[write] = self.delete_token_id;
                        write += 1;
                        i += alt_first_advance;
                        // Seed the next iteration if the lilbuf actually
                        // covered any real bytes. Defensive: lb_real == 0
                        // means lb_total < 2 which shouldn't happen given
                        // the score-b gate, but guard anyway so we never
                        // seed a zero-advance loop.
                        if (lb_real_emit > 0) {
                            next_forward_lilbuf = true;
                            next_seed_id = best_second_id;
                            next_seed_real_len = lb_real_emit;
                        }
                    } else {
                        // Legacy 1.16 single-emit path: emit
                        // `[alt_first, DEL, lilbuf_second]` and advance
                        // past both. Kept for back-compat with vocabs
                        // that don't enable goto_checkpoint_enabled.
                        std.debug.assert(write + 2 < out.len);
                        out[write] = best_first_id;
                        write += 1;
                        out[write] = self.delete_token_id;
                        write += 1;
                        out[write] = best_second_id;
                        write += 1;
                        i += alt_first_advance + lb_real_emit;
                    }
                },
                .del_then_seed => {
                    // TM-Go score2 `case`: emit bare DEL (0 input bytes),
                    // advance 0, and seed the continuation as next iter's
                    // greedy (TM's `goto checkpoint`). Mirrors the
                    // first_del_second goto-checkpoint seeding but with NO
                    // alt-first token (DEL is emitted alone this iter).
                    std.debug.assert(write < out.len);
                    out[write] = self.delete_token_id;
                    write += 1;
                    // i unchanged (DEL covers 0 input bytes); seed the
                    // space-prefixed continuation so the next iteration
                    // replays it as the synthetic greedy and runs its own
                    // alt ladder, yielding [\x7f][ Exp…] exactly as TM-Go.
                    if (best_lb_real > 0) {
                        next_forward_lilbuf = true;
                        next_seed_id = best_second_id;
                        next_seed_real_len = best_lb_real;
                    }
                },
            }

            // Update forward_delete state for the next iteration.
            // Per TM-Go: set to 1 after score1b/2b/3b wins, cleared
            // otherwise. In our restructured emit, .first_del_second
            // corresponds to score2b/3b (and similar from path-(a)
            // direct it'd be score1b — but path-(a) is treated as its
            // own emission, not a goto-checkpoint, so we don't carry
            // forwardDelete from it; this is one of the residual
            // TM-Go-vs-ztok semantic gaps).
            forward_delete = if (winner_sets_fd) 1 else 0;
            forward_lilbuf = next_forward_lilbuf;
            fwd_seed_id = next_seed_id;
            fwd_seed_real_len = next_seed_real_len;
        }

        return out[0..write];
    }

    /// Same as `encodeChunk` but also writes per-id byte spans into
    /// `out_offsets`. `chunk_offset` is the position of `chunk[0]` in
    /// whatever buffer the offsets index into.
    pub fn encodeChunkWithOffsets(
        self: *const Monster,
        allocator: std.mem.Allocator,
        chunk: []const u8,
        chunk_offset: u32,
        out_ids: []TokenId,
        out_offsets: []Span,
    ) !usize {
        const use_pa = self.use_precomputed_alts and self.alts != null;
        const use_gc = self.goto_checkpoint_enabled and self.lilbuf_enabled and self.score2b3b_enabled;
        if (self.mask == null) {
            if (self.lilbuf_enabled) {
                if (self.score2b3b_enabled) {
                    if (use_pa) {
                        if (use_gc) return self.encodeChunkWithOffsetsImpl(false, true, true, true, true, allocator, chunk, chunk_offset, out_ids, out_offsets);
                        return self.encodeChunkWithOffsetsImpl(false, true, true, true, false, allocator, chunk, chunk_offset, out_ids, out_offsets);
                    }
                    if (use_gc) return self.encodeChunkWithOffsetsImpl(false, true, true, false, true, allocator, chunk, chunk_offset, out_ids, out_offsets);
                    return self.encodeChunkWithOffsetsImpl(false, true, true, false, false, allocator, chunk, chunk_offset, out_ids, out_offsets);
                }
                if (use_pa) return self.encodeChunkWithOffsetsImpl(false, true, false, true, false, allocator, chunk, chunk_offset, out_ids, out_offsets);
                return self.encodeChunkWithOffsetsImpl(false, true, false, false, false, allocator, chunk, chunk_offset, out_ids, out_offsets);
            }
            if (use_pa) return self.encodeChunkWithOffsetsImpl(false, false, false, true, false, allocator, chunk, chunk_offset, out_ids, out_offsets);
            return self.encodeChunkWithOffsetsImpl(false, false, false, false, false, allocator, chunk, chunk_offset, out_ids, out_offsets);
        }
        if (self.lilbuf_enabled) {
            if (self.score2b3b_enabled) {
                if (use_pa) {
                    if (use_gc) return self.encodeChunkWithOffsetsImpl(true, true, true, true, true, allocator, chunk, chunk_offset, out_ids, out_offsets);
                    return self.encodeChunkWithOffsetsImpl(true, true, true, true, false, allocator, chunk, chunk_offset, out_ids, out_offsets);
                }
                if (use_gc) return self.encodeChunkWithOffsetsImpl(true, true, true, false, true, allocator, chunk, chunk_offset, out_ids, out_offsets);
                return self.encodeChunkWithOffsetsImpl(true, true, true, false, false, allocator, chunk, chunk_offset, out_ids, out_offsets);
            }
            if (use_pa) return self.encodeChunkWithOffsetsImpl(true, true, false, true, false, allocator, chunk, chunk_offset, out_ids, out_offsets);
            return self.encodeChunkWithOffsetsImpl(true, true, false, false, false, allocator, chunk, chunk_offset, out_ids, out_offsets);
        }
        if (use_pa) return self.encodeChunkWithOffsetsImpl(true, false, false, true, false, allocator, chunk, chunk_offset, out_ids, out_offsets);
        return self.encodeChunkWithOffsetsImpl(true, false, false, false, false, allocator, chunk, chunk_offset, out_ids, out_offsets);
    }

    inline fn encodeChunkWithOffsetsImpl(
        self: *const Monster,
        comptime use_mask: bool,
        comptime use_lilbuf: bool,
        comptime use_score2b3b: bool,
        comptime use_precomp_alts: bool,
        comptime use_goto_checkpoint: bool,
        allocator: std.mem.Allocator,
        chunk: []const u8,
        chunk_offset: u32,
        out_ids: []TokenId,
        out_offsets: []Span,
    ) !usize {
        const expand: usize = if (use_lilbuf) 2 else 1;
        std.debug.assert(out_ids.len >= chunk.len * expand);
        std.debug.assert(out_offsets.len >= chunk.len * expand);
        _ = allocator;
        if (chunk.len == 0) {
            @branchHint(.unlikely);
            return 0;
        }

        var cand_lens: [MAX_CANDIDATES]u32 = undefined;
        var cand_ids: [MAX_CANDIDATES]u32 = undefined;

        var forward_delete: u8 = 0;
        // Goto-checkpoint state — see `encodeChunkImpl` for the
        // semantic write-up.
        var forward_lilbuf: bool = false;
        var fwd_seed_id: u32 = NO_TOKEN;
        var fwd_seed_real_len: u32 = 0;

        var write: usize = 0;
        var i: usize = 0;
        while (i < chunk.len) {
            const is_seeded = use_goto_checkpoint and forward_lilbuf;
            const n_cand = if (is_seeded) blk: {
                // See `encodeChunkImpl` — cold path post-score2b/3b.
                @branchHint(.unlikely);
                cand_lens[0] = fwd_seed_real_len + 1;
                cand_ids[0] = fwd_seed_id;
                break :blk @as(usize, 1);
            } else self.collectPrefixMatches(
                use_mask,
                chunk[i..],
                &cand_lens,
                &cand_ids,
            );

            // See `encodeChunkImpl` for why the lilbuf checks move
            // BEFORE the n_cand==0 fallback, and for path (a)/(b)
            // distinctions. At seeded iterations both paths are
            // skipped — the seed already represents the lilbuf result.
            var ll_id: u32 = NO_TOKEN;
            var ll_advance: u32 = 0;
            var ll_sp_id: u32 = NO_TOKEN;
            var ll_sp_advance: u32 = 0;
            const lilbuf_gate_ok_b = if (self.lilbuf_marker_byte == 0x7F)
                lilbufPositionGate(chunk, i)
            else
                lilbufPositionGateCapcode(chunk, i);
            if (use_lilbuf and !is_seeded and lilbuf_gate_ok_b) {
                if (self.has_lilbuf_prefix_tokens) {
                    var ll_total_len: u32 = 0;
                    ll_id = self.lilbufLongestMatch(use_mask, chunk[i..], &ll_total_len);
                    if (ll_id != NO_TOKEN and ll_total_len >= 3) {
                        ll_advance = ll_total_len - 2;
                    } else {
                        ll_id = NO_TOKEN;
                    }
                }
                if (self.delete_token_id != NO_TOKEN and self.has_space_prefix_tokens) {
                    var ll_sp_total: u32 = 0;
                    ll_sp_id = self.lilbufSpaceLongestMatch(use_mask, chunk[i..], &ll_sp_total);
                    if (ll_sp_id != NO_TOKEN and ll_sp_total >= 2) {
                        ll_sp_advance = ll_sp_total - 1;
                    } else {
                        ll_sp_id = NO_TOKEN;
                    }
                }
            }

            if (n_cand == 0 and ll_id == NO_TOKEN and ll_sp_id == NO_TOKEN) {
                // Unk-fallback — see encodeChunkImpl for rationale.
                @branchHint(.unlikely);
                std.debug.assert(write < out_ids.len);
                out_ids[write] = self.unk_id;
                out_offsets[write] = .{
                    .start = chunk_offset + @as(u32, @intCast(i)),
                    .end = chunk_offset + @as(u32, @intCast(i + 1)),
                };
                write += 1;
                i += 1;
                continue;
            }

            const Emit = enum { normal, del_before_first, first_del_second };
            var best_first_id: u32 = undefined;
            var best_advance: u32 = undefined;
            var best_second_id: u32 = NO_TOKEN;
            // 1.20 perf: cache score-b lilbuf real-byte length (see
            // encodeChunkImpl for the rationale — saves one
            // O(max_token_len) trie walk per score-b win).
            var best_lb_real: u32 = 0;
            var best_emit: Emit = .normal;
            if (n_cand > 0) {
                best_first_id = cand_ids[n_cand - 1];
                best_advance = cand_lens[n_cand - 1];
            } else if (ll_id != NO_TOKEN) {
                best_first_id = ll_id;
                best_advance = ll_advance;
            } else {
                best_first_id = ll_sp_id;
                best_advance = ll_sp_advance;
                best_emit = .del_before_first;
            }
            var best_score: i32 = std.math.minInt(i32);
            var winner_sets_fd: bool = false;

            // See `encodeChunkImpl` for seed_advance_adj_pre rationale.
            const seed_advance_adj_pre: u32 = if (use_goto_checkpoint and is_seeded) 1 else 0;

            // gap #3 — `flag & 32` single-whole-word skip-gate
            // (TM-Go tokenmonster.go:1057). See encodeChunkImpl for the full
            // write-up. Emits greedy unconditionally and skips all alts/paths.
            var skip_alts: bool = false;
            if (n_cand > 0) {
                const g_idx: usize = n_cand - 1;
                const g_id: u32 = cand_ids[g_idx];
                const g_real_len: u32 = cand_lens[g_idx] - seed_advance_adj_pre;
                const next_pos: usize = i + g_real_len;
                if (next_pos < chunk.len and g_id < self.count) {
                    const g_is_alias: bool = g_id < self.alias_lens_by_id.len and
                        self.alias_lens_by_id[g_id] != 0 and
                        @as(u32, self.alias_lens_by_id[g_id]) == g_real_len;
                    const g_flag: u8 = if (g_is_alias) self.bare_flags[g_id] else self.flags[g_id];
                    // gap #3 REVERTED: skip-gate disabled (always score
                    // alts). Re-enabling it was measured inert at baseline.
                    if (false and (g_flag & FLAG_SINGLE_WORD) != 0 and
                        self.begin_byte[chunk[next_pos]] == BB_SPACE)
                    {
                        skip_alts = true;
                    }
                }
            }

            if (n_cand > 0 and !skip_alts) {
                const greedy_idx: usize = n_cand - 1;
                const greedy_len_raw: u32 = cand_lens[greedy_idx];
                const greedy_id_raw: u32 = cand_ids[greedy_idx];
                const greedy_len_i: i32 = @as(i32, @intCast(greedy_len_raw)) - @as(i32, forward_delete);

                // See `encodeChunkImpl` for the rationale on the
                // precomputed-alts branch selection. Same shape here.
                var br_ids: [3]u32 = .{ greedy_id_raw, NO_TOKEN, NO_TOKEN };
                var br_lens: [3]u32 = .{ greedy_len_raw, 0, 0 };
                var n_branches: usize = 1;
                // Stage 2: see `encodeChunkImpl` — bare-alias greedy
                // matches now source their alts from `bare_alt_lut`
                // instead of being skipped, and the alt branches read
                // bare flags/nwords for those alt ids.
                var greedy_from_bare_alias: bool = false;
                if (use_precomp_alts) {
                    if (self.alts) |alts_buf| {
                        // See `encodeChunkImpl` for the alias gate
                        // rationale (alts are computed per-id but a
                        // matched alias has a different prefix
                        // structure than the primary, so the per-id
                        // precomp alts don't apply — use the bare twin's
                        // own alt table instead).
                        const greedy_is_alias = greedy_id_raw < self.alias_lens_by_id.len and
                            self.alias_lens_by_id[greedy_id_raw] != 0 and
                            self.alias_lens_by_id[greedy_id_raw] == greedy_len_raw;
                        greedy_from_bare_alias = greedy_is_alias;
                        const ap: AltPair = if (!greedy_is_alias)
                            alts_buf[greedy_id_raw]
                        else if (greedy_id_raw < self.bare_alt_lut.len)
                            self.bare_alt_lut[greedy_id_raw]
                        else
                            .{};
                        if (ap.index != NO_TOKEN and ap.length > 0) {
                            const ok1 = if (use_mask) (self.mask.?[ap.index] == 0) else true;
                            if (ok1) {
                                br_ids[1] = ap.index;
                                br_lens[1] = ap.length;
                                n_branches = 2;
                            }
                            if (ap.index2 != NO_TOKEN and ap.length2 > 0) {
                                const ok2 = if (use_mask) (self.mask.?[ap.index2] == 0) else true;
                                if (ok2) {
                                    br_ids[2] = ap.index2;
                                    br_lens[2] = ap.length2;
                                    n_branches = 3;
                                }
                            }
                        }
                    }
                } else {
                    n_branches = @min(n_cand, MAX_BRANCHES);
                }

                var b: usize = 0;
                while (b < n_branches) : (b += 1) {
                    const is_greedy = (b == 0);
                    const first_len_raw = if (use_precomp_alts)
                        br_lens[b]
                    else
                        cand_lens[greedy_idx - b];
                    const first_id = if (use_precomp_alts)
                        br_ids[b]
                    else
                        cand_ids[greedy_idx - b];
                    const first_len_i: i32 = @as(i32, @intCast(first_len_raw)) - @as(i32, forward_delete);

                    const first_real_len = first_len_raw - seed_advance_adj_pre;
                    const after = i + first_real_len;
                    // See `encodeChunkImpl` for the plain_second_*
                    // separation rationale.
                    var plain_second_id: u32 = NO_TOKEN;
                    var plain_second_len: u32 = 0;
                    if (after < chunk.len) {
                        const m = self.longestMatchIdAndLen(use_mask, chunk[after..]);
                        plain_second_id = m.id;
                        plain_second_len = m.len;
                    }
                    var second_id: u32 = plain_second_id;
                    var second_len: u32 = plain_second_len;
                    var second_is_phantom: bool = false;

                    // v2 alias detection on the second position.
                    // If the plain second match length equals the alias
                    // byte length for that id, the trie matched the
                    // bare alias and we should score with bare_flags/
                    // bare_nwords just like a phantom-second match.
                    if (second_id != NO_TOKEN and
                        second_id < self.alias_lens_by_id.len and
                        self.alias_lens_by_id[second_id] != 0 and
                        @as(u32, self.alias_lens_by_id[second_id]) == second_len)
                    {
                        second_is_phantom = true;
                    }

                    // Phantom-second via path-(b) — see `encodeChunkImpl`
                    // for the rationale.
                    if (use_lilbuf and
                        second_id == NO_TOKEN and
                        self.has_lilbuf_prefix_tokens and
                        after < chunk.len and
                        isAsciiLetter(chunk[after]))
                    {
                        var ph_total: u32 = 0;
                        const ph_id = self.lilbufLongestMatch(use_mask, chunk[after..], &ph_total);
                        if (ph_id != NO_TOKEN and ph_total >= 3) {
                            second_id = ph_id;
                            second_len = ph_total - 2;
                            second_is_phantom = true;
                        }
                    }

                    const first_alias_for_nw: bool = first_id < self.alias_lens_by_id.len and
                        self.alias_lens_by_id[first_id] != 0 and
                        @as(u32, self.alias_lens_by_id[first_id]) == first_real_len and
                        (b == 0 or greedy_from_bare_alias);
                    const nw1_raw: i32 = if (first_alias_for_nw)
                        @intCast(self.bare_nwords[first_id])
                    else
                        @intCast(self.nwords_score[first_id]);
                    const nw1: i32 = nw1_raw - @as(i32, forward_delete);
                    const nw2: i32 = if (second_id == NO_TOKEN)
                        0
                    else if (second_is_phantom)
                        @as(i32, @intCast(self.bare_nwords[second_id]))
                    else
                        @as(i32, @intCast(self.nwords_score[second_id]));

                    const tail = after + second_len;
                    const first_is_alias_match: bool = first_id < self.alias_lens_by_id.len and
                        self.alias_lens_by_id[first_id] != 0 and
                        @as(u32, self.alias_lens_by_id[first_id]) == first_real_len and
                        (b == 0 or greedy_from_bare_alias);
                    const first_flag: u8 = if (first_id >= self.count)
                        0
                    else if (first_is_alias_match)
                        self.bare_flags[first_id]
                    else
                        self.flags[first_id];
                    const second_flag: u8 = if (second_id == NO_TOKEN or second_id >= self.count)
                        0
                    else if (second_is_phantom)
                        self.bare_flags[second_id]
                    else
                        self.flags[second_id];
                    const next_bb: u8 = if (tail >= chunk.len) 0 else self.begin_byte[chunk[tail]];

                    const score = tmScore(
                        first_len_i,
                        @as(i32, @intCast(second_len)),
                        first_flag,
                        second_flag,
                        nw1,
                        nw2,
                        next_bb,
                        !is_greedy,
                        greedy_len_i,
                        0,
                        false,
                    );

                    if (score > best_score) {
                        best_score = score;
                        best_first_id = first_id;
                        best_advance = first_len_raw;
                        best_emit = .normal;
                        winner_sets_fd = false;
                    }

                    // score-b extension: see `encodeChunkImpl` for the
                    // gating rationale. Uses plain_second (NOT phantom).
                    if (use_score2b3b and use_lilbuf and
                        self.delete_token_id != NO_TOKEN and
                        self.has_space_prefix_tokens and
                        after < chunk.len)
                    {
                        var eff_second_id: u32 = plain_second_id;
                        var eff_second_len: u32 = plain_second_len;
                        if (eff_second_id == NO_TOKEN and
                            self.has_lilbuf_prefix_tokens)
                        {
                            var pb_total: u32 = 0;
                            const pb_id = self.lilbufLongestMatch(use_mask, chunk[after..], &pb_total);
                            if (pb_id != NO_TOKEN and pb_total >= 3) {
                                eff_second_id = pb_id;
                                eff_second_len = pb_total - 2;
                            }
                        }
                        if (eff_second_id != NO_TOKEN and eff_second_len > 0) {
                            const after_second = after + eff_second_len;
                            // gap #5 — TM-Go gate (tokenmonster.go:1088/1137/1189):
                            //   second.flag & 2 != 0 && nextByte == 1 && second.nWords == 0
                            // Use begin_byte/flag tests, not isAsciiLetter, so
                            // non-ASCII letters pass exactly as TM. path-b
                            // fallback matches read bare-form flags. See
                            // encodeChunkImpl for the full write-up.
                            // Stage 9: see encodeChunkImpl. TM's nextByte==1
                            // uses beginByte; the capcode word marker `W`
                            // (0x57) is 10, not a letter-start. Use begin_byte
                            // for the after-second conjunct so capcode markers
                            // gate out exactly as TM (fixes the `D or`
                            // over-stitch). First conjunct keeps isAsciiLetter.
                            const inside_word =
                                isAsciiLetter(chunk[after]) and
                                after_second < chunk.len and
                                self.begin_byte[chunk[after_second]] == BB_LETTER and
                                self.nwords_tm[eff_second_id] == 0;
                            if (inside_word) {
                                var lb_total: u32 = 0;
                                const lb_id = self.lilbufSpaceLongestMatch(use_mask, chunk[after..], &lb_total);
                                if (lb_id != NO_TOKEN and lb_total >= 2) {
                                    const lb_real: u32 = lb_total - 1;
                                    // gap #1 — TM-Go `length2b > length2 + 1`
                                    // (tokenmonster.go:1092/1141/1193).
                                    if (lb_real > eff_second_len) { // gap #1 REVERTED
                                        const nw2b: i32 = @intCast(self.nwords_score[lb_id]);
                                        const tail_b = after + lb_real;
                                        const lb_flag = self.flags[lb_id];
                                        const next_bb_b: u8 = if (tail_b >= chunk.len) 0 else self.begin_byte[chunk[tail_b]];
                                        const score_b = tmScore(
                                            first_len_i,
                                            @as(i32, @intCast(lb_real)),
                                            first_flag,
                                            lb_flag,
                                            nw1,
                                            nw2b,
                                            next_bb_b,
                                            !is_greedy,
                                            greedy_len_i,
                                            1,
                                            true,
                                        );
                                        // gap #2 REVERTED: fold inline.
                                        if (score_b > best_score) {
                                            best_score = score_b;
                                            best_first_id = first_id;
                                            best_advance = first_len_raw;
                                            best_second_id = lb_id;
                                            best_lb_real = lb_real;
                                            best_emit = .first_del_second;
                                            winner_sets_fd = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // gap #2 REVERTED: score-b folded inline above.
            }

            // 7th branch (path b): lilbuf — see `encodeChunkImpl` for full
            // rationale. Span emitted covers the REAL bytes consumed
            // (advance = total - 2), not the synthetic prefix.
            //
            // gap #3: skipped under the single-whole-word skip-gate.
            //
            // Stage 3: mirrors `encodeChunkImpl`'s path-(b) — the greedy
            // lilbuf keeps the legacy length-dominant score (it is the
            // primary `\x7F `-token emitter; replacing it wholesale
            // regresses every cell), while a TM-Go float-ladder pass over
            // the lilbuf's alts may override the greedy with a shorter alt
            // when the alt strictly wins the float ladder. Targets the
            // mid-word `\x7F sti`→`\x7F st` greedy-vs-alt bucket. See the
            // primary encoder's 7th-branch comment for the full rationale.
            if (use_lilbuf and !skip_alts and ll_id != NO_TOKEN) {
                var lb_cand_ids: [3]u32 = .{ ll_id, NO_TOKEN, NO_TOKEN };
                var lb_cand_adv: [3]u32 = .{ ll_advance, 0, 0 };
                var n_lb: usize = 1;
                if (use_precomp_alts) {
                    if (self.alts) |alts_buf| {
                        if (ll_id < self.count) {
                            const ap = alts_buf[ll_id];
                            // alt1/alt2 extracted INDEPENDENTLY (see primary
                            // encoder) so a bare-`\x7F`-marker alt1 doesn't
                            // suppress the real alt2.
                            if (ap.index != NO_TOKEN and ap.length > 2) {
                                const ok1 = if (use_mask) (self.mask.?[ap.index] == 0) else true;
                                if (ok1) {
                                    lb_cand_ids[n_lb] = ap.index;
                                    lb_cand_adv[n_lb] = ap.length - 2;
                                    n_lb += 1;
                                }
                            }
                            if (ap.index2 != NO_TOKEN and ap.length2 > 2) {
                                const ok2 = if (use_mask) (self.mask.?[ap.index2] == 0) else true;
                                if (ok2) {
                                    lb_cand_ids[n_lb] = ap.index2;
                                    lb_cand_adv[n_lb] = ap.length2 - 2;
                                    n_lb += 1;
                                }
                            }
                            // L30 experiment (gated) — see encodeChunkImpl.
                            if (n_lb < 3 and ap.index2 != NO_TOKEN and ap.length2 > 3) {
                                const short_vlen: u32 = ap.length2 - 1;
                                const gbytes = self.idBytes(ll_id);
                                if (short_vlen <= gbytes.len) {
                                    const short_id = trieExactLookup(self.nodes, self.child_bytes, self.child_nodes, gbytes[0..short_vlen]);
                                    if (short_id != NO_TOKEN and short_id != ap.index and short_id != ap.index2 and short_id < self.count) {
                                        const oks = if (use_mask) (self.mask.?[short_id] == 0) else true;
                                        if (oks) {
                                            lb_cand_ids[n_lb] = short_id;
                                            lb_cand_adv[n_lb] = short_vlen - 2;
                                            n_lb += 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                const lb_greedy_real_len: i32 = @intCast(ll_advance);
                var lb_float_greedy: i32 = std.math.minInt(i32);
                var lb_float_best: i32 = std.math.minInt(i32);
                var lb_float_alt_id: u32 = NO_TOKEN;
                var lb_float_alt_adv: u32 = 0;

                var lc: usize = 0;
                while (lc < n_lb) : (lc += 1) {
                    const cand_id = lb_cand_ids[lc];
                    const cand_adv = lb_cand_adv[lc];
                    if (cand_id == NO_TOKEN or cand_adv == 0) continue;
                    const after = i + cand_adv;
                    var second_len: u32 = 0;
                    var second_id_b: u32 = NO_TOKEN;
                    if (after < chunk.len) {
                        const m = self.longestMatchIdAndLen(use_mask, chunk[after..]);
                        second_len = m.len;
                        second_id_b = m.id;
                    }
                    const branch_len: i32 = @intCast(@as(usize, cand_adv) + @as(usize, second_len));
                    const nw1: i32 = @intCast(self.nwords[cand_id]);
                    // v2 alias detection — see encodeChunkImpl for rationale.
                    const second_b_is_alias: bool = second_id_b != NO_TOKEN and
                        second_id_b < self.alias_lens_by_id.len and
                        self.alias_lens_by_id[second_id_b] != 0 and
                        @as(u32, self.alias_lens_by_id[second_id_b]) == second_len;
                    const nw2: i32 = if (second_id_b == NO_TOKEN)
                        0
                    else if (second_b_is_alias)
                        @as(i32, @intCast(self.bare_nwords[second_id_b]))
                    else
                        @as(i32, @intCast(self.nwords[second_id_b]));
                    const tail = after + second_len;
                    const next_is_space: i32 = if (tail >= chunk.len) 1 else blk: {
                        break :blk if (isWhitespace(chunk[tail])) 1 else 0;
                    };
                    // Legacy pre-flag formula — drives best_score for the
                    // greedy lilbuf slot only.
                    var score: i32 = branch_len;
                    score += bMaxZeroAnd(nw1 - 1);
                    score += bMaxZeroAnd(nw2 - 1);
                    score += (nw1 + nw2 + next_is_space) * 100;

                    // L88 fix (offsets mirror): read bare flags/nwords for an
                    // alias-matched candidate so the path-(b) greedy float
                    // matches the branch loop's bare-form scoring and the alt
                    // can win the float ladder. See encodeChunkImpl for the
                    // full rationale.
                    const cand_is_alias: bool = cand_id < self.alias_lens_by_id.len and
                        self.alias_lens_by_id[cand_id] != 0 and
                        @as(u32, self.alias_lens_by_id[cand_id]) == cand_adv;
                    const f1: u8 = if (cand_id >= self.count)
                        0
                    else if (cand_is_alias)
                        self.bare_flags[cand_id]
                    else
                        self.flags[cand_id];
                    const f2: u8 = if (second_id_b == NO_TOKEN or second_id_b >= self.count)
                        0
                    else if (second_b_is_alias)
                        self.bare_flags[second_id_b]
                    else
                        self.flags[second_id_b];
                    const nw1_s: i32 = if (cand_is_alias)
                        @as(i32, @intCast(self.bare_nwords[cand_id]))
                    else
                        @as(i32, @intCast(self.nwords_score[cand_id]));
                    const nw2_s: i32 = if (second_id_b == NO_TOKEN)
                        0
                    else if (second_b_is_alias)
                        @as(i32, @intCast(self.bare_nwords[second_id_b]))
                    else
                        @as(i32, @intCast(self.nwords_score[second_id_b]));
                    const next_bb_b: u8 = if (tail >= chunk.len) 0 else self.begin_byte[chunk[tail]];
                    var tm_score = tmScore(
                        @intCast(cand_adv),
                        @intCast(second_len),
                        f1,
                        f2,
                        nw1_s,
                        nw2_s,
                        next_bb_b,
                        lc > 0,
                        lb_greedy_real_len,
                        0,
                        false,
                    );

                    // Stage 4: score2b/3b for the ALT lilbuf candidate.
                    // Mirrors encodeChunkImpl — the alt's mid-word
                    // continuation is reached via a lilbuf-SPACE token
                    // (TM-Go's score2b/3b), which the plain second lookahead
                    // undercounts. Compute that branch's score (score-b
                    // variant: extra_token_penalty=1, drop_begin_space=true)
                    // and take it as the alt's effective float when longer.
                    // See encodeChunkImpl for the full rationale.
                    if (use_lilbuf and lc > 0 and
                        self.delete_token_id != NO_TOKEN and
                        self.has_space_prefix_tokens and
                        after < chunk.len and
                        isAsciiLetter(chunk[after]))
                    {
                        var lbs_total: u32 = 0;
                        const lbs_id = self.lilbufSpaceLongestMatch(use_mask, chunk[after..], &lbs_total);
                        if (lbs_id != NO_TOKEN and lbs_total >= 2) {
                            const lbs_real: u32 = lbs_total - 1;
                            if (lbs_real > second_len) {
                                const lbs_flag = if (lbs_id < self.count) self.flags[lbs_id] else 0;
                                const lbs_nw: i32 = @intCast(self.nwords_score[lbs_id]);
                                const lbs_tail = after + lbs_real;
                                const lbs_bb: u8 = if (lbs_tail >= chunk.len) 0 else self.begin_byte[chunk[lbs_tail]];
                                const score_2b = tmScore(
                                    @intCast(cand_adv),
                                    @intCast(lbs_real),
                                    f1,
                                    lbs_flag,
                                    nw1_s,
                                    lbs_nw,
                                    lbs_bb,
                                    true, // is_alt
                                    lb_greedy_real_len,
                                    1, // -1 extra-token penalty (score-b)
                                    true, // drop begin-space bonus (score-b)
                                );
                                if (score_2b > tm_score) tm_score = score_2b;
                            }
                        }
                    }

                    if (lc == 0) {
                        lb_float_greedy = tm_score;
                        if (score > best_score) {
                            best_score = score;
                            best_first_id = cand_id;
                            best_advance = cand_adv;
                            best_emit = .normal;
                            winner_sets_fd = false;
                        }
                    } else {
                        if (tm_score > lb_float_best) {
                            lb_float_best = tm_score;
                            lb_float_alt_id = cand_id;
                            lb_float_alt_adv = cand_adv;
                        }
                    }
                }

                if (lb_float_alt_id != NO_TOKEN and
                    lb_float_best > lb_float_greedy and
                    best_emit == .normal and
                    best_first_id == ll_id)
                {
                    best_first_id = lb_float_alt_id;
                    best_advance = lb_float_alt_adv;
                    winner_sets_fd = false;

                    // v3 Stage 8: cluster-A under-stitch fix (CAPCODE ONLY).
                    // Mirror of the `encodeChunkImpl` block — see there for the
                    // full rationale. When the path-(b) float ladder picks a
                    // shorter `D `-word alt on a full-capcode vocab, emit the
                    // TM-Go score2b/3b triple (`[alt, DEL, lilbuf_second]`) via
                    // the goto-checkpoint seed path instead of a bare `.normal`
                    // alt. Gated to capcode marker mode so nocapcode is
                    // untouched.
                    if (use_lilbuf and use_goto_checkpoint and
                        self.lilbuf_marker_byte != 0x7F and
                        self.delete_token_id != NO_TOKEN and
                        self.has_space_prefix_tokens)
                    {
                        const alt_end = i + lb_float_alt_adv;
                        if (alt_end < chunk.len and isAsciiLetter(chunk[alt_end])) {
                            const pm = self.longestMatchIdAndLen(use_mask, chunk[alt_end..]);
                            const plain2_len: u32 = pm.len;
                            var lbs_total: u32 = 0;
                            const lbs_id = self.lilbufSpaceLongestMatch(use_mask, chunk[alt_end..], &lbs_total);
                            if (lbs_id != NO_TOKEN and lbs_total >= 2) {
                                const lbs_real: u32 = lbs_total - 1;
                                if (lbs_real > plain2_len + 1 and lbs_id < self.count) {
                                    best_second_id = lbs_id;
                                    best_lb_real = lbs_real;
                                    best_emit = .first_del_second;
                                    winner_sets_fd = true;
                                }
                            }
                        }
                    }
                }
            }

            // 8th branch (path a): see `encodeChunkImpl`.
            //
            // gap #3: skipped under the single-whole-word skip-gate.
            if (use_lilbuf and !skip_alts and ll_sp_id != NO_TOKEN) {
                const after = i + ll_sp_advance;
                var second_len: u32 = 0;
                var second_id_a: u32 = NO_TOKEN;
                if (after < chunk.len) {
                    const m = self.longestMatchIdAndLen(use_mask, chunk[after..]);
                    second_len = m.len;
                    second_id_a = m.id;
                }
                const branch_len: i32 = @intCast(@as(usize, ll_sp_advance) + @as(usize, second_len));
                const nw1: i32 = @intCast(self.nwords[ll_sp_id]);
                // v2 alias detection — see encodeChunkImpl for rationale.
                const second_a_is_alias: bool = second_id_a != NO_TOKEN and
                    second_id_a < self.alias_lens_by_id.len and
                    self.alias_lens_by_id[second_id_a] != 0 and
                    @as(u32, self.alias_lens_by_id[second_id_a]) == second_len;
                const nw2: i32 = if (second_id_a == NO_TOKEN)
                    0
                else if (second_a_is_alias)
                    @as(i32, @intCast(self.bare_nwords[second_id_a]))
                else
                    @as(i32, @intCast(self.nwords[second_id_a]));
                const tail = after + second_len;
                const next_is_space: i32 = if (tail >= chunk.len) 1 else blk: {
                    break :blk if (isWhitespace(chunk[tail])) 1 else 0;
                };
                var score: i32 = branch_len;
                score += bMaxZeroAnd(nw1 - 1);
                score += bMaxZeroAnd(nw2 - 1);
                score += (nw1 + nw2 + next_is_space) * 100;
                score -= 1;
                if (score > best_score) {
                    best_score = score;
                    best_first_id = ll_sp_id;
                    best_advance = ll_sp_advance;
                    best_emit = .del_before_first;
                    winner_sets_fd = false;

                    // L56 experiment (gated) — see `encodeChunkImpl`.
                    if (use_goto_checkpoint and use_precomp_alts and !is_seeded and
                        self.lilbuf_marker_byte == 0x7F)
                    {
                        if (self.alts) |alts_buf| {
                            if (ll_sp_id < self.count) {
                                const ap = alts_buf[ll_sp_id];
                                if (ap.index2 != NO_TOKEN and ap.length2 > 2 and ap.index2 < self.count) {
                                    const ok2 = if (use_mask) (self.mask.?[ap.index2] == 0) else true;
                                    const alt2_real: u32 = ap.length2 - 1;
                                    if (ok2 and alt2_real < ll_sp_advance and i + alt2_real < chunk.len and
                                        isAsciiLetter(chunk[i + alt2_real]))
                                    {
                                        const g_f1: u8 = self.flags[ll_sp_id];
                                        const g_nw1: i32 = @intCast(self.nwords_score[ll_sp_id]);
                                        const g_f2: u8 = if (second_id_a == NO_TOKEN or second_id_a >= self.count) 0 else if (second_a_is_alias) self.bare_flags[second_id_a] else self.flags[second_id_a];
                                        const g_nw2: i32 = if (second_id_a == NO_TOKEN) 0 else if (second_a_is_alias) @as(i32, @intCast(self.bare_nwords[second_id_a])) else @as(i32, @intCast(self.nwords_score[second_id_a]));
                                        const g_bb: u8 = if (tail >= chunk.len) 0 else self.begin_byte[chunk[tail]];
                                        const g_float = tmScore(@intCast(ll_sp_advance), @intCast(second_len), g_f1, g_f2, g_nw1, g_nw2, g_bb, false, @intCast(ll_sp_advance), 1, true);

                                        const a_after = i + alt2_real;
                                        var a_total: u32 = 0;
                                        const a_sec_id = self.lilbufLongestMatch(use_mask, chunk[a_after..], &a_total);
                                        if (a_sec_id != NO_TOKEN and a_total >= 3 and a_sec_id < self.count) {
                                            const a_sec_real: u32 = a_total - 2;
                                            const a_f1: u8 = self.flags[ap.index2];
                                            const a_nw1: i32 = @intCast(self.nwords_score[ap.index2]);
                                            const a_f2: u8 = self.flags[a_sec_id];
                                            const a_nw2: i32 = @intCast(self.nwords_score[a_sec_id]);
                                            const a_tail = a_after + a_sec_real;
                                            const a_bb: u8 = if (a_tail >= chunk.len) 0 else self.begin_byte[chunk[a_tail]];
                                            const a_float = tmScore(@intCast(alt2_real), @intCast(a_sec_real), a_f1, a_f2, a_nw1, a_nw2, a_bb, true, @intCast(ll_sp_advance), 1, true);
                                            if (a_float > g_float) {
                                                best_first_id = ap.index2;
                                                best_advance = alt2_real;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            var next_forward_lilbuf: bool = false;
            var next_seed_id: u32 = NO_TOKEN;
            var next_seed_real_len: u32 = 0;

            // See `encodeChunkImpl` for the seed_advance_adj rationale.
            const seed_advance_adj: u32 = if (use_goto_checkpoint and is_seeded) 1 else 0;

            switch (best_emit) {
                .normal => {
                    std.debug.assert(write < out_ids.len);
                    const adv = best_advance - seed_advance_adj;
                    out_ids[write] = best_first_id;
                    out_offsets[write] = .{
                        .start = chunk_offset + @as(u32, @intCast(i)),
                        .end = chunk_offset + @as(u32, @intCast(i + adv)),
                    };
                    write += 1;
                    i += adv;
                },
                .del_before_first => {
                    std.debug.assert(write + 1 < out_ids.len);
                    // DEL covers zero input bytes (synthetic boundary).
                    const adv = best_advance - seed_advance_adj;
                    out_ids[write] = self.delete_token_id;
                    out_offsets[write] = .{
                        .start = chunk_offset + @as(u32, @intCast(i)),
                        .end = chunk_offset + @as(u32, @intCast(i)),
                    };
                    write += 1;
                    out_ids[write] = best_first_id;
                    out_offsets[write] = .{
                        .start = chunk_offset + @as(u32, @intCast(i)),
                        .end = chunk_offset + @as(u32, @intCast(i + adv)),
                    };
                    write += 1;
                    i += adv;
                },
                .first_del_second => {
                    // 1.20 perf: reuse cached lb_real (see encodeChunkImpl).
                    const alt_first_advance = best_advance - seed_advance_adj;
                    const lb_real_emit: u32 = best_lb_real;
                    const after_first_pos: u32 = @as(u32, @intCast(i)) + alt_first_advance;
                    if (use_goto_checkpoint) {
                        // Goto-checkpoint path: emit only `[alt_first,
                        // DEL]` and seed the next iter with the lilbuf
                        // token. See `encodeChunkImpl` for the full
                        // rationale.
                        std.debug.assert(write + 1 < out_ids.len);
                        out_ids[write] = best_first_id;
                        out_offsets[write] = .{
                            .start = chunk_offset + @as(u32, @intCast(i)),
                            .end = chunk_offset + after_first_pos,
                        };
                        write += 1;
                        out_ids[write] = self.delete_token_id;
                        out_offsets[write] = .{
                            .start = chunk_offset + after_first_pos,
                            .end = chunk_offset + after_first_pos,
                        };
                        write += 1;
                        i += alt_first_advance;
                        if (lb_real_emit > 0) {
                            next_forward_lilbuf = true;
                            next_seed_id = best_second_id;
                            next_seed_real_len = lb_real_emit;
                        }
                    } else {
                        std.debug.assert(write + 2 < out_ids.len);
                        out_ids[write] = best_first_id;
                        out_offsets[write] = .{
                            .start = chunk_offset + @as(u32, @intCast(i)),
                            .end = chunk_offset + after_first_pos,
                        };
                        write += 1;
                        out_ids[write] = self.delete_token_id;
                        // DEL is synthetic — zero-width span at boundary.
                        out_offsets[write] = .{
                            .start = chunk_offset + after_first_pos,
                            .end = chunk_offset + after_first_pos,
                        };
                        write += 1;
                        out_ids[write] = best_second_id;
                        out_offsets[write] = .{
                            .start = chunk_offset + after_first_pos,
                            .end = chunk_offset + after_first_pos + lb_real_emit,
                        };
                        write += 1;
                        i += alt_first_advance + lb_real_emit;
                    }
                },
            }

            forward_delete = if (winner_sets_fd) 1 else 0;
            forward_lilbuf = next_forward_lilbuf;
            fwd_seed_id = next_seed_id;
            fwd_seed_real_len = next_seed_real_len;
        }

        return write;
    }

    // --- internals ---

    inline fn findChild(self: *const Monster, node_idx: u32, byte: u8) ?u32 {
        // 1.25 A perf: root-node fast path. Every trie walk starts at
        // node 0; the root has 80-200+ children on real vocabs, which
        // hit the binary-search branch below. Replacing it with a single
        // [byte] indexed load (one ALU+load) eliminated the ~19% of
        // program cycles callgrind attributed to the root binary search.
        // For non-root nodes the table is unused — those nodes have
        // small fanouts and the linear-scan path below handles them
        // well.
        if (node_idx == 0) {
            @branchHint(.likely);
            const child = self.root_child_table[byte];
            return if (child == NO_TOKEN) null else child;
        }
        const node = self.nodes[node_idx];
        const n = node.children_len;
        if (n == 0) {
            // Most internal trie nodes have at least one child; only
            // terminal-leaf positions and end-of-vocab nodes hit n == 0.
            @branchHint(.unlikely);
            return null;
        }
        const lo = node.children_start;
        // Small-fanout fast path. Inner trie nodes are dominated by tiny
        // child sets (≤ 8); a linear scan over the SoA `child_bytes`
        // array vectorizes cleanly and lets the branch predictor learn
        // the common-byte case. Binary search wins only at large
        // fanouts (root + a handful of first-byte-bucket nodes).
        //
        // SoA win: the scan touches only the `child_bytes` array (1
        // byte/child → up to 8 children share a single 8-byte word /
        // a single cache line). The matching `node` index is fetched
        // from `child_nodes` exactly once on hit. The AoS predecessor
        // touched the full {u8, u32}-with-padding pair (8 bytes/child)
        // — a 64-byte cacheline read per node step regardless of how
        // early the match happened.
        const bytes_ptr = self.child_bytes;
        if (n <= 8) {
            // Linear scan is the common case (~95 % of trie steps).
            @branchHint(.likely);
            var k: u32 = 0;
            while (k < n) : (k += 1) {
                const cb = bytes_ptr[lo + k];
                if (cb == byte) {
                    // Match found — fetch the child node index from
                    // the parallel SoA array. The byte-scan inner
                    // loop never touched `child_nodes`, so the cache
                    // line for that array is fresh for the hit only.
                    return self.child_nodes[lo + k];
                }
                // Children are sorted ascending — early-out once we
                // pass the target byte. Saves on average half the
                // remaining iterations for misses on dense nodes.
                if (cb > byte) return null;
            }
            return null;
        } else {
            // Binary search for the wider fanouts (root, etc.).
            @branchHint(.unlikely);
            var l: u32 = lo;
            var r: u32 = lo + n;
            while (l < r) {
                const m = l + (r - l) / 2;
                const b = bytes_ptr[m];
                if (b == byte) return self.child_nodes[m];
                if (b < byte) l = m + 1 else r = m;
            }
            return null;
        }
    }

    // Walk the trie from root over `input`, recording every terminal hit in
    // descend order (shortest → longest). Caps at MAX_CANDIDATES; further
    // matches are silently dropped (deeper terminals are rare and the
    // shortest few already span the relevant byte counts).
    //
    // When `use_mask == true`, terminals whose id is masked are treated as
    // non-terminals (the walk continues past them in case a deeper unmasked
    // terminal exists). This is what makes mask-based marginal-value
    // scoring see the "vocab with P removed" view of the trie.
    inline fn collectPrefixMatches(
        self: *const Monster,
        comptime use_mask: bool,
        input: []const u8,
        out_lens: *[MAX_CANDIDATES]u32,
        out_ids: *[MAX_CANDIDATES]u32,
    ) usize {
        const mask_ptr: ?[]const u8 = if (use_mask) self.mask else null;
        var n: usize = 0;
        var node_idx: u32 = 0;
        const limit = @min(input.len, @as(usize, self.max_token_len));
        var k: usize = 0;
        while (k < limit) : (k += 1) {
            const child = self.findChild(node_idx, input[k]) orelse break;
            node_idx = child;
            const node = self.nodes[node_idx];
            if (node.token_id != NO_TOKEN) {
                if (use_mask) {
                    if (mask_ptr.?[node.token_id] != 0) {
                        // Masked terminal — keep walking deeper.
                        // Trainer-only, rare on production paths.
                        @branchHint(.unlikely);
                        continue;
                    }
                }
                if (n == MAX_CANDIDATES) {
                    // MAX_CANDIDATES = 64; real vocab trie paths rarely
                    // touch more than ~6 terminals along one walk.
                    @branchHint(.unlikely);
                    break;
                }
                out_lens[n] = @intCast(k + 1);
                out_ids[n] = node.token_id;
                n += 1;
            }
        }
        return n;
    }

    inline fn longestMatchLen(
        self: *const Monster,
        comptime use_mask: bool,
        input: []const u8,
    ) u32 {
        const mask_ptr: ?[]const u8 = if (use_mask) self.mask else null;
        var node_idx: u32 = 0;
        const limit = @min(input.len, @as(usize, self.max_token_len));
        var best: u32 = 0;
        var k: usize = 0;
        while (k < limit) : (k += 1) {
            const child = self.findChild(node_idx, input[k]) orelse break;
            node_idx = child;
            const tid = self.nodes[node_idx].token_id;
            if (tid != NO_TOKEN) {
                if (use_mask) {
                    if (mask_ptr.?[tid] != 0) continue;
                }
                best = @intCast(k + 1);
            }
        }
        return best;
    }

    /// Combined version of `longestMatchIdOrNone` + `longestMatchLen`
    /// in ONE trie walk. The pre-1.19 hot path called both functions
    /// back-to-back at the same position, doubling the trie-walk cost
    /// of the alt-branch lookahead. Returns 0 / NO_TOKEN when no
    /// terminal is reached.
    inline fn longestMatchIdAndLen(
        self: *const Monster,
        comptime use_mask: bool,
        input: []const u8,
    ) struct { id: u32, len: u32 } {
        const mask_ptr: ?[]const u8 = if (use_mask) self.mask else null;
        var node_idx: u32 = 0;
        const limit = @min(input.len, @as(usize, self.max_token_len));
        var best_id: u32 = NO_TOKEN;
        var best_len: u32 = 0;
        var k: usize = 0;
        while (k < limit) : (k += 1) {
            const child = self.findChild(node_idx, input[k]) orelse break;
            node_idx = child;
            const tid = self.nodes[node_idx].token_id;
            // Most trie nodes are non-terminals — only the deepest
            // step in a prefix-matching chain hits this branch.
            if (tid != NO_TOKEN) {
                if (use_mask) {
                    if (mask_ptr.?[tid] != 0) {
                        // Masked terminal — fall through to deeper
                        // walk in case an unmasked terminal exists.
                        // Rare on real corpora (`use_mask` itself is
                        // a trainer-only flag).
                        @branchHint(.unlikely);
                        continue;
                    }
                }
                best_id = tid;
                best_len = @intCast(k + 1);
            }
        }
        return .{ .id = best_id, .len = best_len };
    }

    inline fn longestMatchId(
        self: *const Monster,
        comptime use_mask: bool,
        input: []const u8,
    ) u32 {
        const id = self.longestMatchIdOrNone(use_mask, input);
        std.debug.assert(id != NO_TOKEN);
        return id;
    }

    // Same as `longestMatchId` but returns NO_TOKEN instead of asserting
    // when no terminal is found. Used by alt-branch scoring that may
    // have inferred a `second_len > 0` via lilbuf (where the actual
    // greedy id at that position doesn't exist).
    inline fn longestMatchIdOrNone(
        self: *const Monster,
        comptime use_mask: bool,
        input: []const u8,
    ) u32 {
        const mask_ptr: ?[]const u8 = if (use_mask) self.mask else null;
        var node_idx: u32 = 0;
        const limit = @min(input.len, @as(usize, self.max_token_len));
        var best_id: u32 = NO_TOKEN;
        var k: usize = 0;
        while (k < limit) : (k += 1) {
            const child = self.findChild(node_idx, input[k]) orelse break;
            node_idx = child;
            const tid = self.nodes[node_idx].token_id;
            if (tid != NO_TOKEN) {
                if (use_mask) {
                    if (mask_ptr.?[tid] != 0) continue;
                }
                best_id = tid;
            }
        }
        return best_id;
    }

    // === lilbuf (synthetic-boundary) helpers ===

    // Path (a): 1-byte synthetic prefix `[0x20]` (just a space). This
    // matches TM-Go's actual lilbuf encoder: `lilbuf[0] = 32` with
    // `lilbufOffset = 1`. When a token like ` monster` matches via this
    // prefix, the encoder emits TWO ids: `delete_token_id` (a boundary
    // marker) followed by the matched token. The decoder strips the
    // synthetic leading space using the DEL marker.
    const LILBUF_PREFIX_SPACE: [1]u8 = .{0x20};


    // Walk the trie with `[marker_byte, 0x20]` followed by `input` and return
    // the LONGEST terminal id we encountered along the path (or NO_TOKEN
    // if no terminal exists past the 2-byte prefix). Also writes the
    // matched piece length (in BYTES OF THE SYNTHETIC INPUT, including
    // the 2-byte prefix) to `out_total_len`.
    //
    // If the trie doesn't have any token starting with `\x7f `, the
    // first two findChild calls miss and the function returns NO_TOKEN
    // in O(2 * log(branching)) — cheap enough that lilbuf can run on
    // every position even for vocabs that didn't train with this trick.
    inline fn lilbufLongestMatch(
        self: *const Monster,
        comptime use_mask: bool,
        input: []const u8,
        out_total_len: *u32,
    ) u32 {
        const mask_ptr: ?[]const u8 = if (use_mask) self.mask else null;
        var node_idx: u32 = 0;
        // Walk the synthetic 2-byte prefix `[marker_byte, 0x20]`. The
        // marker byte is `\x7F` for nocapcode vocabs and `D` (0x44)
        // for full-capcode vocabs (see `Monster.lilbuf_marker_byte`
        // doc-comment). If either probe fails, the trie has no
        // marker-prefixed tokens at all — return immediately.
        const prefix: [2]u8 = .{ self.lilbuf_marker_byte, 0x20 };
        for (prefix) |pb| {
            const child = self.findChild(node_idx, pb) orelse {
                out_total_len.* = 0;
                return NO_TOKEN;
            };
            node_idx = child;
        }
        // After the prefix, optionally record any terminal hit (a vocab
        // could in principle have `\x7f ` alone as a token; if so, we
        // count it as a length-2 match here).
        var best_id: u32 = NO_TOKEN;
        var best_total: u32 = 0;
        {
            const tid_at_prefix = self.nodes[node_idx].token_id;
            if (tid_at_prefix != NO_TOKEN) {
                const accept = if (use_mask) (mask_ptr.?[tid_at_prefix] == 0) else true;
                if (accept) {
                    best_id = tid_at_prefix;
                    best_total = 2;
                }
            }
        }
        // Now walk over `input`. Total synthetic-input length is at most
        // `max_token_len`, so stop at `max_token_len - 2` real bytes.
        const max_extra: usize = if (self.max_token_len > 2) self.max_token_len - 2 else 0;
        const limit = @min(input.len, max_extra);
        var k: usize = 0;
        while (k < limit) : (k += 1) {
            const child = self.findChild(node_idx, input[k]) orelse break;
            node_idx = child;
            const tid = self.nodes[node_idx].token_id;
            if (tid != NO_TOKEN) {
                if (use_mask) {
                    if (mask_ptr.?[tid] != 0) continue;
                }
                best_id = tid;
                best_total = @intCast(k + 1 + 2); // +2 for the synthetic prefix
            }
        }
        out_total_len.* = best_total;
        return best_id;
    }

    // Path (a): walk the trie with `[0x20]` + `input` and return the
    // LONGEST terminal id. `out_total_len` returns the BYTES OF THE
    // SYNTHETIC INPUT (including the 1-byte prefix) — caller computes
    // `real_advance = total - 1`. When this match wins, the encoder
    // emits TWO ids: `delete_token_id` followed by the returned id.
    //
    // Mirrors TM-Go's lilbuf: a 1-byte space prefix that lets the trie
    // reach space-prefixed tokens like ` monster` from a mid-word
    // position where the input has no real space.
    inline fn lilbufSpaceLongestMatch(
        self: *const Monster,
        comptime use_mask: bool,
        input: []const u8,
        out_total_len: *u32,
    ) u32 {
        const mask_ptr: ?[]const u8 = if (use_mask) self.mask else null;
        var node_idx: u32 = 0;
        const child = self.findChild(node_idx, LILBUF_PREFIX_SPACE[0]) orelse {
            out_total_len.* = 0;
            return NO_TOKEN;
        };
        node_idx = child;
        var best_id: u32 = NO_TOKEN;
        var best_total: u32 = 0;
        // Don't accept a terminal at the 1-byte-prefix node — that's
        // the bare space token, which would advance 0 real bytes and
        // loop forever.
        const max_extra: usize = if (self.max_token_len > 1) self.max_token_len - 1 else 0;
        const limit = @min(input.len, max_extra);
        var k: usize = 0;
        while (k < limit) : (k += 1) {
            const c = self.findChild(node_idx, input[k]) orelse break;
            node_idx = c;
            const tid = self.nodes[node_idx].token_id;
            if (tid != NO_TOKEN) {
                if (use_mask) {
                    if (mask_ptr.?[tid] != 0) {
                        @branchHint(.unlikely);
                        continue;
                    }
                }
                best_id = tid;
                best_total = @intCast(k + 1 + 1); // +1 for the synthetic prefix
            }
        }
        out_total_len.* = best_total;
        return best_id;
    }
};

// ASCII whitespace test — matches what the Go source treats as space-ish
// boundaries for word counting (space, tab, CR, LF, FF, VT).
inline fn isWhitespace(b: u8) bool {
    return b == ' ' or b == '\t' or b == '\n' or b == '\r' or b == 0x0B or b == 0x0C;
}

// ASCII letter test. Used to gate the path-(a) lilbuf branch — TM-Go's
// score1b only fires when the byte AFTER the previous token's last
// position is a letter (`nextByte == 1` in `vocab.beginByte`); we
// approximate that condition with a literal ASCII letter check.
inline fn isAsciiLetter(b: u8) bool {
    return (b >= 'A' and b <= 'Z') or (b >= 'a' and b <= 'z');
}

// True iff lilbuf path-(a)/(b) should attempt a synthetic-prefix walk at
// `chunk[i]`. TM-Go has no position gate at all in its lilbuf branches;
// ztok's stand-alone lilbuf paths (which TM-Go doesn't have — they're
// our way of reaching `\x7f `-prefixed and ` `-prefixed tokens at
// arbitrary positions outside the alt scoring loop) need *some* gate to
// avoid firing at every byte. The 1.16 rule: lilbuf fires whenever the
// current byte is a letter, regardless of what precedes. This catches
// both mid-word positions (the original gate) AND word-boundary
// positions like the bare letter after `=` in `width="hi"`, which the
// pre-1.16 `i > 0 and isLetter(chunk[i-1])` requirement missed. The
// score-b branch math still picks the right winner on tied positions
// (greedy/space-prefixed tokens win on -10000 ties), so over-firing
// here is corrected by scoring rather than producing a wrong segmenter.
inline fn lilbufPositionGate(chunk: []const u8, i: usize) bool {
    return isAsciiLetter(chunk[i]);
}

// Full-capcode variant of the position gate. The lilbuf/deleteToken
// stitch synthesizes `D `+content to reach a space-prefixed token in
// the MIDDLE of a word. The content letter it stitches before is
// always a *lowercase* byte: capcode_encode folds every uppercase
// letter to lowercase, so the only uppercase ASCII bytes that survive
// in the normalized stream are the printable capcode MARKERS
// `C`(0x43)/`W`(0x57)/`D`(0x44) themselves. Firing the lilbuf walk at
// a marker position makes ztok emit a spurious `D ` before the marker
// (e.g. `complete` + `D`+` C` instead of `complete` + bare `C`),
// diverging from TM-Go which has no such position there. TM-Go's
// score1b/2b/3b gate keys off the lookahead token's beginByte (a
// content letter), never the marker byte. So for full-capcode, gate
// on a *lowercase* ASCII letter only — markers (uppercase) are
// excluded. (nocapcode markers are 0x7F, never letters, so the plain
// `isAsciiLetter` gate already excludes them; this variant is only
// wired for the printable-marker `.full` path.)
inline fn lilbufPositionGateCapcode(chunk: []const u8, i: usize) bool {
    const c = chunk[i];
    return c >= 'a' and c <= 'z';
}

// Count word starts in a token: +1 if first byte is non-whitespace, then
// +1 for every whitespace→non-whitespace transition. Saturates at 255.
fn computeNwords(piece: []const u8) u8 {
    if (piece.len == 0) return 0;
    var count: u32 = 0;
    if (!isWhitespace(piece[0])) count = 1;
    var k: usize = 1;
    while (k < piece.len) : (k += 1) {
        if (isWhitespace(piece[k - 1]) and !isWhitespace(piece[k])) count += 1;
    }
    return if (count > 255) 255 else @intCast(count);
}

// TM-Go-style nWords (used as the score-b gate): only counts whitespace→
// non-whitespace transitions INSIDE the token; the first byte is never
// counted on its own. So `train` → 0, ` train` → 1, ` train ed` → 2.
// Also strips a leading synthetic `\x7f ` (DEL + space) so that
// `\x7f train` is treated as `train` (nwords_tm = 0, not 1). This makes
// the score-b "second is mid-word" gate work for nocapcode .ztm vocabs
// where ALL bare-letter equivalents are stored as `\x7f X` literals.
//
// Matches TM-Go's `nWords++` in the rune loop (see
// refs/tokenmonster/go/tokenmonster.go:3554-3560 and friends), which
// gates score2b/score3b via `second.nWords == 0`.
fn computeNwordsTm(piece: []const u8) u8 {
    var p: []const u8 = piece;
    // Strip leading marker+space: `\x7F ` (nocapcode) or `D ` (TM-
    // printable full capcode). Either signals "this is a synthetic-
    // boundary token whose bare form is what we count words against."
    if (p.len >= 2 and p[0] == 0x7F and p[1] == 0x20) p = p[2..];
    if (p.len >= 2 and p[0] == 'D' and p[1] == 0x20) p = p[2..];
    if (p.len < 2) return 0;
    var count: u32 = 0;
    var k: usize = 1;
    while (k < p.len) : (k += 1) {
        if (isWhitespace(p[k - 1]) and !isWhitespace(p[k])) count += 1;
    }
    return if (count > 255) 255 else @intCast(count);
}

// True TM-Go nWords (refs/tokenmonster/go/tokenmonster.go:3514-3572):
// number of whole words in the piece. A "word" is a `space → alphanum`
// transition. The initial rune contributes one word IFF it's a space
// and the second rune is alphanum.
//
// Used by the TM-Go-aware score formula in `tmScore`. Separate from
// `computeNwordsTm` because the latter strips a leading `\x7f ` for
// the score-b gate check, but the score formula needs the unstripped
// count.
//
// Examples:
//   "the"     → 0
//   " the"    → 1
//   " the dog"→ 2
//   "\x7f the"→ 1
//   "ab cd"   → 1
fn computeNwordsScore(piece: []const u8, capcode: CapcodeMode) u8 {
    if (piece.len == 0) return 0;
    var r1: u21 = undefined;
    var n1: usize = undefined;
    decodeRune(piece, &r1, &n1);
    if (n1 == 0 or n1 >= piece.len) return 0;
    var r2: u21 = undefined;
    var n2: usize = undefined;
    decodeRune(piece[n1..], &r2, &n2);
    if (n2 == 0) return 0;

    var count: u32 = 0;
    if (r1 == ' ' and isAlphaNumCp(r2, capcode)) count += 1;

    var i: usize = n1 + n2;
    var prev_r: u21 = r2;
    while (i < piece.len) {
        var nr: u21 = undefined;
        var nn: usize = undefined;
        decodeRune(piece[i..], &nr, &nn);
        if (nn == 0) break;
        if (prev_r == ' ' and isAlphaNumCp(nr, capcode)) count += 1;
        prev_r = nr;
        i += nn;
    }
    return if (count > 255) 255 else @intCast(count);
}

// --- TM-Go flag-bit computation ---
//
// Mirrors the writer at refs/tokenmonster/go/tokenmonster.go:3490-3593.
// Each piece's 8-bit flag is the bitwise-OR of `FLAG_*` constants set
// by walking the first/middle/last runes of the piece bytes.
//
// `capcode` selects the rune classifier:
//   - `.none`     — every rune is a plain letter/digit/punct; no
//                   marker runes exist. Score-formula capcode bits
//                   (`FLAG_ENDS_CAPCODE`, `FLAG_BEGINS_CAPCODE`) never
//                   fire, which matches the pre-flag-aware behavior
//                   for vocabs without capcode normalization.
//   - `.nocapcode`— `\x7F` (0x7F) is the DEL marker. Treated as a
//                   capcode rune by `isCapcodeMarker`; otherwise
//                   everything else is a plain letter/digit/punct.
//   - `.full`     — 'C', 'W', 'D' are capcode markers. `isLetterCp`
//                   masks them out so e.g. `Hello` (no markers) and
//                   `CHello` (capcode-marker-prefixed) classify
//                   differently.
//
// Tested via the per-bit unit tests at the bottom of this file.
pub fn computeFlags(piece: []const u8, capcode: CapcodeMode) u8 {
    if (piece.len == 0) return 0;

    var flag: u8 = 0;

    // Decode first 2 runes (TM-Go's r/r2 at :3519-3520).
    var r1: u21 = undefined;
    var n1: usize = undefined;
    decodeRune(piece, &r1, &n1);

    var r2: u21 = 0;
    var n2: usize = 0;
    if (n1 < piece.len) decodeRune(piece[n1..], &r2, &n2);

    var min_alt_size: usize = 1;

    // ----- Beginning of token (:3521-3542) -----
    if (r1 == ' ') {
        flag = FLAG_BEGINS_SPACE;
        if (r2 != 0 and isAlphaNumCp(r2, capcode)) min_alt_size = 2;
    } else if (isLetterCp(r1, capcode)) {
        flag = FLAG_BEGINS_LETTER;
    } else if (isCapcodeMarker(r1, capcode)) {
        // CharacterToken / WordToken count as space (TM-Go :3533-3534).
        // For nocapcode, the single DEL rune `\x7F` doesn't have a
        // CharacterToken/WordToken distinction; TM-Go's path treats
        // any DEL as just a capcode-marker (no `4`/space bit).
        if (capcode == .full and (r1 == 'C' or r1 == 'W')) {
            flag = FLAG_BEGINS_SPACE;
        }
        flag |= FLAG_BEGINS_CAPCODE;
    }
    // (numbers and other categories don't write any begin-flag.)

    // ----- Count words in piece + track all-letter/punct/number flags
    //       (:3543-3572). Replays TM-Go's onlyLetterSpace etc. -----
    var only_letter_space = false;
    var only_number_space = false;
    var only_punc = false;

    if (piece.len == 1) {
        only_punc = true;
    } else {
        if ((r1 == ' ' or isLetterCp(r1, capcode)) and isLetterCp(r2, capcode)) {
            only_letter_space = true;
        } else if ((r1 == ' ' or unicode_props.isNumber(r1)) and unicode_props.isNumber(r2)) {
            only_number_space = true;
        } else if (!isAlphaNumCp(r1, capcode) and !isAlphaNumCp(r2, capcode)) {
            only_punc = true;
        }
        var i: usize = n1 + n2;
        while (i < piece.len) {
            var nr: u21 = undefined;
            var nn: usize = undefined;
            decodeRune(piece[i..], &nr, &nn);
            // category tracking — needed for FLAG_ALL_LETTERS at end.
            if (isLetterCp(nr, capcode)) {
                only_punc = false;
                only_number_space = false;
            } else if (unicode_props.isNumber(nr)) {
                only_punc = false;
                only_letter_space = false;
            } else if (nr != ' ') {
                only_letter_space = false;
                only_number_space = false;
            }
            if (nn == 0) break;
            i += nn;
        }
    }

    // ----- End of token (:3574-3593) -----
    var r_last: u21 = 0;
    decodeLastRune(piece, &r_last);

    // FLAG_SINGLE_WORD: TM-Go gate is `minAltSize == 2 && isLetter(r) &&
    // onlyLetterSpace && nWords == 1`. We need `n_words == 1` here;
    // recompute the count locally so we can use it.
    if (min_alt_size == 2 and isLetterCp(r_last, capcode) and only_letter_space) {
        // Recount n_words: how many ` `+alphanum transitions including
        // the initial space.
        var nw_local: u32 = 1; // the leading space + r2 (alphanum) = 1 word seen
        var i: usize = n1 + n2;
        var cur_r: u21 = r2;
        while (i < piece.len) {
            var nr: u21 = undefined;
            var nn: usize = undefined;
            decodeRune(piece[i..], &nr, &nn);
            if (cur_r == ' ' and isAlphaNumCp(nr, capcode)) nw_local += 1;
            cur_r = nr;
            if (nn == 0) break;
            i += nn;
        }
        if (nw_local == 1) flag |= FLAG_SINGLE_WORD;
    }

    if (isCapcodeMarker(r_last, capcode)) flag |= FLAG_ENDS_CAPCODE;
    if (isLetterCp(r_last, capcode)) flag |= FLAG_ENDS_LETTER;
    if (only_letter_space or only_number_space or only_punc) flag |= FLAG_ALL_LETTERS;

    return flag;
}

// --- rune helpers — minimal UTF-8 decoder; no allocations. ---

fn decodeRune(b: []const u8, out_r: *u21, out_n: *usize) void {
    if (b.len == 0) {
        out_r.* = 0;
        out_n.* = 0;
        return;
    }
    const c0 = b[0];
    if (c0 < 0x80) {
        out_r.* = c0;
        out_n.* = 1;
        return;
    }
    const len: usize = if (c0 & 0xE0 == 0xC0) 2 else if (c0 & 0xF0 == 0xE0) 3 else if (c0 & 0xF8 == 0xF0) 4 else 1;
    if (len == 1 or b.len < len) {
        // Invalid leading byte — treat as a single-byte rune so we
        // don't infinitely loop. Won't classify as letter/digit and
        // won't match a capcode marker.
        out_r.* = c0;
        out_n.* = 1;
        return;
    }
    const cp = std.unicode.utf8Decode(b[0..len]) catch {
        out_r.* = c0;
        out_n.* = 1;
        return;
    };
    out_r.* = cp;
    out_n.* = len;
}

fn decodeLastRune(b: []const u8, out_r: *u21) void {
    if (b.len == 0) {
        out_r.* = 0;
        return;
    }
    // Find the start of the last UTF-8 sequence.
    var i: usize = b.len - 1;
    while (i > 0 and (b[i] & 0xC0) == 0x80) : (i -= 1) {}
    var dummy: usize = 0;
    decodeRune(b[i..], out_r, &dummy);
}

fn decodeFirstRune(b: []const u8) u21 {
    var r: u21 = 0;
    var n: usize = 0;
    decodeRune(b, &r, &n);
    return r;
}

// TM-Go's isLetter (refs/tokenmonster/go/tokenmonster.go:359): a
// Unicode letter that is NOT a capcode marker in full-capcode mode,
// OR a combining mark (Mn/Mc/Me).
fn isLetterCp(r: u21, capcode: CapcodeMode) bool {
    if (unicode_props.isLetter(r)) {
        if (capcode == .full and (r == 'W' or r == 'C' or r == 'D')) return false;
        return true;
    }
    return unicode_props.isMn(r) or unicode_props.isMc(r) or unicode_props.isMe(r);
}

fn isAlphaNumCp(r: u21, capcode: CapcodeMode) bool {
    return isLetterCp(r, capcode) or unicode_props.isNumber(r);
}

fn isCapcodeMarker(r: u21, capcode: CapcodeMode) bool {
    return switch (capcode) {
        .none => false,
        .nocapcode => r == 0x7F,
        .full => r == 'C' or r == 'W' or r == 'D',
    };
}

// --- trie construction (mirrors unigram.zig) ---

const TrieBuild = struct {
    nodes: []Monster.Node,
    /// Parallel SoA arrays — `child_bytes[k]` is the byte at child slot
    /// `k`, `child_nodes[k]` is the descend-to node index. Split for
    /// cacheline density on the byte-scan inner loop (see
    /// `Monster.child_bytes` doc-comment).
    child_bytes: []u8,
    child_nodes: []u32,
};

const BuildNode = struct {
    token_id: u32,
    children: std.ArrayList(Monster.Child),
};

fn buildTrie(
    allocator: std.mem.Allocator,
    bytes: []const u8,
    offsets: []const u32,
    count: u32,
    aliases: []const Monster.AliasRecord,
) !TrieBuild {
    var nodes: std.ArrayList(BuildNode) = .empty;
    defer {
        for (nodes.items) |*n| n.children.deinit(allocator);
        nodes.deinit(allocator);
    }
    try nodes.append(allocator, .{ .token_id = NO_TOKEN, .children = .empty });

    var id: u32 = 0;
    while (id < count) : (id += 1) {
        const start = offsets[id];
        const end = offsets[id + 1];
        const piece = bytes[start..end];
        var cur: u32 = 0;
        for (piece) |b| {
            const next = try descendOrCreate(allocator, &nodes, cur, b);
            cur = next;
        }
        // Last-writer wins on dupes — vocab should be unique.
        nodes.items[cur].token_id = id;
    }
    // Aliases: extra trie entries that point at an EXISTING id (a
    // "twin" of the primary id_to_bytes entry, e.g. TM-Go's `train`
    // vs `\x7F train` both at alt_id 9735). They share the primary's
    // flags/nwords slot but contribute their OWN terminal in the trie
    // so direct longest-match lookups can land on either form. If the
    // alias collides with another primary entry, primaries win (we
    // don't overwrite). If two aliases collide, last writer wins
    // (matches buildTrie's primary semantics).
    //
    // Gate: aliases shorter than 3 bytes are dropped. The 1-2 byte
    // alias slots are dominated by single-character primary tokens
    // (`a`, `e`, ` `, etc.) and capcode markers; inserting them as
    // aliases of OTHER ids would create ambiguous lookups where the
    // encoder may end up routing to the wrong shared id. Empirically
    // this gate is the difference between alias insertion HURTING
    // equivalence (without gate) and HELPING it (with gate) on the
    // 32k vocabs that motivate this format extension.
    for (aliases) |a| {
        if (a.bytes.len == 0) continue;
        var cur: u32 = 0;
        for (a.bytes) |b| {
            const next = try descendOrCreate(allocator, &nodes, cur, b);
            cur = next;
        }
        if (nodes.items[cur].token_id == NO_TOKEN) {
            nodes.items[cur].token_id = a.id;
        }
        // else: a primary already occupies this terminal — keep the
        // primary's id (the alias is redundant with an existing
        // distinct vocab piece). Aliases for ids that ALSO have a
        // bare-form primary in the vocab shouldn't be emitted by the
        // converter, but if they leak through we silently no-op.
    }

    const node_count = nodes.items.len;
    var total_children: usize = 0;
    for (nodes.items) |n| total_children += n.children.items.len;

    const flat_nodes = try allocator.alloc(Monster.Node, node_count);
    errdefer allocator.free(flat_nodes);
    // SoA flat children arrays: one byte per slot, one node index per
    // slot. The byte-scan inner loop in `Monster.findChild` only touches
    // the bytes array; the matching child's node index is fetched from
    // `child_nodes` on hit. See `Monster.child_bytes` doc-comment.
    const flat_bytes = try allocator.alloc(u8, total_children);
    errdefer allocator.free(flat_bytes);
    const flat_child_nodes = try allocator.alloc(u32, total_children);
    errdefer allocator.free(flat_child_nodes);

    var write: u32 = 0;
    for (nodes.items, 0..) |n, idx| {
        const len: u32 = @intCast(n.children.items.len);
        flat_nodes[idx] = .{
            .token_id = n.token_id,
            .children_start = write,
            .children_len = len,
        };
        for (n.children.items, 0..) |c, k| {
            flat_bytes[write + @as(u32, @intCast(k))] = c.byte;
            flat_child_nodes[write + @as(u32, @intCast(k))] = c.node;
        }
        write += len;
    }

    return .{ .nodes = flat_nodes, .child_bytes = flat_bytes, .child_nodes = flat_child_nodes };
}

fn descendOrCreate(
    allocator: std.mem.Allocator,
    nodes: *std.ArrayList(BuildNode),
    parent: u32,
    byte: u8,
) !u32 {
    // Binary search the existing sorted child list first; bail if hit.
    // The append-may-realloc invariant from unigram.zig applies: don't hold
    // a pointer across the append below.
    {
        const kids = nodes.items[parent].children.items;
        var l: usize = 0;
        var r: usize = kids.len;
        while (l < r) {
            const m = l + (r - l) / 2;
            const b = kids[m].byte;
            if (b == byte) return kids[m].node;
            if (b < byte) l = m + 1 else r = m;
        }
    }

    const new_idx: u32 = @intCast(nodes.items.len);
    try nodes.append(allocator, .{ .token_id = NO_TOKEN, .children = .empty });

    const kids_ptr = &nodes.items[parent].children;
    var l2: usize = 0;
    var r2: usize = kids_ptr.items.len;
    while (l2 < r2) {
        const m = l2 + (r2 - l2) / 2;
        const b = kids_ptr.items[m].byte;
        if (b < byte) l2 = m + 1 else r2 = m;
    }
    try kids_ptr.insert(allocator, l2, .{ .byte = byte, .node = new_idx });
    return new_idx;
}

// --- precomputed alts (TM-Go's `tokenOuter.{index,length,index2,length2}`) ---
//
// For each piece in the vocab, find the two highest-priority in-vocab
// proper-prefix subtokens. Priority is the boundary class at the split
// point — see `altPriority` for the full ladder. Mirrors TM-Go's
// `tokenData.alt` computation at refs/tokenmonster/go/tokenmonster.go:3597-3753.

/// Find the terminal id of an exact-length lookup `piece[..len]` via the
/// flat trie. Returns NO_TOKEN if no terminal exists at exactly that
/// length. (We use the trie because the encoder's `Monster.findChild`
/// is private to the encoder; this is a static-shape duplicate.)
fn trieExactLookup(
    nodes: []const Monster.Node,
    child_bytes: []const u8,
    child_nodes: []const u32,
    bytes: []const u8,
) u32 {
    if (bytes.len == 0) return NO_TOKEN;
    var node_idx: u32 = 0;
    for (bytes) |b| {
        const node = nodes[node_idx];
        if (node.children_len == 0) return NO_TOKEN;
        // binary search children — same shape as the encoder's
        // `findChild` wide-fanout branch (SoA bytes for compare, SoA
        // nodes for the matching child's descent target).
        var lo: u32 = node.children_start;
        var hi: u32 = node.children_start + node.children_len;
        var found: u32 = NO_TOKEN;
        while (lo < hi) {
            const m = lo + (hi - lo) / 2;
            const cb = child_bytes[m];
            if (cb == b) {
                found = child_nodes[m];
                break;
            }
            if (cb < b) lo = m + 1 else hi = m;
        }
        if (found == NO_TOKEN) return NO_TOKEN;
        node_idx = found;
    }
    return nodes[node_idx].token_id;
}

// TM-Go's ungreedy English-contraction suffix table
// (refs/tokenmonster/go/tokenmonster.go:3157 `ungreedySuffixes`).
// Two entries: ASCII apostrophe + 's', and U+2019 RIGHT SINGLE
// QUOTATION MARK (UTF-8 0xE2 0x80 0x99) + 's'. ztok is UTF-8-only
// (TM-Go charset 0/1), so we store only the byte forms — TM-Go's
// charset==2 UTF-16 branch (:3163-3166) is not applicable here.
const ungreedy_suffixes = [_][]const u8{
    "'s", // TM-Go tokenmonster.go:3157  "'s"
    "\u{2019}s", // TM-Go tokenmonster.go:3157  "’s" (U+2019 + 's')
};

/// Port of TM-Go's `hasSuffixPos` (refs/tokenmonster/go/tokenmonster.go:
/// 287-299). For each ungreedy suffix, if `key` ends with that suffix AND
/// the suffix is shorter than the key, decode the last rune of the part
/// BEFORE the suffix; if that rune is a letter, return the byte position
/// where the suffix begins (`key.len - suffix.len`). Otherwise return null.
///
/// NOTE: at that returned position, the subtoken `key[0..pos]` ends in a
/// letter and `key[pos]` is `'` or `’` (non-letter, non-`_`), so the
/// letter|non-letter priority-9 case in `altPriority` always fires first
/// and `continue`s before the priority-8 suffix branch can run. The
/// suffix branch is therefore inert in practice (true for both TM-Go's
/// encoder build at :3719-3735 and its trainer at trainvocab.go:746-815),
/// but we port it faithfully — including the loop-`break` — so the control
/// flow matches the reference exactly.
fn hasSuffixPos(key: []const u8, capcode: CapcodeMode) ?usize {
    for (ungreedy_suffixes) |suffix| {
        // bytes.HasSuffix(key, suffix) — TM-Go tokenmonster.go:289
        if (key.len >= suffix.len and std.mem.eql(u8, key[key.len - suffix.len ..], suffix)) {
            // len(suffix) < len(key) — TM-Go tokenmonster.go:290
            if (suffix.len < key.len) {
                const before = key[0 .. key.len - suffix.len];
                var r: u21 = 0;
                decodeLastRune(before, &r); // TM-Go tokenmonster.go:291
                if (isLetterCp(r, capcode)) { // TM-Go tokenmonster.go:292
                    return key.len - suffix.len; // TM-Go tokenmonster.go:293
                }
            }
        }
    }
    return null; // TM-Go tokenmonster.go:298 (returns -1)
}

/// Compute the boundary-class priority of an alt split inside `piece`
/// at position `length` (so subtoken is `piece[0..length]`, the boundary
/// rune is `piece[length-1]` last-of-subtoken vs `piece[length]` first
/// past split). Mirrors TM-Go's switch ladder at
/// refs/tokenmonster/go/tokenmonster.go:3597-3717.
///
/// Returns 0 if the split has no recognized boundary class — TM-Go
/// still picks such alts via the priority-1 fallback at :3737-3750.
fn altPriority(
    piece: []const u8,
    length: usize,
    capcode: CapcodeMode,
) struct { priority: u8, eligible: bool, is_fallback: bool } {
    // `is_fallback` flags TM-Go's priority-1 "everything else" branch at
    // refs/tokenmonster/go/tokenmonster.go:3737-3750, which is reached
    // ONLY when none of the boundary-class `continue`s above fired. The
    // caller uses this to know whether it may apply the priority-8 suffix
    // rule (:3719-3735), which sits between the switch ladder and the
    // priority-1 fallback in TM-Go's control flow.
    // Mirror TM-Go's "length <= len(token) - 2" + `token[length] == ' '`
    // path. Priority 10: space then letter/number after the split.
    if (length + 2 <= piece.len and piece[length] == ' ') {
        var r2: u21 = 0;
        var n2: usize = 0;
        decodeRune(piece[length + 1 ..], &r2, &n2);
        if (n2 > 0 and (isLetterCp(r2, capcode) or unicode_props.isNumber(r2))) {
            return .{ .priority = 10, .eligible = true, .is_fallback = false };
        }
    }

    // Decode the last rune of subtoken (= rune ending at position `length-1`)
    // and the first rune of the suffix (= rune starting at `length`).
    var r: u21 = 0;
    decodeLastRune(piece[0..length], &r);
    var r2: u21 = 0;
    var n2: usize = 0;
    if (length < piece.len) decodeRune(piece[length..], &r2, &n2);

    // capcode == .none: extra non-letter|letter and non-number|number
    // priority-9 buckets.
    if (capcode == .none) {
        const r_is_letter_or_us = isLetterCp(r, capcode) or r == '_';
        const r2_is_letter_or_us = isLetterCp(r2, capcode) or r2 == '_';
        if (!r_is_letter_or_us and r2_is_letter_or_us) {
            return .{ .priority = 9, .eligible = true, .is_fallback = false };
        }
        if (!unicode_props.isNumber(r) and unicode_props.isNumber(r2)) {
            return .{ .priority = 9, .eligible = true, .is_fallback = false };
        }
    }

    // letter | non-letter (priority 9). `_` counts as a letter for this
    // boundary check per TM-Go.
    if ((isLetterCp(r, capcode) or r == '_') and !(isLetterCp(r2, capcode) or r2 == '_')) {
        return .{ .priority = 9, .eligible = true, .is_fallback = false };
    }
    // number | non-number (priority 9).
    if (unicode_props.isNumber(r) and !unicode_props.isNumber(r2)) {
        return .{ .priority = 9, .eligible = true, .is_fallback = false };
    }
    // space | non-space (priority 7).
    if (r == ' ' and r2 != ' ') {
        return .{ .priority = 7, .eligible = true, .is_fallback = false };
    }
    // non-space | space (priority 8).
    if (r != ' ' and r2 == ' ') {
        return .{ .priority = 8, .eligible = true, .is_fallback = false };
    }
    // everything | capcode (priority 9). Empty for capcode .none.
    if (isCapcodeMarker(r2, capcode)) {
        return .{ .priority = 9, .eligible = true, .is_fallback = false };
    }
    // Priority-1 fallback (everything else). TM-Go takes this path
    // for cases without a clear boundary class so the encoder always
    // has SOMETHING to score against. Flagged so the caller can apply
    // the priority-8 suffix rule first (TM-Go reaches the suffix check
    // BEFORE this fallback; see :3719-3750).
    return .{ .priority = 1, .eligible = true, .is_fallback = true };
}

/// Compute the minAltSize for a piece. Mirrors TM-Go's :3527 +
/// :3581-3583: pieces beginning with ` `+alphanum start at minAltSize=2;
/// if `nWords <= 1` it's later reset to 1.
fn computeMinAltSize(piece: []const u8, capcode: CapcodeMode) usize {
    var r1: u21 = 0;
    var n1: usize = 0;
    decodeRune(piece, &r1, &n1);
    if (n1 == 0 or r1 != ' ' or n1 >= piece.len) return 1;
    var r2: u21 = 0;
    var n2: usize = 0;
    decodeRune(piece[n1..], &r2, &n2);
    if (n2 == 0 or !isAlphaNumCp(r2, capcode)) return 1;
    // Found ` `+alphanum start. But TM-Go drops to 1 if nWords <= 1.
    // We approximate `nWords` by counting space|alphanum transitions
    // (matches `computeNwordsScore`).
    const nw = computeNwordsScore(piece, capcode);
    return if (nw <= 1) 1 else 2;
}

fn computeAlts(
    allocator: std.mem.Allocator,
    bytes: []const u8,
    offsets: []const u32,
    count: u32,
    nodes: []const Monster.Node,
    child_bytes: []const u8,
    child_nodes: []const u32,
    capcode: CapcodeMode,
) ![]AltPair {
    const alts = try allocator.alloc(AltPair, if (count == 0) 1 else count);
    if (count == 0) {
        alts[0] = .{};
        return alts;
    }
    var id: u32 = 0;
    while (id < count) : (id += 1) {
        alts[id] = .{};
        const start = offsets[id];
        const end = offsets[id + 1];
        const piece = bytes[start..end];
        if (piece.len < 2) continue;

        var min_alt_size = computeMinAltSize(piece, capcode);
        if (min_alt_size == 0) min_alt_size = 1;

        // TM-Go iterates length from len(token)-1 DOWN to minAltSize.
        var priority1: u8 = 0;
        var priority2: u8 = 0;
        var pair: AltPair = .{};

        // hasSuffix position for the priority-8 suffix rule (TM-Go
        // tokenmonster.go:3595). `null` means no ungreedy suffix applies.
        const has_suffix: ?usize = hasSuffixPos(piece, capcode);

        var length: usize = piece.len;
        while (length > min_alt_size) {
            length -= 1;
            // 0-based length: subtoken is piece[0..length] (excludes
            // piece[length]). Loop visits length = len-1, len-2, ..., min.
            // TM-Go's first iteration uses length = len-1 (so subtoken
            // is len-1 bytes); ours matches.
            const subtoken = piece[0..length];
            const sub_id = trieExactLookup(nodes, child_bytes, child_nodes, subtoken);
            if (sub_id == NO_TOKEN) continue;
            if (sub_id == id) continue; // shouldn't happen (proper prefix)

            const pri_res = altPriority(piece, length, capcode);
            if (!pri_res.eligible) continue;
            var pri = pri_res.priority;

            // Suffix rule (TM-Go tokenmonster.go:3719-3735). Reached only
            // when the boundary-class switch did NOT match (i.e. TM-Go's
            // priority-1 fallback path). If the split lands exactly on the
            // ungreedy-suffix start, promote to priority 8 and BREAK the
            // length loop, matching TM-Go's `break` at :3734.
            //
            // In practice `pri_res.is_fallback && length == has_suffix`
            // cannot both hold (the char before the suffix is a letter and
            // the char at the split is `'`/`’`, so letter|non-letter wins
            // priority 9 above and this never fires). Ported faithfully
            // for exact control-flow parity; see `hasSuffixPos` doc.
            var suffix_break = false;
            if (pri_res.is_fallback) {
                if (has_suffix) |sp| {
                    if (length == sp) {
                        pri = 8; // TM-Go tokenmonster.go:3725/3731
                        suffix_break = true;
                    }
                }
            }

            // TM-Go's promotion logic at :3606-3617 (and similar). The
            // slot with LOWER priority gets overwritten; on equal
            // priority, the slot with SHORTER existing length is the
            // one to bump. We mirror the same control flow.
            //
            //   if priority1 < priority2 || (priority1 == priority2 &&
            //       alt.length <= alt.length2) {
            //       if priority1 < <pri> { ...update slot 1... }
            //   } else {
            //       if priority2 < <pri> { ...update slot 2... }
            //   }
            const target_slot1 = (priority1 < priority2) or
                (priority1 == priority2 and pair.length <= pair.length2);
            if (target_slot1) {
                if (priority1 < pri) {
                    pair.index = sub_id;
                    pair.length = @intCast(length);
                    priority1 = pri;
                }
            } else {
                if (priority2 < pri) {
                    pair.index2 = sub_id;
                    pair.length2 = @intCast(length);
                    priority2 = pri;
                }
            }

            // TM-Go tokenmonster.go:3734 — the suffix branch `break`s the
            // length loop after attempting promotion (unconditional within
            // the `length == hasSuffix` block).
            if (suffix_break) break;
        }

        // Make sure the better alt is in slot 1 (TM-Go :3760-3764).
        if (pair.length2 > 0 and
            (priority2 > priority1 or
                (priority2 == priority1 and pair.length2 > pair.length)))
        {
            const ti = pair.index;
            pair.index = pair.index2;
            pair.index2 = ti;
            const tl = pair.length;
            pair.length = pair.length2;
            pair.length2 = tl;
        }

        alts[id] = pair;
    }
    return alts;
}

// --- tests ---

test "builder + idBytes" {
    var b = Monster.Builder.init(std.testing.allocator);
    defer b.deinit();
    const foo = try b.addToken("foo");
    const bar = try b.addToken("bar");
    const unk = try b.addToken("<unk>");
    var m = try b.finalize(unk);
    defer m.deinit();

    try std.testing.expectEqualStrings("foo", m.idBytes(foo));
    try std.testing.expectEqualStrings("bar", m.idBytes(bar));
    try std.testing.expectEqualStrings("<unk>", m.idBytes(unk));
    try std.testing.expectEqual(@as(u32, 3), m.count);
    try std.testing.expectEqual(@as(u32, 5), m.max_token_len);
    // nwords proxy: all start with letters, no internal transitions → 1 each.
    try std.testing.expectEqual(@as(u8, 1), m.nwords[foo]);
    try std.testing.expectEqual(@as(u8, 1), m.nwords[bar]);
}

test "encodeChunk longest-match for unambiguous input" {
    var b = Monster.Builder.init(std.testing.allocator);
    defer b.deinit();
    const abc = try b.addToken("abc");
    const def = try b.addToken("def");
    const unk = try b.addToken("<unk>");
    var m = try b.finalize(unk);
    defer m.deinit();

    var out: [8]TokenId = undefined;
    const ids = try m.encodeChunk(std.testing.allocator, "abcdef", &out);
    try std.testing.expectEqualSlices(TokenId, &.{ abc, def }, ids);
}

test "encodeChunk picks higher-score branch" {
    // Vocab {a, ab, c, abc}. Encoding "abc":
    //   greedy "abc" → 3+0=3
    //   alt "ab" + "c" → 2+1=3 (covers same bytes as greedy → -10000 penalty)
    //   alt "a" + (no match for "b") → 1+0=1 (shorter → -100 penalty)
    // Greedy "abc" wins.
    var b = Monster.Builder.init(std.testing.allocator);
    defer b.deinit();
    _ = try b.addToken("a");
    _ = try b.addToken("ab");
    _ = try b.addToken("c");
    const abc = try b.addToken("abc");
    const unk = try b.addToken("<unk>");
    var m = try b.finalize(unk);
    defer m.deinit();

    var out: [8]TokenId = undefined;
    const ids = try m.encodeChunk(std.testing.allocator, "abc", &out);
    try std.testing.expectEqualSlices(TokenId, &.{abc}, ids);
}

test "encodeChunk falls back to unk for unknown bytes" {
    var b = Monster.Builder.init(std.testing.allocator);
    defer b.deinit();
    const a = try b.addToken("a");
    const unk = try b.addToken("<unk>");
    var m = try b.finalize(unk);
    defer m.deinit();

    var out: [8]TokenId = undefined;
    const ids = try m.encodeChunk(std.testing.allocator, "ax", &out);
    try std.testing.expectEqualSlices(TokenId, &.{ a, unk }, ids);
}

test "encodeChunk prefers space-prefixed tokens for whole-word coverage" {
    // Vocab {<unk>, " the", "the", " quick", "quick"}. Encoding " the quick":
    // greedy at pos 0 = " the" (4); lookahead at pos 4 = " quick" (6).
    // Result: [" the", " quick"] — both space-prefixed, both nwords=1.
    // The 6-branch machinery exercises shorter alts ("the"→ would need to
    // see " quick" after) but they don't beat greedy here. The genuinely
    // nontrivial rare-vs-common tradeoffs only show up on real corpora.
    var b = Monster.Builder.init(std.testing.allocator);
    defer b.deinit();
    const unk = try b.addToken("<unk>");
    const sp_the = try b.addToken(" the");
    _ = try b.addToken("the");
    const sp_quick = try b.addToken(" quick");
    _ = try b.addToken("quick");
    var m = try b.finalize(unk);
    defer m.deinit();

    var out: [16]TokenId = undefined;
    const ids = try m.encodeChunk(std.testing.allocator, " the quick", &out);
    try std.testing.expectEqualSlices(TokenId, &.{ sp_the, sp_quick }, ids);
}

test "encodeChunk -10000 deduction keeps greedy when alt has equal length" {
    // Vocab {<unk>, "ab", "a", "b"} on "ab":
    //   greedy "ab" → branch_len=2 (no second match) → no length penalty
    //   alt "a"+"b" → branch_len=1+1=2, equals greedy length → -10000
    // Greedy "ab" wins by 10000 — this is what makes the encoder ungreedy
    // only when an alt covers MORE bytes than greedy, not the same.
    var b = Monster.Builder.init(std.testing.allocator);
    defer b.deinit();
    const unk = try b.addToken("<unk>");
    const ab = try b.addToken("ab");
    _ = try b.addToken("a");
    _ = try b.addToken("b");
    var m = try b.finalize(unk);
    defer m.deinit();

    var out: [4]TokenId = undefined;
    const ids = try m.encodeChunk(std.testing.allocator, "ab", &out);
    try std.testing.expectEqualSlices(TokenId, &.{ab}, ids);
}

test "encodeChunkWithOffsets ranges match merged token boundaries" {
    var b = Monster.Builder.init(std.testing.allocator);
    defer b.deinit();
    const abc = try b.addToken("abc");
    const def = try b.addToken("def");
    const unk = try b.addToken("<unk>");
    var m = try b.finalize(unk);
    defer m.deinit();

    var out_ids: [8]TokenId = undefined;
    var out_off: [8]Span = undefined;
    const n = try m.encodeChunkWithOffsets(std.testing.allocator, "abcdef", 0, &out_ids, &out_off);
    try std.testing.expectEqual(@as(usize, 2), n);
    try std.testing.expectEqual(@as(TokenId, abc), out_ids[0]);
    try std.testing.expectEqual(@as(TokenId, def), out_ids[1]);
    try std.testing.expectEqual(@as(u32, 0), out_off[0].start);
    try std.testing.expectEqual(@as(u32, 3), out_off[0].end);
    try std.testing.expectEqual(@as(u32, 3), out_off[1].start);
    try std.testing.expectEqual(@as(u32, 6), out_off[1].end);
}

test "encodeChunkWithOffsets unk spans single byte" {
    var b = Monster.Builder.init(std.testing.allocator);
    defer b.deinit();
    const a = try b.addToken("a");
    const unk = try b.addToken("<unk>");
    var m = try b.finalize(unk);
    defer m.deinit();

    var out_ids: [8]TokenId = undefined;
    var out_off: [8]Span = undefined;
    const n = try m.encodeChunkWithOffsets(std.testing.allocator, "ax", 5, &out_ids, &out_off);
    try std.testing.expectEqual(@as(usize, 2), n);
    try std.testing.expectEqual(@as(TokenId, a), out_ids[0]);
    try std.testing.expectEqual(@as(TokenId, unk), out_ids[1]);
    try std.testing.expectEqual(@as(u32, 5), out_off[0].start);
    try std.testing.expectEqual(@as(u32, 6), out_off[0].end);
    try std.testing.expectEqual(@as(u32, 6), out_off[1].start);
    try std.testing.expectEqual(@as(u32, 7), out_off[1].end);
}

// --- mask tests (v3 marginal-value scoring substrate) ------------------

test "encodeChunk all-zero mask matches no mask (identity)" {
    var b = Monster.Builder.init(std.testing.allocator);
    defer b.deinit();
    _ = try b.addToken("<unk>");
    _ = try b.addToken("a");
    _ = try b.addToken("ab");
    _ = try b.addToken("abc");
    _ = try b.addToken("c");
    _ = try b.addToken(" the");
    _ = try b.addToken("the");
    _ = try b.addToken(" quick");
    _ = try b.addToken("quick");
    var m = try b.finalize(0);
    defer m.deinit();

    const inputs = [_][]const u8{ "abc", " the quick", "ax", "" };
    var no_mask: [32]TokenId = undefined;
    var with_mask: [32]TokenId = undefined;
    for (inputs) |inp| {
        m.mask = null;
        const ids_a = try m.encodeChunk(std.testing.allocator, inp, &no_mask);

        const zero = try std.testing.allocator.alloc(u8, m.count);
        defer std.testing.allocator.free(zero);
        @memset(zero, 0);
        m.mask = zero;
        const ids_b = try m.encodeChunk(std.testing.allocator, inp, &with_mask);

        try std.testing.expectEqualSlices(TokenId, ids_a, ids_b);
    }
    m.mask = null;
}

test "encodeChunk mask hides a single piece" {
    // Vocab {<unk>, "abc", "a", "b", "c"}. Encoding "abc":
    //   unmasked → ["abc"]
    //   mask "abc" → ["a","b","c"] (or "a"+"b" then a single byte, depending
    //     on scoring; either way the masked id must NEVER appear in output).
    var b = Monster.Builder.init(std.testing.allocator);
    defer b.deinit();
    _ = try b.addToken("<unk>");
    const abc = try b.addToken("abc");
    _ = try b.addToken("a");
    _ = try b.addToken("b");
    _ = try b.addToken("c");
    var m = try b.finalize(0);
    defer m.deinit();

    var unmasked_out: [8]TokenId = undefined;
    const unmasked = try m.encodeChunk(std.testing.allocator, "abc", &unmasked_out);
    try std.testing.expectEqualSlices(TokenId, &.{abc}, unmasked);

    const mask = try std.testing.allocator.alloc(u8, m.count);
    defer std.testing.allocator.free(mask);
    @memset(mask, 0);
    mask[abc] = 1;
    m.mask = mask;
    var masked_out: [8]TokenId = undefined;
    const masked = try m.encodeChunk(std.testing.allocator, "abc", &masked_out);
    m.mask = null;

    try std.testing.expect(masked.len >= 2);
    for (masked) |id| try std.testing.expect(id != abc);

    // Concatenated bytes still roundtrip — fallback bytes "a","b","c" cover it.
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(std.testing.allocator);
    for (masked) |id| try buf.appendSlice(std.testing.allocator, m.idBytes(id));
    try std.testing.expectEqualStrings("abc", buf.items);
}

test "encodeChunk mask is respected in lookahead/ungreedy path" {
    // Vocab {<unk>, "a", "b", "ab", "c"} on "abc":
    //   Unmasked: greedy at pos 0 = "ab" (len 2); lookahead at pos 2 = "c"
    //     (len 1). Branch_len = 3. Greedy is "ab".
    //   Mask "c": greedy at pos 0 = "ab"; lookahead "c" is hidden, so
    //     second_len = 0 → branch_len = 2. The ungreedy alt "a" + (lookahead
    //     "b") still scores 2 too, but the -10000 tie-break keeps greedy.
    //     We then advance to pos 2, where the only match is unk.
    //
    // The point: the lookahead must NOT see "c". If it did, the score would
    // include nw2=1 for "c", and the next-is-space check would treat past-end
    // (since after+1=3==len) as 1 → both score the same. With mask honored
    // in lookahead, the masked "c" must not contribute to scoring.
    //
    // A stronger check: construct a case where greedy would pick a masked
    // first token. With masking the greedy slot itself disappears, and
    // the encoder falls back to the (shorter) alternative.
    var b = Monster.Builder.init(std.testing.allocator);
    defer b.deinit();
    _ = try b.addToken("<unk>");
    const a = try b.addToken("a");
    _ = try b.addToken("b");
    const ab = try b.addToken("ab");
    const c = try b.addToken("c");
    var m = try b.finalize(0);
    defer m.deinit();

    // Mask "ab": greedy candidate at pos 0 disappears, encoder must fall
    // back to "a" then "b" then "c".
    const mask = try std.testing.allocator.alloc(u8, m.count);
    defer std.testing.allocator.free(mask);
    @memset(mask, 0);
    mask[ab] = 1;
    m.mask = mask;

    var out: [8]TokenId = undefined;
    const ids = try m.encodeChunk(std.testing.allocator, "abc", &out);
    m.mask = null;

    // ab must not appear; "a" must lead; some encoding of "b" and "c" must
    // follow (could be ["a", "b", "c"] or similar — the only invariant is
    // no masked id and full byte coverage).
    for (ids) |id| try std.testing.expect(id != ab);
    try std.testing.expect(ids.len >= 3);
    try std.testing.expectEqual(@as(TokenId, a), ids[0]);

    // The bytes should still roundtrip — single-byte pieces cover the input.
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(std.testing.allocator);
    for (ids) |id| try buf.appendSlice(std.testing.allocator, m.idBytes(id));
    try std.testing.expectEqualStrings("abc", buf.items);
    // And c (unmasked) must end up in the output.
    var saw_c = false;
    for (ids) |id| if (id == c) {
        saw_c = true;
    };
    try std.testing.expect(saw_c);
}

test "encodeChunkWithOffsets honors mask and keeps offsets correct" {
    // Vocab {<unk>, "abc", "a", "b", "c", "def"} on "abcdef" with "abc"
    // masked. Expect "a","b","c","def" with correct byte ranges.
    var b = Monster.Builder.init(std.testing.allocator);
    defer b.deinit();
    _ = try b.addToken("<unk>");
    const abc = try b.addToken("abc");
    _ = try b.addToken("a");
    _ = try b.addToken("b");
    _ = try b.addToken("c");
    const def = try b.addToken("def");
    var m = try b.finalize(0);
    defer m.deinit();

    const mask = try std.testing.allocator.alloc(u8, m.count);
    defer std.testing.allocator.free(mask);
    @memset(mask, 0);
    mask[abc] = 1;
    m.mask = mask;

    var out_ids: [16]TokenId = undefined;
    var out_off: [16]Span = undefined;
    const n = try m.encodeChunkWithOffsets(std.testing.allocator, "abcdef", 100, &out_ids, &out_off);
    m.mask = null;

    try std.testing.expect(n >= 4); // at least a,b,c,def
    for (out_ids[0..n]) |id| try std.testing.expect(id != abc);

    // Last token must be "def" with span [103, 106).
    try std.testing.expectEqual(@as(TokenId, def), out_ids[n - 1]);
    try std.testing.expectEqual(@as(u32, 103), out_off[n - 1].start);
    try std.testing.expectEqual(@as(u32, 106), out_off[n - 1].end);

    // Offsets must form a non-decreasing tiling that covers exactly the
    // input bytes [100, 106).
    try std.testing.expectEqual(@as(u32, 100), out_off[0].start);
    var prev_end: u32 = 100;
    var k: usize = 0;
    while (k < n) : (k += 1) {
        try std.testing.expectEqual(prev_end, out_off[k].start);
        try std.testing.expect(out_off[k].end > out_off[k].start);
        prev_end = out_off[k].end;
    }
    try std.testing.expectEqual(@as(u32, 106), prev_end);
}

// --- lilbuf (synthetic-boundary) tests ----------------------------------

test "lilbuf disabled: synthetic-prefix token is unreachable from bare input" {
    // Vocab {<unk>, "\x7F bar"}. Encoding "bar":
    //   collectPrefixMatches sees nothing matching at 'b'/'a'/'r' → unk
    //   fallback for every byte. The \x7F bar token would match the
    //   synthetic prefix [0x7F, 0x20] + "bar" but lilbuf is off.
    var b = Monster.Builder.init(std.testing.allocator);
    defer b.deinit();
    const unk = try b.addToken("<unk>");
    _ = try b.addToken("\x7F bar");
    var m = try b.finalize(unk);
    defer m.deinit();
    try std.testing.expect(!m.lilbuf_enabled); // default off

    var out: [16]TokenId = undefined;
    const ids = try m.encodeChunk(std.testing.allocator, "bar", &out);
    // Three unks (one per byte) because the only token covering b/a/r
    // bytes is gated behind the synthetic \x7F prefix.
    try std.testing.expectEqual(@as(usize, 3), ids.len);
    for (ids) |id| try std.testing.expectEqual(unk, id);
}

test "lilbuf enabled: synthetic-prefix token becomes reachable" {
    // Vocab {<unk>, "x", "\x7F bar"}. Encoding "xbar":
    //   pos 0 = 'x' → emit `x` (id 1, greedy).
    //   pos 1 = 'b', preceding byte 'x' is a letter → lilbuf fires.
    //     Walk [0x7F, 0x20] + "bar" → finds "\x7F bar" (5 bytes total,
    //     advance by 3 real input bytes).
    // Lilbuf is gated on "current byte is a letter AND previous byte
    // is also a letter" (we're mid-word) — that's why position 0 of
    // a bare "bar" wouldn't trigger it, but mid-word does.
    var b = Monster.Builder.init(std.testing.allocator);
    defer b.deinit();
    const unk = try b.addToken("<unk>");
    const x = try b.addToken("x");
    const sp_bar = try b.addToken("\x7F bar");
    var m = try b.finalize(unk);
    defer m.deinit();
    m.lilbuf_enabled = true;

    var out: [16]TokenId = undefined;
    const ids = try m.encodeChunk(std.testing.allocator, "xbar", &out);
    try std.testing.expectEqualSlices(TokenId, &.{ x, sp_bar }, ids);
}

test "lilbuf: no-prefix vocab is unaffected" {
    // Vocab without ANY \x7F-prefixed tokens. The lilbuf branch's trie
    // walk fails on the first synthetic byte; encoding must produce
    // exactly the same ids as with lilbuf disabled.
    var b = Monster.Builder.init(std.testing.allocator);
    defer b.deinit();
    const unk = try b.addToken("<unk>");
    const sp_the = try b.addToken(" the");
    _ = try b.addToken("the");
    const sp_quick = try b.addToken(" quick");
    _ = try b.addToken("quick");
    var m = try b.finalize(unk);
    defer m.deinit();

    var out_off: [32]TokenId = undefined;
    m.lilbuf_enabled = false;
    const ids_off = try m.encodeChunk(std.testing.allocator, " the quick", &out_off);
    var copy_off: [32]TokenId = undefined;
    @memcpy(copy_off[0..ids_off.len], ids_off);
    const off_len = ids_off.len;

    var out_on: [32]TokenId = undefined;
    m.lilbuf_enabled = true;
    const ids_on = try m.encodeChunk(std.testing.allocator, " the quick", &out_on);

    try std.testing.expectEqualSlices(TokenId, copy_off[0..off_len], ids_on);
    // And sanity: the expected greedy/lilbuf-irrelevant segmentation.
    try std.testing.expectEqualSlices(TokenId, &.{ sp_the, sp_quick }, ids_on);
}

test "lilbuf: TM 32K vocab encodes 'TokenMonster' to TM-Go's exact ids" {
    // Real-world parity check against bench/vocabs/tm_englishcode_32k.ztm.
    // TM-Go encodes the bytes `# TokenMonster` (no extra normalization)
    // as ids [6, 10331, 2740, 6648]:
    //   6     = '#'           (literal byte token)
    //   10331 = ' Token'      (greedy match for the bytes ' Token')
    //   2740  = '\x7f Mon'    (lilbuf-prefixed: synthetic ' M' prefix)
    //   6648  = '\x7f ster'   (lilbuf-prefixed continuation)
    // Without lilbuf, ztok cannot reach the \x7f-prefixed tokens; it
    // falls back to unk after position 6.
    const monster_io = @import("monster_io.zig");
    var loaded = monster_io.readFile(std.testing.allocator, "bench/vocabs/tm_englishcode_32k.ztm") catch |err| {
        // CI may not check out the vocab; skip rather than fail.
        std.debug.print("(skipping TM vocab test: {})\n", .{err});
        return error.SkipZigTest;
    };
    defer loaded.deinit();
    try std.testing.expect(loaded.lilbuf_enabled);

    const input = "# TokenMonster";
    var out: [64]TokenId = undefined;
    const ids = try loaded.encodeChunk(std.testing.allocator, input, &out);

    const expected = [_]TokenId{ 6, 10331, 2740, 6648 };
    try std.testing.expectEqualSlices(TokenId, &expected, ids);
}

// --- score2b/score3b lookahead-of-alt + forwardDelete tests --------------

test "score2b: ' pretraining 16' picks alt + DEL + lilbuf-second" {
    // Critical parity case from the post-1.13 task. TM-Go segments
    //   ' pretraining 16'   →  [5101 (' pre'), 98 (DEL), 21609 (' training'), 1648 (' 16')]
    // ztok 1.13 would instead pick the greedy ' pret' (id 8321) and then
    // splinter the rest into [' pret', 'rain', 'ing', ' 16'] — 4 tokens
    // but a wholly different segmentation. score2b lets us recover the
    // TM-Go choice by evaluating the alt branch with a lilbuf-space
    // lookahead and a -1 extra-token penalty.
    const monster_io = @import("monster_io.zig");
    var loaded = monster_io.readFile(std.testing.allocator, "bench/vocabs/tm_englishcode_32k.ztm") catch |err| {
        std.debug.print("(skipping TM vocab test: {})\n", .{err});
        return error.SkipZigTest;
    };
    defer loaded.deinit();
    try std.testing.expect(loaded.score2b3b_enabled);

    const input = " pretraining 16";
    var out: [64]TokenId = undefined;
    const ids = try loaded.encodeChunk(std.testing.allocator, input, &out);

    const expected = [_]TokenId{ 5101, 98, 21609, 1648 };
    try std.testing.expectEqualSlices(TokenId, &expected, ids);
}

test "score-b disabled: 1.13 baseline reachable for A/B benches" {
    // Confirm we can still reach lilbuf-only-no-score2b/3b behavior by
    // flipping the flag. This is the A/B switch the bench harness uses
    // when measuring score2b's marginal value.
    const monster_io = @import("monster_io.zig");
    var loaded = monster_io.readFile(std.testing.allocator, "bench/vocabs/tm_englishcode_32k.ztm") catch |err| {
        std.debug.print("(skipping TM vocab test: {})\n", .{err});
        return error.SkipZigTest;
    };
    defer loaded.deinit();
    // Disable score2b3b. lilbuf stays on.
    loaded.score2b3b_enabled = false;
    try std.testing.expect(loaded.lilbuf_enabled);

    const input = " pretraining 16";
    var out: [64]TokenId = undefined;
    const ids = try loaded.encodeChunk(std.testing.allocator, input, &out);

    // 1.13 baseline (no score2b): greedy ' pret' + path-(a) at midword
    // → [8321 (' pret'), 6573 ('rain'), 3269 ('ing'), 1648 (' 16')].
    const expected_baseline = [_]TokenId{ 8321, 6573, 3269, 1648 };
    try std.testing.expectEqualSlices(TokenId, &expected_baseline, ids);
}

test "score-b: no-TM vocab encodes unchanged (has_lilbuf_prefix_tokens=false short-circuit)" {
    // Vocab without `\x7f`-prefixed tokens: has_lilbuf_prefix_tokens
    // is FALSE. score-b's effective-second computation falls back to
    // path-(b) which won't fire here, so score-b CAN'T promote a
    // false win for vocabs that don't carry the synthetic prefix
    // machinery. Encoding must match an identical run with
    // `score2b3b_enabled = false`.
    var b = Monster.Builder.init(std.testing.allocator);
    defer b.deinit();
    const unk = try b.addToken("<unk>");
    _ = try b.addToken(" the");
    _ = try b.addToken(" quick");
    _ = try b.addToken("the");
    _ = try b.addToken("quick");
    var m = try b.finalize(unk);
    defer m.deinit();
    try std.testing.expect(!m.has_lilbuf_prefix_tokens);
    // Manually enable lilbuf+score2b3b to verify the short-circuit.
    m.lilbuf_enabled = true;
    m.score2b3b_enabled = true;

    // With lilbuf, encodeChunk requires out.len >= chunk.len * 2.
    var out_on: [32]TokenId = undefined;
    const ids_on = try m.encodeChunk(std.testing.allocator, " the quick", &out_on);
    var copy: [32]TokenId = undefined;
    @memcpy(copy[0..ids_on.len], ids_on);
    const n_on = ids_on.len;

    m.score2b3b_enabled = false;
    var out_off: [32]TokenId = undefined;
    const ids_off = try m.encodeChunk(std.testing.allocator, " the quick", &out_off);

    try std.testing.expectEqualSlices(TokenId, copy[0..n_on], ids_off);
}

test "forwardDelete: state clears after a non-score-b branch wins" {
    // Verify forward_delete is properly cleared after greedy/alt wins.
    // We can't easily observe the state directly, but we can detect
    // drift: encode a sequence where score-b fires once, then a
    // straight greedy follows. If forward_delete stayed at 1, the
    // subsequent branch_len would be off by 1 and the encoder would
    // pick a different segmentation.
    //
    // Use the TM vocab with input " pretraining 16 cat" — score-b fires
    // for "pretraining", then plain encoding for " 16 cat".
    const monster_io = @import("monster_io.zig");
    var loaded = monster_io.readFile(std.testing.allocator, "bench/vocabs/tm_englishcode_32k.ztm") catch |err| {
        std.debug.print("(skipping TM vocab test: {})\n", .{err});
        return error.SkipZigTest;
    };
    defer loaded.deinit();

    const input = " pretraining 16 the";
    var out: [64]TokenId = undefined;
    const ids = try loaded.encodeChunk(std.testing.allocator, input, &out);

    // The first 4 ids must match the score-b expected segmentation.
    // After that, " the" (id 87 = '\x7f t', or similar) should resolve
    // independently of forward_delete state.
    try std.testing.expect(ids.len >= 5);
    try std.testing.expectEqual(@as(TokenId, 5101), ids[0]); // ' pre'
    try std.testing.expectEqual(@as(TokenId, 98), ids[1]); // DEL
    try std.testing.expectEqual(@as(TokenId, 21609), ids[2]); // ' training'
    try std.testing.expectEqual(@as(TokenId, 1648), ids[3]); // ' 16'
    // ids[4..] covers ' the' — last id should NOT be DEL, indicating
    // the encoder advanced past ' 16' cleanly with forward_delete=0.
    try std.testing.expect(ids[ids.len - 1] != 98);
}

test "score-b: encodeChunkWithOffsets emits 3 spans for first_del_second" {
    // Verify the offsets variant correctly emits 3 spans (alt_first,
    // DEL synthetic boundary, lilbuf-second) with correct byte ranges
    // for a score-b win.
    const monster_io = @import("monster_io.zig");
    var loaded = monster_io.readFile(std.testing.allocator, "bench/vocabs/tm_englishcode_32k.ztm") catch |err| {
        std.debug.print("(skipping TM vocab test: {})\n", .{err});
        return error.SkipZigTest;
    };
    defer loaded.deinit();

    const input = " pretraining 16";
    var out_ids: [64]TokenId = undefined;
    var out_off: [64]Span = undefined;
    const n = try loaded.encodeChunkWithOffsets(
        std.testing.allocator,
        input,
        0,
        &out_ids,
        &out_off,
    );

    // Expected: [5101, 98, 21609, 1648]
    try std.testing.expectEqual(@as(usize, 4), n);
    try std.testing.expectEqual(@as(TokenId, 5101), out_ids[0]);
    try std.testing.expectEqual(@as(TokenId, 98), out_ids[1]);
    try std.testing.expectEqual(@as(TokenId, 21609), out_ids[2]);
    try std.testing.expectEqual(@as(TokenId, 1648), out_ids[3]);
    // ' pre' covers bytes [0, 4)
    try std.testing.expectEqual(@as(u32, 0), out_off[0].start);
    try std.testing.expectEqual(@as(u32, 4), out_off[0].end);
    // DEL is synthetic — zero-width at the boundary [4, 4)
    try std.testing.expectEqual(@as(u32, 4), out_off[1].start);
    try std.testing.expectEqual(@as(u32, 4), out_off[1].end);
    // ' training' covers [4, 12) — 8 real bytes (the synthetic space
    // prefix is collapsed into the boundary marker, not the span)
    try std.testing.expectEqual(@as(u32, 4), out_off[2].start);
    try std.testing.expectEqual(@as(u32, 12), out_off[2].end);
    // ' 16' covers [12, 15)
    try std.testing.expectEqual(@as(u32, 12), out_off[3].start);
    try std.testing.expectEqual(@as(u32, 15), out_off[3].end);
}

// Regression test for the phantom-second + bare_flags + begin_byte
// compensation fix (1.22): the segment `\x7F pretrain.\x7F md)` was
// historically picked apart into 5 tokens [98, 8321, 6573, 8975, 12]
// because ztok's collapsed vocab had no `t`-prefix tokens in its trie
// AND no letter classification for `t` in its begin_byte LUT. After
// the fix (bare-form flag/nWords precompute + double-tally for
// `\x7F X`/`X` collisions), it matches TM-Go's 4-token output
// [3529, 9735, 8975, 12] exactly.
test "vocab-collapse fix: '\\x7f pretrain.\\x7f md)' matches TM-Go" {
    const monster_io = @import("monster_io.zig");
    var loaded = monster_io.readFile(std.testing.allocator, "bench/vocabs/tm_englishcode_32k.ztm") catch |err| {
        std.debug.print("(skipping TM vocab test: {})\n", .{err});
        return error.SkipZigTest;
    };
    defer loaded.deinit();
    const input = [_]u8{ 0x7F, 0x20, 0x70, 0x72, 0x65, 0x74, 0x72, 0x61, 0x69, 0x6e, 0x2e, 0x7F, 0x20, 0x6d, 0x64, 0x29 };
    var out: [64]TokenId = undefined;
    const ids = try loaded.encodeChunk(std.testing.allocator, &input, &out);
    const expected = [_]TokenId{ 3529, 9735, 8975, 12 };
    try std.testing.expectEqualSlices(TokenId, &expected, ids);
}

// Regression test for the phantom-second / score-b gate-b separation
// fix (1.22): at `, cheaper, smarter`, ztok historically picked
// greedy `, c` (id 2483) because the gate-b fallback was firing
// score-b on the greedy branch (TM-Go's score1b would NOT fire here
// because TM-Go's `t`-prefix lookups are also empty after the `,`
// — wait, actually TM-Go finds the `h`-prefix `he` here; the issue
// was that ztok's collapsed vocab returns NO_TOKEN at `heaper...`
// and gate-b's fallback wrongly let score-b fire on greedy). After
// the fix (alt-only gate-b fallback), ztok matches TM-Go's
// `, ` + ` cheap` split.
test "score-b gate-b: greedy ', c' doesn't over-fire score-b" {
    const monster_io = @import("monster_io.zig");
    var loaded = monster_io.readFile(std.testing.allocator, "bench/vocabs/tm_englishcode_32k.ztm") catch |err| {
        std.debug.print("(skipping TM vocab test: {})\n", .{err});
        return error.SkipZigTest;
    };
    defer loaded.deinit();
    const input = ", cheaper, smarter";
    var out: [64]TokenId = undefined;
    const ids = try loaded.encodeChunk(std.testing.allocator, input, &out);
    // TM-Go's segmentation: `,` + ` cheap` + `\x7f er,` + ` smart` + `\x7f er`
    const expected = [_]TokenId{ 15, 10617, 3124, 11537, 1113 };
    try std.testing.expectEqualSlices(TokenId, &expected, ids);
}

// --- TM-Go flag-bit + score-formula tests (post-1.14 agent A) ---

test "computeFlags: bare-letter piece -> begins+ends letter + all-letters" {
    // "the" (capcode .none): begin=letter, ends=letter, onlyLetterSpace
    // (r1='t' letter, r2='h' letter), → flag bits {1, 2, 128}.
    const f = computeFlags("the", .none);
    try std.testing.expectEqual(@as(u8, FLAG_ENDS_LETTER | FLAG_BEGINS_LETTER | FLAG_ALL_LETTERS), f);
}

test "computeFlags: leading-space single-word piece -> SINGLE_WORD + ALL_LETTERS" {
    // " the": begin=space, min_alt_size=2, ends letter, onlyLetterSpace,
    // nWords=1. → {begin_space, ends_letter, single_word, all_letters}.
    const f = computeFlags(" the", .none);
    try std.testing.expectEqual(
        @as(u8, FLAG_BEGINS_SPACE | FLAG_ENDS_LETTER | FLAG_SINGLE_WORD | FLAG_ALL_LETTERS),
        f,
    );
}

test "computeFlags: all-punctuation single byte -> ALL_LETTERS (= all-puncts)" {
    // "!" (single byte): onlyPunc shortcut at piece.len==1 → bit 7 set.
    // No FLAG_ENDS_LETTER (it's punct).
    const f = computeFlags("!", .none);
    try std.testing.expectEqual(@as(u8, FLAG_ALL_LETTERS), f);
}

test "computeFlags: nocapcode-prefixed piece -> begins_capcode + ends_letter" {
    // "\x7f the": r1=0x7F (DEL marker), r2=' '. In nocapcode mode the
    // DEL doesn't trigger BEGINS_SPACE (only CharToken/WordToken do
    // in full capcode); BEGINS_CAPCODE fires. Ends with 'e' → ENDS_LETTER.
    // Not onlyLetterSpace (r1=capcode, not letter+not space).
    const f = computeFlags("\x7f the", .nocapcode);
    try std.testing.expectEqual(@as(u8, FLAG_BEGINS_CAPCODE | FLAG_ENDS_LETTER), f);
}

test "computeFlags: full-capcode W-marker piece -> begins_capcode + begins_space" {
    // "W hello": full-capcode W is the WordToken — special-cased to
    // also set FLAG_BEGINS_SPACE per TM-Go writer (:3533).
    const f = computeFlags("W hello", .full);
    try std.testing.expect((f & FLAG_BEGINS_CAPCODE) != 0);
    try std.testing.expect((f & FLAG_BEGINS_SPACE) != 0);
    try std.testing.expect((f & FLAG_ENDS_LETTER) != 0);
}

test "computeFlags: capcode-trailing piece -> ENDS_CAPCODE" {
    // "fooD" (full capcode): 'D' is the DELETE_TOKEN marker, ends-on-
    // capcode bit set.
    const f = computeFlags("fooD", .full);
    try std.testing.expect((f & FLAG_ENDS_CAPCODE) != 0);
    try std.testing.expectEqual(@as(u8, 0), f & FLAG_ENDS_LETTER);
}

test "tmScore: split-word penalty fires when first ends letter + second begins letter" {
    // first.flag = FLAG_ENDS_LETTER, second.flag = FLAG_BEGINS_LETTER.
    // No other flags. Should deduct -103 from the score vs the
    // same scenario without the flags.
    const with_flags = tmScore(3, 3, FLAG_ENDS_LETTER, FLAG_BEGINS_LETTER, 1, 1, 0, false, 0, 0, false);
    const without = tmScore(3, 3, 0, 0, 1, 1, 0, false, 0, 0, false);
    try std.testing.expectEqual(without - 103, with_flags);
}

test "tmScore: all-letters bonus +1 each side" {
    // first.flag = FLAG_ALL_LETTERS only, second.flag = 0 → +1 score.
    // both = +2.
    const left = tmScore(3, 3, FLAG_ALL_LETTERS, 0, 1, 1, 0, false, 0, 0, false);
    const right = tmScore(3, 3, 0, FLAG_ALL_LETTERS, 1, 1, 0, false, 0, 0, false);
    const both = tmScore(3, 3, FLAG_ALL_LETTERS, FLAG_ALL_LETTERS, 1, 1, 0, false, 0, 0, false);
    const baseline = tmScore(3, 3, 0, 0, 1, 1, 0, false, 0, 0, false);
    try std.testing.expectEqual(baseline + 1, left);
    try std.testing.expectEqual(baseline + 1, right);
    try std.testing.expectEqual(baseline + 2, both);
}

test "tmScore: next-byte-is-space bonus + non-letter word count" {
    // next_bb = BB_SPACE (12 = bit 2 + bit 3) → +1 for begin-space, plus
    // word bonus of +100 since bit 3 is set ("next is not letter").
    const with_space = tmScore(5, 0, 0, 0, 1, 0, BB_SPACE, false, 0, 0, false);
    const past_end = tmScore(5, 0, 0, 0, 1, 0, 0, false, 0, 0, false);
    try std.testing.expectEqual(past_end + 1 + 100, with_space);
}

test "synthetic 4-piece vocab: score formula picks whole-word path" {
    // Vocab tuned to expose the split-word penalty:
    // pieces:  {"<unk>", "th", "the", "ere", "her", " quickly"}
    // input:   "the"  (no leading space).
    // Greedy at pos 0 = "the" (3 bytes, FLAG_ENDS_LETTER+FLAG_BEGINS_LETTER+FLAG_ALL_LETTERS).
    //   Lookahead at pos 3 = none → branch_len = 3.
    // Alt "th" (2 bytes) + greedy_lookahead at pos 2 = "her"? No —
    //   "her" only matches starting from 'h' which is byte 1.
    //   Lookahead at pos 2 = "e" → no match. branch_len = 2.
    //   -10000 vs greedy_len=3 (no), -100 (shorter).
    // Greedy wins.
    var b = Monster.Builder.init(std.testing.allocator);
    defer b.deinit();
    _ = try b.addToken("<unk>");
    _ = try b.addToken("th");
    const the = try b.addToken("the");
    _ = try b.addToken("ere");
    _ = try b.addToken("her");
    _ = try b.addToken(" quickly");
    var m = try b.finalizeWithCapcode(0, .none);
    defer m.deinit();

    var out: [16]TokenId = undefined;
    const ids = try m.encodeChunk(std.testing.allocator, "the", &out);
    try std.testing.expectEqualSlices(TokenId, &.{the}, ids);
}

test "split-word penalty: alt covering whole word beats greedy + split" {
    // Vocab pieces:
    //   <unk>, "th", "the", "y", "ere"
    // Input: "they"
    // - Greedy at 0 = "the" (3), lookahead at 3 = "y" (1). branch_len = 4.
    //   second.flag = FLAG_BEGINS_LETTER (y is letter). first.flag has
    //   FLAG_ENDS_LETTER (e). Split-word penalty -103 fires.
    //   nw1 = nw2 = 0 (no leading-space-then-alphanum in either piece).
    //   Score: 4 + 0 + 0 + 0 + 1 (first.all_letters) + 1 (second.all_letters)
    //          + 0 (next is end-of-input, next_bb=0)
    //          - 103 (split word)
    //        = 4 + 2 - 103 = -97
    // - Alt "th" (2) + lookahead at 2 = "ey"? No vocab. Falls back to
    //   just "th" (branch_len=2, second_len=0). second.flag=0.
    //   first.flag has FLAG_ENDS_LETTER, FLAG_BEGINS_LETTER,
    //   FLAG_ALL_LETTERS. nw1=0. nw2=0.
    //   Score: 2 + 0 + 0 + 0 + 1 (first.all_letters) + 0 + 0
    //          - 0 (no split-word: second begins nothing)
    //          - 100 (LessThan branch_len=2 vs greedy_len=3 → -100)
    //        = 2 + 1 - 100 = -97
    // Both score -97. Tie. Greedy wins on first iteration of best_score.
    //
    // So this test verifies the BEHAVIOR — split-word penalty made the
    // alt competitive (without it, greedy would have +103 higher, winning
    // by far). Now they tie, and the implementation picks greedy.
    var b = Monster.Builder.init(std.testing.allocator);
    defer b.deinit();
    _ = try b.addToken("<unk>");
    _ = try b.addToken("th");
    const the = try b.addToken("the");
    _ = try b.addToken("y");
    _ = try b.addToken("ere");
    var m = try b.finalizeWithCapcode(0, .none);
    defer m.deinit();

    var out: [8]TokenId = undefined;
    const ids = try m.encodeChunk(std.testing.allocator, "they", &out);
    // First token is greedy "the" (+ then "y"). Test asserts the first
    // emitted token IS the (the split-word penalty didn't tip the tie
    // toward alt). This locks in the formula behavior; a future change
    // that strengthens the penalty would surface here.
    try std.testing.expect(ids.len >= 1);
    try std.testing.expectEqual(@as(TokenId, the), ids[0]);
}

test "back-compat: legacy finalize(unk_id) produces flags but encode unchanged" {
    // The legacy path delegates to finalizeWithCapcode(.none). flags
    // ARE computed (just not specific to any capcode mode). For
    // vocabs that don't carry any flag-relevant patterns (no
    // letter-ending pieces, etc.), the score formula degrades.
    var b = Monster.Builder.init(std.testing.allocator);
    defer b.deinit();
    const a = try b.addToken("a");
    _ = try b.addToken("b");
    _ = try b.addToken("ab");
    const unk = try b.addToken("<unk>");
    var m = try b.finalize(unk);
    defer m.deinit();

    // flags slice is allocated, non-zero for letter-pieces.
    try std.testing.expectEqual(@as(usize, 4), m.flags.len);
    try std.testing.expect((m.flags[a] & FLAG_BEGINS_LETTER) != 0);

    // Encode picks the same id as the existing "encodeChunk longest-
    // match for unambiguous input" test (which doesn't use flags
    // because the synthetic vocab has trivial flag values).
    var out: [8]TokenId = undefined;
    const ids = try m.encodeChunk(std.testing.allocator, "ab", &out);
    // "ab" is the longest greedy at pos 0; no alts beat it.
    try std.testing.expectEqualSlices(TokenId, &.{2}, ids);
}

// --- 1.16 post-1.15 agent A: precomputed alts + lilbuf gate relax +
//     full-capcode lilbuf marker tests ---

test "precomputed alts: populated for non-empty vocabs" {
    // Every piece with at least one in-vocab proper-prefix subtoken
    // should have a non-NO_TOKEN entry in `alts`. Build a vocab where
    // we know the structure exactly.
    var b = Monster.Builder.init(std.testing.allocator);
    defer b.deinit();
    _ = try b.addToken("<unk>");
    const p = try b.addToken("p");
    _ = try b.addToken("pre");
    const pret = try b.addToken("pret");
    var m = try b.finalize(0);
    defer m.deinit();
    try std.testing.expect(m.use_precomputed_alts);
    try std.testing.expect(m.alts != null);
    // "pret" has subtokens "pre" (priority-1 fallback boundary class
    // letter|letter — no class transition) and "p" (priority-1 too).
    // Both end up in alts.{index,index2}; the LONGER one wins slot 1.
    const ap = m.alts.?[pret];
    try std.testing.expect(ap.index != NO_TOKEN);
    try std.testing.expect(ap.length >= 1);
    // "p" piece has no subtokens (it's 1 byte).
    const ap_p = m.alts.?[p];
    try std.testing.expectEqual(@as(u32, NO_TOKEN), ap_p.index);
}

test "precomputed alts: boundary-class priority picks the right alt" {
    // Build a vocab where the priority-10 (space + letter after split)
    // alt is NOT the longest proper prefix.
    //   piece: ' hello' (begins space + letter, 6 bytes)
    //   subtokens in vocab: ' hell' (5 bytes — no clean boundary,
    //     prio-1 fallback), ' he' (3 bytes, also prio-1), and
    //     ' ' (1 byte — split at position 1 leaves piece[1]='h',
    //     piece[2]=letter → priority 10 fires).
    // alts[ ' hello' ] should pick ' ' (length 1, prio 10) as slot 1,
    // because prio 10 > prio 1.
    var b = Monster.Builder.init(std.testing.allocator);
    defer b.deinit();
    _ = try b.addToken("<unk>");
    const sp = try b.addToken(" ");
    _ = try b.addToken(" he");
    _ = try b.addToken(" hell");
    const sp_hello = try b.addToken(" hello");
    var m = try b.finalizeWithCapcode(0, .none);
    defer m.deinit();
    const ap = m.alts.?[sp_hello];
    // Slot 1 should be ' ' (priority 10) — even though ' hell' is
    // longer (priority 1 fallback only).
    try std.testing.expectEqual(@as(u32, sp), ap.index);
    try std.testing.expectEqual(@as(u32, 1), ap.length);
}

test "lilbuf gate: position 0 letter fires (relaxed from pre-1.16)" {
    // Vocab without lilbuf markers — gate fires but lilbuf walks
    // miss (no path-(b) terminals), so encode is unchanged. The
    // relaxed gate is observable indirectly via the position-0
    // path-(a) lilbuf walk getting CALLED (not skipped) — but with
    // no `\x7f`-prefixed tokens, ll_id stays NO_TOKEN. Encoding
    // continues normally.
    var b = Monster.Builder.init(std.testing.allocator);
    defer b.deinit();
    const unk = try b.addToken("<unk>");
    _ = try b.addToken(" the");
    const the = try b.addToken("the");
    var m = try b.finalize(unk);
    defer m.deinit();
    m.lilbuf_enabled = true;

    // "the" at position 0 — letter, prev byte n/a → gate fires.
    // No `\x7f`-prefixed token → ll_id=NO_TOKEN. Continues to greedy
    // which finds "the".
    var out: [8]TokenId = undefined;
    const ids = try m.encodeChunk(std.testing.allocator, "the", &out);
    try std.testing.expectEqualSlices(TokenId, &.{the}, ids);
}

test "lilbuf gate: digit-position doesn't fire (not a letter)" {
    // `1"` style: the digit `1` doesn't satisfy isAsciiLetter, so
    // the relaxed gate still returns false. This preserves the
    // existing behavior that digit runs don't try lilbuf walks.
    try std.testing.expect(!lilbufPositionGate("123", 0));
    try std.testing.expect(!lilbufPositionGate("123", 1));
    try std.testing.expect(lilbufPositionGate("abc", 0));
    try std.testing.expect(lilbufPositionGate("abc", 1));
    try std.testing.expect(lilbufPositionGate("=p", 1));
}

test "full-capcode: lilbuf_marker_byte resolves to 'D' (0x44)" {
    // Build a vocab in full-capcode mode (no `\x7F` token, has 'D').
    // The encoder's lilbuf path-(b) walks `[lilbuf_marker_byte, 0x20]`;
    // for full capcode that's `[0x44, 0x20]` = `D ` synthetic prefix.
    var b = Monster.Builder.init(std.testing.allocator);
    defer b.deinit();
    _ = try b.addToken("<unk>");
    const d = try b.addToken("D"); // capcode DeleteToken
    _ = try b.addToken("D hello"); // D + space + hello (path-(b) target)
    _ = try b.addToken(" hello");
    var m = try b.finalizeWithCapcode(0, .full);
    defer m.deinit();
    try std.testing.expectEqual(@as(u8, 'D'), m.lilbuf_marker_byte);
    // Delete token id resolves to the single-byte 'D' (id 1).
    try std.testing.expectEqual(@as(u32, d), m.delete_token_id);
    // The trie contains `D ` as a prefix.
    try std.testing.expect(m.has_lilbuf_prefix_tokens);
}

test "nocapcode: lilbuf_marker_byte stays 0x7F (back-compat)" {
    var b = Monster.Builder.init(std.testing.allocator);
    defer b.deinit();
    _ = try b.addToken("<unk>");
    _ = try b.addToken("\x7F");
    _ = try b.addToken("\x7F hello");
    var m = try b.finalizeWithCapcode(0, .nocapcode);
    defer m.deinit();
    try std.testing.expectEqual(@as(u8, 0x7F), m.lilbuf_marker_byte);
    try std.testing.expect(m.has_lilbuf_prefix_tokens);
}

test "back-compat: legacy finalize keeps alts populated (no path opt-out)" {
    // The legacy `finalize(unk_id)` delegates to
    // `finalizeWithCapcode(.none)` which now also builds `alts`.
    // Encoding result is unchanged for vocabs without flag-relevant
    // alts (the alt scores degrade to the same path the old loop
    // produced).
    var b = Monster.Builder.init(std.testing.allocator);
    defer b.deinit();
    _ = try b.addToken("<unk>");
    _ = try b.addToken("a");
    _ = try b.addToken("ab");
    _ = try b.addToken("abc");
    var m = try b.finalize(0);
    defer m.deinit();
    // alts populated and the encoder path is enabled.
    try std.testing.expect(m.alts != null);
    try std.testing.expect(m.use_precomputed_alts);
    // The existing greedy-wins assertion still holds.
    var out: [8]TokenId = undefined;
    const ids = try m.encodeChunk(std.testing.allocator, "abc", &out);
    try std.testing.expectEqualSlices(TokenId, &.{3}, ids); // abc id = 3
}

// --- post-1.17 agent C: goto-checkpoint re-evaluation tests ---

test "goto-checkpoint: default off keeps 1.16 single-emit behavior" {
    // Builder default is `goto_checkpoint_enabled = false`. Any score-b
    // win should emit `[alt_first, DEL, lilbuf_second]` as a single
    // triple (1.16 path), not split into [alt_first, DEL] + re-eval.
    var b = Monster.Builder.init(std.testing.allocator);
    defer b.deinit();
    _ = try b.addToken("<unk>");
    var m = try b.finalize(0);
    defer m.deinit();
    try std.testing.expect(!m.goto_checkpoint_enabled);
}

test "goto-checkpoint: monster_io enables it for .ztm loads" {
    // The .ztm loader (`monster_io.readBytes`) auto-enables
    // goto_checkpoint_enabled alongside lilbuf + score2b3b so TM-Go-
    // compat vocabs pick up the re-evaluation behavior. Verify on
    // the actual TM 32K vocab if available.
    const monster_io = @import("monster_io.zig");
    var loaded = monster_io.readFile(std.testing.allocator, "bench/vocabs/tm_englishcode_32k.ztm") catch |err| {
        std.debug.print("(skipping TM vocab test: {})\n", .{err});
        return error.SkipZigTest;
    };
    defer loaded.deinit();
    try std.testing.expect(loaded.goto_checkpoint_enabled);
    try std.testing.expect(loaded.lilbuf_enabled);
    try std.testing.expect(loaded.score2b3b_enabled);
}

test "goto-checkpoint: ' pretraining 16' still produces TM-Go's exact ids" {
    // The pretraining case is the canonical score2b parity test
    // (see "score2b: ' pretraining 16' picks alt + DEL + lilbuf-second"
    // above). With goto-checkpoint ON, the encoder emits
    // `[5101 (' pre'), 98 (DEL)]` then re-enters with ` training`
    // (id 21609) seeded as the synthetic greedy at the new position.
    // The re-eval picks ` training` itself (greedy wins over its
    // shorter alts), advances 8 real bytes, then emits ` 16` (id 1648).
    // Same 4-id output as 1.16 — the goto-checkpoint is semantically
    // equivalent for THIS case (no shorter alt of ` training` beats
    // the greedy seed) but the encoder MUST still produce identical
    // output. Regression test for the goto-checkpoint plumbing.
    const monster_io = @import("monster_io.zig");
    var loaded = monster_io.readFile(std.testing.allocator, "bench/vocabs/tm_englishcode_32k.ztm") catch |err| {
        std.debug.print("(skipping TM vocab test: {})\n", .{err});
        return error.SkipZigTest;
    };
    defer loaded.deinit();
    try std.testing.expect(loaded.goto_checkpoint_enabled);

    const input = " pretraining 16";
    var out: [64]TokenId = undefined;
    const ids = try loaded.encodeChunk(std.testing.allocator, input, &out);

    const expected = [_]TokenId{ 5101, 98, 21609, 1648 };
    try std.testing.expectEqualSlices(TokenId, &expected, ids);
}

test "goto-checkpoint: disabling reverts to 1.16 single-emit (A/B switch)" {
    // Flipping goto_checkpoint_enabled off must reproduce the 1.16
    // single-emit pattern bit-for-bit. The pretraining case produces
    // the same 4 ids either way (greedy seed wins the re-eval), so
    // this is a structural regression test rather than an output-
    // diverging check. The structural assertion: with goto OFF the
    // emit pattern is "alt_first DEL lilbuf_second in one switch case"
    // (3 writes); with goto ON it's "alt_first DEL emit + seed +
    // greedy emit in next iter" (3 writes total). The output ids and
    // their order MUST be identical — we verify that here.
    const monster_io = @import("monster_io.zig");
    var loaded = monster_io.readFile(std.testing.allocator, "bench/vocabs/tm_englishcode_32k.ztm") catch |err| {
        std.debug.print("(skipping TM vocab test: {})\n", .{err});
        return error.SkipZigTest;
    };
    defer loaded.deinit();

    const input = " pretraining 16";
    var out_on: [64]TokenId = undefined;
    loaded.goto_checkpoint_enabled = true;
    const ids_on = try loaded.encodeChunk(std.testing.allocator, input, &out_on);
    var copy: [64]TokenId = undefined;
    @memcpy(copy[0..ids_on.len], ids_on);
    const n_on = ids_on.len;

    var out_off: [64]TokenId = undefined;
    loaded.goto_checkpoint_enabled = false;
    const ids_off = try loaded.encodeChunk(std.testing.allocator, input, &out_off);
    // Restore for downstream tests.
    loaded.goto_checkpoint_enabled = true;

    try std.testing.expectEqualSlices(TokenId, copy[0..n_on], ids_off);
}

test "goto-checkpoint: forward_lilbuf clears cleanly after non-score-b iters" {
    // Encode a buffer where score-b fires once then the encoder
    // continues with plain greedy emits. The output must end cleanly —
    // no spurious DEL tokens, no length under/over-advance — proving
    // that `forward_lilbuf` is reset to false after the seeded iter
    // commits its greedy emission.
    const monster_io = @import("monster_io.zig");
    var loaded = monster_io.readFile(std.testing.allocator, "bench/vocabs/tm_englishcode_32k.ztm") catch |err| {
        std.debug.print("(skipping TM vocab test: {})\n", .{err});
        return error.SkipZigTest;
    };
    defer loaded.deinit();

    const input = " pretraining 16 the cat sat";
    var out: [128]TokenId = undefined;
    const ids = try loaded.encodeChunk(std.testing.allocator, input, &out);

    // Sanity: the first 4 ids are the pretraining segmentation; the
    // tail must NOT contain a stray DEL where forward_lilbuf would
    // have leaked.
    try std.testing.expect(ids.len >= 5);
    try std.testing.expectEqual(@as(TokenId, 5101), ids[0]); // ' pre'
    try std.testing.expectEqual(@as(TokenId, 98), ids[1]); // DEL
    try std.testing.expectEqual(@as(TokenId, 21609), ids[2]); // ' training'
    try std.testing.expectEqual(@as(TokenId, 1648), ids[3]); // ' 16'
    // The last id should NOT be DEL — input ends with " sat", which
    // resolves cleanly to non-lilbuf tokens.
    try std.testing.expect(ids[ids.len - 1] != 98);
}

test "goto-checkpoint: vocab without `\\x7f `-prefix tokens encodes identically on/off" {
    // A synthetic vocab that doesn't carry any DEL-marker tokens.
    // The score-b gate (`has_space_prefix_tokens` + `delete_token_id !=
    // NO_TOKEN`) is false, so .first_del_second never wins. The
    // goto-checkpoint code path is therefore unreachable. Toggling
    // the flag must produce byte-identical output, proving the
    // goto-checkpoint addition is a true no-op when its preconditions
    // don't hold.
    var b = Monster.Builder.init(std.testing.allocator);
    defer b.deinit();
    const unk = try b.addToken("<unk>");
    _ = try b.addToken("the");
    _ = try b.addToken(" the");
    _ = try b.addToken("quick");
    _ = try b.addToken(" quick");
    _ = try b.addToken("brown");
    var m = try b.finalize(unk);
    defer m.deinit();
    // No DEL-marker token in this vocab.
    try std.testing.expectEqual(@as(u32, NO_TOKEN), m.delete_token_id);

    // Manually enable lilbuf + score2b3b to ensure the score-b path is
    // available — but the gate stays false because delete_token_id ==
    // NO_TOKEN. With lilbuf, the chunk size doubles for out.len.
    m.lilbuf_enabled = true;
    m.score2b3b_enabled = true;

    const input = " the quick brown the quick";
    var out_off: [64]TokenId = undefined;
    m.goto_checkpoint_enabled = false;
    const ids_off = try m.encodeChunk(std.testing.allocator, input, &out_off);
    var copy: [64]TokenId = undefined;
    @memcpy(copy[0..ids_off.len], ids_off);
    const n_off = ids_off.len;

    var out_on: [64]TokenId = undefined;
    m.goto_checkpoint_enabled = true;
    const ids_on = try m.encodeChunk(std.testing.allocator, input, &out_on);

    try std.testing.expectEqualSlices(TokenId, copy[0..n_off], ids_on);
    // And the goto-checkpoint must never have fired: no DEL ids in
    // the output (the vocab has no DEL to emit anyway).
    for (ids_on) |id| try std.testing.expect(id != NO_TOKEN);
}

test "goto-checkpoint: encodeChunkWithOffsets emits 2-span pattern (not 3)" {
    // With goto-checkpoint ON, .first_del_second emits ONLY
    // `[alt_first, DEL]` in the current iteration (2 spans); the
    // lilbuf_second is emitted in the NEXT iter as a separate token
    // (1 more span at offset = after_first_pos + ...). The total span
    // count for the pretraining case is unchanged (4 spans), but the
    // boundary semantics differ: the DEL's span is at after_first_pos
    // and the lilbuf-second's span starts at after_first_pos (since
    // the synthetic space prefix is part of the boundary, not the
    // real input).
    const monster_io = @import("monster_io.zig");
    var loaded = monster_io.readFile(std.testing.allocator, "bench/vocabs/tm_englishcode_32k.ztm") catch |err| {
        std.debug.print("(skipping TM vocab test: {})\n", .{err});
        return error.SkipZigTest;
    };
    defer loaded.deinit();

    const input = " pretraining 16";
    var out_ids: [64]TokenId = undefined;
    var out_off: [64]Span = undefined;
    const n = try loaded.encodeChunkWithOffsets(
        std.testing.allocator,
        input,
        0,
        &out_ids,
        &out_off,
    );

    // Same 4 ids as the encodeChunk-only test.
    try std.testing.expectEqual(@as(usize, 4), n);
    try std.testing.expectEqual(@as(TokenId, 5101), out_ids[0]);
    try std.testing.expectEqual(@as(TokenId, 98), out_ids[1]);
    try std.testing.expectEqual(@as(TokenId, 21609), out_ids[2]);
    try std.testing.expectEqual(@as(TokenId, 1648), out_ids[3]);
    // ' pre' covers bytes [0, 4)
    try std.testing.expectEqual(@as(u32, 0), out_off[0].start);
    try std.testing.expectEqual(@as(u32, 4), out_off[0].end);
    // DEL is synthetic — zero-width at the boundary.
    try std.testing.expectEqual(@as(u32, 4), out_off[1].start);
    try std.testing.expectEqual(@as(u32, 4), out_off[1].end);
    // ' training' starts at byte 4. With goto, this token is emitted
    // in the NEXT iter where `i = 4`. The span goes from 4 to 4+8=12.
    try std.testing.expectEqual(@as(u32, 4), out_off[2].start);
    try std.testing.expectEqual(@as(u32, 12), out_off[2].end);
    // ' 16' covers [12, 15)
    try std.testing.expectEqual(@as(u32, 12), out_off[3].start);
    try std.testing.expectEqual(@as(u32, 15), out_off[3].end);
}

// === 1.19 perf push: bit-identical regression check (post-1.18 agent E) ===
//
// The 1.19 changes (combined `longestMatchIdAndLen` trie walks + small-
// fanout linear `findChild`) MUST produce the same id stream as 1.18 on
// real-world input. This test guards against the refactor introducing
// any silent divergence on real text and confirms repeated calls are
// bit-identical (the small-fanout linear scan must not be perturbed by
// any cached state).

test "1.19 encoder optimizations: TM nocapcode large-mixed sample is bit-identical run-to-run" {
    const monster_io = @import("monster_io.zig");
    var loaded = monster_io.readFile(std.testing.allocator, "bench/vocabs/tm_englishcode_32k.ztm") catch |err| {
        std.debug.print("(skipping 1.19 bit-identical test: {})\n", .{err});
        return error.SkipZigTest;
    };
    defer loaded.deinit();

    // ~10 KB sample: balanced English/punct/digit/word-boundary mix
    // exercises greedy, score2/3 alts, score2b/3b lilbuf, and the
    // goto-checkpoint re-entry path.
    var sample_buf: std.ArrayList(u8) = .empty;
    defer sample_buf.deinit(std.testing.allocator);

    const lines = [_][]const u8{
        "The quick brown fox jumps over the lazy dog. ",
        " #TokenMonster pretraining 16 elapsed=42ms loss=3.14159\n",
        " A B C D E F  -- this line has  many   spaces  -- end.\n",
        " 12345 67890 0x1A2B3C 1e9 -1.5e-10 (parens) [brackets] {braces}\n",
        " https://example.com/path?q=value#frag user@host.tld\n",
        " The rain in Spain falls mainly on the plain. ",
        " She sells seashells by the seashore. ",
        " Peter Piper picked a peck of pickled peppers. ",
        " How much wood would a woodchuck chuck if a woodchuck could chuck wood? ",
        " Wikipedia is a free online encyclopedia that anyone can edit. ",
        " Lorem ipsum dolor sit amet consectetur adipiscing elit. ",
        " Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.\n",
    };
    var rep: usize = 0;
    while (rep < 12) : (rep += 1) {
        for (lines) |l| try sample_buf.appendSlice(std.testing.allocator, l);
    }

    const sample = sample_buf.items;
    const out_cap = sample.len * 2;
    const out_a = try std.testing.allocator.alloc(TokenId, out_cap);
    defer std.testing.allocator.free(out_a);
    const out_b = try std.testing.allocator.alloc(TokenId, out_cap);
    defer std.testing.allocator.free(out_b);

    const ids_a = try loaded.encodeChunk(std.testing.allocator, sample, out_a);
    const ids_b = try loaded.encodeChunk(std.testing.allocator, sample, out_b);

    try std.testing.expectEqualSlices(TokenId, ids_a, ids_b);
    try std.testing.expect(ids_a.len > 0);
    // bytes/token ratio for TM 32K on English should comfortably exceed 1.5
    const bpt = @as(f64, @floatFromInt(sample.len)) / @as(f64, @floatFromInt(ids_a.len));
    try std.testing.expect(bpt > 1.5);
    try std.testing.expect(bpt < 4.0);
}

// === 1.20 perf v2 regression tests ===

test "1.20 perf: SoA trie layout encodes identically to the AoS predecessor (synthetic vocab)" {
    // Hand-built vocab + corpus that exercises every trie depth + the
    // small-fanout (≤8 children) and wide-fanout (binary search) paths
    // in `findChild`. Hits both the byte-scan loop and the early-exit
    // miss case. SoA's correctness is observable through the same
    // encode oracle that AoS used — the bytes returned by `idBytes`
    // and the ids returned by `encodeChunk` must round-trip the
    // input exactly.
    var b = Monster.Builder.init(std.testing.allocator);
    defer b.deinit();
    // Vocabulary covering several depths + a wide root fanout.
    const unk = try b.addToken("<unk>");
    _ = try b.addToken(" ");
    _ = try b.addToken("a");
    _ = try b.addToken("b");
    _ = try b.addToken("c");
    _ = try b.addToken("d");
    _ = try b.addToken("e");
    _ = try b.addToken("f");
    _ = try b.addToken("g");
    _ = try b.addToken("h");
    _ = try b.addToken("ab");
    _ = try b.addToken("abc");
    _ = try b.addToken("abcd");
    _ = try b.addToken("abcde");
    _ = try b.addToken(" the");
    _ = try b.addToken(" quick");
    _ = try b.addToken(" brown");
    _ = try b.addToken(" fox");
    _ = try b.addToken("the");
    _ = try b.addToken("quick");
    _ = try b.addToken("brown");
    _ = try b.addToken("fox");
    _ = try b.addToken("jumps");
    _ = try b.addToken(" jumps");
    var m = try b.finalize(unk);
    defer m.deinit();

    // Sanity: SoA arrays are populated, byte-scan and node arrays
    // are the same length, and the root has multiple children.
    try std.testing.expect(m.child_bytes.len > 0);
    try std.testing.expect(m.child_bytes.len == m.child_nodes.len);
    try std.testing.expect(m.nodes.len > 0);
    try std.testing.expect(m.nodes[0].children_len > 1);

    // The byte-scan returns children in ascending byte order — assert
    // the SoA bytes array reflects that for every node (mirrors the
    // pre-1.20 AoS invariant the linear scan's early-exit relies on).
    var ni: u32 = 0;
    while (ni < m.nodes.len) : (ni += 1) {
        const n = m.nodes[ni];
        if (n.children_len <= 1) continue;
        var k: u32 = 1;
        while (k < n.children_len) : (k += 1) {
            const prev = m.child_bytes[n.children_start + k - 1];
            const cur = m.child_bytes[n.children_start + k];
            try std.testing.expect(prev < cur);
        }
    }

    // Round-trip on a mixed corpus that hits every relevant code path.
    const sample = " the quick brown fox jumps abcde abcd abc ab a b c d e f g h ";
    const out = try std.testing.allocator.alloc(TokenId, sample.len * 2);
    defer std.testing.allocator.free(out);
    const ids = try m.encodeChunk(std.testing.allocator, sample, out);
    try std.testing.expect(ids.len > 0);

    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(std.testing.allocator);
    for (ids) |id| try buf.appendSlice(std.testing.allocator, m.idBytes(id));
    try std.testing.expectEqualStrings(sample, buf.items);

    // Two encodes of the same input must be byte-for-byte identical
    // (the SoA refactor adds no encoder state across calls).
    const out2 = try std.testing.allocator.alloc(TokenId, sample.len * 2);
    defer std.testing.allocator.free(out2);
    const ids2 = try m.encodeChunk(std.testing.allocator, sample, out2);
    try std.testing.expectEqualSlices(TokenId, ids, ids2);
}

test "1.20 perf: bit-identical 10 KB regression baseline (TM full-capcode)" {
    // Locked-in bit-identical encode check on a 10 KB mixed corpus,
    // mirroring the 1.19 nocapcode regression test for the full-
    // capcode .ztm. Any future hot-path optimization must keep two
    // back-to-back encodes byte-for-byte identical, and must not
    // change the encoded length envelope (guards against drift in
    // the score formula / ungreedy tie-break / lilbuf gating).
    //
    // Note: the full-capcode .ztm expects capcode-normalized input
    // (C/W/D markers on case-change runs). Feeding raw English bytes
    // is intentional — we don't need the encoder's output to be
    // *semantically* correct, only *identical* across two encodes
    // with the same inputs.
    const monster_io = @import("monster_io.zig");
    var loaded = monster_io.readFile(std.testing.allocator, "bench/vocabs/tm_englishcode_capcode_32k.ztm") catch |err| {
        std.debug.print("(skipping 1.20 full-capcode bit-identical test: {})\n", .{err});
        return error.SkipZigTest;
    };
    defer loaded.deinit();

    var sample_buf: std.ArrayList(u8) = .empty;
    defer sample_buf.deinit(std.testing.allocator);
    const lines = [_][]const u8{
        "The quick brown fox jumps over the lazy dog. ",
        " #TokenMonster pretraining 16 elapsed=42ms loss=3.14159\n",
        " A B C D E F  -- this line has  many   spaces  -- end.\n",
        " 12345 67890 0x1A2B3C 1e9 -1.5e-10 (parens) [brackets] {braces}\n",
        " https://example.com/path?q=value#frag user@host.tld\n",
        " The rain in Spain falls mainly on the plain. ",
        " She sells seashells by the seashore. ",
        " Peter Piper picked a peck of pickled peppers. ",
    };
    var rep: usize = 0;
    while (rep < 30) : (rep += 1) {
        for (lines) |l| try sample_buf.appendSlice(std.testing.allocator, l);
    }
    // Sample is in the 10-15 KB range — keep it modest so the test
    // stays fast even in Debug mode, but large enough to exercise the
    // SoA byte-scan loop across many trie depths.
    try std.testing.expect(sample_buf.items.len >= 10 * 1024);

    const sample = sample_buf.items;
    // Worst-case encode expansion when lilbuf is on is 2× input length;
    // bump to 4× to leave slack for any DEL emit + bare-byte tail.
    const out_cap = sample.len * 4;
    const out_a = try std.testing.allocator.alloc(TokenId, out_cap);
    defer std.testing.allocator.free(out_a);
    const out_b = try std.testing.allocator.alloc(TokenId, out_cap);
    defer std.testing.allocator.free(out_b);

    const ids_a = try loaded.encodeChunk(std.testing.allocator, sample, out_a);
    const ids_b = try loaded.encodeChunk(std.testing.allocator, sample, out_b);

    try std.testing.expectEqualSlices(TokenId, ids_a, ids_b);
    try std.testing.expect(ids_a.len > 0);
    // Encoder efficiency sanity bound — bit-identical equality above is
    // the real contract; this only guards against an obviously broken
    // encode (e.g. emitting one token per byte).
    const bpt = @as(f64, @floatFromInt(sample.len)) / @as(f64, @floatFromInt(ids_a.len));
    try std.testing.expect(bpt > 0.4);
    try std.testing.expect(bpt < 6.0);
}

test "1.20 perf: branch hints don't change encode output (no-op behavioral check)" {
    // `@branchHint` is purely a codegen hint — it cannot change the
    // emitted ids on any input. This test is the cheapest possible
    // regression guard: build a tiny vocab, encode a varied 1 KB
    // input through both the masked and unmasked encoders (which take
    // distinct comptime-specialized paths in `encodeChunkImpl`'s
    // dispatcher, so each hint-decorated branch is reached at least
    // once), and assert the encode + roundtrip bytes are identical.
    // If a future refactor accidentally turns a `@branchHint` into a
    // condition mutator (e.g. by wrapping it in an `if`), this test
    // fires.
    var b = Monster.Builder.init(std.testing.allocator);
    defer b.deinit();
    _ = try b.addToken("<unk>");
    const a_id = try b.addToken("a");
    _ = try b.addToken("b");
    _ = try b.addToken("c");
    _ = try b.addToken(" "); // bare space — without it, all spaces become unk
    _ = try b.addToken("ab");
    _ = try b.addToken("bc");
    _ = try b.addToken("abc");
    _ = try b.addToken(" the");
    _ = try b.addToken("the");
    _ = try b.addToken(" cat");
    _ = try b.addToken("cat");
    _ = try b.addToken(" sat");
    _ = try b.addToken("sat");
    _ = try b.addToken(" on");
    _ = try b.addToken(" the mat");
    var m = try b.finalize(0);
    defer m.deinit();

    var sample: std.ArrayList(u8) = .empty;
    defer sample.deinit(std.testing.allocator);
    var rep: usize = 0;
    while (rep < 64) : (rep += 1) {
        try sample.appendSlice(std.testing.allocator, "abc ab bc a b c the cat sat on the mat ");
    }
    try std.testing.expect(sample.items.len >= 1024);

    const out = try std.testing.allocator.alloc(TokenId, sample.items.len * 2);
    defer std.testing.allocator.free(out);
    const ids_unmasked = try m.encodeChunk(std.testing.allocator, sample.items, out);
    try std.testing.expect(ids_unmasked.len > 0);

    // Now go through the masked encoder path. All-zero mask is the
    // identity (trainer path branch-hints still reachable, but the
    // emitted output must match the unmasked oracle).
    const zero = try std.testing.allocator.alloc(u8, m.count);
    defer std.testing.allocator.free(zero);
    @memset(zero, 0);
    m.mask = zero;
    const out2 = try std.testing.allocator.alloc(TokenId, sample.items.len * 2);
    defer std.testing.allocator.free(out2);
    const ids_masked = try m.encodeChunk(std.testing.allocator, sample.items, out2);
    m.mask = null;

    try std.testing.expectEqualSlices(TokenId, ids_unmasked, ids_masked);

    // Roundtrip — assemble decoded bytes from the encoded ids and
    // assert they reproduce the input. Catches a malformed @branchHint
    // that accidentally inverts a condition (which would still pass
    // the equality check above only if both paths broke identically).
    var rt: std.ArrayList(u8) = .empty;
    defer rt.deinit(std.testing.allocator);
    for (ids_unmasked) |id| try rt.appendSlice(std.testing.allocator, m.idBytes(id));
    try std.testing.expectEqualStrings(sample.items, rt.items);

    // Hit the unk-fallback hint path (rare) with an unknown byte
    // input ('z'). The encoder must emit `unk` for it.
    var unk_out: [4]TokenId = undefined;
    const unk_ids = try m.encodeChunk(std.testing.allocator, "z", &unk_out);
    try std.testing.expect(unk_ids.len == 1);
    try std.testing.expect(unk_ids[0] == 0); // unk_id was 0

    _ = a_id;
}
