//! Byte-level BPE tokenizer in the tiktoken format.
//!
//! Vocab is struct-of-arrays: a flat `bytes` buffer plus an `offsets`
//! index so token `id` lives at `bytes[offsets[id]..offsets[id+1]]`.
//! For tiktoken vocabs the token id IS the merge rank.
//!
//! `encodeChunk` keeps the merge state as SoA arrays
//! (`parts_start`, `parts_len`, `ranks`) and maintains the pair-rank
//! array across merges. The inner loop is a contiguous min-scan over
//! `ranks` (private `scanMin` — SIMD-swappable) plus two hashmap
//! lookups per merge for the affected neighbour pairs. Hashmap ops are
//! O(N) per chunk instead of O(N^2).
//!
//! ## 1.16 hot-table trade-off
//!
//! The 1.15 64 KB two-level hot table (`HotEntry` / `HOT_CAPACITY` /
//! `hotLookup`) is a clear win for the batch ×N + pinned production
//! shape: cl100k batch ×48 + `--pin-physical` gains +43 % (247 → 353
//! MB/s) by absorbing the SMT-sibling contention on the 644 KB
//! `by_bytes` StringHashMap into a 64 KB L1d-resident cache. But it's
//! a -16 % regression on cold single-thread cl100k (21.3 → 17.9 MB/s)
//! because the cold StringHashMap stays L2-warm in that regime, so
//! the hot table's miss-path work is pure overhead.
//!
//! Resolution (1.16): the hot table is **opt-in via `LoadOptions`**.
//! The bare `loadTiktokenFile` / `loadTiktokenBytes` constructors are
//! single-shot entry points (script-style / one-off encode) and
//! default the table OFF — single-thread is the relevant deployment
//! shape for those callers. `loadTiktokenFileWithOptions` /
//! `loadTiktokenBytesWithOptions` let batch / pool-aware callers
//! flip `hot_table = true` and get the +43 % win.
//! `hf_bridge.bpeFromHF` (and `WithOptions`) and `vocab_extend` /
//! `vocab_prune` / `doctor` keep the table on by default — those are
//! library / serving paths that overwhelmingly run multi-threaded.
//!
//! The opt-out costs zero extra branches on the hot path: the
//! existing `self.hot_table orelse return null` inside `hotLookup` IS
//! the gate. When `hot_table == null` the lookup short-circuits with
//! a single nullable check (one branch); the cold-table-only path
//! that follows is identical to the pre-1.15 encode loop.

const std = @import("std");
const TokenId = @import("token.zig").TokenId;
const Span = @import("token.zig").Span;

/// Initial segmentation strategy for `encodeChunkScratch`. The default
/// `.bpe_merge` is the canonical tiktoken / HF byte-level path: start
/// with one part per byte and merge adjacent pairs by lowest rank until
/// no more merges apply.
///
/// `.longest_match` is the SP-friendly path: greedily walk the input
/// emitting the longest piece in `by_bytes` at each position. SP-derived
/// vocabs encode pieces like `▁world` (U+2581 + "world") as single
/// tokens. The bpe-merge path can't navigate to those tokens from raw
/// bytes (no intermediate merges are defined), so it falls back to
/// `<0xNN>` byte tokens for every byte. The longest-match path looks
/// those pieces up directly and emits them in a single pass — exactly
/// what SP's own encoder does at the piece boundary level.
///
/// When `.longest_match` is selected, the merge loop still runs over
/// the resulting parts but is a no-op in practice for SP vocabs (no
/// pair of two complete pieces is itself a registered piece). Byte-
/// fallback still applies to bytes not covered by any piece.
///
/// `.optimal` runs a dynamic-programming segmentation that emits the
/// provably FEWEST tokens for the given vocab. Unlike `.bpe_merge`
/// (which follows the trained merge order) and `.longest_match` (which
/// greedily takes the longest piece at each position), the DP considers
/// every vocab token that matches at every position and picks the
/// globally minimal token count. It does NOT reproduce tiktoken / HF
/// byte-identical output — it intentionally beats greedy. See
/// `encodeChunkOptimal` for the algorithm and tie-break rules.
pub const EncodeMode = enum { bpe_merge, longest_match, optimal };

/// Two-level BPE merge-rank layout (1.15, post-1.14 agent E).
///
/// The merge loop in `encodeChunkScratch` / `bpe_heap.encodeWithFallback`
/// fires `by_bytes.get(pair)` against the 100K-entry cl100k StringHashMap
/// (~644 KB working set) on every candidate pair, every iteration. Two
/// SMT siblings sharing an L1d thrash that table. The 1.11 scaling
/// investigation flagged this as the dominant remaining SMT-contention
/// source.
///
/// Fix: a small direct-mapped front cache, sized to fit in L1d. On
/// lookup, hash the key, index the slot, compare a fingerprint plus the
/// inline key bytes, return on hit. On miss (slot empty, fingerprint
/// mismatch, or key collision) fall through to `by_bytes`.
///
/// Entries store the key bytes INLINE (no pointer chase to `self.bytes`),
/// so a hot-path hit is `hash → index → compare → return` with one cache
/// line touched per probe. Pieces longer than `HOT_KEY_INLINE_MAX` skip
/// the hot table; in practice the merge loop's hot pairs are short
/// (typical cl100k average word ~5 bytes, so most byte-pair merges are
/// 2-8 bytes), so the long tail naturally lives on the cold table.
///
/// Default sizing: 4096 entries × 16 B = **64 KB** — half of a typical
/// 32 KB L1d on Zen 3, but small enough to share L1d with the live merge
/// scratch arrays without evicting them. Tune via `HOT_CAPACITY`.
pub const HOT_CAPACITY: u32 = 4096;
/// Direct-mapped slot mask. HOT_CAPACITY must be a power of two.
pub const HOT_MASK: u32 = HOT_CAPACITY - 1;
/// Max inlined key length per entry. Picked so the entry is 16 bytes
/// (`hash:u32 + key_len:u8 + key_bytes:[7]u8 + id:u32`). Pieces longer
/// than this stay on the cold table only.
pub const HOT_KEY_INLINE_MAX: u8 = 7;
/// Reserved fingerprint that marks an empty slot. The chance a real
/// key hashes to exactly 0 is 1 in 2^32; on collision we just lose
/// that entry to the cold path — correctness still holds.
const HOT_EMPTY: u32 = 0;

comptime {
    // Power-of-two check so HOT_MASK is correct.
    if ((HOT_CAPACITY & (HOT_CAPACITY - 1)) != 0) {
        @compileError("HOT_CAPACITY must be a power of two");
    }
}

pub const HotEntry = extern struct {
    /// Non-zero on a populated slot; `HOT_EMPTY` (0) means empty.
    hash: u32 align(4),
    /// Number of valid bytes in `key_bytes`. 0 marks an empty slot too,
    /// but we key emptiness on `hash` since that's a single 32-bit load.
    key_len: u8,
    /// Inline storage for the merge-pair key bytes. The hot table is
    /// only populated for keys with `len <= HOT_KEY_INLINE_MAX`; the
    /// long tail spills to the cold `by_bytes` StringHashMap.
    key_bytes: [HOT_KEY_INLINE_MAX]u8,
    /// Token id == merge rank for tiktoken / HF byte-level vocabs.
    id: TokenId,

    comptime {
        if (@sizeOf(HotEntry) != 16) {
            @compileError("HotEntry must be exactly 16 bytes for a 64 KB / 4096-slot L1d budget");
        }
    }
};

/// 64-bit multiplicative hash specialized for short byte spans. Packs
/// up to 8 bytes into a u64 (zero-padded at the high end), folds in the
/// length, then runs a single multiplication step. ~6 instructions on
/// x86_64 (movzx + shifts + or + imul + xor-fold), branchless on the
/// hot path. Sub-3x faster than FNV-1a's per-byte multiply for the
/// 2-8 byte keys that dominate the merge loop.
pub inline fn hotHash(key: []const u8) u32 {
    // Pack key bytes (max HOT_KEY_INLINE_MAX = 7) into a u64. Unrolled
    // length-keyed branches let LLVM emit straight-line loads per len.
    var w: u64 = @as(u64, key.len);
    switch (key.len) {
        0 => {},
        1 => w = (w << 8) | key[0],
        2 => w = (w << 16) | (@as(u64, key[0]) << 8) | key[1],
        3 => w = (w << 24) | (@as(u64, key[0]) << 16) | (@as(u64, key[1]) << 8) | key[2],
        4 => w = (w << 32) | (@as(u64, key[0]) << 24) | (@as(u64, key[1]) << 16) | (@as(u64, key[2]) << 8) | key[3],
        5 => w = (w << 40) | (@as(u64, key[0]) << 32) | (@as(u64, key[1]) << 24) | (@as(u64, key[2]) << 16) | (@as(u64, key[3]) << 8) | key[4],
        6 => w = (w << 48) | (@as(u64, key[0]) << 40) | (@as(u64, key[1]) << 32) | (@as(u64, key[2]) << 24) | (@as(u64, key[3]) << 16) | (@as(u64, key[4]) << 8) | key[5],
        else => w = (w << 56) | (@as(u64, key[0]) << 48) | (@as(u64, key[1]) << 40) | (@as(u64, key[2]) << 32) | (@as(u64, key[3]) << 24) | (@as(u64, key[4]) << 16) | (@as(u64, key[5]) << 8) | key[6],
    }
    // Single mix step: multiply by a 64-bit prime, xor-fold to u32.
    const mix: u64 = w *% 0x9E3779B97F4A7C15;
    var h: u32 = @truncate(mix ^ (mix >> 32));
    // Force non-zero — 0 is our empty-slot sentinel. Keys that
    // organically hash to 0 (rare) get bumped to 1.
    if (h == 0) h = 1;
    return h;
}

pub const Bpe = struct {
    allocator: std.mem.Allocator,

    /// Concatenated token bytes, back to back.
    bytes: []u8,
    /// `offsets[i]..offsets[i+1]` is token `i`. Length is `count + 1`.
    offsets: []u32,
    count: u32,

    /// Reverse lookup. Keys borrow into `self.bytes`.
    by_bytes: std.StringHashMap(TokenId),

    /// Two-level layout front cache. `null` for vocabs that didn't go
    /// through `buildHotTable` (synthetic test vocabs that bypass the
    /// standard loaders). Populated by `loadTiktokenBytes`, `bpeFromHF`,
    /// and `bpeFromSP`. See `HotEntry` / `HOT_CAPACITY` docs above.
    /// Owned by `Bpe`; freed in `deinit`.
    hot_table: ?[]HotEntry = null,

    /// Optional byte-fallback table: byte value -> token id. Populated by
    /// SP-BPE loaders when the source model has `trainer_spec.byte_fallback`
    /// set and dedicated `<0xNN>` byte tokens in the vocab. After the merge
    /// loop, any length-1 chunk whose raw byte has a mapped id is rewritten
    /// from the multi-byte hashmap lookup result to that mapped id, so an
    /// unknown byte resolves to its dedicated byte token instead of the
    /// `maxInt` sentinel.
    ///
    /// `null` for tiktoken / HF byte-level / non-SP vocabs; the post-pass
    /// is a single nullable-branch with no work when this is null.
    byte_fallback: ?[256]TokenId = null,

    /// See `EncodeMode`. Default `.bpe_merge` keeps tiktoken / HF byte-
    /// level callers bit-identical; `bpeFromSP` flips this to
    /// `.longest_match` so SP vocabs can encode multi-byte pieces in a
    /// single pass without synthesizing intermediate merge rules.
    encode_mode: EncodeMode = .bpe_merge,

    /// Cached maximum piece length in `bytes`, used to cap the per-
    /// position scan in the `.longest_match` path so we don't probe
    /// arbitrarily large prefixes. Recomputed from `offsets` whenever
    /// a fresh `Bpe` is constructed; defaults to 1 for an empty vocab
    /// (no piece longer than that exists yet) and to the actual max
    /// when loaded via tiktoken or `bpeFromSP`. Bumping this above the
    /// real max would only cost extra hashmap probes per position.
    max_piece_len: u32 = 1,

    /// Optional per-token merge priority for the `.longest_match` path.
    /// SP's reference encoder picks among overlapping prefix matches by
    /// score (lowest negative score = highest priority), NOT by raw
    /// length. When this is populated, the longest-match encoder
    /// switches from "pick the longest prefix" to "pick the prefix with
    /// the lowest piece_ranks[id]". Tiktoken / HF byte-level callers
    /// leave this null and the encoder reverts to length-based selection
    /// (which is moot for `.bpe_merge` mode anyway). Owned by `Bpe`.
    piece_ranks: ?[]u32 = null,

    /// HF BPE `ignore_merges` flag. When true, the encoder first checks
    /// whether the entire pretok chunk is already a single piece in
    /// `by_bytes`; if so, that id is emitted directly and the merge loop
    /// is skipped. Llama-3 / GPT-4-style tokenizers set this to bypass
    /// the merge ordering for multilingual pieces that the merge loop
    /// would otherwise mis-split (the merge-rank-by-id heuristic for HF
    /// vocabs can produce a different segmentation than HF's pair-rank
    /// loop for non-ASCII sequences, but the whole-chunk lookup matches
    /// HF unconditionally). See `bpeFromHF` for where this is populated
    /// from the parsed `tokenizer.json`. Default false keeps tiktoken /
    /// older HF BPE behavior bit-identical.
    ignore_merges: bool = false,

    pub fn deinit(self: *Bpe) void {
        self.by_bytes.deinit();
        if (self.bytes.len > 0) self.allocator.free(self.bytes);
        if (self.offsets.len > 0) self.allocator.free(self.offsets);
        if (self.piece_ranks) |pr| self.allocator.free(pr);
        if (self.hot_table) |ht| self.allocator.free(ht);
        self.* = .{
            .allocator = self.allocator,
            .bytes = &.{},
            .offsets = &.{},
            .count = 0,
            .by_bytes = std.StringHashMap(TokenId).init(self.allocator),
            .byte_fallback = null,
            .encode_mode = .bpe_merge,
            .max_piece_len = 1,
            .piece_ranks = null,
            .hot_table = null,
            .ignore_merges = false,
        };
    }

    /// Allocate + populate a hot table from the vocab in id order. Top
    /// pieces by id win the slot on collision (id == rank for tiktoken
    /// vocabs, so this is "most-frequent merge wins"). Pieces longer
    /// than `HOT_KEY_INLINE_MAX` skip the hot table entirely (the cold
    /// `by_bytes` map still serves them).
    ///
    /// Caller is responsible for freeing via `Bpe.deinit` (the returned
    /// slice is assigned to `self.hot_table`).
    pub fn buildHotTable(
        allocator: std.mem.Allocator,
        bytes: []const u8,
        offsets: []const u32,
        count: u32,
    ) ![]HotEntry {
        const table = try allocator.alloc(HotEntry, HOT_CAPACITY);
        errdefer allocator.free(table);
        // Zero-initialize: hash=0 marks empty.
        @memset(table, .{
            .hash = HOT_EMPTY,
            .key_len = 0,
            .key_bytes = [_]u8{0} ** HOT_KEY_INLINE_MAX,
            .id = 0,
        });

        // Iterate in id order — id == merge rank for tiktoken, so low
        // ids are the highest-frequency merges. First-wins on slot
        // collision keeps the more frequent merge in the hot table.
        var r: u32 = 0;
        while (r < count) : (r += 1) {
            const key_len: u32 = offsets[r + 1] - offsets[r];
            if (key_len == 0 or key_len > HOT_KEY_INLINE_MAX) continue;
            const key = bytes[offsets[r] .. offsets[r] + key_len];
            const h = hotHash(key);
            const slot = h & HOT_MASK;
            if (table[slot].hash != HOT_EMPTY) continue; // first-wins
            table[slot] = .{
                .hash = h,
                .key_len = @intCast(key_len),
                .key_bytes = [_]u8{0} ** HOT_KEY_INLINE_MAX,
                .id = r,
            };
            // Memcpy the key bytes inline.
            var i: u32 = 0;
            while (i < key_len) : (i += 1) {
                table[slot].key_bytes[i] = key[i];
            }
        }
        return table;
    }

    /// Atomic counters for hot-table hit / miss instrumentation. Cleared
    /// to zero on process start. Updated only when the `HOT_INSTRUMENT`
    /// compile-time switch is on; the no-op compile path leaves the
    /// merge loop unchanged. Bench code reads these directly.
    pub var hot_hits: u64 = 0;
    pub var hot_misses: u64 = 0;
    /// Compile-time switch — flip to `true` to get a hit-rate readout
    /// from the bench. Adds two atomic-add ops per merge candidate pair
    /// so leave off in the production build.
    pub const HOT_INSTRUMENT = false;

    /// Hot-table front lookup. Returns `null` on miss; caller should
    /// fall through to `by_bytes.get(key)`. Inlined into the merge loop
    /// so the hot path compiles to: length check → hash → slot index →
    /// fingerprint compare → key compare → return.
    pub inline fn hotLookup(self: *const Bpe, key: []const u8) ?TokenId {
        // Long keys are not in the hot table by construction; the cheap
        // length-check kills the load before it would have missed.
        if (key.len == 0 or key.len > HOT_KEY_INLINE_MAX) {
            if (comptime HOT_INSTRUMENT) _ = @atomicRmw(u64, &hot_misses, .Add, 1, .monotonic);
            return null;
        }
        const table = self.hot_table orelse {
            if (comptime HOT_INSTRUMENT) _ = @atomicRmw(u64, &hot_misses, .Add, 1, .monotonic);
            return null;
        };
        const h = hotHash(key);
        const slot = table[h & HOT_MASK];
        // Slot empty: hash=0.
        // Slot occupied with different key: hash mismatch.
        // Same hash but different content (very rare): length or
        // byte-compare mismatch -> caller falls through to cold table.
        if (slot.hash != h) {
            if (comptime HOT_INSTRUMENT) _ = @atomicRmw(u64, &hot_misses, .Add, 1, .monotonic);
            return null;
        }
        if (slot.key_len != key.len) {
            if (comptime HOT_INSTRUMENT) _ = @atomicRmw(u64, &hot_misses, .Add, 1, .monotonic);
            return null;
        }
        // Inline equality check on the inlined key bytes (max 7). std's
        // `mem.eql` lowers to a single-load + compare on small slices,
        // which beats the byte-by-byte loop here when the compiler
        // can't prove key.len == slot.key_len at the call site.
        if (!std.mem.eql(u8, slot.key_bytes[0..slot.key_len], key)) {
            if (comptime HOT_INSTRUMENT) _ = @atomicRmw(u64, &hot_misses, .Add, 1, .monotonic);
            return null;
        }
        if (comptime HOT_INSTRUMENT) _ = @atomicRmw(u64, &hot_hits, .Add, 1, .monotonic);
        return slot.id;
    }

    /// Knobs the `*WithOptions` constructors honor. Additive; new fields
    /// land with sensible defaults so existing callers keep compiling.
    ///
    /// `hot_table`: build the 1.15 two-level 64 KB direct-mapped
    /// merge-rank front cache. See the module header comment for the
    /// performance trade-off (batch ×N + pin: +43 %; single-thread: -16 %).
    /// Default `false` because the *WithOptions constructors are most
    /// useful for callers who already know which deployment shape they're
    /// targeting; the bare `loadTiktokenFile` / `loadTiktokenBytes`
    /// constructors apply the same default (off — single-shot bias).
    pub const LoadOptions = struct {
        hot_table: bool = false,
    };

    /// Read `path` and parse its tiktoken vocab. Uses the global
    /// single-threaded `Io` since loading is a synchronous one-shot.
    ///
    /// Hot table is **off by default** — see module header for why.
    /// Callers targeting batch / pool encode should prefer
    /// `loadTiktokenFileWithOptions(.{ .hot_table = true })`.
    pub fn loadTiktokenFile(allocator: std.mem.Allocator, path: []const u8) !Bpe {
        return loadTiktokenFileWithOptions(allocator, path, .{});
    }

    /// As `loadTiktokenFile` but with explicit `LoadOptions`.
    pub fn loadTiktokenFileWithOptions(
        allocator: std.mem.Allocator,
        path: []const u8,
        opts: LoadOptions,
    ) !Bpe {
        const io = std.Io.Threaded.global_single_threaded.io();
        const contents = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
        defer allocator.free(contents);
        return loadTiktokenBytesWithOptions(allocator, contents, opts);
    }

    /// Parse a tiktoken vocab from an in-memory buffer.
    /// Each non-empty line: `<base64-of-bytes> <decimal-rank>\n`.
    /// Rank order is not assumed; entries are indexed by rank into the
    /// final SoA arrays.
    ///
    /// Hot table is **off by default** — see module header. Use
    /// `loadTiktokenBytesWithOptions` for the explicit-opts form.
    pub fn loadTiktokenBytes(allocator: std.mem.Allocator, contents: []const u8) !Bpe {
        return loadTiktokenBytesWithOptions(allocator, contents, .{});
    }

    /// As `loadTiktokenBytes` but with explicit `LoadOptions`.
    pub fn loadTiktokenBytesWithOptions(
        allocator: std.mem.Allocator,
        contents: []const u8,
        opts: LoadOptions,
    ) !Bpe {
        const decoder = std.base64.standard.Decoder;

        // First pass: count entries and decoded byte size, and discover
        // the maximum rank so we can size offsets exactly.
        var line_count: u32 = 0;
        var total_bytes: usize = 0;
        var max_rank: u32 = 0;
        {
            var it = std.mem.splitScalar(u8, contents, '\n');
            while (it.next()) |raw| {
                const line = std.mem.trimEnd(u8, raw, "\r");
                if (line.len == 0) continue;
                const sp = std.mem.indexOfScalar(u8, line, ' ') orelse return error.InvalidTiktokenLine;
                const b64 = line[0..sp];
                const rank_str = line[sp + 1 ..];
                const rank = try std.fmt.parseInt(u32, rank_str, 10);
                const n = try decoder.calcSizeForSlice(b64);
                line_count += 1;
                total_bytes += n;
                if (rank > max_rank) max_rank = rank;
            }
        }

        if (line_count == 0) {
            return .{
                .allocator = allocator,
                .bytes = &.{},
                .offsets = &.{},
                .count = 0,
                .by_bytes = std.StringHashMap(TokenId).init(allocator),
            };
        }

        // Ranks must form a dense 0..count-1 set for the SoA layout to
        // hold. tiktoken vocabs always satisfy this; reject otherwise.
        if (@as(usize, max_rank) + 1 != line_count) return error.NonContiguousRanks;
        const count = line_count;

        // Allocate the SoA buffers up front.
        const bytes = try allocator.alloc(u8, total_bytes);
        errdefer allocator.free(bytes);
        const offsets = try allocator.alloc(u32, @as(usize, count) + 1);
        errdefer allocator.free(offsets);

        // Side table: stage decodes in arrival order, then re-emit in
        // rank order so the final `bytes` array is laid out by id.
        const Entry = struct { rank: u32, start: u32, len: u32 };
        const entries = try allocator.alloc(Entry, count);
        defer allocator.free(entries);

        const staging = try allocator.alloc(u8, total_bytes);
        defer allocator.free(staging);

        var staging_off: u32 = 0;
        var idx: u32 = 0;
        {
            var it = std.mem.splitScalar(u8, contents, '\n');
            while (it.next()) |raw| {
                const ln = std.mem.trimEnd(u8, raw, "\r");
                if (ln.len == 0) continue;
                const sp = std.mem.indexOfScalar(u8, ln, ' ').?;
                const b64 = ln[0..sp];
                const rank = try std.fmt.parseInt(u32, ln[sp + 1 ..], 10);
                if (rank >= count) return error.NonContiguousRanks;

                const n: u32 = @intCast(try decoder.calcSizeForSlice(b64));
                try decoder.decode(staging[staging_off .. staging_off + n], b64);
                entries[idx] = .{ .rank = rank, .start = staging_off, .len = n };
                staging_off += n;
                idx += 1;
            }
        }

        // Build per-rank source index, catching dupes and gaps.
        const rank_src = try allocator.alloc(u32, count);
        defer allocator.free(rank_src);
        const lens = try allocator.alloc(u32, count);
        defer allocator.free(lens);
        @memset(lens, std.math.maxInt(u32));
        for (entries) |e| {
            if (lens[e.rank] != std.math.maxInt(u32)) return error.DuplicateRank;
            lens[e.rank] = e.len;
            rank_src[e.rank] = e.start;
        }

        // Emit bytes in rank order. offsets[r]..offsets[r+1] is token r.
        offsets[0] = 0;
        var write: u32 = 0;
        for (0..count) |r| {
            const len = lens[r];
            if (len == std.math.maxInt(u32)) return error.MissingRank;
            const src = rank_src[r];
            @memcpy(bytes[write .. write + len], staging[src .. src + len]);
            write += len;
            offsets[r + 1] = write;
        }
        std.debug.assert(write == total_bytes);

        var by_bytes = std.StringHashMap(TokenId).init(allocator);
        errdefer by_bytes.deinit();
        try by_bytes.ensureTotalCapacity(count);
        var max_piece_len: u32 = 1;
        for (0..count) |r| {
            const key = bytes[offsets[r]..offsets[r + 1]];
            try by_bytes.put(key, @intCast(r));
            if (key.len > max_piece_len) max_piece_len = @intCast(key.len);
        }

        // Build the L1d-resident front cache for the BPE merge loop —
        // opt-in via `opts.hot_table`. See `HotEntry` / `HOT_CAPACITY`
        // docs and the module header for the perf trade-off.
        const hot_table: ?[]HotEntry = if (opts.hot_table)
            try buildHotTable(allocator, bytes, offsets, count)
        else
            null;
        errdefer if (hot_table) |ht| allocator.free(ht);

        return .{
            .allocator = allocator,
            .bytes = bytes,
            .offsets = offsets,
            .count = count,
            .by_bytes = by_bytes,
            .max_piece_len = max_piece_len,
            .hot_table = hot_table,
        };
    }

    pub fn idBytes(self: *const Bpe, id: TokenId) []const u8 {
        std.debug.assert(id < self.count);
        return self.bytes[self.offsets[id]..self.offsets[id + 1]];
    }

    /// Sentinel meaning "no merge possible at this index". Picked so a
    /// straight min-scan over `ranks` naturally skips it.
    const RANK_INVALID: u32 = std.math.maxInt(u32);

    /// Max chunk length served from on-stack scratch. Tuned for typical
    /// pre-tokenized chunks (cl100k average is ~3-10 bytes); anything
    /// larger spills to the allocator.
    const STACK_LIMIT: usize = 256;

    /// Min-rank scan over `ranks`. Routes to the SIMD path in
    /// `src/simd_min.zig` which lowers to `vpminud` on AVX2/AVX-512 and
    /// `uminv` on NEON. The narrow path is `@Vector(16, u32)` (one
    /// ZMM on AVX-512, two YMM on AVX2). When the target has AVX-512F
    /// the wide path `@Vector(32, u32)` engages for spans of >= 32
    /// ranks (comptime-gated; compiles out on AVX2/aarch64/wasm). The
    /// scalar tail handles ranks shorter than the vector width.
    const simd_min = @import("simd_min.zig");
    fn scanMin(ranks: []const u32) struct { idx: u32, rank: u32 } {
        const r = simd_min.scanMin(ranks);
        if (r) |hit| return .{ .idx = hit.idx, .rank = hit.rank };
        return .{ .idx = 0, .rank = RANK_INVALID };
    }

    /// Encode `chunk` into `out`. `out.len` must be >= chunk.len, since
    /// a worst-case-no-merges encoding yields one id per byte.
    ///
    /// Convenience wrapper for `encodeChunkScratch` that pulls heap
    /// spillover (chunks > `STACK_LIMIT` bytes) from `self.allocator`.
    /// Hot-path callers should prefer `encodeChunkScratch` with a
    /// per-thread arena to avoid GPA traffic on large chunks.
    pub fn encodeChunk(self: *const Bpe, chunk: []const u8, out: []TokenId) []TokenId {
        return self.encodeChunkScratch(self.allocator, chunk, out);
    }

    /// Same as `encodeChunk` but the heap-spillover scratch (for chunks
    /// > `STACK_LIMIT` bytes, plus the linked-list/heap path for chunks
    /// over `HEAP_THRESHOLD`) is allocated from `scratch` instead of
    /// `self.allocator`. Pass a per-thread `ArenaAllocator` to get
    /// zero-allocation behavior after the first warm-up call.
    ///
    /// `scratch` may be any allocator; with an arena reset between
    /// outer pipeline calls the arena buffers are retained and
    /// subsequent calls hit the arena's fast bump path.
    pub fn encodeChunkScratch(
        self: *const Bpe,
        scratch: std.mem.Allocator,
        chunk: []const u8,
        out: []TokenId,
    ) []TokenId {
        if (chunk.len == 0) return out[0..0];
        std.debug.assert(out.len >= chunk.len);

        // HF BPE `ignore_merges` short-circuit. When the source
        // tokenizer.json sets `model.ignore_merges = true` (Llama-3,
        // GPT-4-style multilingual vocabs), the reference Rust encoder
        // probes the entire pretok chunk against the vocab BEFORE running
        // the merge loop and, on a hit, emits that single id. Without
        // this, the rank-by-id heuristic the legacy byte-level path uses
        // can disagree with HF's pair-rank merge loop on non-ASCII
        // sequences (e.g. ` Федерации` would split into 3 pieces instead
        // of the single id 111112). See `bpeFromHF` for the parse path.
        if (self.ignore_merges) {
            if (self.hotLookup(chunk)) |id| {
                out[0] = id;
                return out[0..1];
            }
            if (self.by_bytes.get(chunk)) |id| {
                out[0] = id;
                return out[0..1];
            }
        }

        // Longest-match mode: walk the input once, emit the longest
        // matching piece at each position, byte-fallback otherwise.
        // SP-derived vocabs use this; tiktoken / HF byte-level stay on
        // the default `.bpe_merge` path below.
        if (self.encode_mode == .longest_match) {
            return self.encodeLongestMatch(scratch, chunk, out);
        }

        // Optimal mode: dynamic-programming minimum-token segmentation.
        // Beats greedy / merge-order encoding with a provable guarantee.
        if (self.encode_mode == .optimal) {
            return self.encodeChunkOptimal(scratch, chunk, out);
        }

        // SoA scratch. Stack-sized for the common path; scratch-allocator
        // fallback (catch unreachable on OOM, same as before) for
        // pathological chunks. `scratch` should typically be a reset-
        // retain ArenaAllocator owned by the calling thread.
        var stack_starts: [STACK_LIMIT]u32 = undefined;
        var stack_lens: [STACK_LIMIT]u32 = undefined;
        var stack_ranks: [STACK_LIMIT]u32 = undefined;

        const parts_start: []u32 = if (chunk.len <= STACK_LIMIT)
            stack_starts[0..chunk.len]
        else
            scratch.alloc(u32, chunk.len) catch unreachable;
        const parts_len: []u32 = if (chunk.len <= STACK_LIMIT)
            stack_lens[0..chunk.len]
        else
            scratch.alloc(u32, chunk.len) catch unreachable;
        const ranks: []u32 = if (chunk.len <= STACK_LIMIT)
            stack_ranks[0..chunk.len]
        else
            scratch.alloc(u32, chunk.len) catch unreachable;

        // Init parts as single bytes. parts_start[i+1] == parts_start[i]
        // + parts_len[i] — the adjacency invariant that lets us look up
        // pair bytes as a single slice into `chunk`.
        var live: u32 = @intCast(chunk.len);
        for (0..chunk.len) |i| {
            parts_start[i] = @intCast(i);
            parts_len[i] = 1;
        }

        // Initial pair ranks: live - 1 lookups. ranks[live-1] is unused
        // (no right neighbour) and is set to RANK_INVALID so the scan
        // never picks it.
        //
        // Lookup order: hot table first, then `by_bytes` on miss. The
        // hot path is ~10 instructions (hash, slot index, fingerprint
        // compare, key bytes compare) and stays in L1d; the cold path
        // is the StringHashMap probe that thrashes when SMT siblings
        // share L1d. See `HotEntry` docs above.
        if (live >= 2) {
            var i: u32 = 0;
            while (i + 1 < live) : (i += 1) {
                const start = parts_start[i];
                const total = parts_len[i] + parts_len[i + 1];
                const key = chunk[start .. start + total];
                if (self.hotLookup(key)) |r| {
                    ranks[i] = r;
                } else if (self.by_bytes.get(key)) |r| {
                    ranks[i] = r;
                } else {
                    ranks[i] = RANK_INVALID;
                }
            }
        }
        if (live > 0) ranks[live - 1] = RANK_INVALID;

        // For long chunks the O(live) shift per merge dominates; the
        // 4-ary heap encoder in `src/bpe_heap.zig` switches to a linked
        // list + O(log live) heap pops. Threshold tuned for the cl100k
        // common case where average chunk length is ~5 bytes; heap path
        // engages only on outliers (long unbroken identifiers, base64,
        // 'aaaa...' style inputs).
        const HEAP_THRESHOLD: u32 = 64;
        if (live > HEAP_THRESHOLD) {
            const heap_path = @import("bpe_heap.zig");
            const n: usize = live;
            // Pulled from `scratch` so a per-thread arena retains capacity
            // across calls. No explicit free — arena owns the lifetime.
            const prev = scratch.alloc(u32, n) catch unreachable;
            const next = scratch.alloc(u32, n) catch unreachable;
            const heap_buf = scratch.alloc(heap_path.HeapNode, 3 * n) catch unreachable;

            // Init linked list.
            for (0..n) |i| {
                prev[i] = if (i == 0) heap_path.NONE else @intCast(i - 1);
                next[i] = if (i + 1 == n) heap_path.NONE else @intCast(i + 1);
            }
            const bf_ptr: ?*const [256]TokenId = if (self.byte_fallback) |*t| t else null;
            const written = heap_path.encodeWithFallback(
                chunk,
                n,
                parts_start,
                parts_len,
                ranks,
                prev,
                next,
                &self.by_bytes,
                bf_ptr,
                heap_buf,
                out,
                self.hot_table,
            );
            return out[0..written];
        }

        // Merge loop. Each iteration: scan ranks[0..live-1] for the min,
        // splice out parts[mi+1], then refresh at most two neighbouring
        // pair-ranks (mi-1 and mi). Two hashmap ops per merge.
        while (live >= 2) {
            const scan = scanMin(ranks[0 .. live - 1]);
            if (scan.rank == RANK_INVALID) break;
            const mi = scan.idx;

            // Merge parts[mi] and parts[mi+1] in-place: extend mi's
            // length, then shift the tail (parts + ranks) down one
            // slot. Shift cost is O(live) per merge; trivial for the
            // small `live` typical of pre-tokenized chunks.
            parts_len[mi] += parts_len[mi + 1];

            var j: u32 = mi + 1;
            while (j + 1 < live) : (j += 1) {
                parts_start[j] = parts_start[j + 1];
                parts_len[j] = parts_len[j + 1];
                ranks[j] = ranks[j + 1];
            }
            live -= 1;

            // Refresh neighbour ranks. After the shift, mi is the
            // merged part; mi-1's right-pair and mi's right-pair are
            // both stale. Same hot-table-first lookup pattern.
            //
            // Left neighbour: pair (mi-1, mi). Only exists if mi > 0.
            if (mi > 0) {
                const li = mi - 1;
                const start = parts_start[li];
                const total = parts_len[li] + parts_len[mi];
                const key = chunk[start .. start + total];
                if (self.hotLookup(key)) |r| {
                    ranks[li] = r;
                } else if (self.by_bytes.get(key)) |r| {
                    ranks[li] = r;
                } else {
                    ranks[li] = RANK_INVALID;
                }
            }
            // Right neighbour: pair (mi, mi+1). Only exists if mi+1 <
            // live; otherwise mi is the last part and its rank slot is
            // unused.
            if (mi + 1 < live) {
                const start = parts_start[mi];
                const total = parts_len[mi] + parts_len[mi + 1];
                const key = chunk[start .. start + total];
                if (self.hotLookup(key)) |r| {
                    ranks[mi] = r;
                } else if (self.by_bytes.get(key)) |r| {
                    ranks[mi] = r;
                } else {
                    ranks[mi] = RANK_INVALID;
                }
            } else {
                ranks[mi] = RANK_INVALID;
            }
        }

        // Resolve final parts to ids. Singletons assume the vocab
        // covers all 256 bytes (tiktoken invariant); a missing key
        // emits maxInt as a sentinel so callers can validate.
        //
        // SP byte-fallback: if a length-1 unmerged part's raw byte is
        // present in the byte-fallback table, prefer that id over the
        // hashmap result. Faithfully reproduces SP's behavior where a
        // byte that the merge loop couldn't grow falls back to its
        // dedicated `<0xNN>` token rather than an unk sentinel. The
        // nullable check on `byte_fallback` is the only cost when the
        // field is unset (the common tiktoken / HF byte-level case).
        const bf_table = self.byte_fallback;
        var w: usize = 0;
        var k: u32 = 0;
        while (k < live) : (k += 1) {
            const start = parts_start[k];
            const len = parts_len[k];
            const key = chunk[start .. start + len];
            var id: TokenId = if (self.hotLookup(key)) |r|
                r
            else
                self.by_bytes.get(key) orelse std.math.maxInt(TokenId);
            if (len == 1) {
                if (bf_table) |*tbl| {
                    const mapped = tbl[chunk[start]];
                    if (mapped != std.math.maxInt(TokenId)) id = mapped;
                }
            }
            out[w] = id;
            w += 1;
        }
        return out[0..w];
    }

    /// UTF-8 codepoint length of the byte at `b[0]`. Returns 1 for
    /// invalid leading bytes — we degrade gracefully into byte-level
    /// fallback rather than panic on malformed input.
    inline fn cpLen(b: u8) usize {
        if (b < 0x80) return 1;
        if (b < 0xC0) return 1; // continuation byte (shouldn't lead); treat as 1
        if (b < 0xE0) return 2;
        if (b < 0xF0) return 3;
        return 4;
    }

    /// Greedy longest-match encoder used by the `.longest_match`
    /// `EncodeMode`. At each position try prefixes of length
    /// `min(remaining, max_piece_len)` down to 1 against `by_bytes`,
    /// emit the first hit, advance. If no prefix matches at all, fall
    /// back to the byte-fallback table (or `maxInt` sentinel if no
    /// table). Doesn't allocate. O(input × max_piece_len) hashmap probes
    /// in the worst case; in practice the longest piece matches almost
    /// always so it's closer to O(input / avg_piece_len).
    ///
    /// SP-derived vocabs (`bpeFromSP` sets this mode) go through a
    /// codepoint-aware BPE merge instead — see `encodeSpBpe` below.
    /// Plain longest-match remains useful for synthetic test vocabs
    /// where no `piece_ranks` table is populated.
    fn encodeLongestMatch(
        self: *const Bpe,
        scratch: std.mem.Allocator,
        chunk: []const u8,
        out: []TokenId,
    ) []TokenId {
        // When piece_ranks is populated, the SP-flavored BPE merge wants
        // to run instead. It needs scratch for its parts arrays and
        // (on long inputs) the heap path's prev/next/heap_buf.
        if (self.piece_ranks != null) {
            return self.encodeSpBpe(scratch, chunk, out);
        }
        const bf_table = self.byte_fallback;
        const max_len: usize = self.max_piece_len;
        var w: usize = 0;
        var pos: usize = 0;
        while (pos < chunk.len) {
            const remaining = chunk.len - pos;
            var try_len = if (remaining < max_len) remaining else max_len;
            var matched: ?TokenId = null;
            while (try_len > 0) : (try_len -= 1) {
                const key = chunk[pos .. pos + try_len];
                if (self.by_bytes.get(key)) |id| {
                    matched = id;
                    break;
                }
            }
            if (matched) |id| {
                out[w] = id;
                w += 1;
                pos += try_len;
            } else {
                // No piece (not even length 1) matched. Fall back to the
                // byte-fallback table; if unset, emit the maxInt sentinel
                // and step one byte forward so callers can validate.
                var id: TokenId = std.math.maxInt(TokenId);
                if (bf_table) |*tbl| {
                    const mapped = tbl[chunk[pos]];
                    if (mapped != std.math.maxInt(TokenId)) id = mapped;
                }
                out[w] = id;
                w += 1;
                pos += 1;
            }
        }
        return out[0..w];
    }

    /// Offsets variant of `encodeLongestMatch`. Each emitted id gets a
    /// span covering the bytes it consumed in the original buffer.
    fn encodeLongestMatchWithOffsets(
        self: *const Bpe,
        scratch: std.mem.Allocator,
        chunk: []const u8,
        chunk_offset: u32,
        out_ids: []TokenId,
        out_offsets: []Span,
    ) usize {
        if (self.piece_ranks != null) {
            return self.encodeSpBpeWithOffsets(scratch, chunk, chunk_offset, out_ids, out_offsets);
        }
        const bf_table = self.byte_fallback;
        const max_len: usize = self.max_piece_len;
        var w: usize = 0;
        var pos: usize = 0;
        while (pos < chunk.len) {
            const remaining = chunk.len - pos;
            var try_len = if (remaining < max_len) remaining else max_len;
            var matched: ?TokenId = null;
            while (try_len > 0) : (try_len -= 1) {
                const key = chunk[pos .. pos + try_len];
                if (self.by_bytes.get(key)) |id| {
                    matched = id;
                    break;
                }
            }
            const span_start: u32 = chunk_offset + @as(u32, @intCast(pos));
            if (matched) |id| {
                out_ids[w] = id;
                out_offsets[w] = .{ .start = span_start, .end = span_start + @as(u32, @intCast(try_len)) };
                w += 1;
                pos += try_len;
            } else {
                var id: TokenId = std.math.maxInt(TokenId);
                if (bf_table) |*tbl| {
                    const mapped = tbl[chunk[pos]];
                    if (mapped != std.math.maxInt(TokenId)) id = mapped;
                }
                out_ids[w] = id;
                out_offsets[w] = .{ .start = span_start, .end = span_start + 1 };
                w += 1;
                pos += 1;
            }
        }
        return w;
    }

    // --- optimal DP minimum-token segmentation ----------------------
    //
    // `dp[i]` = minimum number of tokens needed to encode `chunk[0..i]`.
    // `dp[0] = 0`; everything else starts at the DP_INF sentinel.
    //
    // At each reachable position `i` we enumerate every vocab token that
    // matches starting at `i` (prefixes of length 1..max_piece_len,
    // probed hot-table-first then `by_bytes`), and relax
    //   dp[i + len] = min(dp[i + len], dp[i] + 1)
    // recording a backpointer (prev position + token id) so the chosen
    // segmentation can be reconstructed. Because every relaxation adds
    // exactly one token, scanning positions left-to-right means `dp[i]`
    // is final by the time we expand `i` (all relaxations into `i` come
    // from positions < i). The result is the provably fewest tokens for
    // the given vocab — it beats both `.bpe_merge` (merge-order) and
    // `.longest_match` (greedy) whenever those leave tokens on the table.
    //
    // Coverage: if no vocab piece (not even length 1) matches at `i`, we
    // still relax `dp[i+1]` with a length-1 fallback token (the
    // `byte_fallback` table entry if present, else the `maxInt` sentinel)
    // so the chain always reaches `n`. This mirrors the greedy paths'
    // single-byte fallback and keeps the encoding lossless under
    // `decoder.concat` (the token bytes concatenate back to the input).
    //
    // Deterministic tie-break: when two candidate tokens ending at the
    // same position give the SAME (minimal) count, we keep the one with
    // the LONGER token; if lengths also tie, we keep the one with the
    // LOWER token id. Documented and enforced in `relaxOptimal` below.
    // This guarantees a single, reproducible segmentation for every
    // input regardless of probe / iteration order.
    const DP_INF: u32 = std.math.maxInt(u32);

    /// One backpointer cell for the optimal DP. `count` is `dp[j]`;
    /// `prev` is the start position of the last token; `id`/`len`
    /// describe that token (kept for the tie-break and for emission).
    const OptCell = struct {
        count: u32,
        prev: u32,
        id: TokenId,
        len: u32,
    };

    /// Relax `dp[j]` (cell `c`) with a candidate token of length `len`
    /// and id `tid` whose predecessor is position `prev` with count
    /// `prev_count`. Applies the documented tie-break (prefer longer
    /// token, then lower id) on equal counts. Returns the (possibly
    /// updated) cell.
    inline fn relaxOptimal(c: OptCell, prev: u32, prev_count: u32, len: u32, tid: TokenId) OptCell {
        const cand_count = prev_count + 1;
        if (cand_count < c.count) {
            return .{ .count = cand_count, .prev = prev, .id = tid, .len = len };
        }
        if (cand_count == c.count) {
            // Tie-break: longer token wins; on equal length, lower id.
            if (len > c.len or (len == c.len and tid < c.id)) {
                return .{ .count = cand_count, .prev = prev, .id = tid, .len = len };
            }
        }
        return c;
    }

    /// Optimal (fewest-token) encoder. See the block comment above for
    /// the DP, coverage, and tie-break rules. Allocates one `OptCell`
    /// per byte position (n+1 cells) from `scratch`; stack-resident for
    /// chunks up to `STACK_LIMIT` bytes.
    fn encodeChunkOptimal(
        self: *const Bpe,
        scratch: std.mem.Allocator,
        chunk: []const u8,
        out: []TokenId,
    ) []TokenId {
        const n = chunk.len;
        var stack_cells: [STACK_LIMIT + 1]OptCell = undefined;
        const cells: []OptCell = if (n + 1 <= STACK_LIMIT + 1)
            stack_cells[0 .. n + 1]
        else
            scratch.alloc(OptCell, n + 1) catch unreachable;

        for (cells) |*c| c.* = .{ .count = DP_INF, .prev = 0, .id = std.math.maxInt(TokenId), .len = 0 };
        cells[0].count = 0;

        const bf_table = self.byte_fallback;
        const max_len: usize = self.max_piece_len;

        var i: usize = 0;
        while (i < n) : (i += 1) {
            const base = cells[i].count;
            if (base == DP_INF) continue; // unreachable prefix
            const remaining = n - i;
            const cap = if (remaining < max_len) remaining else max_len;

            var matched_any = false;
            var len: usize = 1;
            while (len <= cap) : (len += 1) {
                const key = chunk[i .. i + len];
                const id: ?TokenId = self.hotLookup(key) orelse self.by_bytes.get(key);
                if (id) |tid| {
                    matched_any = true;
                    const j = i + len;
                    cells[j] = relaxOptimal(cells[j], @intCast(i), base, @intCast(len), tid);
                }
            }

            // Coverage fallback: ensure dp[i+1] is reachable even if no
            // vocab piece matched at this position.
            if (!matched_any) {
                var fid: TokenId = std.math.maxInt(TokenId);
                if (bf_table) |*tbl| {
                    const mapped = tbl[chunk[i]];
                    if (mapped != std.math.maxInt(TokenId)) fid = mapped;
                }
                cells[i + 1] = relaxOptimal(cells[i + 1], @intCast(i), base, 1, fid);
            }
        }

        // Backtrack from n to 0 into a temporary, then reverse into `out`.
        // The number of tokens is `cells[n].count`.
        const total: usize = cells[n].count;
        var pos: usize = n;
        var w: usize = total;
        while (pos > 0) {
            const c = cells[pos];
            w -= 1;
            out[w] = c.id;
            pos = c.prev;
        }
        return out[0..total];
    }

    /// Offsets variant of `encodeChunkOptimal`. Same DP; each emitted id
    /// gets a span covering the bytes it consumed in the original buffer.
    fn encodeChunkOptimalWithOffsets(
        self: *const Bpe,
        scratch: std.mem.Allocator,
        chunk: []const u8,
        chunk_offset: u32,
        out_ids: []TokenId,
        out_offsets: []Span,
    ) usize {
        const n = chunk.len;
        var stack_cells: [STACK_LIMIT + 1]OptCell = undefined;
        const cells: []OptCell = if (n + 1 <= STACK_LIMIT + 1)
            stack_cells[0 .. n + 1]
        else
            scratch.alloc(OptCell, n + 1) catch unreachable;

        for (cells) |*c| c.* = .{ .count = DP_INF, .prev = 0, .id = std.math.maxInt(TokenId), .len = 0 };
        cells[0].count = 0;

        const bf_table = self.byte_fallback;
        const max_len: usize = self.max_piece_len;

        var i: usize = 0;
        while (i < n) : (i += 1) {
            const base = cells[i].count;
            if (base == DP_INF) continue;
            const remaining = n - i;
            const cap = if (remaining < max_len) remaining else max_len;

            var matched_any = false;
            var len: usize = 1;
            while (len <= cap) : (len += 1) {
                const key = chunk[i .. i + len];
                const id: ?TokenId = self.hotLookup(key) orelse self.by_bytes.get(key);
                if (id) |tid| {
                    matched_any = true;
                    const j = i + len;
                    cells[j] = relaxOptimal(cells[j], @intCast(i), base, @intCast(len), tid);
                }
            }
            if (!matched_any) {
                var fid: TokenId = std.math.maxInt(TokenId);
                if (bf_table) |*tbl| {
                    const mapped = tbl[chunk[i]];
                    if (mapped != std.math.maxInt(TokenId)) fid = mapped;
                }
                cells[i + 1] = relaxOptimal(cells[i + 1], @intCast(i), base, 1, fid);
            }
        }

        const total: usize = cells[n].count;
        var pos: usize = n;
        var w: usize = total;
        while (pos > 0) {
            const c = cells[pos];
            w -= 1;
            out_ids[w] = c.id;
            const s: u32 = chunk_offset + c.prev;
            out_offsets[w] = .{ .start = s, .end = s + c.len };
            pos = c.prev;
        }
        return total;
    }

    /// SP-style BPE encoder. Splits the input into UTF-8 codepoint
    /// pieces (each one looked up in `by_bytes`; bytes that aren't a
    /// piece fall back to the byte-fallback table or `maxInt`). Then
    /// runs the canonical BPE merge loop using `piece_ranks[id]` as the
    /// merge priority. SP's training guarantees every intermediate
    /// piece exists in the vocab, so the merge loop navigates from
    /// individual characters all the way up to the longest registered
    /// piece — matching the reference SP encoder's behavior.
    ///
    /// Long inputs (live > HEAP_THRESHOLD codepoints) route to the
    /// 4-ary heap encoder in `bpe_heap.encodeSpBpe`, dropping the
    /// O(live^2) shift cost of the scalar path to O(live log live).
    fn encodeSpBpe(
        self: *const Bpe,
        scratch: std.mem.Allocator,
        chunk: []const u8,
        out: []TokenId,
    ) []TokenId {
        if (chunk.len == 0) return out[0..0];
        const piece_ranks = self.piece_ranks orelse return self.encodeLongestMatch(scratch, chunk, out);
        const bf_table = self.byte_fallback;

        // SoA scratch: one slot per initial codepoint. Worst case is
        // chunk.len single-byte codepoints; allocate that.
        var stack_starts: [STACK_LIMIT]u32 = undefined;
        var stack_lens: [STACK_LIMIT]u32 = undefined;
        var stack_ids: [STACK_LIMIT]TokenId = undefined;
        var stack_ranks: [STACK_LIMIT]u32 = undefined;
        const parts_start: []u32 = if (chunk.len <= STACK_LIMIT)
            stack_starts[0..chunk.len]
        else
            scratch.alloc(u32, chunk.len) catch unreachable;
        const parts_len: []u32 = if (chunk.len <= STACK_LIMIT)
            stack_lens[0..chunk.len]
        else
            scratch.alloc(u32, chunk.len) catch unreachable;
        const part_ids: []TokenId = if (chunk.len <= STACK_LIMIT)
            stack_ids[0..chunk.len]
        else
            scratch.alloc(TokenId, chunk.len) catch unreachable;
        const ranks: []u32 = if (chunk.len <= STACK_LIMIT)
            stack_ranks[0..chunk.len]
        else
            scratch.alloc(u32, chunk.len) catch unreachable;

        // Stage 1: codepoint-level initial segmentation. Each codepoint
        // becomes one part. Look up by_bytes for the whole codepoint; if
        // it's not a piece (rare — SP usually has every base char as a
        // piece) consult byte_fallback for the leading byte and advance
        // one byte.
        var live: u32 = 0;
        var pos: usize = 0;
        while (pos < chunk.len) {
            const cpl = cpLen(chunk[pos]);
            const end = if (pos + cpl > chunk.len) chunk.len else pos + cpl;
            const key = chunk[pos..end];
            var id: TokenId = std.math.maxInt(TokenId);
            if (self.by_bytes.get(key)) |hit| {
                id = hit;
                parts_start[live] = @intCast(pos);
                parts_len[live] = @intCast(end - pos);
                pos = end;
            } else {
                // Codepoint not in vocab as a piece. Use byte-fallback
                // for one byte at a time so multi-byte non-piece chars
                // still get representable ids.
                if (bf_table) |*tbl| {
                    const mapped = tbl[chunk[pos]];
                    if (mapped != std.math.maxInt(TokenId)) id = mapped;
                }
                parts_start[live] = @intCast(pos);
                parts_len[live] = 1;
                pos += 1;
            }
            part_ids[live] = id;
            live += 1;
        }

        // Stage 2: pair ranks. ranks[i] is the merge rank of pair
        // (i, i+1). RANK_INVALID for the trailing slot and for pairs
        // whose concatenation isn't a piece.
        if (live >= 2) {
            var i: u32 = 0;
            while (i + 1 < live) : (i += 1) {
                const s = parts_start[i];
                const total = parts_len[i] + parts_len[i + 1];
                const key = chunk[s .. s + total];
                if (self.by_bytes.get(key)) |id| {
                    ranks[i] = piece_ranks[id];
                } else {
                    ranks[i] = RANK_INVALID;
                }
            }
        }
        if (live > 0) ranks[live - 1] = RANK_INVALID;

        // Long-chunk path: 4-ary heap encoder. Same algorithm, but pair
        // selection is O(log live) per merge instead of O(live), and
        // splicing is a linked-list pointer update instead of an
        // O(live) shift. Engages at the same `HEAP_THRESHOLD` the
        // byte-level path uses, just in codepoint units.
        const HEAP_THRESHOLD: u32 = 64;
        if (live > HEAP_THRESHOLD) {
            const heap_path = @import("bpe_heap.zig");
            const nn: usize = live;
            const prev = scratch.alloc(u32, nn) catch unreachable;
            const nxt = scratch.alloc(u32, nn) catch unreachable;
            const heap_buf = scratch.alloc(heap_path.HeapNode, 3 * nn) catch unreachable;
            for (0..nn) |i| {
                prev[i] = if (i == 0) heap_path.NONE else @intCast(i - 1);
                nxt[i] = if (i + 1 == nn) heap_path.NONE else @intCast(i + 1);
            }
            const written = heap_path.encodeSpBpe(
                chunk,
                nn,
                parts_start,
                parts_len,
                part_ids,
                ranks,
                prev,
                nxt,
                &self.by_bytes,
                piece_ranks,
                heap_buf,
                out,
            );
            return out[0..written];
        }

        // Stage 3: BPE merge loop. Same shape as the byte-level path —
        // scan for min, splice, refresh neighbour ranks.
        while (live >= 2) {
            const scan = scanMin(ranks[0 .. live - 1]);
            if (scan.rank == RANK_INVALID) break;
            const mi = scan.idx;

            // Resolve the merged piece's id from the concatenated bytes.
            const ms = parts_start[mi];
            const merged_total = parts_len[mi] + parts_len[mi + 1];
            const merged_key = chunk[ms .. ms + merged_total];
            const merged_id = self.by_bytes.get(merged_key) orelse unreachable;
            part_ids[mi] = merged_id;
            parts_len[mi] = merged_total;

            // Shift tail.
            var j: u32 = mi + 1;
            while (j + 1 < live) : (j += 1) {
                parts_start[j] = parts_start[j + 1];
                parts_len[j] = parts_len[j + 1];
                part_ids[j] = part_ids[j + 1];
                ranks[j] = ranks[j + 1];
            }
            live -= 1;

            // Refresh neighbour ranks. Two hashmap ops at most.
            if (mi > 0) {
                const li = mi - 1;
                const s = parts_start[li];
                const total = parts_len[li] + parts_len[mi];
                const key = chunk[s .. s + total];
                ranks[li] = if (self.by_bytes.get(key)) |id| piece_ranks[id] else RANK_INVALID;
            }
            if (mi + 1 < live) {
                const s = parts_start[mi];
                const total = parts_len[mi] + parts_len[mi + 1];
                const key = chunk[s .. s + total];
                ranks[mi] = if (self.by_bytes.get(key)) |id| piece_ranks[id] else RANK_INVALID;
            } else {
                ranks[mi] = RANK_INVALID;
            }
        }

        // Stage 4: emit ids.
        var w: usize = 0;
        var k: u32 = 0;
        while (k < live) : (k += 1) {
            out[w] = part_ids[k];
            w += 1;
        }
        return out[0..w];
    }

    /// Offsets variant of `encodeSpBpe`.
    fn encodeSpBpeWithOffsets(
        self: *const Bpe,
        scratch: std.mem.Allocator,
        chunk: []const u8,
        chunk_offset: u32,
        out_ids: []TokenId,
        out_offsets: []Span,
    ) usize {
        if (chunk.len == 0) return 0;
        const piece_ranks = self.piece_ranks orelse
            return self.encodeLongestMatchWithOffsets(scratch, chunk, chunk_offset, out_ids, out_offsets);
        const bf_table = self.byte_fallback;

        var stack_starts: [STACK_LIMIT]u32 = undefined;
        var stack_lens: [STACK_LIMIT]u32 = undefined;
        var stack_ids: [STACK_LIMIT]TokenId = undefined;
        var stack_ranks: [STACK_LIMIT]u32 = undefined;
        const parts_start: []u32 = if (chunk.len <= STACK_LIMIT)
            stack_starts[0..chunk.len]
        else
            scratch.alloc(u32, chunk.len) catch unreachable;
        const parts_len: []u32 = if (chunk.len <= STACK_LIMIT)
            stack_lens[0..chunk.len]
        else
            scratch.alloc(u32, chunk.len) catch unreachable;
        const part_ids: []TokenId = if (chunk.len <= STACK_LIMIT)
            stack_ids[0..chunk.len]
        else
            scratch.alloc(TokenId, chunk.len) catch unreachable;
        const ranks: []u32 = if (chunk.len <= STACK_LIMIT)
            stack_ranks[0..chunk.len]
        else
            scratch.alloc(u32, chunk.len) catch unreachable;

        var live: u32 = 0;
        var pos: usize = 0;
        while (pos < chunk.len) {
            const cpl = cpLen(chunk[pos]);
            const end = if (pos + cpl > chunk.len) chunk.len else pos + cpl;
            const key = chunk[pos..end];
            var id: TokenId = std.math.maxInt(TokenId);
            if (self.by_bytes.get(key)) |hit| {
                id = hit;
                parts_start[live] = @intCast(pos);
                parts_len[live] = @intCast(end - pos);
                pos = end;
            } else {
                if (bf_table) |*tbl| {
                    const mapped = tbl[chunk[pos]];
                    if (mapped != std.math.maxInt(TokenId)) id = mapped;
                }
                parts_start[live] = @intCast(pos);
                parts_len[live] = 1;
                pos += 1;
            }
            part_ids[live] = id;
            live += 1;
        }

        if (live >= 2) {
            var i: u32 = 0;
            while (i + 1 < live) : (i += 1) {
                const s = parts_start[i];
                const total = parts_len[i] + parts_len[i + 1];
                const key = chunk[s .. s + total];
                if (self.by_bytes.get(key)) |id| {
                    ranks[i] = piece_ranks[id];
                } else {
                    ranks[i] = RANK_INVALID;
                }
            }
        }
        if (live > 0) ranks[live - 1] = RANK_INVALID;

        while (live >= 2) {
            const scan = scanMin(ranks[0 .. live - 1]);
            if (scan.rank == RANK_INVALID) break;
            const mi = scan.idx;

            const ms = parts_start[mi];
            const merged_total = parts_len[mi] + parts_len[mi + 1];
            const merged_key = chunk[ms .. ms + merged_total];
            const merged_id = self.by_bytes.get(merged_key) orelse unreachable;
            part_ids[mi] = merged_id;
            parts_len[mi] = merged_total;

            var j: u32 = mi + 1;
            while (j + 1 < live) : (j += 1) {
                parts_start[j] = parts_start[j + 1];
                parts_len[j] = parts_len[j + 1];
                part_ids[j] = part_ids[j + 1];
                ranks[j] = ranks[j + 1];
            }
            live -= 1;

            if (mi > 0) {
                const li = mi - 1;
                const s = parts_start[li];
                const total = parts_len[li] + parts_len[mi];
                const key = chunk[s .. s + total];
                ranks[li] = if (self.by_bytes.get(key)) |id| piece_ranks[id] else RANK_INVALID;
            }
            if (mi + 1 < live) {
                const s = parts_start[mi];
                const total = parts_len[mi] + parts_len[mi + 1];
                const key = chunk[s .. s + total];
                ranks[mi] = if (self.by_bytes.get(key)) |id| piece_ranks[id] else RANK_INVALID;
            } else {
                ranks[mi] = RANK_INVALID;
            }
        }

        var w: usize = 0;
        var k: u32 = 0;
        while (k < live) : (k += 1) {
            out_ids[w] = part_ids[k];
            const sp_start = chunk_offset + parts_start[k];
            out_offsets[w] = .{ .start = sp_start, .end = sp_start + parts_len[k] };
            w += 1;
        }
        return w;
    }

    /// Same as `encodeChunk` but also writes byte ranges (relative to the
    /// buffer the offsets index into) into `out_offsets`. `chunk_offset` is
    /// the position of `chunk[0]` in that buffer. Always takes the scalar
    /// SoA path; long chunks pay an O(live) shift per merge.
    ///
    /// Convenience wrapper for `encodeChunkWithOffsetsScratch` that pulls
    /// heap spillover from `self.allocator`.
    pub fn encodeChunkWithOffsets(
        self: *const Bpe,
        chunk: []const u8,
        chunk_offset: u32,
        out_ids: []TokenId,
        out_offsets: []Span,
    ) usize {
        return self.encodeChunkWithOffsetsScratch(
            self.allocator,
            chunk,
            chunk_offset,
            out_ids,
            out_offsets,
        );
    }

    /// Same as `encodeChunkWithOffsets` but heap-spillover scratch (for
    /// chunks > `STACK_LIMIT` bytes) comes from `scratch` instead of
    /// `self.allocator`. Pass a per-thread `ArenaAllocator` for
    /// zero-alloc behavior after warm-up.
    pub fn encodeChunkWithOffsetsScratch(
        self: *const Bpe,
        scratch: std.mem.Allocator,
        chunk: []const u8,
        chunk_offset: u32,
        out_ids: []TokenId,
        out_offsets: []Span,
    ) usize {
        if (chunk.len == 0) return 0;
        std.debug.assert(out_ids.len >= chunk.len);
        std.debug.assert(out_offsets.len >= chunk.len);

        // HF `ignore_merges` whole-chunk short-circuit. See the
        // matching block in `encodeChunkScratch` for why.
        if (self.ignore_merges) {
            if (self.hotLookup(chunk)) |id| {
                out_ids[0] = id;
                out_offsets[0] = .{ .start = chunk_offset, .end = chunk_offset + @as(u32, @intCast(chunk.len)) };
                return 1;
            }
            if (self.by_bytes.get(chunk)) |id| {
                out_ids[0] = id;
                out_offsets[0] = .{ .start = chunk_offset, .end = chunk_offset + @as(u32, @intCast(chunk.len)) };
                return 1;
            }
        }

        if (self.encode_mode == .longest_match) {
            return self.encodeLongestMatchWithOffsets(scratch, chunk, chunk_offset, out_ids, out_offsets);
        }

        if (self.encode_mode == .optimal) {
            return self.encodeChunkOptimalWithOffsets(scratch, chunk, chunk_offset, out_ids, out_offsets);
        }

        var stack_starts: [STACK_LIMIT]u32 = undefined;
        var stack_lens: [STACK_LIMIT]u32 = undefined;
        var stack_ranks: [STACK_LIMIT]u32 = undefined;

        const parts_start: []u32 = if (chunk.len <= STACK_LIMIT)
            stack_starts[0..chunk.len]
        else
            scratch.alloc(u32, chunk.len) catch unreachable;
        const parts_len: []u32 = if (chunk.len <= STACK_LIMIT)
            stack_lens[0..chunk.len]
        else
            scratch.alloc(u32, chunk.len) catch unreachable;
        const ranks: []u32 = if (chunk.len <= STACK_LIMIT)
            stack_ranks[0..chunk.len]
        else
            scratch.alloc(u32, chunk.len) catch unreachable;

        var live: u32 = @intCast(chunk.len);
        for (0..chunk.len) |i| {
            parts_start[i] = @intCast(i);
            parts_len[i] = 1;
        }

        // Same hot-table-first lookup as `encodeChunkScratch`. Two-level
        // BPE merge-rank layout — see `HotEntry` docs at top of file.
        if (live >= 2) {
            var i: u32 = 0;
            while (i + 1 < live) : (i += 1) {
                const start = parts_start[i];
                const total = parts_len[i] + parts_len[i + 1];
                const key = chunk[start .. start + total];
                if (self.hotLookup(key)) |r| {
                    ranks[i] = r;
                } else if (self.by_bytes.get(key)) |r| {
                    ranks[i] = r;
                } else {
                    ranks[i] = RANK_INVALID;
                }
            }
        }
        if (live > 0) ranks[live - 1] = RANK_INVALID;

        while (live >= 2) {
            const scan = scanMin(ranks[0 .. live - 1]);
            if (scan.rank == RANK_INVALID) break;
            const mi = scan.idx;

            parts_len[mi] += parts_len[mi + 1];

            var j: u32 = mi + 1;
            while (j + 1 < live) : (j += 1) {
                parts_start[j] = parts_start[j + 1];
                parts_len[j] = parts_len[j + 1];
                ranks[j] = ranks[j + 1];
            }
            live -= 1;

            if (mi > 0) {
                const li = mi - 1;
                const start = parts_start[li];
                const total = parts_len[li] + parts_len[mi];
                const key = chunk[start .. start + total];
                if (self.hotLookup(key)) |r| {
                    ranks[li] = r;
                } else if (self.by_bytes.get(key)) |r| {
                    ranks[li] = r;
                } else {
                    ranks[li] = RANK_INVALID;
                }
            }
            if (mi + 1 < live) {
                const start = parts_start[mi];
                const total = parts_len[mi] + parts_len[mi + 1];
                const key = chunk[start .. start + total];
                if (self.hotLookup(key)) |r| {
                    ranks[mi] = r;
                } else if (self.by_bytes.get(key)) |r| {
                    ranks[mi] = r;
                } else {
                    ranks[mi] = RANK_INVALID;
                }
            } else {
                ranks[mi] = RANK_INVALID;
            }
        }

        const bf_table = self.byte_fallback;
        var w: usize = 0;
        var k: u32 = 0;
        while (k < live) : (k += 1) {
            const start = parts_start[k];
            const len = parts_len[k];
            const key = chunk[start .. start + len];
            var id: TokenId = if (self.hotLookup(key)) |r|
                r
            else
                self.by_bytes.get(key) orelse std.math.maxInt(TokenId);
            if (len == 1) {
                if (bf_table) |*tbl| {
                    const mapped = tbl[chunk[start]];
                    if (mapped != std.math.maxInt(TokenId)) id = mapped;
                }
            }
            out_ids[w] = id;
            out_offsets[w] = .{
                .start = chunk_offset + start,
                .end = chunk_offset + start + len,
            };
            w += 1;
        }
        return w;
    }

    // -------- trace variants ---------------------------------------------
    //
    // These mirror `encodeChunkScratch` / `encodeChunkWithOffsetsScratch`
    // but emit one `Trace.merge` record per merge decision (with the
    // pre-merge byte payloads of both halves). They always take the
    // scalar SoA path: the heap-spillover bpe_heap path is skipped so
    // the trace stream stays linear and easy to follow. Tracing is not
    // a hot path; correctness + readability trump worst-case speed.

    /// Same shape as `encodeChunkScratch` plus a non-null `trace` sink.
    /// Each merge step emits `bpe merge pos=… rank=… left=… right=…`
    /// to the trace writer. Returns an error union because the trace
    /// writer can fail.
    pub fn encodeChunkScratchTrace(
        self: *const Bpe,
        scratch: std.mem.Allocator,
        chunk: []const u8,
        out: []TokenId,
        trace: *@import("trace.zig").Trace,
    ) ![]TokenId {
        if (chunk.len == 0) return out[0..0];
        std.debug.assert(out.len >= chunk.len);

        // HF `ignore_merges` whole-chunk short-circuit — same as
        // `encodeChunkScratch`. Trace a single pseudo-record so the
        // output is self-documenting.
        if (self.ignore_merges) {
            if (self.hotLookup(chunk) orelse self.by_bytes.get(chunk)) |id| {
                try trace.pieces("bpe (ignore_merges hit, no merge) chunk", &.{chunk});
                out[0] = id;
                return out[0..1];
            }
        }

        // Longest-match / SP-BPE paths don't expose a merge sequence —
        // they pick whole pieces greedily. Fall back to the non-trace
        // path on those so we never break behaviour, and emit a single
        // pseudo-record telling the user we skipped tracing here.
        if (self.encode_mode == .longest_match) {
            try trace.pieces("bpe (longest_match, no merge trace) input", &.{chunk});
            return self.encodeLongestMatch(scratch, chunk, out);
        }
        if (self.encode_mode == .optimal) {
            try trace.pieces("bpe (optimal DP, no merge trace) input", &.{chunk});
            return self.encodeChunkOptimal(scratch, chunk, out);
        }

        var stack_starts: [STACK_LIMIT]u32 = undefined;
        var stack_lens: [STACK_LIMIT]u32 = undefined;
        var stack_ranks: [STACK_LIMIT]u32 = undefined;

        const parts_start: []u32 = if (chunk.len <= STACK_LIMIT)
            stack_starts[0..chunk.len]
        else
            scratch.alloc(u32, chunk.len) catch unreachable;
        const parts_len: []u32 = if (chunk.len <= STACK_LIMIT)
            stack_lens[0..chunk.len]
        else
            scratch.alloc(u32, chunk.len) catch unreachable;
        const ranks: []u32 = if (chunk.len <= STACK_LIMIT)
            stack_ranks[0..chunk.len]
        else
            scratch.alloc(u32, chunk.len) catch unreachable;

        var live: u32 = @intCast(chunk.len);
        for (0..chunk.len) |i| {
            parts_start[i] = @intCast(i);
            parts_len[i] = 1;
        }

        if (live >= 2) {
            var i: u32 = 0;
            while (i + 1 < live) : (i += 1) {
                const start = parts_start[i];
                const total = parts_len[i] + parts_len[i + 1];
                const key = chunk[start .. start + total];
                if (self.hotLookup(key)) |r| {
                    ranks[i] = r;
                } else if (self.by_bytes.get(key)) |r| {
                    ranks[i] = r;
                } else {
                    ranks[i] = RANK_INVALID;
                }
            }
        }
        if (live > 0) ranks[live - 1] = RANK_INVALID;

        while (live >= 2) {
            const scan = scanMin(ranks[0 .. live - 1]);
            if (scan.rank == RANK_INVALID) break;
            const mi = scan.idx;
            // Trace: capture the two pre-merge byte payloads before we
            // splice. Cheap — both are direct slices into `chunk`.
            const left_start = parts_start[mi];
            const left_len = parts_len[mi];
            const right_start = parts_start[mi + 1];
            const right_len = parts_len[mi + 1];
            try trace.merge(
                mi,
                scan.rank,
                chunk[left_start .. left_start + left_len],
                chunk[right_start .. right_start + right_len],
            );

            parts_len[mi] += parts_len[mi + 1];

            var j: u32 = mi + 1;
            while (j + 1 < live) : (j += 1) {
                parts_start[j] = parts_start[j + 1];
                parts_len[j] = parts_len[j + 1];
                ranks[j] = ranks[j + 1];
            }
            live -= 1;

            if (mi > 0) {
                const li = mi - 1;
                const s2 = parts_start[li];
                const total = parts_len[li] + parts_len[mi];
                const key = chunk[s2 .. s2 + total];
                if (self.hotLookup(key)) |r| {
                    ranks[li] = r;
                } else if (self.by_bytes.get(key)) |r| {
                    ranks[li] = r;
                } else {
                    ranks[li] = RANK_INVALID;
                }
            }
            if (mi + 1 < live) {
                const s2 = parts_start[mi];
                const total = parts_len[mi] + parts_len[mi + 1];
                const key = chunk[s2 .. s2 + total];
                if (self.hotLookup(key)) |r| {
                    ranks[mi] = r;
                } else if (self.by_bytes.get(key)) |r| {
                    ranks[mi] = r;
                } else {
                    ranks[mi] = RANK_INVALID;
                }
            } else {
                ranks[mi] = RANK_INVALID;
            }
        }

        const bf_table = self.byte_fallback;
        var w: usize = 0;
        var k: u32 = 0;
        while (k < live) : (k += 1) {
            const start = parts_start[k];
            const len = parts_len[k];
            const key = chunk[start .. start + len];
            var id: TokenId = if (self.hotLookup(key)) |r|
                r
            else
                self.by_bytes.get(key) orelse std.math.maxInt(TokenId);
            if (len == 1) {
                if (bf_table) |*tbl| {
                    const mapped = tbl[chunk[start]];
                    if (mapped != std.math.maxInt(TokenId)) id = mapped;
                }
            }
            out[w] = id;
            w += 1;
        }
        return out[0..w];
    }

    /// Offsets-emitting trace variant. Same trace records as
    /// `encodeChunkScratchTrace`; emits one Span per final id.
    pub fn encodeChunkWithOffsetsScratchTrace(
        self: *const Bpe,
        scratch: std.mem.Allocator,
        chunk: []const u8,
        chunk_offset: u32,
        out_ids: []TokenId,
        out_offsets: []Span,
        trace: *@import("trace.zig").Trace,
    ) !usize {
        if (chunk.len == 0) return 0;
        std.debug.assert(out_ids.len >= chunk.len);
        std.debug.assert(out_offsets.len >= chunk.len);

        // HF `ignore_merges` whole-chunk short-circuit.
        if (self.ignore_merges) {
            if (self.hotLookup(chunk) orelse self.by_bytes.get(chunk)) |id| {
                try trace.pieces("bpe (ignore_merges hit, no merge) chunk", &.{chunk});
                out_ids[0] = id;
                out_offsets[0] = .{ .start = chunk_offset, .end = chunk_offset + @as(u32, @intCast(chunk.len)) };
                return 1;
            }
        }

        if (self.encode_mode == .longest_match) {
            try trace.pieces("bpe (longest_match, no merge trace) input", &.{chunk});
            return self.encodeLongestMatchWithOffsets(scratch, chunk, chunk_offset, out_ids, out_offsets);
        }
        if (self.encode_mode == .optimal) {
            try trace.pieces("bpe (optimal DP, no merge trace) input", &.{chunk});
            return self.encodeChunkOptimalWithOffsets(scratch, chunk, chunk_offset, out_ids, out_offsets);
        }

        var stack_starts: [STACK_LIMIT]u32 = undefined;
        var stack_lens: [STACK_LIMIT]u32 = undefined;
        var stack_ranks: [STACK_LIMIT]u32 = undefined;

        const parts_start: []u32 = if (chunk.len <= STACK_LIMIT)
            stack_starts[0..chunk.len]
        else
            scratch.alloc(u32, chunk.len) catch unreachable;
        const parts_len: []u32 = if (chunk.len <= STACK_LIMIT)
            stack_lens[0..chunk.len]
        else
            scratch.alloc(u32, chunk.len) catch unreachable;
        const ranks: []u32 = if (chunk.len <= STACK_LIMIT)
            stack_ranks[0..chunk.len]
        else
            scratch.alloc(u32, chunk.len) catch unreachable;

        var live: u32 = @intCast(chunk.len);
        for (0..chunk.len) |i| {
            parts_start[i] = @intCast(i);
            parts_len[i] = 1;
        }

        if (live >= 2) {
            var i: u32 = 0;
            while (i + 1 < live) : (i += 1) {
                const start = parts_start[i];
                const total = parts_len[i] + parts_len[i + 1];
                const key = chunk[start .. start + total];
                if (self.hotLookup(key)) |r| {
                    ranks[i] = r;
                } else if (self.by_bytes.get(key)) |r| {
                    ranks[i] = r;
                } else {
                    ranks[i] = RANK_INVALID;
                }
            }
        }
        if (live > 0) ranks[live - 1] = RANK_INVALID;

        while (live >= 2) {
            const scan = scanMin(ranks[0 .. live - 1]);
            if (scan.rank == RANK_INVALID) break;
            const mi = scan.idx;
            const left_start = parts_start[mi];
            const left_len = parts_len[mi];
            const right_start = parts_start[mi + 1];
            const right_len = parts_len[mi + 1];
            try trace.merge(
                mi,
                scan.rank,
                chunk[left_start .. left_start + left_len],
                chunk[right_start .. right_start + right_len],
            );

            parts_len[mi] += parts_len[mi + 1];

            var j: u32 = mi + 1;
            while (j + 1 < live) : (j += 1) {
                parts_start[j] = parts_start[j + 1];
                parts_len[j] = parts_len[j + 1];
                ranks[j] = ranks[j + 1];
            }
            live -= 1;

            if (mi > 0) {
                const li = mi - 1;
                const s2 = parts_start[li];
                const total = parts_len[li] + parts_len[mi];
                const key = chunk[s2 .. s2 + total];
                if (self.hotLookup(key)) |r| {
                    ranks[li] = r;
                } else if (self.by_bytes.get(key)) |r| {
                    ranks[li] = r;
                } else {
                    ranks[li] = RANK_INVALID;
                }
            }
            if (mi + 1 < live) {
                const s2 = parts_start[mi];
                const total = parts_len[mi] + parts_len[mi + 1];
                const key = chunk[s2 .. s2 + total];
                if (self.hotLookup(key)) |r| {
                    ranks[mi] = r;
                } else if (self.by_bytes.get(key)) |r| {
                    ranks[mi] = r;
                } else {
                    ranks[mi] = RANK_INVALID;
                }
            } else {
                ranks[mi] = RANK_INVALID;
            }
        }

        const bf_table = self.byte_fallback;
        var w: usize = 0;
        var k: u32 = 0;
        while (k < live) : (k += 1) {
            const start = parts_start[k];
            const len = parts_len[k];
            const key = chunk[start .. start + len];
            var id: TokenId = if (self.hotLookup(key)) |r|
                r
            else
                self.by_bytes.get(key) orelse std.math.maxInt(TokenId);
            if (len == 1) {
                if (bf_table) |*tbl| {
                    const mapped = tbl[chunk[start]];
                    if (mapped != std.math.maxInt(TokenId)) id = mapped;
                }
            }
            out_ids[w] = id;
            out_offsets[w] = .{
                .start = chunk_offset + start,
                .end = chunk_offset + start + len,
            };
            w += 1;
        }
        return w;
    }
};

// --- tests -----------------------------------------------------------

const testing = std.testing;

const TestEntry = struct { bytes: []const u8, rank: u32 };

// Build a vocab string from (bytes, rank) pairs. Runtime; not comptime.
fn buildVocabSource(
    allocator: std.mem.Allocator,
    entries: []const TestEntry,
) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    const enc = std.base64.standard.Encoder;
    for (entries) |e| {
        const sz = enc.calcSize(e.bytes.len);
        const tmp = try allocator.alloc(u8, sz);
        defer allocator.free(tmp);
        const encoded = enc.encode(tmp, e.bytes);
        try buf.appendSlice(allocator, encoded);
        try buf.print(allocator, " {d}\n", .{e.rank});
    }
    return buf.toOwnedSlice(allocator);
}

test "loadTiktokenBytes parses simple vocab" {
    const src = try buildVocabSource(testing.allocator, &.{
        .{ .bytes = "a", .rank = 0 },
        .{ .bytes = "b", .rank = 1 },
        .{ .bytes = "ab", .rank = 2 },
        .{ .bytes = "bc", .rank = 3 },
    });
    defer testing.allocator.free(src);

    var bpe = try Bpe.loadTiktokenBytes(testing.allocator, src);
    defer bpe.deinit();

    try testing.expectEqual(@as(u32, 4), bpe.count);
    try testing.expectEqualStrings("a", bpe.idBytes(0));
    try testing.expectEqualStrings("b", bpe.idBytes(1));
    try testing.expectEqualStrings("ab", bpe.idBytes(2));
    try testing.expectEqualStrings("bc", bpe.idBytes(3));
    try testing.expectEqual(@as(?TokenId, 0), bpe.by_bytes.get("a"));
    try testing.expectEqual(@as(?TokenId, 3), bpe.by_bytes.get("bc"));
}

// Build a vocab covering all 256 single-byte tokens, then any extras
// (which must have rank >= 256 and stay contiguous).
//
// Forces `hot_table = true` so the dedicated "hot table" tests below
// have a populated front cache to introspect. The bare-default policy
// (off) is exercised separately by tests that call
// `loadTiktokenBytes` directly.
fn buildByteVocab(
    allocator: std.mem.Allocator,
    extras: []const TestEntry,
) !Bpe {
    var entries: std.ArrayList(TestEntry) = .empty;
    defer entries.deinit(allocator);

    // Single-byte tokens 0..255. We allocate one byte per token so the
    // pointer stays valid for the duration of buildVocabSource.
    var byte_holders: [256][1]u8 = undefined;
    var i: u32 = 0;
    while (i < 256) : (i += 1) {
        byte_holders[i][0] = @intCast(i);
        try entries.append(allocator, .{ .bytes = byte_holders[i][0..1], .rank = i });
    }
    for (extras) |e| try entries.append(allocator, e);

    const src = try buildVocabSource(allocator, entries.items);
    defer allocator.free(src);
    return Bpe.loadTiktokenBytesWithOptions(allocator, src, .{ .hot_table = true });
}

test "encodeChunk single byte" {
    var bpe = try buildByteVocab(testing.allocator, &.{});
    defer bpe.deinit();

    var out: [4]TokenId = undefined;
    const ids = bpe.encodeChunk("a", &out);
    try testing.expectEqualSlices(TokenId, &.{0x61}, ids);
}

test "encodeChunk merges adjacent pair" {
    // {a:0, b:1, ab:2}.
    const src = try buildVocabSource(testing.allocator, &.{
        .{ .bytes = "a", .rank = 0 },
        .{ .bytes = "b", .rank = 1 },
        .{ .bytes = "ab", .rank = 2 },
    });
    defer testing.allocator.free(src);
    var bpe = try Bpe.loadTiktokenBytes(testing.allocator, src);
    defer bpe.deinit();

    var out: [4]TokenId = undefined;
    const ids = bpe.encodeChunk("ab", &out);
    try testing.expectEqualSlices(TokenId, &.{2}, ids);
}

test "encodeChunk picks lowest rank pair" {
    // {a:0, b:1, c:2, bc:3, ab:5}. bc has lower rank than ab so the
    // first merge is bc -> "abc" becomes [a, bc] = [0, 3].
    // (Ranks 0,1,2,3,5 are non-contiguous; we need 0..4. Use ab:4.)
    // Adjusting to keep dense rank assignment while preserving the
    // ordering bc < ab: bc:3, ab:4.
    const src = try buildVocabSource(testing.allocator, &.{
        .{ .bytes = "a", .rank = 0 },
        .{ .bytes = "b", .rank = 1 },
        .{ .bytes = "c", .rank = 2 },
        .{ .bytes = "bc", .rank = 3 },
        .{ .bytes = "ab", .rank = 4 },
    });
    defer testing.allocator.free(src);
    var bpe = try Bpe.loadTiktokenBytes(testing.allocator, src);
    defer bpe.deinit();

    var out: [4]TokenId = undefined;
    const ids = bpe.encodeChunk("abc", &out);
    try testing.expectEqualSlices(TokenId, &.{ 0, 3 }, ids);
}

test "encodeChunk leaves unmergeable bytes as singles" {
    const src = try buildVocabSource(testing.allocator, &.{
        .{ .bytes = "a", .rank = 0 },
        .{ .bytes = "b", .rank = 1 },
    });
    defer testing.allocator.free(src);
    var bpe = try Bpe.loadTiktokenBytes(testing.allocator, src);
    defer bpe.deinit();

    var out: [4]TokenId = undefined;
    const ids = bpe.encodeChunk("ab", &out);
    try testing.expectEqualSlices(TokenId, &.{ 0, 1 }, ids);
}

// --- optimal (minimum-token DP) -------------------------------------

test "optimal beats greedy on a strict-win vocab" {
    // Vocab {a,b,c,d, "ab", "bcd"} over input "abcd".
    //   .bpe_merge: bytes a,b,c,d -> only "ab" merges -> [ab,c,d] = 3.
    //   .optimal:   [a, bcd]                                      = 2.
    // The DP must find the 2-token segmentation that no greedy /
    // merge-order walk reaches here.
    var bpe = try buildByteVocab(testing.allocator, &.{
        .{ .bytes = "ab", .rank = 256 },
        .{ .bytes = "bcd", .rank = 257 },
    });
    defer bpe.deinit();

    var out: [8]TokenId = undefined;

    bpe.encode_mode = .bpe_merge;
    const greedy = bpe.encodeChunk("abcd", &out);
    try testing.expectEqual(@as(usize, 3), greedy.len);

    var out2: [8]TokenId = undefined;
    bpe.encode_mode = .optimal;
    const opt = bpe.encodeChunk("abcd", &out2);
    try testing.expectEqual(@as(usize, 2), opt.len);
    try testing.expect(opt.len < greedy.len);
    // [a, bcd] — a is byte token 0x61, "bcd" is the extra at rank 257.
    try testing.expectEqualSlices(TokenId, &.{ 0x61, 257 }, opt);
}

test "optimal decodes back to exact input (lossless)" {
    var bpe = try buildByteVocab(testing.allocator, &.{
        .{ .bytes = "ab", .rank = 256 },
        .{ .bytes = "bcd", .rank = 257 },
        .{ .bytes = "abcd", .rank = 258 },
    });
    defer bpe.deinit();
    bpe.encode_mode = .optimal;

    const inputs = [_][]const u8{ "abcd", "abcabcd", "dddabcd", "a", "zzz" };
    for (inputs) |inp| {
        var out: [64]TokenId = undefined;
        const ids = bpe.encodeChunk(inp, &out);
        // Reconstruct by concatenating each token's bytes.
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(testing.allocator);
        for (ids) |id| {
            try testing.expect(id < bpe.count); // no unk sentinel: byte vocab covers all 256
            try buf.appendSlice(testing.allocator, bpe.idBytes(id));
        }
        try testing.expectEqualStrings(inp, buf.items);
    }
}

test "optimal token count is <= greedy across inputs" {
    var bpe = try buildByteVocab(testing.allocator, &.{
        .{ .bytes = "ab", .rank = 256 },
        .{ .bytes = "bcd", .rank = 257 },
        .{ .bytes = "cd", .rank = 258 },
        .{ .bytes = "abcd", .rank = 259 },
    });
    defer bpe.deinit();

    const inputs = [_][]const u8{ "abcd", "abcabcd", "dabcd", "abab", "cdcd", "" };
    for (inputs) |inp| {
        var g: [64]TokenId = undefined;
        bpe.encode_mode = .bpe_merge;
        const greedy = bpe.encodeChunk(inp, &g);

        var o: [64]TokenId = undefined;
        bpe.encode_mode = .optimal;
        const opt = bpe.encodeChunk(inp, &o);

        try testing.expect(opt.len <= greedy.len);
    }
}

test "optimal is deterministic and respects the tie-break" {
    // Tie-break: prefer longer token, then lower id. Input "aa" with
    // {a, "aa"} -> both [a,a] (2 tokens) and [aa] (1) exist; minimal is
    // [aa]. Construct a genuine equal-count tie: input "ab" with
    // tokens {a, b, "ab"(longer, id 257), and a synthetic same-length
    // alternative is impossible, so test longer-token preference and
    // determinism across repeated runs.
    var bpe = try buildByteVocab(testing.allocator, &.{
        .{ .bytes = "ab", .rank = 256 },
    });
    defer bpe.deinit();
    bpe.encode_mode = .optimal;

    var prev: [8]TokenId = undefined;
    var prev_len: usize = 0;
    var run: usize = 0;
    while (run < 5) : (run += 1) {
        var out: [8]TokenId = undefined;
        const ids = bpe.encodeChunk("ab", &out);
        // Longer token preferred: [ab] (id 256), not [a, b].
        try testing.expectEqualSlices(TokenId, &.{256}, ids);
        if (run > 0) {
            try testing.expectEqualSlices(TokenId, prev[0..prev_len], ids);
        }
        @memcpy(prev[0..ids.len], ids);
        prev_len = ids.len;
    }
}

test "optimal edge cases: empty and single byte" {
    var bpe = try buildByteVocab(testing.allocator, &.{
        .{ .bytes = "ab", .rank = 256 },
    });
    defer bpe.deinit();
    bpe.encode_mode = .optimal;

    var out: [4]TokenId = undefined;
    const empty = bpe.encodeChunk("", &out);
    try testing.expectEqual(@as(usize, 0), empty.len);

    const single = bpe.encodeChunk("a", &out);
    try testing.expectEqualSlices(TokenId, &.{0x61}, single);
}

test "optimal with offsets covers the input contiguously" {
    var bpe = try buildByteVocab(testing.allocator, &.{
        .{ .bytes = "ab", .rank = 256 },
        .{ .bytes = "bcd", .rank = 257 },
    });
    defer bpe.deinit();
    bpe.encode_mode = .optimal;

    var ids: [16]TokenId = undefined;
    var offs: [16]Span = undefined;
    const n = bpe.encodeChunkWithOffsets("abcd", 100, &ids, &offs);
    try testing.expectEqual(@as(usize, 2), n);
    // Spans must tile [100, 104) with no gaps/overlaps, in order.
    try testing.expectEqual(@as(u32, 100), offs[0].start);
    try testing.expectEqual(offs[0].end, offs[1].start);
    try testing.expectEqual(@as(u32, 104), offs[1].end);
}

test "encodeChunk long repeated chunk merges deeply" {
    // Vocab: 256 byte tokens + ab, abc, abcabc, abcabcabc, abcabcabcabc,
    // abcabcabcabcabcabc (6), then doubling. Pick ranks so each longer
    // merge has the lowest rank, forcing greedy deep merging of
    // "abcabc..." (50 bytes ~= 16 repeats of "abc" + 2 leftover bytes).
    var bpe = try buildByteVocab(testing.allocator, &.{
        .{ .bytes = "ab", .rank = 256 },
        .{ .bytes = "abc", .rank = 257 },
        .{ .bytes = "abcabc", .rank = 258 },
        .{ .bytes = "abcabcabc", .rank = 259 },
        .{ .bytes = "abcabcabcabc", .rank = 260 },
    });
    defer bpe.deinit();

    // 16 repeats of "abc" = 48 bytes + "ab" = 50 bytes.
    var input: [50]u8 = undefined;
    var i: usize = 0;
    while (i < 48) : (i += 3) {
        input[i] = 'a';
        input[i + 1] = 'b';
        input[i + 2] = 'c';
    }
    input[48] = 'a';
    input[49] = 'b';

    var out: [64]TokenId = undefined;
    const ids = bpe.encodeChunk(&input, &out);

    // BPE greedy: merge "ab" first (rank 256), then "ab"+"c" => "abc"
    // (rank 257), then "abc"+"abc" => "abcabc" (rank 258), etc.
    // The final segmentation depends on rank ordering; we just assert
    // the round-trip is exact and the result is meaningfully merged.
    var decoded: std.ArrayList(u8) = .empty;
    defer decoded.deinit(testing.allocator);
    for (ids) |id| try decoded.appendSlice(testing.allocator, bpe.idBytes(id));
    try testing.expectEqualSlices(u8, &input, decoded.items);

    // Deeply merged: should be far fewer than 50 ids. With the vocab
    // above, the merger collapses runs of "abc" into longer tokens.
    try testing.expect(ids.len < 15);
}

test "encodeChunk rank-precomputation finds non-first min" {
    // Input "aabc": pairs are (a,a)=ranked, (a,b)=ranked, (b,c)=lowest.
    // The min-rank pair is the LAST pair, not the first, so a naive
    // first-match scan would be wrong. Confirms scanMin/rank-table
    // correctness independent of scan order.
    // {a:0,b:1,c:2,aa:5,ab:4,bc:3} — bc has the lowest non-singleton rank.
    var bpe = try buildByteVocab(testing.allocator, &.{
        .{ .bytes = "aa", .rank = 258 },
        .{ .bytes = "ab", .rank = 257 },
        .{ .bytes = "bc", .rank = 256 },
    });
    defer bpe.deinit();

    var out: [8]TokenId = undefined;
    const ids = bpe.encodeChunk("aabc", &out);

    // Expected merge trace:
    //   [a,a,b,c] -> bc (rank 256) wins -> [a,a,bc]
    //   pairs now: (a,a)=258, (a,bc)=? not in vocab -> INVALID
    //   so (a,a) wins -> [aa,bc]
    //   pairs: (aa,bc)=? not in vocab -> done.
    // Final ids: [aa, bc] = [258, 256].
    try testing.expectEqual(@as(usize, 2), ids.len);
    try testing.expectEqual(@as(TokenId, 258), ids[0]);
    try testing.expectEqual(@as(TokenId, 256), ids[1]);
}

test "encodeChunkWithOffsets reports byte ranges" {
    // {a:0, b:1, ab:2}. "ab" -> single token covering [0, 2).
    const src = try buildVocabSource(testing.allocator, &.{
        .{ .bytes = "a", .rank = 0 },
        .{ .bytes = "b", .rank = 1 },
        .{ .bytes = "ab", .rank = 2 },
    });
    defer testing.allocator.free(src);
    var bpe = try Bpe.loadTiktokenBytes(testing.allocator, src);
    defer bpe.deinit();

    var out_ids: [4]TokenId = undefined;
    var out_off: [4]Span = undefined;
    const n = bpe.encodeChunkWithOffsets("ab", 0, &out_ids, &out_off);
    try testing.expectEqual(@as(usize, 1), n);
    try testing.expectEqual(@as(TokenId, 2), out_ids[0]);
    try testing.expectEqual(@as(u32, 0), out_off[0].start);
    try testing.expectEqual(@as(u32, 2), out_off[0].end);
}

test "encodeChunkWithOffsets handles partial merges" {
    // {a:0, b:1, x:2}. "abx" -> three singleton tokens, offsets
    // [0,1)[1,2)[2,3). x is given a single-byte token so the chunk fully
    // covers; without 'x' the byte would be a missing-key sentinel.
    var bpe = try buildByteVocab(testing.allocator, &.{});
    defer bpe.deinit();

    var out_ids: [4]TokenId = undefined;
    var out_off: [4]Span = undefined;
    const n = bpe.encodeChunkWithOffsets("abx", 0, &out_ids, &out_off);
    try testing.expectEqual(@as(usize, 3), n);
    try testing.expectEqual(@as(u32, 0), out_off[0].start);
    try testing.expectEqual(@as(u32, 1), out_off[0].end);
    try testing.expectEqual(@as(u32, 1), out_off[1].start);
    try testing.expectEqual(@as(u32, 2), out_off[1].end);
    try testing.expectEqual(@as(u32, 2), out_off[2].start);
    try testing.expectEqual(@as(u32, 3), out_off[2].end);
}

test "encodeChunkWithOffsets respects chunk_offset" {
    var bpe = try buildByteVocab(testing.allocator, &.{
        .{ .bytes = "ab", .rank = 256 },
    });
    defer bpe.deinit();

    var out_ids: [4]TokenId = undefined;
    var out_off: [4]Span = undefined;
    const n = bpe.encodeChunkWithOffsets("ab", 10, &out_ids, &out_off);
    try testing.expectEqual(@as(usize, 1), n);
    try testing.expectEqual(@as(u32, 10), out_off[0].start);
    try testing.expectEqual(@as(u32, 12), out_off[0].end);
}

test "Bpe.byte_fallback defaults to null on tiktoken vocabs" {
    // Loaders that don't opt into byte_fallback (tiktoken / HF byte-level)
    // must produce a Bpe with `byte_fallback == null`, so the encode
    // post-pass is a no-op single null-check branch.
    const src = try buildVocabSource(testing.allocator, &.{
        .{ .bytes = "a", .rank = 0 },
        .{ .bytes = "b", .rank = 1 },
        .{ .bytes = "ab", .rank = 2 },
    });
    defer testing.allocator.free(src);
    var bpe = try Bpe.loadTiktokenBytes(testing.allocator, src);
    defer bpe.deinit();

    try testing.expect(bpe.byte_fallback == null);
}

test "encodeChunk byte_fallback rewrites unmerged singletons" {
    // Build a vocab manually that mimics SP byte-fallback: every byte
    // has a token, plus a multi-byte "ab" merge. Then set the
    // byte_fallback table to remap a few bytes to far-away ids and
    // confirm those remappings take effect after merging.
    var bpe = try buildByteVocab(testing.allocator, &.{
        .{ .bytes = "ab", .rank = 256 },
    });
    defer bpe.deinit();

    // Manually populate the table: remap byte 'z' (0x7a) to id 999 (a
    // sentinel value, doesn't have to exist in the vocab; the encoder
    // doesn't validate). Other bytes left as maxInt (no remap), so the
    // post-pass keeps the hashmap-resolved id from the byte vocab.
    var table: [256]TokenId = undefined;
    @memset(&table, std.math.maxInt(TokenId));
    table['z'] = 999;
    bpe.byte_fallback = table;

    var out: [4]TokenId = undefined;

    // "ab" merges; the post-pass sees a length-2 chunk and skips it.
    {
        const ids = bpe.encodeChunk("ab", &out);
        try testing.expectEqualSlices(TokenId, &.{256}, ids);
    }

    // "z" is unmerged length-1: post-pass rewrites it to 999.
    {
        const ids = bpe.encodeChunk("z", &out);
        try testing.expectEqualSlices(TokenId, &.{999}, ids);
    }

    // "azb" -> 'a' (no remap, stays 0x61), 'z' (remapped to 999), 'b'
    // (no remap, stays 0x62). No "ab" merge because the 'z' splits them.
    {
        const ids = bpe.encodeChunk("azb", &out);
        try testing.expectEqualSlices(TokenId, &.{ 0x61, 999, 0x62 }, ids);
    }
}

// --- 1.15 two-level merge-rank layout tests --------------------------

test "hot table: every hot entry agrees with by_bytes lookup" {
    // Regression check: for every populated hot slot, the inline key+id
    // must roundtrip through `by_bytes.get` to the same id. Catches
    // any future bug in `buildHotTable`'s population or `hotLookup`'s
    // compare logic.
    var bpe = try buildByteVocab(testing.allocator, &.{
        .{ .bytes = "ab", .rank = 256 },
        .{ .bytes = "cd", .rank = 257 },
        .{ .bytes = "abc", .rank = 258 },
        .{ .bytes = "abcd", .rank = 259 },
        .{ .bytes = "abcde", .rank = 260 },
        .{ .bytes = "abcdef", .rank = 261 },
        .{ .bytes = "abcdefg", .rank = 262 },
    });
    defer bpe.deinit();

    const table = bpe.hot_table orelse return error.HotTableMissing;
    var checked: u32 = 0;
    for (table) |slot| {
        if (slot.hash == 0) continue; // empty slot
        const key = slot.key_bytes[0..slot.key_len];
        const cold = bpe.by_bytes.get(key) orelse return error.MissingInColdTable;
        try testing.expectEqual(cold, slot.id);
        const hot = bpe.hotLookup(key) orelse return error.HotLookupMiss;
        try testing.expectEqual(slot.id, hot);
        checked += 1;
    }
    // 256 single-byte pieces + 7 short multi-byte pieces all qualify
    // (len <= HOT_KEY_INLINE_MAX). At 4096 slots vs ~263 keys we expect
    // close to zero collisions, so almost every key lives in the table.
    try testing.expect(checked >= 250);
}

test "hot table: vocab smaller than HOT_CAPACITY covers all short keys" {
    // 256 single-byte pieces, all length 1 → fit in the hot table with
    // very few collisions. Verify every length-1 byte returns its id
    // via `hotLookup`. (Some byte slots can collide; assert at least
    // 240 of 256 hit, which is the worst-case load factor for 256
    // FNV-1a-hashed 1-byte keys into 4096 slots.)
    var bpe = try buildByteVocab(testing.allocator, &.{});
    defer bpe.deinit();

    var hits: u32 = 0;
    var b: u32 = 0;
    while (b < 256) : (b += 1) {
        const byte = [_]u8{@intCast(b)};
        if (bpe.hotLookup(&byte)) |id| {
            try testing.expectEqual(@as(TokenId, b), id);
            hits += 1;
        }
    }
    try testing.expect(hits >= 240);
}

test "hot table: no false positives on hash collision" {
    // Construct a tiny vocab where one piece is known to occupy a slot;
    // then probe random short keys that aren't in the vocab and
    // confirm `hotLookup` returns null for all of them (not a stale
    // id from a different key hashing to the same slot).
    const src = try buildVocabSource(testing.allocator, &.{
        .{ .bytes = "x", .rank = 0 },
        .{ .bytes = "y", .rank = 1 },
        .{ .bytes = "xy", .rank = 2 },
    });
    defer testing.allocator.free(src);
    var bpe = try Bpe.loadTiktokenBytesWithOptions(testing.allocator, src, .{ .hot_table = true });
    defer bpe.deinit();

    // None of these are vocab entries; `hotLookup` must NOT spuriously
    // return an id.
    const noise_keys = [_][]const u8{
        "a", "b", "z", "ab", "zz", "abc", "yzx", "qrstu", "abcdefg",
    };
    for (noise_keys) |k| {
        try testing.expectEqual(@as(?TokenId, null), bpe.hotLookup(k));
    }
    // But the known entries must hit.
    try testing.expectEqual(@as(?TokenId, 0), bpe.hotLookup("x"));
    try testing.expectEqual(@as(?TokenId, 1), bpe.hotLookup("y"));
    try testing.expectEqual(@as(?TokenId, 2), bpe.hotLookup("xy"));
}

test "hot table: encode bit-identical with and without hot table" {
    // Build a fully-loaded vocab via the normal path (hot_table populated),
    // run encodeChunk against a corpus of inputs, then null the hot
    // table and re-encode the same inputs. Outputs must be byte-identical.
    var bpe = try buildByteVocab(testing.allocator, &.{
        .{ .bytes = "ab", .rank = 256 },
        .{ .bytes = "cd", .rank = 257 },
        .{ .bytes = "ef", .rank = 258 },
        .{ .bytes = "abcd", .rank = 259 },
        .{ .bytes = "abcdef", .rank = 260 },
    });
    defer bpe.deinit();

    const inputs = [_][]const u8{
        "ab",         "cd",        "abcd",          "abcdef",
        "abcdefab",   "efcdab",    "abcdefabcdef",  "xyzab",
        "abcdefg",    "a",         "f",             "abcabcabc",
    };

    var with_hot: [128]TokenId = undefined;
    var without_hot: [128]TokenId = undefined;
    for (inputs) |inp| {
        const a = bpe.encodeChunk(inp, &with_hot);
        // Stash the hot table, null it out for the cold-only path.
        const saved = bpe.hot_table;
        bpe.hot_table = null;
        const b = bpe.encodeChunk(inp, &without_hot);
        bpe.hot_table = saved;
        try testing.expectEqualSlices(TokenId, a, b);
    }
}

test "hot table: size sanity — at most 256 KB per Bpe" {
    // 4096 entries × 16 bytes = 64 KB. The task spec caps at 256 KB
    // per loaded BPE — this asserts the sizing constant stays in range
    // if someone bumps HOT_CAPACITY without realizing the cost.
    const bytes_per_bpe: usize = @as(usize, HOT_CAPACITY) * @sizeOf(HotEntry);
    try testing.expect(bytes_per_bpe <= 256 * 1024);
    try testing.expectEqual(@as(usize, 16), @sizeOf(HotEntry));
}

test "hot table: long keys never enter the hot table" {
    // Keys longer than HOT_KEY_INLINE_MAX (7) must skip population so
    // the inline-bytes equality check in `hotLookup` can't compare
    // against truncated storage.
    var bpe = try buildByteVocab(testing.allocator, &.{
        .{ .bytes = "abcdefgh", .rank = 256 }, // 8 bytes — too long
        .{ .bytes = "abcdefghij", .rank = 257 }, // 10 bytes — too long
        .{ .bytes = "abcdefg", .rank = 258 }, // 7 bytes — just fits
    });
    defer bpe.deinit();

    // The 7-byte key must be in the hot table.
    try testing.expectEqual(@as(?TokenId, 258), bpe.hotLookup("abcdefg"));
    // The 8- and 10-byte keys must NOT be in the hot table (hotLookup
    // returns null due to the length precheck).
    try testing.expectEqual(@as(?TokenId, null), bpe.hotLookup("abcdefgh"));
    try testing.expectEqual(@as(?TokenId, null), bpe.hotLookup("abcdefghij"));
    // But they're still reachable through by_bytes (the cold path).
    try testing.expectEqual(@as(?TokenId, 256), bpe.by_bytes.get("abcdefgh"));
    try testing.expectEqual(@as(?TokenId, 257), bpe.by_bytes.get("abcdefghij"));
}

// --- 1.16 hot-table opt-out tests -----------------------------------

test "1.16: loadTiktokenBytes defaults to hot_table=off" {
    // Single-shot constructors bias toward single-thread workloads
    // where the hot table is a -16% regression on cl100k. Confirm the
    // bare loader leaves `hot_table` null so the encode loop short-
    // circuits past `hotLookup` with one nullable check.
    const src = try buildVocabSource(testing.allocator, &.{
        .{ .bytes = "a", .rank = 0 },
        .{ .bytes = "b", .rank = 1 },
        .{ .bytes = "ab", .rank = 2 },
    });
    defer testing.allocator.free(src);
    var bpe = try Bpe.loadTiktokenBytes(testing.allocator, src);
    defer bpe.deinit();

    try testing.expect(bpe.hot_table == null);
}

test "1.16: loadTiktokenBytesWithOptions(.{ .hot_table = true }) populates it" {
    const src = try buildVocabSource(testing.allocator, &.{
        .{ .bytes = "a", .rank = 0 },
        .{ .bytes = "b", .rank = 1 },
        .{ .bytes = "ab", .rank = 2 },
    });
    defer testing.allocator.free(src);
    var bpe = try Bpe.loadTiktokenBytesWithOptions(testing.allocator, src, .{ .hot_table = true });
    defer bpe.deinit();

    try testing.expect(bpe.hot_table != null);
    // Sanity: at least one short key is reachable via hotLookup.
    try testing.expectEqual(@as(?TokenId, 2), bpe.hotLookup("ab"));
}

test "1.16: encode bit-identical with hot_table on vs off" {
    // Build the same vocab twice — once with hot_table on, once off.
    // Encode a battery of inputs through each Bpe and confirm
    // byte-equal outputs. Regression guard for any future divergence
    // between the hot-table and cold-only paths.
    var entries: std.ArrayList(TestEntry) = .empty;
    defer entries.deinit(testing.allocator);
    var byte_holders: [256][1]u8 = undefined;
    for (0..256) |i| {
        byte_holders[i][0] = @intCast(i);
        try entries.append(testing.allocator, .{ .bytes = byte_holders[i][0..1], .rank = @intCast(i) });
    }
    try entries.append(testing.allocator, .{ .bytes = "ab", .rank = 256 });
    try entries.append(testing.allocator, .{ .bytes = "cd", .rank = 257 });
    try entries.append(testing.allocator, .{ .bytes = "ef", .rank = 258 });
    try entries.append(testing.allocator, .{ .bytes = "abcd", .rank = 259 });
    try entries.append(testing.allocator, .{ .bytes = "abcdef", .rank = 260 });
    try entries.append(testing.allocator, .{ .bytes = "abcdefgh", .rank = 261 });

    const src = try buildVocabSource(testing.allocator, entries.items);
    defer testing.allocator.free(src);

    var bpe_off = try Bpe.loadTiktokenBytesWithOptions(testing.allocator, src, .{ .hot_table = false });
    defer bpe_off.deinit();
    var bpe_on = try Bpe.loadTiktokenBytesWithOptions(testing.allocator, src, .{ .hot_table = true });
    defer bpe_on.deinit();

    try testing.expect(bpe_off.hot_table == null);
    try testing.expect(bpe_on.hot_table != null);

    // Short inputs that stay on the inline merge-loop path
    // (< HEAP_THRESHOLD == 64 bytes). The heap path is covered by
    // existing `hot table: encode bit-identical with and without hot
    // table` test via the null-the-pointer pattern.
    const short_inputs = [_][]const u8{
        "ab",                   "cd",        "abcd",
        "abcdef",               "abcdefab",  "efcdab",
        "abcdefabcdef",         "xyzab",     "abcdefgh",
        "a",                    "f",         "abcabcabc",
        "abcdefabcdefabcdefab",
    };
    var out_off: [256]TokenId = undefined;
    var out_on: [256]TokenId = undefined;
    for (short_inputs) |inp| {
        const a = bpe_off.encodeChunk(inp, &out_off);
        const b = bpe_on.encodeChunk(inp, &out_on);
        try testing.expectEqualSlices(TokenId, a, b);
    }

    // Boundary case at HEAP_THRESHOLD (64 bytes) — last input that
    // stays on the inline path. Use an arena to absorb any scratch
    // alloc the larger path might do without leaking through the
    // testing GPA.
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const arena_alloc = arena.allocator();
    const med_input = try arena_alloc.alloc(u8, 64);
    for (med_input, 0..) |*c, i| c.* = "abcdef"[i % 6];
    var med_out_off: [128]TokenId = undefined;
    var med_out_on: [128]TokenId = undefined;
    const ma = bpe_off.encodeChunkScratch(arena_alloc, med_input, &med_out_off);
    const mb = bpe_on.encodeChunkScratch(arena_alloc, med_input, &med_out_on);
    try testing.expectEqualSlices(TokenId, ma, mb);

    // Long input (> HEAP_THRESHOLD) — heap path engaged on both sides.
    // Arena absorbs the heap path's `prev/next/heap_buf` scratch.
    const long_input = try arena_alloc.alloc(u8, 200);
    for (long_input, 0..) |*c, i| c.* = "abcdef"[i % 6];
    var long_out_off: [256]TokenId = undefined;
    var long_out_on: [256]TokenId = undefined;
    const la = bpe_off.encodeChunkScratch(arena_alloc, long_input, &long_out_off);
    const lb = bpe_on.encodeChunkScratch(arena_alloc, long_input, &long_out_on);
    try testing.expectEqualSlices(TokenId, la, lb);
}

// --- ignore_merges regression tests --------------------------------
//
// Llama-3 / GPT-4-style HF BPE vocabs ship `model.ignore_merges = true`
// in tokenizer.json. The HF Rust BPE encoder probes the entire pretok
// chunk against the vocab BEFORE running the merge loop and, on a hit,
// emits that single id directly. Without that short-circuit ztok's
// legacy byte-level merge loop uses `id` as the pair-rank, which can
// produce a different segmentation than HF's true pair-rank loop for
// multilingual sequences. Symptom on the 2026-05 stress sweep:
// Llama-3 × multilingual stuck at 95.0% match; the diverging tokens
// for line 1 (" Федерации") decoded to identical bytes but ztok split
// to 3 ids while HF emitted the single id 111112.

test "ignore_merges: whole-chunk lookup short-circuits the merge loop" {
    // Synthetic vocab modeled on the Llama-3 ` Федерации` symptom from
    // the stress sweep: the whole piece exists in the vocab but the
    // merge loop can't reach it because the intermediate pair pieces
    // (e.g. "abcd", "cdef") aren't in the vocab. Without
    // ignore_merges the loop gets stuck at 3 pieces; with it the
    // whole-chunk lookup short-circuits to a single id.
    //
    //   pieces: bytes 0..255,
    //           "ab" (256), "cd" (257), "ef" (258), "abcdef" (259)
    //   merge loop on "abcdef":
    //     (a,b)→256, (c,d)→257, (e,f)→258 all fire,
    //     leaving [ab, cd, ef]; (ab,cd)→"abcd"→missing,
    //     (cd,ef)→"cdef"→missing → no more valid pairs.
    var bpe = try buildByteVocab(testing.allocator, &.{
        .{ .bytes = "ab", .rank = 256 },
        .{ .bytes = "cd", .rank = 257 },
        .{ .bytes = "ef", .rank = 258 },
        .{ .bytes = "abcdef", .rank = 259 },
    });
    defer bpe.deinit();

    // Baseline: ignore_merges off -> merge loop stalls at [256,257,258].
    try testing.expect(!bpe.ignore_merges);
    var out: [16]TokenId = undefined;
    {
        const ids = bpe.encodeChunk("abcdef", &out);
        try testing.expectEqualSlices(TokenId, &.{ 256, 257, 258 }, ids);
    }

    // Flip ignore_merges on -> whole-chunk lookup wins, single id 259.
    bpe.ignore_merges = true;
    {
        const ids = bpe.encodeChunk("abcdef", &out);
        try testing.expectEqualSlices(TokenId, &.{259}, ids);
    }
    // Inputs that aren't a single vocab piece still fall through to
    // the normal merge loop. "abcdefg" has no whole-chunk match, so
    // ignore_merges must NOT swallow it as a single id.
    {
        const ids = bpe.encodeChunk("abcdefg", &out);
        try testing.expectEqualSlices(TokenId, &.{ 256, 257, 258, 0x67 }, ids);
    }
}

test "ignore_merges: non-ASCII multi-codepoint chunk maps to a single id" {
    // Mirrors the failing Llama-3 byte sequence from the stress sweep.
    // The byte-level encoded form of " Федерации" (UTF-8 bytes wrapped
    // through GPT-2 byte_to_unicode) is 38 bytes — well under the 64-
    // byte HEAP_THRESHOLD so this stays on the SoA inline path.
    // Vocab: bytes 0..255 + the whole chunk as a single piece (id 256).
    // The merge-rank-by-id heuristic over single bytes would never
    // reach the whole piece via byte-pair merges (no pair entries
    // exist in this synthetic vocab), so the only way to emit the
    // whole-chunk id is via the ignore_merges short-circuit.
    const fed_bytes = "\xC4\xA0\xC3\x90\xC2\xA4\xC3\x90\xC2\xB5\xC3\x90\xC2\xB4" ++
        "\xC3\x90\xC2\xB5\xC3\x91\xC4\xA2\xC3\x90\xC2\xB0\xC3\x91\xC4\xA8" ++
        "\xC3\x90\xC2\xB8\xC3\x90\xC2\xB8";
    var bpe = try buildByteVocab(testing.allocator, &.{
        .{ .bytes = fed_bytes, .rank = 256 },
    });
    defer bpe.deinit();
    bpe.ignore_merges = true;

    var out: [64]TokenId = undefined;
    const ids = bpe.encodeChunk(fed_bytes, &out);
    try testing.expectEqualSlices(TokenId, &.{256}, ids);
}

test "ignore_merges: offsets variant short-circuits identically" {
    // Same vocab shape as the first ignore_merges test; check that the
    // *WithOffsets path also short-circuits and emits a single span
    // covering the entire chunk. Required for the streaming /
    // offsets-emitting bridge clients (cli_eval, hf_bridge tests).
    var bpe = try buildByteVocab(testing.allocator, &.{
        .{ .bytes = "ab", .rank = 256 },
        .{ .bytes = "cd", .rank = 257 },
        .{ .bytes = "ef", .rank = 258 },
        .{ .bytes = "abcdef", .rank = 259 },
    });
    defer bpe.deinit();
    bpe.ignore_merges = true;

    var out_ids: [8]TokenId = undefined;
    var out_off: [8]Span = undefined;
    const n = bpe.encodeChunkWithOffsets("abcdef", 10, &out_ids, &out_off);
    try testing.expectEqual(@as(usize, 1), n);
    try testing.expectEqual(@as(TokenId, 259), out_ids[0]);
    try testing.expectEqual(@as(u32, 10), out_off[0].start);
    try testing.expectEqual(@as(u32, 16), out_off[0].end);
}
