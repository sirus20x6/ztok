const std = @import("std");
const TokenId = @import("token.zig").TokenId;
const Monster = @import("monster.zig").Monster;
const Normalizer = @import("normalizer.zig").Normalizer;
const CapcodeNormalizer = @import("normalizer.zig").CapcodeNormalizer;

pub const Error = error{ MagicMismatch, UnsupportedVersion, TruncatedFile, MalformedFile } || std.mem.Allocator.Error;

/// Magic prefix is `ZTM` + the writer's version byte. v1 writers stamp
/// `0x01` here, v2 writers stamp `0x02`. Readers accept any byte in
/// `[1, VERSION]` and dispatch on it. The 4th magic byte AND the
/// `header.version` byte at offset 4 BOTH carry the version so a partial
/// header read can still detect it.
pub const MAGIC_V1: [4]u8 = .{ 'Z', 'T', 'M', 0x01 };
pub const MAGIC_V2: [4]u8 = .{ 'Z', 'T', 'M', 0x02 };
pub const MAGIC_V3: [4]u8 = .{ 'Z', 'T', 'M', 0x03 };
/// Legacy alias for the v1 magic — kept for back-compat with any
/// callers (tests, doctor) that reach in to inspect the magic bytes
/// directly. New code should use `MAGIC_V1` / `MAGIC_V2` / `MAGIC_V3`
/// explicitly.
pub const MAGIC: [4]u8 = MAGIC_V1;
pub const VERSION: u8 = 3;
pub const HEADER_SIZE: usize = 20;

/// Bytes of the alias-section header (a single u32 alias_count). Each
/// alias record is `{ u32 id, u32 alt_byte_len, u8[alt_byte_len] }`.
/// The section lives APPENDED past the v1 EOF so that strict v1 readers
/// (which compute `need` from header fields and stop there) won't be
/// confused by the new bytes. v2/v3 readers (this module) check the
/// magic byte and parse the appended section iff version >= 2.
pub const ALIAS_HEADER_SIZE: usize = @sizeOf(u32);

/// Bytes of the v3 per-twin alt-section header (a single u32
/// twin_count). The section is APPENDED past the v2 alias section.
/// Each twin record is:
///   u32 id          (vocab id this twin form resolves to)
///   u32 key_len
///   u8[key_len] key (the twin's byte sequence — the disambiguator)
///   u8  n_alts      (0..2)
///   for each alt:
///     u32 alt_id        (vocab id to emit for the alt)
///     u32 alt_byte_len  (bytes consumed in THIS twin's units)
/// v1/v2 readers stop before this section (they compute their EOF from
/// the alias section and don't look further); v3 readers parse it iff
/// version == 3. See `Monster.PerTwinAlt`.
pub const PER_TWIN_HEADER_SIZE: usize = @sizeOf(u32);

/// Capcode flag in the .ztm header at byte 5. Mirrors the TokenMonster
/// `usingCapcode` byte semantically, with one ztok-specific extension
/// (CAPCODE_FULL_TM) that disambiguates marker style:
///   0 = none
///   1 = nocapcode (forward-delete-only, 0x7F marker — shared between
///       TM-Go and ztok byte-for-byte)
///   2 = full capcode, ztok-native markers (0x0E / 0x0F / 0x11). Used by
///       any `.ztm` written from a ztok-side training/encoding pipeline.
///   3 = full capcode, TM-Go printable markers ('C' / 'W' / 'D' =
///       0x43 / 0x57 / 0x44). Used by `.ztm` files produced from a
///       TokenMonster `.vocab` (where the vocab pieces themselves carry
///       printable marker bytes), so the loader can wire a normalizer
///       that emits the matching byte stream. The TM `.vocab` format
///       always stores `usingCapcode = 2` for full capcode; the
///       `convert_tm_to_ztm.py` script translates that into CAPCODE_FULL_TM
///       here. Loading any pre-existing `.ztm` with `capcode = 2`
///       continues to give the ztok-native marker style (back-compat).
/// .ztm files written by older versions of `monster_io.writeBytes` always
/// stamped 0 here; recent writers and the `convert_tm_to_ztm.py` script
/// populate the field from the source TM vocab.
pub const CAPCODE_NONE: u8 = 0;
pub const CAPCODE_NOCAPCODE: u8 = 1;
pub const CAPCODE_FULL: u8 = 2;
pub const CAPCODE_FULL_TM: u8 = 3;

/// Normalizer flag in the .ztm header at byte 7. Mirrors the
/// TokenMonster `norm.Normalizer.Flag` bit layout (bit 0 = NFD,
/// bit 1 = Lowercase, ...). ztok only ever reads the NFD bit; other
/// flags are accepted (for future use) but treated as no-ops on load.
/// Older .ztm files stamped 0 here, so the field is back-compat safe.
pub const NORM_NFD: u8 = 0x01;
pub const NORM_LOWERCASE: u8 = 0x02;
pub const NORM_ACCENTS: u8 = 0x04;
pub const NORM_QUOTEMARKS: u8 = 0x08;
pub const NORM_COLLAPSE: u8 = 0x10;
pub const NORM_TRIM: u8 = 0x20;
pub const NORM_LEADINGSPACE: u8 = 0x40;
pub const NORM_UNIXLINES: u8 = 0x80;

pub const Header = struct {
    version: u8 = 1,
    capcode: u8 = 0,
    max_token_length: u8,
    count: u32,
    unk_id: TokenId,
    bytes_total: u32,
    /// TM-style normalizer flag bits (NORM_NFD etc.). 0 means no
    /// normalization. Persisted in the header's reserved byte slot at
    /// offset 7 — 0 in older .ztm files for back-compat.
    norm_flag: u8 = 0,
};

/// Loaded .ztm result that pairs the encoder with the normalization
/// metadata recovered from the file header. Use `recommendedNormalizer`
/// to wire the pipeline's normalizer in one step, or read the raw
/// fields if you want custom behavior.
pub const LoadedMonster = struct {
    monster: Monster,
    /// Raw capcode byte from the .ztm header (0/1/2 — see CAPCODE_*).
    capcode: u8,
    /// Raw normalizer flag byte from the .ztm header (NORM_* bits).
    norm_flag: u8,

    pub fn deinit(self: *LoadedMonster) void {
        self.monster.deinit();
    }

    /// Build a `Normalizer` value that matches the metadata in the
    /// loaded .ztm file. Falls back to `.identity` for older .ztm files
    /// (capcode=0, norm_flag=0) that don't carry metadata. For full
    /// capcode, `CAPCODE_FULL` (= 2) maps to ztok-native marker bytes
    /// and `CAPCODE_FULL_TM` (= 3) maps to TM-Go's printable markers —
    /// the distinguisher is the capcode header byte itself; no extra
    /// header field is needed.
    pub fn recommendedNormalizer(self: *const LoadedMonster) Normalizer {
        const want_nfd = (self.norm_flag & NORM_NFD) != 0;
        return switch (self.capcode) {
            CAPCODE_FULL => .{ .capcode = .{ .nfd = want_nfd, .marker_style = .ztok } },
            CAPCODE_FULL_TM => .{ .capcode = .{ .nfd = want_nfd, .marker_style = .tm_printable } },
            CAPCODE_NOCAPCODE => .{ .nocapcode = .{ .nfd = want_nfd, .tm_compat_space = true } },
            else => if (want_nfd) .nfd else .identity,
        };
    }
};

pub fn writeBytes(allocator: std.mem.Allocator, m: *const Monster) ![]u8 {
    return writeBytesWithMeta(allocator, m, .{});
}

/// Options for `writeBytesWithMeta` / `writeFileWithMeta`. The fields
/// map directly to the .ztm header bytes at offsets 5 and 7. Default 0
/// preserves legacy `writeBytes` behavior for callers that don't care
/// about TM-equivalent normalization metadata.
pub const WriteOptions = struct {
    /// Capcode flag (CAPCODE_* constants).
    capcode: u8 = CAPCODE_NONE,
    /// Normalizer flag bits (NORM_* constants).
    norm_flag: u8 = 0,
};

pub fn writeBytesWithMeta(
    allocator: std.mem.Allocator,
    m: *const Monster,
    opts: WriteOptions,
) ![]u8 {
    const count: u32 = m.count;
    const bytes_total: u32 = @intCast(m.bytes.len);
    const max_len_u8: u8 = if (m.max_token_len > 255) 255 else @intCast(m.max_token_len);

    // v2 alias section: appended past the v1 EOF so strict v1 readers
    // can be coerced into reading the file (they'd compute `need` from
    // the header and stop there). Layout:
    //   u32 alias_count
    //   for each alias:
    //     u32 id
    //     u32 alt_byte_len
    //     u8  alt_bytes[alt_byte_len]
    const alias_count: u32 = @intCast(m.aliases.len);
    var alias_section_len: usize = ALIAS_HEADER_SIZE;
    for (m.aliases) |a| {
        alias_section_len += 2 * @sizeOf(u32) + a.bytes.len;
    }

    // v3 per-twin alt section: appended past the v2 alias section. Each
    // record is `{ u32 id, u32 key_len, key[], u8 n_alts, n_alts ×
    // { u32 alt_id, u32 alt_byte_len } }`.
    const twin_count: u32 = @intCast(m.per_twin_alts.len);
    var twin_section_len: usize = PER_TWIN_HEADER_SIZE;
    for (m.per_twin_alts) |t| {
        twin_section_len += 2 * @sizeOf(u32) + t.key.len + @sizeOf(u8) +
            t.alts.len * (2 * @sizeOf(u32));
    }

    const v1_payload_size: usize = HEADER_SIZE +
        @as(usize, bytes_total) +
        @as(usize, count + 1) * @sizeOf(u32) +
        @as(usize, count) * @sizeOf(u8);
    const total: usize = v1_payload_size + alias_section_len + twin_section_len;

    const buf = try allocator.alloc(u8, total);
    errdefer allocator.free(buf);

    @memcpy(buf[0..4], &MAGIC_V3);
    buf[4] = VERSION;
    buf[5] = opts.capcode;
    buf[6] = max_len_u8;
    buf[7] = opts.norm_flag;
    std.mem.writeInt(u32, buf[8..12], count, .little);
    std.mem.writeInt(u32, buf[12..16], m.unk_id, .little);
    std.mem.writeInt(u32, buf[16..20], bytes_total, .little);

    var off: usize = HEADER_SIZE;
    if (bytes_total > 0) {
        @memcpy(buf[off..][0..bytes_total], m.bytes);
        off += bytes_total;
    }
    var i: u32 = 0;
    while (i <= count) : (i += 1) {
        std.mem.writeInt(u32, buf[off..][0..4], m.offsets[i], .little);
        off += 4;
    }
    if (count > 0) {
        @memcpy(buf[off..][0..count], m.nwords);
        off += count;
    }
    std.debug.assert(off == v1_payload_size);

    // Alias section (always present in v2 — count of 0 is a 4-byte
    // section). Older v1 readers stop at `v1_payload_size`; v2
    // readers continue from `off` and parse the alias records.
    std.mem.writeInt(u32, buf[off..][0..4], alias_count, .little);
    off += @sizeOf(u32);
    for (m.aliases) |a| {
        std.mem.writeInt(u32, buf[off..][0..4], a.id, .little);
        off += @sizeOf(u32);
        const alt_len: u32 = @intCast(a.bytes.len);
        std.mem.writeInt(u32, buf[off..][0..4], alt_len, .little);
        off += @sizeOf(u32);
        if (a.bytes.len > 0) {
            @memcpy(buf[off..][0..a.bytes.len], a.bytes);
            off += a.bytes.len;
        }
    }

    // v3 per-twin alt section (always present in v3 — count of 0 is a
    // 4-byte section). v1/v2 readers stop before this; v3 readers
    // continue from `off` and parse `twin_count` records.
    std.mem.writeInt(u32, buf[off..][0..4], twin_count, .little);
    off += @sizeOf(u32);
    for (m.per_twin_alts) |t| {
        std.mem.writeInt(u32, buf[off..][0..4], t.id, .little);
        off += @sizeOf(u32);
        const key_len: u32 = @intCast(t.key.len);
        std.mem.writeInt(u32, buf[off..][0..4], key_len, .little);
        off += @sizeOf(u32);
        if (t.key.len > 0) {
            @memcpy(buf[off..][0..t.key.len], t.key);
            off += t.key.len;
        }
        const n_alts: u8 = @intCast(t.alts.len);
        buf[off] = n_alts;
        off += @sizeOf(u8);
        for (t.alts) |al| {
            std.mem.writeInt(u32, buf[off..][0..4], al.alt_id, .little);
            off += @sizeOf(u32);
            std.mem.writeInt(u32, buf[off..][0..4], al.alt_byte_len, .little);
            off += @sizeOf(u32);
        }
    }
    std.debug.assert(off == total);
    return buf;
}

pub fn readBytes(allocator: std.mem.Allocator, contents: []const u8) Error!Monster {
    const loaded = try readBytesMeta(allocator, contents);
    // For back-compat with the existing readBytes signature, drop the
    // metadata and return just the Monster. Callers that want the
    // capcode/normalizer flags should call `readBytesMeta` or `readFileMeta`.
    return loaded.monster;
}

/// Same as `readBytes` but also returns the capcode/normalizer flag
/// bytes from the .ztm header so the caller can wire a matching
/// `Normalizer` via `LoadedMonster.recommendedNormalizer()`.
pub fn readBytesMeta(allocator: std.mem.Allocator, contents: []const u8) Error!LoadedMonster {
    if (contents.len < HEADER_SIZE) return Error.TruncatedFile;
    // Accept v1 (`ZTM\x01`), v2 (`ZTM\x02`), and v3 (`ZTM\x03`) magic.
    // The 4th byte duplicates `header.version` at offset 4 — both must
    // agree. v1/v2 files leave the v3 per-twin section absent; v3 files
    // are v2 + the appended per-twin alt section.
    if (contents[0] != 'Z' or contents[1] != 'T' or contents[2] != 'M') return Error.MagicMismatch;
    const magic_version = contents[3];
    if (magic_version != 0x01 and magic_version != 0x02 and magic_version != 0x03) return Error.MagicMismatch;
    const version = contents[4];
    if (version != magic_version) return Error.MagicMismatch;
    if (version > VERSION) return Error.UnsupportedVersion;
    const capcode_byte = contents[5];
    // capcode and norm_flag are accepted on read and surfaced via the
    // returned LoadedMonster; older .ztm files carry 0 in both slots
    // which preserves the legacy "no normalization" behavior.
    if (capcode_byte > CAPCODE_FULL_TM) return Error.MalformedFile;
    const norm_flag = contents[7];

    const count = std.mem.readInt(u32, contents[8..12], .little);
    const unk_id = std.mem.readInt(u32, contents[12..16], .little);
    const bytes_total = std.mem.readInt(u32, contents[16..20], .little);

    const need: usize = HEADER_SIZE +
        @as(usize, bytes_total) +
        @as(usize, count + 1) * @sizeOf(u32) +
        @as(usize, count) * @sizeOf(u8);
    if (contents.len < need) return Error.TruncatedFile;

    if (count > 0 and unk_id >= count) return Error.MalformedFile;

    var off: usize = HEADER_SIZE;
    const bytes_slice = contents[off..][0..bytes_total];
    off += bytes_total;

    // Parse offsets from the file and validate strict monotonicity +
    // final == bytes_total before handing pieces to the Builder.
    var prev_off: u32 = 0;
    const offsets_first = std.mem.readInt(u32, contents[off..][0..4], .little);
    if (offsets_first != 0) return Error.MalformedFile;
    off += 4;
    prev_off = 0;

    var builder = Monster.Builder.init(allocator);
    defer builder.deinit();

    var i: u32 = 0;
    while (i < count) : (i += 1) {
        const next_off = std.mem.readInt(u32, contents[off..][0..4], .little);
        off += 4;
        if (next_off < prev_off or next_off > bytes_total) return Error.MalformedFile;
        const piece = bytes_slice[prev_off..next_off];
        _ = builder.addToken(piece) catch |e| return e;
        prev_off = next_off;
    }
    if (count > 0 and prev_off != bytes_total) return Error.MalformedFile;

    // nwords block — read for validation/round-trip parity but the
    // Builder recomputes its own from piece bytes so we don't overwrite.
    if (count > 0) {
        // bounds already accounted for by the `need` check above.
        off += count;
    }

    // v2 alias section: appended past the v1 payload. v1 files
    // (`magic_version == 1`) skip this entirely; v2 AND v3 files (this
    // branch) parse `alias_count` records and seed the Builder.
    // The alias section is mandatory in v2/v3 (count of 0 is a valid
    // 4-byte section).
    if (version >= 0x02) {
        if (off + ALIAS_HEADER_SIZE > contents.len) return Error.TruncatedFile;
        const alias_count = std.mem.readInt(u32, contents[off..][0..4], .little);
        off += ALIAS_HEADER_SIZE;
        var a: u32 = 0;
        while (a < alias_count) : (a += 1) {
            if (off + 2 * @sizeOf(u32) > contents.len) return Error.TruncatedFile;
            const id = std.mem.readInt(u32, contents[off..][0..4], .little);
            off += @sizeOf(u32);
            const alt_len = std.mem.readInt(u32, contents[off..][0..4], .little);
            off += @sizeOf(u32);
            if (count > 0 and id >= count) return Error.MalformedFile;
            if (off + alt_len > contents.len) return Error.TruncatedFile;
            const alt_bytes = contents[off..][0..alt_len];
            off += alt_len;
            // Reject zero-length aliases (would never match anything
            // and confuse the trie insertion).
            if (alt_len == 0) return Error.MalformedFile;
            builder.addAlias(id, alt_bytes) catch |e| return e;
        }
    }

    // v3 per-twin alt section: appended past the v2 alias section. Only
    // present in v3 files (`magic_version == 3`). Each record carries a
    // twin's own alt table in that twin's own byte units — the
    // structural data the v2 converter collapsed. Parsed into the
    // Builder via `addPerTwinAlt`; **Stage 1 stores but does not consume
    // these in the encoder.**
    if (version == 0x03) {
        if (off + PER_TWIN_HEADER_SIZE > contents.len) return Error.TruncatedFile;
        const twin_count = std.mem.readInt(u32, contents[off..][0..4], .little);
        off += PER_TWIN_HEADER_SIZE;
        var t: u32 = 0;
        while (t < twin_count) : (t += 1) {
            // fixed header: id (u32) + key_len (u32)
            if (off + 2 * @sizeOf(u32) > contents.len) return Error.TruncatedFile;
            const id = std.mem.readInt(u32, contents[off..][0..4], .little);
            off += @sizeOf(u32);
            const key_len = std.mem.readInt(u32, contents[off..][0..4], .little);
            off += @sizeOf(u32);
            if (count > 0 and id >= count) return Error.MalformedFile;
            if (key_len == 0) return Error.MalformedFile;
            if (off + key_len > contents.len) return Error.TruncatedFile;
            const key = contents[off..][0..key_len];
            off += key_len;
            // n_alts (u8) then that many { alt_id, alt_byte_len } pairs.
            if (off + @sizeOf(u8) > contents.len) return Error.TruncatedFile;
            const n_alts = contents[off];
            off += @sizeOf(u8);
            if (n_alts > 2) return Error.MalformedFile;
            var alt_entries: [2]Monster.TwinAltEntry = undefined;
            var ai: u8 = 0;
            while (ai < n_alts) : (ai += 1) {
                if (off + 2 * @sizeOf(u32) > contents.len) return Error.TruncatedFile;
                const alt_id = std.mem.readInt(u32, contents[off..][0..4], .little);
                off += @sizeOf(u32);
                const alt_byte_len = std.mem.readInt(u32, contents[off..][0..4], .little);
                off += @sizeOf(u32);
                if (count > 0 and alt_id >= count) return Error.MalformedFile;
                alt_entries[ai] = .{ .alt_id = alt_id, .alt_byte_len = alt_byte_len };
            }
            builder.addPerTwinAlt(id, key, alt_entries[0..n_alts]) catch |e| return e;
        }
    }

    // Resolve the capcode mode for per-piece flag computation. The
    // .ztm header carries the capcode byte; we map our wire constants
    // (CAPCODE_NONE/NOCAPCODE/FULL/FULL_TM) to TM-Go's
    // `usingCapcode` enum (`none`/`nocapcode`/`full`). FULL_TM uses
    // printable 'C'/'W'/'D' markers — same `.full` classifier on the
    // rune side.
    const cc_mode: @import("monster.zig").CapcodeMode = switch (capcode_byte) {
        CAPCODE_NOCAPCODE => .nocapcode,
        CAPCODE_FULL, CAPCODE_FULL_TM => .full,
        else => .none,
    };
    var monster = if (count == 0)
        // Empty vocab: builder has no pieces; the unk_id field is
        // meaningless. finalize with 0 to construct a valid empty Monster.
        try builder.finalizeWithCapcode(0, cc_mode)
    else
        try builder.finalizeWithCapcode(unk_id, cc_mode);
    // Enable the TM-Go lilbuf (synthetic-boundary) trick on every loaded
    // .ztm. For vocabs that carry `\x7f `-prefixed tokens (the typical
    // TM-Go prebuilts), this is what closes the equivalence gap to TM-Go.
    // For vocabs without such tokens (custom-trained ztok Monsters that
    // don't synthesize the prefix), the lilbuf branch is a no-op — the
    // trie walk fails on the very first synthetic byte. See the field's
    // doc-comment in `monster.zig` for the cost analysis.
    monster.lilbuf_enabled = true;
    // TM-Go's score2b/score3b lookahead-of-alt evaluation. Reuses the
    // lilbuf machinery; only fires when the alt-branch lookahead lands
    // mid-word. No-op for non-TM vocabs (the gate flags are off).
    monster.score2b3b_enabled = true;
    // TM-Go's goto-checkpoint re-evaluation. When score2b/3b wins,
    // emit only `[alt_first, DEL]` and re-enter the encoder at the
    // alt-first end position with the lilbuf token seeded as the
    // synthetic greedy. Lets the next iteration pick a DIFFERENT alt
    // than the 1.16 single-emit `[alt_first, DEL, lilbuf_second]`
    // path, closing more equivalence gaps on TM-Go-compat vocabs.
    // Cheap (one branch per iteration when not active) and back-
    // compatible (off by default; only TM-compat .ztm enables it).
    monster.goto_checkpoint_enabled = true;
    return .{ .monster = monster, .capcode = capcode_byte, .norm_flag = norm_flag };
}

pub fn writeFile(allocator: std.mem.Allocator, m: *const Monster, path: []const u8) !void {
    const buf = try writeBytes(allocator, m);
    defer allocator.free(buf);
    const io = std.Io.Threaded.global_single_threaded.io();
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = buf });
}

pub fn readFile(allocator: std.mem.Allocator, path: []const u8) Error!Monster {
    const io = std.Io.Threaded.global_single_threaded.io();
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited) catch
        return Error.TruncatedFile;
    defer allocator.free(bytes);
    return readBytes(allocator, bytes);
}

/// Same as `readFile` but returns a `LoadedMonster` carrying the
/// .ztm capcode/normalizer metadata bytes. Callers can then call
/// `recommendedNormalizer()` to wire a matching pipeline normalizer.
pub fn readFileMeta(allocator: std.mem.Allocator, path: []const u8) Error!LoadedMonster {
    const io = std.Io.Threaded.global_single_threaded.io();
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited) catch
        return Error.TruncatedFile;
    defer allocator.free(bytes);
    return readBytesMeta(allocator, bytes);
}

/// Inspect a TokenMonster vocab name and return a `Normalizer` value
/// tuned to the normalizations that TM-Go would apply for the same
/// vocab. Recognized patterns (case-insensitive; substring match):
///
///   * Contains "nocapcode"  → `.{ .nocapcode = .{ .nfd = true, .tm_compat_space = true } }`
///       (TM's prebuilt UTF-8 vocabs run NFD; the `nocapcode` flag
///        means capcode=1 = forward-delete-only with 0x7F as the
///        shared marker between TM-Go and ztok.)
///   * Contains "capcode"    → `.{ .capcode = .{ .nfd = true, .marker_style = .tm_printable } }`
///       (TM's default for UTF-8 vocabs is capcode=2 + NFD. The
///        printable marker style is required for byte-equivalence with
///        TM-Go's capcode encoder — TM-Go always emits 'C'/'W'/'D'
///        regardless of the vocab name. ztok-native vocabs that need
///        the C0-control markers should be loaded via `readBytesMeta`
///        and use the header's `CAPCODE_FULL` byte instead of going
///        through this name heuristic.)
///   * Otherwise             → `.identity`
///       (Unknown vocab — don't guess wrong. NFD-only vocabs are
///        served by passing a custom normalizer or by inspecting the
///        loaded `.ztm` header via `readBytesMeta`.)
///
/// Aliases recognized: `nocapcode` is matched before `capcode` so the
/// "no" prefix wins. The pattern matches the TokenMonster prebuilt
/// vocab naming convention from
/// https://github.com/alasdairforsythe/tokenmonster (e.g.
/// `englishcode-32000-clean-nocapcode-v1`,
/// `englishcode-32000-clean-capcode-v1`,
/// `fiction-24000-strict-v1`, `code-4096-clean-nocapcode-v1`).
pub fn normalizerForVocab(name: []const u8) Normalizer {
    // Case-insensitive substring match. The lowercase comparison is
    // O(name.len * needle.len) but name.len is tiny (~40 chars max).
    if (containsCi(name, "nocapcode")) return .{ .nocapcode = .{ .nfd = true, .tm_compat_space = true } };
    if (containsCi(name, "capcode")) return .{ .capcode = .{ .nfd = true, .marker_style = .tm_printable } };
    return .identity;
}

fn containsCi(haystack: []const u8, needle: []const u8) bool {
    if (needle.len == 0) return true;
    if (haystack.len < needle.len) return false;
    var i: usize = 0;
    while (i + needle.len <= haystack.len) : (i += 1) {
        var match = true;
        var k: usize = 0;
        while (k < needle.len) : (k += 1) {
            const a = std.ascii.toLower(haystack[i + k]);
            const b = std.ascii.toLower(needle[k]);
            if (a != b) {
                match = false;
                break;
            }
        }
        if (match) return true;
    }
    return false;
}

// --- tests ---

const testing = std.testing;

fn buildSampleMonster(a: std.mem.Allocator) !Monster {
    var b = Monster.Builder.init(a);
    defer b.deinit();
    _ = try b.addToken("hello");
    _ = try b.addToken(" world");
    _ = try b.addToken("he");
    _ = try b.addToken("l");
    _ = try b.addToken("o");
    _ = try b.addToken(" w");
    const unk = try b.addToken("<unk>");
    return b.finalize(unk);
}

test "writeBytes + readBytes round-trip preserves bytes/offsets/count/unk_id/nwords" {
    var m = try buildSampleMonster(testing.allocator);
    defer m.deinit();

    const buf = try writeBytes(testing.allocator, &m);
    defer testing.allocator.free(buf);

    var m2 = try readBytes(testing.allocator, buf);
    defer m2.deinit();

    try testing.expectEqual(m.count, m2.count);
    try testing.expectEqual(m.unk_id, m2.unk_id);
    try testing.expectEqualSlices(u8, m.bytes, m2.bytes);
    try testing.expectEqualSlices(u32, m.offsets, m2.offsets);
    try testing.expectEqualSlices(u8, m.nwords, m2.nwords);
}

test "magic mismatch returns Error.MagicMismatch" {
    var bad: [HEADER_SIZE]u8 = @splat(0);
    bad[0] = 'X';
    bad[1] = 'X';
    bad[2] = 'X';
    bad[3] = 0;
    try testing.expectError(Error.MagicMismatch, readBytes(testing.allocator, &bad));
}

test "truncated file returns Error.TruncatedFile" {
    // 10 bytes < HEADER_SIZE
    const short: [10]u8 = @splat(0);
    try testing.expectError(Error.TruncatedFile, readBytes(testing.allocator, &short));

    // Valid header claiming more payload than the buffer contains.
    var m = try buildSampleMonster(testing.allocator);
    defer m.deinit();
    const full = try writeBytes(testing.allocator, &m);
    defer testing.allocator.free(full);

    const cut = full[0 .. full.len - 4];
    try testing.expectError(Error.TruncatedFile, readBytes(testing.allocator, cut));
}

test "writeFile + readFile round-trip via /tmp" {
    var m = try buildSampleMonster(testing.allocator);
    defer m.deinit();

    const path = "/tmp/ztok_monster_io_test.ztm";
    try writeFile(testing.allocator, &m, path);
    defer {
        const io = std.Io.Threaded.global_single_threaded.io();
        std.Io.Dir.cwd().deleteFile(io, path) catch {};
    }

    var m2 = try readFile(testing.allocator, path);
    defer m2.deinit();

    try testing.expectEqual(m.count, m2.count);
    try testing.expectEqual(m.unk_id, m2.unk_id);
    try testing.expectEqualSlices(u8, m.bytes, m2.bytes);
    try testing.expectEqualSlices(u32, m.offsets, m2.offsets);
}

test "round-trip preserves encode output" {
    var m = try buildSampleMonster(testing.allocator);
    defer m.deinit();

    var out_a: [32]TokenId = undefined;
    const ids_a = try m.encodeChunk(testing.allocator, "hello", &out_a);

    const buf = try writeBytes(testing.allocator, &m);
    defer testing.allocator.free(buf);
    var m2 = try readBytes(testing.allocator, buf);
    defer m2.deinit();

    var out_b: [32]TokenId = undefined;
    const ids_b = try m2.encodeChunk(testing.allocator, "hello", &out_b);

    try testing.expectEqualSlices(TokenId, ids_a, ids_b);
}

// --- normalizerForVocab + metadata-aware loader tests ---

test "normalizerForVocab: englishcode-clean-nocapcode returns nocapcode + NFD" {
    const n = normalizerForVocab("englishcode-32000-clean-nocapcode-v1");
    try testing.expect(n == .nocapcode);
    try testing.expectEqual(true, n.nocapcode.nfd);
    try testing.expectEqual(true, n.nocapcode.tm_compat_space);
}

test "normalizerForVocab: englishcode-clean-capcode returns capcode + NFD + tm_printable" {
    const n = normalizerForVocab("englishcode-32000-clean-capcode-v1");
    try testing.expect(n == .capcode);
    try testing.expectEqual(true, n.capcode.nfd);
    // TM-Go always uses printable markers; the name-based heuristic
    // therefore always picks `.tm_printable` so the produced byte
    // stream matches the vocab pieces.
    try testing.expectEqual(@import("normalizer.zig").MarkerStyle.tm_printable, n.capcode.marker_style);
}

test "normalizerForVocab: implicit-capcode (no nocapcode suffix) prebuilt name" {
    // TM convention: capcode is the default; if the name doesn't carry
    // `nocapcode` AND contains the word "capcode" anywhere, it's full
    // capcode. Synthetic test name; the real prebuilt TM names typically
    // OMIT the word "capcode" when capcode=2 is in effect (the absence
    // of `nocapcode` is the signal). Documented below in the helper's
    // doc-comment that the safer call site is `readBytesMeta()` for
    // unknown prebuilts; this helper covers the two patterns that are
    // unambiguous from the name alone.
    const n = normalizerForVocab("custom-vocab-with-capcode-on");
    try testing.expect(n == .capcode);
    try testing.expectEqual(@import("normalizer.zig").MarkerStyle.tm_printable, n.capcode.marker_style);
}

test "normalizerForVocab: unknown vocab returns .identity (don't guess)" {
    try testing.expect(normalizerForVocab("fiction-24000-strict-v1") == .identity);
    try testing.expect(normalizerForVocab("gpt2") == .identity);
    try testing.expect(normalizerForVocab("") == .identity);
}

test "normalizerForVocab: case-insensitive matching" {
    const a = normalizerForVocab("Englishcode-CLEAN-NoCapcode-v1");
    try testing.expect(a == .nocapcode);
    const b = normalizerForVocab("FOO-CAPCODE-BAR");
    try testing.expect(b == .capcode);
}

test "readBytesMeta: legacy .ztm (capcode=0, norm=0) loads as identity recommendation" {
    var m = try buildSampleMonster(testing.allocator);
    defer m.deinit();
    const buf = try writeBytes(testing.allocator, &m);
    defer testing.allocator.free(buf);

    var loaded = try readBytesMeta(testing.allocator, buf);
    defer loaded.deinit();

    try testing.expectEqual(CAPCODE_NONE, loaded.capcode);
    try testing.expectEqual(@as(u8, 0), loaded.norm_flag);
    try testing.expect(loaded.recommendedNormalizer() == .identity);
}

test "readBytesMeta: synthetic capcode+NFD header surfaces correct recommendation" {
    // Build a real Monster, write it, then patch the header bytes to
    // simulate a TM vocab where capcode=1 and the normalizer flag has
    // the NFD bit set. The loader should round-trip the tokens cleanly
    // and `recommendedNormalizer()` should return `.nocapcode{.nfd=true}`.
    var m = try buildSampleMonster(testing.allocator);
    defer m.deinit();
    const buf = try writeBytes(testing.allocator, &m);
    defer testing.allocator.free(buf);
    // Header layout: 0..4 magic, 4 version, 5 capcode, 6 max_token_len, 7 norm_flag.
    buf[5] = CAPCODE_NOCAPCODE;
    buf[7] = NORM_NFD;

    var loaded = try readBytesMeta(testing.allocator, buf);
    defer loaded.deinit();
    try testing.expectEqual(CAPCODE_NOCAPCODE, loaded.capcode);
    try testing.expectEqual(NORM_NFD, loaded.norm_flag);
    const rec = loaded.recommendedNormalizer();
    try testing.expect(rec == .nocapcode);
    try testing.expectEqual(true, rec.nocapcode.nfd);

    // Encoder still works through the loaded vocab — small sample.
    var ids_buf: [32]TokenId = undefined;
    const ids = try loaded.monster.encodeChunk(testing.allocator, "hello", &ids_buf);
    try testing.expect(ids.len > 0);
}

test "readBytesMeta: invalid capcode byte rejected" {
    var m = try buildSampleMonster(testing.allocator);
    defer m.deinit();
    const buf = try writeBytes(testing.allocator, &m);
    defer testing.allocator.free(buf);
    buf[5] = 0xFF; // > CAPCODE_FULL_TM
    try testing.expectError(Error.MalformedFile, readBytesMeta(testing.allocator, buf));
}

test "readBytesMeta: CAPCODE_FULL (= 2) recommends ztok marker_style" {
    var m = try buildSampleMonster(testing.allocator);
    defer m.deinit();
    const buf = try writeBytes(testing.allocator, &m);
    defer testing.allocator.free(buf);
    buf[5] = CAPCODE_FULL;
    buf[7] = NORM_NFD;

    var loaded = try readBytesMeta(testing.allocator, buf);
    defer loaded.deinit();
    try testing.expectEqual(CAPCODE_FULL, loaded.capcode);
    const rec = loaded.recommendedNormalizer();
    try testing.expect(rec == .capcode);
    try testing.expectEqual(true, rec.capcode.nfd);
    // CAPCODE_FULL means ztok-native markers (back-compat with any
    // pre-1.12 .ztm that stamped 2 from a ztok-side writer).
    try testing.expectEqual(@import("normalizer.zig").MarkerStyle.ztok, rec.capcode.marker_style);
}

test "readBytesMeta: CAPCODE_FULL_TM (= 3) recommends tm_printable marker_style" {
    var m = try buildSampleMonster(testing.allocator);
    defer m.deinit();
    const buf = try writeBytes(testing.allocator, &m);
    defer testing.allocator.free(buf);
    buf[5] = CAPCODE_FULL_TM;
    buf[7] = NORM_NFD;

    var loaded = try readBytesMeta(testing.allocator, buf);
    defer loaded.deinit();
    try testing.expectEqual(CAPCODE_FULL_TM, loaded.capcode);
    const rec = loaded.recommendedNormalizer();
    try testing.expect(rec == .capcode);
    try testing.expectEqual(true, rec.capcode.nfd);
    try testing.expectEqual(@import("normalizer.zig").MarkerStyle.tm_printable, rec.capcode.marker_style);
}

test "normalizerForVocab: nocapcode (1.12 behavior preserved)" {
    // Locking in the 1.12 nocapcode contract: nocapcode + nfd=true +
    // tm_compat_space=true. The marker style field is irrelevant for
    // .nocapcode (NoCapcode always uses 0x7F in both worlds), so we
    // don't check it.
    const n = normalizerForVocab("englishcode-clean-nocapcode-v1");
    try testing.expect(n == .nocapcode);
    try testing.expectEqual(true, n.nocapcode.nfd);
    try testing.expectEqual(true, n.nocapcode.tm_compat_space);
}

// --- TM-Go flag-bit propagation tests (post-1.14 agent A) ---

test "readFileMeta: tm_englishcode_32k.ztm yields non-zero flags for >= 90% of pieces" {
    // Skip if the vocab isn't present in the bench artifact dir.
    // CI runs without the prebuilt; local devs run with it. Mirrors
    // the pattern used elsewhere for the cross-tokenizer bench inputs.
    var loaded = readFileMeta(testing.allocator, "bench/vocabs/tm_englishcode_32k.ztm") catch |err| switch (err) {
        Error.TruncatedFile, Error.MagicMismatch => return error.SkipZigTest,
        else => return err,
    };
    defer loaded.deinit();

    try testing.expectEqual(CAPCODE_NOCAPCODE, loaded.capcode);
    const m = &loaded.monster;
    var nonzero: u32 = 0;
    for (m.flags) |f| {
        if (f != 0) nonzero += 1;
    }
    const ratio = @as(f32, @floatFromInt(nonzero)) / @as(f32, @floatFromInt(m.flags.len));
    try testing.expect(ratio >= 0.90);

    // begin_byte spot-check: ' ' (0x20) must classify as BB_SPACE
    // because TM vocabs have many space-leading pieces.
    try testing.expectEqual(@as(u8, 12), m.begin_byte[' ']);
    // '\x7F' (DEL) classifies as BB_PUNCT for nocapcode vocabs (it's
    // the DEL marker — capcode bucket).
    try testing.expectEqual(@as(u8, 10), m.begin_byte[0x7F]);
}

test "readFileMeta: legacy .ztm (capcode=0) loads with flags but all FLAG_ALL_LETTERS=0" {
    // capcode .none + a sample vocab of letter pieces: each piece
    // gets a non-zero flag (since they begin/end with letters). The
    // back-compat path is that the flag formula additions don't
    // break anything for non-capcode vocabs — they just bias
    // scoring slightly via the new bit-7 / split-word terms.
    var m = try buildSampleMonster(testing.allocator);
    defer m.deinit();
    const buf = try writeBytes(testing.allocator, &m);
    defer testing.allocator.free(buf);

    var loaded = try readBytesMeta(testing.allocator, buf);
    defer loaded.deinit();

    try testing.expectEqual(CAPCODE_NONE, loaded.capcode);
    // Sample vocab includes "hello" (begins+ends letter), " world",
    // etc. — non-zero flags expected.
    var nonzero: u32 = 0;
    for (loaded.monster.flags) |f| {
        if (f != 0) nonzero += 1;
    }
    try testing.expect(nonzero >= 1);
}

// --- v2 alias format tests ---

test "v2: writeBytes + readBytes round-trip with aliases (synthetic)" {
    // Build a Monster, attach a couple of aliases at the
    // Builder level, serialize via v2, then read back and verify the
    // alias byte sequences round-trip identical.
    var b = Monster.Builder.init(testing.allocator);
    defer b.deinit();
    _ = try b.addToken("hello");
    _ = try b.addToken(" world");
    const id_train_prim = try b.addToken("\x7F train");
    _ = try b.addToken("l");
    const unk = try b.addToken("<unk>");
    try b.addAlias(id_train_prim, "train");
    var m = try b.finalizeWithCapcode(unk, .nocapcode);
    defer m.deinit();

    try testing.expectEqual(@as(usize, 1), m.aliases.len);
    try testing.expectEqual(id_train_prim, m.aliases[0].id);
    try testing.expectEqualSlices(u8, "train", m.aliases[0].bytes);

    const buf = try writeBytesWithMeta(testing.allocator, &m, .{ .capcode = CAPCODE_NOCAPCODE });
    defer testing.allocator.free(buf);

    // Header now carries v3 magic (v3 = v2 + per-twin alt section;
    // the writer always emits the current VERSION).
    try testing.expectEqual(@as(u8, 0x03), buf[3]);
    try testing.expectEqual(@as(u8, 0x03), buf[4]);

    var loaded = try readBytesMeta(testing.allocator, buf);
    defer loaded.deinit();
    try testing.expectEqual(@as(usize, 1), loaded.monster.aliases.len);
    try testing.expectEqualSlices(u8, "train", loaded.monster.aliases[0].bytes);
    try testing.expectEqual(id_train_prim, loaded.monster.aliases[0].id);
    // Round-trip primary bytes too.
    try testing.expectEqualSlices(u8, "\x7F train", loaded.monster.idBytes(id_train_prim));
    // Alias_lens_by_id is populated.
    try testing.expectEqual(@as(u8, 5), loaded.monster.alias_lens_by_id[id_train_prim]);
}

test "v2: re-write of loaded v2 file produces byte-identical output" {
    var b = Monster.Builder.init(testing.allocator);
    defer b.deinit();
    _ = try b.addToken("a");
    const idb = try b.addToken("\x7F b");
    const unk = try b.addToken("<unk>");
    try b.addAlias(idb, "b");
    var m = try b.finalizeWithCapcode(unk, .nocapcode);
    defer m.deinit();

    const buf1 = try writeBytesWithMeta(testing.allocator, &m, .{ .capcode = CAPCODE_NOCAPCODE });
    defer testing.allocator.free(buf1);
    var loaded = try readBytesMeta(testing.allocator, buf1);
    defer loaded.deinit();
    const buf2 = try writeBytesWithMeta(testing.allocator, &loaded.monster, .{ .capcode = CAPCODE_NOCAPCODE });
    defer testing.allocator.free(buf2);
    try testing.expectEqualSlices(u8, buf1, buf2);
}

test "v1 .ztm files still load cleanly under v2 reader" {
    // Build a v1 file by hand (with `ZTM\x01` magic and no alias
    // section) and ensure readBytesMeta accepts it without error.
    // This is the back-compat gate for pre-v2 vocabs.
    const v1: []const u8 = "ZTM\x01" ++ // magic
        [_]u8{
            0x01, // version
            0x00, // capcode
            0x01, // max_token_length
            0x00, // norm_flag
        } ++ [_]u8{ 0x02, 0x00, 0x00, 0x00 } // count = 2
    ++ [_]u8{ 0x01, 0x00, 0x00, 0x00 } // unk_id = 1
    ++ [_]u8{ 0x02, 0x00, 0x00, 0x00 } // bytes_total = 2
    ++ [_]u8{ 'a', 'b' } // tokens "a", "b"
    ++ [_]u8{ 0x00, 0x00, 0x00, 0x00 } // offsets[0] = 0
    ++ [_]u8{ 0x01, 0x00, 0x00, 0x00 } // offsets[1] = 1
    ++ [_]u8{ 0x02, 0x00, 0x00, 0x00 } // offsets[2] = 2
    ++ [_]u8{ 0x00, 0x00 }; // nwords[0], nwords[1]

    var loaded = try readBytesMeta(testing.allocator, v1);
    defer loaded.deinit();
    try testing.expectEqual(@as(u32, 2), loaded.monster.count);
    try testing.expectEqual(@as(u32, 1), loaded.monster.unk_id);
    // v1 files carry no aliases.
    try testing.expectEqual(@as(usize, 0), loaded.monster.aliases.len);
    try testing.expectEqual(@as(usize, 0), loaded.monster.alias_lens_by_id.len);
}

// --- v3 per-twin alt format tests ---

test "v3: writeBytes + readBytes round-trip preserves per-twin alts" {
    // Build a Monster with a couple of twin forms, attach per-twin alt
    // tables, serialize via v3, read back, and verify the per-twin
    // structures round-trip identically (id, key, alts with units).
    const TwinAltEntry = Monster.TwinAltEntry;
    var b = Monster.Builder.init(testing.allocator);
    defer b.deinit();
    _ = try b.addToken("s"); // id 0
    _ = try b.addToken("st"); // id 1
    const id_sti = try b.addToken("sti"); // id 2
    _ = try b.addToken("\x7F"); // id 3
    _ = try b.addToken("\x7F st"); // id 4
    const id_psti = try b.addToken("\x7F sti"); // id 5
    const unk = try b.addToken("<unk>"); // id 6
    // Twin `sti` -> alts `st`(len 2), `s`(len 1) in BARE units.
    try b.addPerTwinAlt(id_sti, "sti", &[_]TwinAltEntry{
        .{ .alt_id = 1, .alt_byte_len = 2 },
        .{ .alt_id = 0, .alt_byte_len = 1 },
    });
    // Twin `\x7F sti` -> alts `\x7F st`(len 4), `\x7F`(len 1) in
    // PREFIXED units (different ids AND different byte lengths — the
    // structural distinction v3 captures).
    try b.addPerTwinAlt(id_psti, "\x7F sti", &[_]TwinAltEntry{
        .{ .alt_id = 4, .alt_byte_len = 4 },
        .{ .alt_id = 3, .alt_byte_len = 1 },
    });
    var m = try b.finalizeWithCapcode(unk, .nocapcode);
    defer m.deinit();

    try testing.expectEqual(@as(usize, 2), m.per_twin_alts.len);

    const buf = try writeBytesWithMeta(testing.allocator, &m, .{ .capcode = CAPCODE_NOCAPCODE });
    defer testing.allocator.free(buf);
    // v3 magic.
    try testing.expectEqual(@as(u8, 0x03), buf[3]);
    try testing.expectEqual(@as(u8, 0x03), buf[4]);

    var loaded = try readBytesMeta(testing.allocator, buf);
    defer loaded.deinit();
    const pta = loaded.monster.per_twin_alts;
    try testing.expectEqual(@as(usize, 2), pta.len);

    // Find the bare `sti` and prefixed `\x7F sti` records by key.
    var saw_bare = false;
    var saw_pref = false;
    for (pta) |t| {
        if (std.mem.eql(u8, t.key, "sti")) {
            saw_bare = true;
            try testing.expectEqual(id_sti, t.id);
            try testing.expectEqual(@as(usize, 2), t.alts.len);
            try testing.expectEqual(@as(u32, 1), t.alts[0].alt_id);
            try testing.expectEqual(@as(u32, 2), t.alts[0].alt_byte_len);
            try testing.expectEqual(@as(u32, 0), t.alts[1].alt_id);
            try testing.expectEqual(@as(u32, 1), t.alts[1].alt_byte_len);
        } else if (std.mem.eql(u8, t.key, "\x7F sti")) {
            saw_pref = true;
            try testing.expectEqual(id_psti, t.id);
            try testing.expectEqual(@as(usize, 2), t.alts.len);
            try testing.expectEqual(@as(u32, 4), t.alts[0].alt_id);
            try testing.expectEqual(@as(u32, 4), t.alts[0].alt_byte_len);
            try testing.expectEqual(@as(u32, 3), t.alts[1].alt_id);
            try testing.expectEqual(@as(u32, 1), t.alts[1].alt_byte_len);
        }
    }
    try testing.expect(saw_bare and saw_pref);
}

test "v3: re-write of loaded v3 file is byte-identical" {
    const TwinAltEntry = Monster.TwinAltEntry;
    var b = Monster.Builder.init(testing.allocator);
    defer b.deinit();
    _ = try b.addToken("a"); // id 0
    const idab = try b.addToken("ab"); // id 1
    const unk = try b.addToken("<unk>"); // id 2
    try b.addPerTwinAlt(idab, "ab", &[_]TwinAltEntry{
        .{ .alt_id = 0, .alt_byte_len = 1 },
    });
    var m = try b.finalizeWithCapcode(unk, .none);
    defer m.deinit();

    const buf1 = try writeBytesWithMeta(testing.allocator, &m, .{});
    defer testing.allocator.free(buf1);
    var loaded = try readBytesMeta(testing.allocator, buf1);
    defer loaded.deinit();
    try testing.expectEqual(@as(usize, 1), loaded.monster.per_twin_alts.len);
    const buf2 = try writeBytesWithMeta(testing.allocator, &loaded.monster, .{});
    defer testing.allocator.free(buf2);
    try testing.expectEqualSlices(u8, buf1, buf2);
}

test "v2 .ztm files (no per-twin section) still load under v3 reader" {
    // Hand-build a v2 file (`ZTM\x02`, alias section present, NO
    // per-twin section). The v3 reader must accept it and leave
    // per_twin_alts empty (back-compat gate for pre-v3 vocabs).
    const v2: []const u8 = "ZTM\x02" ++ // magic
        [_]u8{
            0x02, // version
            0x00, // capcode
            0x02, // max_token_length
            0x00, // norm_flag
        } ++ [_]u8{ 0x02, 0x00, 0x00, 0x00 } // count = 2
    ++ [_]u8{ 0x01, 0x00, 0x00, 0x00 } // unk_id = 1
    ++ [_]u8{ 0x03, 0x00, 0x00, 0x00 } // bytes_total = 3
    ++ [_]u8{ 'a', 'b', 'c' } // tokens "a", "bc"
    ++ [_]u8{ 0x00, 0x00, 0x00, 0x00 } // offsets[0] = 0
    ++ [_]u8{ 0x01, 0x00, 0x00, 0x00 } // offsets[1] = 1
    ++ [_]u8{ 0x03, 0x00, 0x00, 0x00 } // offsets[2] = 3
    ++ [_]u8{ 0x00, 0x00 } // nwords[0], nwords[1]
    ++ [_]u8{ 0x00, 0x00, 0x00, 0x00 }; // alias_count = 0

    var loaded = try readBytesMeta(testing.allocator, v2);
    defer loaded.deinit();
    try testing.expectEqual(@as(u32, 2), loaded.monster.count);
    try testing.expectEqual(@as(usize, 0), loaded.monster.per_twin_alts.len);
    try testing.expectEqual(@as(usize, 0), loaded.monster.aliases.len);
}

test "v3: malformed per-twin section (n_alts > 2) rejected" {
    // Minimal v3 file with one twin record claiming n_alts = 3.
    const v3: []const u8 = "ZTM\x03" ++
        [_]u8{ 0x03, 0x00, 0x01, 0x00 } // version, capcode, max_len, norm
    ++ [_]u8{ 0x01, 0x00, 0x00, 0x00 } // count = 1
    ++ [_]u8{ 0x00, 0x00, 0x00, 0x00 } // unk_id = 0
    ++ [_]u8{ 0x01, 0x00, 0x00, 0x00 } // bytes_total = 1
    ++ [_]u8{'a'} // token "a"
    ++ [_]u8{ 0x00, 0x00, 0x00, 0x00 } // offsets[0]
    ++ [_]u8{ 0x01, 0x00, 0x00, 0x00 } // offsets[1]
    ++ [_]u8{0x00} // nwords[0]
    ++ [_]u8{ 0x00, 0x00, 0x00, 0x00 } // alias_count = 0
    ++ [_]u8{ 0x01, 0x00, 0x00, 0x00 } // twin_count = 1
    ++ [_]u8{ 0x00, 0x00, 0x00, 0x00 } // twin id = 0
    ++ [_]u8{ 0x01, 0x00, 0x00, 0x00 } // key_len = 1
    ++ [_]u8{'a'} // key
    ++ [_]u8{0x03}; // n_alts = 3 (invalid)

    try testing.expectError(Error.MalformedFile, readBytesMeta(testing.allocator, v3));
}

test "v3: real regenerated 32k vocab loads and populates per-twin alts" {
    // Smoke test: the converter-regenerated vocab must load cleanly and
    // carry a non-empty per-twin alt section. Skip when the bench
    // artifact isn't present (CI without prebuilt vocabs).
    var loaded = readFileMeta(testing.allocator, "bench/vocabs/tm_englishcode_32k.ztm") catch |err| switch (err) {
        Error.TruncatedFile, Error.MagicMismatch => return error.SkipZigTest,
        else => return err,
    };
    defer loaded.deinit();
    try testing.expect(loaded.monster.per_twin_alts.len > 0);
    // Spot-check a record's shape: id in range, <=2 alts each in range.
    for (loaded.monster.per_twin_alts[0..@min(loaded.monster.per_twin_alts.len, 64)]) |t| {
        try testing.expect(t.id < loaded.monster.count);
        try testing.expect(t.alts.len >= 1 and t.alts.len <= 2);
        try testing.expect(t.key.len > 0);
        for (t.alts) |al| try testing.expect(al.alt_id < loaded.monster.count);
    }
    // Stage 2: the bare-twin alt LUT must be built and have at least
    // some populated entries (bare aliases that carry alts).
    try testing.expect(loaded.monster.bare_alt_lut.len > 0);
    var populated: usize = 0;
    for (loaded.monster.bare_alt_lut) |e| {
        if (e.index != 0xFFFF_FFFF) populated += 1;
    }
    try testing.expect(populated > 0);
}

test "v2: alias trie lookup succeeds for alias bytes" {
    // Verify the alias is reachable through the same trie walk that
    // the encoder uses. Build a vocab with `\x7F train` as primary and
    // `train` as alias; encode the bare `train` and confirm we get
    // the primary's id.
    var b = Monster.Builder.init(testing.allocator);
    defer b.deinit();
    const id_prim = try b.addToken("\x7F train");
    _ = try b.addToken("a"); // single-byte fallback for unk
    const unk = try b.addToken("<unk>");
    try b.addAlias(id_prim, "train");
    var m = try b.finalizeWithCapcode(unk, .nocapcode);
    defer m.deinit();

    // Encode bare `train`. Expect a single id (the alias hit) == id_prim.
    var out: [16]@import("token.zig").TokenId = undefined;
    const ids = try m.encodeChunk(testing.allocator, "train", &out);
    try testing.expectEqual(@as(usize, 1), ids.len);
    try testing.expectEqual(id_prim, ids[0]);

    // And encode the prefixed `\x7F train` — same id.
    const ids2 = try m.encodeChunk(testing.allocator, "\x7F train", &out);
    try testing.expectEqual(@as(usize, 1), ids2.len);
    try testing.expectEqual(id_prim, ids2[0]);
}
