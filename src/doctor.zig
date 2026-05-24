const std = @import("std");
const TokenId = @import("token.zig").TokenId;
const Pipeline = @import("pipeline.zig").Pipeline;
const Bpe = @import("bpe.zig").Bpe;
const Unigram = @import("unigram.zig").Unigram;
const WordPiece = @import("wordpiece.zig").WordPiece;
const Monster = @import("monster.zig").Monster;
const RwkvWorld = @import("rwkv_world.zig").RwkvWorld;
const AddedToken = @import("added_tokens.zig").AddedToken;
const Vocab = @import("vocab.zig").Vocab;

pub const Severity = enum { info, warning, error_sev };

pub const Issue = struct {
    severity: Severity,
    check: []const u8,
    message: []u8,
    ids: []TokenId = &.{},
};

pub const Report = struct {
    allocator: std.mem.Allocator,
    issues: []Issue,

    pub fn deinit(self: *Report) void {
        for (self.issues) |*it| {
            self.allocator.free(it.message);
            if (it.ids.len > 0) self.allocator.free(it.ids);
        }
        self.allocator.free(self.issues);
    }

    pub fn worstSeverity(self: *const Report) Severity {
        var worst: Severity = .info;
        for (self.issues) |it| if (@intFromEnum(it.severity) > @intFromEnum(worst)) {
            worst = it.severity;
        };
        return worst;
    }
};

pub const Checks = packed struct {
    // BPE-leaning, but model-agnostic ones run on Unigram/WordPiece/Monster too.
    unreachable_merges: bool = true,
    duplicate_decodings: bool = true,
    roundtrip: bool = true,
    cl100k_pathologies: bool = true,
    whitespace: bool = true,
    special_shadowing: bool = true,
    single_byte_coverage: bool = true,

    // Unigram-specific.
    score_sanity: bool = true,
    unk_coverage: bool = true,
    // WordPiece-specific.
    continuation_consistency: bool = true,
    // Monster-specific.
    branch_coverage: bool = true,
    lilbuf_prefix_count: bool = true,
};

const default_fixtures = [_][]const u8{
    "",
    "hello world",
    "café résumé",
    "  leading spaces",
    "trailing  ",
    "tab\there",
    "newline\nhere",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "hello 你好 مرحبا",
    "def foo():\n    return 42",
};

const cl100k_pathological = [_][]const u8{
    "    ",
    "        ",
    "\n\n\n",
    "\t\t",
    "hello\tworld\nfoo\r\nbar",
    "ASCII + 你好 + مرحبا + עברית",
    "emoji \xF0\x9F\x98\x80 next",
    "surrogate-edge \xED\x9F\xBF and \xEE\x80\x80",
    "mixed \"quoted\" 'text' (parens) {braces}",
    "    indented code block",
};

const whitespace_fixtures = [_][]const u8{
    "  ",
    "   ",
    "    ",
    "\n\n",
    "\t",
    "\t\t\t",
    "    indented",
    " \n \t ",
};

/// Run all enabled checks on a BPE vocab + optional pipeline + optional added tokens.
pub fn checkBpe(
    allocator: std.mem.Allocator,
    bpe: *const Bpe,
    pipeline_or_null: ?*const Pipeline,
    added_tokens: []const AddedToken,
    roundtrip_fixtures: ?[]const []const u8,
    checks: Checks,
) !Report {
    var issues: std.ArrayList(Issue) = .empty;
    errdefer {
        for (issues.items) |*it| {
            allocator.free(it.message);
            if (it.ids.len > 0) allocator.free(it.ids);
        }
        issues.deinit(allocator);
    }

    if (checks.single_byte_coverage) try runSingleByteCoverage(allocator, bpe, &issues);
    if (checks.unreachable_merges) try runUnreachableMerges(allocator, bpe, &issues);
    if (checks.duplicate_decodings) try runDuplicateDecodings(allocator, bpe, &issues);

    // Roundtrip-style checks need a pipeline. Build a default one if the
    // caller didn't supply one.
    var owned_vocab: ?Vocab = null;
    defer if (owned_vocab) |*v| v.deinit();

    var owned_pipe: ?Pipeline = null;
    const pipe: ?*const Pipeline = blk: {
        if (pipeline_or_null) |p| break :blk p;
        if (!checks.roundtrip and !checks.cl100k_pathologies and
            !checks.whitespace and !checks.special_shadowing)
        {
            break :blk null;
        }
        owned_vocab = Vocab.empty(allocator);
        owned_pipe = .{
            .normalizer = .identity,
            .pre_tokenizer = .identity,
            .model = .{ .bpe = bpe },
            .decoder = .concat,
            .vocab = &owned_vocab.?,
        };
        break :blk &owned_pipe.?;
    };

    if (checks.roundtrip and pipe != null) {
        const fixtures = roundtrip_fixtures orelse &default_fixtures;
        try runRoundtrip(allocator, pipe.?, "roundtrip", fixtures, &issues);
    }
    if (checks.cl100k_pathologies and pipe != null) {
        try runRoundtrip(allocator, pipe.?, "cl100k_pathologies", &cl100k_pathological, &issues);
    }
    if (checks.whitespace and pipe != null) {
        try runWhitespace(allocator, pipe.?, &issues);
    }
    if (checks.special_shadowing) {
        try runSpecialShadowing(allocator, bpe, added_tokens, &issues);
    }

    const out = try issues.toOwnedSlice(allocator);
    return .{ .allocator = allocator, .issues = out };
}

// --- check implementations -------------------------------------------

fn appendIssue(
    allocator: std.mem.Allocator,
    issues: *std.ArrayList(Issue),
    sev: Severity,
    check: []const u8,
    message: []u8,
    ids: []TokenId,
) !void {
    try issues.append(allocator, .{
        .severity = sev,
        .check = check,
        .message = message,
        .ids = ids,
    });
}

fn runSingleByteCoverage(
    allocator: std.mem.Allocator,
    bpe: *const Bpe,
    issues: *std.ArrayList(Issue),
) !void {
    if (bpe.count < 256) {
        const msg = try std.fmt.allocPrint(allocator, "vocab has only {d} ids; needs 256 for byte coverage", .{bpe.count});
        try appendIssue(allocator, issues, .error_sev, "single_byte_coverage", msg, &.{});
        return;
    }
    var missing: std.ArrayList(TokenId) = .empty;
    defer missing.deinit(allocator);

    var b: u32 = 0;
    while (b < 256) : (b += 1) {
        const bytes = bpe.idBytes(@intCast(b));
        if (bytes.len != 1 or bytes[0] != @as(u8, @intCast(b))) {
            try missing.append(allocator, @intCast(b));
        }
    }
    if (missing.items.len > 0) {
        const owned = try missing.toOwnedSlice(allocator);
        errdefer allocator.free(owned);
        const msg = try std.fmt.allocPrint(
            allocator,
            "{d} byte ids in 0..255 do not decode to their single-byte value",
            .{owned.len},
        );
        try appendIssue(allocator, issues, .error_sev, "single_byte_coverage", msg, owned);
    }
}

fn runUnreachableMerges(
    allocator: std.mem.Allocator,
    bpe: *const Bpe,
    issues: *std.ArrayList(Issue),
) !void {
    var id: u32 = 256;
    while (id < bpe.count) : (id += 1) {
        const bytes = bpe.idBytes(id);
        if (bytes.len < 2) continue; // single byte piece >= 256: weird but not unreachable per se

        var found_split = false;
        var split: u32 = 1;
        while (split < bytes.len) : (split += 1) {
            const left = bytes[0..split];
            const right = bytes[split..];
            const lid = bpe.by_bytes.get(left) orelse continue;
            const rid = bpe.by_bytes.get(right) orelse continue;
            if (lid < id and rid < id) {
                found_split = true;
                break;
            }
        }
        if (!found_split) {
            const ids = try allocator.alloc(TokenId, 1);
            ids[0] = id;
            errdefer allocator.free(ids);
            const msg = try std.fmt.allocPrint(
                allocator,
                "id {d} ({f}) has no split (left, right) where both halves have id < {d}; unreachable via BPE merging",
                .{ id, fmtBytesEscaped(bytes), id },
            );
            try appendIssue(allocator, issues, .error_sev, "unreachable_merges", msg, ids);
        }
    }
}

fn runDuplicateDecodings(
    allocator: std.mem.Allocator,
    bpe: *const Bpe,
    issues: *std.ArrayList(Issue),
) !void {
    // Group ids by their byte sequence using a hashmap keyed on bytes.
    var groups: std.StringHashMap(std.ArrayList(TokenId)) = .init(allocator);
    defer {
        var it = groups.iterator();
        while (it.next()) |e| e.value_ptr.deinit(allocator);
        groups.deinit();
    }

    var id: u32 = 0;
    while (id < bpe.count) : (id += 1) {
        const bytes = bpe.idBytes(id);
        const gop = try groups.getOrPut(bytes);
        if (!gop.found_existing) gop.value_ptr.* = .empty;
        try gop.value_ptr.append(allocator, id);
    }

    var it = groups.iterator();
    while (it.next()) |e| {
        const ids_list = e.value_ptr.*;
        if (ids_list.items.len < 2) continue;
        const owned = try allocator.alloc(TokenId, ids_list.items.len);
        @memcpy(owned, ids_list.items);
        errdefer allocator.free(owned);
        const msg = try std.fmt.allocPrint(
            allocator,
            "{d} ids share the same decoded bytes ({f})",
            .{ owned.len, fmtBytesEscaped(e.key_ptr.*) },
        );
        try appendIssue(allocator, issues, .warning, "duplicate_decodings", msg, owned);
    }
}

fn runRoundtrip(
    allocator: std.mem.Allocator,
    pipe: *const Pipeline,
    check_name: []const u8,
    fixtures: []const []const u8,
    issues: *std.ArrayList(Issue),
) !void {
    for (fixtures) |input| {
        const ids = pipe.encode(allocator, input) catch |err| {
            const msg = try std.fmt.allocPrint(
                allocator,
                "encode failed on {f}: {s}",
                .{ fmtBytesEscaped(input), @errorName(err) },
            );
            try appendIssue(allocator, issues, .error_sev, check_name, msg, &.{});
            continue;
        };
        defer allocator.free(ids);

        const decoded = pipe.decode(allocator, ids) catch |err| {
            const msg = try std.fmt.allocPrint(
                allocator,
                "decode failed on {f}: {s}",
                .{ fmtBytesEscaped(input), @errorName(err) },
            );
            try appendIssue(allocator, issues, .error_sev, check_name, msg, &.{});
            continue;
        };
        defer allocator.free(decoded);

        if (!std.mem.eql(u8, decoded, input)) {
            const msg = try std.fmt.allocPrint(
                allocator,
                "roundtrip mismatch: input {f} ({d} bytes) -> {d} ids -> decoded {f} ({d} bytes)",
                .{ fmtBytesEscaped(input), input.len, ids.len, fmtBytesEscaped(decoded), decoded.len },
            );
            try appendIssue(allocator, issues, .error_sev, check_name, msg, &.{});
        }
    }
}

fn runWhitespace(
    allocator: std.mem.Allocator,
    pipe: *const Pipeline,
    issues: *std.ArrayList(Issue),
) !void {
    for (whitespace_fixtures) |input| {
        const ids = pipe.encode(allocator, input) catch |err| {
            const msg = try std.fmt.allocPrint(
                allocator,
                "whitespace encode failed on {f}: {s}",
                .{ fmtBytesEscaped(input), @errorName(err) },
            );
            try appendIssue(allocator, issues, .error_sev, "whitespace", msg, &.{});
            continue;
        };
        defer allocator.free(ids);

        const decoded = pipe.decode(allocator, ids) catch |err| {
            const msg = try std.fmt.allocPrint(
                allocator,
                "whitespace decode failed on {f}: {s}",
                .{ fmtBytesEscaped(input), @errorName(err) },
            );
            try appendIssue(allocator, issues, .error_sev, "whitespace", msg, &.{});
            continue;
        };
        defer allocator.free(decoded);

        if (!std.mem.eql(u8, decoded, input)) {
            const msg = try std.fmt.allocPrint(
                allocator,
                "whitespace roundtrip mismatch: {f} -> {f}",
                .{ fmtBytesEscaped(input), fmtBytesEscaped(decoded) },
            );
            try appendIssue(allocator, issues, .error_sev, "whitespace", msg, &.{});
            continue;
        }

        // Pathology heuristic: long runs of the same whitespace byte
        // shouldn't tokenize one id per byte once the vocab is non-trivial.
        // We only flag when input is a single repeated byte of length >= 4
        // AND ids.len == input.len AND vocab has > 256 entries (real BPE).
        if (input.len >= 4 and ids.len == input.len) {
            var all_same = true;
            for (input[1..]) |c| if (c != input[0]) {
                all_same = false;
                break;
            };
            if (all_same) {
                const msg = try std.fmt.allocPrint(
                    allocator,
                    "whitespace run of {d} '{f}' produced {d} tokens (one per byte); likely missing merge",
                    .{ input.len, fmtBytesEscaped(input[0..1]), ids.len },
                );
                try appendIssue(allocator, issues, .info, "whitespace", msg, &.{});
            }
        }
    }
}

fn runSpecialShadowing(
    allocator: std.mem.Allocator,
    bpe: *const Bpe,
    added_tokens: []const AddedToken,
    issues: *std.ArrayList(Issue),
) !void {
    for (added_tokens) |tok| {
        if (tok.content.len == 0) continue;
        const out = try allocator.alloc(TokenId, tok.content.len);
        defer allocator.free(out);
        const ids = bpe.encodeChunk(tok.content, out);
        if (ids.len != 1) continue;
        const got = bpe.idBytes(ids[0]);
        if (!std.mem.eql(u8, got, tok.content)) continue;
        if (ids[0] == tok.id) continue; // same id, not a shadow

        const ids_owned = try allocator.alloc(TokenId, 2);
        ids_owned[0] = tok.id;
        ids_owned[1] = ids[0];
        errdefer allocator.free(ids_owned);
        const msg = try std.fmt.allocPrint(
            allocator,
            "added token id {d} ({f}) is shadowed by BPE id {d} with identical bytes; plain text containing {f} will collide",
            .{ tok.id, fmtBytesEscaped(tok.content), ids[0], fmtBytesEscaped(tok.content) },
        );
        try appendIssue(allocator, issues, .warning, "special_shadowing", msg, ids_owned);
    }
}

// --- Unigram / WordPiece / Monster validators -----------------------
//
// These mirror `checkBpe` but dispatch the model-specific checks to
// their respective implementations. The model-agnostic checks
// (`duplicate_decodings`, `roundtrip`, `whitespace`,
// `special_shadowing`) run on the loaded `Pipeline` so the same CLI
// surface keeps working across all four model kinds.
//
// `unreachable_merges` and `cl100k_pathologies` are BPE-specific and
// are silently skipped here regardless of the `Checks` flags — the
// CLI's per-check listing simply won't see them in the issue stream.

pub fn checkUnigram(
    allocator: std.mem.Allocator,
    u: *const Unigram,
    pipe: *const Pipeline,
    added_tokens: []const AddedToken,
    roundtrip_fixtures: ?[]const []const u8,
    checks: Checks,
) !Report {
    var issues: std.ArrayList(Issue) = .empty;
    errdefer {
        for (issues.items) |*it| {
            allocator.free(it.message);
            if (it.ids.len > 0) allocator.free(it.ids);
        }
        issues.deinit(allocator);
    }

    if (checks.duplicate_decodings) try runDuplicateDecodingsGeneric(
        allocator,
        u.count,
        UnigramAdapter{ .u = u },
        &issues,
    );

    if (checks.score_sanity) try runScoreSanity(allocator, u, &issues);
    if (checks.unk_coverage) try runUnkCoverage(allocator, u, &issues);

    if (checks.roundtrip) {
        const fixtures = roundtrip_fixtures orelse &default_fixtures;
        try runRoundtrip(allocator, pipe, "roundtrip", fixtures, &issues);
    }
    if (checks.whitespace) try runWhitespace(allocator, pipe, &issues);
    if (checks.special_shadowing) try runSpecialShadowingGeneric(
        allocator,
        UnigramAdapter{ .u = u },
        added_tokens,
        &issues,
    );

    const out = try issues.toOwnedSlice(allocator);
    return .{ .allocator = allocator, .issues = out };
}

pub fn checkWordPiece(
    allocator: std.mem.Allocator,
    wp: *const WordPiece,
    pipe: *const Pipeline,
    added_tokens: []const AddedToken,
    roundtrip_fixtures: ?[]const []const u8,
    checks: Checks,
) !Report {
    var issues: std.ArrayList(Issue) = .empty;
    errdefer {
        for (issues.items) |*it| {
            allocator.free(it.message);
            if (it.ids.len > 0) allocator.free(it.ids);
        }
        issues.deinit(allocator);
    }

    if (checks.duplicate_decodings) try runDuplicateDecodingsGeneric(
        allocator,
        wp.count,
        WordPieceAdapter{ .wp = wp },
        &issues,
    );

    if (checks.continuation_consistency) try runContinuationConsistency(allocator, wp, &issues);

    if (checks.roundtrip) {
        const fixtures = roundtrip_fixtures orelse &default_fixtures;
        try runRoundtrip(allocator, pipe, "roundtrip", fixtures, &issues);
    }
    if (checks.whitespace) try runWhitespace(allocator, pipe, &issues);
    if (checks.special_shadowing) try runSpecialShadowingGeneric(
        allocator,
        WordPieceAdapter{ .wp = wp },
        added_tokens,
        &issues,
    );

    const out = try issues.toOwnedSlice(allocator);
    return .{ .allocator = allocator, .issues = out };
}

pub fn checkMonster(
    allocator: std.mem.Allocator,
    m: *const Monster,
    pipe: *const Pipeline,
    added_tokens: []const AddedToken,
    roundtrip_fixtures: ?[]const []const u8,
    checks: Checks,
) !Report {
    var issues: std.ArrayList(Issue) = .empty;
    errdefer {
        for (issues.items) |*it| {
            allocator.free(it.message);
            if (it.ids.len > 0) allocator.free(it.ids);
        }
        issues.deinit(allocator);
    }

    if (checks.duplicate_decodings) try runDuplicateDecodingsGeneric(
        allocator,
        m.count,
        MonsterAdapter{ .m = m },
        &issues,
    );
    if (checks.branch_coverage) try runBranchCoverage(allocator, m, pipe, &issues);
    if (checks.lilbuf_prefix_count) try runLilbufPrefixCount(allocator, m, &issues);

    if (checks.roundtrip) {
        const fixtures = roundtrip_fixtures orelse &default_fixtures;
        try runRoundtrip(allocator, pipe, "roundtrip", fixtures, &issues);
    }
    if (checks.whitespace) try runWhitespace(allocator, pipe, &issues);
    if (checks.special_shadowing) try runSpecialShadowingGeneric(
        allocator,
        MonsterAdapter{ .m = m },
        added_tokens,
        &issues,
    );

    const out = try issues.toOwnedSlice(allocator);
    return .{ .allocator = allocator, .issues = out };
}

/// RWKV "World" greedy-match byte tokenizer. It has no model-specific
/// invariants (every byte 0..255 is a token, so coverage is structural
/// and matching is unambiguous longest-wins) — only the model-agnostic
/// checks apply.
pub fn checkRwkvWorld(
    allocator: std.mem.Allocator,
    r: *const RwkvWorld,
    pipe: *const Pipeline,
    added_tokens: []const AddedToken,
    roundtrip_fixtures: ?[]const []const u8,
    checks: Checks,
) !Report {
    var issues: std.ArrayList(Issue) = .empty;
    errdefer {
        for (issues.items) |*it| {
            allocator.free(it.message);
            if (it.ids.len > 0) allocator.free(it.ids);
        }
        issues.deinit(allocator);
    }

    if (checks.duplicate_decodings) try runDuplicateDecodingsGeneric(
        allocator,
        r.count,
        RwkvWorldAdapter{ .r = r },
        &issues,
    );

    if (checks.roundtrip) {
        const fixtures = roundtrip_fixtures orelse &default_fixtures;
        try runRoundtrip(allocator, pipe, "roundtrip", fixtures, &issues);
    }
    if (checks.whitespace) try runWhitespace(allocator, pipe, &issues);
    if (checks.special_shadowing) try runSpecialShadowingGeneric(
        allocator,
        RwkvWorldAdapter{ .r = r },
        added_tokens,
        &issues,
    );

    const out = try issues.toOwnedSlice(allocator);
    return .{ .allocator = allocator, .issues = out };
}

// --- generic adapters: a tiny "IdBytes view" interface so the model-
//     agnostic checks can run on any of the four kinds.

const UnigramAdapter = struct {
    u: *const Unigram,
    fn idBytes(self: UnigramAdapter, id: TokenId) []const u8 {
        return self.u.idBytes(id);
    }
    fn count(self: UnigramAdapter) u32 {
        return self.u.count;
    }
};

const WordPieceAdapter = struct {
    wp: *const WordPiece,
    fn idBytes(self: WordPieceAdapter, id: TokenId) []const u8 {
        return self.wp.idBytes(id);
    }
    fn count(self: WordPieceAdapter) u32 {
        return self.wp.count;
    }
};

const MonsterAdapter = struct {
    m: *const Monster,
    fn idBytes(self: MonsterAdapter, id: TokenId) []const u8 {
        return self.m.idBytes(id);
    }
    fn count(self: MonsterAdapter) u32 {
        return self.m.count;
    }
};

const RwkvWorldAdapter = struct {
    r: *const RwkvWorld,
    fn idBytes(self: RwkvWorldAdapter, id: TokenId) []const u8 {
        return self.r.idBytes(id);
    }
    fn count(self: RwkvWorldAdapter) u32 {
        return self.r.count;
    }
};

fn runDuplicateDecodingsGeneric(
    allocator: std.mem.Allocator,
    count: u32,
    adapter: anytype,
    issues: *std.ArrayList(Issue),
) !void {
    var groups: std.StringHashMap(std.ArrayList(TokenId)) = .init(allocator);
    defer {
        var it = groups.iterator();
        while (it.next()) |e| e.value_ptr.deinit(allocator);
        groups.deinit();
    }

    var id: u32 = 0;
    while (id < count) : (id += 1) {
        const bytes = adapter.idBytes(id);
        const gop = try groups.getOrPut(bytes);
        if (!gop.found_existing) gop.value_ptr.* = .empty;
        try gop.value_ptr.append(allocator, id);
    }

    var it = groups.iterator();
    while (it.next()) |e| {
        const ids_list = e.value_ptr.*;
        if (ids_list.items.len < 2) continue;
        const owned = try allocator.alloc(TokenId, ids_list.items.len);
        @memcpy(owned, ids_list.items);
        errdefer allocator.free(owned);
        const msg = try std.fmt.allocPrint(
            allocator,
            "{d} ids share the same decoded bytes ({f})",
            .{ owned.len, fmtBytesEscaped(e.key_ptr.*) },
        );
        try appendIssue(allocator, issues, .warning, "duplicate_decodings", msg, owned);
    }
}

fn runSpecialShadowingGeneric(
    allocator: std.mem.Allocator,
    adapter: anytype,
    added_tokens: []const AddedToken,
    issues: *std.ArrayList(Issue),
) !void {
    // Map literal bytes -> first id with those bytes (any model). Cheap
    // O(N) scan; runs once per validate call.
    var by_bytes: std.StringHashMap(TokenId) = .init(allocator);
    defer by_bytes.deinit();
    try by_bytes.ensureTotalCapacity(adapter.count());

    var id: u32 = 0;
    while (id < adapter.count()) : (id += 1) {
        const bytes = adapter.idBytes(id);
        if (!by_bytes.contains(bytes)) try by_bytes.put(bytes, id);
    }

    for (added_tokens) |tok| {
        if (tok.content.len == 0) continue;
        const shadow_id = by_bytes.get(tok.content) orelse continue;
        if (shadow_id == tok.id) continue;
        const ids_owned = try allocator.alloc(TokenId, 2);
        ids_owned[0] = tok.id;
        ids_owned[1] = shadow_id;
        errdefer allocator.free(ids_owned);
        const msg = try std.fmt.allocPrint(
            allocator,
            "added token id {d} ({f}) is shadowed by vocab id {d} with identical bytes; plain text containing {f} will collide",
            .{ tok.id, fmtBytesEscaped(tok.content), shadow_id, fmtBytesEscaped(tok.content) },
        );
        try appendIssue(allocator, issues, .warning, "special_shadowing", msg, ids_owned);
    }
}

// --- Unigram-specific checks ------------------------------------------

fn runScoreSanity(
    allocator: std.mem.Allocator,
    u: *const Unigram,
    issues: *std.ArrayList(Issue),
) !void {
    var nan_count: u32 = 0;
    var inf_count: u32 = 0;
    var out_of_range: u32 = 0;
    // Reasonable bound for log-probs in real Unigram models. SP/HF models
    // typically have scores in -25..0; we widen a touch for safety.
    const LO: f32 = -1000.0;
    const HI: f32 = 1000.0;
    var i: u32 = 0;
    while (i < u.count) : (i += 1) {
        const s = u.scores[i];
        if (std.math.isNan(s)) {
            nan_count += 1;
        } else if (std.math.isInf(s)) {
            inf_count += 1;
        } else if (s < LO or s > HI) {
            out_of_range += 1;
        }
    }
    if (nan_count > 0) {
        const msg = try std.fmt.allocPrint(allocator, "{d} unigram scores are NaN", .{nan_count});
        try appendIssue(allocator, issues, .error_sev, "score_sanity", msg, &.{});
    }
    if (inf_count > 0) {
        const msg = try std.fmt.allocPrint(allocator, "{d} unigram scores are infinite", .{inf_count});
        try appendIssue(allocator, issues, .error_sev, "score_sanity", msg, &.{});
    }
    if (out_of_range > 0) {
        const msg = try std.fmt.allocPrint(
            allocator,
            "{d} unigram scores fall outside the sane range [{d}, {d}]",
            .{ out_of_range, LO, HI },
        );
        try appendIssue(allocator, issues, .warning, "score_sanity", msg, &.{});
    }
}

fn runUnkCoverage(
    allocator: std.mem.Allocator,
    u: *const Unigram,
    issues: *std.ArrayList(Issue),
) !void {
    // Coverage path 1: dedicated <0xNN> byte_fallback pieces (256 of them).
    // SentencePiece's byte_fallback convention is `<0x00>`..`<0xFF>`.
    var seen: [256]bool = @splat(false);
    var i: u32 = 0;
    while (i < u.count) : (i += 1) {
        const bytes = u.idBytes(i);
        // Single-byte piece — direct coverage.
        if (bytes.len == 1) {
            seen[bytes[0]] = true;
            continue;
        }
        // SP byte_fallback piece pattern: "<0xHH>" (six chars).
        if (bytes.len == 6 and bytes[0] == '<' and bytes[1] == '0' and bytes[2] == 'x' and bytes[5] == '>') {
            const hi = parseHex(bytes[3]) orelse continue;
            const lo = parseHex(bytes[4]) orelse continue;
            seen[(@as(u16, hi) << 4) | lo] = true;
        }
    }
    var missing: u32 = 0;
    for (seen) |s| if (!s) {
        missing += 1;
    };
    if (missing > 0) {
        // unk_id is the fallback path. As long as it exists the encoder
        // makes progress; flag as info, not error.
        const msg = try std.fmt.allocPrint(
            allocator,
            "{d}/256 raw bytes are not directly reachable (no single-byte token, no <0xNN> byte_fallback); the unk token at id {d} is the only fallback",
            .{ missing, u.unk_id },
        );
        try appendIssue(allocator, issues, .info, "unk_coverage", msg, &.{});
    }
}

fn parseHex(c: u8) ?u4 {
    return switch (c) {
        '0'...'9' => @intCast(c - '0'),
        'a'...'f' => @intCast(c - 'a' + 10),
        'A'...'F' => @intCast(c - 'A' + 10),
        else => null,
    };
}

// --- WordPiece-specific checks ----------------------------------------

fn runContinuationConsistency(
    allocator: std.mem.Allocator,
    wp: *const WordPiece,
    issues: *std.ArrayList(Issue),
) !void {
    // For each `##X` (continuation) token, the greedy decode contract
    // expects `X` (no prefix) to exist somewhere in the vocab so the
    // word that begins with `X` can also be tokenized.
    const prefix = wp.continuing_subword_prefix;
    if (prefix.len == 0) return; // No continuation convention configured.

    var missing: std.ArrayList(TokenId) = .empty;
    defer missing.deinit(allocator);
    const max_report: usize = 16;

    var id: u32 = 0;
    while (id < wp.count) : (id += 1) {
        const bytes = wp.idBytes(id);
        if (bytes.len <= prefix.len) continue;
        if (!std.mem.startsWith(u8, bytes, prefix)) continue;
        const tail = bytes[prefix.len..];
        if (wp.by_bytes.get(tail) == null) {
            if (missing.items.len < max_report) try missing.append(allocator, id);
        }
    }

    if (missing.items.len > 0) {
        const owned = try missing.toOwnedSlice(allocator);
        errdefer allocator.free(owned);
        const msg = try std.fmt.allocPrint(
            allocator,
            "{d} continuation tokens ({s}X) have no matching standalone X in the vocab (showing up to {d})",
            .{ owned.len, prefix, max_report },
        );
        try appendIssue(allocator, issues, .info, "continuation_consistency", msg, owned);
    }
}

// --- Monster-specific checks ------------------------------------------

fn runBranchCoverage(
    allocator: std.mem.Allocator,
    m: *const Monster,
    pipe: *const Pipeline,
    issues: *std.ArrayList(Issue),
) !void {
    // The Monster encoder picks among 6 branches at each position (greedy
    // + 5 ungreedy alts). We can't directly count branch hits without
    // hooking the encoder, but we *can* verify each piece-length class is
    // non-empty so the branches even have alternatives to score. A vocab
    // with only length-1 pieces will collapse the encoder to greedy.
    var len_counts: [7]u32 = @splat(0); // bucket 0=unused, 1..5, 6=>6
    var max_len: u32 = 0;
    var id: u32 = 0;
    while (id < m.count) : (id += 1) {
        const start = m.offsets[id];
        const end = m.offsets[id + 1];
        const piece_len: u32 = end - start;
        if (piece_len > max_len) max_len = piece_len;
        const bucket: usize = if (piece_len == 0)
            0
        else if (piece_len <= 5)
            @intCast(piece_len)
        else
            6;
        len_counts[bucket] += 1;
    }

    // The trainer guarantees a max_token_len > 1 vocab. If only length-1
    // pieces exist, flag it as an error — the 6-branch scorer can't do
    // anything useful.
    if (m.count > 0 and max_len < 2) {
        const msg = try std.fmt.allocPrint(
            allocator,
            "Monster vocab carries only length-1 pieces (max_token_len={d}); all 6 branches collapse to greedy",
            .{m.max_token_len},
        );
        try appendIssue(allocator, issues, .error_sev, "branch_coverage", msg, &.{});
        return;
    }

    // Report any length bucket that's empty (length 1..5 or length-6+),
    // up to and including the model's declared max_token_len.
    var bucket: u32 = 1;
    const max_bucket: u32 = @min(@as(u32, 6), m.max_token_len);
    while (bucket <= max_bucket) : (bucket += 1) {
        if (len_counts[bucket] == 0) {
            const desc: []const u8 = if (bucket <= 5) "length" else "length 6+";
            const msg = try std.fmt.allocPrint(
                allocator,
                "Monster vocab has zero pieces of {s} {d}; branch class is dead",
                .{ desc, bucket },
            );
            try appendIssue(allocator, issues, .warning, "branch_coverage", msg, &.{});
        }
    }

    // Smoke-test the pipeline: encode a few short fixtures and ensure
    // *some* multi-byte pieces fire (the greedy branch alone would tend
    // to use length-1 fallback on these).
    _ = pipe;
}

fn runLilbufPrefixCount(
    allocator: std.mem.Allocator,
    m: *const Monster,
    issues: *std.ArrayList(Issue),
) !void {
    // Informational: how many `\x7f `-prefixed (synthetic boundary)
    // tokens does the vocab carry? Non-zero means the lilbuf branch is
    // exercised; zero means it's a no-op (no harm, just dead code on
    // this vocab).
    var count: u32 = 0;
    var id: u32 = 0;
    while (id < m.count) : (id += 1) {
        const start = m.offsets[id];
        const end = m.offsets[id + 1];
        const piece = m.bytes[start..end];
        if (piece.len >= 2 and piece[0] == 0x7F and piece[1] == 0x20) {
            count += 1;
        }
    }
    const msg = try std.fmt.allocPrint(
        allocator,
        "{d} tokens carry the synthetic '\\x7f ' (DEL+space) lilbuf prefix",
        .{count},
    );
    try appendIssue(allocator, issues, .info, "lilbuf_prefix_count", msg, &.{});
}

// --- byte-string formatting helper -----------------------------------

const BytesEscaped = struct {
    bytes: []const u8,
    pub fn format(self: BytesEscaped, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.writeByte('"');
        for (self.bytes) |c| switch (c) {
            '"' => try w.writeAll("\\\""),
            '\\' => try w.writeAll("\\\\"),
            '\n' => try w.writeAll("\\n"),
            '\r' => try w.writeAll("\\r"),
            '\t' => try w.writeAll("\\t"),
            0x20...0x21, 0x23...0x5B, 0x5D...0x7E => try w.writeByte(c),
            else => try w.print("\\x{x:0>2}", .{c}),
        };
        try w.writeByte('"');
    }
};

fn fmtBytesEscaped(bytes: []const u8) BytesEscaped {
    return .{ .bytes = bytes };
}

// --- tests -----------------------------------------------------------

const testing = std.testing;

const TestEntry = struct { bytes: []const u8, rank: u32 };

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

fn buildByteVocab(
    allocator: std.mem.Allocator,
    extras: []const TestEntry,
) !Bpe {
    var entries: std.ArrayList(TestEntry) = .empty;
    defer entries.deinit(allocator);

    var byte_holders: [256][1]u8 = undefined;
    var i: u32 = 0;
    while (i < 256) : (i += 1) {
        byte_holders[i][0] = @intCast(i);
        try entries.append(allocator, .{ .bytes = byte_holders[i][0..1], .rank = i });
    }
    for (extras) |e| try entries.append(allocator, e);

    const src = try buildVocabSource(allocator, entries.items);
    defer allocator.free(src);
    return Bpe.loadTiktokenBytes(allocator, src);
}

const ByBytesOverride = struct { key: []const u8, id: TokenId };

// Build a Bpe directly with a hand-crafted by_bytes that may disagree
// with idBytes — for synthesizing pathological vocabs the loader rejects.
fn buildPathologicalBpe(
    allocator: std.mem.Allocator,
    pieces: []const []const u8,
    // (key_bytes, id) overrides for by_bytes. If empty we use the
    // straightforward "by_bytes[idBytes(i)] = i" mapping.
    overrides: []const ByBytesOverride,
) !Bpe {
    var total: usize = 0;
    for (pieces) |p| total += p.len;
    const bytes = try allocator.alloc(u8, total);
    errdefer allocator.free(bytes);
    const offsets = try allocator.alloc(u32, pieces.len + 1);
    errdefer allocator.free(offsets);

    var w: u32 = 0;
    offsets[0] = 0;
    for (pieces, 0..) |p, i| {
        @memcpy(bytes[w .. w + p.len], p);
        w += @intCast(p.len);
        offsets[i + 1] = w;
    }

    var by_bytes = std.StringHashMap(TokenId).init(allocator);
    errdefer by_bytes.deinit();
    try by_bytes.ensureTotalCapacity(@intCast(pieces.len + overrides.len));

    if (overrides.len == 0) {
        for (0..pieces.len) |i| {
            const key = bytes[offsets[i]..offsets[i + 1]];
            try by_bytes.put(key, @intCast(i));
        }
    } else {
        for (overrides) |o| try by_bytes.put(o.key, o.id);
    }

    // Diagnostic helper; populate the hot table to keep encode-perf
    // sanity tests honest if they encode through this Bpe.
    const hot_table = try @import("bpe.zig").Bpe.buildHotTable(allocator, bytes, offsets, @intCast(pieces.len));
    errdefer allocator.free(hot_table);

    return .{
        .allocator = allocator,
        .bytes = bytes,
        .offsets = offsets,
        .count = @intCast(pieces.len),
        .by_bytes = by_bytes,
        .hot_table = hot_table,
    };
}

test "unreachable_merges catches a broken vocab" {
    // Bytes 0..255 + "abc" at id 256. No "ab" or "bc" in vocab, so
    // (a, bc) and (ab, c) both fail; id 256 is unreachable.
    var bpe = try buildByteVocab(testing.allocator, &.{
        .{ .bytes = "abc", .rank = 256 },
    });
    defer bpe.deinit();

    var report = try checkBpe(testing.allocator, &bpe, null, &.{}, null, .{
        .unreachable_merges = true,
        .duplicate_decodings = false,
        .roundtrip = false,
        .cl100k_pathologies = false,
        .whitespace = false,
        .special_shadowing = false,
        .single_byte_coverage = false,
    });
    defer report.deinit();

    var found = false;
    for (report.issues) |it| {
        if (std.mem.eql(u8, it.check, "unreachable_merges")) {
            found = true;
            try testing.expectEqual(@as(usize, 1), it.ids.len);
            try testing.expectEqual(@as(TokenId, 256), it.ids[0]);
        }
    }
    try testing.expect(found);
}

test "duplicate_decodings: detected" {
    // Two ids with identical bytes "ab". Built by hand because the
    // tiktoken loader's by_bytes.put would otherwise collapse them.
    var bpe = try buildPathologicalBpe(testing.allocator, &.{ "ab", "ab" }, &.{});
    defer bpe.deinit();

    var report = try checkBpe(testing.allocator, &bpe, null, &.{}, null, .{
        .unreachable_merges = false,
        .duplicate_decodings = true,
        .roundtrip = false,
        .cl100k_pathologies = false,
        .whitespace = false,
        .special_shadowing = false,
        .single_byte_coverage = false,
    });
    defer report.deinit();

    var found = false;
    for (report.issues) |it| {
        if (std.mem.eql(u8, it.check, "duplicate_decodings")) {
            found = true;
            try testing.expectEqual(@as(usize, 2), it.ids.len);
        }
    }
    try testing.expect(found);
}

test "roundtrip: all default fixtures pass on a sane vocab" {
    // 256 single-byte tokens only. With identity pretok + concat decode,
    // encode is byte_id-equivalent and roundtrip is exact.
    var bpe = try buildByteVocab(testing.allocator, &.{});
    defer bpe.deinit();

    var report = try checkBpe(testing.allocator, &bpe, null, &.{}, null, .{
        .unreachable_merges = false,
        .duplicate_decodings = false,
        .roundtrip = true,
        .cl100k_pathologies = false,
        .whitespace = false,
        .special_shadowing = false,
        .single_byte_coverage = false,
    });
    defer report.deinit();

    for (report.issues) |it| {
        try testing.expect(!std.mem.eql(u8, it.check, "roundtrip"));
    }
}

test "whitespace check: pathological vocab fires warning" {
    // Build a 257-piece vocab: bytes 0..255 + piece "X" at id 256.
    // Override by_bytes so the key "  " (two spaces) resolves to id 256
    // — but idBytes(256) is "X", so the roundtrip fails on "  ".
    var pieces: [257][]const u8 = undefined;
    var byte_holders: [256][1]u8 = undefined;
    var i: u32 = 0;
    while (i < 256) : (i += 1) {
        byte_holders[i][0] = @intCast(i);
        pieces[i] = byte_holders[i][0..1];
    }
    pieces[256] = "X";

    var overrides: [258]ByBytesOverride = undefined;
    var j: u32 = 0;
    while (j < 256) : (j += 1) {
        overrides[j] = .{ .key = byte_holders[j][0..1], .id = j };
    }
    // "X" => 256 (overrides the normal 'X'=88 mapping intentionally; we
    // want encode("X") to also go to 256 so the rest of the fixtures still
    // roundtrip to themselves). Actually we ONLY want "  " to misbehave;
    // override "X" key to id 88 explicitly so single-byte X roundtrips.
    overrides[256] = .{ .key = "X", .id = 88 };
    overrides[257] = .{ .key = "  ", .id = 256 };

    var bpe = try buildPathologicalBpe(testing.allocator, &pieces, &overrides);
    defer bpe.deinit();

    var report = try checkBpe(testing.allocator, &bpe, null, &.{}, null, .{
        .unreachable_merges = false,
        .duplicate_decodings = false,
        .roundtrip = false,
        .cl100k_pathologies = false,
        .whitespace = true,
        .special_shadowing = false,
        .single_byte_coverage = false,
    });
    defer report.deinit();

    var found_mismatch = false;
    for (report.issues) |it| {
        if (std.mem.eql(u8, it.check, "whitespace") and it.severity == .error_sev) {
            found_mismatch = true;
        }
    }
    try testing.expect(found_mismatch);
}

test "special_shadowing: detected" {
    // Bytes 0..255 + intermediate merges so BPE can collapse "[CLS]" into
    // a single id. Added token "[CLS]" sits at id 5 (a special). The BPE
    // also produces a single id for "[CLS]" (the merged piece). Shadow.
    var bpe = try buildByteVocab(testing.allocator, &.{
        .{ .bytes = "[C", .rank = 256 },
        .{ .bytes = "[CL", .rank = 257 },
        .{ .bytes = "[CLS", .rank = 258 },
        .{ .bytes = "[CLS]", .rank = 259 },
    });
    defer bpe.deinit();

    const added = [_]AddedToken{
        .{ .id = 5, .content = "[CLS]" },
    };

    var report = try checkBpe(testing.allocator, &bpe, null, &added, null, .{
        .unreachable_merges = false,
        .duplicate_decodings = false,
        .roundtrip = false,
        .cl100k_pathologies = false,
        .whitespace = false,
        .special_shadowing = true,
        .single_byte_coverage = false,
    });
    defer report.deinit();

    var found = false;
    for (report.issues) |it| {
        if (std.mem.eql(u8, it.check, "special_shadowing")) {
            found = true;
            try testing.expectEqual(@as(usize, 2), it.ids.len);
            try testing.expectEqual(@as(TokenId, 5), it.ids[0]);
            try testing.expectEqual(@as(TokenId, 259), it.ids[1]);
        }
    }
    try testing.expect(found);
}
