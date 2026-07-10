//! Library-level helpers backing the `ztok validate` and `ztok roundtrip`
//! CLI subcommands. Kept out of `src/main.zig` so they can be unit-tested
//! against the root module without spawning a subprocess.
//!
//! The functions here take an already-loaded `*const Bpe` (validate) or
//! `*const Pipeline` (roundtrip), plus a `*std.Io.Writer`, and produce
//! human- or machine-readable output. They never read flags or files
//! directly — that's the CLI's job.

const std = @import("std");
const TokenId = @import("token.zig").TokenId;
const Bpe = @import("bpe.zig").Bpe;
const Unigram = @import("unigram.zig").Unigram;
const WordPiece = @import("wordpiece.zig").WordPiece;
const Monster = @import("monster.zig").Monster;
const RwkvWorld = @import("rwkv_world.zig").RwkvWorld;
const Pipeline = @import("pipeline.zig").Pipeline;
const Vocab = @import("vocab.zig").Vocab;
const doctor = @import("doctor.zig");

pub const Format = enum { text, json };

/// A loaded model dispatched to the appropriate `doctor.check*` function.
/// Kept thin — owners hold the concrete value and pass a pointer; the
/// CLI's `runValidateAny` and `runRoundtrip` peek the kind and call the
/// right validator. `vocab_size` is forwarded for human-readable output.
pub const ModelKind = enum { bpe, unigram, wordpiece, monster, rwkv_world };

pub const LoadedModel = union(ModelKind) {
    bpe: *const Bpe,
    unigram: *const Unigram,
    wordpiece: *const WordPiece,
    monster: *const Monster,
    rwkv_world: *const RwkvWorld,

    pub fn vocabSize(self: LoadedModel) u32 {
        return switch (self) {
            .bpe => |b| b.count,
            .unigram => |u| u.count,
            .wordpiece => |w| w.count,
            .monster => |m| m.count,
            .rwkv_world => |r| r.count,
        };
    }
};

/// Returns whether a given check name applies to the given model kind.
/// The CLI uses this to skip per-check rows for checks that don't apply
/// to the loaded model (e.g. `unreachable_merges` for Unigram).
pub fn checkAppliesToModel(name: []const u8, kind: ModelKind) bool {
    // Model-agnostic (apply to all four kinds).
    if (std.mem.eql(u8, name, "duplicate_decodings")) return true;
    if (std.mem.eql(u8, name, "roundtrip")) return true;
    if (std.mem.eql(u8, name, "whitespace")) return true;
    if (std.mem.eql(u8, name, "special_shadowing")) return true;
    // BPE-only.
    if (std.mem.eql(u8, name, "unreachable_merges")) return kind == .bpe;
    if (std.mem.eql(u8, name, "cl100k_pathologies")) return kind == .bpe;
    if (std.mem.eql(u8, name, "single_byte_coverage")) return kind == .bpe;
    // Unigram-only.
    if (std.mem.eql(u8, name, "score_sanity")) return kind == .unigram;
    if (std.mem.eql(u8, name, "unk_coverage")) return kind == .unigram;
    // WordPiece-only.
    if (std.mem.eql(u8, name, "continuation_consistency")) return kind == .wordpiece;
    // Monster-only.
    if (std.mem.eql(u8, name, "branch_coverage")) return kind == .monster;
    if (std.mem.eql(u8, name, "lilbuf_prefix_count")) return kind == .monster;
    return false;
}

pub const ValidateOptions = struct {
    /// Subset of doctor checks to run. Defaults to all.
    checks: doctor.Checks = .{},
    /// Optional caller-provided fixtures for the `roundtrip` check.
    fixtures: ?[]const []const u8 = null,
    /// Output format.
    format: Format = .text,
};

pub const ValidateResult = struct {
    warnings: u32,
    errors: u32,

    pub fn exitCode(self: ValidateResult) u8 {
        return if (self.errors > 0) 1 else 0;
    }
};

const CHECK_ORDER = [_][]const u8{
    // BPE-friendly (also model-agnostic for some).
    "unreachable_merges",
    "duplicate_decodings",
    "roundtrip",
    "cl100k_pathologies",
    "whitespace",
    "special_shadowing",
    "single_byte_coverage",
    // Unigram-specific.
    "score_sanity",
    "unk_coverage",
    // WordPiece-specific.
    "continuation_consistency",
    // Monster-specific.
    "branch_coverage",
    "lilbuf_prefix_count",
};

fn checkEnabled(checks: doctor.Checks, name: []const u8) bool {
    if (std.mem.eql(u8, name, "unreachable_merges")) return checks.unreachable_merges;
    if (std.mem.eql(u8, name, "duplicate_decodings")) return checks.duplicate_decodings;
    if (std.mem.eql(u8, name, "roundtrip")) return checks.roundtrip;
    if (std.mem.eql(u8, name, "cl100k_pathologies")) return checks.cl100k_pathologies;
    if (std.mem.eql(u8, name, "whitespace")) return checks.whitespace;
    if (std.mem.eql(u8, name, "special_shadowing")) return checks.special_shadowing;
    if (std.mem.eql(u8, name, "single_byte_coverage")) return checks.single_byte_coverage;
    if (std.mem.eql(u8, name, "score_sanity")) return checks.score_sanity;
    if (std.mem.eql(u8, name, "unk_coverage")) return checks.unk_coverage;
    if (std.mem.eql(u8, name, "continuation_consistency")) return checks.continuation_consistency;
    if (std.mem.eql(u8, name, "branch_coverage")) return checks.branch_coverage;
    if (std.mem.eql(u8, name, "lilbuf_prefix_count")) return checks.lilbuf_prefix_count;
    return false;
}

/// Parse a comma-separated list of check names into a `doctor.Checks`
/// mask. Unknown names are reported via `unknown_out` (caller-allocated
/// slice; first unknown wins) so the CLI can surface a friendly error.
pub fn parseChecks(spec: []const u8, unknown_out: *?[]const u8) ?doctor.Checks {
    var checks: doctor.Checks = .{
        .unreachable_merges = false,
        .duplicate_decodings = false,
        .roundtrip = false,
        .cl100k_pathologies = false,
        .whitespace = false,
        .special_shadowing = false,
        .single_byte_coverage = false,
        .score_sanity = false,
        .unk_coverage = false,
        .continuation_consistency = false,
        .branch_coverage = false,
        .lilbuf_prefix_count = false,
    };
    var it = std.mem.splitScalar(u8, spec, ',');
    while (it.next()) |raw| {
        const name = std.mem.trim(u8, raw, " \t");
        if (name.len == 0) continue;
        if (std.mem.eql(u8, name, "unreachable_merges")) {
            checks.unreachable_merges = true;
        } else if (std.mem.eql(u8, name, "duplicate_decodings")) {
            checks.duplicate_decodings = true;
        } else if (std.mem.eql(u8, name, "roundtrip")) {
            checks.roundtrip = true;
        } else if (std.mem.eql(u8, name, "cl100k_pathologies")) {
            checks.cl100k_pathologies = true;
        } else if (std.mem.eql(u8, name, "whitespace")) {
            checks.whitespace = true;
        } else if (std.mem.eql(u8, name, "special_shadowing")) {
            checks.special_shadowing = true;
        } else if (std.mem.eql(u8, name, "single_byte_coverage")) {
            checks.single_byte_coverage = true;
        } else if (std.mem.eql(u8, name, "score_sanity")) {
            checks.score_sanity = true;
        } else if (std.mem.eql(u8, name, "unk_coverage")) {
            checks.unk_coverage = true;
        } else if (std.mem.eql(u8, name, "continuation_consistency")) {
            checks.continuation_consistency = true;
        } else if (std.mem.eql(u8, name, "branch_coverage")) {
            checks.branch_coverage = true;
        } else if (std.mem.eql(u8, name, "lilbuf_prefix_count")) {
            checks.lilbuf_prefix_count = true;
        } else {
            unknown_out.* = name;
            return null;
        }
    }
    return checks;
}

/// Run the doctor and write a report to `out`. Returns aggregate counts
/// so the caller can pick the exit code.
///
/// Back-compat wrapper that defaults the model kind to `.bpe` — keeps
/// the original 1.13 CLI surface unchanged.
pub fn runValidate(
    allocator: std.mem.Allocator,
    bpe: *const Bpe,
    pipeline: ?*const Pipeline,
    opts: ValidateOptions,
    out: *std.Io.Writer,
) !ValidateResult {
    return runValidateAny(allocator, .{ .bpe = bpe }, pipeline, opts, out);
}

/// Dispatch validate to the appropriate `doctor.check*` based on the
/// loaded model's kind. Used by the post-1.13 multi-format
/// `ztok validate` path.
pub fn runValidateAny(
    allocator: std.mem.Allocator,
    loaded: LoadedModel,
    pipeline: ?*const Pipeline,
    opts: ValidateOptions,
    out: *std.Io.Writer,
) !ValidateResult {
    // Build a pipeline on the fly if the caller didn't supply one. The
    // model-agnostic checks (roundtrip, whitespace) need one; for BPE
    // the existing `checkBpe` already synthesizes one internally, but
    // for the other three kinds we don't have that fallback path.
    var owned_vocab: ?Vocab = null;
    defer if (owned_vocab) |*v| v.deinit();
    var owned_pipe: ?Pipeline = null;
    const pipe_eff: ?*const Pipeline = blk: {
        if (pipeline != null) break :blk pipeline;
        if (loaded == .bpe) break :blk null; // checkBpe builds its own.
        owned_vocab = Vocab.empty(allocator);
        const model_val: @import("model.zig").Model = switch (loaded) {
            .bpe => unreachable,
            .unigram => |u| .{ .unigram = u },
            .wordpiece => |w| .{ .wordpiece = w },
            .monster => |m| .{ .monster = m },
            .rwkv_world => |r| .{ .rwkv_world = r },
        };
        owned_pipe = .{
            .normalizer = .identity,
            .pre_tokenizer = .identity,
            .model = model_val,
            .decoder = .concat,
            .vocab = &owned_vocab.?,
        };
        break :blk &owned_pipe.?;
    };

    var report = switch (loaded) {
        .bpe => |b| try doctor.checkBpe(allocator, b, pipe_eff, &.{}, opts.fixtures, opts.checks),
        .unigram => |u| try doctor.checkUnigram(allocator, u, pipe_eff.?, &.{}, opts.fixtures, opts.checks),
        .wordpiece => |w| try doctor.checkWordPiece(allocator, w, pipe_eff.?, &.{}, opts.fixtures, opts.checks),
        .monster => |m| try doctor.checkMonster(allocator, m, pipe_eff.?, &.{}, opts.fixtures, opts.checks),
        .rwkv_world => |r| try doctor.checkRwkvWorld(allocator, r, pipe_eff.?, &.{}, opts.fixtures, opts.checks),
    };
    defer report.deinit();

    // Bucket issues by check name for per-check tallies.
    var per_check_warnings = [_]u32{0} ** CHECK_ORDER.len;
    var per_check_errors = [_]u32{0} ** CHECK_ORDER.len;
    var per_check_infos = [_]u32{0} ** CHECK_ORDER.len;
    var total_warnings: u32 = 0;
    var total_errors: u32 = 0;

    for (report.issues) |it| {
        const slot = checkIndex(it.check) orelse continue;
        switch (it.severity) {
            .warning => {
                per_check_warnings[slot] += 1;
                total_warnings += 1;
            },
            .error_sev => {
                per_check_errors[slot] += 1;
                total_errors += 1;
            },
            .info => per_check_infos[slot] += 1,
        }
    }

    switch (opts.format) {
        .text => try writeText(
            out,
            opts.checks,
            loaded,
            report.issues,
            per_check_warnings,
            per_check_errors,
            per_check_infos,
            opts.fixtures,
            total_warnings,
            total_errors,
        ),
        .json => try writeJson(
            out,
            opts.checks,
            loaded,
            report.issues,
            per_check_warnings,
            per_check_errors,
            per_check_infos,
            total_warnings,
            total_errors,
        ),
    }

    return .{ .warnings = total_warnings, .errors = total_errors };
}

fn checkIndex(name: []const u8) ?usize {
    for (CHECK_ORDER, 0..) |c, i| {
        if (std.mem.eql(u8, c, name)) return i;
    }
    return null;
}

fn writeText(
    out: *std.Io.Writer,
    checks: doctor.Checks,
    loaded: LoadedModel,
    issues: []const doctor.Issue,
    per_check_warnings: [CHECK_ORDER.len]u32,
    per_check_errors: [CHECK_ORDER.len]u32,
    per_check_infos: [CHECK_ORDER.len]u32,
    fixtures: ?[]const []const u8,
    total_warnings: u32,
    total_errors: u32,
) !void {
    const kind: ModelKind = std.meta.activeTag(loaded);
    for (CHECK_ORDER, 0..) |name, idx| {
        if (!checkEnabled(checks, name)) continue;
        if (!checkAppliesToModel(name, kind)) continue;
        const w = per_check_warnings[idx];
        const e = per_check_errors[idx];
        const i = per_check_infos[idx];
        // For single_byte_coverage the doctor packs all missing byte ids
        // into one issue's .ids — surface that count to the renderer.
        var missing_bytes: u32 = 0;
        if (std.mem.eql(u8, name, "single_byte_coverage")) {
            for (issues) |it| if (std.mem.eql(u8, it.check, name)) {
                missing_bytes += @intCast(it.ids.len);
            };
        }
        const marker: []const u8 = if (e > 0) "x" else if (w > 0) "!" else "-";
        const status_word: []const u8 = if (e > 0) "FAIL" else if (w > 0) "WARN" else "OK";
        try out.print("  {s} {s:<24} {s:<4} ", .{ marker, name, status_word });
        try renderDetail(out, name, w, e, i, loaded, fixtures, missing_bytes);
        try out.writeByte('\n');

        // Emit per-issue lines for non-pass checks.
        if (w > 0 or e > 0 or i > 0) {
            for (issues) |it| {
                if (!std.mem.eql(u8, it.check, name)) continue;
                const sev_str = switch (it.severity) {
                    .info => "info",
                    .warning => "warn",
                    .error_sev => "error",
                };
                try out.print("      [{s}] {s}", .{ sev_str, it.message });
                if (it.ids.len > 0) {
                    try out.writeAll(" (ids: ");
                    for (it.ids, 0..) |id, k| {
                        if (k > 0) try out.writeAll(", ");
                        try out.print("{d}", .{id});
                    }
                    try out.writeAll(")");
                }
                try out.writeAll("\n");
            }
        }
    }

    try out.print("\nSummary: {d} warning{s}, {d} error{s}\n", .{
        total_warnings, plural(total_warnings),
        total_errors,   plural(total_errors),
    });
}

fn plural(n: u32) []const u8 {
    return if (n == 1) "" else "s";
}

/// Inline counter string after the check name, e.g.
/// `(10/10 fixtures)`, `(0 issues)`, `(256/256)`.
/// `missing_bytes` is non-zero only when single_byte_coverage flagged
/// specific byte ids; it lets the renderer report `(0/256)` rather than
/// `(255/256)` when 256 bytes are missing in one bundled issue.
fn renderDetail(
    out: *std.Io.Writer,
    name: []const u8,
    warnings: u32,
    errors: u32,
    infos: u32,
    loaded: LoadedModel,
    fixtures: ?[]const []const u8,
    missing_bytes: u32,
) !void {
    if (std.mem.eql(u8, name, "single_byte_coverage")) {
        const sz = loaded.vocabSize();
        const covered = if (sz >= 256)
            @as(u32, 256) - @min(missing_bytes, 256)
        else
            sz;
        try out.print("({d}/256)", .{covered});
    } else if (std.mem.eql(u8, name, "roundtrip")) {
        const default_n: u32 = 10;
        const n: u32 = if (fixtures) |f| @intCast(f.len) else default_n;
        const ok = if (n >= errors) n - errors else 0;
        try out.print("({d}/{d} fixtures)", .{ ok, n });
    } else if (std.mem.eql(u8, name, "cl100k_pathologies")) {
        const n: u32 = 10;
        const ok = if (n >= errors) n - errors else 0;
        try out.print("({d}/{d} fixtures)", .{ ok, n });
    } else {
        const total = warnings + errors + infos;
        try out.print("({d} issue{s})", .{ total, plural(total) });
    }
}

// --- JSON output ----------------------------------------------------------

fn writeJsonString(out: *std.Io.Writer, s: []const u8) !void {
    try out.writeByte('"');
    for (s) |c| switch (c) {
        '"' => try out.writeAll("\\\""),
        '\\' => try out.writeAll("\\\\"),
        '\n' => try out.writeAll("\\n"),
        '\r' => try out.writeAll("\\r"),
        '\t' => try out.writeAll("\\t"),
        0x08 => try out.writeAll("\\b"),
        0x0C => try out.writeAll("\\f"),
        0x00...0x07, 0x0B, 0x0E...0x1F => try out.print("\\u{x:0>4}", .{c}),
        else => try out.writeByte(c),
    };
    try out.writeByte('"');
}

fn writeJson(
    out: *std.Io.Writer,
    checks: doctor.Checks,
    loaded: LoadedModel,
    issues: []const doctor.Issue,
    per_check_warnings: [CHECK_ORDER.len]u32,
    per_check_errors: [CHECK_ORDER.len]u32,
    per_check_infos: [CHECK_ORDER.len]u32,
    total_warnings: u32,
    total_errors: u32,
) !void {
    const kind: ModelKind = std.meta.activeTag(loaded);
    const kind_str: []const u8 = switch (kind) {
        .bpe => "bpe",
        .unigram => "unigram",
        .wordpiece => "wordpiece",
        .monster => "monster",
        .rwkv_world => "rwkv_world",
    };
    try out.writeAll("{\"model_kind\":");
    try writeJsonString(out, kind_str);
    try out.writeAll(",\"checks\":[");
    var first_check = true;
    for (CHECK_ORDER, 0..) |name, idx| {
        if (!checkEnabled(checks, name)) continue;
        if (!checkAppliesToModel(name, kind)) continue;
        if (!first_check) try out.writeAll(",");
        first_check = false;
        const w = per_check_warnings[idx];
        const e = per_check_errors[idx];
        const status: []const u8 = if (e > 0) "fail" else if (w > 0) "warn" else "pass";

        try out.writeAll("{\"name\":");
        try writeJsonString(out, name);
        try out.writeAll(",\"status\":");
        try writeJsonString(out, status);
        try out.print(",\"warnings\":{d},\"errors\":{d},\"infos\":{d}", .{
            w, e, per_check_infos[idx],
        });
        try out.writeAll(",\"issues\":[");
        var first_issue = true;
        for (issues) |it| {
            if (!std.mem.eql(u8, it.check, name)) continue;
            if (!first_issue) try out.writeAll(",");
            first_issue = false;
            const sev_str = switch (it.severity) {
                .info => "info",
                .warning => "warning",
                .error_sev => "error",
            };
            try out.writeAll("{\"severity\":");
            try writeJsonString(out, sev_str);
            try out.writeAll(",\"message\":");
            try writeJsonString(out, it.message);
            try out.writeAll(",\"ids\":[");
            for (it.ids, 0..) |id, k| {
                if (k > 0) try out.writeAll(",");
                try out.print("{d}", .{id});
            }
            try out.writeAll("]}");
        }
        try out.writeAll("]}");
    }
    try out.print("],\"summary\":{{\"warnings\":{d},\"errors\":{d}}}}}\n", .{
        total_warnings, total_errors,
    });
}

// --- roundtrip --------------------------------------------------------

pub const RoundtripOptions = struct {
    summary_only: bool = false,
};

pub const RoundtripResult = struct {
    total_lines: u32,
    ok_lines: u32,
    bytes_processed: u64,

    pub fn allOk(self: RoundtripResult) bool {
        return self.ok_lines == self.total_lines;
    }
    pub fn exitCode(self: RoundtripResult) u8 {
        return if (self.allOk()) 0 else 1;
    }
};

/// Run round-trip (encode then decode then compare) for each line in
/// `input_text`. A trailing newline is treated as an empty final line
/// terminator, not a separate line. If `input_text` contains no
/// newlines, the whole buffer is treated as a single line.
pub fn runRoundtrip(
    allocator: std.mem.Allocator,
    pipeline: *const Pipeline,
    input_text: []const u8,
    opts: RoundtripOptions,
    out: *std.Io.Writer,
) !RoundtripResult {
    var result: RoundtripResult = .{ .total_lines = 0, .ok_lines = 0, .bytes_processed = 0 };

    var line_no: u32 = 0;
    var iter = std.mem.splitScalar(u8, input_text, '\n');
    var last_was_newline = false;
    if (input_text.len > 0 and input_text[input_text.len - 1] == '\n') last_was_newline = true;

    while (iter.next()) |raw| {
        // Skip the empty trailing field produced by a terminal newline.
        if (iter.peek() == null and raw.len == 0 and last_was_newline) break;

        line_no += 1;
        result.total_lines += 1;
        result.bytes_processed += raw.len;

        const ids = pipeline.encode(allocator, raw) catch |err| {
            if (!opts.summary_only) {
                try out.print("line {d}: ENCODE_ERROR ({s}) [{d} bytes]\n", .{
                    line_no, @errorName(err), raw.len,
                });
            }
            continue;
        };
        defer allocator.free(ids);

        const decoded = pipeline.decode(allocator, ids) catch |err| {
            if (!opts.summary_only) {
                try out.print("line {d}: DECODE_ERROR ({s}) [{d} bytes, {d} ids]\n", .{
                    line_no, @errorName(err), raw.len, ids.len,
                });
            }
            continue;
        };
        defer allocator.free(decoded);

        if (std.mem.eql(u8, decoded, raw)) {
            result.ok_lines += 1;
            if (!opts.summary_only) {
                try out.print("line {d}: OK ({d} bytes -> {d} ids)\n", .{
                    line_no, raw.len, ids.len,
                });
            }
        } else if (!opts.summary_only) {
            try out.print("line {d}: MISMATCH ({d} bytes -> {d} ids -> {d} bytes)\n", .{
                line_no, raw.len, ids.len, decoded.len,
            });
            // Locate first diverging byte for the report.
            const max = @min(raw.len, decoded.len);
            var diff_at: usize = max;
            for (0..max) |k| if (raw[k] != decoded[k]) {
                diff_at = k;
                break;
            };
            try out.print("    first byte diff at offset {d}\n", .{diff_at});
            // Surface the diverging token ids (those whose decoded slice
            // crosses or sits past `diff_at`). We re-decode each id
            // individually since the pipeline doesn't expose per-id byte
            // spans through .decode. This is a small CLI nicety; for
            // hot-path inspection callers should use encodeWithOffsets.
            var byte_cursor: usize = 0;
            var first_bad_id: ?u32 = null;
            var first_bad_pos: usize = 0;
            for (ids, 0..) |id, k| {
                const single = [_]TokenId{id};
                const piece = pipeline.decode(allocator, &single) catch break;
                defer allocator.free(piece);
                if (byte_cursor + piece.len > diff_at) {
                    first_bad_id = id;
                    first_bad_pos = k;
                    break;
                }
                byte_cursor += piece.len;
            }
            if (first_bad_id) |id| {
                try out.print("    first divergent id at position {d}: {d}\n", .{
                    first_bad_pos, id,
                });
            }
        }
    }

    if (opts.summary_only) {
        const pct: f64 = if (result.total_lines == 0) 100.0 else (@as(f64, @floatFromInt(result.ok_lines)) * 100.0) / @as(f64, @floatFromInt(result.total_lines));
        try out.print("{d} lines, {d} round-trip OK ({d:.1}%), {d} bytes processed\n", .{
            result.total_lines, result.ok_lines, pct, result.bytes_processed,
        });
    } else {
        try out.print("\nsummary: {d}/{d} lines OK, {d} bytes processed\n", .{
            result.ok_lines, result.total_lines, result.bytes_processed,
        });
    }
    return result;
}

// --- tests -----------------------------------------------------------

const testing = std.testing;

const TestEntry = struct { bytes: []const u8, rank: u32 };

fn buildVocabSource(allocator: std.mem.Allocator, entries: []const TestEntry) ![]u8 {
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

fn buildByteVocab(allocator: std.mem.Allocator, extras: []const TestEntry) !Bpe {
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

fn captureOutput(comptime cap: usize) type {
    return struct {
        buf: [cap]u8 = undefined,
        writer: std.Io.Writer = undefined,

        const Self = @This();
        fn init(self: *Self) void {
            self.writer = .fixed(&self.buf);
        }
        fn slice(self: *Self) []const u8 {
            return self.writer.buffered();
        }
    };
}

test "runValidate text output lists every default check" {
    var bpe = try buildByteVocab(testing.allocator, &.{});
    defer bpe.deinit();

    var cap: captureOutput(8192) = .{};
    cap.init();

    const r = try runValidate(testing.allocator, &bpe, null, .{}, &cap.writer);
    const s = cap.slice();

    // All seven check names should appear in default output.
    inline for ([_][]const u8{
        "unreachable_merges",
        "duplicate_decodings",
        "roundtrip",
        "cl100k_pathologies",
        "whitespace",
        "special_shadowing",
        "single_byte_coverage",
    }) |name| {
        try testing.expect(std.mem.indexOf(u8, s, name) != null);
    }
    try testing.expect(std.mem.indexOf(u8, s, "Summary:") != null);
    try testing.expectEqual(@as(u32, 0), r.errors);
}

test "runValidate JSON output parses back as valid JSON" {
    var bpe = try buildByteVocab(testing.allocator, &.{});
    defer bpe.deinit();

    var cap: captureOutput(16384) = .{};
    cap.init();

    _ = try runValidate(testing.allocator, &bpe, null, .{ .format = .json }, &cap.writer);
    const s = cap.slice();

    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, s, .{});
    defer parsed.deinit();

    const root = parsed.value;
    try testing.expect(root == .object);
    try testing.expect(root.object.get("checks") != null);
    try testing.expect(root.object.get("summary") != null);

    const checks_arr = root.object.get("checks").?.array;
    try testing.expectEqual(@as(usize, 7), checks_arr.items.len);

    // Sanity: every check object has name/status/issues.
    for (checks_arr.items) |c| {
        try testing.expect(c.object.get("name") != null);
        try testing.expect(c.object.get("status") != null);
        try testing.expect(c.object.get("issues") != null);
    }
}

test "runValidate with --checks subset only runs that check" {
    var bpe = try buildByteVocab(testing.allocator, &.{});
    defer bpe.deinit();

    const checks: doctor.Checks = .{
        .unreachable_merges = false,
        .duplicate_decodings = false,
        .roundtrip = true,
        .cl100k_pathologies = false,
        .whitespace = false,
        .special_shadowing = false,
        .single_byte_coverage = false,
    };

    var cap: captureOutput(8192) = .{};
    cap.init();
    _ = try runValidate(testing.allocator, &bpe, null, .{ .checks = checks }, &cap.writer);
    const s = cap.slice();

    try testing.expect(std.mem.indexOf(u8, s, "roundtrip") != null);
    // Disabled checks should not appear as their own section.
    try testing.expect(std.mem.indexOf(u8, s, "unreachable_merges") == null);
    try testing.expect(std.mem.indexOf(u8, s, "single_byte_coverage") == null);
}

test "runValidate exits non-zero when a fixture round-trip fails" {
    // 255 bytes only — missing byte 0xFF entirely. Single-byte-coverage
    // fails AND any fixture containing 0xFF (we'll inject one) cannot
    // round-trip.
    var entries: std.ArrayList(TestEntry) = .empty;
    defer entries.deinit(testing.allocator);
    var holders: [255][1]u8 = undefined;
    var i: u32 = 0;
    while (i < 255) : (i += 1) {
        holders[i][0] = @intCast(i);
        try entries.append(testing.allocator, .{ .bytes = holders[i][0..1], .rank = i });
    }
    const src = try buildVocabSource(testing.allocator, entries.items);
    defer testing.allocator.free(src);
    var bpe = try Bpe.loadTiktokenBytes(testing.allocator, src);
    defer bpe.deinit();

    var cap: captureOutput(8192) = .{};
    cap.init();
    const r = try runValidate(testing.allocator, &bpe, null, .{}, &cap.writer);
    try testing.expect(r.errors > 0);
    try testing.expect(r.exitCode() != 0);
}

test "runRoundtrip summary mode counts OK lines correctly" {
    var bpe = try buildByteVocab(testing.allocator, &.{});
    defer bpe.deinit();
    var v = Vocab.empty(testing.allocator);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .{ .bpe = &bpe },
        .decoder = .concat,
        .vocab = &v,
    };

    const input = "hello\nworld\nfoo bar\n";
    var cap: captureOutput(2048) = .{};
    cap.init();
    const r = try runRoundtrip(testing.allocator, &pipe, input, .{ .summary_only = true }, &cap.writer);
    try testing.expectEqual(@as(u32, 3), r.total_lines);
    try testing.expectEqual(@as(u32, 3), r.ok_lines);
    try testing.expectEqual(@as(u64, 17), r.bytes_processed);
    try testing.expect(r.allOk());
    try testing.expectEqual(@as(u8, 0), r.exitCode());

    const s = cap.slice();
    try testing.expect(std.mem.indexOf(u8, s, "3 lines") != null);
    try testing.expect(std.mem.indexOf(u8, s, "100.0%") != null);
}

test "runRoundtrip per-line mode emits OK/MISMATCH lines" {
    var bpe = try buildByteVocab(testing.allocator, &.{});
    defer bpe.deinit();
    var v = Vocab.empty(testing.allocator);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .{ .bpe = &bpe },
        .decoder = .concat,
        .vocab = &v,
    };

    const input = "first line\nsecond line\n";
    var cap: captureOutput(2048) = .{};
    cap.init();
    const r = try runRoundtrip(testing.allocator, &pipe, input, .{}, &cap.writer);
    try testing.expectEqual(@as(u32, 2), r.total_lines);
    try testing.expectEqual(@as(u32, 2), r.ok_lines);

    const s = cap.slice();
    try testing.expect(std.mem.indexOf(u8, s, "line 1: OK") != null);
    try testing.expect(std.mem.indexOf(u8, s, "line 2: OK") != null);
}

test "parseChecks accepts known names and reports unknown" {
    var unknown: ?[]const u8 = null;
    const ok = parseChecks("roundtrip,whitespace", &unknown);
    try testing.expect(ok != null);
    try testing.expect(unknown == null);
    try testing.expect(ok.?.roundtrip);
    try testing.expect(ok.?.whitespace);
    try testing.expect(!ok.?.unreachable_merges);

    var unknown2: ?[]const u8 = null;
    const bad = parseChecks("roundtrip,bogus", &unknown2);
    try testing.expect(bad == null);
    try testing.expect(unknown2 != null);
    try testing.expectEqualStrings("bogus", unknown2.?);
}

test "parseChecks accepts model-specific check names" {
    var unknown: ?[]const u8 = null;
    const ok = parseChecks(
        "score_sanity,unk_coverage,continuation_consistency,branch_coverage,lilbuf_prefix_count",
        &unknown,
    );
    try testing.expect(ok != null);
    try testing.expect(unknown == null);
    try testing.expect(ok.?.score_sanity);
    try testing.expect(ok.?.unk_coverage);
    try testing.expect(ok.?.continuation_consistency);
    try testing.expect(ok.?.branch_coverage);
    try testing.expect(ok.?.lilbuf_prefix_count);
    // BPE-only checks not in the list stay off.
    try testing.expect(!ok.?.unreachable_merges);
}

// --- multi-format runValidateAny smoke tests ---

// Build a tiny Unigram with bytes a/b/c + the merged "abc" piece + an unk.
fn buildSampleUnigram(a: std.mem.Allocator) !Unigram {
    var b = Unigram.Builder.init(a);
    defer b.deinit();
    _ = try b.addToken("<unk>", -10.0);
    _ = try b.addToken("a", -2.0);
    _ = try b.addToken("b", -2.0);
    _ = try b.addToken("c", -2.0);
    _ = try b.addToken("ab", -1.0);
    return b.finalize(0);
}

test "runValidateAny on Unigram emits Unigram-specific checks only" {
    var u = try buildSampleUnigram(testing.allocator);
    defer u.deinit();
    var v = Vocab.empty(testing.allocator);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .{ .unigram = &u },
        .decoder = .concat,
        .vocab = &v,
    };

    var cap: captureOutput(8192) = .{};
    cap.init();
    _ = try runValidateAny(testing.allocator, .{ .unigram = &u }, &pipe, .{}, &cap.writer);
    const s = cap.slice();

    // Unigram-specific checks must appear.
    try testing.expect(std.mem.indexOf(u8, s, "score_sanity") != null);
    try testing.expect(std.mem.indexOf(u8, s, "unk_coverage") != null);
    // BPE-only checks must NOT appear (filtered by checkAppliesToModel).
    try testing.expect(std.mem.indexOf(u8, s, "unreachable_merges") == null);
    try testing.expect(std.mem.indexOf(u8, s, "cl100k_pathologies") == null);
    try testing.expect(std.mem.indexOf(u8, s, "single_byte_coverage") == null);
    // Monster-only checks must NOT appear either.
    try testing.expect(std.mem.indexOf(u8, s, "branch_coverage") == null);
    try testing.expect(std.mem.indexOf(u8, s, "lilbuf_prefix_count") == null);
}

test "runValidateAny JSON output is valid JSON for Unigram and Monster" {
    // Unigram half.
    {
        var u = try buildSampleUnigram(testing.allocator);
        defer u.deinit();
        var v = Vocab.empty(testing.allocator);
        defer v.deinit();
        const pipe: Pipeline = .{
            .normalizer = .identity,
            .pre_tokenizer = .identity,
            .model = .{ .unigram = &u },
            .decoder = .concat,
            .vocab = &v,
        };
        var cap: captureOutput(16384) = .{};
        cap.init();
        _ = try runValidateAny(testing.allocator, .{ .unigram = &u }, &pipe, .{ .format = .json }, &cap.writer);
        const s = cap.slice();
        var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, s, .{});
        defer parsed.deinit();
        try testing.expectEqualStrings("unigram", parsed.value.object.get("model_kind").?.string);
    }
    // Monster half — tiny synthetic vocab so the test is fast.
    {
        var b = Monster.Builder.init(testing.allocator);
        defer b.deinit();
        _ = try b.addToken("a");
        _ = try b.addToken("b");
        _ = try b.addToken("ab");
        _ = try b.addToken("abc");
        const unk = try b.addToken("<unk>");
        var m = try b.finalize(unk);
        defer m.deinit();
        var v = Vocab.empty(testing.allocator);
        defer v.deinit();
        const pipe: Pipeline = .{
            .normalizer = .identity,
            .pre_tokenizer = .identity,
            .model = .{ .monster = &m },
            .decoder = .concat,
            .vocab = &v,
        };
        var cap: captureOutput(16384) = .{};
        cap.init();
        _ = try runValidateAny(testing.allocator, .{ .monster = &m }, &pipe, .{ .format = .json }, &cap.writer);
        const s = cap.slice();
        var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, s, .{});
        defer parsed.deinit();
        try testing.expectEqualStrings("monster", parsed.value.object.get("model_kind").?.string);
    }
}

test "runValidateAny flags broken Monster vocab via branch_coverage" {
    // Build a Monster vocab with only length-1 pieces; branch_coverage
    // should fire an error and set exit code to 1.
    var b = Monster.Builder.init(testing.allocator);
    defer b.deinit();
    _ = try b.addToken("a");
    _ = try b.addToken("b");
    _ = try b.addToken("c");
    var m = try b.finalize(0); // unk = id 0 = 'a' (test only checks branch_coverage)
    defer m.deinit();
    var v = Vocab.empty(testing.allocator);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .{ .monster = &m },
        .decoder = .concat,
        .vocab = &v,
    };
    const checks: doctor.Checks = .{
        .unreachable_merges = false,
        .duplicate_decodings = false,
        .roundtrip = false,
        .cl100k_pathologies = false,
        .whitespace = false,
        .special_shadowing = false,
        .single_byte_coverage = false,
        .score_sanity = false,
        .unk_coverage = false,
        .continuation_consistency = false,
        .branch_coverage = true,
        .lilbuf_prefix_count = false,
    };
    var cap: captureOutput(4096) = .{};
    cap.init();
    const r = try runValidateAny(testing.allocator, .{ .monster = &m }, &pipe, .{ .checks = checks }, &cap.writer);
    try testing.expect(r.errors > 0);
    try testing.expectEqual(@as(u8, 1), r.exitCode());
}

test "runRoundtrip works on Unigram (summary mode)" {
    var u = try buildSampleUnigram(testing.allocator);
    defer u.deinit();
    var v = Vocab.empty(testing.allocator);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .{ .unigram = &u },
        .decoder = .concat,
        .vocab = &v,
    };

    const input = "ab\nabc\n";
    var cap: captureOutput(2048) = .{};
    cap.init();
    const r = try runRoundtrip(testing.allocator, &pipe, input, .{ .summary_only = true }, &cap.writer);
    try testing.expectEqual(@as(u32, 2), r.total_lines);
    // Both should roundtrip — "ab" via the ab piece, "abc" via ab+c.
    try testing.expectEqual(@as(u32, 2), r.ok_lines);
    try testing.expect(r.allOk());
}

test "runRoundtrip works on Monster (summary mode)" {
    // Tiny Monster: a, b, c, ab, abc + unk. Identity normalizer; encode
    // then decode should reproduce ASCII inputs exactly.
    var b = Monster.Builder.init(testing.allocator);
    defer b.deinit();
    _ = try b.addToken("a");
    _ = try b.addToken("b");
    _ = try b.addToken("c");
    _ = try b.addToken("ab");
    _ = try b.addToken("abc");
    const unk = try b.addToken("<unk>");
    var m = try b.finalize(unk);
    defer m.deinit();
    var v = Vocab.empty(testing.allocator);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .{ .monster = &m },
        .decoder = .concat,
        .vocab = &v,
    };
    const input = "abc\nab\n";
    var cap: captureOutput(2048) = .{};
    cap.init();
    const r = try runRoundtrip(testing.allocator, &pipe, input, .{ .summary_only = true }, &cap.writer);
    try testing.expectEqual(@as(u32, 2), r.total_lines);
    try testing.expectEqual(@as(u32, 2), r.ok_lines);
}

test "loadPipelineAutoDetect for HF Unigram routes to the Unigram validator" {
    // Inline T5-style HF JSON with a 3-token Unigram model. Write to
    // /tmp, point loadPipelineAutoDetect at it, expect a .unigram variant.
    // (We test the loader by going through the public surface — auto_detect
    // identifies it as `.hf_json`, the loader inspects `model.type`, and
    // dispatches to `unigramFromHF`.)
    const json =
        \\{
        \\  "added_tokens": [],
        \\  "model": {
        \\    "type": "Unigram",
        \\    "unk_id": 0,
        \\    "vocab": [
        \\      ["<unk>", -10.0],
        \\      ["a", -2.0],
        \\      ["b", -2.0],
        \\      ["ab", -1.0]
        \\    ]
        \\  }
        \\}
    ;
    const io = std.Io.Threaded.global_single_threaded.io();
    const path = "/tmp/ztok_cli_unigram_hf.json";
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = json });
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    // Re-implement the dispatch directly to avoid an additional cross-
    // module import; the production version lives in `main.zig`.
    const fmt = try @import("auto_detect.zig").detectFile(path);
    try testing.expectEqual(@import("auto_detect.zig").Format.hf_json, fmt);
    var hf = try @import("hf_json.zig").loadFromFile(testing.allocator, path);
    defer hf.deinit();
    try testing.expectEqual(@import("hf_json.zig").ModelKind.unigram, hf.model_kind);
    var u = try @import("hf_bridge.zig").unigramFromHF(testing.allocator, &hf);
    defer u.deinit();

    var v = Vocab.empty(testing.allocator);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .{ .unigram = &u },
        .decoder = .concat,
        .vocab = &v,
    };
    var cap: captureOutput(4096) = .{};
    cap.init();
    const result = try runValidateAny(testing.allocator, .{ .unigram = &u }, &pipe, .{}, &cap.writer);
    _ = result;
    const s = cap.slice();
    // Unigram-specific check must appear; confirms routing to checkUnigram.
    try testing.expect(std.mem.indexOf(u8, s, "score_sanity") != null);
}
