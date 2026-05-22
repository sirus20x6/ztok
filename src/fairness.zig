//! Multilingual tokenization-fairness metric.
//!
//! A tokenizer tuned for English can compress English text into very few
//! tokens while shredding other languages into many more. Because LLM cost,
//! latency, and effective context length are all measured in *tokens*, that
//! disparity charges the under-served languages a hidden "token premium":
//! the same semantic content costs more to send and leaves less room in the
//! window. This module quantifies that disparity from pre-counted statistics.
//!
//! It is deliberately self-contained: it imports only `std`, never reads
//! files, and never tokenizes anything. The caller counts tokens and units
//! (codepoints preferred, bytes acceptable) per language and hands us the
//! tallies; we turn them into a `Report`.
//!
//! Core quantity is **fertility** = tokens / unit. Lower is better (fewer
//! tokens to represent the same amount of text). Everything else is derived:
//!
//!   - per-language **premium**       = fertility / best_fertility   (>= 1)
//!   - **max_min_ratio**              = max_fertility / min_fertility (worst-case disparity)
//!   - **gini**                       = Gini coefficient of the fertility
//!                                      distribution (0 = perfectly fair,
//!                                      -> 1 = maximally unfair)
//!   - **mean_fertility**             = arithmetic mean of per-language fertility
//!
//! Languages with zero units carry no information and are skipped (guarded)
//! rather than producing NaN/inf.

const std = @import("std");

/// Pre-counted statistics for a single language. The module does not produce
/// these; the caller tokenizes its corpus and tallies them.
///
/// `units` should be Unicode codepoints (preferred — a script-neutral measure
/// of "how much text") but may be bytes if codepoints are unavailable; just be
/// consistent across all languages in one report.
pub const LangStat = struct {
    name: []const u8,
    tokens: u64,
    units: u64,

    /// tokens / units, or `null` when there are no units to divide by.
    pub fn fertility(self: LangStat) ?f64 {
        if (self.units == 0) return null;
        return @as(f64, @floatFromInt(self.tokens)) / @as(f64, @floatFromInt(self.units));
    }
};

/// Per-language fairness result. `name` aliases the input `LangStat.name`,
/// so it stays valid only as long as the caller's slices do.
pub const LangReport = struct {
    name: []const u8,
    tokens: u64,
    units: u64,
    /// tokens / units. Lower is better.
    fertility: f64,
    /// fertility / best_fertility. The baseline language is exactly 1.0;
    /// everyone else is >= 1.0 and reads as "pays N times as many tokens
    /// per unit of text as the most-compressible language".
    premium: f64,
};

/// Aggregate fairness report over all languages with non-zero units.
pub const Report = struct {
    /// Per-language detail, in the same order as the (kept) input. Owned by
    /// this struct; free with `deinit`.
    langs: []LangReport,
    /// Index into `langs` of the lowest-fertility (baseline) language.
    best_index: usize,
    /// Index into `langs` of the highest-fertility (worst-served) language.
    worst_index: usize,
    /// Lowest per-language fertility (the baseline used for premiums).
    best_fertility: f64,
    /// Highest per-language fertility.
    worst_fertility: f64,
    /// worst_fertility / best_fertility. 1.0 == every language equally served.
    max_min_ratio: f64,
    /// Arithmetic mean of per-language fertility.
    mean_fertility: f64,
    /// Gini coefficient of the fertility distribution. 0 == perfectly fair,
    /// approaches 1 as one language hogs all the disparity.
    gini: f64,

    pub fn deinit(self: *Report, allocator: std.mem.Allocator) void {
        allocator.free(self.langs);
        self.* = undefined;
    }

    /// Number of languages that contributed to the report (units > 0).
    pub fn count(self: Report) usize {
        return self.langs.len;
    }

    pub fn best(self: Report) LangReport {
        return self.langs[self.best_index];
    }

    pub fn worst(self: Report) LangReport {
        return self.langs[self.worst_index];
    }
};

/// Errors `analyze` can return.
pub const Error = error{
    /// No input language had a positive `units` count, so no fertility could
    /// be computed for anyone.
    NoData,
} || std.mem.Allocator.Error;

/// Build a fairness `Report` from per-language counts.
///
/// Languages with `units == 0` are silently skipped. If that leaves nothing,
/// returns `error.NoData`. The returned `Report` owns an allocation; call
/// `Report.deinit`.
pub fn analyze(allocator: std.mem.Allocator, stats: []const LangStat) Error!Report {
    // Filter to languages that carry information.
    var kept: std.ArrayList(LangReport) = .empty;
    errdefer kept.deinit(allocator);

    for (stats) |s| {
        const f = s.fertility() orelse continue; // skip units == 0
        try kept.append(allocator, .{
            .name = s.name,
            .tokens = s.tokens,
            .units = s.units,
            .fertility = f,
            .premium = 1.0, // filled in once we know the baseline
        });
    }

    if (kept.items.len == 0) {
        // `errdefer kept.deinit` above frees the (empty) buffer.
        return error.NoData;
    }

    const langs = try kept.toOwnedSlice(allocator);
    errdefer allocator.free(langs);

    // Locate best (min) and worst (max) fertility, accumulate the mean.
    var best_index: usize = 0;
    var worst_index: usize = 0;
    var sum: f64 = 0;
    for (langs, 0..) |l, i| {
        sum += l.fertility;
        if (l.fertility < langs[best_index].fertility) best_index = i;
        if (l.fertility > langs[worst_index].fertility) worst_index = i;
    }

    const n_f: f64 = @floatFromInt(langs.len);
    const mean_fertility = sum / n_f;
    const best_fertility = langs[best_index].fertility;
    const worst_fertility = langs[worst_index].fertility;

    // Premiums relative to the most-compressible language. best_fertility is
    // strictly positive here only if tokens > 0; guard the all-zero-token case
    // (every fertility 0) so we report premium 1.0 rather than 0/0.
    for (langs) |*l| {
        l.premium = if (best_fertility > 0) l.fertility / best_fertility else 1.0;
    }

    const max_min_ratio = if (best_fertility > 0)
        worst_fertility / best_fertility
    else
        1.0;

    return .{
        .langs = langs,
        .best_index = best_index,
        .worst_index = worst_index,
        .best_fertility = best_fertility,
        .worst_fertility = worst_fertility,
        .max_min_ratio = max_min_ratio,
        .mean_fertility = mean_fertility,
        .gini = giniCoefficient(langs),
    };
}

/// Gini coefficient of the fertility distribution.
///
/// Uses the mean-absolute-difference form:
///
///     G = ( sum_i sum_j |x_i - x_j| ) / ( 2 * n^2 * mean(x) )
///
/// which is 0 when every value is equal (perfect fairness) and rises toward 1
/// as the distribution concentrates. With a single value the sum is 0, so a
/// lone language is trivially fair (G = 0). If the mean is 0 (all fertilities
/// zero) the disparity is undefined-but-trivially-fair, so we return 0.
fn giniCoefficient(langs: []const LangReport) f64 {
    const n = langs.len;
    if (n <= 1) return 0;

    var sum: f64 = 0;
    var abs_diff_sum: f64 = 0;
    for (langs, 0..) |a, i| {
        sum += a.fertility;
        // j > i then double, avoiding the j == i zero terms.
        var j: usize = i + 1;
        while (j < n) : (j += 1) {
            abs_diff_sum += @abs(a.fertility - langs[j].fertility);
        }
    }
    abs_diff_sum *= 2; // we only summed the upper triangle

    const n_f: f64 = @floatFromInt(n);
    const mean = sum / n_f;
    if (mean == 0) return 0;

    return abs_diff_sum / (2.0 * n_f * n_f * mean);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

/// `a` and `b` agree to within `eps` (absolute).
fn approx(a: f64, b: f64, eps: f64) bool {
    return @abs(a - b) <= eps;
}

test "empty input returns NoData" {
    try testing.expectError(error.NoData, analyze(testing.allocator, &.{}));
}

test "all-zero-units input returns NoData" {
    const stats = [_]LangStat{
        .{ .name = "en", .tokens = 100, .units = 0 },
        .{ .name = "ja", .tokens = 50, .units = 0 },
    };
    try testing.expectError(error.NoData, analyze(testing.allocator, &stats));
}

test "single language: gini 0, premium 1, ratio 1" {
    const stats = [_]LangStat{
        .{ .name = "en", .tokens = 300, .units = 1000 },
    };
    var r = try analyze(testing.allocator, &stats);
    defer r.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 1), r.count());
    try testing.expect(approx(r.langs[0].fertility, 0.3, 1e-12));
    try testing.expect(approx(r.langs[0].premium, 1.0, 1e-12));
    try testing.expect(approx(r.best_fertility, 0.3, 1e-12));
    try testing.expect(approx(r.worst_fertility, 0.3, 1e-12));
    try testing.expect(approx(r.max_min_ratio, 1.0, 1e-12));
    try testing.expect(approx(r.mean_fertility, 0.3, 1e-12));
    try testing.expect(approx(r.gini, 0.0, 1e-12));
    try testing.expectEqual(@as(usize, 0), r.best_index);
    try testing.expectEqual(@as(usize, 0), r.worst_index);
}

test "languages with zero units are skipped, not fatal" {
    const stats = [_]LangStat{
        .{ .name = "en", .tokens = 100, .units = 500 }, // fertility 0.2
        .{ .name = "skipme", .tokens = 9, .units = 0 }, // dropped
        .{ .name = "de", .tokens = 200, .units = 500 }, // fertility 0.4
    };
    var r = try analyze(testing.allocator, &stats);
    defer r.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 2), r.count());
    try testing.expectEqualStrings("en", r.langs[0].name);
    try testing.expectEqualStrings("de", r.langs[1].name);
}

test "premiums and max/min ratio against a known baseline" {
    // en is the most compressible (lowest fertility) so it is the baseline.
    const stats = [_]LangStat{
        .{ .name = "en", .tokens = 100, .units = 1000 }, // 0.10  (baseline)
        .{ .name = "fr", .tokens = 150, .units = 1000 }, // 0.15  -> 1.5x
        .{ .name = "th", .tokens = 400, .units = 1000 }, // 0.40  -> 4.0x  (worst)
    };
    var r = try analyze(testing.allocator, &stats);
    defer r.deinit(testing.allocator);

    try testing.expectEqual(@as(usize, 0), r.best_index);
    try testing.expectEqual(@as(usize, 2), r.worst_index);
    try testing.expect(approx(r.best_fertility, 0.10, 1e-12));
    try testing.expect(approx(r.worst_fertility, 0.40, 1e-12));

    try testing.expect(approx(r.langs[0].premium, 1.0, 1e-12));
    try testing.expect(approx(r.langs[1].premium, 1.5, 1e-12));
    try testing.expect(approx(r.langs[2].premium, 4.0, 1e-12));

    // worst / best = 0.40 / 0.10 = 4.0
    try testing.expect(approx(r.max_min_ratio, 4.0, 1e-12));
    // mean of {0.10, 0.15, 0.40} = 0.65/3
    try testing.expect(approx(r.mean_fertility, 0.65 / 3.0, 1e-12));
}

test "hand-computed Gini for {1,2,3}" {
    // Choose units so fertility == {1, 2, 3} exactly.
    // For x = {1,2,3}: mean = 2.
    //   sum_{i<j} |xi-xj| = |1-2|+|1-3|+|2-3| = 1+2+1 = 4
    //   full double sum   = 2 * 4 = 8
    //   G = 8 / (2 * n^2 * mean) = 8 / (2 * 9 * 2) = 8/36 = 0.2222...
    const stats = [_]LangStat{
        .{ .name = "a", .tokens = 1, .units = 1 }, // 1.0
        .{ .name = "b", .tokens = 2, .units = 1 }, // 2.0
        .{ .name = "c", .tokens = 3, .units = 1 }, // 3.0
    };
    var r = try analyze(testing.allocator, &stats);
    defer r.deinit(testing.allocator);

    try testing.expect(approx(r.gini, 8.0 / 36.0, 1e-12));
    try testing.expect(approx(r.mean_fertility, 2.0, 1e-12));
    try testing.expect(approx(r.max_min_ratio, 3.0, 1e-12)); // 3/1
}

test "perfectly equal fertilities give Gini 0 and ratio 1" {
    const stats = [_]LangStat{
        .{ .name = "a", .tokens = 30, .units = 100 },
        .{ .name = "b", .tokens = 60, .units = 200 },
        .{ .name = "c", .tokens = 90, .units = 300 },
    };
    var r = try analyze(testing.allocator, &stats);
    defer r.deinit(testing.allocator);

    try testing.expect(approx(r.gini, 0.0, 1e-12));
    try testing.expect(approx(r.max_min_ratio, 1.0, 1e-12));
    for (r.langs) |l| try testing.expect(approx(l.premium, 1.0, 1e-12));
}

test "all-zero-token languages: defined, fair, no division by zero" {
    const stats = [_]LangStat{
        .{ .name = "a", .tokens = 0, .units = 100 },
        .{ .name = "b", .tokens = 0, .units = 200 },
    };
    var r = try analyze(testing.allocator, &stats);
    defer r.deinit(testing.allocator);

    try testing.expect(approx(r.best_fertility, 0.0, 1e-12));
    try testing.expect(approx(r.gini, 0.0, 1e-12));
    try testing.expect(approx(r.max_min_ratio, 1.0, 1e-12));
    for (r.langs) |l| try testing.expect(approx(l.premium, 1.0, 1e-12));
}

test "LangStat.fertility guards zero units" {
    try testing.expectEqual(@as(?f64, null), (LangStat{ .name = "x", .tokens = 5, .units = 0 }).fertility());
    try testing.expect(approx((LangStat{ .name = "x", .tokens = 5, .units = 10 }).fertility().?, 0.5, 1e-12));
}
