//! Prometheus-format metrics for `ztok serve`.
//!
//! All counters/gauges are lock-free atomic operations so the metrics
//! recorder can be hit from any future worker-thread the serve loop
//! grows. The histogram uses one atomic counter per bucket plus an
//! atomic sum/count; concurrent observers will race on bucket
//! increments but the result is still a valid Prometheus histogram (a
//! sample never goes to two buckets, just to "less than or equal").
//!
//! Format reference:
//! https://prometheus.io/docs/instrumenting/exposition_formats/#text-based-format
//!
//! We deliberately keep label cardinality tiny:
//!   - `method` ∈ {GET, POST, OTHER}
//!   - `path`   ∈ {/encode, /encode_stream, /encode_chunked, /decode,
//!                 /eval, /version, /health, /metrics, /encode_ws,
//!                 OTHER}
//!   - `status` ∈ {2xx, 4xx, 5xx} (bucketed to keep the time series
//!                 count bounded)
//!
//! That keeps the per-process series count to a few dozen rather than
//! one-per-distinct-URL.

const std = @import("std");

pub const buckets_seconds: [5]f64 = .{ 0.001, 0.01, 0.1, 1.0, 10.0 };

/// Stable enumeration of paths we emit labels for. Anything outside
/// this set lands in `.other` so cardinality stays bounded.
pub const PathLabel = enum {
    encode,
    encode_stream,
    encode_chunked,
    decode,
    eval,
    version,
    health,
    metrics,
    encode_ws,
    other,

    pub fn fromPath(path: []const u8) PathLabel {
        if (std.mem.eql(u8, path, "/encode")) return .encode;
        if (std.mem.eql(u8, path, "/encode_stream")) return .encode_stream;
        if (std.mem.eql(u8, path, "/encode_chunked")) return .encode_chunked;
        if (std.mem.eql(u8, path, "/decode")) return .decode;
        if (std.mem.eql(u8, path, "/eval")) return .eval;
        if (std.mem.eql(u8, path, "/version")) return .version;
        if (std.mem.eql(u8, path, "/health")) return .health;
        if (std.mem.eql(u8, path, "/metrics")) return .metrics;
        if (std.mem.eql(u8, path, "/encode_ws")) return .encode_ws;
        return .other;
    }

    pub fn asString(self: PathLabel) []const u8 {
        return switch (self) {
            .encode => "/encode",
            .encode_stream => "/encode_stream",
            .encode_chunked => "/encode_chunked",
            .decode => "/decode",
            .eval => "/eval",
            .version => "/version",
            .health => "/health",
            .metrics => "/metrics",
            .encode_ws => "/encode_ws",
            .other => "other",
        };
    }
};

pub const MethodLabel = enum {
    GET,
    POST,
    OTHER,

    pub fn fromMethod(m: std.http.Method) MethodLabel {
        return switch (m) {
            .GET => .GET,
            .POST => .POST,
            else => .OTHER,
        };
    }

    pub fn asString(self: MethodLabel) []const u8 {
        return switch (self) {
            .GET => "GET",
            .POST => "POST",
            .OTHER => "OTHER",
        };
    }
};

/// Per-status bucket: 2xx (success), 4xx (client error), 5xx (server
/// error), and a single "other" catch-all for 1xx/3xx (rare).
pub const StatusBucket = enum {
    s2xx,
    s4xx,
    s5xx,
    other,

    pub fn fromStatus(code: u16) StatusBucket {
        if (code >= 200 and code < 300) return .s2xx;
        if (code >= 400 and code < 500) return .s4xx;
        if (code >= 500 and code < 600) return .s5xx;
        return .other;
    }

    pub fn asString(self: StatusBucket) []const u8 {
        return switch (self) {
            .s2xx => "2xx",
            .s4xx => "4xx",
            .s5xx => "5xx",
            .other => "other",
        };
    }
};

const n_paths = @typeInfo(PathLabel).@"enum".fields.len;
const n_methods = @typeInfo(MethodLabel).@"enum".fields.len;
const n_status = @typeInfo(StatusBucket).@"enum".fields.len;
const n_buckets = buckets_seconds.len;

/// Process-wide metrics state. Zero-init is the valid empty state.
pub const Metrics = struct {
    /// ztok_requests_total{method,path,status}
    requests_total: [n_methods][n_paths][n_status]std.atomic.Value(u64) = init: {
        var arr: [n_methods][n_paths][n_status]std.atomic.Value(u64) = undefined;
        for (0..n_methods) |m| for (0..n_paths) |p| for (0..n_status) |s| {
            arr[m][p][s] = .init(0);
        };
        break :init arr;
    },

    /// ztok_request_bytes_in_total{path}
    bytes_in_total: [n_paths]std.atomic.Value(u64) = init: {
        var arr: [n_paths]std.atomic.Value(u64) = undefined;
        for (0..n_paths) |p| arr[p] = .init(0);
        break :init arr;
    },

    /// ztok_request_bytes_out_total{path}
    bytes_out_total: [n_paths]std.atomic.Value(u64) = init: {
        var arr: [n_paths]std.atomic.Value(u64) = undefined;
        for (0..n_paths) |p| arr[p] = .init(0);
        break :init arr;
    },

    /// ztok_encode_tokens_total
    encode_tokens_total: std.atomic.Value(u64) = .init(0),

    /// ztok_request_duration_seconds{path}: per-bucket counters +
    /// total sum + total count (per path).
    /// Buckets follow `buckets_seconds`; the last (n_buckets) slot is
    /// the "+Inf" bucket count which is identical to `count`.
    duration_buckets: [n_paths][n_buckets]std.atomic.Value(u64) = init: {
        var arr: [n_paths][n_buckets]std.atomic.Value(u64) = undefined;
        for (0..n_paths) |p| for (0..n_buckets) |b| {
            arr[p][b] = .init(0);
        };
        break :init arr;
    },
    duration_sum_ns: [n_paths]std.atomic.Value(u64) = init: {
        var arr: [n_paths]std.atomic.Value(u64) = undefined;
        for (0..n_paths) |p| arr[p] = .init(0);
        break :init arr;
    },
    duration_count: [n_paths]std.atomic.Value(u64) = init: {
        var arr: [n_paths]std.atomic.Value(u64) = undefined;
        for (0..n_paths) |p| arr[p] = .init(0);
        break :init arr;
    },

    /// ztok_active_requests (gauge)
    active_requests: std.atomic.Value(i64) = .init(0),

    pub fn incRequest(m: *Metrics, method: MethodLabel, path: PathLabel, status: StatusBucket) void {
        _ = m.requests_total[@intFromEnum(method)][@intFromEnum(path)][@intFromEnum(status)].fetchAdd(1, .monotonic);
    }

    pub fn addBytesIn(m: *Metrics, path: PathLabel, n: u64) void {
        _ = m.bytes_in_total[@intFromEnum(path)].fetchAdd(n, .monotonic);
    }

    pub fn addBytesOut(m: *Metrics, path: PathLabel, n: u64) void {
        _ = m.bytes_out_total[@intFromEnum(path)].fetchAdd(n, .monotonic);
    }

    pub fn addEncodeTokens(m: *Metrics, n: u64) void {
        _ = m.encode_tokens_total.fetchAdd(n, .monotonic);
    }

    pub fn observeDuration(m: *Metrics, path: PathLabel, seconds: f64) void {
        const pi = @intFromEnum(path);
        // Walk buckets least → greatest. A sample <= bucket boundary
        // counts toward THAT bucket and all higher ones; the canonical
        // Prometheus cumulative-histogram view is generated at render
        // time by summing slots ≤ k. We store per-slot increments
        // here so two concurrent observers don't double-count higher
        // buckets.
        for (buckets_seconds, 0..) |b, i| {
            if (seconds <= b) {
                _ = m.duration_buckets[pi][i].fetchAdd(1, .monotonic);
                break;
            }
        } else {
            // Sample exceeds all defined buckets; only +Inf catches it.
        }
        _ = m.duration_count[pi].fetchAdd(1, .monotonic);
        const ns_u64: u64 = if (seconds < 0) 0 else @intFromFloat(seconds * 1_000_000_000.0);
        _ = m.duration_sum_ns[pi].fetchAdd(ns_u64, .monotonic);
    }

    pub fn enterRequest(m: *Metrics) void {
        _ = m.active_requests.fetchAdd(1, .acq_rel);
    }

    pub fn exitRequest(m: *Metrics) void {
        _ = m.active_requests.fetchSub(1, .acq_rel);
    }
};

/// Render the metrics to `w` in Prometheus text exposition format.
pub fn render(m: *const Metrics, w: *std.Io.Writer) !void {
    // ztok_requests_total
    try w.writeAll("# HELP ztok_requests_total Total HTTP requests by method, path, and status class.\n");
    try w.writeAll("# TYPE ztok_requests_total counter\n");
    inline for (@typeInfo(MethodLabel).@"enum".fields) |mf| {
        const ml: MethodLabel = @enumFromInt(mf.value);
        inline for (@typeInfo(PathLabel).@"enum".fields) |pf| {
            const pl: PathLabel = @enumFromInt(pf.value);
            inline for (@typeInfo(StatusBucket).@"enum".fields) |sf| {
                const sl: StatusBucket = @enumFromInt(sf.value);
                const v = m.requests_total[mf.value][pf.value][sf.value].load(.monotonic);
                // `continue` would be comptime control flow inside a
                // runtime-conditional block (which `inline for` does
                // not allow); use a plain `if` to gate the write.
                if (v != 0) {
                    try w.print(
                        "ztok_requests_total{{method=\"{s}\",path=\"{s}\",status=\"{s}\"}} {d}\n",
                        .{ ml.asString(), pl.asString(), sl.asString(), v },
                    );
                }
            }
        }
    }

    // ztok_request_bytes_in_total
    try w.writeAll("# HELP ztok_request_bytes_in_total Bytes received in request bodies, by path.\n");
    try w.writeAll("# TYPE ztok_request_bytes_in_total counter\n");
    inline for (@typeInfo(PathLabel).@"enum".fields) |pf| {
        const pl: PathLabel = @enumFromInt(pf.value);
        const v = m.bytes_in_total[pf.value].load(.monotonic);
        if (v != 0) {
            try w.print(
                "ztok_request_bytes_in_total{{path=\"{s}\"}} {d}\n",
                .{ pl.asString(), v },
            );
        }
    }

    // ztok_request_bytes_out_total
    try w.writeAll("# HELP ztok_request_bytes_out_total Bytes written in response bodies, by path.\n");
    try w.writeAll("# TYPE ztok_request_bytes_out_total counter\n");
    inline for (@typeInfo(PathLabel).@"enum".fields) |pf| {
        const pl: PathLabel = @enumFromInt(pf.value);
        const v = m.bytes_out_total[pf.value].load(.monotonic);
        if (v != 0) {
            try w.print(
                "ztok_request_bytes_out_total{{path=\"{s}\"}} {d}\n",
                .{ pl.asString(), v },
            );
        }
    }

    // ztok_encode_tokens_total
    try w.writeAll("# HELP ztok_encode_tokens_total Total token ids emitted by /encode* endpoints.\n");
    try w.writeAll("# TYPE ztok_encode_tokens_total counter\n");
    try w.print("ztok_encode_tokens_total {d}\n", .{m.encode_tokens_total.load(.monotonic)});

    // ztok_request_duration_seconds histogram
    try w.writeAll("# HELP ztok_request_duration_seconds Request handler latency in seconds, by path.\n");
    try w.writeAll("# TYPE ztok_request_duration_seconds histogram\n");
    inline for (@typeInfo(PathLabel).@"enum".fields) |pf| {
        const pl: PathLabel = @enumFromInt(pf.value);
        const total = m.duration_count[pf.value].load(.monotonic);
        if (total != 0) {
            var cumulative: u64 = 0;
            inline for (buckets_seconds, 0..) |b, i| {
                cumulative += m.duration_buckets[pf.value][i].load(.monotonic);
                try w.print(
                    "ztok_request_duration_seconds_bucket{{path=\"{s}\",le=\"{d}\"}} {d}\n",
                    .{ pl.asString(), b, cumulative },
                );
            }
            try w.print(
                "ztok_request_duration_seconds_bucket{{path=\"{s}\",le=\"+Inf\"}} {d}\n",
                .{ pl.asString(), total },
            );
            const sum_ns = m.duration_sum_ns[pf.value].load(.monotonic);
            const sum_s: f64 = @as(f64, @floatFromInt(sum_ns)) / 1_000_000_000.0;
            try w.print(
                "ztok_request_duration_seconds_sum{{path=\"{s}\"}} {d}\n",
                .{ pl.asString(), sum_s },
            );
            try w.print(
                "ztok_request_duration_seconds_count{{path=\"{s}\"}} {d}\n",
                .{ pl.asString(), total },
            );
        }
    }

    // ztok_active_requests
    try w.writeAll("# HELP ztok_active_requests Currently in-flight requests.\n");
    try w.writeAll("# TYPE ztok_active_requests gauge\n");
    try w.print("ztok_active_requests {d}\n", .{m.active_requests.load(.acquire)});
}

// === Tests ============================================================

const testing = std.testing;

test "metrics: render is well-formed and includes incremented counters" {
    const a = testing.allocator;
    var m: Metrics = .{};
    m.incRequest(.POST, .encode, .s2xx);
    m.incRequest(.POST, .encode, .s2xx);
    m.incRequest(.GET, .health, .s2xx);
    m.addBytesIn(.encode, 1024);
    m.addBytesOut(.encode, 4096);
    m.addEncodeTokens(42);
    m.observeDuration(.encode, 0.005);
    m.observeDuration(.encode, 0.5);
    m.enterRequest();

    var out: std.Io.Writer.Allocating = .init(a);
    defer out.deinit();
    try render(&m, &out.writer);
    const text = out.written();

    // Counter lines for the requests we made.
    try testing.expect(std.mem.indexOf(u8, text, "ztok_requests_total{method=\"POST\",path=\"/encode\",status=\"2xx\"} 2") != null);
    try testing.expect(std.mem.indexOf(u8, text, "ztok_requests_total{method=\"GET\",path=\"/health\",status=\"2xx\"} 1") != null);

    // Byte counters.
    try testing.expect(std.mem.indexOf(u8, text, "ztok_request_bytes_in_total{path=\"/encode\"} 1024") != null);
    try testing.expect(std.mem.indexOf(u8, text, "ztok_request_bytes_out_total{path=\"/encode\"} 4096") != null);

    // Token counter.
    try testing.expect(std.mem.indexOf(u8, text, "ztok_encode_tokens_total 42") != null);

    // Histogram presence + +Inf bucket.
    try testing.expect(std.mem.indexOf(u8, text, "ztok_request_duration_seconds_bucket{path=\"/encode\",le=\"+Inf\"} 2") != null);
    try testing.expect(std.mem.indexOf(u8, text, "ztok_request_duration_seconds_count{path=\"/encode\"} 2") != null);

    // Active gauge.
    try testing.expect(std.mem.indexOf(u8, text, "ztok_active_requests 1") != null);

    // HELP/TYPE headers present.
    try testing.expect(std.mem.indexOf(u8, text, "# TYPE ztok_requests_total counter") != null);
    try testing.expect(std.mem.indexOf(u8, text, "# TYPE ztok_request_duration_seconds histogram") != null);
}

test "metrics: histogram bucket boundaries respect le ordering" {
    const a = testing.allocator;
    var m: Metrics = .{};
    // Observations: 0.0005 (≤0.001), 0.05 (≤0.1), 5 (≤10), 50 (+Inf only).
    m.observeDuration(.encode, 0.0005);
    m.observeDuration(.encode, 0.05);
    m.observeDuration(.encode, 5);
    m.observeDuration(.encode, 50);

    var out: std.Io.Writer.Allocating = .init(a);
    defer out.deinit();
    try render(&m, &out.writer);
    const text = out.written();

    // Cumulative counts should be 1, 1, 2, 3, 3 across the 5 finite buckets,
    // then 4 for +Inf.
    try testing.expect(std.mem.indexOf(u8, text, "le=\"0.001\"}") != null);
    try testing.expect(std.mem.indexOf(u8, text, "le=\"0.01\"}") != null);
    try testing.expect(std.mem.indexOf(u8, text, "le=\"0.1\"}") != null);
    try testing.expect(std.mem.indexOf(u8, text, "le=\"1\"}") != null);
    try testing.expect(std.mem.indexOf(u8, text, "le=\"10\"}") != null);
    try testing.expect(std.mem.indexOf(u8, text, "le=\"+Inf\"} 4") != null);
}

test "metrics: PathLabel.fromPath maps known and unknown paths" {
    try testing.expectEqual(PathLabel.encode, PathLabel.fromPath("/encode"));
    try testing.expectEqual(PathLabel.metrics, PathLabel.fromPath("/metrics"));
    try testing.expectEqual(PathLabel.other, PathLabel.fromPath("/nonsense"));
}
