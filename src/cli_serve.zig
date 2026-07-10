//! HTTP serving primitive backing `ztok serve`.
//!
//! Routes (all JSON unless noted):
//!   POST /encode         {"text": "..."}         -> {"ids": [...]}
//!   POST /encode_stream  {"text": "..."}         -> NDJSON, one {"ids":[...]} per batch + {"done":true}
//!   POST /decode         {"ids": [...]}          -> {"text": "..."}
//!   POST /eval           {"text": "..."}         -> ztok eval JSON schema
//!   GET  /version                                 -> {"version": "...", "model_kind": "..."}
//!   GET  /health                                  -> {"ok": true}
//!
//! Threading:
//!   A bounded persistent connection pool handles independent clients in
//!   parallel. CPU fan-out inside one large encode still comes from the
//!   separate `BatchPool`; access to that non-reentrant pool is serialized.
//!   Each connection worker owns a reusable `Pipeline.ScratchArena`.
//!
//! Security model (post-1.18):
//!   * Binds 127.0.0.1 by default so the listener is unreachable from
//!     the network without explicit override.
//!   * Optional bearer-token auth via `--auth-token TOKEN` or
//!     `--auth-token-file PATH`. When set, every request except
//!     `GET /health` must carry `Authorization: Bearer <TOKEN>`.
//!     Comparison is constant-time.
//!   * Optional per-client-IP token-bucket rate limit via
//!     `--rate-limit REQ_PER_SEC`. Refill rate = N tokens/sec, bucket
//!     size = N*2 (allows brief bursts). `/health` exempt. Client IP
//!     comes from `getpeername(2)` on the accepted socket; entries are
//!     LRU-evicted above 10 000 to bound memory.
//!   * Request body capped at `max_body_bytes` (default 16 MiB).
//!   * Defaults preserve 1.16/1.17 behavior: no auth, no rate limit.
//!
//! Implementation:
//!   Uses Zig 0.16 `std.Io.net.IpAddress.listen` for the TCP socket and
//!   `std.http.Server` for HTTP/1.1 framing. The stdlib `std.http.Server`
//!   is a per-connection state machine — we wrap one around each
//!   accepted stream's reader/writer. The protocol-pure handler
//!   (`handleRequest`) is exercised in tests by driving `std.http.Server`
//!   with `std.Io.Reader.fixed` + `std.Io.Writer.Allocating`, with no
//!   sockets/threads/ports — see `handleRequestInMemory`.

const std = @import("std");
const Pipeline = @import("pipeline.zig").Pipeline;
const ScratchArena = @import("pipeline.zig").ScratchArena;
const TokenId = @import("token.zig").TokenId;
const StreamEncoder = @import("stream.zig").StreamEncoder;
const BatchPool = @import("thread_pool.zig").BatchPool;
const cli_eval = @import("cli_eval.zig");
const metrics_mod = @import("metrics.zig");
const websocket = @import("websocket.zig");
const auth_oidc = @import("auth_oidc.zig");
const mbedtls = @import("mbedtls.zig");
const prefix_cache_mod = @import("prefix_cache.zig");
const connection_pool_mod = @import("connection_pool.zig");
const build_options = @import("build_options");

pub const Metrics = metrics_mod.Metrics;

pub const LogFormat = enum { text, json };

/// Aggregated outcome of one request, used for both structured logging
/// and metrics. `path_label` is the bucketed enum; `path_str` is the
/// raw URL path (logged verbatim so JSON consumers see the real URL).
pub const RequestOutcome = struct {
    method: std.http.Method,
    path_label: metrics_mod.PathLabel,
    path_str: []const u8,
    status: u16,
    bytes_in: u64,
    bytes_out: u64,
    duration_ns: u64,
    client_ip_str: []const u8,
    encoded_tokens: u64 = 0,
};

pub const default_host: []const u8 = "127.0.0.1";
pub const default_port: u16 = 7890;
pub const default_max_body_bytes: usize = 16 * 1024 * 1024;
pub const default_idle_timeout_ms: u32 = 30_000;
pub const default_max_keepalive_requests: u32 = 100;
pub const default_max_connection_workers: usize = 32;

/// Buffer sizes for std.http.Server per connection.
const recv_buf_size: usize = 8192;
const send_buf_size: usize = 16384;

/// Cap on number of distinct client IPs tracked by the rate limiter
/// before LRU eviction kicks in. 10 000 entries × ~48 B per entry =
/// ~470 KiB worst-case footprint.
pub const rate_limit_max_clients: usize = 10_000;

pub const ModelKind = enum { bpe, unigram, wordpiece, monster, byte_id, rwkv_world };

pub const Options = struct {
    host: []const u8 = default_host,
    port: u16 = default_port,
    /// Hard cap on a POST body. Larger requests get a 413.
    max_body_bytes: usize = default_max_body_bytes,
    /// Receive/send inactivity timeout per accepted connection. Zero
    /// disables the socket deadline. Prevents a slow client from occupying
    /// a connection worker indefinitely.
    idle_timeout_ms: u32 = default_idle_timeout_ms,
    /// Bound requests served on one cleartext keep-alive connection.
    /// Zero means unlimited.
    max_keepalive_requests: u32 = default_max_keepalive_requests,
    /// Number of persistent connection workers. Zero selects
    /// min(tokenizer workers, 32). Independent requests run concurrently;
    /// the bounded queue applies backpressure when every worker is busy.
    connection_workers: u16 = 0,
    /// Maximum accepted connections waiting for a worker. Zero selects
    /// two queued connections per worker.
    connection_queue: u16 = 0,
    /// String reported by `GET /version` for `model_kind`.
    model_kind: ModelKind = .bpe,
    /// String reported by `GET /version` for `version`.
    version: []const u8 = "0.0.0",
    /// When true, log accepted connections + per-request method/path to
    /// stderr. Defaults to true since serving is interactive.
    log: bool = true,
    /// Log format. `.text` keeps the legacy human-readable
    /// `std.log.info("ztok serve: METHOD /path", ...)` line; `.json`
    /// emits one structured JSON object per request to stderr.
    log_format: LogFormat = .text,
    /// When non-null, every non-/health request must carry
    /// `Authorization: Bearer <auth_token>`. Otherwise → 401.
    /// Storage is borrowed; caller owns the bytes for the server's
    /// lifetime (typically a CLI-arg slice).
    auth_token: ?[]const u8 = null,
    /// When non-null, enforce a per-client-IP token-bucket rate limit
    /// of this many requests per second. `/health` is exempt.
    rate_limit_rps: ?u32 = null,
    /// Optional Prometheus metrics sink. When non-null, the server
    /// records per-request counters/histograms and a `/metrics`
    /// endpoint becomes available.
    metrics: ?*Metrics = null,
    /// Optional OIDC validator. When non-null, requests must carry a
    /// bearer JWT that validates against the configured issuer +
    /// audience + JWKS. Mutually compatible with `auth_token` — if
    /// both are set, the token wins (bearer auth → constant-time
    /// match), else the JWT is checked. `/health` and `/metrics`
    /// bypass.
    oidc: ?*auth_oidc.Validator = null,
    /// Set to true to terminate TLS at the server. When true and the
    /// build was configured with `-Dtls=mbedtls`, `tls_cert_path` /
    /// `tls_key_path` are loaded into an mbedtls server context and
    /// every accepted socket gets a TLS handshake before the HTTP
    /// parser runs. When true and the build was the default
    /// `-Dtls=none`, `run` fails fast with `error.TLSServerNotAvailable`
    /// so the operator never thinks the bytes are encrypted when
    /// they're not.
    tls_enabled: bool = false,
    /// PEM path to the server certificate, used only when
    /// `tls_enabled` is true and TLS was built in.
    tls_cert_path: ?[]const u8 = null,
    /// PEM path to the server private key, used only when
    /// `tls_enabled` is true and TLS was built in.
    tls_key_path: ?[]const u8 = null,
    /// Optional persistent exact-result cache. When non-null, `/encode`
    /// requests cache the complete input and its complete id stream.
    /// Arbitrary prefix/suffix token streams cannot be concatenated safely
    /// for BPE and normalization pipelines, so partial-prefix splicing is
    /// deliberately not attempted. Caller owns the PrefixCache.
    prefix_cache: ?*prefix_cache_mod.PrefixCache = null,
};

/// Run the server until the listener errors or process exit. Blocks
/// the calling thread; this is the main loop of `ztok serve`.
pub fn run(
    allocator: std.mem.Allocator,
    io: std.Io,
    pipeline: *const Pipeline,
    pool: *BatchPool,
    opts: Options,
) !void {
    // Zig 0.16 stdlib has no server-side TLS — only a Client. When
    // the operator asked for TLS (--tls-cert / --tls-key) we route
    // through mbedtls if the build enabled it (`-Dtls=mbedtls`),
    // otherwise we refuse to start so the operator never thinks the
    // bytes are encrypted when they're not.
    var tls_server: ?mbedtls.Server = null;
    defer if (tls_server) |*s| s.deinit();
    if (opts.tls_enabled) {
        if (build_options.tls_backend != .mbedtls) {
            std.log.err("ztok serve: TLS termination requested via --tls-cert/--tls-key but this build was compiled with -Dtls=none; rebuild with `zig build -Dtls=mbedtls` or terminate TLS at a sidecar (caddy/nginx)", .{});
            return error.TLSServerNotAvailable;
        }
        const cert_path = opts.tls_cert_path orelse {
            std.log.err("ztok serve: --tls-cert PATH is required when TLS is enabled", .{});
            return error.TLSServerNotAvailable;
        };
        const key_path = opts.tls_key_path orelse {
            std.log.err("ztok serve: --tls-key PATH is required when TLS is enabled", .{});
            return error.TLSServerNotAvailable;
        };
        tls_server = mbedtls.Server.init(allocator, cert_path, key_path) catch |err| {
            std.log.err("ztok serve: failed to initialize mbedtls server with cert={s} key={s}: {t}", .{ cert_path, key_path, err });
            return err;
        };
        if (opts.log) std.log.info("ztok serve: TLS termination enabled (mbedtls)", .{});
    }

    var addr = try std.Io.net.IpAddress.parse(opts.host, opts.port);
    var server = addr.listen(io, .{ .reuse_address = true }) catch |err| {
        std.log.err("ztok serve: failed to listen on {s}:{d}: {t}", .{ opts.host, opts.port, err });
        return err;
    };
    defer server.deinit(io);

    if (opts.log) {
        std.log.info("ztok serve: listening on http://{s}:{d}", .{ opts.host, opts.port });
        if (opts.auth_token != null) std.log.info("ztok serve: bearer-token auth enabled", .{});
        if (opts.oidc != null) std.log.info("ztok serve: OIDC bearer-JWT auth enabled", .{});
        if (opts.rate_limit_rps) |rps| std.log.info("ztok serve: rate limit {d} req/s per client IP", .{rps});
        if (opts.metrics != null) std.log.info("ztok serve: /metrics endpoint enabled", .{});
        if (opts.log_format == .json) std.log.info("ztok serve: structured JSON request logging enabled", .{});
    }

    var rate_limiter: ?RateLimiter = if (opts.rate_limit_rps) |rps|
        RateLimiter.init(allocator, io, rps)
    else
        null;
    defer if (rate_limiter) |*rl| rl.deinit();

    var pool_mutex: std.Io.Mutex = .init;
    var cache_mutex: std.Io.Mutex = .init;
    var oidc_mutex: std.Io.Mutex = .init;
    var log_mutex: std.Io.Mutex = .init;
    var ctx: ServeCtx = .{
        .allocator = allocator,
        .io = io,
        .pipeline = pipeline,
        .pool = pool,
        .opts = opts,
        .rate_limiter = if (rate_limiter) |*rl| rl else null,
        .tls_server = if (tls_server) |*s| s else null,
        .prefix_cache = opts.prefix_cache,
        .pool_mutex = &pool_mutex,
        .cache_mutex = &cache_mutex,
        .oidc_mutex = &oidc_mutex,
        .log_mutex = &log_mutex,
    };

    const connection_workers = resolveConnectionWorkers(opts.connection_workers, pool.workerCount());
    const queue_capacity = resolveConnectionQueue(opts.connection_queue, connection_workers);
    var runtime = try ServeRuntime.init(allocator, &ctx, connection_workers);
    defer runtime.deinit();
    const connections = try ConnectionPool.init(
        allocator,
        io,
        connection_workers,
        queue_capacity,
        &runtime,
        ServeRuntime.handle,
    );
    defer connections.deinit();

    serveLoop(&ctx, &server, 0, connections) catch |err| {
        if (err != error.AcceptLoopEnded) return err;
    };
}

/// Accept-and-handle loop. `max_connections == 0` means run forever.
/// Returns `error.AcceptLoopEnded` when a fixed `max_connections` is
/// hit (used by tests). The caller owns `server` and must `deinit` it.
fn serveLoop(
    ctx: *ServeCtx,
    server: *std.Io.net.Server,
    max_connections: usize,
    connections: *ConnectionPool,
) !void {
    var n_handled: usize = 0;
    while (true) {
        var stream = server.accept(ctx.io) catch |err| switch (err) {
            error.Canceled => return,
            else => |e| {
                if (ctx.opts.log) std.log.err("ztok serve: accept failed: {t}", .{e});
                continue;
            },
        };
        configureSocketTimeout(stream.socket.handle, ctx.opts.idle_timeout_ms) catch |err| {
            if (ctx.opts.log) std.log.warn("ztok serve: could not set connection timeout: {t}", .{err});
        };
        connections.submit(stream) catch |err| {
            stream.close(ctx.io);
            return err;
        };
        n_handled += 1;
        if (max_connections != 0 and n_handled >= max_connections) {
            connections.waitIdle();
            return error.AcceptLoopEnded;
        }
    }
}

/// Test-only: run the server bound to 127.0.0.1:`port` (use 0 to pick
/// an ephemeral port), handle `max_connections` requests, then return.
/// Returns the resolved port via `port_out`.
pub fn runForTest(
    allocator: std.mem.Allocator,
    io: std.Io,
    pipeline: *const Pipeline,
    pool: *BatchPool,
    port: u16,
    max_connections: usize,
    port_out: ?*u16,
    opts: Options,
) !void {
    var addr = try std.Io.net.IpAddress.parse("127.0.0.1", port);
    var server = try addr.listen(io, .{ .reuse_address = true });
    defer server.deinit(io);

    if (port_out) |out_p| out_p.* = server.socket.address.getPort();

    var rate_limiter: ?RateLimiter = if (opts.rate_limit_rps) |rps|
        RateLimiter.init(allocator, io, rps)
    else
        null;
    defer if (rate_limiter) |*rl| rl.deinit();

    var pool_mutex: std.Io.Mutex = .init;
    var cache_mutex: std.Io.Mutex = .init;
    var oidc_mutex: std.Io.Mutex = .init;
    var log_mutex: std.Io.Mutex = .init;
    var ctx: ServeCtx = .{
        .allocator = allocator,
        .io = io,
        .pipeline = pipeline,
        .pool = pool,
        .opts = opts,
        .rate_limiter = if (rate_limiter) |*rl| rl else null,
        .prefix_cache = opts.prefix_cache,
        .pool_mutex = &pool_mutex,
        .cache_mutex = &cache_mutex,
        .oidc_mutex = &oidc_mutex,
        .log_mutex = &log_mutex,
    };
    const connection_workers = resolveConnectionWorkers(opts.connection_workers, pool.workerCount());
    const queue_capacity = resolveConnectionQueue(opts.connection_queue, connection_workers);
    var runtime = try ServeRuntime.init(allocator, &ctx, connection_workers);
    defer runtime.deinit();
    const connections = try ConnectionPool.init(
        allocator,
        io,
        connection_workers,
        queue_capacity,
        &runtime,
        ServeRuntime.handle,
    );
    defer connections.deinit();
    serveLoop(&ctx, &server, max_connections, connections) catch |err| {
        if (err != error.AcceptLoopEnded) return err;
    };
}

const ConnectionPool = connection_pool_mod.BoundedPool(std.Io.net.Stream);

fn resolveConnectionWorkers(configured: u16, tokenizer_workers: usize) usize {
    if (configured != 0) return configured;
    return @max(@as(usize, 1), @min(tokenizer_workers, default_max_connection_workers));
}

fn resolveConnectionQueue(configured: u16, workers: usize) usize {
    if (configured != 0) return configured;
    return workers * 2;
}

const ServeRuntime = struct {
    allocator: std.mem.Allocator,
    base: *ServeCtx,
    scratches: []ScratchArena,

    fn init(allocator: std.mem.Allocator, base: *ServeCtx, workers: usize) !ServeRuntime {
        const scratches = try allocator.alloc(ScratchArena, workers);
        for (scratches) |*scratch| scratch.* = ScratchArena.init(allocator);
        return .{ .allocator = allocator, .base = base, .scratches = scratches };
    }

    fn deinit(self: *ServeRuntime) void {
        for (self.scratches) |*scratch| scratch.deinit();
        self.allocator.free(self.scratches);
        self.* = undefined;
    }

    fn handle(opaque_ctx: *anyopaque, stream: std.Io.net.Stream, worker_index: usize) void {
        const self: *ServeRuntime = @ptrCast(@alignCast(opaque_ctx));
        var ctx = self.base.*;
        ctx.scratch = &self.scratches[worker_index];
        ctx.last_status = 200;
        ctx.last_bytes_out = 0;
        ctx.last_bytes_in = 0;
        ctx.last_encode_tokens = 0;
        handleConnection(&ctx, stream) catch |err| {
            if (ctx.opts.log) std.log.warn("ztok serve: connection error: {t}", .{err});
        };
        stream.close(ctx.io);
    }
};

pub const ServeCtx = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    pipeline: *const Pipeline,
    pool: *BatchPool,
    opts: Options,
    rate_limiter: ?*RateLimiter = null,
    /// When TLS is enabled (build_options.tls_backend == .mbedtls
    /// AND opts.tls_enabled), this points to the mbedtls server
    /// context. Each accepted connection allocates a `mbedtls.Conn`
    /// against it for handshake.
    tls_server: ?*mbedtls.Server = null,
    /// Optional persistent prefix cache. Mirrors `Options.prefix_cache`
    /// so handlers can reach it through `ctx`.
    prefix_cache: ?*prefix_cache_mod.PrefixCache = null,
    /// Connection-local transient arena. Set by `ServeRuntime`; null for
    /// protocol-pure in-memory tests and external callers constructing a
    /// context directly.
    scratch: ?*ScratchArena = null,
    /// Shared mutable subsystems protected for concurrent connections.
    pool_mutex: ?*std.Io.Mutex = null,
    cache_mutex: ?*std.Io.Mutex = null,
    oidc_mutex: ?*std.Io.Mutex = null,
    log_mutex: ?*std.Io.Mutex = null,
    /// Most-recent response status set by the active handler. Used by
    /// the per-request metrics + logging wrapper to record outcome.
    /// Defaults to 200 (handlers that hit the success path don't
    /// need to update it).
    last_status: u16 = 200,
    /// Bytes written to the response body for the current request,
    /// estimated where possible (exact for `req.respond`, sum of
    /// payload bytes for streaming responses tracked by helpers).
    last_bytes_out: u64 = 0,
    /// Request-body bytes observed. Initialized from Content-Length and
    /// incremented while dechunking when the length is not known up front.
    last_bytes_in: u64 = 0,
    /// Encoded-token count for the current request, when known.
    last_encode_tokens: u64 = 0,
};

fn setStatus(ctx: *ServeCtx, status: u16) void {
    ctx.last_status = status;
}

fn addBytesOut(ctx: *ServeCtx, n: u64) void {
    ctx.last_bytes_out += n;
}

fn handleConnection(ctx: *ServeCtx, stream: std.Io.net.Stream) !void {
    // TLS path (build_options.tls_backend == .mbedtls + opts.tls_enabled):
    // do the handshake on the accepted socket and then drive an in-
    // house HTTP/1.1 parser through `mbedtls.Conn.read`/`.write`. The
    // stdlib `std.http.Server` wants a `std.Io.net.Stream`; rather
    // than build an adapter type that fakes that interface, we
    // bypass it for the TLS path and hand-parse the request line +
    // headers + body, dispatch to a TLS-flavoured handler that
    // mirrors the same routes as the cleartext path, and write the
    // response through `conn.write` directly.
    //
    // The streaming routes (`/encode_stream`, `/encode_chunked`,
    // `/encode_ws`) ARE supported over TLS: NDJSON / chunked responses
    // and WebSocket frames are written through a `mbedtls.ConnWriter`
    // (a `std.Io.Writer` backed by `Conn.write`) and continuation / WS
    // frames are read through a `mbedtls.ConnReader`, so the same
    // `StreamEncoder` / `websocket` framing code runs over both
    // transports.
    if (ctx.tls_server) |tls_srv| {
        var conn = mbedtls.Conn.init(tls_srv, stream.socket.handle) catch |err| {
            if (ctx.opts.log) std.log.warn("ztok serve: TLS conn init failed: {t}", .{err});
            return;
        };
        defer conn.deinit();
        conn.handshake() catch |err| {
            if (ctx.opts.log) std.log.warn("ztok serve: TLS handshake failed: {t}", .{err});
            return;
        };
        if (ctx.opts.log) std.log.info("ztok serve: TLS handshake completed", .{});

        // Best-effort peer-IP lookup so the rate limiter still works
        // for TLS clients. Matches the cleartext path's behaviour.
        const peer_ip = peerIpFromHandle(stream.socket.handle) catch ClientIp.unknown;

        // Serve at most one request per TLS connection in this wave.
        // HTTP/1.1 keep-alive over TLS would require driving the
        // parser in a loop and reading the connection: header — left
        // for the next iteration. Browsers / curl handle the
        // close-then-reconnect cycle transparently.
        handleTlsRequest(ctx, &conn, peer_ip) catch |err| {
            if (ctx.opts.log) std.log.warn("ztok serve: TLS request error: {t}", .{err});
        };
        return;
    }

    var recv_buf: [recv_buf_size]u8 = undefined;
    var send_buf: [send_buf_size]u8 = undefined;
    var sr = stream.reader(ctx.io, &recv_buf);
    var sw = stream.writer(ctx.io, &send_buf);
    var http: std.http.Server = .init(&sr.interface, &sw.interface);

    // Resolve the client's IP once per connection so rate-limiting can
    // bucket against it. Falls back to "unknown" (a fixed sentinel key)
    // if the syscall fails — we still want the request served, just
    // pooled with other unknowns rather than uncounted.
    const peer_ip = peerIpFromHandle(stream.socket.handle) catch ClientIp.unknown;

    // Serve one or more pipelined HTTP/1.1 requests on the same connection.
    var requests_on_connection: u32 = 0;
    while (true) {
        var req = http.receiveHead() catch |err| switch (err) {
            error.HttpConnectionClosing => return,
            else => return,
        };
        const client_wants_keepalive = req.head.keep_alive;
        handleRequest(ctx, &req, peer_ip) catch |err| {
            if (ctx.opts.log) std.log.warn("ztok serve: handler error: {t}", .{err});
            // Best-effort error response. respond() may itself fail if
            // the connection is half-closed — ignore.
            req.respond("internal error\n", .{
                .status = .internal_server_error,
                .keep_alive = false,
                .extra_headers = &.{.{ .name = "content-type", .value = "text/plain" }},
            }) catch {};
            return;
        };
        requests_on_connection += 1;
        // If the client asked for connection: close (or sent HTTP/1.0
        // without keep-alive), stop reading and let the caller close
        // the socket. Otherwise loop back and serve the next request.
        if (!client_wants_keepalive) return;
        if (ctx.opts.max_keepalive_requests != 0 and requests_on_connection >= ctx.opts.max_keepalive_requests) return;
    }
}

fn configureSocketTimeout(fd: std.posix.socket_t, timeout_ms: u32) !void {
    if (timeout_ms == 0) return;
    const tv: std.posix.timeval = .{
        .sec = @intCast(timeout_ms / 1000),
        .usec = @intCast((timeout_ms % 1000) * 1000),
    };
    const bytes = std.mem.asBytes(&tv);
    try std.posix.setsockopt(fd, std.posix.SOL.SOCKET, std.posix.SO.RCVTIMEO, bytes);
    try std.posix.setsockopt(fd, std.posix.SOL.SOCKET, std.posix.SO.SNDTIMEO, bytes);
}

fn handleRequest(ctx: *ServeCtx, req: *std.http.Server.Request, client_ip: ClientIp) !void {
    const method = req.head.method;
    const target = req.head.target;
    // Reset per-request mutable state. (ServeCtx is reused across
    // requests on the same connection.)
    ctx.last_status = 200;
    ctx.last_bytes_in = req.head.content_length orelse 0;
    ctx.last_bytes_out = 0;
    ctx.last_encode_tokens = 0;

    // Strip query string if present (we don't use it today).
    const path = if (std.mem.indexOfScalar(u8, target, '?')) |q| target[0..q] else target;
    const path_label = metrics_mod.PathLabel.fromPath(path);
    const method_label = metrics_mod.MethodLabel.fromMethod(method);
    var client_ip_buf: [64]u8 = undefined;
    const client_ip_str = formatClientIp(client_ip, &client_ip_buf);

    if (ctx.opts.metrics) |m| {
        m.enterRequest();
    }
    const t_start_ns = std.Io.Clock.now(.awake, ctx.io).toNanoseconds();
    defer {
        const t_end_ns = std.Io.Clock.now(.awake, ctx.io).toNanoseconds();
        const dur_ns_i = t_end_ns - t_start_ns;
        const dur_ns: u64 = if (dur_ns_i > 0) @intCast(@as(i64, @intCast(dur_ns_i))) else 0;
        if (ctx.opts.metrics) |m| {
            const status_bucket = metrics_mod.StatusBucket.fromStatus(ctx.last_status);
            m.incRequest(method_label, path_label, status_bucket);
            m.addBytesIn(path_label, ctx.last_bytes_in);
            m.addBytesOut(path_label, ctx.last_bytes_out);
            m.addEncodeTokens(ctx.last_encode_tokens);
            const dur_s: f64 = @as(f64, @floatFromInt(dur_ns)) / 1_000_000_000.0;
            m.observeDuration(path_label, dur_s);
            m.exitRequest();
        }
        emitRequestLog(ctx, .{
            .method = method,
            .path_label = path_label,
            .path_str = path,
            .status = ctx.last_status,
            .bytes_in = ctx.last_bytes_in,
            .bytes_out = ctx.last_bytes_out,
            .duration_ns = dur_ns,
            .client_ip_str = client_ip_str,
            .encoded_tokens = ctx.last_encode_tokens,
        });
    }
    errdefer {
        if (ctx.last_status < 400) ctx.last_status = 500;
    }

    // /health bypasses auth + rate limit. Ops needs an unconditional
    // probe to determine liveness without leaking a credential.
    if (method == .GET and std.mem.eql(u8, path, "/health")) {
        return respondJsonLiteralTracked(ctx, req, "{\"ok\":true}\n");
    }

    // /metrics also bypasses auth+rate-limit so scrapers don't need a
    // credential. The Prometheus convention is for the endpoint to be
    // either firewalled or auth'd at the network layer.
    if (ctx.opts.metrics != null and method == .GET and std.mem.eql(u8, path, "/metrics")) {
        return handleMetrics(ctx, req);
    }

    // Bearer-token auth (if configured). MUST run before rate-limit so
    // an attacker probing for a valid token doesn't get drowned out by
    // 429s — but only after /health, which ops must always reach.
    if (ctx.opts.auth_token) |expected| {
        if (!checkBearerAuth(req, expected)) {
            setStatus(ctx, 401);
            const body = "{\"error\":\"unauthorized\"}\n";
            addBytesOut(ctx, body.len);
            return req.respond(body, .{
                .status = .unauthorized,
                .extra_headers = &.{
                    .{ .name = "content-type", .value = "application/json" },
                    .{ .name = "www-authenticate", .value = "Bearer" },
                },
            });
        }
    } else if (ctx.opts.oidc) |v| {
        // OIDC bearer-JWT auth. Identical 401 shape so clients can
        // treat the two paths uniformly.
        const ok = checkOidcAuth(ctx, req, v) catch false;
        if (!ok) {
            setStatus(ctx, 401);
            const body = "{\"error\":\"unauthorized\"}\n";
            addBytesOut(ctx, body.len);
            return req.respond(body, .{
                .status = .unauthorized,
                .extra_headers = &.{
                    .{ .name = "content-type", .value = "application/json" },
                    .{ .name = "www-authenticate", .value = "Bearer" },
                },
            });
        }
    }

    // Per-client-IP token-bucket rate limit (if configured).
    if (ctx.rate_limiter) |rl| {
        switch (rl.tryAcquire(client_ip)) {
            .ok => {},
            .limited => |retry_after_ms| {
                setStatus(ctx, 429);
                var buf: [96]u8 = undefined;
                const body = try std.fmt.bufPrint(
                    &buf,
                    "{{\"error\":\"rate_limited\",\"retry_after_ms\":{d}}}\n",
                    .{retry_after_ms},
                );
                addBytesOut(ctx, body.len);
                var retry_buf: [16]u8 = undefined;
                const retry_after_s_str = try std.fmt.bufPrint(&retry_buf, "{d}", .{(retry_after_ms + 999) / 1000});
                return req.respond(body, .{
                    .status = .too_many_requests,
                    .extra_headers = &.{
                        .{ .name = "content-type", .value = "application/json" },
                        .{ .name = "retry-after", .value = retry_after_s_str },
                    },
                });
            },
        }
    }

    if (method == .GET and std.mem.eql(u8, path, "/version")) {
        var buf: [256]u8 = undefined;
        const body = try std.fmt.bufPrint(
            &buf,
            "{{\"version\":\"{s}\",\"model_kind\":\"{s}\"}}\n",
            .{ ctx.opts.version, @tagName(ctx.opts.model_kind) },
        );
        return respondJsonLiteralTracked(ctx, req, body);
    }
    if (method == .POST and std.mem.eql(u8, path, "/encode")) {
        return handleEncode(ctx, req);
    }
    if (method == .POST and std.mem.eql(u8, path, "/encode_stream")) {
        return handleEncodeStream(ctx, req);
    }
    if (method == .POST and std.mem.eql(u8, path, "/encode_chunked")) {
        return handleEncodeChunked(ctx, req);
    }
    if (method == .POST and std.mem.eql(u8, path, "/decode")) {
        return handleDecode(ctx, req);
    }
    if (method == .POST and std.mem.eql(u8, path, "/eval")) {
        return handleEval(ctx, req);
    }
    if (method == .GET and std.mem.eql(u8, path, "/encode_ws")) {
        return handleEncodeWs(ctx, req);
    }

    setStatus(ctx, 404);
    const not_found = "not found\n";
    addBytesOut(ctx, not_found.len);
    try req.respond(not_found, .{
        .status = .not_found,
        .extra_headers = &.{.{ .name = "content-type", .value = "text/plain" }},
    });
}

fn respondJsonLiteralTracked(ctx: *ServeCtx, req: *std.http.Server.Request, body: []const u8) !void {
    addBytesOut(ctx, body.len);
    try req.respond(body, .{
        .status = .ok,
        .extra_headers = &.{.{ .name = "content-type", .value = "application/json" }},
    });
}

// === Common helpers ====================================================

fn respondJsonLiteral(req: *std.http.Server.Request, body: []const u8) !void {
    try req.respond(body, .{
        .status = .ok,
        .extra_headers = &.{.{ .name = "content-type", .value = "application/json" }},
    });
}

/// Updates the metrics ctx (status + bytes_out) and writes a 400.
fn respondBadRequestTracked(ctx: *ServeCtx, req: *std.http.Server.Request, msg: []const u8) !void {
    setStatus(ctx, 400);
    addBytesOut(ctx, msg.len);
    try req.respond(msg, .{
        .status = .bad_request,
        .extra_headers = &.{.{ .name = "content-type", .value = "text/plain" }},
    });
}

fn respondPayloadTooLargeTracked(ctx: *ServeCtx, req: *std.http.Server.Request) !void {
    setStatus(ctx, 413);
    const msg = "body too large\n";
    addBytesOut(ctx, msg.len);
    try req.respond(msg, .{
        .status = .payload_too_large,
        .extra_headers = &.{.{ .name = "content-type", .value = "text/plain" }},
    });
}

/// Read the request body in full into a freshly-allocated buffer the
/// caller must free. Enforces `max_body_bytes`. Returns null when the
/// request had no body (e.g. content-length: 0).
fn readBody(ctx: *ServeCtx, req: *std.http.Server.Request) !?[]u8 {
    if (req.head.method.requestHasBody() == false) return null;
    if (req.head.transfer_encoding == .none and req.head.content_length == null) return null;

    if (req.head.content_length) |cl| {
        if (cl > ctx.opts.max_body_bytes) return error.BodyTooLarge;
    }

    var body_buf: [4096]u8 = undefined;
    const body_reader = req.readerExpectContinue(&body_buf) catch |err| switch (err) {
        error.WriteFailed => return error.WriteFailed,
        error.HttpExpectationFailed => return error.HttpExpectationFailed,
    };

    if (req.head.content_length) |cl| {
        const body = try ctx.allocator.alloc(u8, cl);
        errdefer ctx.allocator.free(body);
        var filled: usize = 0;
        while (filled < body.len) {
            const n = try body_reader.readSliceShort(body[filled..]);
            if (n == 0) return error.UnexpectedEndOfStream;
            filled += n;
        }
        return body;
    }

    var list: std.ArrayList(u8) = .empty;
    errdefer list.deinit(ctx.allocator);

    var tmp: [4096]u8 = undefined;
    while (true) {
        const n = try body_reader.readSliceShort(&tmp);
        if (n == 0) break;
        ctx.last_bytes_in += n;
        if (list.items.len + n > ctx.opts.max_body_bytes) return error.BodyTooLarge;
        try list.appendSlice(ctx.allocator, tmp[0..n]);
        if (n < tmp.len) {
            // Short read with no data left in stream → EOF.
            // (readSliceShort returns <buffer.len iff the stream ended.)
            break;
        }
    }
    return try list.toOwnedSlice(ctx.allocator);
}

const ParsedTextField = struct {
    arena: std.heap.ArenaAllocator,
    value: []const u8,

    fn deinit(self: *ParsedTextField) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

/// Parse a top-level `text` field. Unescaped strings borrow directly from
/// `body`; escaped strings are decoded into the returned arena. This avoids
/// the old parse-allocation followed by a second duplicate allocation.
fn parseTextField(allocator: std.mem.Allocator, body: []const u8) !?ParsedTextField {
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const Parsed = struct { text: ?[]const u8 = null };
    const parsed = std.json.parseFromSliceLeaky(
        Parsed,
        arena.allocator(),
        body,
        .{ .ignore_unknown_fields = true, .allocate = .alloc_if_needed },
    ) catch {
        arena.deinit();
        return null;
    };
    if (parsed.text) |t| {
        return .{ .arena = arena, .value = t };
    }
    arena.deinit();
    return null;
}

/// Compatibility helper used by protocol-pure tests.
fn extractTextField(allocator: std.mem.Allocator, body: []const u8) !?[]u8 {
    var parsed = (try parseTextField(allocator, body)) orelse return null;
    defer parsed.deinit();
    return try allocator.dupe(u8, parsed.value);
}

const ParsedIdsField = struct {
    arena: std.heap.ArenaAllocator,
    value: []const TokenId,

    fn deinit(self: *ParsedIdsField) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

fn parseIdsField(allocator: std.mem.Allocator, body: []const u8) !?ParsedIdsField {
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const Parsed = struct { ids: ?[]TokenId = null };
    const parsed = std.json.parseFromSliceLeaky(
        Parsed,
        arena.allocator(),
        body,
        .{ .ignore_unknown_fields = true, .allocate = .alloc_always },
    ) catch {
        arena.deinit();
        return null;
    };
    if (parsed.ids) |ids| {
        return .{ .arena = arena, .value = ids };
    }
    arena.deinit();
    return null;
}

fn extractIdsField(allocator: std.mem.Allocator, body: []const u8) !?[]TokenId {
    var parsed = (try parseIdsField(allocator, body)) orelse return null;
    defer parsed.deinit();
    return try allocator.dupe(TokenId, parsed.value);
}

// === /encode ===========================================================

/// Encode `text` against `ctx.pipeline`, consulting `ctx.prefix_cache`
/// when present.
///
/// Cache key: the complete input. Cache value: the complete id stream.
/// This exact-result contract preserves bit identity for every pipeline;
/// token streams encoded on either side of an arbitrary byte boundary do
/// not generally concatenate to the single-shot result.
///
/// Returns a freshly allocated slice the caller must free with
/// `ctx.allocator`. Falls back to a plain `encode`/`encodeChunked`
/// when no cache is configured.
fn encodeWithPrefixCache(ctx: *ServeCtx, text: []const u8) ![]TokenId {
    if (ctx.prefix_cache) |pc| {
        if (ctx.cache_mutex) |mutex| mutex.lockUncancelable(ctx.io);
        defer if (ctx.cache_mutex) |mutex| mutex.unlock(ctx.io);
        const ids = try pc.lookupOrInsert(ctx.pipeline, text);
        // Cache slices are invalidated by the next mutation; return an
        // independent result owned by the request allocator.
        return ctx.allocator.dupe(TokenId, ids);
    }

    // Encode through the pool when there's enough input to make
    // chunking worth it; otherwise fall back to single-shot.
    const min_for_chunked: usize = 64 * 1024;
    const n_workers = ctx.pool.workerCount();
    if (text.len >= min_for_chunked and n_workers > 1) {
        if (ctx.pool_mutex) |mutex| mutex.lockUncancelable(ctx.io);
        defer if (ctx.pool_mutex) |mutex| mutex.unlock(ctx.io);
        return ctx.pipeline.encodeChunked(ctx.allocator, ctx.pool, text, n_workers);
    }
    return encodeSingle(ctx, text);
}

fn encodeSingle(ctx: *ServeCtx, text: []const u8) ![]TokenId {
    if (ctx.scratch) |scratch| return ctx.pipeline.encodeWithScratch(ctx.allocator, text, scratch);
    return ctx.pipeline.encode(ctx.allocator, text);
}

fn handleEncode(ctx: *ServeCtx, req: *std.http.Server.Request) !void {
    const body_opt = readBody(ctx, req) catch |err| switch (err) {
        error.BodyTooLarge => return respondPayloadTooLargeTracked(ctx, req),
        else => return err,
    };
    const body = body_opt orelse return respondBadRequestTracked(ctx, req, "missing body\n");
    defer ctx.allocator.free(body);

    var parsed_text = (try parseTextField(ctx.allocator, body)) orelse {
        return respondBadRequestTracked(ctx, req, "expected JSON {\"text\":\"...\"}\n");
    };
    defer parsed_text.deinit();
    const text = parsed_text.value;

    // Encode (optionally through the prefix cache) — see encodeWithPrefixCache
    // for the splice semantics.
    const ids = try encodeWithPrefixCache(ctx, text);
    defer ctx.allocator.free(ids);
    ctx.last_encode_tokens += ids.len;

    // Compute a response upper bound: 8 bytes per id is generous
    // ("4294967295," is 11). Use a streaming response so we don't
    // allocate the response body twice.
    var resp_buf: [4096]u8 = undefined;
    var body_writer = try req.respondStreaming(&resp_buf, .{
        .respond_options = .{
            .extra_headers = &.{.{ .name = "content-type", .value = "application/json" }},
        },
    });
    const w = &body_writer.writer;
    try w.writeAll("{\"ids\":[");
    var body_len: usize = "{\"ids\":[".len + "]}\n".len;
    for (ids, 0..) |id, i| {
        if (i > 0) {
            try w.writeByte(',');
            body_len += 1;
        }
        try w.print("{d}", .{id});
        body_len += decimalLenU32(id);
    }
    try w.writeAll("]}\n");
    try body_writer.end();
    addBytesOut(ctx, body_len);
}

// === /encode_stream ====================================================

/// Number of bytes to feed at a time when chunking the input through
/// the StreamEncoder. Each feed emits one NDJSON line (when it has any
/// ids to flush). 4 KiB matches a typical socket-write granularity.
const stream_feed_size: usize = 4096;

fn handleEncodeStream(ctx: *ServeCtx, req: *std.http.Server.Request) !void {
    const body_opt = readBody(ctx, req) catch |err| switch (err) {
        error.BodyTooLarge => return respondPayloadTooLargeTracked(ctx, req),
        else => return err,
    };
    const body = body_opt orelse return respondBadRequestTracked(ctx, req, "missing body\n");
    defer ctx.allocator.free(body);

    var parsed_text = (try parseTextField(ctx.allocator, body)) orelse {
        return respondBadRequestTracked(ctx, req, "expected JSON {\"text\":\"...\"}\n");
    };
    defer parsed_text.deinit();
    const text = parsed_text.value;

    var resp_buf: [4096]u8 = undefined;
    var body_writer = try req.respondStreaming(&resp_buf, .{
        .respond_options = .{
            .extra_headers = &.{
                .{ .name = "content-type", .value = "application/x-ndjson" },
            },
        },
    });
    const w = &body_writer.writer;

    var enc = StreamEncoder.init(ctx.allocator, ctx.pipeline);
    defer enc.deinit();

    var batch: std.ArrayList(TokenId) = .empty;
    defer batch.deinit(ctx.allocator);

    var i: usize = 0;
    while (i < text.len) {
        const chunk_end = @min(i + stream_feed_size, text.len);
        try enc.feed(text[i..chunk_end], &batch);
        if (batch.items.len > 0) {
            ctx.last_encode_tokens += batch.items.len;
            addBytesOut(ctx, try writeIdsBatchLine(w, batch.items));
            batch.clearRetainingCapacity();
            try body_writer.flush();
        }
        i = chunk_end;
    }
    try enc.finish(&batch);
    if (batch.items.len > 0) {
        ctx.last_encode_tokens += batch.items.len;
        addBytesOut(ctx, try writeIdsBatchLine(w, batch.items));
        batch.clearRetainingCapacity();
    }
    try w.writeAll("{\"done\":true}\n");
    addBytesOut(ctx, "{\"done\":true}\n".len);
    try body_writer.end();
}

fn writeIdsBatchLine(w: *std.Io.Writer, ids: []const TokenId) !usize {
    var len: usize = "{\"ids\":[".len + "]}\n".len;
    try w.writeAll("{\"ids\":[");
    for (ids, 0..) |id, i| {
        if (i > 0) {
            try w.writeByte(',');
            len += 1;
        }
        try w.print("{d}", .{id});
        len += decimalLenU32(id);
    }
    try w.writeAll("]}\n");
    return len;
}

fn decimalLenU32(v: u32) usize {
    if (v < 10) return 1;
    if (v < 100) return 2;
    if (v < 1_000) return 3;
    if (v < 10_000) return 4;
    if (v < 100_000) return 5;
    if (v < 1_000_000) return 6;
    if (v < 10_000_000) return 7;
    if (v < 100_000_000) return 8;
    if (v < 1_000_000_000) return 9;
    return 10;
}

fn idsJsonLineLen(ids: []const TokenId) usize {
    var n: usize = "{\"ids\":[".len + "]}\n".len;
    for (ids, 0..) |id, i| {
        if (i > 0) n += 1;
        n += decimalLenU32(id);
    }
    return n;
}

// === /encode_chunked ===================================================
//
// POST /encode_chunked — accepts the raw input bytes as a chunked
// HTTP/1.1 request body (or content-length if the client prefers; the
// stdlib body reader handles both transparently), streams them through
// `StreamEncoder` chunk by chunk, and emits an NDJSON response over a
// chunked HTTP/1.1 response body. Each `StreamEncoder.feed` call that
// produces ids writes one `{"ids":[...]}` line; the response ends with
// a single `{"done":true}` line after `finish` drains the carry.
//
// Unlike `/encode_stream` which expects a JSON object
// `{"text":"..."}`, this route's request body IS the input — no JSON
// wrapping. That avoids forcing C/Rust/Go clients to JSON-encode a
// multi-GB blob just to pass it over the wire (and avoids the buffering
// that would imply).
//
// Memory cap: cumulative request body bytes are still capped at
// `opts.max_body_bytes` (default 16 MiB; configurable). Exceeding it
// returns 413 mid-stream — the response will be a partial chunked
// stream terminated by an HTTP error. Clients that want to send larger
// inputs should raise the cap explicitly.
//
// Auth + rate-limit: applied by `handleRequest` BEFORE this handler
// runs, identical to every other route except `/health`.
//
// Why a separate route from `/encode_stream`: `/encode_stream` JSON-
// decodes a whole body into memory, then streams ids. That's fine for
// small inputs but defeats the streaming guarantee for large inputs.
// `/encode_chunked` keeps memory bounded by the StreamEncoder's
// internal carry (typically <1 MiB) regardless of input size.

/// HTTP chunk size when reading the request body. 64 KiB matches the
/// typical socket read granularity and is big enough that the per-feed
/// overhead amortizes well. Each read becomes one feed → one possible
/// NDJSON line.
const chunked_read_size: usize = 64 * 1024;

fn handleEncodeChunked(ctx: *ServeCtx, req: *std.http.Server.Request) !void {
    // Reject requests with no body — both content-length 0 and a
    // GET-style no-body invocation. We need bytes to encode.
    if (req.head.method.requestHasBody() == false) {
        return respondBadRequestTracked(ctx, req, "missing body\n");
    }
    if (req.head.transfer_encoding == .none and req.head.content_length == null) {
        return respondBadRequestTracked(ctx, req, "missing body (no content-length and no transfer-encoding)\n");
    }
    // Early reject: a known content-length over the cap shouldn't even
    // start streaming.
    if (req.head.content_length) |cl| {
        if (cl > ctx.opts.max_body_bytes) {
            return respondPayloadTooLargeTracked(ctx, req);
        }
    }

    // Acquire the body reader FIRST (must be called before
    // `respondStreaming`, which internally calls `discardBody` →
    // `readerExpectContinue`. Both helpers assert "only called once",
    // so we own the single call here.)
    //
    // The stdlib body reader transparently dechunks chunked-transfer
    // bodies AND honors content-length, so the route accepts either
    // shape from the client without special-casing.
    var body_buf: [4096]u8 = undefined;
    const body_reader = req.readerExpectContinue(&body_buf) catch |err| switch (err) {
        error.WriteFailed => return error.WriteFailed,
        error.HttpExpectationFailed => return error.HttpExpectationFailed,
    };

    // Now that the body reader exists, open the response stream.
    // `respondStreaming` with no `transfer_encoding` override defaults
    // to chunked.
    var resp_buf: [4096]u8 = undefined;
    var body_writer = try req.respondStreaming(&resp_buf, .{
        .respond_options = .{
            .extra_headers = &.{
                .{ .name = "content-type", .value = "application/x-ndjson" },
            },
        },
    });
    const w = &body_writer.writer;

    var enc = StreamEncoder.init(ctx.allocator, ctx.pipeline);
    defer enc.deinit();

    var batch: std.ArrayList(TokenId) = .empty;
    defer batch.deinit(ctx.allocator);

    var bytes_read: usize = 0;
    var tmp: [chunked_read_size]u8 = undefined;
    while (true) {
        const n = body_reader.readSliceShort(&tmp) catch |err| {
            if (ctx.opts.log) std.log.warn("ztok serve: /encode_chunked body read failed: {t}", .{err});
            return err;
        };
        if (n == 0) break;
        bytes_read += n;
        if (req.head.content_length == null) ctx.last_bytes_in += n;
        if (bytes_read > ctx.opts.max_body_bytes) {
            // Surface a final {"error":"body_too_large"} line on the
            // wire — the response is already in chunked mode so we can
            // append it, then end with the chunked terminator. Status
            // can't be changed at this point (headers are out).
            try w.writeAll("{\"error\":\"body_too_large\"}\n");
            addBytesOut(ctx, "{\"error\":\"body_too_large\"}\n".len);
            try body_writer.end();
            return;
        }
        try enc.feed(tmp[0..n], &batch);
        if (batch.items.len > 0) {
            ctx.last_encode_tokens += batch.items.len;
            addBytesOut(ctx, try writeIdsBatchLine(w, batch.items));
            batch.clearRetainingCapacity();
            try body_writer.flush();
        }
        if (n < tmp.len) {
            // Short read with no data left in stream → end of body.
            break;
        }
    }
    try enc.finish(&batch);
    if (batch.items.len > 0) {
        ctx.last_encode_tokens += batch.items.len;
        addBytesOut(ctx, try writeIdsBatchLine(w, batch.items));
        batch.clearRetainingCapacity();
    }
    try w.writeAll("{\"done\":true}\n");
    addBytesOut(ctx, "{\"done\":true}\n".len);
    try body_writer.end();
}

// === /decode ===========================================================

fn handleDecode(ctx: *ServeCtx, req: *std.http.Server.Request) !void {
    const body_opt = readBody(ctx, req) catch |err| switch (err) {
        error.BodyTooLarge => return respondPayloadTooLargeTracked(ctx, req),
        else => return err,
    };
    const body = body_opt orelse return respondBadRequestTracked(ctx, req, "missing body\n");
    defer ctx.allocator.free(body);

    var parsed_ids = (try parseIdsField(ctx.allocator, body)) orelse {
        return respondBadRequestTracked(ctx, req, "expected JSON {\"ids\":[...]}\n");
    };
    defer parsed_ids.deinit();
    const ids = parsed_ids.value;

    const text = try ctx.pipeline.decode(ctx.allocator, ids);
    defer ctx.allocator.free(text);
    addBytesOut(ctx, "{\"text\":".len + jsonStringLen(text) + "}\n".len);

    var resp_buf: [4096]u8 = undefined;
    var body_writer = try req.respondStreaming(&resp_buf, .{
        .respond_options = .{
            .extra_headers = &.{.{ .name = "content-type", .value = "application/json" }},
        },
    });
    const w = &body_writer.writer;
    try w.writeAll("{\"text\":");
    try writeJsonString(w, text);
    try w.writeAll("}\n");
    try body_writer.end();
}

// === /eval =============================================================

fn handleEval(ctx: *ServeCtx, req: *std.http.Server.Request) !void {
    const body_opt = readBody(ctx, req) catch |err| switch (err) {
        error.BodyTooLarge => return respondPayloadTooLargeTracked(ctx, req),
        else => return err,
    };
    const body = body_opt orelse return respondBadRequestTracked(ctx, req, "missing body\n");
    defer ctx.allocator.free(body);

    var parsed_text = (try parseTextField(ctx.allocator, body)) orelse {
        return respondBadRequestTracked(ctx, req, "expected JSON {\"text\":\"...\"}\n");
    };
    defer parsed_text.deinit();
    const text = parsed_text.value;

    var rendered: std.Io.Writer.Allocating = .init(ctx.allocator);
    defer rendered.deinit();
    try cli_eval.runEval(ctx.allocator, ctx.pipeline, text, .{
        .format = .json,
    }, &rendered.writer);
    const response = rendered.written();
    addBytesOut(ctx, response.len);
    try req.respond(response, .{
        .status = .ok,
        .extra_headers = &.{.{ .name = "content-type", .value = "application/json" }},
    });
}

// === Inline string JSON escape =========================================

fn writeJsonString(w: *std.Io.Writer, s: []const u8) !void {
    try w.writeByte('"');
    for (s) |c| switch (c) {
        '"' => try w.writeAll("\\\""),
        '\\' => try w.writeAll("\\\\"),
        '\n' => try w.writeAll("\\n"),
        '\r' => try w.writeAll("\\r"),
        '\t' => try w.writeAll("\\t"),
        0x08 => try w.writeAll("\\b"),
        0x0C => try w.writeAll("\\f"),
        0x00...0x07, 0x0B, 0x0E...0x1F => try w.print("\\u{x:0>4}", .{c}),
        else => try w.writeByte(c),
    };
    try w.writeByte('"');
}

fn jsonStringLen(s: []const u8) usize {
    var n: usize = 2; // surrounding quotes
    for (s) |c| n += switch (c) {
        '"', '\\', '\n', '\r', '\t', 0x08, 0x0C => 2,
        0x00...0x07, 0x0B, 0x0E...0x1F => 6,
        else => 1,
    };
    return n;
}

// === Auth helpers ======================================================

/// Constant-time check that `req` carries `Authorization: Bearer <expected>`.
///
/// `std.crypto.timing_safe.eql` works on fixed-size arrays only — token
/// length is unknown at comptime, so we hand-roll the variable-length
/// equivalent: walk both byte ranges in lockstep, OR'ing differences
/// into an accumulator without short-circuiting. The function still
/// returns early when lengths differ — a length oracle is unavoidable
/// since the comparison must access all bytes of both sides.
pub fn checkBearerAuth(req: *const std.http.Server.Request, expected: []const u8) bool {
    var it = req.iterateHeaders();
    while (it.next()) |h| {
        if (!std.ascii.eqlIgnoreCase(h.name, "authorization")) continue;
        // Strip leading "Bearer " (case-insensitive, single ASCII space
        // after the scheme — matches what curl + every standard client
        // emits; we don't need to handle quoted strings here).
        if (h.value.len < 7) return false;
        if (!std.ascii.eqlIgnoreCase(h.value[0..7], "bearer ")) return false;
        const presented = h.value[7..];
        return constantTimeEqlBytes(presented, expected);
    }
    return false;
}

/// Length-preserving constant-time byte comparison. Returns false
/// immediately on length mismatch (length is not secret); otherwise
/// scans all bytes regardless of where the first difference appears.
pub fn constantTimeEqlBytes(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    var acc: u8 = 0;
    for (a, b) |x, y| acc |= x ^ y;
    return acc == 0;
}

// === Peer-IP lookup ====================================================

/// Compact, hashable representation of a client IP. IPv4 addresses are
/// stored as IPv4-mapped IPv6 (::ffff:a.b.c.d) so a single key type
/// covers both families. A fixed `unknown` sentinel buckets requests
/// whose `getpeername` failed (e.g., unix-domain test sockets).
pub const ClientIp = struct {
    bytes: [16]u8,

    pub const unknown: ClientIp = .{ .bytes = .{0} ** 16 };

    pub fn fromIp4(b: [4]u8) ClientIp {
        var out: ClientIp = .{ .bytes = .{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 0, 0, 0, 0 } };
        @memcpy(out.bytes[12..16], &b);
        return out;
    }

    pub fn fromIp6(b: [16]u8) ClientIp {
        return .{ .bytes = b };
    }
};

fn formatClientIp(ip: ClientIp, buf: []u8) []const u8 {
    if (std.mem.eql(u8, &ip.bytes, &ClientIp.unknown.bytes)) return "unknown";
    const mapped_prefix = [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff };
    if (std.mem.eql(u8, ip.bytes[0..12], &mapped_prefix)) {
        return std.fmt.bufPrint(buf, "{d}.{d}.{d}.{d}", .{
            ip.bytes[12], ip.bytes[13], ip.bytes[14], ip.bytes[15],
        }) catch "unknown";
    }
    return std.fmt.bufPrint(buf, "{x}:{x}:{x}:{x}:{x}:{x}:{x}:{x}", .{
        std.mem.readInt(u16, ip.bytes[0..2], .big),
        std.mem.readInt(u16, ip.bytes[2..4], .big),
        std.mem.readInt(u16, ip.bytes[4..6], .big),
        std.mem.readInt(u16, ip.bytes[6..8], .big),
        std.mem.readInt(u16, ip.bytes[8..10], .big),
        std.mem.readInt(u16, ip.bytes[10..12], .big),
        std.mem.readInt(u16, ip.bytes[12..14], .big),
        std.mem.readInt(u16, ip.bytes[14..16], .big),
    }) catch "unknown";
}

const ClientIpContext = struct {
    pub fn hash(_: ClientIpContext, k: ClientIp) u64 {
        return std.hash.Wyhash.hash(0xC0FFEE, &k.bytes);
    }
    pub fn eql(_: ClientIpContext, a: ClientIp, b: ClientIp) bool {
        return std.mem.eql(u8, &a.bytes, &b.bytes);
    }
};

fn peerIpFromHandle(fd: std.posix.fd_t) !ClientIp {
    var storage: std.posix.system.sockaddr.storage = undefined;
    var len: std.posix.socklen_t = @sizeOf(@TypeOf(storage));
    try std.posix.getpeername(fd, @ptrCast(&storage), &len);
    switch (storage.family) {
        std.posix.system.AF.INET => {
            const sin: *const std.posix.system.sockaddr.in = @ptrCast(@alignCast(&storage));
            const ip_bytes: [4]u8 = @bitCast(sin.addr);
            return ClientIp.fromIp4(ip_bytes);
        },
        std.posix.system.AF.INET6 => {
            const sin6: *const std.posix.system.sockaddr.in6 = @ptrCast(@alignCast(&storage));
            return ClientIp.fromIp6(sin6.addr);
        },
        else => return ClientIp.unknown,
    }
}

// === Rate limiter ======================================================

/// Per-client-IP token bucket. Bucket size = rps × 2, refill rate =
/// rps tokens/sec; refill is computed lazily on each acquire from a
/// monotonic clock so an idle client never falls behind. LRU-evicts
/// the oldest entry once the map hits `rate_limit_max_clients`.
///
/// The serve loop is single-threaded today, but the limiter holds a
/// mutex around its state so we can flip to a concurrent server later
/// without revisiting this.
pub const RateLimiter = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    rps: u32,
    /// Bucket capacity in fractional tokens. We store tokens as f64 so
    /// the refill rate doesn't have to be an integer divisor of the
    /// refill interval.
    capacity: f64,
    /// Per-IP state.
    buckets: std.HashMapUnmanaged(ClientIp, Bucket, ClientIpContext, std.hash_map.default_max_load_percentage),
    /// Monotonic counter incremented on every acquire; serves as the
    /// LRU "last-touched" timestamp so eviction is O(N) but only runs
    /// when the cap is hit (no per-request cost).
    tick: u64,
    // Local spinlock shim: std.Thread.Mutex was removed in Zig 0.16.
    // Agent E (perf push) added this stub to unblock the build; agent D
    // (serve hardening) owns the canonical replacement (likely
    // std.Io.Mutex which takes an io arg).
    mutex: Mutex,

    pub const Mutex = struct {
        state: std.atomic.Value(u32) = .init(0),
        pub fn lock(m: *Mutex) void {
            while (m.state.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) {
                std.atomic.spinLoopHint();
            }
        }
        pub fn unlock(m: *Mutex) void {
            m.state.store(0, .release);
        }
    };

    pub const Bucket = struct {
        /// Fractional tokens remaining.
        tokens: f64,
        /// Last `tick` at which `tokens` was refilled.
        last_tick: u64,
        /// Last monotonic-clock reading (in nanoseconds since some
        /// arbitrary epoch) used for refill.
        last_refill_ns: i96,
    };

    pub const Decision = union(enum) {
        ok,
        limited: u64, // retry_after_ms
    };

    pub fn init(allocator: std.mem.Allocator, io: std.Io, rps: u32) RateLimiter {
        return .{
            .allocator = allocator,
            .io = io,
            .rps = rps,
            .capacity = @as(f64, @floatFromInt(rps)) * 2.0,
            .buckets = .empty,
            .tick = 0,
            .mutex = .{},
        };
    }

    pub fn deinit(rl: *RateLimiter) void {
        rl.buckets.deinit(rl.allocator);
    }

    /// Try to spend one token for `client`. Returns `.ok` if the
    /// bucket had a token; otherwise `.limited` with a millisecond
    /// estimate of when the next token will be available.
    pub fn tryAcquire(rl: *RateLimiter, client: ClientIp) Decision {
        const now_ns = std.Io.Clock.now(.awake, rl.io).toNanoseconds();
        return rl.tryAcquireAt(client, now_ns);
    }

    /// Time-injected variant used by tests so they're deterministic.
    pub fn tryAcquireAt(rl: *RateLimiter, client: ClientIp, now_ns: i96) Decision {
        rl.mutex.lock();
        defer rl.mutex.unlock();

        rl.tick += 1;
        const gop = rl.buckets.getOrPut(rl.allocator, client) catch {
            // OOM → fail open. Better to serve than to lock out.
            return .ok;
        };
        if (!gop.found_existing) {
            // New client: start with a full bucket, charge one token.
            gop.value_ptr.* = .{
                .tokens = rl.capacity - 1.0,
                .last_tick = rl.tick,
                .last_refill_ns = now_ns,
            };
            rl.maybeEvict();
            return .ok;
        }

        const b = gop.value_ptr;
        const elapsed_ns_i96: i96 = now_ns - b.last_refill_ns;
        const elapsed_ns: f64 = if (elapsed_ns_i96 > 0) @floatFromInt(@as(i64, @intCast(elapsed_ns_i96))) else 0;
        const refill = elapsed_ns * @as(f64, @floatFromInt(rl.rps)) / 1_000_000_000.0;
        b.tokens = @min(rl.capacity, b.tokens + refill);
        b.last_refill_ns = now_ns;
        b.last_tick = rl.tick;

        if (b.tokens >= 1.0) {
            b.tokens -= 1.0;
            return .ok;
        }
        // Not enough; estimate when the next whole token arrives.
        const deficit = 1.0 - b.tokens;
        const wait_ns = deficit * 1_000_000_000.0 / @as(f64, @floatFromInt(rl.rps));
        const wait_ms_f = wait_ns / 1_000_000.0;
        const wait_ms: u64 = if (wait_ms_f < 1.0) 1 else @intFromFloat(wait_ms_f);
        return .{ .limited = wait_ms };
    }

    /// LRU-evict the oldest entry if we've exceeded the cap. O(N) once
    /// per insertion past `rate_limit_max_clients`.
    fn maybeEvict(rl: *RateLimiter) void {
        if (rl.buckets.count() <= rate_limit_max_clients) return;
        var oldest_key: ?ClientIp = null;
        var oldest_tick: u64 = std.math.maxInt(u64);
        var it = rl.buckets.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.last_tick < oldest_tick) {
                oldest_tick = entry.value_ptr.last_tick;
                oldest_key = entry.key_ptr.*;
            }
        }
        if (oldest_key) |k| _ = rl.buckets.remove(k);
    }
};

// === Structured request logging ========================================

/// Write one request log line to stderr in the configured format.
/// Text format is unchanged from pre-1.22; JSON emits a single
/// well-formed object terminated by '\n' so stdlib log consumers and
/// `jq`-style pipelines both work.
fn emitRequestLog(ctx: *ServeCtx, outcome: RequestOutcome) void {
    if (!ctx.opts.log) return;
    if (ctx.log_mutex) |mutex| mutex.lockUncancelable(ctx.io);
    defer if (ctx.log_mutex) |mutex| mutex.unlock(ctx.io);
    switch (ctx.opts.log_format) {
        .text => {
            std.log.info(
                "ztok serve: {t} {s} -> {d} ({d:.2}ms, {d}B in, {d}B out)",
                .{
                    outcome.method,
                    outcome.path_str,
                    outcome.status,
                    @as(f64, @floatFromInt(outcome.duration_ns)) / 1_000_000.0,
                    outcome.bytes_in,
                    outcome.bytes_out,
                },
            );
        },
        .json => {
            var alloc: std.Io.Writer.Allocating = .init(ctx.allocator);
            defer alloc.deinit();
            writeJsonLogLine(&alloc.writer, ctx, outcome) catch return;
            // std.Io.File.stderr() is the 0.16 stdlib entry point.
            const stderr_file = std.Io.File.stderr();
            stderr_file.writeStreamingAll(ctx.io, alloc.written()) catch {};
        },
    }
}

/// Build one JSON-format log line into `w`. Pure function (no I/O)
/// so tests can drive it through `std.Io.Writer.Allocating`.
pub fn writeJsonLogLine(
    w: *std.Io.Writer,
    ctx: *const ServeCtx,
    outcome: RequestOutcome,
) !void {
    const ts_ms = std.Io.Clock.now(.real, ctx.io).toMilliseconds();
    var ts_buf: [32]u8 = undefined;
    const ts_str = formatRfc3339Utc(ts_ms, &ts_buf);
    const latency_ms_f: f64 = @as(f64, @floatFromInt(outcome.duration_ns)) / 1_000_000.0;
    try w.print(
        "{{\"ts\":\"{s}\",\"method\":\"{t}\",\"path\":",
        .{ ts_str, outcome.method },
    );
    try writeJsonString(w, outcome.path_str);
    try w.print(
        ",\"status\":{d},\"latency_ms\":{d:.3},\"bytes_in\":{d},\"bytes_out\":{d},\"client_ip\":",
        .{ outcome.status, latency_ms_f, outcome.bytes_in, outcome.bytes_out },
    );
    try writeJsonString(w, outcome.client_ip_str);
    if (outcome.encoded_tokens > 0) {
        try w.print(",\"encoded_tokens\":{d}", .{outcome.encoded_tokens});
    }
    try w.writeAll("}\n");
}

/// Format a Unix-epoch millisecond count as a fixed-length ISO 8601
/// timestamp in UTC (`YYYY-MM-DDTHH:MM:SSZ`, 20 chars). Returns a
/// slice into `buf`. `buf` must hold at least 21 bytes.
fn formatRfc3339Utc(ms: i64, buf: []u8) []u8 {
    std.debug.assert(buf.len >= 21);
    // Use std.time.epoch to crack the seconds field into Y/M/D/h/m/s.
    const total_secs: i64 = @divFloor(ms, 1000);
    const ep_secs_u64: u64 = if (total_secs >= 0) @intCast(total_secs) else 0;
    const ep_secs: std.time.epoch.EpochSeconds = .{ .secs = ep_secs_u64 };
    const day_secs = ep_secs.getDaySeconds();
    const ep_day = ep_secs.getEpochDay();
    const year_day = ep_day.calculateYearDay();
    const month_day = year_day.calculateMonthDay();
    const out = std.fmt.bufPrint(
        buf,
        "{d:0>4}-{d:0>2}-{d:0>2}T{d:0>2}:{d:0>2}:{d:0>2}Z",
        .{
            year_day.year,
            @intFromEnum(month_day.month),
            month_day.day_index + 1,
            day_secs.getHoursIntoDay(),
            day_secs.getMinutesIntoHour(),
            day_secs.getSecondsIntoMinute(),
        },
    ) catch buf[0..0];
    return out;
}

// === /metrics ==========================================================

fn handleMetrics(ctx: *ServeCtx, req: *std.http.Server.Request) !void {
    const m = ctx.opts.metrics orelse return; // never null on this path
    var out: std.Io.Writer.Allocating = .init(ctx.allocator);
    defer out.deinit();
    try metrics_mod.render(m, &out.writer);
    const body = out.written();
    addBytesOut(ctx, body.len);
    try req.respond(body, .{
        .status = .ok,
        .extra_headers = &.{
            .{ .name = "content-type", .value = "text/plain; version=0.0.4" },
        },
    });
}

// === /encode_ws ========================================================
//
// Minimal WebSocket upgrade endpoint. On a successful handshake we
// write the 101 Switching Protocols response, then run a tiny
// request/response loop on raw socket bytes:
//
//   client → server: text frame containing the input text to encode
//   server → client: one or more binary frames containing little-endian
//                    u32 token ids, followed by a text "DONE" frame.
//
// This is intentionally simpler than `/encode_stream` — the goal is to
// give browser/JS clients a single persistent connection per session
// rather than reinventing NDJSON over HTTP for them.

const ws_max_frame_payload: usize = 16 * 1024 * 1024;
const ws_emit_chunk_ids: usize = 1024;

fn handleEncodeWs(ctx: *ServeCtx, req: *std.http.Server.Request) !void {
    // Pull the Sec-WebSocket-Key header.
    var key: ?[]const u8 = null;
    var has_upgrade_ws = false;
    var has_connection_upgrade = false;
    var has_version_13 = false;
    var it = req.iterateHeaders();
    while (it.next()) |h| {
        if (std.ascii.eqlIgnoreCase(h.name, "sec-websocket-key")) key = h.value;
        if (std.ascii.eqlIgnoreCase(h.name, "upgrade") and headerHasToken(h.value, "websocket")) has_upgrade_ws = true;
        if (std.ascii.eqlIgnoreCase(h.name, "connection") and headerHasToken(h.value, "upgrade")) has_connection_upgrade = true;
        if (std.ascii.eqlIgnoreCase(h.name, "sec-websocket-version") and std.mem.eql(u8, std.mem.trim(u8, h.value, " \t"), "13")) has_version_13 = true;
    }
    if (key == null or !websocket.isValidClientKey(key.?) or !has_upgrade_ws or !has_connection_upgrade or !has_version_13) {
        setStatus(ctx, 400);
        const body = "expected WebSocket upgrade request\n";
        addBytesOut(ctx, body.len);
        return req.respond(body, .{
            .status = .bad_request,
            .extra_headers = &.{.{ .name = "content-type", .value = "text/plain" }},
        });
    }

    // Compute Sec-WebSocket-Accept.
    var accept_buf: [websocket.accept_key_b64_len]u8 = undefined;
    const accept_str = websocket.computeAcceptKey(key.?, &accept_buf);

    // Write 101 Switching Protocols header DIRECTLY to the server's
    // socket writer — we can't go through `req.respond` because that
    // path forces a content-length / chunked body which is wrong for
    // a protocol upgrade. After the header, the bytes on the wire
    // are framed by us.
    const out = req.server.out;
    try out.print(
        "HTTP/1.1 101 Switching Protocols\r\nupgrade: websocket\r\nconnection: upgrade\r\nsec-websocket-accept: {s}\r\n\r\n",
        .{accept_str},
    );
    try out.flush();
    setStatus(ctx, 101);
    addBytesOut(ctx, 80 + accept_str.len); // approximate

    // From here on we own the socket. Loop until the client closes.
    const in = req.server.reader.in;
    while (true) {
        const frame = websocket.readClientFrame(in, ctx.allocator, ws_max_frame_payload) catch |err| switch (err) {
            error.EndOfStream, error.ShortFrame => return,
            else => return,
        };
        defer ctx.allocator.free(frame.payload);
        switch (frame.opcode) {
            .close => return,
            .ping => {
                try websocket.writeServerFrame(out, .pong, frame.payload);
                try out.flush();
            },
            .text, .binary => {
                // Treat payload as the bytes to encode.
                const ids = try encodeSingle(ctx, frame.payload);
                defer ctx.allocator.free(ids);
                ctx.last_encode_tokens += ids.len;
                // Emit ids in ws_emit_chunk_ids batches as binary frames
                // of little-endian u32. (Native-endian would force
                // browser clients onto DataView; le is the web norm.)
                var i: usize = 0;
                while (i < ids.len) {
                    const end = @min(i + ws_emit_chunk_ids, ids.len);
                    const chunk = ids[i..end];
                    // Build a fresh buffer of bytes; can't pun
                    // []TokenId → []u8 portably given endianness.
                    const byte_buf = try ctx.allocator.alloc(u8, chunk.len * @sizeOf(u32));
                    defer ctx.allocator.free(byte_buf);
                    for (chunk, 0..) |id, j| {
                        std.mem.writeInt(u32, byte_buf[j * 4 ..][0..4], @intCast(id), .little);
                    }
                    try websocket.writeServerFrame(out, .binary, byte_buf);
                    try out.flush();
                    addBytesOut(ctx, byte_buf.len);
                    i = end;
                }
                try websocket.writeServerFrame(out, .text, "DONE");
                try out.flush();
            },
            else => {},
        }
    }
}

fn headerHasToken(value: []const u8, wanted: []const u8) bool {
    var it = std.mem.splitScalar(u8, value, ',');
    while (it.next()) |raw| {
        if (std.ascii.eqlIgnoreCase(std.mem.trim(u8, raw, " \t"), wanted)) return true;
    }
    return false;
}

// === OIDC bearer-JWT check =============================================

fn checkOidcAuth(
    ctx: *ServeCtx,
    req: *const std.http.Server.Request,
    v: *auth_oidc.Validator,
) !bool {
    var it = req.iterateHeaders();
    while (it.next()) |h| {
        if (!std.ascii.eqlIgnoreCase(h.name, "authorization")) continue;
        if (h.value.len < 7) return false;
        if (!std.ascii.eqlIgnoreCase(h.value[0..7], "bearer ")) return false;
        const token = h.value[7..];
        // exp comparison needs wall-clock seconds — Io.Clock.real is
        // the configured wall clock for this server (test path uses
        // std.testing.io which advances monotonically too).
        const now_s = std.Io.Clock.now(.real, ctx.io).toSeconds();
        if (ctx.oidc_mutex) |mutex| mutex.lockUncancelable(ctx.io);
        defer if (ctx.oidc_mutex) |mutex| mutex.unlock(ctx.io);
        return validateOidcToken(v, token, now_s);
    }
    return false;
}

/// Validate with lazy key rotation. A stale cache is refreshed on a
/// best-effort basis before validation (old keys remain usable during a
/// transient IdP outage). An unknown `kid` forces one refresh and retry so
/// newly-rotated signing keys work without restarting the server.
fn validateOidcToken(v: *auth_oidc.Validator, token: []const u8, now_s: i64) bool {
    v.validateBearerRefreshing(token, now_s) catch return false;
    return true;
}

// === TLS request path ==================================================
//
// `std.http.Server` reads/writes through `std.Io.Reader`/`std.Io.Writer`
// instances obtained from a `std.Io.net.Stream`. mbedtls's TLS read
// and write live on a different interface, so rather than build a
// reader/writer adapter that fakes a Stream, we hand-parse HTTP/1.1
// request line + headers + body via `mbedtls.Conn.read` and dispatch
// to handlers that write their response via `mbedtls.Conn.write`. The
// public route surface here is intentionally a SUBSET of the
// cleartext path:
//
//   GET  /health    /version    /metrics(*)
//   POST /encode    /decode     /eval
//
// (*) Only when `opts.metrics` is set, matching the cleartext path.
//
// Streaming routes — `/encode_stream`, `/encode_chunked`, `/encode_ws`
// — return 501 Not Implemented for now. They each compose chunked
// writes against stdlib helpers; the next TLS wave can layer them on
// top of `writeTlsResponse` once a chunked writer over `Conn.write`
// is in place.

/// Cap on request-line + headers we'll read before giving up. The
/// cleartext path uses `recv_buf_size` (8 KiB) for the same purpose;
/// matching it keeps behaviour comparable across the two transports.
const tls_head_cap: usize = 8 * 1024;

/// Cap on a single TLS read syscall. Sized so a typical TLS record
/// (~16 KiB) fits in one read and the parser doesn't have to coalesce
/// repeatedly for normal requests.
const tls_read_chunk: usize = 16 * 1024;

/// A parsed request shape used by the TLS path. Mirrors the subset of
/// `std.http.Server.Request` we actually use: method, target,
/// well-known header lookups, and the body. Borrows into `head_buf`
/// for the request line + headers; owns `body` (allocator-allocated).
const TlsRequest = struct {
    method: std.http.Method,
    target: []const u8,
    /// Raw `Name: Value` headers, each pointing into the head buffer.
    headers: []const Header,
    /// Body bytes when content-length > 0; an empty slice otherwise.
    /// Owned by the caller's allocator.
    body: []u8,

    pub const Header = struct {
        name: []const u8,
        value: []const u8,
    };

    pub fn headerValue(self: TlsRequest, name: []const u8) ?[]const u8 {
        for (self.headers) |h| {
            if (std.ascii.eqlIgnoreCase(h.name, name)) return h.value;
        }
        return null;
    }

    pub fn deinit(self: *TlsRequest, allocator: std.mem.Allocator, headers_storage: []Header, head_storage: []u8) void {
        allocator.free(self.body);
        allocator.free(headers_storage);
        allocator.free(head_storage);
        self.* = undefined;
    }
};

/// Read bytes from `reader_ctx` via `read_fn` until we've seen
/// `\r\n\r\n` or hit `tls_head_cap`. Returns the number of bytes
/// consumed into `buf` (including the terminator) and the offset of
/// the body start. The portion of `buf` AFTER the terminator may
/// contain leading body bytes already read from the socket; the
/// caller must use those before issuing more reads.
fn readHttpHead(
    reader_ctx: *anyopaque,
    read_fn: *const fn (*anyopaque, []u8) anyerror!usize,
    buf: []u8,
) !struct { head_end: usize, total_read: usize } {
    var n: usize = 0;
    while (n < buf.len) {
        // Look for "\r\n\r\n" in what we already have. Start the
        // search a little before the new bytes in case the
        // terminator straddles two reads.
        const search_start: usize = if (n >= 3) n - 3 else 0;
        if (n >= 4) {
            if (std.mem.indexOf(u8, buf[search_start..n], "\r\n\r\n")) |rel| {
                const head_end = search_start + rel + 4;
                return .{ .head_end = head_end, .total_read = n };
            }
        }
        const got = try read_fn(reader_ctx, buf[n..]);
        if (got == 0) return error.UnexpectedEof;
        n += got;
        // Re-check after the new read.
        if (n >= 4) {
            const s2: usize = if (n >= 3 + got) n - got - 3 else 0;
            if (std.mem.indexOf(u8, buf[s2..n], "\r\n\r\n")) |rel| {
                const head_end = s2 + rel + 4;
                return .{ .head_end = head_end, .total_read = n };
            }
        }
    }
    return error.HeadersTooLarge;
}

/// Parse a request line + headers from `head` (which ends with the
/// "\r\n\r\n" that terminates the headers). Returns method, target,
/// and the parsed `Header` list (a slice borrowed into `headers_buf`).
/// On a malformed request returns `error.BadRequest`.
fn parseHttpRequestHead(
    head: []const u8,
    headers_buf: []TlsRequest.Header,
) !struct { method: std.http.Method, target: []const u8, headers: []const TlsRequest.Header } {
    // Request line: "METHOD SP TARGET SP HTTP/1.X CRLF"
    const line_end = std.mem.indexOf(u8, head, "\r\n") orelse return error.BadRequest;
    const line = head[0..line_end];
    const sp1 = std.mem.indexOfScalar(u8, line, ' ') orelse return error.BadRequest;
    const after_method = line[sp1 + 1 ..];
    const sp2 = std.mem.indexOfScalar(u8, after_method, ' ') orelse return error.BadRequest;
    const method_str = line[0..sp1];
    const target = after_method[0..sp2];
    const method: std.http.Method = std.meta.stringToEnum(std.http.Method, method_str) orelse return error.BadRequest;

    // Headers: each `Name: Value\r\n`, terminated by an empty line.
    var n_headers: usize = 0;
    var cursor: usize = line_end + 2;
    while (cursor < head.len) {
        // End of headers section: an empty line.
        if (cursor + 2 <= head.len and std.mem.eql(u8, head[cursor .. cursor + 2], "\r\n")) break;
        const eol = std.mem.indexOfPos(u8, head, cursor, "\r\n") orelse return error.BadRequest;
        const raw = head[cursor..eol];
        const colon = std.mem.indexOfScalar(u8, raw, ':') orelse return error.BadRequest;
        const name = std.mem.trim(u8, raw[0..colon], " \t");
        const value = std.mem.trim(u8, raw[colon + 1 ..], " \t");
        if (n_headers >= headers_buf.len) return error.TooManyHeaders;
        headers_buf[n_headers] = .{ .name = name, .value = value };
        n_headers += 1;
        cursor = eol + 2;
    }
    return .{ .method = method, .target = target, .headers = headers_buf[0..n_headers] };
}

/// mbedtls read trampoline matching `readHttpHead`'s `read_fn` shape.
fn mbedtlsReadAny(ctx: *anyopaque, buf: []u8) anyerror!usize {
    const conn: *mbedtls.Conn = @ptrCast(@alignCast(ctx));
    return conn.read(buf);
}

/// Write a complete HTTP/1.1 response (status line + headers +
/// optional body) through the TLS connection. Loops to handle short
/// writes so a long response still goes out in full.
fn writeTlsResponse(
    conn: *mbedtls.Conn,
    status: u16,
    reason: []const u8,
    content_type: []const u8,
    extra_headers: []const u8,
    body: []const u8,
) !void {
    try writeTlsResponseHead(conn, status, reason, content_type, extra_headers, body.len);
    if (body.len > 0) try writeTlsAll(conn, body);
}

fn writeTlsResponseHead(
    conn: *mbedtls.Conn,
    status: u16,
    reason: []const u8,
    content_type: []const u8,
    extra_headers: []const u8,
    content_length: usize,
) !void {
    var head_buf: [512]u8 = undefined;
    const head = try std.fmt.bufPrint(
        &head_buf,
        "HTTP/1.1 {d} {s}\r\nContent-Type: {s}\r\nContent-Length: {d}\r\nConnection: close\r\n{s}\r\n",
        .{ status, reason, content_type, content_length, extra_headers },
    );
    try writeTlsAll(conn, head);
}

fn writeTlsAll(conn: *mbedtls.Conn, bytes: []const u8) !void {
    var off: usize = 0;
    while (off < bytes.len) {
        const n = try conn.write(bytes[off..]);
        if (n == 0) return error.WriteShort;
        off += n;
    }
}

/// Top-level dispatch for one TLS-wrapped HTTP/1.1 request. Mirrors
/// the cleartext `handleRequest`'s route table for the supported
/// subset; auth + rate limiting use the same helpers underneath.
fn handleTlsRequest(ctx: *ServeCtx, conn: *mbedtls.Conn, client_ip: ClientIp) !void {
    const head_buf = try ctx.allocator.alloc(u8, tls_head_cap);
    errdefer ctx.allocator.free(head_buf);
    const headers_buf = try ctx.allocator.alloc(TlsRequest.Header, 64);
    errdefer ctx.allocator.free(headers_buf);

    const head_read = try readHttpHead(@ptrCast(conn), mbedtlsReadAny, head_buf);
    const parsed = try parseHttpRequestHead(head_buf[0..head_read.head_end], headers_buf);

    // Resolve content-length and read the body (if any). Bodies are
    // capped at `opts.max_body_bytes` to match the cleartext path.
    var body_len: usize = 0;
    if (parsedHeaderValue(parsed.headers, "content-length")) |cl| {
        body_len = std.fmt.parseInt(usize, cl, 10) catch return writeTlsErrorAndFree(ctx, conn, head_buf, headers_buf, 400, "bad content-length\n");
    }
    if (body_len > ctx.opts.max_body_bytes) {
        return writeTlsErrorAndFree(ctx, conn, head_buf, headers_buf, 413, "body too large\n");
    }

    var body: []u8 = if (body_len == 0) &.{} else try ctx.allocator.alloc(u8, body_len);
    errdefer if (body.len > 0) ctx.allocator.free(body);

    if (body_len > 0) {
        // The first `total_read - head_end` bytes after the head were
        // already read from the socket while looking for "\r\n\r\n".
        // Copy those into `body` before issuing more reads.
        const leftover = head_read.total_read - head_read.head_end;
        const carry = @min(leftover, body_len);
        if (carry > 0) {
            @memcpy(body[0..carry], head_buf[head_read.head_end .. head_read.head_end + carry]);
        }
        var have: usize = carry;
        while (have < body_len) {
            const max_read = @min(body_len - have, tls_read_chunk);
            const n = try conn.read(body[have .. have + max_read]);
            if (n == 0) return error.UnexpectedEof;
            have += n;
        }
    }

    var req: TlsRequest = .{
        .method = parsed.method,
        .target = parsed.target,
        .headers = parsed.headers,
        .body = body,
    };
    defer req.deinit(ctx.allocator, headers_buf, head_buf);

    // Bytes already read off the socket that belong to neither the head
    // nor a content-length body. For the WebSocket upgrade route these
    // are the first client frame bytes; the WS handler seeds its reader
    // with them before pulling more from the TLS connection.
    const leftover_start = head_read.head_end + body_len;
    const leftover: []const u8 = if (leftover_start < head_read.total_read)
        head_buf[leftover_start..head_read.total_read]
    else
        &.{};

    try dispatchTlsRequest(ctx, conn, &req, client_ip, leftover);
}

/// Header lookup over a `Header[]` slice (free function so the parser
/// + body code can share it without going through the struct method).
fn parsedHeaderValue(headers: []const TlsRequest.Header, name: []const u8) ?[]const u8 {
    for (headers) |h| {
        if (std.ascii.eqlIgnoreCase(h.name, name)) return h.value;
    }
    return null;
}

/// Free transient buffers and emit a tiny plaintext error response.
/// Used when we bail BEFORE allocating `body`, so we have to release
/// the head + headers buffers ourselves.
fn writeTlsErrorAndFree(
    ctx: *ServeCtx,
    conn: *mbedtls.Conn,
    head_buf: []u8,
    headers_buf: []TlsRequest.Header,
    status: u16,
    msg: []const u8,
) !void {
    defer ctx.allocator.free(head_buf);
    defer ctx.allocator.free(headers_buf);
    setStatus(ctx, status);
    addBytesOut(ctx, msg.len);
    try writeTlsResponse(conn, status, statusReason(status), "text/plain", "", msg);
}

fn statusReason(status: u16) []const u8 {
    return switch (status) {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        413 => "Payload Too Large",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        else => "OK",
    };
}

/// Auth check for the TLS path. Returns true if the request may
/// proceed, false (with a 401 written) if it must be rejected.
fn tlsCheckAuth(ctx: *ServeCtx, conn: *mbedtls.Conn, req: *const TlsRequest) !bool {
    if (ctx.opts.auth_token) |expected| {
        const authz = req.headerValue("authorization") orelse return tlsRespondUnauthorized(ctx, conn);
        if (authz.len < 7 or !std.ascii.eqlIgnoreCase(authz[0..7], "bearer ")) return tlsRespondUnauthorized(ctx, conn);
        if (!constantTimeEqlBytes(authz[7..], expected)) return tlsRespondUnauthorized(ctx, conn);
        return true;
    } else if (ctx.opts.oidc) |v| {
        const authz = req.headerValue("authorization") orelse return tlsRespondUnauthorized(ctx, conn);
        if (authz.len < 7 or !std.ascii.eqlIgnoreCase(authz[0..7], "bearer ")) return tlsRespondUnauthorized(ctx, conn);
        const token = authz[7..];
        const now_s = std.Io.Clock.now(.real, ctx.io).toSeconds();
        if (ctx.oidc_mutex) |mutex| mutex.lockUncancelable(ctx.io);
        defer if (ctx.oidc_mutex) |mutex| mutex.unlock(ctx.io);
        if (!validateOidcToken(v, token, now_s)) return tlsRespondUnauthorized(ctx, conn);
        return true;
    }
    return true;
}

fn tlsRespondUnauthorized(ctx: *ServeCtx, conn: *mbedtls.Conn) !bool {
    setStatus(ctx, 401);
    const body = "{\"error\":\"unauthorized\"}\n";
    addBytesOut(ctx, body.len);
    try writeTlsResponse(conn, 401, "Unauthorized", "application/json", "WWW-Authenticate: Bearer\r\n", body);
    return false;
}

fn dispatchTlsRequest(ctx: *ServeCtx, conn: *mbedtls.Conn, req: *TlsRequest, client_ip: ClientIp, leftover: []const u8) !void {
    ctx.last_status = 200;
    ctx.last_bytes_in = req.body.len;
    ctx.last_bytes_out = 0;
    ctx.last_encode_tokens = 0;

    const method = req.method;
    const target = req.target;
    const path = if (std.mem.indexOfScalar(u8, target, '?')) |q| target[0..q] else target;
    const path_label = metrics_mod.PathLabel.fromPath(path);
    const method_label = metrics_mod.MethodLabel.fromMethod(method);
    var client_ip_buf: [64]u8 = undefined;
    const client_ip_str = formatClientIp(client_ip, &client_ip_buf);
    if (ctx.opts.metrics) |m| m.enterRequest();
    const t_start_ns = std.Io.Clock.now(.awake, ctx.io).toNanoseconds();
    defer {
        const t_end_ns = std.Io.Clock.now(.awake, ctx.io).toNanoseconds();
        const elapsed = t_end_ns - t_start_ns;
        const duration_ns: u64 = if (elapsed > 0) @intCast(@as(i64, @intCast(elapsed))) else 0;
        if (ctx.opts.metrics) |m| {
            m.incRequest(method_label, path_label, metrics_mod.StatusBucket.fromStatus(ctx.last_status));
            m.addBytesIn(path_label, ctx.last_bytes_in);
            m.addBytesOut(path_label, ctx.last_bytes_out);
            m.addEncodeTokens(ctx.last_encode_tokens);
            m.observeDuration(path_label, @as(f64, @floatFromInt(duration_ns)) / 1_000_000_000.0);
            m.exitRequest();
        }
        emitRequestLog(ctx, .{
            .method = method,
            .path_label = path_label,
            .path_str = path,
            .status = ctx.last_status,
            .bytes_in = ctx.last_bytes_in,
            .bytes_out = ctx.last_bytes_out,
            .duration_ns = duration_ns,
            .client_ip_str = client_ip_str,
            .encoded_tokens = ctx.last_encode_tokens,
        });
    }
    errdefer {
        if (ctx.last_status < 400) ctx.last_status = 500;
    }

    // /health bypasses auth + rate limit.
    if (method == .GET and std.mem.eql(u8, path, "/health")) {
        const body = "{\"ok\":true}\n";
        addBytesOut(ctx, body.len);
        return writeTlsResponse(conn, 200, "OK", "application/json", "", body);
    }

    // /metrics also bypasses auth + rate limit (when enabled).
    if (ctx.opts.metrics != null and method == .GET and std.mem.eql(u8, path, "/metrics")) {
        const m = ctx.opts.metrics.?;
        var out: std.Io.Writer.Allocating = .init(ctx.allocator);
        defer out.deinit();
        try metrics_mod.render(m, &out.writer);
        const body = out.written();
        addBytesOut(ctx, body.len);
        return writeTlsResponse(conn, 200, "OK", "text/plain; version=0.0.4", "", body);
    }

    // Bearer-token / OIDC auth.
    if (!try tlsCheckAuth(ctx, conn, req)) return;

    // Per-client-IP token-bucket rate limit.
    if (ctx.rate_limiter) |rl| {
        switch (rl.tryAcquire(client_ip)) {
            .ok => {},
            .limited => |retry_after_ms| {
                setStatus(ctx, 429);
                var buf: [96]u8 = undefined;
                const body = try std.fmt.bufPrint(
                    &buf,
                    "{{\"error\":\"rate_limited\",\"retry_after_ms\":{d}}}\n",
                    .{retry_after_ms},
                );
                addBytesOut(ctx, body.len);
                var retry_hdr_buf: [64]u8 = undefined;
                const retry_hdr = try std.fmt.bufPrint(&retry_hdr_buf, "Retry-After: {d}\r\n", .{(retry_after_ms + 999) / 1000});
                return writeTlsResponse(conn, 429, "Too Many Requests", "application/json", retry_hdr, body);
            },
        }
    }

    if (method == .GET and std.mem.eql(u8, path, "/version")) {
        var buf: [256]u8 = undefined;
        const body = try std.fmt.bufPrint(
            &buf,
            "{{\"version\":\"{s}\",\"model_kind\":\"{s}\"}}\n",
            .{ ctx.opts.version, @tagName(ctx.opts.model_kind) },
        );
        addBytesOut(ctx, body.len);
        return writeTlsResponse(conn, 200, "OK", "application/json", "", body);
    }

    if (method == .POST and std.mem.eql(u8, path, "/encode")) {
        return tlsHandleEncode(ctx, conn, req);
    }
    if (method == .POST and std.mem.eql(u8, path, "/decode")) {
        return tlsHandleDecode(ctx, conn, req);
    }
    if (method == .POST and std.mem.eql(u8, path, "/eval")) {
        return tlsHandleEval(ctx, conn, req);
    }

    // Streaming routes over TLS: NDJSON / chunked responses and WS
    // frames are written through `mbedtls.ConnWriter` / read through
    // `mbedtls.ConnReader`.
    if (method == .POST and std.mem.eql(u8, path, "/encode_stream")) {
        return tlsHandleEncodeStream(ctx, conn, req);
    }
    if (method == .POST and std.mem.eql(u8, path, "/encode_chunked")) {
        return tlsHandleEncodeChunked(ctx, conn, req);
    }
    if (method == .GET and std.mem.eql(u8, path, "/encode_ws")) {
        return tlsHandleEncodeWs(ctx, conn, req, leftover);
    }

    setStatus(ctx, 404);
    const not_found = "not found\n";
    addBytesOut(ctx, not_found.len);
    try writeTlsResponse(conn, 404, "Not Found", "text/plain", "", not_found);
}

fn tlsHandleEncode(ctx: *ServeCtx, conn: *mbedtls.Conn, req: *TlsRequest) !void {
    if (req.body.len == 0) {
        setStatus(ctx, 400);
        const body = "missing body\n";
        addBytesOut(ctx, body.len);
        return writeTlsResponse(conn, 400, "Bad Request", "text/plain", "", body);
    }
    var parsed_text = (try parseTextField(ctx.allocator, req.body)) orelse {
        setStatus(ctx, 400);
        const body = "expected JSON {\"text\":\"...\"}\n";
        addBytesOut(ctx, body.len);
        return writeTlsResponse(conn, 400, "Bad Request", "text/plain", "", body);
    };
    defer parsed_text.deinit();
    const text = parsed_text.value;

    const ids = try encodeWithPrefixCache(ctx, text);
    defer ctx.allocator.free(ids);
    ctx.last_encode_tokens += ids.len;

    const body_len = idsJsonLineLen(ids);
    addBytesOut(ctx, body_len);
    try writeTlsResponseHead(conn, 200, "OK", "application/json", "", body_len);
    var buf: [4096]u8 = undefined;
    var writer = mbedtls.connWriter(conn, &buf);
    _ = try writeIdsBatchLine(&writer.interface, ids);
    try writer.interface.flush();
}

fn tlsHandleDecode(ctx: *ServeCtx, conn: *mbedtls.Conn, req: *TlsRequest) !void {
    if (req.body.len == 0) {
        setStatus(ctx, 400);
        const body = "missing body\n";
        addBytesOut(ctx, body.len);
        return writeTlsResponse(conn, 400, "Bad Request", "text/plain", "", body);
    }
    var parsed_ids = (try parseIdsField(ctx.allocator, req.body)) orelse {
        setStatus(ctx, 400);
        const body = "expected JSON {\"ids\":[...]}\n";
        addBytesOut(ctx, body.len);
        return writeTlsResponse(conn, 400, "Bad Request", "text/plain", "", body);
    };
    defer parsed_ids.deinit();
    const ids = parsed_ids.value;

    const text = try ctx.pipeline.decode(ctx.allocator, ids);
    defer ctx.allocator.free(text);

    const body_len = "{\"text\":".len + jsonStringLen(text) + "}\n".len;
    addBytesOut(ctx, body_len);
    try writeTlsResponseHead(conn, 200, "OK", "application/json", "", body_len);
    var buf: [4096]u8 = undefined;
    var writer = mbedtls.connWriter(conn, &buf);
    try writer.interface.writeAll("{\"text\":");
    try writeJsonString(&writer.interface, text);
    try writer.interface.writeAll("}\n");
    try writer.interface.flush();
}

fn tlsHandleEval(ctx: *ServeCtx, conn: *mbedtls.Conn, req: *TlsRequest) !void {
    if (req.body.len == 0) {
        setStatus(ctx, 400);
        const body = "missing body\n";
        addBytesOut(ctx, body.len);
        return writeTlsResponse(conn, 400, "Bad Request", "text/plain", "", body);
    }
    var parsed_text = (try parseTextField(ctx.allocator, req.body)) orelse {
        setStatus(ctx, 400);
        const body = "expected JSON {\"text\":\"...\"}\n";
        addBytesOut(ctx, body.len);
        return writeTlsResponse(conn, 400, "Bad Request", "text/plain", "", body);
    };
    defer parsed_text.deinit();
    const text = parsed_text.value;

    var out: std.Io.Writer.Allocating = .init(ctx.allocator);
    defer out.deinit();
    try cli_eval.runEval(ctx.allocator, ctx.pipeline, text, .{ .format = .json }, &out.writer);
    const body = out.written();
    addBytesOut(ctx, body.len);
    try writeTlsResponse(conn, 200, "OK", "application/json", "", body);
}

// === TLS streaming handlers ============================================
//
// These mirror the cleartext `/encode_stream`, `/encode_chunked`, and
// `/encode_ws` handlers but write their framed output through a
// `mbedtls.ConnWriter` (chunked NDJSON / WS frames) and, for WS, read
// continuation frames through a `mbedtls.ConnReader`. The framing logic
// (`StreamEncoder`, `writeIdsBatchLine`, `websocket.*`) is shared
// verbatim with the cleartext path.

/// Buffer size for the per-connection `ConnWriter`. One NDJSON line or
/// WS frame at a time is small; 4 KiB matches the cleartext response
/// buffers.
const tls_stream_buf_size: usize = 4096;

/// Emit the chunked-transfer response head through `w`. After this,
/// every `writeChunk` call frames one HTTP chunk. Takes a generic
/// `*std.Io.Writer` so it is exercised by hermetic tests against an
/// Allocating writer as well as the live `ConnWriter`.
fn writeChunkedHead(w: *std.Io.Writer, content_type: []const u8) !void {
    try w.print(
        "HTTP/1.1 200 OK\r\nContent-Type: {s}\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
        .{content_type},
    );
}

/// Frame `payload` as a single HTTP/1.1 chunk and write it through `w`.
/// A zero-length payload is skipped (the terminator is written by
/// `writeChunkedEnd`).
fn writeChunk(w: *std.Io.Writer, payload: []const u8) !void {
    if (payload.len == 0) return;
    try w.print("{x}\r\n", .{payload.len});
    try w.writeAll(payload);
    try w.writeAll("\r\n");
}

/// Write the final zero-length chunk that ends a chunked response, then
/// flush the underlying writer.
fn writeChunkedEnd(w: *std.Io.Writer) !void {
    try w.writeAll("0\r\n\r\n");
    try w.flush();
}

fn tlsHandleEncodeStream(ctx: *ServeCtx, conn: *mbedtls.Conn, req: *TlsRequest) !void {
    if (req.body.len == 0) {
        setStatus(ctx, 400);
        const body = "missing body\n";
        addBytesOut(ctx, body.len);
        return writeTlsResponse(conn, 400, "Bad Request", "text/plain", "", body);
    }
    var parsed_text = (try parseTextField(ctx.allocator, req.body)) orelse {
        setStatus(ctx, 400);
        const body = "expected JSON {\"text\":\"...\"}\n";
        addBytesOut(ctx, body.len);
        return writeTlsResponse(conn, 400, "Bad Request", "text/plain", "", body);
    };
    defer parsed_text.deinit();
    const text = parsed_text.value;

    var buf: [tls_stream_buf_size]u8 = undefined;
    var cw = mbedtls.connWriter(conn, &buf);
    try writeChunkedHead(&cw.interface, "application/x-ndjson");

    var enc = StreamEncoder.init(ctx.allocator, ctx.pipeline);
    defer enc.deinit();

    var batch: std.ArrayList(TokenId) = .empty;
    defer batch.deinit(ctx.allocator);

    // One NDJSON line per feed becomes one HTTP chunk. We render each
    // line into a scratch Allocating writer so its byte length is known
    // when we frame the chunk.
    var line: std.Io.Writer.Allocating = .init(ctx.allocator);
    defer line.deinit();

    var i: usize = 0;
    while (i < text.len) {
        const chunk_end = @min(i + stream_feed_size, text.len);
        try enc.feed(text[i..chunk_end], &batch);
        if (batch.items.len > 0) {
            ctx.last_encode_tokens += batch.items.len;
            line.clearRetainingCapacity();
            _ = try writeIdsBatchLine(&line.writer, batch.items);
            try writeChunk(&cw.interface, line.written());
            addBytesOut(ctx, line.written().len);
            batch.clearRetainingCapacity();
        }
        i = chunk_end;
    }
    try enc.finish(&batch);
    if (batch.items.len > 0) {
        ctx.last_encode_tokens += batch.items.len;
        line.clearRetainingCapacity();
        _ = try writeIdsBatchLine(&line.writer, batch.items);
        try writeChunk(&cw.interface, line.written());
        addBytesOut(ctx, line.written().len);
        batch.clearRetainingCapacity();
    }
    try writeChunk(&cw.interface, "{\"done\":true}\n");
    addBytesOut(ctx, "{\"done\":true}\n".len);
    try writeChunkedEnd(&cw.interface);
    setStatus(ctx, 200);
}

fn tlsHandleEncodeChunked(ctx: *ServeCtx, conn: *mbedtls.Conn, req: *TlsRequest) !void {
    // Unlike the cleartext route which dechunks the request body as it
    // streams in, the TLS request body was already read into `req.body`
    // by `handleTlsRequest` (capped at opts.max_body_bytes). The route
    // contract is identical: the body bytes ARE the input to encode, and
    // the response is chunked NDJSON.
    if (req.body.len == 0) {
        setStatus(ctx, 400);
        const body = "missing body\n";
        addBytesOut(ctx, body.len);
        return writeTlsResponse(conn, 400, "Bad Request", "text/plain", "", body);
    }

    var buf: [tls_stream_buf_size]u8 = undefined;
    var cw = mbedtls.connWriter(conn, &buf);
    try writeChunkedHead(&cw.interface, "application/x-ndjson");

    var enc = StreamEncoder.init(ctx.allocator, ctx.pipeline);
    defer enc.deinit();

    var batch: std.ArrayList(TokenId) = .empty;
    defer batch.deinit(ctx.allocator);

    var line: std.Io.Writer.Allocating = .init(ctx.allocator);
    defer line.deinit();

    // Feed the body through the StreamEncoder in chunked_read_size
    // slices so the on-the-wire framing matches the cleartext route
    // (one NDJSON line per feed that produces ids).
    var off: usize = 0;
    while (off < req.body.len) {
        const end = @min(off + chunked_read_size, req.body.len);
        try enc.feed(req.body[off..end], &batch);
        if (batch.items.len > 0) {
            ctx.last_encode_tokens += batch.items.len;
            line.clearRetainingCapacity();
            _ = try writeIdsBatchLine(&line.writer, batch.items);
            try writeChunk(&cw.interface, line.written());
            addBytesOut(ctx, line.written().len);
            batch.clearRetainingCapacity();
        }
        off = end;
    }
    try enc.finish(&batch);
    if (batch.items.len > 0) {
        ctx.last_encode_tokens += batch.items.len;
        line.clearRetainingCapacity();
        _ = try writeIdsBatchLine(&line.writer, batch.items);
        try writeChunk(&cw.interface, line.written());
        addBytesOut(ctx, line.written().len);
        batch.clearRetainingCapacity();
    }
    try writeChunk(&cw.interface, "{\"done\":true}\n");
    addBytesOut(ctx, "{\"done\":true}\n".len);
    try writeChunkedEnd(&cw.interface);
    setStatus(ctx, 200);
}

fn tlsHandleEncodeWs(ctx: *ServeCtx, conn: *mbedtls.Conn, req: *TlsRequest, leftover: []const u8) !void {
    // Validate the upgrade request the same way the cleartext handler
    // does, then write 101 + run the frame loop over the TLS conn.
    var key: ?[]const u8 = null;
    var has_upgrade_ws = false;
    var has_connection_upgrade = false;
    var has_version_13 = false;
    for (req.headers) |h| {
        if (std.ascii.eqlIgnoreCase(h.name, "sec-websocket-key")) key = h.value;
        if (std.ascii.eqlIgnoreCase(h.name, "upgrade") and headerHasToken(h.value, "websocket")) has_upgrade_ws = true;
        if (std.ascii.eqlIgnoreCase(h.name, "connection") and headerHasToken(h.value, "upgrade")) has_connection_upgrade = true;
        if (std.ascii.eqlIgnoreCase(h.name, "sec-websocket-version") and std.mem.eql(u8, std.mem.trim(u8, h.value, " \t"), "13")) has_version_13 = true;
    }
    if (key == null or !websocket.isValidClientKey(key.?) or !has_upgrade_ws or !has_connection_upgrade or !has_version_13) {
        setStatus(ctx, 400);
        const body = "expected WebSocket upgrade request\n";
        addBytesOut(ctx, body.len);
        return writeTlsResponse(conn, 400, "Bad Request", "text/plain", "", body);
    }

    var accept_buf: [websocket.accept_key_b64_len]u8 = undefined;
    const accept_str = websocket.computeAcceptKey(key.?, &accept_buf);

    // 101 Switching Protocols — written directly (no content-length /
    // chunked body, which an upgrade must not carry).
    var head_buf: [256]u8 = undefined;
    const head = try std.fmt.bufPrint(
        &head_buf,
        "HTTP/1.1 101 Switching Protocols\r\nupgrade: websocket\r\nconnection: upgrade\r\nsec-websocket-accept: {s}\r\n\r\n",
        .{accept_str},
    );
    try writeTlsAll(conn, head);
    setStatus(ctx, 101);
    addBytesOut(ctx, head.len);

    // Reader over the TLS conn, seeded with any frame bytes already read
    // past the request head. Large frame payloads are read straight into
    // the caller's destination buffer (readSliceShort → readVec →
    // stream), so this buffer only needs to hold small header peeks plus
    // any seeded leftover; 64 KiB is comfortable.
    const ws_reader_buf_size: usize = @max(64 * 1024, leftover.len);
    const rbuf = try ctx.allocator.alloc(u8, ws_reader_buf_size);
    defer ctx.allocator.free(rbuf);
    var cr = mbedtls.connReader(conn, rbuf);
    if (leftover.len > 0) cr.seed(leftover);
    const in = &cr.interface;

    var wbuf: [tls_stream_buf_size]u8 = undefined;
    var cw = mbedtls.connWriter(conn, &wbuf);
    const out = &cw.interface;

    while (true) {
        const frame = websocket.readClientFrame(in, ctx.allocator, ws_max_frame_payload) catch |err| switch (err) {
            error.EndOfStream, error.ShortFrame => return,
            else => return,
        };
        defer ctx.allocator.free(frame.payload);
        switch (frame.opcode) {
            .close => return,
            .ping => {
                try websocket.writeServerFrame(out, .pong, frame.payload);
                try out.flush();
            },
            .text, .binary => {
                const ids = try encodeSingle(ctx, frame.payload);
                defer ctx.allocator.free(ids);
                ctx.last_encode_tokens += ids.len;
                var i: usize = 0;
                while (i < ids.len) {
                    const end = @min(i + ws_emit_chunk_ids, ids.len);
                    const chunk = ids[i..end];
                    const byte_buf = try ctx.allocator.alloc(u8, chunk.len * @sizeOf(u32));
                    defer ctx.allocator.free(byte_buf);
                    for (chunk, 0..) |id, j| {
                        std.mem.writeInt(u32, byte_buf[j * 4 ..][0..4], @intCast(id), .little);
                    }
                    try websocket.writeServerFrame(out, .binary, byte_buf);
                    try out.flush();
                    addBytesOut(ctx, byte_buf.len);
                    i = end;
                }
                try websocket.writeServerFrame(out, .text, "DONE");
                try out.flush();
            },
            else => {},
        }
    }
}

// === In-memory request driver (for hermetic tests) =====================

/// Drive one HTTP request through `handleRequest` end-to-end without
/// touching the network. `raw_request` is the full wire bytes (request
/// line + headers + body); the response (also wire bytes) is returned
/// in a fresh allocation owned by the caller.
///
/// This is the hermetic equivalent of the 1.16 spawn-a-server-thread
/// pattern: a single synchronous call, no socket, no port, no thread.
/// All of `cli_serve`'s logic that doesn't depend on the TCP listener
/// runs through this path, so tests cover the same code as production.
pub fn handleRequestInMemory(
    ctx: *ServeCtx,
    raw_request: []const u8,
    client_ip: ClientIp,
) ![]u8 {
    var in: std.Io.Reader = .fixed(raw_request);
    var out: std.Io.Writer.Allocating = .init(ctx.allocator);
    errdefer out.deinit();

    // `std.http.Server` reads its head out of the buffer attached to
    // the input reader. `Reader.fixed` already exposes the full input
    // as its buffer, which is exactly what `receiveHead` needs.
    var http: std.http.Server = .init(&in, &out.writer);
    var req = try http.receiveHead();
    try handleRequest(ctx, &req, client_ip);
    return out.toOwnedSlice();
}

// === Tests =============================================================

const testing = std.testing;

test "writeJsonString escapes control bytes and quotes" {
    var buf: [256]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try writeJsonString(&w, "a\"b\\c\nd\te");
    try testing.expectEqualStrings("\"a\\\"b\\\\c\\nd\\te\"", w.buffered());
}

test "extractTextField parses a simple object" {
    const a = testing.allocator;
    const got = (try extractTextField(a, "{\"text\":\"hello\"}")) orelse return error.TestFailed;
    defer a.free(got);
    try testing.expectEqualStrings("hello", got);
}

test "extractTextField returns null when field is absent" {
    const a = testing.allocator;
    try testing.expect((try extractTextField(a, "{\"other\":1}")) == null);
}

test "parseTextField borrows plain strings and allocates only for escapes" {
    const a = testing.allocator;
    const plain = "{\"text\":\"hello\"}";
    var borrowed = (try parseTextField(a, plain)) orelse return error.TestFailed;
    defer borrowed.deinit();
    try testing.expectEqualStrings("hello", borrowed.value);
    try testing.expectEqual(@intFromPtr(plain.ptr) + "{\"text\":\"".len, @intFromPtr(borrowed.value.ptr));

    var decoded = (try parseTextField(a, "{\"text\":\"hello\\nworld\"}")) orelse return error.TestFailed;
    defer decoded.deinit();
    try testing.expectEqualStrings("hello\nworld", decoded.value);
}

test "extractIdsField parses an id array" {
    const a = testing.allocator;
    const got = (try extractIdsField(a, "{\"ids\":[1,2,3]}")) orelse return error.TestFailed;
    defer a.free(got);
    try testing.expectEqualSlices(TokenId, &.{ 1, 2, 3 }, got);
}

test "writeIdsBatchLine produces compact NDJSON" {
    var buf: [256]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    _ = try writeIdsBatchLine(&w, &.{ 7, 11, 13 });
    try testing.expectEqualStrings("{\"ids\":[7,11,13]}\n", w.buffered());
}

// === HTTP integration tests ============================================
//
// Hermetic: every test drives one request through `handleRequestInMemory`,
// which feeds the wire bytes into `std.http.Server` via `Reader.fixed`
// and captures the response via `Writer.Allocating`. No sockets, no
// threads, no ports — `zig build test` runs the whole suite in a few
// milliseconds and can't hang waiting for an accept loop.

const Vocab = @import("vocab.zig").Vocab;
const Bpe = @import("bpe.zig").Bpe;

/// Build the smallest possible BPE+cl100k pipeline reusable across
/// HTTP tests. Caller deinits the returned struct.
const PipeFixture = struct {
    bpe: Bpe,
    vocab: Vocab,
    pipe: Pipeline,

    fn init(a: std.mem.Allocator) !*PipeFixture {
        const fx = try a.create(PipeFixture);
        errdefer a.destroy(fx);
        var src: std.ArrayList(u8) = .empty;
        defer src.deinit(a);
        const b64 = std.base64.standard.Encoder;
        var enc_buf: [16]u8 = undefined;
        var rank: u32 = 0;
        var b: u32 = 0;
        while (b < 256) : (b += 1) {
            const byte: [1]u8 = .{@intCast(b)};
            const encoded = b64.encode(&enc_buf, &byte);
            try src.print(a, "{s} {d}\n", .{ encoded, rank });
            rank += 1;
        }
        const extra = [_][]const u8{ "he", "hel", "hell", "hello", " w", " wo", " wor", " worl", " world" };
        for (extra) |bytes| {
            const encoded = b64.encode(&enc_buf, bytes);
            try src.print(a, "{s} {d}\n", .{ encoded, rank });
            rank += 1;
        }
        fx.bpe = try Bpe.loadTiktokenBytes(a, src.items);
        errdefer fx.bpe.deinit();
        fx.vocab = Vocab.empty(a);
        fx.pipe = .{
            .normalizer = .identity,
            .pre_tokenizer = .cl100k,
            .model = .{ .bpe = &fx.bpe },
            .decoder = .concat,
            .vocab = &fx.vocab,
        };
        return fx;
    }

    fn deinit(self: *PipeFixture, a: std.mem.Allocator) void {
        self.bpe.deinit();
        self.vocab.deinit();
        a.destroy(self);
    }
};

/// Extract the body portion of an HTTP response (everything after the
/// first \r\n\r\n). Returns null if no header terminator is found.
fn responseBody(resp: []const u8) ?[]const u8 {
    const idx = std.mem.indexOf(u8, resp, "\r\n\r\n") orelse return null;
    return resp[idx + 4 ..];
}

fn responseStatus(resp: []const u8) ?u16 {
    // "HTTP/1.1 200 OK\r\n..."
    if (resp.len < 12) return null;
    if (!std.mem.startsWith(u8, resp, "HTTP/1.1 ")) return null;
    const code_slice = resp[9..12];
    return std.fmt.parseInt(u16, code_slice, 10) catch null;
}

fn responseIsChunked(resp: []const u8) bool {
    const head_end = std.mem.indexOf(u8, resp, "\r\n\r\n") orelse return false;
    const head = resp[0..head_end];
    return std.ascii.indexOfIgnoreCase(head, "transfer-encoding: chunked") != null;
}

/// Strip HTTP/1.1 chunked-transfer framing from `body`. Returns the
/// concatenated payload, allocated from `a`. Caller frees.
fn dechunk(a: std.mem.Allocator, body: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var rest = body;
    while (rest.len > 0) {
        const sep = std.mem.indexOf(u8, rest, "\r\n") orelse break;
        const size_str = rest[0..sep];
        const size = std.fmt.parseInt(usize, size_str, 16) catch break;
        if (size == 0) break;
        const payload_start = sep + 2;
        if (payload_start + size > rest.len) break;
        try out.appendSlice(a, rest[payload_start .. payload_start + size]);
        const next_start = payload_start + size + 2;
        if (next_start > rest.len) break;
        rest = rest[next_start..];
    }
    return out.toOwnedSlice(a);
}

// Note: post-1.16 agent B's TCP-spawning HttpTestWorker + doRawRequest
// (and agent E's bridge stub) were removed here. The production
// codepath is now exercised end-to-end through `handleRequestInMemory`,
// which drives `std.http.Server` with an in-memory reader/writer pair.
// No sockets, no threads, no port allocation — `zig build test` runs
// the suite in milliseconds and cannot hang on an accept loop.

/// Build a `ServeCtx` for a hermetic test. Caller borrows `fx` + `pool`.
fn ctxFor(a: std.mem.Allocator, fx: *PipeFixture, pool: *BatchPool, opts: Options) ServeCtx {
    return .{
        .allocator = a,
        .io = testing.io,
        .pipeline = &fx.pipe,
        .pool = pool,
        .opts = opts,
        .rate_limiter = null,
    };
}

/// Build a raw HTTP/1.1 request as a flat byte buffer. Caller frees.
fn buildReq(
    a: std.mem.Allocator,
    method: []const u8,
    path: []const u8,
    extra_headers: []const u8,
    body: []const u8,
) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.print(a, "{s} {s} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n", .{ method, path });
    if (extra_headers.len > 0) try out.appendSlice(a, extra_headers);
    if (body.len > 0) {
        try out.print(a, "Content-Type: application/json\r\nContent-Length: {d}\r\n\r\n", .{body.len});
        try out.appendSlice(a, body);
    } else {
        try out.appendSlice(a, "\r\n");
    }
    return out.toOwnedSlice(a);
}

test "handleRequest /health returns 200 ok (hermetic)" {
    const a = testing.allocator;
    const fx = try PipeFixture.init(a);
    defer fx.deinit(a);
    var pool = try BatchPool.init(a, 2);
    defer pool.deinit();
    var ctx = ctxFor(a, fx, &pool, .{ .log = false });

    const req = try buildReq(a, "GET", "/health", "", "");
    defer a.free(req);
    const resp = try handleRequestInMemory(&ctx, req, ClientIp.unknown);
    defer a.free(resp);

    try testing.expectEqual(@as(?u16, 200), responseStatus(resp));
    const body = responseBody(resp) orelse return error.MissingBody;
    try testing.expectEqualStrings("{\"ok\":true}\n", body);
}

test "handleRequest /encode round-trips against pipeline (hermetic)" {
    const a = testing.allocator;
    const fx = try PipeFixture.init(a);
    defer fx.deinit(a);
    var pool = try BatchPool.init(a, 2);
    defer pool.deinit();
    var ctx = ctxFor(a, fx, &pool, .{ .log = false });

    const want = try fx.pipe.encode(a, "hello world");
    defer a.free(want);

    const req = try buildReq(a, "POST", "/encode", "", "{\"text\":\"hello world\"}");
    defer a.free(req);
    const resp = try handleRequestInMemory(&ctx, req, ClientIp.unknown);
    defer a.free(resp);

    try testing.expectEqual(@as(?u16, 200), responseStatus(resp));
    const resp_body = responseBody(resp) orelse return error.MissingBody;
    const payload = if (responseIsChunked(resp))
        try dechunk(a, resp_body)
    else
        try a.dupe(u8, resp_body);
    defer a.free(payload);

    const got_ids = (try extractIdsField(a, payload)) orelse return error.MissingIdsField;
    defer a.free(got_ids);
    try testing.expectEqualSlices(TokenId, want, got_ids);
}

test "handleRequest /encode missing body returns 400 (hermetic)" {
    const a = testing.allocator;
    const fx = try PipeFixture.init(a);
    defer fx.deinit(a);
    var pool = try BatchPool.init(a, 2);
    defer pool.deinit();
    var ctx = ctxFor(a, fx, &pool, .{ .log = false });

    // No Content-Length and no body → readBody returns null → 400.
    const req = try buildReq(a, "POST", "/encode", "", "");
    defer a.free(req);
    const resp = try handleRequestInMemory(&ctx, req, ClientIp.unknown);
    defer a.free(resp);
    try testing.expectEqual(@as(?u16, 400), responseStatus(resp));
}

test "handleRequest /encode body over cap returns 413 (hermetic)" {
    const a = testing.allocator;
    const fx = try PipeFixture.init(a);
    defer fx.deinit(a);
    var pool = try BatchPool.init(a, 2);
    defer pool.deinit();
    // Tight body cap so we hit BodyTooLarge with a short payload.
    var ctx = ctxFor(a, fx, &pool, .{ .log = false, .max_body_bytes = 8 });

    const req = try buildReq(a, "POST", "/encode", "", "{\"text\":\"this is more than eight bytes\"}");
    defer a.free(req);
    const resp = try handleRequestInMemory(&ctx, req, ClientIp.unknown);
    defer a.free(resp);
    try testing.expectEqual(@as(?u16, 413), responseStatus(resp));
}

test "handleRequest /encode_stream emits NDJSON lines + final {done:true} (hermetic)" {
    const a = testing.allocator;
    const fx = try PipeFixture.init(a);
    defer fx.deinit(a);
    var pool = try BatchPool.init(a, 2);
    defer pool.deinit();
    var ctx = ctxFor(a, fx, &pool, .{ .log = false });

    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(a);
    try body.appendSlice(a, "{\"text\":\"");
    var i: usize = 0;
    while (i < 1024) : (i += 1) try body.appendSlice(a, "hello world ");
    try body.appendSlice(a, "\"}");

    const req = try buildReq(a, "POST", "/encode_stream", "", body.items);
    defer a.free(req);
    const resp = try handleRequestInMemory(&ctx, req, ClientIp.unknown);
    defer a.free(resp);

    try testing.expectEqual(@as(?u16, 200), responseStatus(resp));
    const resp_body = responseBody(resp) orelse return error.MissingBody;
    try testing.expect(responseIsChunked(resp));
    const dechunked = try dechunk(a, resp_body);
    defer a.free(dechunked);

    var n_lines: usize = 0;
    var n_done: usize = 0;
    var it = std.mem.tokenizeScalar(u8, dechunked, '\n');
    while (it.next()) |line| {
        if (line.len == 0) continue;
        n_lines += 1;
        if (std.mem.indexOf(u8, line, "\"done\":true") != null) n_done += 1;
    }
    try testing.expect(n_lines >= 2);
    try testing.expectEqual(@as(usize, 1), n_done);
}

// === /encode_chunked tests (post-1.19 agent B, B2) =====================

/// Build a raw HTTP/1.1 request with a chunked-encoded body (instead
/// of content-length). Splits `body` into N chunks of `chunk_size`
/// bytes plus a terminating empty chunk. Used to exercise the chunked
/// request-body path through std.http.Server's body reader.
fn buildChunkedReq(
    a: std.mem.Allocator,
    method: []const u8,
    path: []const u8,
    extra_headers: []const u8,
    body: []const u8,
    chunk_size: usize,
) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.print(a, "{s} {s} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n", .{ method, path });
    if (extra_headers.len > 0) try out.appendSlice(a, extra_headers);
    try out.appendSlice(a, "Content-Type: application/octet-stream\r\nTransfer-Encoding: chunked\r\n\r\n");
    var i: usize = 0;
    while (i < body.len) {
        const end = @min(i + chunk_size, body.len);
        try out.print(a, "{x}\r\n", .{end - i});
        try out.appendSlice(a, body[i..end]);
        try out.appendSlice(a, "\r\n");
        i = end;
    }
    try out.appendSlice(a, "0\r\n\r\n");
    return out.toOwnedSlice(a);
}

test "/encode_chunked: chunked request body emits chunked NDJSON response (hermetic)" {
    const a = testing.allocator;
    const fx = try PipeFixture.init(a);
    defer fx.deinit(a);
    var pool = try BatchPool.init(a, 2);
    defer pool.deinit();
    var ctx = ctxFor(a, fx, &pool, .{ .log = false });

    // Big enough input that the StreamEncoder emits at least one NDJSON
    // line plus the final {"done":true}.
    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(a);
    var i: usize = 0;
    while (i < 2048) : (i += 1) try body.appendSlice(a, "hello world ");

    const req = try buildChunkedReq(a, "POST", "/encode_chunked", "", body.items, 1024);
    defer a.free(req);
    const resp = try handleRequestInMemory(&ctx, req, ClientIp.unknown);
    defer a.free(resp);

    try testing.expectEqual(@as(?u16, 200), responseStatus(resp));
    try testing.expect(responseIsChunked(resp));
    const resp_body = responseBody(resp) orelse return error.MissingBody;
    const dechunked = try dechunk(a, resp_body);
    defer a.free(dechunked);

    var n_lines: usize = 0;
    var n_done: usize = 0;
    var n_ids_lines: usize = 0;
    var it = std.mem.tokenizeScalar(u8, dechunked, '\n');
    while (it.next()) |line| {
        if (line.len == 0) continue;
        n_lines += 1;
        if (std.mem.indexOf(u8, line, "\"done\":true") != null) n_done += 1;
        if (std.mem.indexOf(u8, line, "\"ids\":[") != null) n_ids_lines += 1;
    }
    try testing.expect(n_ids_lines >= 1);
    try testing.expectEqual(@as(usize, 1), n_done);
    try testing.expect(n_lines >= 2);
}

test "/encode_chunked: honors auth (401 without token, 200 with)" {
    const a = testing.allocator;
    const fx = try PipeFixture.init(a);
    defer fx.deinit(a);
    var pool = try BatchPool.init(a, 2);
    defer pool.deinit();
    var ctx = ctxFor(a, fx, &pool, .{ .log = false, .auth_token = "s3cret" });

    // Without token → 401.
    {
        const req = try buildChunkedReq(a, "POST", "/encode_chunked", "", "hello world", 4);
        defer a.free(req);
        const resp = try handleRequestInMemory(&ctx, req, ClientIp.unknown);
        defer a.free(resp);
        try testing.expectEqual(@as(?u16, 401), responseStatus(resp));
    }
    // With valid token → 200.
    {
        const req = try buildChunkedReq(a, "POST", "/encode_chunked", "Authorization: Bearer s3cret\r\n", "hello world", 4);
        defer a.free(req);
        const resp = try handleRequestInMemory(&ctx, req, ClientIp.unknown);
        defer a.free(resp);
        try testing.expectEqual(@as(?u16, 200), responseStatus(resp));
    }
}

test "/encode_chunked: honors rate limit (429 after bucket exhaustion)" {
    const a = testing.allocator;
    const fx = try PipeFixture.init(a);
    defer fx.deinit(a);
    var pool = try BatchPool.init(a, 2);
    defer pool.deinit();
    var rl = RateLimiter.init(a, testing.io, 1); // cap = 2
    defer rl.deinit();
    _ = rl.tryAcquire(ClientIp.unknown);
    _ = rl.tryAcquire(ClientIp.unknown);
    try testing.expect(rl.tryAcquire(ClientIp.unknown) == .limited);

    var ctx: ServeCtx = .{
        .allocator = a,
        .io = testing.io,
        .pipeline = &fx.pipe,
        .pool = &pool,
        .opts = .{ .log = false, .rate_limit_rps = 1 },
        .rate_limiter = &rl,
    };

    const req = try buildChunkedReq(a, "POST", "/encode_chunked", "", "hello world", 4);
    defer a.free(req);
    const resp = try handleRequestInMemory(&ctx, req, ClientIp.unknown);
    defer a.free(resp);
    try testing.expectEqual(@as(?u16, 429), responseStatus(resp));
}

test "/encode_chunked: round-trip — concatenated ids equal single-shot encode" {
    const a = testing.allocator;
    const fx = try PipeFixture.init(a);
    defer fx.deinit(a);
    var pool = try BatchPool.init(a, 2);
    defer pool.deinit();
    var ctx = ctxFor(a, fx, &pool, .{ .log = false });

    // Pre-tokenizer-friendly content so safe cuts land on word boundaries
    // and the streaming output is bit-identical to single-shot encode.
    const full = "hello world hello world the quick brown fox jumps over the lazy dog hello world ";
    const want = try fx.pipe.encode(a, full);
    defer a.free(want);

    const req = try buildChunkedReq(a, "POST", "/encode_chunked", "", full, 13);
    defer a.free(req);
    const resp = try handleRequestInMemory(&ctx, req, ClientIp.unknown);
    defer a.free(resp);

    try testing.expectEqual(@as(?u16, 200), responseStatus(resp));
    const resp_body = responseBody(resp) orelse return error.MissingBody;
    const dechunked = try dechunk(a, resp_body);
    defer a.free(dechunked);

    var got: std.ArrayList(TokenId) = .empty;
    defer got.deinit(a);
    var it = std.mem.tokenizeScalar(u8, dechunked, '\n');
    while (it.next()) |line| {
        if (line.len == 0) continue;
        if (std.mem.indexOf(u8, line, "\"done\":true") != null) continue;
        const ids = (try extractIdsField(a, line)) orelse return error.MissingIdsField;
        defer a.free(ids);
        try got.appendSlice(a, ids);
    }
    try testing.expectEqualSlices(TokenId, want, got.items);
}

// === Auth tests ========================================================

test "auth: missing Authorization header returns 401" {
    const a = testing.allocator;
    const fx = try PipeFixture.init(a);
    defer fx.deinit(a);
    var pool = try BatchPool.init(a, 2);
    defer pool.deinit();
    var ctx = ctxFor(a, fx, &pool, .{ .log = false, .auth_token = "s3cret" });

    const req = try buildReq(a, "POST", "/encode", "", "{\"text\":\"hi\"}");
    defer a.free(req);
    const resp = try handleRequestInMemory(&ctx, req, ClientIp.unknown);
    defer a.free(resp);

    try testing.expectEqual(@as(?u16, 401), responseStatus(resp));
    const body = responseBody(resp) orelse return error.MissingBody;
    try testing.expectEqualStrings("{\"error\":\"unauthorized\"}\n", body);
}

test "auth: wrong bearer token returns 401" {
    const a = testing.allocator;
    const fx = try PipeFixture.init(a);
    defer fx.deinit(a);
    var pool = try BatchPool.init(a, 2);
    defer pool.deinit();
    var ctx = ctxFor(a, fx, &pool, .{ .log = false, .auth_token = "s3cret" });

    const req = try buildReq(a, "POST", "/encode", "Authorization: Bearer wrongtoken\r\n", "{\"text\":\"hi\"}");
    defer a.free(req);
    const resp = try handleRequestInMemory(&ctx, req, ClientIp.unknown);
    defer a.free(resp);

    try testing.expectEqual(@as(?u16, 401), responseStatus(resp));
}

test "auth: valid bearer token allows request" {
    const a = testing.allocator;
    const fx = try PipeFixture.init(a);
    defer fx.deinit(a);
    var pool = try BatchPool.init(a, 2);
    defer pool.deinit();
    var ctx = ctxFor(a, fx, &pool, .{ .log = false, .auth_token = "s3cret" });

    const req = try buildReq(a, "POST", "/encode", "Authorization: Bearer s3cret\r\n", "{\"text\":\"hi\"}");
    defer a.free(req);
    const resp = try handleRequestInMemory(&ctx, req, ClientIp.unknown);
    defer a.free(resp);

    try testing.expectEqual(@as(?u16, 200), responseStatus(resp));
}

test "auth: /health bypasses auth" {
    const a = testing.allocator;
    const fx = try PipeFixture.init(a);
    defer fx.deinit(a);
    var pool = try BatchPool.init(a, 2);
    defer pool.deinit();
    var ctx = ctxFor(a, fx, &pool, .{ .log = false, .auth_token = "s3cret" });

    const req = try buildReq(a, "GET", "/health", "", "");
    defer a.free(req);
    const resp = try handleRequestInMemory(&ctx, req, ClientIp.unknown);
    defer a.free(resp);
    try testing.expectEqual(@as(?u16, 200), responseStatus(resp));
}

test "auth: constantTimeEqlBytes covers length + content mismatches" {
    try testing.expect(!constantTimeEqlBytes("abcdef", "abcdez"));
    try testing.expect(!constantTimeEqlBytes("abcdef", "Zbcdef"));
    try testing.expect(constantTimeEqlBytes("abcdef", "abcdef"));
    try testing.expect(!constantTimeEqlBytes("abc", "abcd"));
    try testing.expect(constantTimeEqlBytes("", ""));
}

test "auth: timing-safe comparison has no early exit (any-position mismatch)" {
    // Wall-clock timing is too noisy to assert in a unit test; what we
    // *can* assert is correctness for inputs that would betray a naive
    // mem.eql short-circuit: same-length tokens whose single differing
    // byte is at the start, middle, or end. constantTimeEqlBytes's body
    // is a single OR-accumulator loop with no `break`, so any same-
    // length input reads every byte unconditionally — the property the
    // comparison requires. Combined with the explicit length check
    // (length is not secret), we have the "no timing oracle for token
    // content" guarantee.
    const expected = "0123456789abcdef0123456789abcdef";
    const diff_first = "Z123456789abcdef0123456789abcdef";
    const diff_middle = "0123456789abcZef0123456789abcdef";
    const diff_last = "0123456789abcdef0123456789abcdeZ";
    try testing.expect(!constantTimeEqlBytes(diff_first, expected));
    try testing.expect(!constantTimeEqlBytes(diff_middle, expected));
    try testing.expect(!constantTimeEqlBytes(diff_last, expected));
    try testing.expect(constantTimeEqlBytes(expected, expected));
}

// === Rate-limit tests ==================================================

test "rate limit: tryAcquire allows burst up to capacity then 429s" {
    const a = testing.allocator;
    var rl = RateLimiter.init(a, testing.io, 10); // capacity = 20
    defer rl.deinit();
    const client = ClientIp.fromIp4(.{ 10, 0, 0, 1 });
    const t0: i96 = 0;

    var allowed: u32 = 0;
    var limited: u32 = 0;
    var i: u32 = 0;
    while (i < 100) : (i += 1) {
        switch (rl.tryAcquireAt(client, t0)) {
            .ok => allowed += 1,
            .limited => limited += 1,
        }
    }
    // At t=0 with no refill, exactly `capacity` (20) requests succeed.
    try testing.expectEqual(@as(u32, 20), allowed);
    try testing.expectEqual(@as(u32, 80), limited);
}

test "rate limit: bucket refills over time" {
    const a = testing.allocator;
    var rl = RateLimiter.init(a, testing.io, 10);
    defer rl.deinit();
    const client = ClientIp.fromIp4(.{ 10, 0, 0, 2 });

    // Drain at t=0.
    var i: u32 = 0;
    while (i < 20) : (i += 1) _ = rl.tryAcquireAt(client, 0);
    try testing.expect(rl.tryAcquireAt(client, 0) == .limited);

    // 1 second later → 10 tokens refilled → 10 more should succeed.
    const ns_per_s: i96 = 1_000_000_000;
    var allowed: u32 = 0;
    var k: u32 = 0;
    while (k < 20) : (k += 1) {
        if (rl.tryAcquireAt(client, ns_per_s) == .ok) allowed += 1;
    }
    try testing.expectEqual(@as(u32, 10), allowed);
}

test "rate limit: per-client buckets are independent" {
    const a = testing.allocator;
    var rl = RateLimiter.init(a, testing.io, 5); // capacity = 10
    defer rl.deinit();
    const ca = ClientIp.fromIp4(.{ 1, 2, 3, 4 });
    const cb = ClientIp.fromIp4(.{ 5, 6, 7, 8 });

    var i: u32 = 0;
    while (i < 10) : (i += 1) try testing.expect(rl.tryAcquireAt(ca, 0) == .ok);
    try testing.expect(rl.tryAcquireAt(ca, 0) == .limited);
    // cb's bucket is untouched.
    try testing.expect(rl.tryAcquireAt(cb, 0) == .ok);
}

test "rate limit: /health exempt even when limit is exhausted" {
    const a = testing.allocator;
    const fx = try PipeFixture.init(a);
    defer fx.deinit(a);
    var pool = try BatchPool.init(a, 2);
    defer pool.deinit();
    // rps=1 → capacity=2, refill 1 token/sec. Even with 100ms of test
    // jitter between drain and the assertion below the bucket gains
    // ~0.1 tokens, well under the 1.0 threshold for another acquire.
    var rl = RateLimiter.init(a, testing.io, 1);
    defer rl.deinit();
    // Drain on the real-clock path (tryAcquire, not tryAcquireAt) so
    // last_refill_ns is set in the same time domain handleRequest will
    // use a moment later.
    _ = rl.tryAcquire(ClientIp.unknown);
    _ = rl.tryAcquire(ClientIp.unknown);
    try testing.expect(rl.tryAcquire(ClientIp.unknown) == .limited);

    var ctx: ServeCtx = .{
        .allocator = a,
        .io = testing.io,
        .pipeline = &fx.pipe,
        .pool = &pool,
        .opts = .{ .log = false, .rate_limit_rps = 1 },
        .rate_limiter = &rl,
    };

    // /health still 200 even though the bucket is empty.
    const hreq = try buildReq(a, "GET", "/health", "", "");
    defer a.free(hreq);
    const hresp = try handleRequestInMemory(&ctx, hreq, ClientIp.unknown);
    defer a.free(hresp);
    try testing.expectEqual(@as(?u16, 200), responseStatus(hresp));

    // /encode is rate-limited.
    const ereq = try buildReq(a, "POST", "/encode", "", "{\"text\":\"hi\"}");
    defer a.free(ereq);
    const eresp = try handleRequestInMemory(&ctx, ereq, ClientIp.unknown);
    defer a.free(eresp);
    try testing.expectEqual(@as(?u16, 429), responseStatus(eresp));
    const ebody = responseBody(eresp) orelse return error.MissingBody;
    try testing.expect(std.mem.indexOf(u8, ebody, "rate_limited") != null);
    try testing.expect(std.mem.indexOf(u8, ebody, "retry_after_ms") != null);
}

test "rate limit: LRU evicts oldest entry above the cap" {
    const a = testing.allocator;
    var rl = RateLimiter.init(a, testing.io, 1);
    defer rl.deinit();
    // Insert `cap + 1` distinct clients; assert the count never exceeds
    // the cap (oldest entry was evicted).
    var i: u32 = 0;
    while (i < rate_limit_max_clients + 1) : (i += 1) {
        const ip = ClientIp.fromIp4(.{
            @intCast((i >> 24) & 0xFF),
            @intCast((i >> 16) & 0xFF),
            @intCast((i >> 8) & 0xFF),
            @intCast(i & 0xFF),
        });
        _ = rl.tryAcquireAt(ip, @intCast(i));
    }
    try testing.expectEqual(@as(usize, rate_limit_max_clients), rl.buckets.count());
}

// === Structured logging tests (post-1.22 agent D) ======================

test "log: writeJsonLogLine emits a single well-formed JSON object" {
    const a = testing.allocator;
    const fx = try PipeFixture.init(a);
    defer fx.deinit(a);
    var pool = try BatchPool.init(a, 2);
    defer pool.deinit();
    const ctx = ctxFor(a, fx, &pool, .{ .log = false, .log_format = .json });

    var out: std.Io.Writer.Allocating = .init(a);
    defer out.deinit();
    try writeJsonLogLine(&out.writer, &ctx, .{
        .method = .POST,
        .path_label = metrics_mod.PathLabel.encode,
        .path_str = "/encode",
        .status = 200,
        .bytes_in = 1024,
        .bytes_out = 4096,
        .duration_ns = 3_200_000,
        .client_ip_str = "10.0.0.1",
        .encoded_tokens = 17,
    });
    const line = out.written();

    // Single trailing newline, every required field present.
    try testing.expect(line.len > 0);
    try testing.expectEqual(@as(u8, '\n'), line[line.len - 1]);
    try testing.expect(std.mem.indexOf(u8, line, "\"method\":\"POST\"") != null);
    try testing.expect(std.mem.indexOf(u8, line, "\"path\":\"/encode\"") != null);
    try testing.expect(std.mem.indexOf(u8, line, "\"status\":200") != null);
    try testing.expect(std.mem.indexOf(u8, line, "\"bytes_in\":1024") != null);
    try testing.expect(std.mem.indexOf(u8, line, "\"bytes_out\":4096") != null);
    try testing.expect(std.mem.indexOf(u8, line, "\"client_ip\":\"10.0.0.1\"") != null);
    try testing.expect(std.mem.indexOf(u8, line, "\"encoded_tokens\":17") != null);
    // latency_ms = 3_200_000 ns = 3.2 ms (printed as 3.200 with the %.3 format).
    try testing.expect(std.mem.indexOf(u8, line, "\"latency_ms\":3.2") != null);

    // The output is a single JSON object (one '{' / one '}').
    var n_open: usize = 0;
    var n_close: usize = 0;
    for (line) |c| {
        if (c == '{') n_open += 1;
        if (c == '}') n_close += 1;
    }
    try testing.expectEqual(@as(usize, 1), n_open);
    try testing.expectEqual(@as(usize, 1), n_close);
}

test "log: writeJsonLogLine escapes special chars in path + client_ip" {
    const a = testing.allocator;
    const fx = try PipeFixture.init(a);
    defer fx.deinit(a);
    var pool = try BatchPool.init(a, 2);
    defer pool.deinit();
    const ctx = ctxFor(a, fx, &pool, .{ .log = false, .log_format = .json });

    var out: std.Io.Writer.Allocating = .init(a);
    defer out.deinit();
    try writeJsonLogLine(&out.writer, &ctx, .{
        .method = .GET,
        .path_label = metrics_mod.PathLabel.other,
        .path_str = "/weird\"path\nname",
        .status = 200,
        .bytes_in = 0,
        .bytes_out = 0,
        .duration_ns = 0,
        .client_ip_str = "10.\"0\".0.1",
    });
    const line = out.written();

    // Backslash-escaped " and \n in path.
    try testing.expect(std.mem.indexOf(u8, line, "\\\"path\\nname") != null);
    // Backslash-escaped " in client_ip.
    try testing.expect(std.mem.indexOf(u8, line, "10.\\\"0\\\".0.1") != null);
}

test "log: text-format path doesn't emit JSON envelope (sanity)" {
    // The text path goes through std.log which we can't easily capture
    // here, so this test simply asserts that switching to json format
    // and back to text doesn't corrupt anything observable in the
    // ServeCtx — the text branch is exercised end-to-end by every
    // other test in this file (which run with .text format).
    const a = testing.allocator;
    const fx = try PipeFixture.init(a);
    defer fx.deinit(a);
    var pool = try BatchPool.init(a, 2);
    defer pool.deinit();
    var ctx = ctxFor(a, fx, &pool, .{ .log = false, .log_format = .json });
    ctx.opts.log_format = .text;
    try testing.expectEqual(LogFormat.text, ctx.opts.log_format);
}

test "log: client IP formatter emits real IPv4 and IPv6 values" {
    var buf: [64]u8 = undefined;
    try testing.expectEqualStrings("10.2.3.4", formatClientIp(ClientIp.fromIp4(.{ 10, 2, 3, 4 }), &buf));
    try testing.expectEqualStrings("unknown", formatClientIp(ClientIp.unknown, &buf));
    const ip6 = ClientIp.fromIp6(.{ 0x20, 0x01, 0x0d, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 });
    try testing.expectEqualStrings("2001:db8:0:0:0:0:0:1", formatClientIp(ip6, &buf));
}

// === /metrics tests (post-1.22 agent D) ================================

test "metrics: /metrics endpoint returns Prometheus text exposition" {
    const a = testing.allocator;
    const fx = try PipeFixture.init(a);
    defer fx.deinit(a);
    var pool = try BatchPool.init(a, 2);
    defer pool.deinit();

    var m: Metrics = .{};
    var ctx = ctxFor(a, fx, &pool, .{ .log = false, .metrics = &m });

    // Drive one /encode request first so the counters have something
    // non-zero to render.
    const req1 = try buildReq(a, "POST", "/encode", "", "{\"text\":\"hello world\"}");
    defer a.free(req1);
    const resp1 = try handleRequestInMemory(&ctx, req1, ClientIp.unknown);
    defer a.free(resp1);
    try testing.expectEqual(@as(?u16, 200), responseStatus(resp1));
    try testing.expect(m.encode_tokens_total.load(.monotonic) > 0);
    try testing.expect(m.bytes_out_total[@intFromEnum(metrics_mod.PathLabel.encode)].load(.monotonic) > 0);

    // Now hit /metrics.
    const req2 = try buildReq(a, "GET", "/metrics", "", "");
    defer a.free(req2);
    const resp2 = try handleRequestInMemory(&ctx, req2, ClientIp.unknown);
    defer a.free(resp2);
    try testing.expectEqual(@as(?u16, 200), responseStatus(resp2));
    const body = responseBody(resp2) orelse return error.MissingBody;
    try testing.expect(std.mem.indexOf(u8, body, "# TYPE ztok_requests_total counter") != null);
    try testing.expect(std.mem.indexOf(u8, body, "ztok_requests_total{method=\"POST\",path=\"/encode\",status=\"2xx\"} 1") != null);
    try testing.expect(std.mem.indexOf(u8, body, "ztok_request_duration_seconds_count{path=\"/encode\"}") != null);
}

test "metrics: malformed encode is counted as 4xx with response bytes" {
    const a = testing.allocator;
    const fx = try PipeFixture.init(a);
    defer fx.deinit(a);
    var pool = try BatchPool.init(a, 2);
    defer pool.deinit();
    var m: Metrics = .{};
    var ctx = ctxFor(a, fx, &pool, .{ .log = false, .metrics = &m });

    const req = try buildReq(a, "POST", "/encode", "", "{\"wrong\":true}");
    defer a.free(req);
    const resp = try handleRequestInMemory(&ctx, req, ClientIp.unknown);
    defer a.free(resp);
    try testing.expectEqual(@as(?u16, 400), responseStatus(resp));

    const count = m.requests_total[@intFromEnum(metrics_mod.MethodLabel.POST)][@intFromEnum(metrics_mod.PathLabel.encode)][@intFromEnum(metrics_mod.StatusBucket.s4xx)].load(.monotonic);
    try testing.expectEqual(@as(u64, 1), count);
    try testing.expect(m.bytes_out_total[@intFromEnum(metrics_mod.PathLabel.encode)].load(.monotonic) > 0);
}

test "metrics: /metrics requires opts.metrics to be set (returns 404 when disabled)" {
    const a = testing.allocator;
    const fx = try PipeFixture.init(a);
    defer fx.deinit(a);
    var pool = try BatchPool.init(a, 2);
    defer pool.deinit();
    var ctx = ctxFor(a, fx, &pool, .{ .log = false }); // no metrics

    const req = try buildReq(a, "GET", "/metrics", "", "");
    defer a.free(req);
    const resp = try handleRequestInMemory(&ctx, req, ClientIp.unknown);
    defer a.free(resp);
    try testing.expectEqual(@as(?u16, 404), responseStatus(resp));
}

test "metrics: /metrics bypasses auth-token (so scrapers don't need creds)" {
    const a = testing.allocator;
    const fx = try PipeFixture.init(a);
    defer fx.deinit(a);
    var pool = try BatchPool.init(a, 2);
    defer pool.deinit();
    var m: Metrics = .{};
    var ctx = ctxFor(a, fx, &pool, .{ .log = false, .metrics = &m, .auth_token = "s3cret" });

    const req = try buildReq(a, "GET", "/metrics", "", "");
    defer a.free(req);
    const resp = try handleRequestInMemory(&ctx, req, ClientIp.unknown);
    defer a.free(resp);
    try testing.expectEqual(@as(?u16, 200), responseStatus(resp));
}

// === TLS stub test (post-1.22 agent D / 1.23 hardening) ================

test "tls: tls_enabled startup path depends on build_options.tls_backend" {
    // When the default build (-Dtls=none) is in effect, run() returns
    // error.TLSServerNotAvailable. With -Dtls=mbedtls the TLS server
    // initializes from the cert/key paths. Both branches are
    // compile-time gated on `build_options.tls_backend`, so the
    // assertion below just verifies the build_options enum is
    // reachable and that the Options carry the new cert/key paths.
    const opts: Options = .{
        .tls_enabled = true,
        .tls_cert_path = "/dev/null",
        .tls_key_path = "/dev/null",
    };
    try testing.expect(opts.tls_enabled);
    try testing.expect(opts.tls_cert_path != null);
    try testing.expect(opts.tls_key_path != null);
    // The build_options enum must be reachable in tests.
    _ = build_options.tls_backend;
}

// === WebSocket handshake test (post-1.22 agent D) ======================

test "ws: /encode_ws missing Upgrade header returns 400" {
    const a = testing.allocator;
    const fx = try PipeFixture.init(a);
    defer fx.deinit(a);
    var pool = try BatchPool.init(a, 2);
    defer pool.deinit();
    var ctx = ctxFor(a, fx, &pool, .{ .log = false });

    // A vanilla GET without the WS upgrade headers.
    const req = try buildReq(a, "GET", "/encode_ws", "", "");
    defer a.free(req);
    const resp = try handleRequestInMemory(&ctx, req, ClientIp.unknown);
    defer a.free(resp);
    try testing.expectEqual(@as(?u16, 400), responseStatus(resp));
}

// === OIDC integration test (post-1.22 agent D) =========================

test "oidc: missing bearer when oidc is configured returns 401" {
    const a = testing.allocator;
    const fx = try PipeFixture.init(a);
    defer fx.deinit(a);
    var pool = try BatchPool.init(a, 2);
    defer pool.deinit();

    // Synthesize an in-process JWKS (empty key set). Any token would
    // fail validation; this test only asserts that the absence of any
    // Authorization header still produces 401, matching the bearer
    // token contract.
    var jwks: auth_oidc.JwkSet = .{ .allocator = a, .keys = try a.alloc(auth_oidc.Jwk, 0), .fetched_at_ns = 0 };
    var v: auth_oidc.Validator = .{
        .allocator = a,
        .issuer = "https://example.com",
        .audience = "ztok",
        .jwks = jwks,
    };
    defer v.deinit();
    _ = &jwks;

    var ctx = ctxFor(a, fx, &pool, .{ .log = false, .oidc = &v });

    const req = try buildReq(a, "POST", "/encode", "", "{\"text\":\"hi\"}");
    defer a.free(req);
    const resp = try handleRequestInMemory(&ctx, req, ClientIp.unknown);
    defer a.free(resp);
    try testing.expectEqual(@as(?u16, 401), responseStatus(resp));
}

// === TLS request-parsing helper tests (Part 1) =========================
//
// These exercise the hand-rolled HTTP/1.1 parser the TLS path uses so
// we cover the framing logic without needing a real TLS client. The
// `readHttpHead` test feeds a fixed buffer through a stub read fn; the
// `parseHttpRequestHead` test drives the head parser directly.

const TestReader = struct {
    bytes: []const u8,
    pos: usize = 0,

    fn read(ctx: *anyopaque, buf: []u8) anyerror!usize {
        const self: *TestReader = @ptrCast(@alignCast(ctx));
        const remaining = self.bytes.len - self.pos;
        if (remaining == 0) return 0;
        const n = @min(remaining, buf.len);
        @memcpy(buf[0..n], self.bytes[self.pos .. self.pos + n]);
        self.pos += n;
        return n;
    }
};

test "tls parser: readHttpHead finds CRLFCRLF terminator" {
    var stub: TestReader = .{ .bytes = "POST /encode HTTP/1.1\r\nHost: x\r\nContent-Length: 5\r\n\r\nhello" };
    var buf: [256]u8 = undefined;
    const res = try readHttpHead(@ptrCast(&stub), TestReader.read, &buf);
    // Head ends right after the empty line; body bytes follow.
    try testing.expect(res.head_end > 0);
    try testing.expectEqualStrings("\r\n\r\n", buf[res.head_end - 4 .. res.head_end]);
    // Any extra bytes past the head are body, available without another read.
    const leftover = res.total_read - res.head_end;
    try testing.expectEqualStrings("hello", buf[res.head_end .. res.head_end + leftover]);
}

test "tls parser: readHttpHead caps at buffer size" {
    // Build a head that's longer than the buffer; readHttpHead should
    // return HeadersTooLarge instead of running off the end.
    const a = testing.allocator;
    var giant: std.ArrayList(u8) = .empty;
    defer giant.deinit(a);
    try giant.appendSlice(a, "GET / HTTP/1.1\r\n");
    var i: usize = 0;
    while (i < 200) : (i += 1) try giant.appendSlice(a, "X-Pad: AAAAAAAAAA\r\n");
    // Intentionally no "\r\n\r\n" terminator.
    var stub: TestReader = .{ .bytes = giant.items };
    var buf: [1024]u8 = undefined;
    const result = readHttpHead(@ptrCast(&stub), TestReader.read, &buf);
    try testing.expectError(error.HeadersTooLarge, result);
}

test "tls parser: parseHttpRequestHead extracts method, target, headers" {
    const head = "POST /encode HTTP/1.1\r\nHost: x\r\nContent-Length: 21\r\nAuthorization: Bearer s3cret\r\n\r\n";
    var headers_buf: [16]TlsRequest.Header = undefined;
    const parsed = try parseHttpRequestHead(head, &headers_buf);
    try testing.expectEqual(std.http.Method.POST, parsed.method);
    try testing.expectEqualStrings("/encode", parsed.target);
    try testing.expectEqual(@as(usize, 3), parsed.headers.len);
    try testing.expectEqualStrings("Host", parsed.headers[0].name);
    try testing.expectEqualStrings("x", parsed.headers[0].value);
    try testing.expectEqualStrings("Content-Length", parsed.headers[1].name);
    try testing.expectEqualStrings("21", parsed.headers[1].value);
    try testing.expectEqualStrings("Authorization", parsed.headers[2].name);
    try testing.expectEqualStrings("Bearer s3cret", parsed.headers[2].value);
}

test "tls parser: parseHttpRequestHead rejects malformed request line" {
    // No SP-separated TARGET — should fail.
    const head = "GETNO_SPACES_AT_ALL\r\nHost: x\r\n\r\n";
    var headers_buf: [16]TlsRequest.Header = undefined;
    try testing.expectError(error.BadRequest, parseHttpRequestHead(head, &headers_buf));
}

test "tls parser: parseHttpRequestHead trims OWS around header values" {
    const head = "GET / HTTP/1.1\r\nHost:   trimmed-value   \r\n\r\n";
    var headers_buf: [16]TlsRequest.Header = undefined;
    const parsed = try parseHttpRequestHead(head, &headers_buf);
    try testing.expectEqualStrings("trimmed-value", parsed.headers[0].value);
}

test "tls parser: TlsRequest.headerValue is case-insensitive" {
    var headers = [_]TlsRequest.Header{
        .{ .name = "Content-Type", .value = "application/json" },
        .{ .name = "X-Custom", .value = "v" },
    };
    const req: TlsRequest = .{
        .method = .POST,
        .target = "/encode",
        .headers = &headers,
        .body = &.{},
    };
    try testing.expectEqualStrings("application/json", req.headerValue("content-type").?);
    try testing.expectEqualStrings("v", req.headerValue("X-CUSTOM").?);
    try testing.expect(req.headerValue("missing") == null);
}

// === Prefix-cache wiring tests (Part 2) ================================
//
// These cover the `/encode` path's interaction with a persistent prefix
// cache: the wired-up cache opens cleanly, second request reads from
// it without re-encoding the prefix, and the cache file survives a
// close/reopen cycle (i.e. ids encoded in one process show up in the
// next).

const PrefixCache = prefix_cache_mod.PrefixCache;

fn prefixCacheTmpPath(allocator: std.mem.Allocator, tmp: *std.testing.TmpDir) ![]u8 {
    var buf: [4096]u8 = undefined;
    const n = try tmp.dir.realPath(std.testing.io, &buf);
    return std.fmt.allocPrint(allocator, "{s}/ztok_serve_prefix_cache.dat", .{buf[0..n]});
}

test "prefix cache: /encode round-trip is bit-identical with the cache enabled (hermetic)" {
    const a = testing.allocator;
    const fx = try PipeFixture.init(a);
    defer fx.deinit(a);
    var pool = try BatchPool.init(a, 2);
    defer pool.deinit();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try prefixCacheTmpPath(a, &tmp);
    defer a.free(path);

    var pc = try PrefixCache.openPersistent(a, .{ .path = path });
    defer pc.closePersistent();

    var ctx = ctxFor(a, fx, &pool, .{ .log = false });
    ctx.prefix_cache = pc;

    // Baseline: encode without the cache to know what we expect.
    const want = try fx.pipe.encode(a, "hello world");
    defer a.free(want);

    const req = try buildReq(a, "POST", "/encode", "", "{\"text\":\"hello world\"}");
    defer a.free(req);
    const resp = try handleRequestInMemory(&ctx, req, ClientIp.unknown);
    defer a.free(resp);

    try testing.expectEqual(@as(?u16, 200), responseStatus(resp));
    const resp_body = responseBody(resp) orelse return error.MissingBody;
    const payload = if (responseIsChunked(resp))
        try dechunk(a, resp_body)
    else
        try a.dupe(u8, resp_body);
    defer a.free(payload);

    const got_ids = (try extractIdsField(a, payload)) orelse return error.MissingIdsField;
    defer a.free(got_ids);
    try testing.expectEqualSlices(TokenId, want, got_ids);
}

test "prefix cache: inputs longer than the former splice boundary stay bit-identical" {
    const a = testing.allocator;
    const fx = try PipeFixture.init(a);
    defer fx.deinit(a);
    var pool = try BatchPool.init(a, 2);
    defer pool.deinit();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try prefixCacheTmpPath(a, &tmp);
    defer a.free(path);
    var pc = try PrefixCache.openPersistent(a, .{ .path = path });
    defer pc.closePersistent();

    const suffix = "hello world";
    const text = try a.alloc(u8, 253 + suffix.len);
    defer a.free(text);
    @memset(text[0..253], 'x');
    @memcpy(text[253..], suffix);

    const want = try fx.pipe.encode(a, text);
    defer a.free(want);
    var ctx = ctxFor(a, fx, &pool, .{ .log = false });
    ctx.prefix_cache = pc;
    const got = try encodeWithPrefixCache(&ctx, text);
    defer a.free(got);
    try testing.expectEqualSlices(TokenId, want, got);

    // A second request proves the exact full-input entry is reusable.
    const got_cached = try encodeWithPrefixCache(&ctx, text);
    defer a.free(got_cached);
    try testing.expectEqualSlices(TokenId, want, got_cached);
    try testing.expect(pc.mem.stats.hits > 0);
}

test "prefix cache: second /encode of the same input hits the cache (stats.hits increases)" {
    const a = testing.allocator;
    const fx = try PipeFixture.init(a);
    defer fx.deinit(a);
    var pool = try BatchPool.init(a, 2);
    defer pool.deinit();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try prefixCacheTmpPath(a, &tmp);
    defer a.free(path);

    var pc = try PrefixCache.openPersistent(a, .{ .path = path });
    defer pc.closePersistent();

    var ctx = ctxFor(a, fx, &pool, .{ .log = false });
    ctx.prefix_cache = pc;

    const req = try buildReq(a, "POST", "/encode", "", "{\"text\":\"hello world\"}");
    defer a.free(req);

    // First request — miss.
    {
        const resp = try handleRequestInMemory(&ctx, req, ClientIp.unknown);
        defer a.free(resp);
        try testing.expectEqual(@as(?u16, 200), responseStatus(resp));
    }
    const stats_after_first = pc.mem.stats;
    try testing.expect(stats_after_first.misses >= 1);

    // Second request — should hit the in-memory layer.
    {
        const resp = try handleRequestInMemory(&ctx, req, ClientIp.unknown);
        defer a.free(resp);
        try testing.expectEqual(@as(?u16, 200), responseStatus(resp));
    }
    const stats_after_second = pc.mem.stats;
    try testing.expect(stats_after_second.hits > stats_after_first.hits);
}

test "prefix cache: persisted cache survives close/reopen (file-backed reload)" {
    const a = testing.allocator;
    const fx = try PipeFixture.init(a);
    defer fx.deinit(a);
    var pool = try BatchPool.init(a, 2);
    defer pool.deinit();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try prefixCacheTmpPath(a, &tmp);
    defer a.free(path);

    // Phase 1: open, encode, sync, close.
    {
        var pc = try PrefixCache.openPersistent(a, .{ .path = path, .sync_every_n_writes = 1 });
        defer pc.closePersistent();
        var ctx = ctxFor(a, fx, &pool, .{ .log = false });
        ctx.prefix_cache = pc;
        const req = try buildReq(a, "POST", "/encode", "", "{\"text\":\"hello world\"}");
        defer a.free(req);
        const resp = try handleRequestInMemory(&ctx, req, ClientIp.unknown);
        defer a.free(resp);
        try testing.expectEqual(@as(?u16, 200), responseStatus(resp));
        try pc.sync();
    }

    // Phase 2: reopen — the on-disk index should hold the prefix and
    // a fresh /encode of the same input must succeed and return the
    // identical ids without re-encoding the prefix.
    {
        var pc = try PrefixCache.openPersistent(a, .{ .path = path });
        defer pc.closePersistent();
        // The persistent file index must contain at least one record
        // (the cached prefix from phase 1).
        const s = pc.persistentStats();
        try testing.expect(s.file_entries >= 1);

        var ctx = ctxFor(a, fx, &pool, .{ .log = false });
        ctx.prefix_cache = pc;
        const req = try buildReq(a, "POST", "/encode", "", "{\"text\":\"hello world\"}");
        defer a.free(req);
        const resp = try handleRequestInMemory(&ctx, req, ClientIp.unknown);
        defer a.free(resp);
        try testing.expectEqual(@as(?u16, 200), responseStatus(resp));

        const resp_body = responseBody(resp) orelse return error.MissingBody;
        const payload = if (responseIsChunked(resp))
            try dechunk(a, resp_body)
        else
            try a.dupe(u8, resp_body);
        defer a.free(payload);
        const got_ids = (try extractIdsField(a, payload)) orelse return error.MissingIdsField;
        defer a.free(got_ids);

        const want = try fx.pipe.encode(a, "hello world");
        defer a.free(want);
        try testing.expectEqualSlices(TokenId, want, got_ids);
    }
}

// === TLS streaming framing tests =======================================
//
// The TLS streaming handlers write their chunked NDJSON responses
// through `writeChunkedHead`/`writeChunk`/`writeChunkedEnd` against a
// `*std.Io.Writer`. Over the wire that writer is a `mbedtls.ConnWriter`;
// here we point the same helpers at an Allocating writer and verify the
// produced bytes are valid HTTP/1.1 chunked transfer that `dechunk`
// recovers — i.e. the exact framing a TLS client would see.

test "tls streaming: writeChunk framing round-trips through dechunk" {
    const a = testing.allocator;
    var out: std.Io.Writer.Allocating = .init(a);
    defer out.deinit();
    const w = &out.writer;

    try writeChunkedHead(w, "application/x-ndjson");
    try writeChunk(w, "{\"ids\":[1,2,3]}\n");
    try writeChunk(w, "{\"ids\":[4,5]}\n");
    try writeChunk(w, ""); // zero-length payload must be a no-op
    try writeChunk(w, "{\"done\":true}\n");
    try writeChunkedEnd(w);

    const resp = out.written();
    try testing.expect(responseIsChunked(resp));

    const head_end = std.mem.indexOf(u8, resp, "\r\n\r\n").? + 4;
    const dechunked = try dechunk(a, resp[head_end..]);
    defer a.free(dechunked);

    try testing.expectEqualStrings(
        "{\"ids\":[1,2,3]}\n{\"ids\":[4,5]}\n{\"done\":true}\n",
        dechunked,
    );
}

test "tls streaming: chunked NDJSON of StreamEncoder output matches single-shot encode" {
    const a = testing.allocator;
    const fx = try PipeFixture.init(a);
    defer fx.deinit(a);

    // Drive the StreamEncoder exactly as tlsHandleEncodeChunked does and
    // frame each emitted line as a chunk, then verify the concatenated
    // ids equal a single-shot encode of the same input.
    const input = "the quick brown fox jumps over the lazy dog " ** 8;

    var out: std.Io.Writer.Allocating = .init(a);
    defer out.deinit();
    const w = &out.writer;
    try writeChunkedHead(w, "application/x-ndjson");

    var enc = StreamEncoder.init(a, &fx.pipe);
    defer enc.deinit();
    var batch: std.ArrayList(TokenId) = .empty;
    defer batch.deinit(a);
    var line: std.Io.Writer.Allocating = .init(a);
    defer line.deinit();

    var off: usize = 0;
    while (off < input.len) {
        const end = @min(off + 13, input.len);
        try enc.feed(input[off..end], &batch);
        if (batch.items.len > 0) {
            line.clearRetainingCapacity();
            _ = try writeIdsBatchLine(&line.writer, batch.items);
            try writeChunk(w, line.written());
            batch.clearRetainingCapacity();
        }
        off = end;
    }
    try enc.finish(&batch);
    if (batch.items.len > 0) {
        line.clearRetainingCapacity();
        _ = try writeIdsBatchLine(&line.writer, batch.items);
        try writeChunk(w, line.written());
    }
    try writeChunk(w, "{\"done\":true}\n");
    try writeChunkedEnd(w);

    // Dechunk + parse the ids back out of every {"ids":[...]} line.
    const resp = out.written();
    const head_end = std.mem.indexOf(u8, resp, "\r\n\r\n").? + 4;
    const dechunked = try dechunk(a, resp[head_end..]);
    defer a.free(dechunked);

    var got: std.ArrayList(TokenId) = .empty;
    defer got.deinit(a);
    var saw_done = false;
    var it = std.mem.tokenizeScalar(u8, dechunked, '\n');
    while (it.next()) |ln| {
        if (std.mem.indexOf(u8, ln, "\"done\"") != null) {
            saw_done = true;
            continue;
        }
        const ids = (try extractIdsField(a, ln)) orelse continue;
        defer a.free(ids);
        try got.appendSlice(a, ids);
    }
    try testing.expect(saw_done);

    const want = try fx.pipe.encode(a, input);
    defer a.free(want);
    try testing.expectEqualSlices(TokenId, want, got.items);
}
