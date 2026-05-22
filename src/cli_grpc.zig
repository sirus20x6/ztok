//! gRPC-Web serving primitive backing `ztok grpc-serve`.
//!
//! Why gRPC-Web instead of gRPC-over-HTTP/2:
//!   Zig 0.16's stdlib does not ship an HTTP/2 server. gRPC-Web
//!   (https://github.com/grpc/grpc-web) defines a wire format that
//!   carries the same gRPC frames (1-byte flag + 4-byte BE length +
//!   protobuf payload) over plain HTTP/1.1, with the content-type
//!   `application/grpc-web+proto`. The encoding/decoding routes are
//!   identical; the only thing that changes vs HTTP/2 framing is
//!   that responses include a trailing frame (flag = 0x80) carrying
//!   ASCII grpc-status / grpc-message headers, rather than HTTP/2
//!   trailers. Every modern gRPC client (grpc-web, grpc.aio, Envoy
//!   proxy, etc.) speaks this format natively.
//!
//! Routes (POST, content-type: application/grpc-web+proto):
//!   /ztok.Tokenizer/Encode  -> EncodeRequest  → EncodeResponse
//!   /ztok.Tokenizer/Decode  -> DecodeRequest  → DecodeResponse
//!   /ztok.Tokenizer/Eval    -> EvalRequest    → EvalResponse
//!   GET /health             -> {"ok":true}    (HTTP/1.1, JSON)
//!
//! Threading + security: identical to `cli_serve.zig` — same accept
//! loop, same per-request handler. No auth/rate-limit knobs yet; add
//! when the HTTP server gets a real bearer-token wrapper.

const std = @import("std");
const Pipeline = @import("pipeline.zig").Pipeline;
const TokenId = @import("token.zig").TokenId;
const BatchPool = @import("thread_pool.zig").BatchPool;
const proto_min = @import("proto_min.zig");
const eval = @import("eval.zig");

pub const default_host: []const u8 = "127.0.0.1";
pub const default_port: u16 = 7891;
pub const default_max_body_bytes: usize = 16 * 1024 * 1024;

const recv_buf_size: usize = 8192;
const send_buf_size: usize = 16384;

pub const content_type_grpc_web_proto: []const u8 = "application/grpc-web+proto";

pub const ModelKind = enum { bpe, unigram, wordpiece, monster, byte_id };

pub const Options = struct {
    host: []const u8 = default_host,
    port: u16 = default_port,
    max_body_bytes: usize = default_max_body_bytes,
    model_kind: ModelKind = .bpe,
    version: []const u8 = "0.0.0",
    log: bool = true,
};

/// gRPC status codes. Subset — the full table lives in
/// https://grpc.github.io/grpc/core/md_doc_statuscodes.html.
pub const GrpcStatus = enum(u32) {
    ok = 0,
    canceled = 1,
    invalid_argument = 3,
    internal = 13,
    unimplemented = 12,
    resource_exhausted = 8,
};

pub fn run(
    allocator: std.mem.Allocator,
    io: std.Io,
    pipeline: *const Pipeline,
    pool: *BatchPool,
    opts: Options,
) !void {
    var addr = try std.Io.net.IpAddress.parse(opts.host, opts.port);
    var server = addr.listen(io, .{ .reuse_address = true }) catch |err| {
        std.log.err("ztok grpc-serve: failed to listen on {s}:{d}: {t}", .{ opts.host, opts.port, err });
        return err;
    };
    defer server.deinit(io);

    if (opts.log) {
        std.log.info("ztok grpc-serve: listening on http://{s}:{d} (gRPC-Web over HTTP/1.1)", .{ opts.host, opts.port });
    }

    var ctx: ServeCtx = .{
        .allocator = allocator,
        .io = io,
        .pipeline = pipeline,
        .pool = pool,
        .opts = opts,
    };

    serveLoop(&ctx, &server, 0) catch |err| {
        if (err != error.AcceptLoopEnded) return err;
    };
}

fn serveLoop(ctx: *ServeCtx, server: *std.Io.net.Server, max_connections: usize) !void {
    var n_handled: usize = 0;
    while (true) {
        var stream = server.accept(ctx.io) catch |err| switch (err) {
            error.Canceled => return,
            else => |e| {
                if (ctx.opts.log) std.log.err("ztok grpc-serve: accept failed: {t}", .{e});
                continue;
            },
        };
        handleConnection(ctx, stream) catch |err| {
            if (ctx.opts.log) std.log.warn("ztok grpc-serve: connection error: {t}", .{err});
        };
        stream.close(ctx.io);
        n_handled += 1;
        if (max_connections != 0 and n_handled >= max_connections) return error.AcceptLoopEnded;
    }
}

const ServeCtx = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    pipeline: *const Pipeline,
    pool: *BatchPool,
    opts: Options,
};

fn handleConnection(ctx: *ServeCtx, stream: std.Io.net.Stream) !void {
    var recv_buf: [recv_buf_size]u8 = undefined;
    var send_buf: [send_buf_size]u8 = undefined;
    var sr = stream.reader(ctx.io, &recv_buf);
    var sw = stream.writer(ctx.io, &send_buf);
    var http: std.http.Server = .init(&sr.interface, &sw.interface);

    while (true) {
        var req = http.receiveHead() catch |err| switch (err) {
            error.HttpConnectionClosing => return,
            else => return,
        };
        const keep = req.head.keep_alive;
        handleRequest(ctx, &req) catch |err| {
            if (ctx.opts.log) std.log.warn("ztok grpc-serve: handler error: {t}", .{err});
            req.respond("internal error\n", .{
                .status = .internal_server_error,
                .keep_alive = false,
                .extra_headers = &.{.{ .name = "content-type", .value = "text/plain" }},
            }) catch {};
            return;
        };
        if (!keep) return;
    }
}

fn handleRequest(ctx: *ServeCtx, req: *std.http.Server.Request) !void {
    const method = req.head.method;
    const target = req.head.target;
    if (ctx.opts.log) std.log.info("ztok grpc-serve: {t} {s}", .{ method, target });

    const path = if (std.mem.indexOfScalar(u8, target, '?')) |q| target[0..q] else target;

    if (method == .GET and std.mem.eql(u8, path, "/health")) {
        return req.respond("{\"ok\":true}\n", .{
            .status = .ok,
            .extra_headers = &.{.{ .name = "content-type", .value = "application/json" }},
        });
    }

    if (method == .POST and std.mem.eql(u8, path, "/ztok.Tokenizer/Encode")) {
        return handleEncode(ctx, req);
    }
    if (method == .POST and std.mem.eql(u8, path, "/ztok.Tokenizer/Decode")) {
        return handleDecode(ctx, req);
    }
    if (method == .POST and std.mem.eql(u8, path, "/ztok.Tokenizer/Eval")) {
        return handleEval(ctx, req);
    }

    try req.respond("not found\n", .{
        .status = .not_found,
        .extra_headers = &.{.{ .name = "content-type", .value = "text/plain" }},
    });
}

// === Body reading ======================================================

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

    var list: std.ArrayList(u8) = .empty;
    errdefer list.deinit(ctx.allocator);
    var tmp: [4096]u8 = undefined;
    while (true) {
        const n = try body_reader.readSliceShort(&tmp);
        if (n == 0) break;
        if (list.items.len + n > ctx.opts.max_body_bytes) return error.BodyTooLarge;
        try list.appendSlice(ctx.allocator, tmp[0..n]);
        if (n < tmp.len) break;
    }
    return try list.toOwnedSlice(ctx.allocator);
}

// === Response framing ==================================================

/// Send a successful gRPC-Web response: one data frame containing
/// `proto_payload`, followed by a trailers frame (`grpc-status: 0`).
fn respondGrpcOk(req: *std.http.Server.Request, allocator: std.mem.Allocator, proto_payload: []const u8) !void {
    // Build the full body (data frame + trailers frame) in a single
    // allocation so we can respond with a known content-length. The
    // `respondStreaming` path also works but content-length keeps the
    // wire smaller and is what grpc-web clients prefer for short
    // responses.
    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(allocator);
    try proto_min.writeFrame(&body, allocator, .data, proto_payload);
    const trailers = try proto_min.buildTrailerPayload(allocator, @intFromEnum(GrpcStatus.ok), "");
    defer allocator.free(trailers);
    try proto_min.writeFrame(&body, allocator, .trailers, trailers);

    try req.respond(body.items, .{
        .status = .ok,
        .extra_headers = &.{
            .{ .name = "content-type", .value = content_type_grpc_web_proto },
            .{ .name = "grpc-encoding", .value = "identity" },
            .{ .name = "grpc-accept-encoding", .value = "identity" },
        },
    });
}

/// Send a gRPC-Web error: HTTP 200 with no data frame, only a trailers
/// frame carrying the non-OK status. (gRPC errors are conveyed in
/// trailers, not the HTTP status code, so any successful HTTP response
/// is correct here.)
fn respondGrpcError(req: *std.http.Server.Request, allocator: std.mem.Allocator, status: GrpcStatus, message: []const u8) !void {
    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(allocator);
    const trailers = try proto_min.buildTrailerPayload(allocator, @intFromEnum(status), message);
    defer allocator.free(trailers);
    try proto_min.writeFrame(&body, allocator, .trailers, trailers);

    try req.respond(body.items, .{
        .status = .ok,
        .extra_headers = &.{
            .{ .name = "content-type", .value = content_type_grpc_web_proto },
        },
    });
}

// === /ztok.Tokenizer/Encode ===========================================

fn handleEncode(ctx: *ServeCtx, req: *std.http.Server.Request) !void {
    const body_opt = readBody(ctx, req) catch |err| switch (err) {
        error.BodyTooLarge => return respondGrpcError(req, ctx.allocator, .resource_exhausted, "body too large"),
        else => return err,
    };
    const body = body_opt orelse return respondGrpcError(req, ctx.allocator, .invalid_argument, "missing body");
    defer ctx.allocator.free(body);

    const frame = proto_min.readFrame(body) catch {
        return respondGrpcError(req, ctx.allocator, .invalid_argument, "malformed grpc-web frame");
    };
    if (frame.flag & 0x80 != 0) {
        // First-byte high bit set → trailers frame, not data. We don't
        // accept compressed (0x01) bodies either — fail clean.
        return respondGrpcError(req, ctx.allocator, .invalid_argument, "expected data frame");
    }

    const decoded_req = proto_min.EncodeRequest.decode(frame.payload) catch {
        return respondGrpcError(req, ctx.allocator, .invalid_argument, "malformed EncodeRequest");
    };

    // `text` and `raw` both encode bytes-to-ids; if both are set, we
    // prefer `text` (the named string field). When only `raw` is set we
    // treat its bytes as the input directly.
    const input: []const u8 = if (decoded_req.text.len > 0) decoded_req.text else decoded_req.raw;

    const min_for_chunked: usize = 64 * 1024;
    const n_workers = ctx.pool.workerCount();
    const ids = if (input.len >= min_for_chunked and n_workers > 1)
        try ctx.pipeline.encodeChunked(ctx.allocator, ctx.pool, input, n_workers)
    else
        try ctx.pipeline.encode(ctx.allocator, input);
    defer ctx.allocator.free(ids);

    var resp_bytes: std.ArrayList(u8) = .empty;
    defer resp_bytes.deinit(ctx.allocator);
    const resp: proto_min.EncodeResponse = .{ .ids = ids };
    try resp.encode(&resp_bytes, ctx.allocator);

    try respondGrpcOk(req, ctx.allocator, resp_bytes.items);
}

// === /ztok.Tokenizer/Decode ===========================================

fn handleDecode(ctx: *ServeCtx, req: *std.http.Server.Request) !void {
    const body_opt = readBody(ctx, req) catch |err| switch (err) {
        error.BodyTooLarge => return respondGrpcError(req, ctx.allocator, .resource_exhausted, "body too large"),
        else => return err,
    };
    const body = body_opt orelse return respondGrpcError(req, ctx.allocator, .invalid_argument, "missing body");
    defer ctx.allocator.free(body);

    const frame = proto_min.readFrame(body) catch {
        return respondGrpcError(req, ctx.allocator, .invalid_argument, "malformed grpc-web frame");
    };
    if (frame.flag & 0x80 != 0) {
        return respondGrpcError(req, ctx.allocator, .invalid_argument, "expected data frame");
    }

    const decoded_req = proto_min.DecodeRequest.decode(ctx.allocator, frame.payload) catch {
        return respondGrpcError(req, ctx.allocator, .invalid_argument, "malformed DecodeRequest");
    };
    defer ctx.allocator.free(decoded_req.ids);

    const text = try ctx.pipeline.decode(ctx.allocator, decoded_req.ids);
    defer ctx.allocator.free(text);

    var resp_bytes: std.ArrayList(u8) = .empty;
    defer resp_bytes.deinit(ctx.allocator);
    const resp: proto_min.DecodeResponse = .{ .text = text };
    try resp.encode(&resp_bytes, ctx.allocator);

    try respondGrpcOk(req, ctx.allocator, resp_bytes.items);
}

// === /ztok.Tokenizer/Eval =============================================

fn handleEval(ctx: *ServeCtx, req: *std.http.Server.Request) !void {
    const body_opt = readBody(ctx, req) catch |err| switch (err) {
        error.BodyTooLarge => return respondGrpcError(req, ctx.allocator, .resource_exhausted, "body too large"),
        else => return err,
    };
    const body = body_opt orelse return respondGrpcError(req, ctx.allocator, .invalid_argument, "missing body");
    defer ctx.allocator.free(body);

    const frame = proto_min.readFrame(body) catch {
        return respondGrpcError(req, ctx.allocator, .invalid_argument, "malformed grpc-web frame");
    };
    if (frame.flag & 0x80 != 0) {
        return respondGrpcError(req, ctx.allocator, .invalid_argument, "expected data frame");
    }

    const decoded_req = proto_min.EvalRequest.decode(frame.payload) catch {
        return respondGrpcError(req, ctx.allocator, .invalid_argument, "malformed EvalRequest");
    };

    // max_lines == 0 means "no cap". When set, we trim the input at the
    // start of the (max_lines+1)-th line so eval only sees that many.
    const input = if (decoded_req.max_lines == 0)
        decoded_req.text
    else
        truncateToMaxLines(decoded_req.text, decoded_req.max_lines);

    var report = try eval.evaluate(ctx.allocator, ctx.pipeline, input, .{});
    defer report.deinit();

    // Use codepoints as the fertility denominator — this is the
    // standard "tokens per codepoint" fertility measure that
    // eval.evaluate already uses for its per-script metrics. Falls back
    // to bytes when the corpus has no codepoints (e.g. empty input).
    const fertility: f64 = blk: {
        if (report.corpus_codepoints == 0) break :blk 0.0;
        const tt: f64 = @floatFromInt(report.total_tokens);
        const cp: f64 = @floatFromInt(report.corpus_codepoints);
        break :blk tt / cp;
    };

    const eval_resp: proto_min.EvalResponse = .{
        .tokens = report.total_tokens,
        .bytes = report.corpus_bytes,
        .fertility = fertility,
    };
    var resp_bytes: std.ArrayList(u8) = .empty;
    defer resp_bytes.deinit(ctx.allocator);
    try eval_resp.encode(&resp_bytes, ctx.allocator);

    try respondGrpcOk(req, ctx.allocator, resp_bytes.items);
}

fn truncateToMaxLines(text: []const u8, max_lines: u64) []const u8 {
    if (max_lines == 0) return text;
    var count: u64 = 0;
    var i: usize = 0;
    while (i < text.len) : (i += 1) {
        if (text[i] == '\n') {
            count += 1;
            if (count >= max_lines) return text[0 .. i + 1];
        }
    }
    return text;
}

// === In-memory request driver (hermetic tests) =========================

/// Drive one HTTP request through `handleRequest` end-to-end without
/// touching the network. Caller owns the returned response bytes.
pub fn handleRequestInMemory(
    ctx: *ServeCtx,
    raw_request: []const u8,
) ![]u8 {
    var in: std.Io.Reader = .fixed(raw_request);
    var out: std.Io.Writer.Allocating = .init(ctx.allocator);
    errdefer out.deinit();
    var http: std.http.Server = .init(&in, &out.writer);
    var req = try http.receiveHead();
    try handleRequest(ctx, &req);
    return out.toOwnedSlice();
}

// === Tests =============================================================

const testing = std.testing;
const Vocab = @import("vocab.zig").Vocab;
const Bpe = @import("bpe.zig").Bpe;

/// Build the smallest possible BPE+cl100k pipeline reusable across
/// gRPC tests. Mirrors the fixture in cli_serve.zig.
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

fn ctxFor(a: std.mem.Allocator, fx: *PipeFixture, pool: *BatchPool, opts: Options) ServeCtx {
    return .{
        .allocator = a,
        .io = testing.io,
        .pipeline = &fx.pipe,
        .pool = pool,
        .opts = opts,
    };
}

/// Build a raw HTTP/1.1 POST request carrying a gRPC-Web framed
/// protobuf body. Caller owns the returned slice.
fn buildGrpcReq(
    a: std.mem.Allocator,
    path: []const u8,
    proto_payload: []const u8,
) ![]u8 {
    var framed: std.ArrayList(u8) = .empty;
    defer framed.deinit(a);
    try proto_min.writeFrame(&framed, a, .data, proto_payload);

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.print(a, "POST {s} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n", .{path});
    try out.print(
        a,
        "Content-Type: {s}\r\nContent-Length: {d}\r\n\r\n",
        .{ content_type_grpc_web_proto, framed.items.len },
    );
    try out.appendSlice(a, framed.items);
    return out.toOwnedSlice(a);
}

fn responseBody(resp: []const u8) ?[]const u8 {
    const idx = std.mem.indexOf(u8, resp, "\r\n\r\n") orelse return null;
    return resp[idx + 4 ..];
}

fn responseStatus(resp: []const u8) ?u16 {
    if (resp.len < 12) return null;
    if (!std.mem.startsWith(u8, resp, "HTTP/1.1 ")) return null;
    return std.fmt.parseInt(u16, resp[9..12], 10) catch null;
}

fn responseHasHeader(resp: []const u8, name: []const u8, value: []const u8) bool {
    const head_end = std.mem.indexOf(u8, resp, "\r\n\r\n") orelse return false;
    const head = resp[0..head_end];
    var it = std.mem.splitSequence(u8, head, "\r\n");
    while (it.next()) |line| {
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
        const k = std.mem.trim(u8, line[0..colon], " ");
        const v = std.mem.trim(u8, line[colon + 1 ..], " ");
        if (std.ascii.eqlIgnoreCase(k, name) and std.mem.eql(u8, v, value)) return true;
    }
    return false;
}

/// Walk all gRPC-Web frames in `body` and return the first data
/// frame's payload + the trailer payload as text. Caller borrows.
const SplitFrames = struct {
    data: ?[]const u8 = null,
    trailers: ?[]const u8 = null,
};

fn splitFrames(body: []const u8) !SplitFrames {
    var out: SplitFrames = .{};
    var rest = body;
    while (rest.len > 0) {
        const fr = try proto_min.readFrame(rest);
        if (fr.flag & 0x80 != 0) {
            out.trailers = fr.payload;
        } else if (out.data == null) {
            out.data = fr.payload;
        }
        if (rest.len < fr.consumed) break;
        rest = rest[fr.consumed..];
    }
    return out;
}

test "grpc /health returns 200 ok (hermetic)" {
    const a = testing.allocator;
    const fx = try PipeFixture.init(a);
    defer fx.deinit(a);
    var pool = try BatchPool.init(a, 2);
    defer pool.deinit();
    var ctx = ctxFor(a, fx, &pool, .{ .log = false });

    const req = try a.dupe(u8, "GET /health HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n");
    defer a.free(req);
    const resp = try handleRequestInMemory(&ctx, req);
    defer a.free(resp);

    try testing.expectEqual(@as(?u16, 200), responseStatus(resp));
    const body = responseBody(resp) orelse return error.MissingBody;
    try testing.expectEqualStrings("{\"ok\":true}\n", body);
}

test "grpc /Encode round-trips against pipeline (hermetic)" {
    const a = testing.allocator;
    const fx = try PipeFixture.init(a);
    defer fx.deinit(a);
    var pool = try BatchPool.init(a, 2);
    defer pool.deinit();
    var ctx = ctxFor(a, fx, &pool, .{ .log = false });

    const want = try fx.pipe.encode(a, "hello world");
    defer a.free(want);

    // Build the EncodeRequest protobuf body.
    var preq_bytes: std.ArrayList(u8) = .empty;
    defer preq_bytes.deinit(a);
    const preq: proto_min.EncodeRequest = .{ .text = "hello world" };
    try preq.encode(&preq_bytes, a);

    const raw = try buildGrpcReq(a, "/ztok.Tokenizer/Encode", preq_bytes.items);
    defer a.free(raw);
    const resp = try handleRequestInMemory(&ctx, raw);
    defer a.free(resp);

    try testing.expectEqual(@as(?u16, 200), responseStatus(resp));
    try testing.expect(responseHasHeader(resp, "content-type", content_type_grpc_web_proto));

    const body = responseBody(resp) orelse return error.MissingBody;
    const frames = try splitFrames(body);
    try testing.expect(frames.data != null);
    try testing.expect(frames.trailers != null);

    // Trailer must report grpc-status: 0.
    try testing.expect(std.mem.indexOf(u8, frames.trailers.?, "grpc-status: 0") != null);

    const got = try proto_min.EncodeResponse.decode(a, frames.data.?);
    defer a.free(got.ids);
    try testing.expectEqualSlices(TokenId, want, got.ids);
}

test "grpc /Decode round-trips against pipeline (hermetic)" {
    const a = testing.allocator;
    const fx = try PipeFixture.init(a);
    defer fx.deinit(a);
    var pool = try BatchPool.init(a, 2);
    defer pool.deinit();
    var ctx = ctxFor(a, fx, &pool, .{ .log = false });

    // First encode "hi" to get a real id sequence.
    const ids = try fx.pipe.encode(a, "hi");
    defer a.free(ids);
    const want_text = try fx.pipe.decode(a, ids);
    defer a.free(want_text);

    var preq_bytes: std.ArrayList(u8) = .empty;
    defer preq_bytes.deinit(a);
    const preq: proto_min.DecodeRequest = .{ .ids = ids };
    try preq.encode(&preq_bytes, a);

    const raw = try buildGrpcReq(a, "/ztok.Tokenizer/Decode", preq_bytes.items);
    defer a.free(raw);
    const resp = try handleRequestInMemory(&ctx, raw);
    defer a.free(resp);

    try testing.expectEqual(@as(?u16, 200), responseStatus(resp));
    const body = responseBody(resp) orelse return error.MissingBody;
    const frames = try splitFrames(body);
    try testing.expect(frames.data != null);
    const got = try proto_min.DecodeResponse.decode(frames.data.?);
    try testing.expectEqualStrings(want_text, got.text);
}

test "grpc /Eval reports tokens + bytes + fertility (hermetic)" {
    const a = testing.allocator;
    const fx = try PipeFixture.init(a);
    defer fx.deinit(a);
    var pool = try BatchPool.init(a, 2);
    defer pool.deinit();
    var ctx = ctxFor(a, fx, &pool, .{ .log = false });

    const text = "hello world\nhello world\n";

    var preq_bytes: std.ArrayList(u8) = .empty;
    defer preq_bytes.deinit(a);
    const preq: proto_min.EvalRequest = .{ .text = text };
    try preq.encode(&preq_bytes, a);

    const raw = try buildGrpcReq(a, "/ztok.Tokenizer/Eval", preq_bytes.items);
    defer a.free(raw);
    const resp = try handleRequestInMemory(&ctx, raw);
    defer a.free(resp);

    try testing.expectEqual(@as(?u16, 200), responseStatus(resp));
    const body = responseBody(resp) orelse return error.MissingBody;
    const frames = try splitFrames(body);
    try testing.expect(frames.data != null);
    const got = try proto_min.EvalResponse.decode(frames.data.?);
    try testing.expect(got.tokens > 0);
    try testing.expectEqual(@as(u64, text.len), got.bytes);
    try testing.expect(got.fertility > 0.0);
}

test "grpc /Eval honors max_lines truncation (hermetic)" {
    const a = testing.allocator;
    const fx = try PipeFixture.init(a);
    defer fx.deinit(a);
    var pool = try BatchPool.init(a, 2);
    defer pool.deinit();
    var ctx = ctxFor(a, fx, &pool, .{ .log = false });

    // Eight identical lines; max_lines=1 should cap us at exactly the
    // first line's bytes ("hello world\n" = 12 bytes).
    const text = "hello world\nhello world\nhello world\nhello world\nhello world\nhello world\nhello world\nhello world\n";

    var preq_bytes: std.ArrayList(u8) = .empty;
    defer preq_bytes.deinit(a);
    const preq: proto_min.EvalRequest = .{ .text = text, .max_lines = 1 };
    try preq.encode(&preq_bytes, a);

    const raw = try buildGrpcReq(a, "/ztok.Tokenizer/Eval", preq_bytes.items);
    defer a.free(raw);
    const resp = try handleRequestInMemory(&ctx, raw);
    defer a.free(resp);

    const body = responseBody(resp) orelse return error.MissingBody;
    const frames = try splitFrames(body);
    try testing.expect(frames.data != null);
    const got = try proto_min.EvalResponse.decode(frames.data.?);
    try testing.expectEqual(@as(u64, 12), got.bytes);
}

test "grpc unknown route returns 404 (hermetic)" {
    const a = testing.allocator;
    const fx = try PipeFixture.init(a);
    defer fx.deinit(a);
    var pool = try BatchPool.init(a, 2);
    defer pool.deinit();
    var ctx = ctxFor(a, fx, &pool, .{ .log = false });

    var preq_bytes: std.ArrayList(u8) = .empty;
    defer preq_bytes.deinit(a);
    const preq: proto_min.EncodeRequest = .{ .text = "x" };
    try preq.encode(&preq_bytes, a);

    const raw = try buildGrpcReq(a, "/ztok.Tokenizer/Bogus", preq_bytes.items);
    defer a.free(raw);
    const resp = try handleRequestInMemory(&ctx, raw);
    defer a.free(resp);

    try testing.expectEqual(@as(?u16, 404), responseStatus(resp));
}

test "grpc /Encode with malformed frame returns invalid_argument trailer (hermetic)" {
    const a = testing.allocator;
    const fx = try PipeFixture.init(a);
    defer fx.deinit(a);
    var pool = try BatchPool.init(a, 2);
    defer pool.deinit();
    var ctx = ctxFor(a, fx, &pool, .{ .log = false });

    // Build a request with content-type grpc-web+proto but a payload
    // that's too short to be a frame.
    var raw: std.ArrayList(u8) = .empty;
    defer raw.deinit(a);
    const body = [_]u8{ 0x00, 0x00 }; // 2 bytes — frames need 5 minimum
    try raw.print(
        a,
        "POST /ztok.Tokenizer/Encode HTTP/1.1\r\nHost: x\r\nConnection: close\r\nContent-Type: {s}\r\nContent-Length: {d}\r\n\r\n",
        .{ content_type_grpc_web_proto, body.len },
    );
    try raw.appendSlice(a, &body);

    const resp = try handleRequestInMemory(&ctx, raw.items);
    defer a.free(resp);
    try testing.expectEqual(@as(?u16, 200), responseStatus(resp));
    const resp_body = responseBody(resp) orelse return error.MissingBody;
    const frames = try splitFrames(resp_body);
    try testing.expect(frames.data == null);
    try testing.expect(frames.trailers != null);
    try testing.expect(std.mem.indexOf(u8, frames.trailers.?, "grpc-status: 3") != null);
}

test "grpc /Encode with empty body returns invalid_argument trailer (hermetic)" {
    const a = testing.allocator;
    const fx = try PipeFixture.init(a);
    defer fx.deinit(a);
    var pool = try BatchPool.init(a, 2);
    defer pool.deinit();
    var ctx = ctxFor(a, fx, &pool, .{ .log = false });

    var raw: std.ArrayList(u8) = .empty;
    defer raw.deinit(a);
    try raw.print(
        a,
        "POST /ztok.Tokenizer/Encode HTTP/1.1\r\nHost: x\r\nConnection: close\r\nContent-Type: {s}\r\n\r\n",
        .{content_type_grpc_web_proto},
    );

    const resp = try handleRequestInMemory(&ctx, raw.items);
    defer a.free(resp);
    const resp_body = responseBody(resp) orelse return error.MissingBody;
    const frames = try splitFrames(resp_body);
    try testing.expect(frames.trailers != null);
    try testing.expect(std.mem.indexOf(u8, frames.trailers.?, "grpc-status: 3") != null);
}

test "grpc /Encode round-trips when text is sent in `raw` field (hermetic)" {
    const a = testing.allocator;
    const fx = try PipeFixture.init(a);
    defer fx.deinit(a);
    var pool = try BatchPool.init(a, 2);
    defer pool.deinit();
    var ctx = ctxFor(a, fx, &pool, .{ .log = false });

    const want = try fx.pipe.encode(a, "hello");
    defer a.free(want);

    var preq_bytes: std.ArrayList(u8) = .empty;
    defer preq_bytes.deinit(a);
    const preq: proto_min.EncodeRequest = .{ .raw = "hello" };
    try preq.encode(&preq_bytes, a);

    const raw = try buildGrpcReq(a, "/ztok.Tokenizer/Encode", preq_bytes.items);
    defer a.free(raw);
    const resp = try handleRequestInMemory(&ctx, raw);
    defer a.free(resp);

    const body = responseBody(resp) orelse return error.MissingBody;
    const frames = try splitFrames(body);
    try testing.expect(frames.data != null);
    const got = try proto_min.EncodeResponse.decode(a, frames.data.?);
    defer a.free(got.ids);
    try testing.expectEqualSlices(TokenId, want, got.ids);
}
