//! Minimal Zig binding layer over MbedTLS for `ztok serve`'s
//! server-side TLS termination. Compiled in only when
//! `-Dtls=mbedtls` is passed to `zig build`; the default build skips
//! this file entirely.
//!
//! Scope:
//!   * Server-side TLS (mbedtls endpoint = IS_SERVER).
//!   * One `Server` per process, holding the config + cert + key + RNG.
//!   * One `Conn` per accepted socket, doing handshake + read/write
//!     through a caller-supplied send/recv pair so the actual socket
//!     I/O can stay on `std.Io.net.Stream` semantics.
//!
//! What's intentionally not here:
//!   * Client mode, mutual TLS, OCSP, session caches — none of which
//!     `ztok serve` needs in its first TLS wave.
//!   * A custom mbedtls config build — we rely on the system
//!     `libmbedtls`'s default features. If a user builds against a
//!     stripped mbedtls (e.g. without TLS 1.2 cipher suites enabled),
//!     handshake will fail at runtime with a clear log line; we
//!     don't try to detect the feature gap at build time.
//!
//! Lifecycle:
//!   var s = try Server.init(allocator, "cert.pem", "key.pem");
//!   defer s.deinit();
//!   while (...) {
//!     const fd = accept(...);
//!     var c = try Conn.init(&s, fd);
//!     defer c.deinit();
//!     try c.handshake();
//!     // read/write via c.read(buf), c.write(buf)
//!   }

const std = @import("std");
const build_options = @import("build_options");

// Compiled away unless the build was configured with `-Dtls=mbedtls`.
// The `if` guards every reference to `c` below; in the `.none` path
// the whole struct is unreachable and the file's only public surface
// is the `available` const.

pub const available: bool = build_options.tls_backend == .mbedtls;

const c = if (available) @cImport({
    @cInclude("mbedtls/ssl.h");
    @cInclude("mbedtls/entropy.h");
    @cInclude("mbedtls/ctr_drbg.h");
    @cInclude("mbedtls/x509_crt.h");
    @cInclude("mbedtls/pk.h");
    @cInclude("mbedtls/error.h");
    @cInclude("mbedtls/net_sockets.h");
}) else struct {};

pub const Error = error{
    MbedTlsNotAvailable,
    MbedTlsConfigInitFailed,
    MbedTlsCertParseFailed,
    MbedTlsKeyParseFailed,
    MbedTlsSeedFailed,
    MbedTlsConfDefaultsFailed,
    MbedTlsOwnCertFailed,
    MbedTlsSetupFailed,
    MbedTlsHandshakeFailed,
    MbedTlsReadFailed,
    MbedTlsWriteFailed,
};

pub const Server = if (!available) struct {
    pub fn init(_: std.mem.Allocator, _: []const u8, _: []const u8) Error!Server {
        return Error.MbedTlsNotAvailable;
    }
    pub fn deinit(_: *Server) void {}
} else struct {
    allocator: std.mem.Allocator,
    conf: c.mbedtls_ssl_config,
    cert: c.mbedtls_x509_crt,
    key: c.mbedtls_pk_context,
    entropy: c.mbedtls_entropy_context,
    ctr_drbg: c.mbedtls_ctr_drbg_context,

    /// Initialize a server-side TLS configuration loaded from PEM
    /// files at `cert_path` and `key_path`. Returned by value; caller
    /// must keep the pointer stable (mbedtls stores it in the conf).
    /// On error, all allocated mbedtls state is released.
    pub fn init(allocator: std.mem.Allocator, cert_path: []const u8, key_path: []const u8) Error!Server {
        var s: Server = undefined;
        s.allocator = allocator;
        c.mbedtls_ssl_config_init(&s.conf);
        c.mbedtls_x509_crt_init(&s.cert);
        c.mbedtls_pk_init(&s.key);
        c.mbedtls_entropy_init(&s.entropy);
        c.mbedtls_ctr_drbg_init(&s.ctr_drbg);
        errdefer {
            c.mbedtls_ctr_drbg_free(&s.ctr_drbg);
            c.mbedtls_entropy_free(&s.entropy);
            c.mbedtls_pk_free(&s.key);
            c.mbedtls_x509_crt_free(&s.cert);
            c.mbedtls_ssl_config_free(&s.conf);
        }

        // Zero-terminate the file paths for the C API. 4096 is the
        // POSIX PATH_MAX upper bound; if a user's cert path is
        // longer the operator has bigger problems than this length
        // check.
        if (cert_path.len >= 4096 or key_path.len >= 4096) return Error.MbedTlsCertParseFailed;
        var cert_buf: [4096]u8 = undefined;
        var key_buf: [4096]u8 = undefined;
        @memcpy(cert_buf[0..cert_path.len], cert_path);
        cert_buf[cert_path.len] = 0;
        @memcpy(key_buf[0..key_path.len], key_path);
        key_buf[key_path.len] = 0;

        // Seed the RNG with a fixed pers-string ("ztok-serve") so log
        // diffs across boots are at least slightly distinguishable
        // from each other in entropy traces.
        const pers = "ztok-serve";
        if (c.mbedtls_ctr_drbg_seed(
            &s.ctr_drbg,
            c.mbedtls_entropy_func,
            &s.entropy,
            @ptrCast(pers.ptr),
            pers.len,
        ) != 0) return Error.MbedTlsSeedFailed;

        if (c.mbedtls_x509_crt_parse_file(&s.cert, @ptrCast(&cert_buf)) != 0)
            return Error.MbedTlsCertParseFailed;

        // Newer mbedtls keyfile parser needs an RNG (added for RSA
        // blinding during key validation). We pass our seeded
        // ctr_drbg.
        if (c.mbedtls_pk_parse_keyfile(
            &s.key,
            @ptrCast(&key_buf),
            null,
            c.mbedtls_ctr_drbg_random,
            &s.ctr_drbg,
        ) != 0) return Error.MbedTlsKeyParseFailed;

        if (c.mbedtls_ssl_config_defaults(
            &s.conf,
            c.MBEDTLS_SSL_IS_SERVER,
            c.MBEDTLS_SSL_TRANSPORT_STREAM,
            c.MBEDTLS_SSL_PRESET_DEFAULT,
        ) != 0) return Error.MbedTlsConfDefaultsFailed;

        c.mbedtls_ssl_conf_rng(&s.conf, c.mbedtls_ctr_drbg_random, &s.ctr_drbg);

        if (c.mbedtls_ssl_conf_own_cert(&s.conf, &s.cert, &s.key) != 0)
            return Error.MbedTlsOwnCertFailed;

        return s;
    }

    pub fn deinit(s: *Server) void {
        c.mbedtls_ctr_drbg_free(&s.ctr_drbg);
        c.mbedtls_entropy_free(&s.entropy);
        c.mbedtls_pk_free(&s.key);
        c.mbedtls_x509_crt_free(&s.cert);
        c.mbedtls_ssl_config_free(&s.conf);
    }
};

/// One accepted-and-handshaken TLS connection. Wraps the socket file
/// descriptor in mbedtls's BIO callbacks.
pub const Conn = if (!available) struct {
    pub fn init(_: *Server, _: std.posix.fd_t) Error!Conn {
        return Error.MbedTlsNotAvailable;
    }
    pub fn deinit(_: *Conn) void {}
    pub fn handshake(_: *Conn) Error!void {
        return Error.MbedTlsNotAvailable;
    }
    pub fn read(_: *Conn, _: []u8) Error!usize {
        return Error.MbedTlsNotAvailable;
    }
    pub fn write(_: *Conn, _: []const u8) Error!usize {
        return Error.MbedTlsNotAvailable;
    }
} else struct {
    ssl: c.mbedtls_ssl_context,
    fd: c_int,

    pub fn init(server: *Server, fd: std.posix.fd_t) Error!Conn {
        var conn: Conn = undefined;
        conn.fd = @intCast(fd);
        c.mbedtls_ssl_init(&conn.ssl);
        errdefer c.mbedtls_ssl_free(&conn.ssl);

        if (c.mbedtls_ssl_setup(&conn.ssl, &server.conf) != 0)
            return Error.MbedTlsSetupFailed;

        // Bind the socket fd via mbedtls's net layer-ish callbacks.
        // We can't take mbedtls_net_send/_recv directly because that
        // requires the mbedtls_net_context wrapping; using raw
        // send/recv via our own trampolines means we don't depend on
        // libmbedtls's net symbol at all.
        c.mbedtls_ssl_set_bio(
            &conn.ssl,
            &conn.fd,
            sendCb,
            recvCb,
            null,
        );

        return conn;
    }

    pub fn deinit(conn: *Conn) void {
        c.mbedtls_ssl_free(&conn.ssl);
    }

    pub fn handshake(conn: *Conn) Error!void {
        var ret = c.mbedtls_ssl_handshake(&conn.ssl);
        while (ret == c.MBEDTLS_ERR_SSL_WANT_READ or ret == c.MBEDTLS_ERR_SSL_WANT_WRITE) {
            ret = c.mbedtls_ssl_handshake(&conn.ssl);
        }
        if (ret != 0) return Error.MbedTlsHandshakeFailed;
    }

    pub fn read(conn: *Conn, buf: []u8) Error!usize {
        const r = c.mbedtls_ssl_read(&conn.ssl, buf.ptr, buf.len);
        if (r < 0) return Error.MbedTlsReadFailed;
        return @intCast(r);
    }

    pub fn write(conn: *Conn, buf: []const u8) Error!usize {
        const r = c.mbedtls_ssl_write(&conn.ssl, buf.ptr, buf.len);
        if (r < 0) return Error.MbedTlsWriteFailed;
        return @intCast(r);
    }

    /// mbedtls BIO send callback. `ctx` is the `*c_int` fd we stashed
    /// via `ssl_set_bio`. Returns bytes written or a negative
    /// MBEDTLS_ERR_NET_* code on failure.
    fn sendCb(ctx: ?*anyopaque, buf: [*c]const u8, len: usize) callconv(.c) c_int {
        const fd_ptr: *c_int = @ptrCast(@alignCast(ctx.?));
        const n = std.c.send(fd_ptr.*, buf, len, 0);
        if (n < 0) return c.MBEDTLS_ERR_NET_SEND_FAILED;
        return @intCast(n);
    }

    fn recvCb(ctx: ?*anyopaque, buf: [*c]u8, len: usize) callconv(.c) c_int {
        const fd_ptr: *c_int = @ptrCast(@alignCast(ctx.?));
        const n = std.c.recv(fd_ptr.*, buf, len, 0);
        if (n < 0) return c.MBEDTLS_ERR_NET_RECV_FAILED;
        if (n == 0) return c.MBEDTLS_ERR_SSL_PEER_CLOSE_NOTIFY;
        return @intCast(n);
    }
};

// === std.Io adapters over a TLS Conn ===================================
//
// The streaming routes (`/encode_stream`, `/encode_chunked`,
// `/encode_ws`) are written against `std.Io.Writer` / `std.Io.Reader`
// (NDJSON line writing, WebSocket frame parsing). To run them over TLS
// we wrap a `Conn` in tiny adapters that route `drain`/`stream` through
// `Conn.write`/`Conn.read`. mbedtls's TLS read/write are synchronous and
// blocking, so the adapters just loop until the whole request is
// satisfied — no partial-write bookkeeping leaks to the caller.
//
// These take a `*Conn` (which exists in both the stub and real
// branches) so the file still compiles with `-Dtls=none`; at runtime the
// stub `Conn.read`/`.write` return `MbedTlsNotAvailable`, which only
// matters on a TLS path that never executes without `-Dtls=mbedtls`.

/// A buffered `std.Io.Writer` whose drained bytes are written to the
/// wrapped TLS `Conn`. Construct via `connWriter`; call `.writer` to get
/// the `*std.Io.Writer` to hand to framing code, and `.writer.flush()`
/// before dropping it to push any buffered tail through TLS.
pub const ConnWriter = struct {
    conn: *Conn,
    interface: std.Io.Writer,

    fn drain(io_w: *std.Io.Writer, data: []const []const u8, splat: usize) std.Io.Writer.Error!usize {
        const self: *ConnWriter = @alignCast(@fieldParentPtr("interface", io_w));
        var written: usize = 0;
        // First flush whatever is buffered in the interface.
        const buffered = io_w.buffered();
        writeAll(self.conn, buffered) catch return error.WriteFailed;
        written += buffered.len;
        if (data.len == 0) return io_w.consume(written);
        // Then the explicit data slices; the last one repeats `splat`
        // times per the Writer drain contract.
        for (data[0 .. data.len - 1]) |bytes| {
            writeAll(self.conn, bytes) catch return error.WriteFailed;
            written += bytes.len;
        }
        const pattern = data[data.len - 1];
        var i: usize = 0;
        while (i < splat) : (i += 1) {
            writeAll(self.conn, pattern) catch return error.WriteFailed;
            written += pattern.len;
        }
        return io_w.consume(written);
    }

    fn writeAll(conn: *Conn, bytes: []const u8) Error!void {
        var off: usize = 0;
        while (off < bytes.len) {
            const n = try conn.write(bytes[off..]);
            if (n == 0) return Error.MbedTlsWriteFailed;
            off += n;
        }
    }
};

/// Build a `ConnWriter` over `conn`, buffering through `buffer`.
pub fn connWriter(conn: *Conn, buffer: []u8) ConnWriter {
    return .{
        .conn = conn,
        .interface = .{
            .buffer = buffer,
            .vtable = &.{ .drain = ConnWriter.drain },
        },
    };
}

/// A `std.Io.Reader` whose fills are sourced from the wrapped TLS
/// `Conn`. Construct via `connReader`; call `.interface` to get the
/// `*std.Io.Reader` to hand to framing code. A `Conn.read` returning 0
/// (TLS peer close-notify) surfaces as `error.EndOfStream`.
pub const ConnReader = struct {
    conn: *Conn,
    interface: std.Io.Reader,

    fn stream(io_r: *std.Io.Reader, w: *std.Io.Writer, limit: std.Io.Limit) std.Io.Reader.StreamError!usize {
        const self: *ConnReader = @alignCast(@fieldParentPtr("interface", io_r));
        // Read directly into the destination writer's unused buffer
        // space, bounded by `limit`. Using the writable region keeps us
        // on the "stream into a fixed Writer" path the default readVec
        // drives.
        const dest = limit.slice(try w.writableSliceGreedy(1));
        const n = self.conn.read(dest) catch return error.ReadFailed;
        if (n == 0) return error.EndOfStream;
        w.advance(n);
        return n;
    }

    /// Optional leading bytes (already read off the socket while parsing
    /// the request head) are prepended by seeding the reader buffer.
    pub fn seed(self: *ConnReader, bytes: []const u8) void {
        std.debug.assert(bytes.len <= self.interface.buffer.len);
        @memcpy(self.interface.buffer[0..bytes.len], bytes);
        self.interface.seek = 0;
        self.interface.end = bytes.len;
    }
};

/// Build a `ConnReader` over `conn`, buffering through `buffer`.
pub fn connReader(conn: *Conn, buffer: []u8) ConnReader {
    return .{
        .conn = conn,
        .interface = .{
            .buffer = buffer,
            .seek = 0,
            .end = 0,
            .vtable = &.{ .stream = ConnReader.stream },
        },
    };
}

// === Tests =============================================================

const testing = std.testing;

test "mbedtls: available flag matches build option" {
    // Trivial: the constant must match what build_options says, so a
    // build broken at the boundary fails here rather than at the
    // first runtime mbedtls call.
    try testing.expectEqual(build_options.tls_backend == .mbedtls, available);
}

// Embedded 2048-bit RSA self-signed cert + key, generated once via
//   openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem \
//     -out cert.pem -days 36500 -subj "/CN=ztok-test" -batch
// The private key is here ONLY for offline-only test use against a
// hard-coded `/CN=ztok-test` self-signed cert; it must never be
// repurposed to sign real traffic.
const test_cert_pem =
    "-----BEGIN CERTIFICATE-----\n" ++
    "MIIDCzCCAfOgAwIBAgIUazCgWIPyre1U1DyZc5zRGOXtxg4wDQYJKoZIhvcNAQEL\n" ++
    "BQAwFDESMBAGA1UEAwwJenRvay10ZXN0MCAXDTI2MDUxOTIyNTQ0NFoYDzIxMjYw\n" ++
    "NDI1MjI1NDQ0WjAUMRIwEAYDVQQDDAl6dG9rLXRlc3QwggEiMA0GCSqGSIb3DQEB\n" ++
    "AQUAA4IBDwAwggEKAoIBAQDyP9Q4C+fnQXpnfA2cV3g94Mzz6Yz0Jq6jADEju63+\n" ++
    "n3Iz5XPWpeKSPg7mjpjrCsogUVnhtk823yRKIbcZuDaG86KLYcCJPL07GgjTpbHF\n" ++
    "HZzkyj+odv2PmNHJwOSdLScS3vUHs+l5ka8hMCqFslN0m6T9/qcHyxEYJ+3UOMrM\n" ++
    "DZ962QnQj1OBbm6samg7bUAVnZ9rsNiikhAZuUYR9brPuLMAPCM4L+qtv4MJNNJY\n" ++
    "ipoOtDxxJguPVaOf/e9gQOAsM1RCg8F5WmLsI7+qYA0lodU0E6oBiEcbLC+GA4T4\n" ++
    "VS4GxYCRG+fWd4poPnZZu9pyqdjdz1eXSbjAMUhynfTXAgMBAAGjUzBRMB0GA1Ud\n" ++
    "DgQWBBSJGaxSUmRmPKyOZeJhUOg+pPN3eTAfBgNVHSMEGDAWgBSJGaxSUmRmPKyO\n" ++
    "ZeJhUOg+pPN3eTAPBgNVHRMBAf8EBTADAQH/MA0GCSqGSIb3DQEBCwUAA4IBAQCT\n" ++
    "E37vlwr78o7Ajv3/czv6icyORbhjNS8LiqtSNyoiA0e5k6R4xK7F5s1ArnIk6JE+\n" ++
    "LM+mI0PLDVNGCnBbRg75lgfJ4vLELnVGAPa4wb8pH237b60XcczNU9WgELU3fr5X\n" ++
    "vQ5YsIbkYjWp+Em/laxp9AhKQ1c5+I+cQ70aIx3T3TcJKDJQjDqrLZE7pzFnmSIT\n" ++
    "j/OaLOIik4ekUDY3z6uY/c7fjHsakGOgBF8T0jginepnyxQ9ncdl9aZpotV10ZJj\n" ++
    "WhYxVGeC8un4REfgeua2Th4Qq0YrrxXztZVyvrmqAqr6q4G+nLTmpX033OX1go6H\n" ++
    "VfclUlKXHljrZB2WqSmm\n" ++
    "-----END CERTIFICATE-----\n";

const test_key_pem =
    "-----BEGIN PRIVATE KEY-----\n" ++
    "MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDyP9Q4C+fnQXpn\n" ++
    "fA2cV3g94Mzz6Yz0Jq6jADEju63+n3Iz5XPWpeKSPg7mjpjrCsogUVnhtk823yRK\n" ++
    "IbcZuDaG86KLYcCJPL07GgjTpbHFHZzkyj+odv2PmNHJwOSdLScS3vUHs+l5ka8h\n" ++
    "MCqFslN0m6T9/qcHyxEYJ+3UOMrMDZ962QnQj1OBbm6samg7bUAVnZ9rsNiikhAZ\n" ++
    "uUYR9brPuLMAPCM4L+qtv4MJNNJYipoOtDxxJguPVaOf/e9gQOAsM1RCg8F5WmLs\n" ++
    "I7+qYA0lodU0E6oBiEcbLC+GA4T4VS4GxYCRG+fWd4poPnZZu9pyqdjdz1eXSbjA\n" ++
    "MUhynfTXAgMBAAECggEARHGlT6qJfIzC/T8PB2etSOpdbeLEWO0e9V3mBF8QA3tg\n" ++
    "RGplZrWaxM/03M5YRTxYrHXfq8abLfkw4yMQfRtPiKSIfdICGKRJIMwzxzyu8+7w\n" ++
    "d7Hu93WbIXm/eD3gOcpamlnVKDZ8VkVDkmBt+zVNoAojvUG4Rprouwb5Crd7ENiR\n" ++
    "xsPAHfi2XbOt/L1SATnMPYR5tJ3Oexdy+qzpvDML8oC5zZ+VL6bjb2RaShjMkxgK\n" ++
    "24LSpa8OuGn5mrqdodbXIFl9go7BxU3T+G89IfIQDGgrVIXhDMSWDsD/uFijGYnG\n" ++
    "VLMLLFDd0Y7sYdzcqj/H4P+/0VPsCAIj2LQOrkAJQQKBgQD7cLXpodF5pDRX7NUV\n" ++
    "1fYWglecHpPq519fqVEl0PMELDsXZvfIYG7s1noVBFBCWTQpycJiB15vkIoL89Lo\n" ++
    "eXSeNNwOngz8mWcDbUK55l5nyN+ogwd1eMlBDqN/Tzu9vDAi0aCa5E5n6NL1eeg7\n" ++
    "7jQb7jcL3iRgTFBk06CjmgmYlwKBgQD2pHNCjn/aUI8ELoyp6UKFsntJUblAlsMI\n" ++
    "j6dkiyHVJp3ExCfCac/6KpYumipki5pLeNRR2EGrTrQ6I+FN6rmGS85lAUOTWBYH\n" ++
    "n71E0jyx3GuTQqevlELNycwksi6SV9F04hpbYydZjn/T1UZUF6whlIaWbakGhAXE\n" ++
    "EJ+ObHfNwQKBgQCEorkjTDwe6bK+6uygvyQ4TXt/nFW05WZXJQ7sXuPCwL5PIv70\n" ++
    "UYJSJvVxXrwjs8Cjho2mfnKfcWSQ14bbIS6WQhYPE+qP2TARC7LWM6J7JuhskOn/\n" ++
    "Wr73NYyjnJ1MAhh2VZReAK8nexbFbRBHhOkyDqA0/3K65abG+SfVBW2ocwKBgC0M\n" ++
    "SEkFwfBb+mMnJWX7Rr0opj/z/0P+xUyRAF/q0Zke3n4L3b8ymFv230tPuSJ4JQxX\n" ++
    "21+/ge8KBvz/hK04i/4tZlsoafFFi3CFCorBY3iQ680PxZTaHYF8tB6XtM3h4E3a\n" ++
    "5jl+2LcQweQs9hVx5WyUtihPiym3f14aMypOQWuBAoGBAONvpUlsH/QMG2W0a1jD\n" ++
    "lStoHPKZ1IrKg/zhPLbwEKVJ1J8cPBKT8ic33YJ9DuZxGPCO4QBLem9ys1eQXuuB\n" ++
    "ioPjo/inEYiEfgN1+mTpSBSk6eRJdd+C9yiegE0v/S8ebJe/rZiy+602I50QPLF8\n" ++
    "Z8NKNSJ6uZp8//vzmEG1ZWk5\n" ++
    "-----END PRIVATE KEY-----\n";

// Hermetic test: cert+key parse + config_defaults + ssl_setup against
// an embedded self-signed cert/key. Gated on `available` so the
// default `zig build test` (no mbedtls) skips it cleanly. With
// `-Dtls=mbedtls` it actually exercises the full server-init path.
// We don't run a real I/O handshake here — that needs two sockets +
// a TLS client, more than a hermetic unit test is meant to do.
test "mbedtls: Server.init parses an embedded self-signed cert + key (only when -Dtls=mbedtls)" {
    if (!available) return error.SkipZigTest;

    // mbedtls_x509_crt_parse_file insists on a path. Write the
    // embedded cert+key to /tmp; delete on exit.
    const cert_path = "/tmp/ztok-mbedtls-test-cert.pem";
    const key_path = "/tmp/ztok-mbedtls-test-key.pem";

    // Zig 0.16: file IO goes through std.Io.Dir; use the threaded
    // single-threaded IO since this test only does a few syscalls.
    const io = std.Io.Threaded.global_single_threaded.io();
    const root = std.Io.Dir.cwd();
    try root.writeFile(io, .{ .sub_path = cert_path, .data = test_cert_pem });
    try root.writeFile(io, .{ .sub_path = key_path, .data = test_key_pem });
    defer root.deleteFile(io, cert_path) catch {};
    defer root.deleteFile(io, key_path) catch {};

    var server = try Server.init(testing.allocator, cert_path, key_path);
    defer server.deinit();

    // Smoke: ssl_setup against a fresh ssl context should succeed —
    // this proves config_defaults + own_cert + rng were all wired
    // correctly during Server.init.
    var ssl_ctx: c.mbedtls_ssl_context = undefined;
    c.mbedtls_ssl_init(&ssl_ctx);
    defer c.mbedtls_ssl_free(&ssl_ctx);
    const rc = c.mbedtls_ssl_setup(&ssl_ctx, &server.conf);
    try testing.expectEqual(@as(c_int, 0), rc);
}
