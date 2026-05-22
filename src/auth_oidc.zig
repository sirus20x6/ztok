//! OAuth2 / OIDC bearer-JWT validation for `ztok serve`.
//!
//! Supported algorithms:
//!   * HS256 (HMAC-SHA256) — JWK kty="oct", k=base64url(secret).
//!   * RS256 (RSASSA-PKCS1-v1_5 / SHA-256) — JWK kty="RSA",
//!     n=base64url(modulus), e=base64url(exponent). The verifier
//!     reuses `std.crypto.Certificate.rsa.PublicKey` +
//!     `PKCS1v1_5Signature.verify` from stdlib (no extra deps).
//!
//! Discovery + JWKS:
//!   * `Validator.initFromDiscovery(allocator, http_fetch, issuer, audience)`
//!     GETs `<issuer>/.well-known/openid-configuration`, parses out
//!     `jwks_uri`, GETs that, and builds the validator. The HTTP
//!     dependency is injected as a `HttpFetchFn` so tests can stub
//!     the network (hermetic) and so the production path can plumb
//!     `std.http.Client` without dragging `std.Io` into every call
//!     site.
//!   * JWKS TTL is 10 minutes by default. The TTL is checked at
//!     validate time so a refresh only happens on the next request
//!     after expiry, not on a background timer. A `kid` miss also
//!     triggers a refresh (a rotated key would otherwise yield
//!     UnknownKid until TTL expiry).
//!
//! The validator is hermetic: tests inject a JWKS via `parseJwks` /
//! the `HttpFetchFn` stub and never touch the network. The CLI uses
//! `initFromDiscovery(..., realHttpFetch, ...)` to plumb the real
//! `std.http.Client`.

const std = @import("std");
const HmacSha256 = std.crypto.auth.hmac.sha2.HmacSha256;
const Sha256 = std.crypto.hash.sha2.Sha256;
const cert_rsa = std.crypto.Certificate.rsa;

/// One entry from a JWKS. Fields are populated based on `kty`:
///   * "oct" / HS256 → `k` holds the base64url-decoded secret.
///   * "RSA" / RS256 → `n` (modulus) and `e` (exponent) hold the
///     base64url-decoded big-endian byte strings.
pub const Jwk = struct {
    kid: ?[]const u8,
    alg: []const u8,
    kty: []const u8,
    /// HS256 key material (base64url-decoded `k`).
    k: ?[]const u8 = null,
    /// RS256 RSA modulus (base64url-decoded big-endian bytes).
    n: ?[]const u8 = null,
    /// RS256 RSA public exponent (base64url-decoded big-endian bytes).
    e: ?[]const u8 = null,
};

pub const JwkSet = struct {
    allocator: std.mem.Allocator,
    keys: []Jwk,
    /// Monotonic-clock nanoseconds at which the keys were last fetched.
    fetched_at_ns: i128,

    pub fn deinit(self: *JwkSet) void {
        for (self.keys) |k| {
            if (k.kid) |s| self.allocator.free(s);
            self.allocator.free(k.alg);
            self.allocator.free(k.kty);
            if (k.k) |s| self.allocator.free(s);
            if (k.n) |s| self.allocator.free(s);
            if (k.e) |s| self.allocator.free(s);
        }
        self.allocator.free(self.keys);
    }

    /// Find the first JWK whose `kid` matches `kid` (or any key if
    /// `kid` is null). Returns null when no match.
    pub fn find(self: *const JwkSet, kid: ?[]const u8) ?*const Jwk {
        for (self.keys) |*k| {
            if (kid == null) return k;
            if (k.kid) |kk| {
                if (std.mem.eql(u8, kk, kid.?)) return k;
            }
        }
        return null;
    }

    /// Like `find` but also requires `alg` to match. Useful when the
    /// JWKS contains multiple keys with the same `kid` for different
    /// algorithms (rare but valid per RFC 7517 §4.4).
    pub fn findByKidAlg(self: *const JwkSet, kid: ?[]const u8, alg: []const u8) ?*const Jwk {
        for (self.keys) |*k| {
            if (!std.mem.eql(u8, k.alg, alg)) continue;
            if (kid == null) return k;
            if (k.kid) |kk| {
                if (std.mem.eql(u8, kk, kid.?)) return k;
            }
        }
        return null;
    }
};

pub const ValidateError = error{
    /// Bearer header is absent or malformed.
    MissingBearer,
    /// JWT didn't have 3 dot-separated parts.
    Malformed,
    /// base64url decode failure on header / payload / signature.
    InvalidEncoding,
    /// `alg` header is unsupported (only HS256 and RS256 are wired in).
    UnsupportedAlg,
    /// `kid` doesn't match any JWK in the set.
    UnknownKid,
    /// HMAC or RSA signature didn't verify.
    BadSignature,
    /// `iss` claim didn't match the configured issuer.
    BadIssuer,
    /// `aud` claim didn't include the configured audience.
    BadAudience,
    /// `exp` claim is in the past (or absent when required).
    Expired,
    /// Out-of-memory during JSON parse / base64 decode.
    OutOfMemory,
    /// Unexpected JSON shape in header or payload.
    BadClaims,
    /// RSA modulus length is unsupported (not 128/256/384/512 bytes).
    UnsupportedRsaModulus,
    /// JWK is missing required RSA fields (n or e).
    InvalidRsaKey,
};

/// Pluggable HTTP GET. Returns the response body as a freshly-allocated
/// slice owned by the caller, or an error. Injected into
/// `Validator.initFromDiscovery` so tests can stub the network.
pub const HttpFetchFn = *const fn (
    ctx: ?*anyopaque,
    allocator: std.mem.Allocator,
    url: []const u8,
) anyerror![]u8;

pub const Validator = struct {
    allocator: std.mem.Allocator,
    issuer: []const u8,
    audience: []const u8,
    /// 10-minute JWKS TTL by default. Time provided by `now_ns()`.
    jwks_ttl_ns: i128 = 10 * 60 * 1_000_000_000,
    jwks: JwkSet,
    /// JWKS URI used for refresh on TTL expiry or kid miss. Owned by
    /// the validator (allocated by `initFromDiscovery`). Null when the
    /// validator was constructed from a static JWKS in tests.
    jwks_uri: ?[]u8 = null,
    /// HTTP fetch callback used to refresh the JWKS. Null when no
    /// refresh path is wired (static-JWKS test construction).
    http_fetch: ?HttpFetchFn = null,
    /// Opaque context passed to `http_fetch`. Typically a `*std.http.Client`.
    http_ctx: ?*anyopaque = null,
    /// Issuer/audience storage. When initFromDiscovery allocates them,
    /// owned_strings is set so deinit can free.
    owned_strings: bool = false,

    pub fn deinit(self: *Validator) void {
        self.jwks.deinit();
        if (self.jwks_uri) |s| self.allocator.free(s);
        if (self.owned_strings) {
            self.allocator.free(self.issuer);
            self.allocator.free(self.audience);
        }
    }

    /// Refresh the JWKS in-place by re-fetching `jwks_uri`. No-op if
    /// the validator wasn't built via `initFromDiscovery`. Updates
    /// `fetched_at_ns` on success; on failure the old JWKS is kept and
    /// the error is returned (caller can log + continue).
    pub fn refreshJwks(self: *Validator) !void {
        const uri = self.jwks_uri orelse return; // static jwks, no-op
        const fetch = self.http_fetch orelse return;
        const body = try fetch(self.http_ctx, self.allocator, uri);
        defer self.allocator.free(body);
        var new_jwks = try parseJwks(self.allocator, body, wallNanos());
        // Swap in atomically; only deinit the old set after the new one
        // is fully built so a refresh failure leaves us with the old.
        var old = self.jwks;
        self.jwks = new_jwks;
        old.deinit();
        _ = &new_jwks; // already moved
    }

    /// True if the JWKS has been in cache longer than `jwks_ttl_ns`
    /// (using the wall clock — `parseJwks` records monotonic ns at
    /// fetch time and `refreshJwks` updates it via `std.time.nanoTimestamp`).
    pub fn jwksIsStale(self: *const Validator) bool {
        const now = wallNanos();
        return (now - self.jwks.fetched_at_ns) > self.jwks_ttl_ns;
    }

    /// Validate the bearer header value (the part after `Bearer `).
    /// `now_unix_seconds` is injected for hermetic testing.
    pub fn validateBearer(
        self: *const Validator,
        token: []const u8,
        now_unix_seconds: i64,
    ) ValidateError!void {
        // Split header.payload.signature on '.'.
        const dot1 = std.mem.indexOfScalar(u8, token, '.') orelse return error.Malformed;
        const rest = token[dot1 + 1 ..];
        const dot2_rel = std.mem.indexOfScalar(u8, rest, '.') orelse return error.Malformed;
        const dot2 = dot1 + 1 + dot2_rel;
        const header_b64 = token[0..dot1];
        const payload_b64 = token[dot1 + 1 .. dot2];
        const sig_b64 = token[dot2 + 1 ..];

        // Decode header.
        var header_buf: [1024]u8 = undefined;
        const header_bytes = b64UrlDecodeInto(header_b64, &header_buf) catch return error.InvalidEncoding;
        var header_arena = std.heap.ArenaAllocator.init(self.allocator);
        defer header_arena.deinit();
        const Header = struct { alg: ?[]u8 = null, kid: ?[]u8 = null, typ: ?[]u8 = null };
        const header = std.json.parseFromSliceLeaky(
            Header,
            header_arena.allocator(),
            header_bytes,
            .{ .ignore_unknown_fields = true, .allocate = .alloc_always },
        ) catch return error.BadClaims;
        const alg = header.alg orelse return error.UnsupportedAlg;
        const is_hs256 = std.mem.eql(u8, alg, "HS256");
        const is_rs256 = std.mem.eql(u8, alg, "RS256");
        if (!is_hs256 and !is_rs256) return error.UnsupportedAlg;

        // Find a matching JWK (must match both kid and alg so we don't
        // try to verify an RS256 JWT against an HS256 key with the
        // same kid, or vice versa).
        const jwk = self.jwks.findByKidAlg(header.kid, alg) orelse return error.UnknownKid;

        // Decode the signature once. RS256 sigs are modulus-sized;
        // HS256 sigs are 32 bytes. Both fit in a 512-byte stack buffer
        // (RSA-4096 = 512-byte sig).
        var sig_buf: [512]u8 = undefined;
        const sig_bytes = b64UrlDecodeInto(sig_b64, &sig_buf) catch return error.InvalidEncoding;

        if (is_hs256) {
            const key_material = jwk.k orelse return error.UnknownKid;

            // Verify signature over `header_b64.payload_b64` with HMAC-SHA256.
            var mac: [HmacSha256.mac_length]u8 = undefined;
            var hmac = HmacSha256.init(key_material);
            hmac.update(header_b64);
            hmac.update(".");
            hmac.update(payload_b64);
            hmac.final(&mac);

            if (sig_bytes.len != mac.len) return error.BadSignature;
            const sig_arr: [HmacSha256.mac_length]u8 = sig_bytes[0..HmacSha256.mac_length].*;
            if (!std.crypto.timing_safe.eql([HmacSha256.mac_length]u8, mac, sig_arr)) return error.BadSignature;
        } else {
            // RS256.
            const n_bytes = jwk.n orelse return error.InvalidRsaKey;
            const e_bytes = jwk.e orelse return error.InvalidRsaKey;
            try verifyRs256(header_b64, payload_b64, sig_bytes, n_bytes, e_bytes);
        }

        // Decode payload + check claims.
        // Payloads can be larger than headers (group memberships etc.),
        // so 16 KiB.
        var payload_buf: [16 * 1024]u8 = undefined;
        const payload_bytes = b64UrlDecodeInto(payload_b64, &payload_buf) catch return error.InvalidEncoding;
        var payload_arena = std.heap.ArenaAllocator.init(self.allocator);
        defer payload_arena.deinit();
        // `aud` may be either a string OR an array of strings; we
        // tolerate both via std.json.Value rather than a hard struct.
        const payload_val = std.json.parseFromSliceLeaky(
            std.json.Value,
            payload_arena.allocator(),
            payload_bytes,
            .{ .ignore_unknown_fields = true, .allocate = .alloc_always },
        ) catch return error.BadClaims;
        if (payload_val != .object) return error.BadClaims;
        const obj = payload_val.object;

        // iss
        if (obj.get("iss")) |v| {
            if (v != .string) return error.BadIssuer;
            if (!std.mem.eql(u8, v.string, self.issuer)) return error.BadIssuer;
        } else {
            return error.BadIssuer;
        }

        // aud
        var aud_ok = false;
        if (obj.get("aud")) |v| switch (v) {
            .string => |s| aud_ok = std.mem.eql(u8, s, self.audience),
            .array => |arr| {
                for (arr.items) |item| {
                    if (item == .string and std.mem.eql(u8, item.string, self.audience)) {
                        aud_ok = true;
                        break;
                    }
                }
            },
            else => {},
        };
        if (!aud_ok) return error.BadAudience;

        // exp
        if (obj.get("exp")) |v| switch (v) {
            .integer => |i| if (i <= now_unix_seconds) return error.Expired,
            .float => |f| if (@as(i64, @intFromFloat(f)) <= now_unix_seconds) return error.Expired,
            else => return error.Expired,
        } else {
            return error.Expired;
        }
    }
};

/// Construct a Validator by:
///   1) GETting `<issuer>/.well-known/openid-configuration`
///   2) parsing the discovery doc and extracting `jwks_uri`
///   3) GETting the JWKS and decoding it
///   4) returning a validator that retains `jwks_uri` + `fetch` for
///      future refreshes
///
/// `fetch` lets tests stub the HTTP path. For production, pass
/// `httpFetchWithStdClient` and a `*std.http.Client` ctx.
///
/// Returns a heap-allocated `Validator` owned by the caller — call
/// `deinit()` then `allocator.destroy()` to release.
pub fn initFromDiscovery(
    allocator: std.mem.Allocator,
    fetch: HttpFetchFn,
    http_ctx: ?*anyopaque,
    issuer_url: []const u8,
    audience: []const u8,
) !*Validator {
    // 1) Discovery doc.
    var discovery_url_buf: [1024]u8 = undefined;
    const trimmed_issuer = std.mem.trimEnd(u8, issuer_url, "/");
    const disc_url = try std.fmt.bufPrint(
        &discovery_url_buf,
        "{s}/.well-known/openid-configuration",
        .{trimmed_issuer},
    );
    const disc_body = try fetch(http_ctx, allocator, disc_url);
    defer allocator.free(disc_body);

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const disc_val = try std.json.parseFromSliceLeaky(
        std.json.Value,
        arena.allocator(),
        disc_body,
        .{ .ignore_unknown_fields = true, .allocate = .alloc_always },
    );
    if (disc_val != .object) return error.BadDiscoveryDoc;
    const jwks_uri_v = disc_val.object.get("jwks_uri") orelse return error.BadDiscoveryDoc;
    if (jwks_uri_v != .string) return error.BadDiscoveryDoc;
    const jwks_uri_owned = try allocator.dupe(u8, jwks_uri_v.string);
    errdefer allocator.free(jwks_uri_owned);

    // 2) JWKS.
    const jwks_body = try fetch(http_ctx, allocator, jwks_uri_owned);
    defer allocator.free(jwks_body);
    var jwks = try parseJwks(allocator, jwks_body, wallNanos());
    errdefer jwks.deinit();

    // 3) Heap-allocate the validator so the pointer stays stable
    //    across cli_serve borrow + refresh calls.
    const issuer_owned = try allocator.dupe(u8, trimmed_issuer);
    errdefer allocator.free(issuer_owned);
    const audience_owned = try allocator.dupe(u8, audience);
    errdefer allocator.free(audience_owned);

    const v = try allocator.create(Validator);
    v.* = .{
        .allocator = allocator,
        .issuer = issuer_owned,
        .audience = audience_owned,
        .jwks = jwks,
        .jwks_uri = jwks_uri_owned,
        .http_fetch = fetch,
        .http_ctx = http_ctx,
        .owned_strings = true,
    };
    return v;
}

/// Production HTTP fetch backed by `std.http.Client`. `ctx` must point
/// to a `*std.http.Client` whose IO context is alive for the duration
/// of the call. Caller-supplied client lets the validator share a
/// pool across discovery + JWKS + refresh fetches.
pub fn httpFetchWithStdClient(
    ctx: ?*anyopaque,
    allocator: std.mem.Allocator,
    url: []const u8,
) anyerror![]u8 {
    const client_ptr = ctx orelse return error.NoHttpClient;
    const client: *std.http.Client = @ptrCast(@alignCast(client_ptr));

    var body: std.ArrayList(u8) = .empty;
    errdefer body.deinit(allocator);
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();

    const result = client.fetch(.{
        .location = .{ .url = url },
        .method = .GET,
        .response_writer = &aw.writer,
    }) catch |err| return err;
    if (@intFromEnum(result.status) < 200 or @intFromEnum(result.status) >= 300) {
        return error.HttpStatusError;
    }
    return try aw.toOwnedSlice();
}

/// Decode a JWKS document (an object with a `keys` array) and return
/// a fresh `JwkSet`. `fetched_at_ns` lets the caller plug in a real
/// monotonic-clock timestamp or zero for tests.
pub fn parseJwks(
    allocator: std.mem.Allocator,
    bytes: []const u8,
    fetched_at_ns: i128,
) !JwkSet {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const parsed = try std.json.parseFromSliceLeaky(
        std.json.Value,
        arena.allocator(),
        bytes,
        .{ .ignore_unknown_fields = true, .allocate = .alloc_always },
    );
    if (parsed != .object) return error.BadClaims;
    const keys_val = parsed.object.get("keys") orelse return error.BadClaims;
    if (keys_val != .array) return error.BadClaims;

    var out: std.ArrayList(Jwk) = .empty;
    errdefer {
        for (out.items) |k| {
            if (k.kid) |s| allocator.free(s);
            allocator.free(k.alg);
            allocator.free(k.kty);
            if (k.k) |s| allocator.free(s);
            if (k.n) |s| allocator.free(s);
            if (k.e) |s| allocator.free(s);
        }
        out.deinit(allocator);
    }
    for (keys_val.array.items) |entry| {
        if (entry != .object) continue;
        const o = entry.object;
        const kty_v = o.get("kty") orelse continue;
        const alg_v = o.get("alg") orelse continue;
        if (kty_v != .string or alg_v != .string) continue;
        const kty = try allocator.dupe(u8, kty_v.string);
        errdefer allocator.free(kty);
        const alg = try allocator.dupe(u8, alg_v.string);
        errdefer allocator.free(alg);
        var kid: ?[]u8 = null;
        if (o.get("kid")) |kv| if (kv == .string) {
            kid = try allocator.dupe(u8, kv.string);
        };
        errdefer if (kid) |s| allocator.free(s);
        var k_material: ?[]u8 = null;
        errdefer if (k_material) |s| allocator.free(s);
        if (o.get("k")) |kv| if (kv == .string) {
            k_material = try b64UrlDecodeAlloc(allocator, kv.string);
        };
        var n_material: ?[]u8 = null;
        errdefer if (n_material) |s| allocator.free(s);
        if (o.get("n")) |kv| if (kv == .string) {
            n_material = try b64UrlDecodeAlloc(allocator, kv.string);
        };
        var e_material: ?[]u8 = null;
        errdefer if (e_material) |s| allocator.free(s);
        if (o.get("e")) |kv| if (kv == .string) {
            e_material = try b64UrlDecodeAlloc(allocator, kv.string);
        };
        try out.append(allocator, .{
            .kid = kid,
            .alg = alg,
            .kty = kty,
            .k = k_material,
            .n = n_material,
            .e = e_material,
        });
    }
    return .{
        .allocator = allocator,
        .keys = try out.toOwnedSlice(allocator),
        .fetched_at_ns = fetched_at_ns,
    };
}

/// Wall-clock nanoseconds since the Unix epoch. `std.time.nanoTimestamp`
/// was retired from std in Zig 0.16 in favor of `std.Io.Clock`; we can't
/// thread an `Io` handle through the Validator without a bigger
/// refactor, so the JWKS TTL helpers fall back to direct libc.
/// Returns 0 if the syscall fails — the TTL check then trips false
/// and the next request still works (it just doesn't refresh).
fn wallNanos() i128 {
    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(.REALTIME, &ts) != 0) return 0;
    return @as(i128, ts.sec) * std.time.ns_per_s + @as(i128, ts.nsec);
}

/// Heap-allocate the base64url-decoded bytes of `src`. Caller frees.
fn b64UrlDecodeAlloc(allocator: std.mem.Allocator, src: []const u8) ![]u8 {
    const buf = try allocator.alloc(u8, b64UrlDecodedLen(src) + 4);
    errdefer allocator.free(buf);
    const got = b64UrlDecodeInto(src, buf) catch return error.BadClaims;
    if (got.len != buf.len) {
        return try allocator.realloc(buf, got.len);
    }
    return buf;
}

/// Verify a JWT's RS256 signature. `header_b64` and `payload_b64` are
/// the JWT segments verbatim (no padding); `sig` is the raw signature
/// bytes (already base64url-decoded). `n_bytes` and `e_bytes` are the
/// big-endian modulus and exponent from the JWK.
fn verifyRs256(
    header_b64: []const u8,
    payload_b64: []const u8,
    sig: []const u8,
    n_bytes: []const u8,
    e_bytes: []const u8,
) ValidateError!void {
    // Trim a leading 0x00 padding byte off the modulus the way the
    // stdlib Certificate path does it — JWKs sometimes carry it
    // because base64url(BIGNUM) round-trips to leading zeros.
    var modulus = n_bytes;
    while (modulus.len > 0 and modulus[0] == 0) modulus = modulus[1..];
    if (modulus.len == 0) return error.InvalidRsaKey;
    if (e_bytes.len == 0 or e_bytes.len > 4) return error.InvalidRsaKey;
    if (sig.len != modulus.len) return error.BadSignature;

    const pub_key = cert_rsa.PublicKey.fromBytes(e_bytes, modulus) catch return error.InvalidRsaKey;

    // `PKCS1v1_5Signature.verify` is generic over modulus_len at
    // comptime. JWKs in the wild come in 2048/3072/4096-bit flavors
    // (256 / 384 / 512 bytes); 1024-bit (128 bytes) is technically
    // legal but already deprecated by most IdPs. We support all four
    // sizes the stdlib Certificate path supports.
    switch (modulus.len) {
        inline 128, 256, 384, 512 => |mod_len| {
            const sig_arr: [mod_len]u8 = sig[0..mod_len].*;
            cert_rsa.PKCS1v1_5Signature.concatVerify(
                mod_len,
                sig_arr,
                &.{ header_b64, ".", payload_b64 },
                pub_key,
                Sha256,
            ) catch return error.BadSignature;
        },
        else => return error.UnsupportedRsaModulus,
    }
}

/// Conservative upper bound on the decoded length of `in` characters
/// of base64url input. The exact length depends on padding, which we
/// adjust for inside `b64UrlDecodeInto`.
fn b64UrlDecodedLen(in: []const u8) usize {
    return (in.len * 3 + 3) / 4;
}

/// base64url-decode `in` into `buf`. Tolerates missing padding (which
/// JWT mandates omitting). Returns a slice into `buf`.
fn b64UrlDecodeInto(in: []const u8, buf: []u8) ![]u8 {
    // Pad the input to a multiple of 4 with '=' so we can use the
    // standard base64 url-safe decoder.
    var padded_buf: [4096]u8 = undefined;
    const pad_n: usize = (4 - in.len % 4) % 4;
    if (in.len + pad_n > padded_buf.len) return error.InvalidEncoding;
    @memcpy(padded_buf[0..in.len], in);
    var i: usize = 0;
    while (i < pad_n) : (i += 1) padded_buf[in.len + i] = '=';
    const padded = padded_buf[0 .. in.len + pad_n];
    const decoder = std.base64.url_safe.Decoder;
    const decoded_len = decoder.calcSizeForSlice(padded) catch return error.InvalidEncoding;
    if (decoded_len > buf.len) return error.InvalidEncoding;
    decoder.decode(buf[0..decoded_len], padded) catch return error.InvalidEncoding;
    return buf[0..decoded_len];
}

// === Tests ============================================================

const testing = std.testing;

/// Build a JWT with the given header + payload + HMAC-SHA256 key.
/// Caller frees the returned slice.
fn buildHs256Jwt(
    a: std.mem.Allocator,
    header_json: []const u8,
    payload_json: []const u8,
    key: []const u8,
) ![]u8 {
    const enc = std.base64.url_safe_no_pad.Encoder;
    const h_buf = try a.alloc(u8, enc.calcSize(header_json.len));
    defer a.free(h_buf);
    const h_b64 = enc.encode(h_buf, header_json);
    const p_buf = try a.alloc(u8, enc.calcSize(payload_json.len));
    defer a.free(p_buf);
    const p_b64 = enc.encode(p_buf, payload_json);

    var mac: [HmacSha256.mac_length]u8 = undefined;
    var hmac = HmacSha256.init(key);
    hmac.update(h_b64);
    hmac.update(".");
    hmac.update(p_b64);
    hmac.final(&mac);

    const s_buf = try a.alloc(u8, enc.calcSize(mac.len));
    defer a.free(s_buf);
    const s_b64 = enc.encode(s_buf, &mac);

    return std.fmt.allocPrint(a, "{s}.{s}.{s}", .{ h_b64, p_b64, s_b64 });
}

test "oidc: HS256 token with correct iss/aud/exp validates" {
    const a = testing.allocator;
    const key = "supersecretkey0123456789";
    // Build a JWKS with this key. `k` field is the base64url-no-pad
    // encoded raw key bytes per RFC 7518 §6.4.
    const enc = std.base64.url_safe_no_pad.Encoder;
    var k_buf: [128]u8 = undefined;
    const k_b64 = enc.encode(k_buf[0..enc.calcSize(key.len)], key);
    const jwks_json = try std.fmt.allocPrint(
        a,
        "{{\"keys\":[{{\"kid\":\"k1\",\"alg\":\"HS256\",\"kty\":\"oct\",\"k\":\"{s}\"}}]}}",
        .{k_b64},
    );
    defer a.free(jwks_json);
    var jwks = try parseJwks(a, jwks_json, 0);
    var v: Validator = .{
        .allocator = a,
        .issuer = "https://example.com",
        .audience = "ztok",
        .jwks = jwks,
    };
    defer v.deinit();
    _ = &jwks;

    const header = "{\"alg\":\"HS256\",\"typ\":\"JWT\",\"kid\":\"k1\"}";
    const payload = "{\"iss\":\"https://example.com\",\"aud\":\"ztok\",\"exp\":9999999999}";
    const jwt = try buildHs256Jwt(a, header, payload, key);
    defer a.free(jwt);

    try v.validateBearer(jwt, 1_000_000_000);
}

test "oidc: HS256 token signed with wrong key is rejected" {
    const a = testing.allocator;
    const real_key = "supersecretkey0123456789";
    const wrong_key = "wrongkey0000000000000000";
    const enc = std.base64.url_safe_no_pad.Encoder;
    var k_buf: [128]u8 = undefined;
    const k_b64 = enc.encode(k_buf[0..enc.calcSize(real_key.len)], real_key);
    const jwks_json = try std.fmt.allocPrint(
        a,
        "{{\"keys\":[{{\"kid\":\"k1\",\"alg\":\"HS256\",\"kty\":\"oct\",\"k\":\"{s}\"}}]}}",
        .{k_b64},
    );
    defer a.free(jwks_json);
    var v: Validator = .{
        .allocator = a,
        .issuer = "https://example.com",
        .audience = "ztok",
        .jwks = try parseJwks(a, jwks_json, 0),
    };
    defer v.deinit();

    const header = "{\"alg\":\"HS256\",\"typ\":\"JWT\",\"kid\":\"k1\"}";
    const payload = "{\"iss\":\"https://example.com\",\"aud\":\"ztok\",\"exp\":9999999999}";
    const jwt = try buildHs256Jwt(a, header, payload, wrong_key);
    defer a.free(jwt);

    try testing.expectError(error.BadSignature, v.validateBearer(jwt, 1_000_000_000));
}

test "oidc: expired exp claim fails" {
    const a = testing.allocator;
    const key = "secret_secret_secret_secret";
    const enc = std.base64.url_safe_no_pad.Encoder;
    var k_buf: [128]u8 = undefined;
    const k_b64 = enc.encode(k_buf[0..enc.calcSize(key.len)], key);
    const jwks_json = try std.fmt.allocPrint(
        a,
        "{{\"keys\":[{{\"kid\":\"k1\",\"alg\":\"HS256\",\"kty\":\"oct\",\"k\":\"{s}\"}}]}}",
        .{k_b64},
    );
    defer a.free(jwks_json);
    var v: Validator = .{
        .allocator = a,
        .issuer = "iss",
        .audience = "aud",
        .jwks = try parseJwks(a, jwks_json, 0),
    };
    defer v.deinit();

    // exp = 100, now = 200 → Expired.
    const header = "{\"alg\":\"HS256\",\"typ\":\"JWT\",\"kid\":\"k1\"}";
    const payload = "{\"iss\":\"iss\",\"aud\":\"aud\",\"exp\":100}";
    const jwt = try buildHs256Jwt(a, header, payload, key);
    defer a.free(jwt);

    try testing.expectError(error.Expired, v.validateBearer(jwt, 200));
}

test "oidc: ES256 (an algorithm we don't implement) is rejected with UnsupportedAlg" {
    const a = testing.allocator;
    const key = "x";
    const enc = std.base64.url_safe_no_pad.Encoder;
    var k_buf: [128]u8 = undefined;
    const k_b64 = enc.encode(k_buf[0..enc.calcSize(key.len)], key);
    const jwks_json = try std.fmt.allocPrint(
        a,
        "{{\"keys\":[{{\"kid\":\"k1\",\"alg\":\"HS256\",\"kty\":\"oct\",\"k\":\"{s}\"}}]}}",
        .{k_b64},
    );
    defer a.free(jwks_json);
    var v: Validator = .{
        .allocator = a,
        .issuer = "iss",
        .audience = "aud",
        .jwks = try parseJwks(a, jwks_json, 0),
    };
    defer v.deinit();

    // Header advertises ES256 (P-256 ECDSA — not wired in this build).
    // RS256 is now supported and would route to the RS256 path; ES256
    // still trips UnsupportedAlg in the alg dispatch.
    const header = "{\"alg\":\"ES256\",\"typ\":\"JWT\",\"kid\":\"k1\"}";
    const payload = "{\"iss\":\"iss\",\"aud\":\"aud\",\"exp\":9999999999}";
    const jwt = try buildHs256Jwt(a, header, payload, key);
    defer a.free(jwt);

    try testing.expectError(error.UnsupportedAlg, v.validateBearer(jwt, 0));
}

test "oidc: aud array form is accepted when audience is one of the entries" {
    const a = testing.allocator;
    const key = "k0_k0_k0_k0_k0_k0_k0_k0_k0_k0_xx";
    const enc = std.base64.url_safe_no_pad.Encoder;
    var k_buf: [128]u8 = undefined;
    const k_b64 = enc.encode(k_buf[0..enc.calcSize(key.len)], key);
    const jwks_json = try std.fmt.allocPrint(
        a,
        "{{\"keys\":[{{\"kid\":\"k1\",\"alg\":\"HS256\",\"kty\":\"oct\",\"k\":\"{s}\"}}]}}",
        .{k_b64},
    );
    defer a.free(jwks_json);
    var v: Validator = .{
        .allocator = a,
        .issuer = "iss",
        .audience = "svcB",
        .jwks = try parseJwks(a, jwks_json, 0),
    };
    defer v.deinit();

    const header = "{\"alg\":\"HS256\",\"typ\":\"JWT\",\"kid\":\"k1\"}";
    const payload = "{\"iss\":\"iss\",\"aud\":[\"svcA\",\"svcB\"],\"exp\":9999999999}";
    const jwt = try buildHs256Jwt(a, header, payload, key);
    defer a.free(jwt);

    try v.validateBearer(jwt, 0);
}

// === RS256 tests =======================================================
//
// The RSA-2048 keypair below was generated once via openssl + a Python
// helper that signs three JWTs (good, bad-signature, wrong-kid) and
// dumps the JWK n / e fields. The private key never enters the repo;
// only the public modulus + the precomputed JWT signatures are checked
// in. See `/tmp/gen_jwt.py` in the 1.23 hardening commit history for
// the generator if these ever need to be regenerated.

const rs256_n_b64u = "58kk3d_3GzwrcE7iHKeKTh4aL7PEb3-pKu7_paTI_AKQsGgx4gAesc-YKKwsFf6FUrS20_70Ay002KJf0FrGnANoW1hz6slkT7x3j_C3EBTgJHfDO38o8ZbgwnptQZjb0Gv3J5psS8TapFJV5AUr8_zRAX4MRa3TsdSlEFlt60CukaVUQzG1RmZfG3ymixEllbYsNzAjecA57xBp5RU7EP3VLSdhZDMoQ4yDjLF6zc0ry4Hjg12xR8Pt4qc4hNA7W-BoQa8HFBwHggKN-6NKeH24REOt643pl0RfKWWrBTUpZGdD2uVz2x-i7Ocj0xOnSFB5oMQHFALIpGMjRO5MYw";
const rs256_e_b64u = "AQAB";
const rs256_jwt_good = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6InJzMSJ9.eyJpc3MiOiJodHRwczovL2V4YW1wbGUuY29tIiwiYXVkIjoienRvayIsImV4cCI6OTk5OTk5OTk5OX0.1kd4WAIqRwYelzsXn5FeOmSAx0NZvzu_MI0Cevk7Fd8Vtv_BhiH0TDmToNFQJl2qbnYuAOzaHPkC4KFqGQyQ2Z7p7Eky-V2gmlkaKgOvzuA6QDvx4VqT_XsVolVu4-1lQZlC1pKKh5rcm4amwyUorsAo700CoX1V-oG4TGiXhfp-32x-BXJoAagaCWcv7bcKb0uhExzJ47kKa0_MNh0zN-inBD2TStyVNSivT4aAtgJy6K-1NFlnqnHilRnOylQckTPDTGuX8A63RCGJa_27NhXJ_WyCmh0QKd6Um75CCOYDaGbCMWL0wuncZXx3BzAT0Ru7onLVRguSXiVkK8fCew";
const rs256_jwt_bad_sig = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6InJzMSJ9.eyJpc3MiOiJodHRwczovL2V4YW1wbGUuY29tIiwiYXVkIjoienRvayIsImV4cCI6OTk5OTk5OTk5OX0.1kd4WAIqRwYelzsXn5FeOmSAx0NZvzu_MI0Cevk7Fd8Vtv_BhiH0TDmToNFQJl2qbnYuAOzaHPkC4KFqGQyQ2Z7p7Eky-V2gmlkaKgOvzuA6QDvx4VqT_XsVolVu4-1lQZlC1pKKh5rcm4amwyUorsAo700CoX1V-oG4TGiXhfp-32x-BXJoAagaCWcv7bcKb0uhExzJ47kKa0_MNh0zN-inBD2TStyVNSivT4aAtgJy6K-1NFlnqnHilRnOylQckTPDTGuX8A63RCGJa_27NhXJ_WyCmh0QKd6Um75CCOYDaGbCMWL0wuncZXx3BzAT0Ru7onLVRguSXiVkK8fChA";
const rs256_jwt_wrong_kid = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Im90aGVyIn0.eyJpc3MiOiJodHRwczovL2V4YW1wbGUuY29tIiwiYXVkIjoienRvayIsImV4cCI6OTk5OTk5OTk5OX0.2tynK2qVK6Qoa-lIQTxnTm6Tcufee2DrRGCYr7739zSijkLwJdqqttGOyVpz24zhTgWlkVlUo2aKJWO9MkeByMZZj5zCfPBnboSVmQSmwY5o0XdPgEYQ88Mo0lFGMMDgwh5lfVz_O2J_OiP3UFJWYmhn7_IdrlzYtoNozqGTye0Dp2d9VleGKCGHKvCTl2O4kOw0Uz2mgAieRwWo7LzOoZ_tQTCYP15KoaYgUeukCrmnInM5GHpS2diPYrvx6tzfrwTrPlwihajhD4ePztvBHQ_xaS5nb6M5C-cRgbC5qSAKDJspLX-BLx88k1fAlKgBy8fuCh6acKMS1B8wEONT7A";

fn buildRs256Jwks(a: std.mem.Allocator) ![]u8 {
    return std.fmt.allocPrint(
        a,
        "{{\"keys\":[{{\"kid\":\"rs1\",\"alg\":\"RS256\",\"kty\":\"RSA\",\"n\":\"{s}\",\"e\":\"{s}\"}}]}}",
        .{ rs256_n_b64u, rs256_e_b64u },
    );
}

test "oidc: RS256 token with matching JWK validates" {
    const a = testing.allocator;
    const jwks_json = try buildRs256Jwks(a);
    defer a.free(jwks_json);
    var v: Validator = .{
        .allocator = a,
        .issuer = "https://example.com",
        .audience = "ztok",
        .jwks = try parseJwks(a, jwks_json, 0),
    };
    defer v.deinit();

    try v.validateBearer(rs256_jwt_good, 1_000_000_000);
}

test "oidc: RS256 token with flipped signature byte is rejected" {
    const a = testing.allocator;
    const jwks_json = try buildRs256Jwks(a);
    defer a.free(jwks_json);
    var v: Validator = .{
        .allocator = a,
        .issuer = "https://example.com",
        .audience = "ztok",
        .jwks = try parseJwks(a, jwks_json, 0),
    };
    defer v.deinit();

    try testing.expectError(error.BadSignature, v.validateBearer(rs256_jwt_bad_sig, 1_000_000_000));
}

test "oidc: RS256 token with kid that isn't in the JWKS is rejected with UnknownKid" {
    const a = testing.allocator;
    const jwks_json = try buildRs256Jwks(a);
    defer a.free(jwks_json);
    var v: Validator = .{
        .allocator = a,
        .issuer = "https://example.com",
        .audience = "ztok",
        .jwks = try parseJwks(a, jwks_json, 0),
    };
    defer v.deinit();

    try testing.expectError(error.UnknownKid, v.validateBearer(rs256_jwt_wrong_kid, 1_000_000_000));
}

// === Discovery + JWKS auto-fetch tests =================================
//
// Hermetic: the HTTP fetch is a function pointer the validator calls
// for both the discovery doc and the JWKS. The test stub returns the
// canned discovery JSON and the canned JWKS bytes — no socket.

const DiscoveryStubCtx = struct {
    allocator: std.mem.Allocator,
    issuer: []const u8,
    jwks_uri: []const u8,
    jwks_body: []const u8,
    discovery_calls: u32 = 0,
    jwks_calls: u32 = 0,
};

fn discoveryStubFetch(ctx: ?*anyopaque, allocator: std.mem.Allocator, url: []const u8) anyerror![]u8 {
    const c: *DiscoveryStubCtx = @ptrCast(@alignCast(ctx.?));

    if (std.mem.endsWith(u8, url, "/.well-known/openid-configuration")) {
        c.discovery_calls += 1;
        return try std.fmt.allocPrint(
            allocator,
            "{{\"issuer\":\"{s}\",\"jwks_uri\":\"{s}\"}}",
            .{ c.issuer, c.jwks_uri },
        );
    }
    if (std.mem.eql(u8, url, c.jwks_uri)) {
        c.jwks_calls += 1;
        return try allocator.dupe(u8, c.jwks_body);
    }
    return error.StubUnknownUrl;
}

test "oidc: initFromDiscovery pulls discovery doc + JWKS via injected fetch" {
    const a = testing.allocator;
    const jwks_body = try buildRs256Jwks(a);
    defer a.free(jwks_body);

    var ctx: DiscoveryStubCtx = .{
        .allocator = a,
        .issuer = "https://example.com",
        .jwks_uri = "https://example.com/oauth2/jwks",
        .jwks_body = jwks_body,
    };

    const v = try initFromDiscovery(
        a,
        discoveryStubFetch,
        @ptrCast(&ctx),
        "https://example.com",
        "ztok",
    );
    defer {
        v.deinit();
        a.destroy(v);
    }

    try testing.expect(v.jwks.keys.len == 1);
    try testing.expectEqualStrings("rs1", v.jwks.keys[0].kid.?);
    try testing.expectEqualStrings("RS256", v.jwks.keys[0].alg);
    try testing.expect(v.jwks.keys[0].n != null);

    // The fetch path must have made exactly one discovery + one jwks call.
    try testing.expectEqual(@as(u32, 1), ctx.discovery_calls);
    try testing.expectEqual(@as(u32, 1), ctx.jwks_calls);

    // The validator should now verify a real RS256 JWT signed by the
    // injected key.
    try v.validateBearer(rs256_jwt_good, 1_000_000_000);
}

test "oidc: refreshJwks re-pulls the JWKS via the stored fetch fn" {
    const a = testing.allocator;
    const jwks_body = try buildRs256Jwks(a);
    defer a.free(jwks_body);

    var ctx: DiscoveryStubCtx = .{
        .allocator = a,
        .issuer = "https://example.com",
        .jwks_uri = "https://example.com/oauth2/jwks",
        .jwks_body = jwks_body,
    };

    const v = try initFromDiscovery(
        a,
        discoveryStubFetch,
        @ptrCast(&ctx),
        "https://example.com",
        "ztok",
    );
    defer {
        v.deinit();
        a.destroy(v);
    }

    try v.refreshJwks();
    // Discovery URL count is still 1 (we don't re-pull it), but the
    // JWKS URL was hit a second time.
    try testing.expectEqual(@as(u32, 1), ctx.discovery_calls);
    try testing.expectEqual(@as(u32, 2), ctx.jwks_calls);

    // After refresh, the JWKS still verifies the test JWT.
    try v.validateBearer(rs256_jwt_good, 1_000_000_000);
}
