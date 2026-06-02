//! Browser-WASM entry module for ztok.
//!
//! `wasm32-freestanding` doesn't have `std.Io.Threaded`, `std.fs`, or
//! any of the syscall-backed allocators. The default `c_api.zig` /
//! `root.zig` pull a lot of that in transitively (file constructors,
//! `auto_detect.zig`, `monster_io.zig`, etc.), which fails to compile
//! on freestanding.
//!
//! This module is a deliberately narrow surface that:
//!   * Re-exports only the pieces a browser caller needs (BPE +
//!     Pipeline + decode).
//!   * Replaces every file-based constructor with a bytes-based one.
//!   * Provides a minimal `panic` and exports a single linear-memory
//!     allocator (`ztok_malloc` / `ztok_free`) so JS can hand the wasm
//!     instance the vocab + input bytes without a custom Module().
//!
//! Targets `wasm32-freestanding`, `single_threaded = true`. The
//! BatchPool degenerates to a serial loop on this target (see
//! `thread_pool.zig` header), so the encode path is single-thread —
//! exactly what we want in a browser tab.

const std = @import("std");
const builtin = @import("builtin");

// Pull in only the modules we need. Critically: NOT `c_api`, NOT
// `auto_detect`, NOT `monster_io` (the file paths in those modules
// drag in `std.Io.Threaded` which doesn't compile freestanding).
const Pipeline = @import("pipeline.zig").Pipeline;
const Vocab = @import("vocab.zig").Vocab;
const Normalizer = @import("normalizer.zig").Normalizer;
const PreTokenizer = @import("pretok.zig").PreTokenizer;
const Model = @import("model.zig").Model;
const Decoder = @import("decoder.zig").Decoder;
const TokenId = @import("token.zig").TokenId;
const Bpe = @import("bpe.zig").Bpe;

// `wasm32-freestanding` has no host allocator. Use Zig 0.16's
// `WasmAllocator`, which walks the linear-memory page table via
// `@wasmMemoryGrow`. This is exactly what `std.heap.wasm_allocator` is
// supposed to be — keep the indirection so a future stdlib rename
// doesn't break us.
const gpa: std.mem.Allocator = std.heap.wasm_allocator;

const VERSION = "1.28.0-wasm";

// --- status codes -----------------------------------------------------
// Mirrors `c_api.zig`'s codes so JS can rely on the same constants.
const ZTOK_OK: i32 = 0;
const ZTOK_ERR_OUT_OF_MEMORY: i32 = 1;
const ZTOK_ERR_INVALID_INPUT: i32 = 2;
const ZTOK_ERR_BUFFER_TOO_SMALL: i32 = 3;
const ZTOK_ERR_INTERNAL: i32 = 99;

// --- handle wrapping ---------------------------------------------------
//
// The browser surface is BPE-only for now — that's what cl100k_base
// gives us, and what tiktoken-js exposes. If we want Unigram/WordPiece
// in the browser later, add another `_bytes` constructor that wraps
// the matching loader from `hf_json` (which is pure-bytes already).
const PipelineHandle = struct {
    pipeline: Pipeline,
    vocab: Vocab,
    bpe: Bpe,
};

fn mapErr(e: anyerror) i32 {
    return switch (e) {
        error.OutOfMemory => ZTOK_ERR_OUT_OF_MEMORY,
        else => ZTOK_ERR_INTERNAL,
    };
}

fn setStatus(out: ?*i32, v: i32) void {
    if (out) |p| p.* = v;
}

// --- linear-memory plumbing for JS callers ----------------------------
//
// JS can't easily share its `Uint8Array` with the wasm instance — the
// canonical pattern is: JS asks the wasm side for `len` bytes via
// `ztok_malloc`, writes vocab/input into that pointer, then passes
// the pointer + len into the encode/load entry points. `ztok_free`
// returns the buffer to the allocator afterwards.
//
// We prefix every allocation with an 8-byte length header so the free
// side can recover the size without the JS caller round-tripping it.
const HeaderLen = u64;
const header_size: usize = @sizeOf(HeaderLen);

export fn ztok_malloc(n: usize) ?[*]u8 {
    if (n == 0) return null;
    const total = header_size + n;
    const buf = gpa.alignedAlloc(u8, .@"8", total) catch return null;
    const hdr_ptr: *HeaderLen = @ptrCast(@alignCast(buf.ptr));
    hdr_ptr.* = @intCast(n);
    return buf.ptr + header_size;
}

export fn ztok_free(p: ?[*]u8) void {
    const pp = p orelse return;
    const raw = pp - header_size;
    const hdr_ptr: *HeaderLen = @ptrCast(@alignCast(raw));
    const n: usize = @intCast(hdr_ptr.*);
    const total = header_size + n;
    const aligned: [*]align(8) u8 = @alignCast(raw);
    gpa.free(aligned[0..total]);
}

// --- BPE construction --------------------------------------------------
//
// Bytes-in (no filesystem) variant of `ztok_pipeline_new_bpe_from_tiktoken`.
// JS passes a pointer to a `cl100k_base.tiktoken`-format buffer.
// Defaults: identity normalizer + cl100k pretok + concat decoder, which
// matches what `cl100k_base` actually needs.
export fn ztok_pipeline_new_bpe_from_tiktoken_bytes(
    bytes_ptr: ?[*]const u8,
    bytes_len: usize,
    out_status: ?*i32,
) ?*PipelineHandle {
    const ptr = bytes_ptr orelse {
        setStatus(out_status, ZTOK_ERR_INVALID_INPUT);
        return null;
    };
    if (bytes_len == 0) {
        setStatus(out_status, ZTOK_ERR_INVALID_INPUT);
        return null;
    }
    const contents = ptr[0..bytes_len];

    var bpe = Bpe.loadTiktokenBytes(gpa, contents) catch |e| {
        setStatus(out_status, mapErr(e));
        return null;
    };

    const h = gpa.create(PipelineHandle) catch |e| {
        bpe.deinit();
        setStatus(out_status, mapErr(e));
        return null;
    };
    h.* = .{
        .pipeline = undefined,
        .vocab = Vocab.empty(gpa),
        .bpe = bpe,
    };
    h.pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .cl100k,
        .model = .{ .bpe = &h.bpe },
        .decoder = .concat,
        .vocab = &h.vocab,
    };
    setStatus(out_status, ZTOK_OK);
    return h;
}

export fn ztok_pipeline_free(p: ?*PipelineHandle) void {
    const pp = p orelse return;
    pp.bpe.deinit();
    pp.vocab.deinit();
    gpa.destroy(pp);
}

// --- encode / decode ---------------------------------------------------
//
// JS calls `ztok_encode(p, input_ptr, input_len, &out_ids_ptr, &out_len)`.
// We allocate the result buffer with `ztok_malloc` so JS can read it,
// then free with `ztok_free`. The double-out-pointer pattern is the
// idiomatic wasm convention for returning multiple scalars.
//
// `out_ids_ptr` receives the pointer to the result buffer (or 0 on
// error / empty); `out_len_ptr` receives the id count.
export fn ztok_encode(
    p: ?*PipelineHandle,
    input_ptr: ?[*]const u8,
    input_len: usize,
    out_ids_ptr: ?*u32, // pointer-to-u32 holding the result pointer
    out_len_ptr: ?*u32,
) i32 {
    const pp = p orelse return ZTOK_ERR_INVALID_INPUT;
    const ip = input_ptr orelse return ZTOK_ERR_INVALID_INPUT;
    const op = out_ids_ptr orelse return ZTOK_ERR_INVALID_INPUT;
    const ol = out_len_ptr orelse return ZTOK_ERR_INVALID_INPUT;

    const ids = pp.pipeline.encode(gpa, ip[0..input_len]) catch |e| {
        op.* = 0;
        ol.* = 0;
        return mapErr(e);
    };
    defer gpa.free(ids);

    if (ids.len == 0) {
        op.* = 0;
        ol.* = 0;
        return ZTOK_OK;
    }

    const byte_len = ids.len * @sizeOf(TokenId);
    const out_buf = ztok_malloc(byte_len) orelse {
        op.* = 0;
        ol.* = 0;
        return ZTOK_ERR_OUT_OF_MEMORY;
    };
    const out_as_ids: [*]TokenId = @ptrCast(@alignCast(out_buf));
    @memcpy(out_as_ids[0..ids.len], ids);

    op.* = @intCast(@intFromPtr(out_buf));
    ol.* = @intCast(ids.len);
    return ZTOK_OK;
}

export fn ztok_decode(
    p: ?*PipelineHandle,
    ids_ptr: ?[*]const TokenId,
    ids_len: usize,
    out_bytes_ptr: ?*u32,
    out_len_ptr: ?*u32,
) i32 {
    const pp = p orelse return ZTOK_ERR_INVALID_INPUT;
    const ip = ids_ptr orelse return ZTOK_ERR_INVALID_INPUT;
    const op = out_bytes_ptr orelse return ZTOK_ERR_INVALID_INPUT;
    const ol = out_len_ptr orelse return ZTOK_ERR_INVALID_INPUT;

    const bytes = pp.pipeline.decode(gpa, ip[0..ids_len]) catch |e| {
        op.* = 0;
        ol.* = 0;
        return mapErr(e);
    };
    defer gpa.free(bytes);

    if (bytes.len == 0) {
        op.* = 0;
        ol.* = 0;
        return ZTOK_OK;
    }

    const out_buf = ztok_malloc(bytes.len) orelse {
        op.* = 0;
        ol.* = 0;
        return ZTOK_ERR_OUT_OF_MEMORY;
    };
    @memcpy(out_buf[0..bytes.len], bytes);
    op.* = @intCast(@intFromPtr(out_buf));
    ol.* = @intCast(bytes.len);
    return ZTOK_OK;
}

// --- misc exports ------------------------------------------------------

// Return a `*const u8` to the version C-string and its length so JS can
// `TextDecoder.decode(new Uint8Array(memory.buffer, ptr, len))`.
// (Wasm32 has no concept of NUL-terminated strings at the JS boundary —
// passing length separately is the clean play.)
export fn ztok_version_ptr() [*]const u8 {
    return VERSION.ptr;
}

export fn ztok_version_len() u32 {
    return @intCast(VERSION.len);
}

// --- panic / OOB plumbing ---------------------------------------------
//
// `wasm32-freestanding` has no default panic handler. Without one,
// `@panic` and integer overflow / bounds checks would emit a reference
// to `std.builtin.default_panic` which itself transitively pulls in
// `std.debug.print` + stderr writer + ... none of which exist
// freestanding. So we install our own minimal panic that just traps
// (the wasm `unreachable` instruction). The browser will surface this
// as a `RuntimeError: unreachable executed`.
//
// In ReleaseSmall / ReleaseFast the runtime safety checks are off, so
// this handler effectively never fires unless the user's input
// triggers an actual `@panic("foo")`. ReleaseSafe (which we use for
// tests) would call this from any overflow, so we KEEP the build
// optimization at ReleaseSmall and document that JS-facing wasm runs
// without the runtime checks.
pub const Panic = struct {
    pub fn call(_: []const u8, _: ?usize) noreturn {
        @trap();
    }
    pub fn sentinelMismatch(_: anytype, _: anytype) noreturn {
        @trap();
    }
    pub fn unwrapError(_: anyerror) noreturn {
        @trap();
    }
    pub fn outOfBounds(_: usize, _: usize) noreturn {
        @trap();
    }
    pub fn startGreaterThanEnd(_: usize, _: usize) noreturn {
        @trap();
    }
    pub fn inactiveUnionField(_: anytype, _: anytype) noreturn {
        @trap();
    }
    pub fn sliceCastLenRemainder(_: usize) noreturn {
        @trap();
    }
    pub fn reachedUnreachable() noreturn {
        @trap();
    }
    pub fn unwrapNull() noreturn {
        @trap();
    }
    pub fn castToNull() noreturn {
        @trap();
    }
    pub fn incorrectAlignment() noreturn {
        @trap();
    }
    pub fn invalidErrorCode() noreturn {
        @trap();
    }
    pub fn castTruncatedData() noreturn {
        @trap();
    }
    pub fn negativeToUnsigned() noreturn {
        @trap();
    }
    pub fn integerOverflow() noreturn {
        @trap();
    }
    pub fn shlOverflow() noreturn {
        @trap();
    }
    pub fn shrOverflow() noreturn {
        @trap();
    }
    pub fn divideByZero() noreturn {
        @trap();
    }
    pub fn exactDivisionRemainder() noreturn {
        @trap();
    }
    pub fn integerPartOutOfBounds() noreturn {
        @trap();
    }
    pub fn corruptSwitch() noreturn {
        @trap();
    }
    pub fn shiftRhsTooBig() noreturn {
        @trap();
    }
    pub fn invalidEnumValue() noreturn {
        @trap();
    }
    pub fn forLenMismatch() noreturn {
        @trap();
    }
    pub fn copyLenMismatch() noreturn {
        @trap();
    }
    pub fn memcpyAlias() noreturn {
        @trap();
    }
    pub fn noreturnReturned() noreturn {
        @trap();
    }
    pub const messages = std.debug.SimplePanic.messages;
};
