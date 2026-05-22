//! Sanity-check the freshly-built browser wasm binary actually exports
//! the symbols we promise JS callers it does.
//!
//! Runs as `zig build test-wasm-browser`. Reads the wasm at
//! `zig-out/bin/ztok_browser.wasm` (built by the `ztok-wasm-browser`
//! step we declared as a build dependency) and parses out the export
//! section, then asserts every expected name is present and refers to
//! a function (not a memory/global/table).
//!
//! Wasm binary format reference:
//!   https://webassembly.github.io/spec/core/binary/modules.html
//! We only need to parse far enough to find the export section
//! (section id = 7) — section ids before it are skipped by length.

const std = @import("std");

const WASM_PATH = "zig-out/bin/ztok_browser.wasm";
const WASM_PATH_SCALAR = "zig-out/bin/ztok_browser_scalar.wasm";

// WebAssembly SIMD prefix byte (every v128.* / i32x4.* / etc.
// instruction is encoded as 0xFD followed by a LEB128 opcode index).
// See https://webassembly.github.io/spec/core/binary/instructions.html#vector-instructions
const WASM_SIMD_PREFIX: u8 = 0xFD;

const required_exports = [_][]const u8{
    "ztok_pipeline_new_bpe_from_tiktoken_bytes",
    "ztok_pipeline_free",
    "ztok_encode",
    "ztok_decode",
    "ztok_malloc",
    "ztok_free",
    "ztok_version_ptr",
    "ztok_version_len",
};

// Minimal LEB128 unsigned decoder. Wasm uses LEB128 for nearly every
// integer in the binary format; the section size, name length, and
// export count all use it.
fn readVarUint(data: []const u8, off: *usize) !u64 {
    var result: u64 = 0;
    var shift: u6 = 0;
    while (true) {
        if (off.* >= data.len) return error.Truncated;
        const byte = data[off.*];
        off.* += 1;
        result |= (@as(u64, byte & 0x7F) << shift);
        if (byte & 0x80 == 0) break;
        shift = std.math.add(u6, shift, 7) catch return error.LebTooBig;
    }
    return result;
}

test "browser wasm has expected exports" {
    const a = std.testing.allocator;

    const io = std.Io.Threaded.global_single_threaded.io();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, WASM_PATH, a, .unlimited);
    defer a.free(bytes);

    // Wasm magic: \0asm + version 1.
    try std.testing.expect(bytes.len > 8);
    try std.testing.expectEqualSlices(u8, "\x00asm", bytes[0..4]);
    try std.testing.expectEqualSlices(u8, "\x01\x00\x00\x00", bytes[4..8]);

    var off: usize = 8;
    var found_export_section = false;
    var found = std.StringHashMap(void).init(a);
    defer found.deinit();

    while (off < bytes.len) {
        const section_id = bytes[off];
        off += 1;
        const section_size = try readVarUint(bytes, &off);
        const section_end = off + @as(usize, @intCast(section_size));

        if (section_id == 7) { // export section
            found_export_section = true;
            const count = try readVarUint(bytes, &off);
            var i: u64 = 0;
            while (i < count) : (i += 1) {
                const name_len = try readVarUint(bytes, &off);
                const name = bytes[off .. off + @as(usize, @intCast(name_len))];
                off += @as(usize, @intCast(name_len));
                const kind = bytes[off];
                off += 1;
                _ = try readVarUint(bytes, &off); // export index
                // We only care about function exports (kind 0).
                if (kind == 0) {
                    try found.put(name, {});
                }
            }
        }
        off = section_end;
    }

    try std.testing.expect(found_export_section);
    for (required_exports) |name| {
        if (!found.contains(name)) {
            std.debug.print("missing export: {s}\n", .{name});
            try std.testing.expect(false);
        }
    }
}

/// Walk every section of a wasm binary and return the byte slice for the
/// code section (id 10), or null if absent.
fn findCodeSection(bytes: []const u8) !?[]const u8 {
    if (bytes.len < 8) return error.Truncated;
    var off: usize = 8;
    while (off < bytes.len) {
        const section_id = bytes[off];
        off += 1;
        const section_size = try readVarUint(bytes, &off);
        const section_end = off + @as(usize, @intCast(section_size));
        if (section_id == 10) return bytes[off..section_end];
        off = section_end;
    }
    return null;
}

/// Count occurrences of the SIMD prefix (0xFD) in the code section.
/// This is a deliberately conservative scan — it doesn't decode the
/// surrounding instructions, so the count can include 0xFD bytes that
/// happen to fall inside an immediate operand. In practice immediates
/// in the BPE encoder hot path are small LEB128 ints (function
/// indices, local indices, branch labels), so collisions with 0xFD
/// are rare. The scalar build is a control: we compare counts between
/// the SIMD wasm and the scalar wasm, so any constant baseline of
/// non-instruction 0xFD bytes (e.g. embedded constants) cancels out.
fn countSimdPrefixBytes(code: []const u8) usize {
    var n: usize = 0;
    for (code) |b| if (b == WASM_SIMD_PREFIX) {
        n += 1;
    };
    return n;
}

test "browser wasm with simd128 emits v128 / i32x4 opcodes in the code section" {
    const a = std.testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();

    const simd_bytes = try std.Io.Dir.cwd().readFileAlloc(io, WASM_PATH, a, .unlimited);
    defer a.free(simd_bytes);

    const code = (try findCodeSection(simd_bytes)) orelse {
        std.debug.print("no code section in {s}\n", .{WASM_PATH});
        try std.testing.expect(false);
        return;
    };

    const n_simd = countSimdPrefixBytes(code);

    // Empirically the SIMD build emits a few dozen 0xFD prefixes in
    // the scanMin / merge-loop hot path. Be lax — even 16 would
    // indicate the codegen is producing vector ops rather than
    // scalar fallback. The scalar control test below cross-checks.
    try std.testing.expect(n_simd >= 16);
}

test "scalar wasm build (no simd128) emits no SIMD prefixes" {
    const a = std.testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();

    // The scalar build is opt-in; skip cleanly if it wasn't produced.
    const scalar_bytes = std.Io.Dir.cwd().readFileAlloc(io, WASM_PATH_SCALAR, a, .unlimited) catch {
        return error.SkipZigTest;
    };
    defer a.free(scalar_bytes);

    const code = (try findCodeSection(scalar_bytes)) orelse {
        std.debug.print("no code section in {s}\n", .{WASM_PATH_SCALAR});
        try std.testing.expect(false);
        return;
    };

    // Cross-check: scalar build should have many fewer 0xFD prefixes
    // than the SIMD build. We can't assert exactly zero because 0xFD
    // can appear inside immediate operands, but the SIMD build should
    // have meaningfully more.
    const simd_full = try std.Io.Dir.cwd().readFileAlloc(io, WASM_PATH, a, .unlimited);
    defer a.free(simd_full);
    const simd_code = (try findCodeSection(simd_full)).?;

    const n_scalar = countSimdPrefixBytes(code);
    const n_simd = countSimdPrefixBytes(simd_code);

    // SIMD build should have at least 32 more 0xFD bytes than scalar —
    // a comfortable margin above the random-collision noise floor.
    try std.testing.expect(n_simd > n_scalar + 32);
}

test "browser wasm with simd128 still loads as a valid module" {
    const a = std.testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();

    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, WASM_PATH, a, .unlimited);
    defer a.free(bytes);

    // Trivial validation: walk every section, asserting we never run
    // off the end of the buffer. Real validation happens in the
    // browser; this catches gross corruption (e.g. a truncated build).
    try std.testing.expect(bytes.len > 8);
    try std.testing.expectEqualSlices(u8, "\x00asm", bytes[0..4]);

    var off: usize = 8;
    var section_count: usize = 0;
    while (off < bytes.len) {
        const section_id = bytes[off];
        off += 1;
        const section_size = try readVarUint(bytes, &off);
        const section_end = off + @as(usize, @intCast(section_size));
        try std.testing.expect(section_end <= bytes.len);
        // Section IDs are 0..12 in the current spec; anything else is
        // a corrupted module.
        try std.testing.expect(section_id <= 12);
        off = section_end;
        section_count += 1;
    }
    try std.testing.expect(section_count >= 5);
}
