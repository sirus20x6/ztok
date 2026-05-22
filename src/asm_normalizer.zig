//! x86-64 "normalized instruction route" classifier.
//!
//! ztok's tokenization overlay lets each token carry side-channels. Two of
//! those channels — `opcode_class` and `operand_class` — are meant to hold
//! *normalized* instruction classes for binary / disassembly inputs. The
//! normalization masks out immediates, displacements and absolute addresses
//! so that instructions that differ only in those bytes collapse to the same
//! pair of classes. For example both
//!
//!     48 8B 05 11 22 33 44      ; mov rax, [rip+0x44332211]
//!     48 8B 05 AA BB CC DD      ; mov rax, [rip+0xddccbbaa]
//!
//! decode to opcode_class=`mov` / operand_class=`reg_mem_riprel`, regardless of
//! the four displacement bytes.
//!
//! This module is intentionally self-contained: it imports only `std` so that
//! `zig test src/asm_normalizer.zig` runs standalone. The orchestrator wires
//! the classes into the encoder elsewhere; nothing here touches other ztok
//! modules.
//!
//! Scope. We decode a *defensible subset* of x86-64 — enough to walk a code
//! stream instruction-by-instruction without ever reading past the slice or
//! looping forever:
//!
//!   * legacy prefixes (0x66/0x67/segment/lock/rep) and the REX prefix block
//!   * common one-byte opcodes: MOV r/m<->r and MOV r/m,imm; PUSH/POP r64;
//!     ADD/SUB/XOR/CMP r/m<->r and the 0x80/0x81/0x83 group with imm; LEA;
//!     CALL rel32 (E8); JMP rel32 (E9) / rel8 (EB); Jcc short (70..7F) and
//!     Jcc near (0F 80..8F); RET (C3) and RET imm16 (C2); NOP (90); INT3 (CC)
//!   * full ModRM / SIB / displacement / immediate length computation
//!
//! Anything we don't recognize advances exactly one byte and is reported as
//! `Class.unknown` with `Operand.none`. That guarantees forward progress: the
//! walk always terminates and never indexes out of bounds.

const std = @import("std");

/// Normalized mnemonic family (the `opcode_class` side-channel value).
///
/// Backed by `u32` because the overlay channel stores a `u32`; callers cast
/// with `@intFromEnum`.
pub const Class = enum(u32) {
    unknown = 0,
    mov,
    lea,
    push,
    pop,
    add,
    sub,
    xor,
    cmp,
    call,
    jmp,
    jcc,
    ret,
    nop,
    int3,
};

/// Normalized addressing form (the `operand_class` side-channel value).
///
/// Immediates / displacements / absolute targets are *not* encoded here — that
/// is the whole point of normalization. Two instructions with the same family
/// and the same `Operand` are considered the same "route".
pub const Operand = enum(u32) {
    none = 0,
    /// register, register (ModRM mod==11)
    reg_reg,
    /// register <-> memory, general ModRM memory form (mod 00/01/10, not RIP)
    reg_mem,
    /// register <-> RIP-relative memory (mod==00, rm==101, no SIB)
    reg_mem_riprel,
    /// register / memory with an immediate source
    reg_imm,
    /// single register operand (push/pop r64, etc.)
    reg,
    /// 32-bit relative branch target (call/jmp/jcc near)
    rel32,
    /// 8-bit relative branch target (jmp short / jcc short)
    rel8,
    /// 16-bit immediate operand with no register (ret imm16)
    imm16,
};

/// Result of decoding a single instruction.
pub const Decoded = struct {
    /// Total length of the decoded instruction in bytes. Always >= 1 and never
    /// larger than the input slice (the decoder clamps and degrades to
    /// `unknown` rather than read past the end).
    len: usize,
    /// `@intFromEnum` of a `Class`.
    opcode_class: u32,
    /// `@intFromEnum` of an `Operand`.
    operand_class: u32,

    pub fn class(self: Decoded) Class {
        return @enumFromInt(self.opcode_class);
    }
    pub fn operand(self: Decoded) Operand {
        return @enumFromInt(self.operand_class);
    }
};

fn make(len: usize, c: Class, o: Operand) Decoded {
    return .{
        .len = len,
        .opcode_class = @intFromEnum(c),
        .operand_class = @intFromEnum(o),
    };
}

/// A single byte we couldn't decode. One byte of progress, no operand.
fn unknownByte() Decoded {
    return make(1, .unknown, .none);
}

const Rex = struct {
    present: bool = false,
    w: bool = false, // REX.W — 64-bit operand size
};

/// Compute the number of bytes consumed by a ModRM byte and everything that
/// trails it that belongs to the *addressing* part: the optional SIB byte and
/// the displacement. Does NOT include any immediate (the caller adds that).
///
/// Returns the consumed length (modrm + sib + disp) starting at `modrm`, and
/// the normalized `Operand` addressing form. `addr32` reflects a 0x67 prefix
/// (32-bit address size) which only changes disp width for the 16-bit corner —
/// in 64-bit mode disp is 8 or 32 bits regardless, so it is informational here.
///
/// `avail` is how many bytes remain in the buffer starting at the modrm byte.
/// If the addressing bytes would overrun, we report the form but the caller
/// detects the overrun via the returned length vs `avail`.
const ModRmInfo = struct {
    /// bytes consumed for modrm + sib + disp
    len: usize,
    form: Operand,
};

/// Full ModRM decode that also accounts for the SIB base==101 special case,
/// which requires looking at the SIB byte. `rest` is the slice starting at the
/// ModRM byte. Returns null if the addressing bytes would overrun `rest`.
fn decodeModRmFull(rest: []const u8) ?ModRmInfo {
    if (rest.len < 1) return null;
    const modrm = rest[0];
    const mod: u2 = @intCast(modrm >> 6);
    const rm: u3 = @intCast(modrm & 0b111);

    if (mod == 0b11) return .{ .len = 1, .form = .reg_reg };

    var len: usize = 1; // modrm
    var form: Operand = .reg_mem;
    var has_sib = false;
    var sib_base: u3 = 0;

    if (rm == 0b100) {
        // SIB byte required
        if (rest.len < 2) return null;
        has_sib = true;
        sib_base = @intCast(rest[1] & 0b111);
        len += 1;
    }

    switch (mod) {
        0b00 => {
            if (rm == 0b101) {
                // RIP-relative disp32 (no SIB on this rm)
                len += 4;
                form = .reg_mem_riprel;
            } else if (has_sib and sib_base == 0b101) {
                // SIB with no base register -> disp32 follows
                len += 4;
            }
            // else: no displacement
        },
        0b01 => len += 1, // disp8
        0b10 => len += 4, // disp32
        0b11 => unreachable,
    }

    if (len > rest.len) return null;
    return .{ .len = len, .form = form };
}

/// Decode a single instruction at the start of `bytes`.
///
/// Never reads past `bytes`. Always returns `len >= 1` (so a buffer walk makes
/// progress) and `len <= bytes.len`. On any malformed / truncated / unknown
/// sequence it falls back to a single `unknown` byte.
pub fn next(bytes: []const u8) Decoded {
    if (bytes.len == 0) return make(0, .unknown, .none);

    var i: usize = 0;
    var rex: Rex = .{};
    var two_byte = false;

    // ---- legacy prefixes -------------------------------------------------
    // Operand-size (66), address-size (67), segment overrides, lock, rep/repne.
    // We consume them but, for classification, treat them as transparent.
    while (i < bytes.len) : (i += 1) {
        switch (bytes[i]) {
            0x66, 0x67, 0xF0, 0xF2, 0xF3, 0x2E, 0x36, 0x3E, 0x26, 0x64, 0x65 => {},
            else => break,
        }
    }
    if (i >= bytes.len) return unknownByte();

    // ---- REX prefix ------------------------------------------------------
    if (bytes[i] >= 0x40 and bytes[i] <= 0x4F) {
        rex.present = true;
        rex.w = (bytes[i] & 0b1000) != 0;
        i += 1;
        if (i >= bytes.len) return unknownByte();
    }

    var op = bytes[i];
    i += 1;

    // ---- two-byte opcode escape -----------------------------------------
    if (op == 0x0F) {
        if (i >= bytes.len) return unknownByte();
        two_byte = true;
        op = bytes[i];
        i += 1;
    }

    // `rest` is the slice starting right after the (final) opcode byte.
    const rest = bytes[i..];

    if (two_byte) {
        // Jcc near: 0F 80..8F + rel32
        if (op >= 0x80 and op <= 0x8F) {
            const total = i + 4;
            if (total > bytes.len) return unknownByte();
            return make(total, .jcc, .rel32);
        }
        // Unknown two-byte opcode: degrade to a single byte (the 0x0F) so the
        // walk re-syncs on the next byte rather than guessing a length.
        return unknownByte();
    }

    // ---- one-byte opcode space ------------------------------------------
    switch (op) {
        // MOV r/m,r and r,r/m  (88/89 store, 8A/8B load)
        0x88, 0x89, 0x8A, 0x8B => return modRmInsn(i, rest, .mov, 0),
        // LEA r,m  (always memory form)
        0x8D => return modRmInsn(i, rest, .lea, 0),

        // ADD/SUB/XOR/CMP  r/m<->r families (Eb/Ev forms)
        0x00, 0x01, 0x02, 0x03 => return modRmInsn(i, rest, .add, 0),
        0x28, 0x29, 0x2A, 0x2B => return modRmInsn(i, rest, .sub, 0),
        0x30, 0x31, 0x32, 0x33 => return modRmInsn(i, rest, .xor, 0),
        0x38, 0x39, 0x3A, 0x3B => return modRmInsn(i, rest, .cmp, 0),

        // ADD/SUB/XOR/CMP eAX, imm  (no ModRM; 04/2C/34/3C take imm8,
        // 05/2D/35/3D take imm16/32 by op-size). Reg is implicit eAX.
        0x04, 0x2C, 0x34, 0x3C => {
            const fam = aluFamily(op);
            const total = i + 1; // imm8
            if (total > bytes.len) return unknownByte();
            return make(total, fam, .reg_imm);
        },
        0x05, 0x2D, 0x35, 0x3D => {
            const fam = aluFamily(op);
            const imm: usize = 4; // imm32 in 64-bit mode (66h would make it 2)
            const total = i + imm;
            if (total > bytes.len) return unknownByte();
            return make(total, fam, .reg_imm);
        },

        // Immediate group 1: 80/81/83 /digit  (ADD/OR/ADC/SBB/AND/SUB/XOR/CMP)
        // The /digit in ModRM.reg picks the family; we map the common ones and
        // fall back to ADD-family classification for the rest (still a valid
        // route: r/m, imm). 80 -> imm8, 81 -> imm32, 83 -> imm8 (sign-extended).
        0x80, 0x81, 0x83 => {
            if (rest.len < 1) return unknownByte();
            const info = decodeModRmFull(rest) orelse return unknownByte();
            const fam = group1Family(rest[0]);
            const imm_len: usize = if (op == 0x81) 4 else 1;
            const total = i + info.len + imm_len;
            if (total > bytes.len) return unknownByte();
            // Operand is r/m + imm regardless of mod -> reg_imm.
            return make(total, fam, .reg_imm);
        },

        // MOV r/m, imm  (C6 -> imm8, C7 -> imm32)
        0xC6, 0xC7 => {
            const info = decodeModRmFull(rest) orelse return unknownByte();
            const imm_len: usize = if (op == 0xC7) 4 else 1;
            const total = i + info.len + imm_len;
            if (total > bytes.len) return unknownByte();
            return make(total, .mov, .reg_imm);
        },

        // MOV r64, imm64 / r32, imm32  (B8..BF +rd). REX.W -> imm64.
        0xB8, 0xB9, 0xBA, 0xBB, 0xBC, 0xBD, 0xBE, 0xBF => {
            const imm_len: usize = if (rex.w) 8 else 4;
            const total = i + imm_len;
            if (total > bytes.len) return unknownByte();
            return make(total, .mov, .reg_imm);
        },

        // PUSH r64 / POP r64  (50..57 push, 58..5F pop)
        0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57 => return make(i, .push, .reg),
        0x58, 0x59, 0x5A, 0x5B, 0x5C, 0x5D, 0x5E, 0x5F => return make(i, .pop, .reg),

        // CALL rel32
        0xE8 => {
            const total = i + 4;
            if (total > bytes.len) return unknownByte();
            return make(total, .call, .rel32);
        },
        // JMP rel32
        0xE9 => {
            const total = i + 4;
            if (total > bytes.len) return unknownByte();
            return make(total, .jmp, .rel32);
        },
        // JMP rel8
        0xEB => {
            const total = i + 1;
            if (total > bytes.len) return unknownByte();
            return make(total, .jmp, .rel8);
        },

        // Jcc short  70..7F + rel8
        0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x7B, 0x7C, 0x7D, 0x7E, 0x7F => {
            const total = i + 1;
            if (total > bytes.len) return unknownByte();
            return make(total, .jcc, .rel8);
        },

        // RET near
        0xC3 => return make(i, .ret, .none),
        // RET imm16
        0xC2 => {
            const total = i + 2;
            if (total > bytes.len) return unknownByte();
            return make(total, .ret, .imm16);
        },

        // NOP
        0x90 => return make(i, .nop, .none),
        // INT3
        0xCC => return make(i, .int3, .none),

        else => return unknownByte(),
    }
}

/// Helper for opcodes that are `opcode + ModRM [+ SIB + disp]` with no
/// immediate. `after_op` is the absolute index just past the opcode byte;
/// `rest` is `bytes[after_op..]`. Returns the full instruction `Decoded`.
fn modRmInsn(after_op: usize, rest: []const u8, c: Class, extra_imm: usize) Decoded {
    const info = decodeModRmFull(rest) orelse return unknownByte();
    // decodeModRmFull already bounds-checked info.len against rest.len, and
    // after_op == bytes.len - rest.len, so after_op + info.len <= bytes.len.
    // Current callers pass extra_imm == 0; the parameter exists for completeness.
    return make(after_op + info.len + extra_imm, c, info.form);
}

fn aluFamily(op: u8) Class {
    return switch (op & 0xF8) {
        0x00 => .add,
        0x28 => .sub,
        0x30 => .xor,
        0x38 => .cmp,
        else => .add,
    };
}

/// Map the /digit field of an 80/81/83 group-1 ModRM byte to a family. Only
/// ADD(0)/SUB(5)/XOR(6)/CMP(7) are distinguished; OR/ADC/SBB/AND fall back to
/// ADD-family (they are all "r/m, imm" arithmetic routes — the operand class
/// is what matters for normalization).
fn group1Family(modrm: u8) Class {
    const digit: u3 = @intCast((modrm >> 3) & 0b111);
    return switch (digit) {
        5 => .sub,
        6 => .xor,
        7 => .cmp,
        else => .add,
    };
}

/// Walk an entire buffer, invoking `visit` once per decoded instruction.
/// Guaranteed to terminate: `next` always returns `len >= 1` for non-empty
/// input. `visit` receives the byte offset and the `Decoded` result.
pub fn walk(bytes: []const u8, comptime Ctx: type, ctx: Ctx, comptime visit: fn (Ctx, usize, Decoded) void) void {
    var off: usize = 0;
    while (off < bytes.len) {
        const d = next(bytes[off..]);
        const step = if (d.len == 0) 1 else d.len;
        visit(ctx, off, d);
        off += step;
    }
}

/// Convenience: count how many instructions a buffer decodes into.
pub fn countInstructions(bytes: []const u8) usize {
    var off: usize = 0;
    var n: usize = 0;
    while (off < bytes.len) {
        const d = next(bytes[off..]);
        off += if (d.len == 0) 1 else d.len;
        n += 1;
    }
    return n;
}

// ===========================================================================
// Tests
// ===========================================================================

const testing = std.testing;

fn expectInsn(bytes: []const u8, want_len: usize, want_class: Class, want_op: Operand) !void {
    const d = next(bytes);
    try testing.expectEqual(want_len, d.len);
    try testing.expectEqual(want_class, d.class());
    try testing.expectEqual(want_op, d.operand());
}

test "empty input" {
    const d = next(&.{});
    try testing.expectEqual(@as(usize, 0), d.len);
    try testing.expectEqual(Class.unknown, d.class());
}

test "MOV rax, rbx reg_reg (48 89 d8)" {
    try expectInsn(&.{ 0x48, 0x89, 0xd8 }, 3, .mov, .reg_reg);
}

test "MOV rax, [rip+disp] reg_mem_riprel (48 8b 05 ..)" {
    try expectInsn(&.{ 0x48, 0x8b, 0x05, 0x11, 0x22, 0x33, 0x44 }, 7, .mov, .reg_mem_riprel);
}

test "RIP-relative class is invariant to displacement bytes" {
    const a = next(&.{ 0x48, 0x8b, 0x05, 0x11, 0x22, 0x33, 0x44 });
    const b = next(&.{ 0x48, 0x8b, 0x05, 0xAA, 0xBB, 0xCC, 0xDD });
    try testing.expectEqual(a.opcode_class, b.opcode_class);
    try testing.expectEqual(a.operand_class, b.operand_class);
    try testing.expectEqual(a.len, b.len);
}

test "CALL rel32 (e8 ..) invariant to target" {
    try expectInsn(&.{ 0xe8, 0x00, 0x00, 0x00, 0x00 }, 5, .call, .rel32);
    const a = next(&.{ 0xe8, 0x00, 0x00, 0x00, 0x00 });
    const b = next(&.{ 0xe8, 0xde, 0xad, 0xbe, 0xef });
    try testing.expectEqual(a.opcode_class, b.opcode_class);
    try testing.expectEqual(a.operand_class, b.operand_class);
}

test "JMP rel32 (e9) and rel8 (eb)" {
    try expectInsn(&.{ 0xe9, 0x01, 0x02, 0x03, 0x04 }, 5, .jmp, .rel32);
    try expectInsn(&.{ 0xeb, 0xfe }, 2, .jmp, .rel8);
}

test "RET (c3) and RET imm16 (c2)" {
    try expectInsn(&.{0xc3}, 1, .ret, .none);
    try expectInsn(&.{ 0xc2, 0x08, 0x00 }, 3, .ret, .imm16);
}

test "JE rel8 (74 10) is jcc rel8, invariant to target" {
    try expectInsn(&.{ 0x74, 0x10 }, 2, .jcc, .rel8);
    const a = next(&.{ 0x74, 0x10 });
    const b = next(&.{ 0x74, 0xF0 });
    try testing.expectEqual(a.opcode_class, b.opcode_class);
    try testing.expectEqual(a.operand_class, b.operand_class);
}

test "Jcc near (0f 84 ..) is jcc rel32" {
    try expectInsn(&.{ 0x0f, 0x84, 0x00, 0x00, 0x00, 0x00 }, 6, .jcc, .rel32);
}

test "PUSH/POP r64 (50 / 5b)" {
    try expectInsn(&.{0x50}, 1, .push, .reg);
    try expectInsn(&.{0x5b}, 1, .pop, .reg);
    // with REX.B the encoding gains a prefix byte
    try expectInsn(&.{ 0x41, 0x50 }, 2, .push, .reg);
}

test "NOP (90) and INT3 (cc)" {
    try expectInsn(&.{0x90}, 1, .nop, .none);
    try expectInsn(&.{0xcc}, 1, .int3, .none);
}

test "LEA rax, [rip+disp] (48 8d 05 ..)" {
    try expectInsn(&.{ 0x48, 0x8d, 0x05, 0x00, 0x00, 0x00, 0x00 }, 7, .lea, .reg_mem_riprel);
}

test "ADD/SUB/XOR/CMP reg_reg forms" {
    try expectInsn(&.{ 0x48, 0x01, 0xd8 }, 3, .add, .reg_reg); // add rax, rbx
    try expectInsn(&.{ 0x48, 0x29, 0xd8 }, 3, .sub, .reg_reg); // sub rax, rbx
    try expectInsn(&.{ 0x48, 0x31, 0xd8 }, 3, .xor, .reg_reg); // xor rax, rbx
    try expectInsn(&.{ 0x48, 0x39, 0xd8 }, 3, .cmp, .reg_reg); // cmp rax, rbx
}

test "XOR eax,eax (31 c0) no REX" {
    try expectInsn(&.{ 0x31, 0xc0 }, 2, .xor, .reg_reg);
}

test "group1 imm: add r/m,imm8 via 83 /0 (48 83 c0 08)" {
    // add rax, 8 ; invariant to imm
    const a = next(&.{ 0x48, 0x83, 0xc0, 0x08 });
    const b = next(&.{ 0x48, 0x83, 0xc0, 0x7f });
    try testing.expectEqual(@as(usize, 4), a.len);
    try testing.expectEqual(Class.add, a.class());
    try testing.expectEqual(Operand.reg_imm, a.operand());
    try testing.expectEqual(a.opcode_class, b.opcode_class);
    try testing.expectEqual(a.operand_class, b.operand_class);
}

test "group1 imm: cmp r/m,imm32 via 81 /7 (48 81 f8 .. ..)" {
    try expectInsn(&.{ 0x48, 0x81, 0xf8, 0x00, 0x01, 0x00, 0x00 }, 7, .cmp, .reg_imm);
}

test "group1 family selection sub/xor (83 /5, 83 /6)" {
    try expectInsn(&.{ 0x48, 0x83, 0xe8, 0x01 }, 4, .sub, .reg_imm); // sub rax,1
    try expectInsn(&.{ 0x48, 0x83, 0xf0, 0x01 }, 4, .xor, .reg_imm); // xor rax,1
}

test "MOV r64, imm64 (48 b8 .. x8)" {
    const bytes = [_]u8{ 0x48, 0xb8, 0, 1, 2, 3, 4, 5, 6, 7 };
    try expectInsn(&bytes, 10, .mov, .reg_imm);
}

test "MOV r32, imm32 (b8 .. x4) no REX" {
    try expectInsn(&.{ 0xb8, 0x00, 0x00, 0x00, 0x00 }, 5, .mov, .reg_imm);
}

test "MOV r/m, imm32 (c7 /0): c7 c0 .. -> reg_imm" {
    try expectInsn(&.{ 0xc7, 0xc0, 0x00, 0x00, 0x00, 0x00 }, 6, .mov, .reg_imm);
}

test "MOV with SIB no-disp (48 8b 04 24) -> reg_mem" {
    // mov rax, [rsp]  (modrm=04 -> rm=100 SIB; sib=24 base=rsp)
    try expectInsn(&.{ 0x48, 0x8b, 0x04, 0x24 }, 4, .mov, .reg_mem);
}

test "MOV with SIB base==101 disp32 (48 8b 04 25 .. ..) -> reg_mem" {
    // mov rax, [disp32] absolute via SIB no-base
    try expectInsn(&.{ 0x48, 0x8b, 0x04, 0x25, 0x00, 0x00, 0x00, 0x00 }, 8, .mov, .reg_mem);
}

test "MOV disp8 mem (48 8b 40 08) -> reg_mem" {
    // mov rax, [rax+8]
    try expectInsn(&.{ 0x48, 0x8b, 0x40, 0x08 }, 4, .mov, .reg_mem);
}

test "operand-size prefix is transparent (66 prefix on mov)" {
    // 66 89 d8 -> mov ax, bx ; still mov reg_reg, prefix consumed
    try expectInsn(&.{ 0x66, 0x89, 0xd8 }, 3, .mov, .reg_reg);
}

test "unknown byte advances exactly one ( d6 is undefined)" {
    try expectInsn(&.{ 0xd6, 0xd6 }, 1, .unknown, .none);
}

test "truncated MOV rip-rel degrades to unknown one byte" {
    // missing displacement bytes -> we must not read past the slice
    const d = next(&.{ 0x48, 0x8b, 0x05, 0x11 });
    try testing.expectEqual(Class.unknown, d.class());
    try testing.expectEqual(@as(usize, 1), d.len);
}

test "truncated CALL degrades to unknown one byte" {
    const d = next(&.{ 0xe8, 0x00, 0x00 });
    try testing.expectEqual(Class.unknown, d.class());
    try testing.expectEqual(@as(usize, 1), d.len);
}

test "lone REX prefix at end -> unknown, no overrun" {
    const d = next(&.{0x48});
    try testing.expectEqual(Class.unknown, d.class());
    try testing.expectEqual(@as(usize, 1), d.len);
}

test "lone 0f escape at end -> unknown" {
    const d = next(&.{0x0f});
    try testing.expectEqual(Class.unknown, d.class());
    try testing.expectEqual(@as(usize, 1), d.len);
}

test "walk terminates and counts a small program" {
    // push rbp; mov rbp,rsp; xor eax,eax; pop rbp; ret
    const prog = [_]u8{
        0x55, // push rbp
        0x48, 0x89, 0xe5, // mov rbp, rsp
        0x31, 0xc0, // xor eax, eax
        0x5d, // pop rbp
        0xc3, // ret
    };
    try testing.expectEqual(@as(usize, 5), countInstructions(&prog));
}

test "walk over garbage never loops and consumes everything" {
    const junk = [_]u8{ 0xff, 0xfe, 0x06, 0x07, 0x0e, 0x9b };
    // Every byte here is unknown-ish; count must equal byte count (1 each)
    // except 0x9b is itself a (consumed) prefix-like? It's FWAIT, unknown to us.
    var off: usize = 0;
    var iters: usize = 0;
    while (off < junk.len) {
        const d = next(junk[off..]);
        try testing.expect(d.len >= 1);
        off += d.len;
        iters += 1;
        try testing.expect(iters <= junk.len); // guaranteed progress
    }
    try testing.expectEqual(junk.len, off);
}

test "Decoded.class/operand helpers round-trip" {
    const d = make(3, .mov, .reg_reg);
    try testing.expectEqual(Class.mov, d.class());
    try testing.expectEqual(Operand.reg_reg, d.operand());
    try testing.expectEqual(@as(u32, @intFromEnum(Class.mov)), d.opcode_class);
}

test "immediate-only differences never change class across families" {
    // mov r/m,imm32 with two different immediates
    const m1 = next(&.{ 0xc7, 0xc0, 0xde, 0xad, 0xbe, 0xef });
    const m2 = next(&.{ 0xc7, 0xc0, 0x00, 0x00, 0x00, 0x00 });
    try testing.expectEqual(m1.opcode_class, m2.opcode_class);
    try testing.expectEqual(m1.operand_class, m2.operand_class);
    try testing.expectEqual(m1.len, m2.len);

    // disp8 differences on a memory mov
    const d1 = next(&.{ 0x48, 0x8b, 0x40, 0x08 });
    const d2 = next(&.{ 0x48, 0x8b, 0x40, 0x7f });
    try testing.expectEqual(d1.opcode_class, d2.opcode_class);
    try testing.expectEqual(d1.operand_class, d2.operand_class);
}
