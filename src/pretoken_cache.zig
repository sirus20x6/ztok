//! Bounded worker-local cache for pretoken -> token-id sequences.
//!
//! Dataset-scale tokenization encounters the same short words and punctuation
//! runs millions of times. Re-running BPE for each occurrence dominates the
//! cold file path even though the mapping is deterministic. This direct-mapped
//! cache stores the common case inline: a pretoken of at most 15 bytes that
//! encodes to at most four IDs. Longer keys/results simply bypass the cache.
//!
//! The cache is deliberately caller-owned and never consulted by ordinary
//! `Pipeline.encode`; chunked/file callers opt in with an explicit entry count.

const std = @import("std");
const builtin = @import("builtin");
const TokenId = @import("token.zig").TokenId;

pub const max_key_bytes: usize = 15;
pub const max_token_ids: usize = 4;
pub const spill_flag: u64 = 0x80;
const max_spill_token_ids: usize = 0x7F;
const max_inline_token_id: TokenId = 0x00FF_FFFF;

const Entry = extern struct {
    key_lo: u64 align(32),
    key_hi: u64,
    value: u64,
    extension: u64,
};

comptime {
    if (@sizeOf(Entry) != 32) @compileError("pretoken cache Entry must remain 32 bytes");
    if (@alignOf(Entry) != 32) @compileError("pretoken cache Entry must remain 32-byte aligned");
}

pub const Stats = struct {
    hits: u64 = 0,
    home_hits: u64 = 0,
    displaced_hits: u64 = 0,
    misses: u64 = 0,
    inserts: u64 = 0,
    bypasses: u64 = 0,
};

pub const PreparedKey = struct {
    lo: u64 = 0,
    hi: u64 = 0,
    hash: u64 = 0,
};

/// Register-sized result of probing only the key's home cache-line pair.
/// On a miss, `value` and `extension` are unspecified and must be ignored.
pub const ProbeResult = struct {
    value: u64,
    extension: u64,
    found: bool,
};

/// A chunk-scoped snapshot of the cache table. Keeping the base and folded
/// pair mask outside the emit loop prevents repeated loads through `Cache`.
/// The current cache never grows, so a view remains valid until deinit.
pub const ProbeView = struct {
    base: [*]const Entry,
    pair_mask: usize,

    inline fn pairPtr(self: ProbeView, hash: u64) [*]const Entry {
        return self.base + (@as(usize, hash) & self.pair_mask);
    }

    pub inline fn prefetch(self: ProbeView, prepared: PreparedKey) void {
        if (prepared.hi == 0) return;
        @prefetch(self.pairPtr(prepared.hash), .{ .rw = .read, .locality = 3, .cache = .data });
    }

    /// Probe the overwhelmingly common displacement-0/1 case without a
    /// data-dependent entry-pointer load. Both values are loaded from the
    /// already-prefetched line, then selected with integer masks.
    pub inline fn probePair(self: ProbeView, prepared: PreparedKey) ProbeResult {
        const pair = self.pairPtr(prepared.hash);
        const e0 = pair[0];
        const e1 = pair[1];
        const m0 = e0.key_lo == prepared.lo and e0.key_hi == prepared.hi;
        const m1 = e1.key_lo == prepared.lo and e1.key_hi == prepared.hi;
        var value = e1.value;
        var extension = e1.extension;
        if (comptime builtin.cpu.arch == .x86_64) {
            // LLVM otherwise tends to select an Entry pointer and perform a
            // dependent value load. Select already-loaded register values so
            // both halves of the prefetched cache line issue in parallel.
            asm volatile ("test %[match], %[match]\n\t" ++
                    "cmovne %[value0], %[value]\n\t" ++
                    "cmovne %[extension0], %[extension]"
                : [value] "+r" (value),
                  [extension] "+r" (extension),
                : [match] "r" (@as(u64, @intFromBool(m0))),
                  [value0] "r" (e0.value),
                  [extension0] "r" (e0.extension),
                : .{ .cc = true });
        } else {
            const select0 = @as(u64, @bitCast(-%@as(i64, @intFromBool(m0))));
            value = (e0.value & select0) | (e1.value & ~select0);
            extension = (e0.extension & select0) | (e1.extension & ~select0);
        }
        return .{
            .value = value,
            .extension = extension,
            .found = prepared.hi != 0 and (m0 or m1),
        };
    }
};

pub const Cache = struct {
    allocator: std.mem.Allocator,
    spill_allocator: std.mem.Allocator,
    entries: []Entry,
    spill_ids: std.ArrayList(TokenId) = .empty,
    mask: usize,
    len: usize = 0,
    stats: Stats = .{},
    stats_enabled: bool = false,

    /// `requested_entries` is rounded up to a power of two. Zero creates a
    /// disabled cache and performs no allocation.
    pub fn init(allocator: std.mem.Allocator, requested_entries: usize) !Cache {
        if (requested_entries == 0) return .{
            .allocator = allocator,
            .spill_allocator = allocator,
            .entries = &.{},
            .mask = 0,
            .len = 0,
        };
        const entry_capacity = std.math.ceilPowerOfTwo(usize, requested_entries) catch
            return error.CacheTooLarge;
        const table_bytes = entry_capacity * @sizeOf(Entry);
        const huge_page_bytes = 2 * 1024 * 1024;
        const entry_allocator = if (table_bytes >= huge_page_bytes)
            std.heap.page_allocator
        else
            allocator;
        const entries = if (table_bytes >= huge_page_bytes)
            try std.heap.page_allocator.alignedAlloc(
                Entry,
                .fromByteUnits(huge_page_bytes),
                entry_capacity,
            )
        else
            try allocator.alloc(Entry, entry_capacity);
        // Request THP before first touch; zeroing first faults 4 KiB pages and
        // leaves random probes dominated by dTLB misses on multi-megabyte
        // tables. Failure is only a lost optimization.
        if (builtin.os.tag == .linux and table_bytes >= huge_page_bytes) {
            const bytes = std.mem.sliceAsBytes(entries);
            _ = std.os.linux.madvise(bytes.ptr, bytes.len, std.os.linux.MADV.HUGEPAGE);
        }
        @memset(entries, std.mem.zeroes(Entry));
        return .{
            .allocator = entry_allocator,
            .spill_allocator = allocator,
            .entries = entries,
            .mask = entry_capacity - 1,
            .len = 0,
        };
    }

    pub fn deinit(self: *Cache) void {
        self.spill_ids.deinit(self.spill_allocator);
        if (self.entries.len > 0) self.allocator.free(self.entries);
        self.* = .{
            .allocator = self.allocator,
            .spill_allocator = self.spill_allocator,
            .entries = &.{},
            .mask = 0,
            .len = 0,
        };
    }

    pub inline fn get(self: *Cache, key: []const u8, out: []TokenId) ?usize {
        const prepared = prepare(key);
        return self.getPrepared(prepared, out);
    }

    pub inline fn getPrepared(self: *Cache, prepared: PreparedKey, out: []TokenId) ?usize {
        if (self.entries.len == 0 or prepared.hi == 0) {
            if (self.stats_enabled) self.stats.bypasses += 1;
            return null;
        }
        const home_idx = @as(usize, prepared.hash) & self.mask & ~@as(usize, 1);
        var idx = home_idx;
        while (true) {
            const e0 = &self.entries[idx];
            const e1 = &self.entries[idx + 1];
            const hit = if (e0.key_lo == prepared.lo and e0.key_hi == prepared.hi)
                e0
            else if (e1.key_lo == prepared.lo and e1.key_hi == prepared.hi)
                e1
            else
                null;
            if (hit) |entry| {
                if (self.stats_enabled) {
                    self.stats.hits += 1;
                    if (idx == home_idx) {
                        self.stats.home_hits += 1;
                    } else {
                        self.stats.displaced_hits += 1;
                    }
                }
                const count: usize = @intCast(entry.value & 0x7F);
                std.debug.assert(count >= 1 and out.len >= count);
                if ((entry.value & spill_flag) != 0) {
                    const offset: usize = @intCast(entry.extension);
                    std.debug.assert(offset + count <= self.spill_ids.items.len);
                    @memcpy(out[0..count], self.spill_ids.items[offset .. offset + count]);
                    return count;
                }
                std.debug.assert(count <= max_token_ids);
                out[0] = @truncate((entry.value >> 8) & 0x00FF_FFFF);
                if (count >= 2) out[1] = @truncate((entry.value >> 32) & 0x00FF_FFFF);
                if (count >= 3) out[2] = @truncate(entry.extension);
                if (count >= 4) out[3] = @truncate(entry.extension >> 32);
                return count;
            }
            // Empty key terminates the pair walk. All live packed keys carry
            // a non-zero length byte, so (0,0) is an unambiguous sentinel.
            if ((e0.key_lo | e0.key_hi) == 0 or (e1.key_lo | e1.key_hi) == 0) break;
            idx = (idx + 2) & self.mask;
        }
        if (self.stats_enabled) self.stats.misses += 1;
        return null;
    }

    pub inline fn put(self: *Cache, key: []const u8, ids: []const TokenId) void {
        self.putPrepared(prepare(key), ids);
    }

    pub inline fn putPrepared(self: *Cache, prepared: PreparedKey, ids: []const TokenId) void {
        if (self.entries.len == 0 or prepared.hi == 0 or ids.len == 0 or
            ids.len > max_spill_token_ids)
        {
            return;
        }
        // Stop at 75% load so every unsuccessful pair walk is guaranteed to
        // find an empty slot quickly. The cache is a bounded accelerator;
        // bypassing new long-tail entries is preferable to rehashing during
        // a timed file encode.
        if ((self.len + 1) * 4 > self.entries.len * 3) return;
        var idx = @as(usize, prepared.hash) & self.mask & ~@as(usize, 1);
        while (true) : (idx = (idx + 2) & self.mask) {
            if ((self.entries[idx].key_lo | self.entries[idx].key_hi) == 0) break;
            if ((self.entries[idx + 1].key_lo | self.entries[idx + 1].key_hi) == 0) {
                idx += 1;
                break;
            }
        }
        self.insertPreparedAt(idx, prepared, ids);
    }

    inline fn insertPreparedAt(
        self: *Cache,
        idx: usize,
        prepared: PreparedKey,
        ids: []const TokenId,
    ) void {
        const entry = &self.entries[idx];
        entry.* = std.mem.zeroes(Entry);
        entry.key_lo = prepared.lo;
        entry.key_hi = prepared.hi;
        const inline_value = ids.len <= max_token_ids and
            ids[0] <= max_inline_token_id and
            (ids.len < 2 or ids[1] <= max_inline_token_id);
        if (inline_value) {
            entry.value = @as(u64, @intCast(ids.len)) |
                (@as(u64, ids[0]) << 8) |
                (if (ids.len >= 2) @as(u64, ids[1]) << 32 else 0);
            entry.extension = (if (ids.len >= 3) @as(u64, ids[2]) else 0) |
                (if (ids.len >= 4) @as(u64, ids[3]) << 32 else 0);
        } else {
            const offset = self.spill_ids.items.len;
            self.spill_ids.appendSlice(self.spill_allocator, ids) catch {
                entry.* = std.mem.zeroes(Entry);
                return;
            };
            entry.value = spill_flag | @as(u64, @intCast(ids.len));
            entry.extension = @intCast(offset);
        }
        self.len += 1;
        if (self.stats_enabled) self.stats.inserts += 1;
    }

    pub fn capacity(self: *const Cache) usize {
        return self.entries.len;
    }

    pub inline fn probeView(self: *const Cache) ProbeView {
        std.debug.assert(self.entries.len != 0);
        return .{
            .base = self.entries.ptr,
            .pair_mask = self.mask & ~@as(usize, 1),
        };
    }

    pub inline fn recordHomeHit(self: *Cache) void {
        if (self.stats_enabled) {
            self.stats.hits += 1;
            self.stats.home_hits += 1;
        }
    }

    pub fn enableStats(self: *Cache, enabled: bool) void {
        self.stats_enabled = enabled;
    }

    pub inline fn prefetchL2(self: *const Cache, prepared: PreparedKey) void {
        if (self.entries.len == 0 or prepared.hi == 0) return;
        const entry = &self.entries[@as(usize, prepared.hash) & self.mask & ~@as(usize, 1)];
        @prefetch(entry, .{ .rw = .read, .locality = 2, .cache = .data });
    }

    pub inline fn prefetchL1(self: *const Cache, prepared: PreparedKey) void {
        if (self.entries.len == 0 or prepared.hi == 0) return;
        const entry = &self.entries[@as(usize, prepared.hash) & self.mask & ~@as(usize, 1)];
        @prefetch(entry, .{ .rw = .read, .locality = 3, .cache = .data });
    }

    const PackedKey = struct { lo: u64, hi: u64 };

    pub inline fn prepare(key: []const u8) PreparedKey {
        if (key.len == 0 or key.len > max_key_bytes) return .{};
        const key_packed = packKey(key);
        return .{
            .lo = key_packed.lo,
            .hi = key_packed.hi,
            .hash = keyHash(key_packed.lo, key_packed.hi),
        };
    }

    inline fn packKey(key: []const u8) PackedKey {
        std.debug.assert(key.len > 0 and key.len <= max_key_bytes);
        var lo: u64 = 0;
        var hi: u64 = 0;

        // One unaligned load is substantially cheaper than a generic
        // variable-length hash/copy for the overwhelmingly common short key.
        // Keep it inside the current page so a key near an allocation's end
        // can never touch an unmapped page.
        const page_offset = @intFromPtr(key.ptr) & 4095;
        if (page_offset <= 4096 - 16) {
            // Reconstruct from the address after the page-boundary proof.
            // This intentionally communicates the guarded over-read to Zig's
            // compile-time bounds checker for short string literals.
            const p0: *align(1) const u64 = @ptrFromInt(@intFromPtr(key.ptr));
            const p1: *align(1) const u64 = @ptrFromInt(@intFromPtr(key.ptr) + 8);
            lo = p0.*;
            hi = p1.*;
        } else {
            var bytes: [16]u8 = [_]u8{0} ** 16;
            @memcpy(bytes[0..key.len], key);
            lo = std.mem.readInt(u64, bytes[0..8], .little);
            hi = std.mem.readInt(u64, bytes[8..16], .little);
        }

        // Mask the bytes beyond the slice without a variable-length memset.
        // Both dynamic shifts lower to one scalar instruction; the two halves
        // avoid compiler-rt u128 shift helpers on x86-64.
        if (key.len < 8) {
            const bits: u6 = @intCast(key.len * 8);
            lo &= (@as(u64, 1) << bits) - 1;
            hi = 0;
        } else if (key.len == 8) {
            hi = 0;
        } else {
            const bits: u6 = @intCast((key.len - 8) * 8);
            hi &= (@as(u64, 1) << bits) - 1;
        }
        // Length occupies the top byte, making every live key non-zero and
        // distinguishing equal prefixes of different lengths.
        hi |= @as(u64, @intCast(key.len)) << 56;
        return .{ .lo = lo, .hi = hi };
    }

    inline fn keyHash(lo: u64, hi: u64) u64 {
        if (comptime builtin.cpu.arch == .x86_64 and
            std.Target.x86.featureSetHas(builtin.cpu.features, .sse4_2))
        {
            var hash: u64 = 0;
            asm volatile ("crc32q %[lo], %[hash]"
                : [hash] "+r" (hash),
                : [lo] "r" (lo),
            );
            asm volatile ("crc32q %[hi], %[hash]"
                : [hash] "+r" (hash),
                : [hi] "r" (hi),
            );
            return hash;
        }
        var hash = (lo ^ std.math.rotr(u64, hi, 25)) *% 0x9E3779B97F4A7C15;
        hash ^= hash >> 32;
        return hash;
    }
};

test "short pretoken cache hit, replacement, and bypass" {
    var cache = try Cache.init(std.testing.allocator, 3);
    defer cache.deinit();
    cache.enableStats(true);
    try std.testing.expectEqual(@as(usize, 4), cache.capacity());

    var out: [4]TokenId = undefined;
    try std.testing.expect(cache.get(" hello", &out) == null);
    cache.put(" hello", &.{ 17, 29 });
    const n = cache.get(" hello", &out).?;
    try std.testing.expectEqualSlices(TokenId, &.{ 17, 29 }, out[0..n]);
    try std.testing.expect(cache.get(" world", &out) == null);

    var spill_out: [8]TokenId = undefined;
    cache.put("abcdef", &.{ 1, 2, 3, 4, 5, 6 });
    const spill_n = cache.get("abcdef", &spill_out).?;
    try std.testing.expectEqualSlices(TokenId, &.{ 1, 2, 3, 4, 5, 6 }, spill_out[0..spill_n]);

    const long = "this key is longer than fifteen bytes";
    cache.put(long, &.{1});
    try std.testing.expect(cache.get(long, &out) == null);
    try std.testing.expectEqual(@as(u64, 2), cache.stats.hits);
}
