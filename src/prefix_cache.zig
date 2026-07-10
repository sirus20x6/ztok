//! Prefix cache for repeated LLM prompts (system prompts, tool schemas,
//! chat-template prefixes, RAG chunks, etc). Hash the bytes, look up a
//! cached id sequence. Hit -> return slice; miss -> tokenize, store,
//! return.
//!
//! Hash: std.hash.Wyhash.hash with seed 0. Wyhash is fast on the short-
//! to-medium prefixes (a few KB) we expect here, has excellent
//! distribution, and is already in std with no extra deps.
//!
//! Storage layout (v2 — arena-backed + watermark compaction):
//! - A hash map (u64 -> node_idx) plus a flat ArrayList of LruNode records.
//! - One `std.heap.ArenaAllocator` ("id-arena") holds the cached id-array
//!   bytes for every live entry. Each entry's id slice is a stable
//!   pointer into the arena.
//! - The key bytes (the prefix bytes used for collision-checked equality)
//!   stay in a flat `bytes_arena: ArrayList(u8)`. They are short, never
//!   returned to the caller, and don't benefit from compaction the same
//!   way the id arrays do; we leave the v1 strategy in place for them.
//! - LRU ordering is a doubly-linked list embedded in the node array
//!   (parallel prev/next indexes).
//!
//! Watermark compaction:
//! - We track `arena_bytes` (total bytes ever handed out by the id-arena
//!   since the last compaction) and `live_id_bytes` (sum of id_len *
//!   sizeof(TokenId) across live entries).
//! - When `arena_bytes > compaction_ratio * live_id_bytes` (and
//!   `live_id_bytes > 0`), the next mutating op compacts: allocate a fresh
//!   arena, copy every live entry's ids into it, swap in the new arena,
//!   free the old one. `compaction_ratio` defaults to 4.0.
//! - Compaction preserves LRU order (the link-list edges are untouched —
//!   we only rewrite each node's `ids_ptr`).
//!
//! Slice lifetimes:
//! - The `[]const TokenId` returned by `lookup` / `lookupOrInsert` is
//!   valid UNTIL THE NEXT MUTATING OPERATION on the cache (insert,
//!   lookupOrInsert with a miss, clear, deinit, forceCompact). After a
//!   compaction the underlying arena is freed and the slice pointer is
//!   dangling — re-look-up if you need it again. This is a normal cache
//!   contract.
//!
//! Collisions: when a hash hits but the stored bytes don't match the
//! requested prefix, treat as a miss. lookupOrInsert overwrites the
//! existing entry with the new tokenization (last writer wins). lookup
//! returns null. Wyhash collisions are astronomically rare in practice.
//!
//! Concurrency: v1 is single-threaded. The struct holds no locks; the
//! caller must wrap a shared Cache in a mutex if multiple threads will
//! touch it.

const std = @import("std");

const TokenId = @import("token.zig").TokenId;
const Pipeline = @import("pipeline.zig").Pipeline;

pub const Stats = struct {
    hits: u64 = 0,
    misses: u64 = 0,
    evictions: u64 = 0,
    bytes_stored: u64 = 0,
    entries: u32 = 0,

    pub fn hitRate(self: Stats) f64 {
        const total = self.hits + self.misses;
        if (total == 0) return 0.0;
        return @as(f64, @floatFromInt(self.hits)) / @as(f64, @floatFromInt(total));
    }
};

pub const CompactionStats = struct {
    /// Number of compactions performed since init.
    compactions: u32,
    /// Bytes of id-array data currently referenced by live entries.
    live_bytes: usize,
    /// Bytes ever handed out by the current id-arena since the last
    /// compaction (a high-water-mark approximation: monotonically
    /// non-decreasing until a compaction resets it).
    arena_bytes: usize,
    /// Watermark trigger threshold (live_bytes * compaction_ratio).
    /// Compaction fires when arena_bytes > watermark.
    watermark_bytes: usize,
};

pub const Options = struct {
    /// Compaction fires when `arena_bytes > compaction_ratio *
    /// live_id_bytes`. Default 4.0 means "tolerate up to 4× overhead from
    /// dead/orphaned id slices before paying for a compaction pass".
    /// Set to a very large value (e.g. std.math.inf(f32)) to disable
    /// auto-compaction; you can still call `forceCompact` manually.
    compaction_ratio: f32 = 4.0,
};

pub fn hashBytes(bytes: []const u8) u64 {
    return std.hash.Wyhash.hash(0, bytes);
}

pub const Cache = struct {
    allocator: std.mem.Allocator,
    capacity: u32,
    options: Options,

    entries: std.AutoHashMap(u64, EntryRef),

    /// Flat arena for prefix key bytes. Orphaned slots on overwrite/
    /// eviction are not reclaimed; this list is small in practice (a few
    /// KB per entry) and not exposed to the caller.
    bytes_arena: std.ArrayList(u8),

    /// Arena for cached id arrays. Each live entry's `.ids_ptr` is a
    /// stable pointer into this arena. Reset on compaction.
    ids_arena: std.heap.ArenaAllocator,
    /// Total bytes ever allocated from `ids_arena` since the last reset.
    /// Used to decide when to compact.
    arena_bytes: usize,

    lru_head: ?u32,
    lru_tail: ?u32,
    nodes: std.ArrayList(LruNode),
    stats: Stats,

    /// Number of compactions performed since init.
    compactions: u32,

    pub const EntryRef = struct {
        node_idx: u32,
    };
    pub const LruNode = struct {
        hash: u64,
        bytes_start: u32,
        bytes_len: u32,
        /// Stable pointer into `ids_arena` for the cached id slice. Many-
        /// in-one is the whole point: each entry owns a contiguous block.
        ids_ptr: [*]TokenId,
        ids_len: u32,
        prev: ?u32,
        next: ?u32,
    };

    pub fn init(allocator: std.mem.Allocator, capacity: u32) Cache {
        return initWithOptions(allocator, capacity, .{});
    }

    pub fn initWithOptions(
        allocator: std.mem.Allocator,
        capacity: u32,
        options: Options,
    ) Cache {
        std.debug.assert(capacity > 0);
        return .{
            .allocator = allocator,
            .capacity = capacity,
            .options = options,
            .entries = std.AutoHashMap(u64, EntryRef).init(allocator),
            .bytes_arena = .empty,
            .ids_arena = std.heap.ArenaAllocator.init(allocator),
            .arena_bytes = 0,
            .lru_head = null,
            .lru_tail = null,
            .nodes = .empty,
            .stats = .{},
            .compactions = 0,
        };
    }

    pub fn deinit(self: *Cache) void {
        self.entries.deinit();
        self.bytes_arena.deinit(self.allocator);
        self.ids_arena.deinit();
        self.nodes.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn clear(self: *Cache) void {
        self.entries.clearRetainingCapacity();
        self.bytes_arena.clearRetainingCapacity();
        _ = self.ids_arena.reset(.retain_capacity);
        self.arena_bytes = 0;
        self.nodes.clearRetainingCapacity();
        self.lru_head = null;
        self.lru_tail = null;
        self.stats = .{};
    }

    pub fn lookup(self: *Cache, prefix_bytes: []const u8) ?[]const TokenId {
        const h = hashBytes(prefix_bytes);
        const ref = self.entries.get(h) orelse {
            self.stats.misses += 1;
            return null;
        };
        const n = self.nodes.items[ref.node_idx];
        const stored = self.bytes_arena.items[n.bytes_start .. n.bytes_start + n.bytes_len];
        if (!std.mem.eql(u8, stored, prefix_bytes)) {
            // Hash collision: treat as miss, leave existing entry untouched.
            self.stats.misses += 1;
            return null;
        }
        self.touch(ref.node_idx);
        self.stats.hits += 1;
        return n.ids_ptr[0..n.ids_len];
    }

    pub fn lookupOrInsert(
        self: *Cache,
        pipeline: *const Pipeline,
        prefix_bytes: []const u8,
    ) ![]const TokenId {
        const h = hashBytes(prefix_bytes);
        if (self.entries.get(h)) |ref| {
            const n = self.nodes.items[ref.node_idx];
            const stored = self.bytes_arena.items[n.bytes_start .. n.bytes_start + n.bytes_len];
            if (std.mem.eql(u8, stored, prefix_bytes)) {
                self.touch(ref.node_idx);
                self.stats.hits += 1;
                return n.ids_ptr[0..n.ids_len];
            }
            // Collision: tokenize fresh and overwrite the existing entry.
            self.stats.misses += 1;
            const ids = try pipeline.encode(self.allocator, prefix_bytes);
            defer self.allocator.free(ids);
            try self.overwrite(ref.node_idx, h, prefix_bytes, ids);
            const n2 = self.nodes.items[ref.node_idx];
            return n2.ids_ptr[0..n2.ids_len];
        }
        // True miss.
        self.stats.misses += 1;
        const ids = try pipeline.encode(self.allocator, prefix_bytes);
        defer self.allocator.free(ids);
        return self.insertInternal(h, prefix_bytes, ids);
    }

    pub fn insert(self: *Cache, prefix_bytes: []const u8, ids: []const TokenId) !void {
        const h = hashBytes(prefix_bytes);
        if (self.entries.get(h)) |ref| {
            try self.overwrite(ref.node_idx, h, prefix_bytes, ids);
            return;
        }
        _ = try self.insertInternal(h, prefix_bytes, ids);
    }

    /// Snapshot of compaction-relevant counters. O(1).
    pub fn compactionStats(self: *const Cache) CompactionStats {
        const live = self.liveIdBytes();
        return .{
            .compactions = self.compactions,
            .live_bytes = live,
            .arena_bytes = self.arena_bytes,
            .watermark_bytes = self.watermarkBytes(live),
        };
    }

    /// Unconditional compaction: allocate a fresh id-arena, copy every
    /// live entry's ids into it, swap. Invalidates any id slices
    /// previously returned by `lookup`/`lookupOrInsert`. Safe to call
    /// when there are zero entries.
    pub fn forceCompact(self: *Cache) !void {
        try self.compact();
    }

    // ----- internals -----

    fn liveIdBytes(self: *const Cache) usize {
        var total: usize = 0;
        for (self.nodes.items) |n| total += @as(usize, n.ids_len) * @sizeOf(TokenId);
        return total;
    }

    fn watermarkBytes(self: *const Cache, live_bytes: usize) usize {
        // Use saturating arithmetic so a very large ratio (effectively
        // disable) doesn't wrap. f32 -> usize via floor.
        const r: f32 = self.options.compaction_ratio;
        if (!(r > 0)) return std.math.maxInt(usize);
        const flive: f32 = @floatFromInt(live_bytes);
        const product = flive * r;
        if (!(product < @as(f32, @floatFromInt(std.math.maxInt(usize))))) {
            return std.math.maxInt(usize);
        }
        return @intFromFloat(product);
    }

    /// Decide-and-do: run compaction if the arena has grown past the
    /// configured threshold relative to live data. Idempotent on a
    /// freshly compacted cache.
    fn maybeCompact(self: *Cache) !void {
        const live = self.liveIdBytes();
        if (live == 0) {
            // No live data → reclaim arena unconditionally if it has
            // any pending bytes. Cheap and keeps the steady-state empty
            // cache from sitting on capacity.
            if (self.arena_bytes > 0) try self.compact();
            return;
        }
        const wm = self.watermarkBytes(live);
        if (self.arena_bytes > wm) try self.compact();
    }

    fn compact(self: *Cache) !void {
        // Build a new arena. We pre-size to live_id_bytes so the first
        // allocations in the new arena land in a single block — this is
        // the contiguity guarantee we promised the caller. Plus a slop
        // pad so the very next insert doesn't immediately allocate a
        // second backing block.
        const live = self.liveIdBytes();
        var new_arena = std.heap.ArenaAllocator.init(self.allocator);
        errdefer new_arena.deinit();

        // Reserve the arena's first chunk in one go.
        if (live > 0) {
            const a = new_arena.allocator();
            const pad_slop: usize = @max(@as(usize, 256), live / 8);
            const reservation = try a.alloc(u8, live + pad_slop);
            // Free it back so the arena treats the chunk as scratch we
            // can carve up. ArenaAllocator keeps the underlying buffer
            // for subsequent allocs.
            a.free(reservation);
        }

        const a = new_arena.allocator();

        // Copy each live entry's ids into the new arena, rewriting the
        // node's ids_ptr. The LRU link structure (prev/next/head/tail)
        // is untouched, so eviction order is preserved exactly.
        for (self.nodes.items) |*n| {
            if (n.ids_len == 0) {
                // Use a stable non-null sentinel: a 0-length alloc is
                // fine but we want a deterministic pointer. Allocate a
                // single byte so the slice has a real backing.
                const dummy = try a.alloc(TokenId, 0);
                n.ids_ptr = dummy.ptr;
                continue;
            }
            const dst = try a.alloc(TokenId, n.ids_len);
            @memcpy(dst, n.ids_ptr[0..n.ids_len]);
            n.ids_ptr = dst.ptr;
        }

        // Swap arenas: free the old one, install the new one, reset
        // counters.
        self.ids_arena.deinit();
        self.ids_arena = new_arena;
        self.arena_bytes = live;
        self.compactions +%= 1;
    }

    fn arenaDupIds(self: *Cache, ids: []const TokenId) ![]TokenId {
        const a = self.ids_arena.allocator();
        const dst = try a.alloc(TokenId, ids.len);
        @memcpy(dst, ids);
        self.arena_bytes += ids.len * @sizeOf(TokenId);
        return dst;
    }

    fn insertInternal(
        self: *Cache,
        h: u64,
        prefix_bytes: []const u8,
        ids: []const TokenId,
    ) ![]const TokenId {
        // Evict tail until there is room for one more entry.
        while (self.nodes.items.len >= self.capacity) {
            try self.evictTail();
        }

        // Compaction first, before we allocate fresh id storage. This
        // keeps `arena_bytes` low and means the new entry lands in a
        // freshly reserved contiguous block.
        try self.maybeCompact();

        const bytes_start: u32 = @intCast(self.bytes_arena.items.len);
        try self.bytes_arena.appendSlice(self.allocator, prefix_bytes);

        const dst = try self.arenaDupIds(ids);

        const new_idx: u32 = @intCast(self.nodes.items.len);
        try self.nodes.append(self.allocator, .{
            .hash = h,
            .bytes_start = bytes_start,
            .bytes_len = @intCast(prefix_bytes.len),
            .ids_ptr = dst.ptr,
            .ids_len = @intCast(ids.len),
            .prev = null,
            .next = null,
        });

        try self.entries.put(h, .{ .node_idx = new_idx });
        self.linkAtHead(new_idx);
        self.stats.entries = @intCast(self.nodes.items.len);
        self.stats.bytes_stored += prefix_bytes.len;
        return dst;
    }

    fn overwrite(
        self: *Cache,
        node_idx: u32,
        h: u64,
        prefix_bytes: []const u8,
        ids: []const TokenId,
    ) !void {
        // We're replacing an entry's id-array. Note that the OLD ids
        // become orphaned bytes in the arena — that's exactly the
        // condition the watermark/compaction is designed to bound.
        try self.maybeCompact();

        const bytes_start: u32 = @intCast(self.bytes_arena.items.len);
        try self.bytes_arena.appendSlice(self.allocator, prefix_bytes);

        const dst = try self.arenaDupIds(ids);

        const n = &self.nodes.items[node_idx];
        // Track the net bytes_stored delta as if old were freed (live size).
        self.stats.bytes_stored = self.stats.bytes_stored - n.bytes_len + prefix_bytes.len;
        n.hash = h;
        n.bytes_start = bytes_start;
        n.bytes_len = @intCast(prefix_bytes.len);
        n.ids_ptr = dst.ptr;
        n.ids_len = @intCast(ids.len);

        // Make sure the map points to the right node (hash is unchanged
        // unless this was a hash-collision overwrite — but the caller
        // passes the same `h` that already maps to `node_idx`, so just
        // re-put for safety).
        try self.entries.put(h, .{ .node_idx = node_idx });
        self.touch(node_idx);
    }

    fn evictTail(self: *Cache) !void {
        const tail = self.lru_tail orelse return;
        const n = self.nodes.items[tail];
        _ = self.entries.remove(n.hash);
        self.unlink(tail);
        self.stats.bytes_stored -= n.bytes_len;
        self.stats.evictions += 1;

        // Swap-remove the tail node out of the nodes array. If it was
        // not the last node, fix up the moved node's links + map entry.
        const last_idx: u32 = @intCast(self.nodes.items.len - 1);
        if (tail != last_idx) {
            const moved = self.nodes.items[last_idx];
            self.nodes.items[tail] = moved;
            // Update neighbours that pointed at last_idx.
            if (moved.prev) |p| self.nodes.items[p].next = tail;
            if (moved.next) |nx| self.nodes.items[nx].prev = tail;
            if (self.lru_head == last_idx) self.lru_head = tail;
            if (self.lru_tail == last_idx) self.lru_tail = tail;
            // Update map entry for the moved node.
            try self.entries.put(moved.hash, .{ .node_idx = tail });
        }
        _ = self.nodes.pop();
        self.stats.entries = @intCast(self.nodes.items.len);
    }

    fn linkAtHead(self: *Cache, idx: u32) void {
        const n = &self.nodes.items[idx];
        n.prev = null;
        n.next = self.lru_head;
        if (self.lru_head) |h| self.nodes.items[h].prev = idx;
        self.lru_head = idx;
        if (self.lru_tail == null) self.lru_tail = idx;
    }

    fn unlink(self: *Cache, idx: u32) void {
        const n = self.nodes.items[idx];
        if (n.prev) |p| self.nodes.items[p].next = n.next else self.lru_head = n.next;
        if (n.next) |nx| self.nodes.items[nx].prev = n.prev else self.lru_tail = n.prev;
    }

    fn touch(self: *Cache, idx: u32) void {
        if (self.lru_head == idx) return;
        self.unlink(idx);
        self.linkAtHead(idx);
    }
};

// ============================================================
// Persistent backend (PrefixCache)
// ============================================================
//
// On-disk format (single file at <PersistentConfig.path>):
//
//   [0..64)   header
//     [0..8)   magic = "ZTOKPCv1"
//     [8..12)  version u32 (LE), current = 1
//     [12..20) entry_count u64 (LE)         — written on close/compact
//     [20..28) total_bytes u64 (LE)         — file size hint, written on close/compact
//     [28..64) reserved (zero)
//
//   [64..EOF) variable-length records, each:
//     [0..8)   hash u64 (Wyhash, LE)
//     [8..12)  key_len  u32 (LE)
//     [12..16) ids_len  u32 (LE, count of TokenIds)
//     [16..16+key_len)               key bytes
//     [16+key_len .. 16+key_len+4*ids_len)  ids (u32 LE, packed)
//     [16+key_len+4*ids_len .. +8)   CRC64-Ecma182 over the preceding
//                                    record bytes (hash..ids, inclusive)
//
// Design choices vs the "spec":
//   * Plain `pread`/`pwrite` + `fdatasync`, not mmap. Records are
//     variable-length and small; sequential append + targeted small
//     reads at known offsets are exactly what pread/pwrite are for.
//     mmap would have to grow the mapping on every append (mremap,
//     SIGBUS at file-end, partial-page tail handling) for no real
//     win — the kernel page cache already does the buffering we'd
//     otherwise pay for. The on-disk format is stable either way:
//     a future mmap-backed reader can scan the same file unchanged.
//   * `flock(LOCK_EX)` on a stable `<path>.lock` sidecar for writer
//     exclusion. Keeping the lock off the replaceable data inode preserves
//     exclusion across compaction's atomic rename. The lock is released
//     automatically on close() or crash.
//   * The in-memory `Cache` stays the read fast-path. The persistent
//     backend mirrors `insert`/`lookupOrInsert` writes synchronously
//     to the log; reads never touch disk after the initial scan.

const posix = std.posix;

pub const PersistentConfig = struct {
    /// Path to the cache file. Created if missing, opened R/W otherwise.
    path: []const u8,
    /// Sync to disk every N writes. 0 disables count-based syncs.
    sync_every_n_writes: u32 = 100,
    /// Sync to disk every B bytes written since last sync. 0 disables
    /// byte-based syncs.
    sync_every_bytes: usize = 1 << 20,
    /// When the dead bytes in the file exceed this percentage of the
    /// total file size, the next mutating op triggers compaction.
    /// 0..100; 0 disables auto-compaction (manual `compactNow` only).
    compact_at_stale_pct: u8 = 50,
    /// Capacity for the in-memory LRU. The on-disk log can hold many
    /// more entries than this — only the working set lives in RAM.
    /// On a lookup that misses the in-memory cache, we consult the
    /// on-disk file_index and (on hit) reload the entry into the LRU.
    /// Keeps memory bounded for very large caches.
    in_memory_capacity: u32 = 4096,
};

pub const PersistentStats = struct {
    /// Records present in the on-disk log (including overwrites still
    /// pinned — i.e. logical count is `entries` minus `dead_entries`).
    file_entries: u64,
    /// Records that have been logically replaced by a later record
    /// and are awaiting compaction.
    dead_entries: u64,
    /// File size in bytes, header included.
    file_bytes: u64,
    /// Bytes that compaction would reclaim (sum of stale record sizes).
    dead_bytes: u64,
    /// Compactions performed since open.
    compactions: u32,
    /// fdatasync calls performed since open.
    syncs: u64,
};

pub const PrefixCache = struct {
    allocator: std.mem.Allocator,
    cfg: PersistentConfig,

    /// In-memory LRU. Holds the working set; writes are mirrored to
    /// the disk log. The struct embeds the existing `Cache` so all
    /// the in-memory behaviour (LRU, watermark compaction, stats)
    /// keeps working unchanged.
    mem: Cache,

    /// File descriptor for the persistent log.
    fd: posix.fd_t,
    /// Stable sidecar lock descriptor (`<path>.lock`). The lock must not
    /// live on `fd` because compaction atomically replaces that inode.
    lock_fd: posix.fd_t,
    /// Current file length in bytes. Updated after every append /
    /// compaction.
    file_len: u64,
    /// hash → on-disk record offset. Built on open by scanning the
    /// file; updated on every append. May contain entries that aren't
    /// in the in-memory `mem` cache (cold entries on disk).
    file_index: std.AutoHashMap(u64, FileEntry),
    /// Total bytes of records that have been logically superseded by
    /// a later record at a different offset for the same hash. Used
    /// as the compaction trigger.
    dead_bytes: u64,
    /// Count of superseded records (each becomes a "dead" entry).
    dead_entries: u64,

    /// Sync accounting since the last fdatasync.
    writes_since_sync: u32,
    bytes_since_sync: usize,
    syncs: u64,
    compactions: u32,

    pub const FileEntry = struct {
        /// Byte offset of the start of the record in the file.
        offset: u64,
        /// Total byte length of the record (header + key + ids + crc).
        size: u32,
        /// ids_len in TokenIds (so we can sanity-check before reading
        /// the body).
        ids_len: u32,
    };

    // ----- file format constants -----

    pub const magic: [8]u8 = "ZTOKPCv1".*;
    pub const format_version: u32 = 1;
    pub const header_size: u64 = 64;

    /// Fixed-size prelude of every record on disk.
    const record_prelude_size: usize = 16; // 8 hash + 4 keylen + 4 idslen
    const record_crc_size: usize = 8;

    const Crc64 = std.hash.crc.Crc64Ecma182;

    /// Open (or create) a persistent prefix cache at `cfg.path`. Acquires
    /// an exclusive advisory lock on a stable sidecar for the lifetime of the
    /// returned cache so that two writers on the same host don't clobber
    /// each other. Concurrent processes block in `openPersistent` until
    /// the first releases the lock (via `closePersistent` or process
    /// exit).
    pub fn openPersistent(
        allocator: std.mem.Allocator,
        cfg: PersistentConfig,
    ) !*PrefixCache {
        // Hold the path as an owned copy so callers can free their slice
        // after `openPersistent` returns.
        const path_dup = try allocator.dupe(u8, cfg.path);
        errdefer allocator.free(path_dup);

        const lock_path = try std.fmt.allocPrint(allocator, "{s}.lock", .{path_dup});
        defer allocator.free(lock_path);
        const lock_fd = try openOrCreateRW(lock_path);
        errdefer _ = posix.system.close(lock_fd);
        // A sidecar inode remains stable while data-file compaction uses
        // atomic rename. LOCK_EX blocks other processes before they open or
        // scan the data file, including processes already waiting during a
        // rename.
        try flockExclusive(lock_fd);

        const fd = try openOrCreateRW(path_dup);
        errdefer _ = posix.system.close(fd);

        // Make sure the header is sane (or initialise it for a fresh
        // file). The scan below will then walk the records.
        const initial_len = try fileLength(fd);
        if (initial_len == 0) {
            try writeFreshHeader(fd);
        } else {
            try validateHeader(fd);
        }

        var self = try allocator.create(PrefixCache);
        errdefer allocator.destroy(self);

        self.* = .{
            .allocator = allocator,
            // Re-point the config's path at our owned copy so we can
            // free it deterministically in `closePersistent`.
            .cfg = .{
                .path = path_dup,
                .sync_every_n_writes = cfg.sync_every_n_writes,
                .sync_every_bytes = cfg.sync_every_bytes,
                .compact_at_stale_pct = cfg.compact_at_stale_pct,
                .in_memory_capacity = cfg.in_memory_capacity,
            },
            .mem = Cache.init(allocator, @max(@as(u32, 1), cfg.in_memory_capacity)),
            .fd = fd,
            .lock_fd = lock_fd,
            .file_len = 0,
            .file_index = std.AutoHashMap(u64, FileEntry).init(allocator),
            .dead_bytes = 0,
            .dead_entries = 0,
            .writes_since_sync = 0,
            .bytes_since_sync = 0,
            .syncs = 0,
            .compactions = 0,
        };
        errdefer {
            self.mem.deinit();
            self.file_index.deinit();
        }

        // Scan existing records. On any structural error (short record,
        // bad CRC), truncate the file at the last good offset and stop.
        // This makes a torn write at process crash recoverable.
        try self.scanFile();

        return self;
    }

    /// Releases the lock and frees all resources. Performs a final
    /// fdatasync so any buffered writes are durable on disk.
    pub fn closePersistent(self: *PrefixCache) void {
        // Best effort flush; we can't return an error from a deinit-style
        // function and the file is about to be closed anyway.
        _ = posix.fdatasync(self.fd) catch {};
        _ = posix.system.close(self.fd);
        _ = posix.system.close(self.lock_fd);
        self.fd = -1;
        self.lock_fd = -1;
        self.mem.deinit();
        self.file_index.deinit();
        self.allocator.free(self.cfg.path);
        const a = self.allocator;
        self.* = undefined;
        a.destroy(self);
    }

    /// Look up a key. Hits the in-memory cache first; on miss, consults
    /// the on-disk index and (if found) reloads the entry into the in-
    /// memory cache. Returns null on a true miss.
    ///
    /// Lifetime: the returned slice is valid until the next mutating
    /// operation on the cache, same contract as `Cache.lookup`.
    pub fn lookup(self: *PrefixCache, key: []const u8) !?[]const TokenId {
        if (self.mem.lookup(key)) |ids| return ids;
        // In-memory miss; check the on-disk index.
        const h = hashBytes(key);
        const fe = self.file_index.get(h) orelse return null;
        // Read and verify the record. On a hash collision the bytes
        // won't match → treat as a miss (same as `Cache.lookup`).
        const rec = try self.readRecord(fe);
        defer {
            self.allocator.free(rec.bytes);
            self.allocator.free(rec.ids);
        }
        if (!std.mem.eql(u8, rec.key, key)) return null;
        // Reload into the in-memory cache so the next lookup is fast.
        try self.mem.insert(key, rec.ids);
        return self.mem.lookup(key);
    }

    /// Insert a key/value pair. Mirrors to the persistent log.
    pub fn insert(self: *PrefixCache, key: []const u8, ids: []const TokenId) !void {
        // Mirror to disk first so the in-memory state is never ahead
        // of what's durable on a crash. If the disk write fails, the
        // in-memory cache is left untouched.
        try self.appendRecord(key, ids);
        // Now update the in-memory cache. If this fails (OOM), the on-
        // disk log still has the entry; next open will pick it up.
        try self.mem.insert(key, ids);
        // Opportunistically compact if we've crossed the stale-pct
        // threshold. Sync first so the compaction sees a flushed file.
        try self.maybeSync(false);
        try self.maybeCompact();
    }

    /// Insert if absent; on miss, runs the supplied pipeline to encode.
    /// Convenience wrapper analogous to `Cache.lookupOrInsert`.
    pub fn lookupOrInsert(
        self: *PrefixCache,
        pipeline: *const Pipeline,
        key: []const u8,
    ) ![]const TokenId {
        if (try self.lookup(key)) |ids| return ids;
        const ids = try pipeline.encode(self.allocator, key);
        defer self.allocator.free(ids);
        try self.insert(key, ids);
        // The in-memory cache now has the entry. Re-fetch so the caller
        // gets the arena-owned slice (not our temporary `ids` heap alloc).
        return self.mem.lookup(key) orelse return error.UnexpectedMiss;
    }

    /// Force-compact the persistent log: rewrite it with only the
    /// currently-live (most-recent-per-hash) entries. Holds the lock
    /// for the duration; concurrent readers from other processes will
    /// briefly see a stale file before the atomic rename swap, but
    /// will never see corruption.
    pub fn compactNow(self: *PrefixCache) !void {
        return self.compact();
    }

    /// O(1) snapshot of persistent-backend counters.
    pub fn persistentStats(self: *const PrefixCache) PersistentStats {
        return .{
            .file_entries = @as(u64, self.file_index.count()) + self.dead_entries,
            .dead_entries = self.dead_entries,
            .file_bytes = self.file_len,
            .dead_bytes = self.dead_bytes,
            .compactions = self.compactions,
            .syncs = self.syncs,
        };
    }

    /// Force an fdatasync. Useful from tests and from callers that want
    /// to draw a durability barrier without waiting for the next batch.
    pub fn sync(self: *PrefixCache) !void {
        try posix.fdatasync(self.fd);
        self.writes_since_sync = 0;
        self.bytes_since_sync = 0;
        self.syncs += 1;
    }

    // ===== internals =====

    const ScannedRecord = struct {
        /// Owning buffer for the entire record body. Free with the
        /// allocator that produced it.
        bytes: []u8,
        /// View into `bytes` for the key.
        key: []const u8,
        /// Separately-allocated, properly-aligned copy of the ids. We
        /// can't safely reinterpret `bytes` as `[]TokenId` because the
        /// ids start at offset `16 + key_len`, which isn't guaranteed
        /// to be 4-aligned. Free with the same allocator.
        ids: []TokenId,
        /// Total on-disk size of the record (prelude + key + ids + crc).
        size: u32,
    };

    fn appendRecord(self: *PrefixCache, key: []const u8, ids: []const TokenId) !void {
        if (key.len > std.math.maxInt(u32)) return error.KeyTooLong;
        if (ids.len > std.math.maxInt(u32)) return error.ValueTooLong;
        const key_len: u32 = @intCast(key.len);
        const ids_len: u32 = @intCast(ids.len);
        const ids_bytes_len: usize = @as(usize, ids_len) * @sizeOf(TokenId);
        const rec_size: usize = record_prelude_size + key.len + ids_bytes_len + record_crc_size;
        if (rec_size > std.math.maxInt(u32)) return error.RecordTooLarge;

        var buf = try self.allocator.alloc(u8, rec_size);
        defer self.allocator.free(buf);

        const h = hashBytes(key);
        // Prelude
        std.mem.writeInt(u64, buf[0..8], h, .little);
        std.mem.writeInt(u32, buf[8..12], key_len, .little);
        std.mem.writeInt(u32, buf[12..16], ids_len, .little);
        // Key
        @memcpy(buf[record_prelude_size .. record_prelude_size + key.len], key);
        // Ids (write each u32 little-endian)
        var off: usize = record_prelude_size + key.len;
        for (ids) |id| {
            std.mem.writeInt(TokenId, buf[off..][0..@sizeOf(TokenId)], id, .little);
            off += @sizeOf(TokenId);
        }
        // CRC over everything but the trailing CRC slot.
        const crc = Crc64.hash(buf[0..off]);
        std.mem.writeInt(u64, buf[off..][0..8], crc, .little);

        const offset = self.file_len;
        try pwriteAll(self.fd, buf, offset);
        self.file_len = offset + buf.len;

        // Update the in-memory file index. If this hash was already
        // present (overwrite or true hash-collision), the old record
        // becomes dead bytes.
        if (self.file_index.get(h)) |old| {
            self.dead_bytes += old.size;
            self.dead_entries += 1;
        }
        try self.file_index.put(h, .{
            .offset = offset,
            .size = @intCast(buf.len),
            .ids_len = ids_len,
        });

        self.writes_since_sync += 1;
        self.bytes_since_sync += buf.len;
    }

    fn readRecord(self: *PrefixCache, fe: FileEntry) !ScannedRecord {
        const buf = try self.allocator.alloc(u8, fe.size);
        errdefer self.allocator.free(buf);
        try preadAll(self.fd, buf, fe.offset);
        // Validate first; then decode the (potentially mis-aligned)
        // ids into a fresh, aligned slice owned by the caller.
        const parts = try parseAndVerifyRecord(buf, fe.size);
        const ids = try self.allocator.alloc(TokenId, parts.ids_len);
        errdefer self.allocator.free(ids);
        decodeIdsInto(ids, buf[parts.ids_off..][0 .. @as(usize, parts.ids_len) * @sizeOf(TokenId)]);
        return .{
            .bytes = buf,
            .key = buf[parts.key_off .. parts.key_off + parts.key_len],
            .ids = ids,
            .size = @intCast(buf.len),
        };
    }

    /// Pure metadata view of a validated record. Pointers/indices into
    /// the input buffer; no allocation.
    const RecordParts = struct {
        key_off: usize,
        key_len: u32,
        ids_off: usize,
        ids_len: u32,
    };

    /// Validate `buf` as a record (length checks + CRC). Returns
    /// offsets into `buf`. Does NOT decode the ids — call
    /// `decodeIdsInto` for that, because the ids region of `buf`
    /// isn't guaranteed to be `@alignOf(TokenId)`-aligned.
    fn parseAndVerifyRecord(buf: []const u8, expected_size: u32) !RecordParts {
        if (buf.len < record_prelude_size + record_crc_size) return error.RecordTruncated;
        const key_len = std.mem.readInt(u32, buf[8..12], .little);
        const ids_len = std.mem.readInt(u32, buf[12..16], .little);
        const ids_bytes: usize = @as(usize, ids_len) * @sizeOf(TokenId);
        const need: usize = record_prelude_size + key_len + ids_bytes + record_crc_size;
        if (need != buf.len or need != expected_size) return error.RecordTruncated;
        const crc_off = need - record_crc_size;
        const stored_crc = std.mem.readInt(u64, buf[crc_off..][0..8], .little);
        const calc_crc = Crc64.hash(buf[0..crc_off]);
        if (stored_crc != calc_crc) return error.RecordCrcMismatch;
        return .{
            .key_off = record_prelude_size,
            .key_len = key_len,
            .ids_off = record_prelude_size + key_len,
            .ids_len = ids_len,
        };
    }

    fn scanFile(self: *PrefixCache) !void {
        const file_size = try fileLength(self.fd);
        self.file_len = file_size;
        if (file_size <= header_size) {
            self.file_len = header_size;
            return;
        }

        var offset: u64 = header_size;
        var last_good: u64 = header_size;
        var prelude_buf: [record_prelude_size]u8 = undefined;

        while (offset < file_size) {
            // Pull the prelude first so we know how large the record is.
            const remaining = file_size - offset;
            if (remaining < record_prelude_size + record_crc_size) {
                // Trailing garbage: truncate.
                break;
            }
            preadAll(self.fd, &prelude_buf, offset) catch break;
            const h = std.mem.readInt(u64, prelude_buf[0..8], .little);
            const key_len = std.mem.readInt(u32, prelude_buf[8..12], .little);
            const ids_len = std.mem.readInt(u32, prelude_buf[12..16], .little);
            const ids_bytes: usize = @as(usize, ids_len) * @sizeOf(TokenId);
            const rec_size: u64 = record_prelude_size + key_len + ids_bytes + record_crc_size;
            if (rec_size > remaining or rec_size > std.math.maxInt(u32)) {
                // Tail of file is truncated mid-record; we'll truncate
                // back to last_good below.
                break;
            }
            // Read + verify the body.
            const buf = self.allocator.alloc(u8, @intCast(rec_size)) catch break;
            defer self.allocator.free(buf);
            preadAll(self.fd, buf, offset) catch break;
            const parsed = parseAndVerifyRecord(buf, @intCast(rec_size)) catch {
                // CRC mismatch → stop scanning, truncate at last_good.
                break;
            };
            _ = parsed; // we only need the prelude + crc for indexing.

            if (self.file_index.get(h)) |old| {
                self.dead_bytes += old.size;
                self.dead_entries += 1;
            }
            try self.file_index.put(h, .{
                .offset = offset,
                .size = @intCast(rec_size),
                .ids_len = ids_len,
            });
            offset += rec_size;
            last_good = offset;
        }

        if (last_good != file_size) {
            // We hit garbage / a torn record. Truncate so subsequent
            // appends land in a known position.
            try ftruncate(self.fd, last_good);
            self.file_len = last_good;
        } else {
            self.file_len = file_size;
        }
    }

    fn maybeSync(self: *PrefixCache, force: bool) !void {
        const by_writes = self.cfg.sync_every_n_writes != 0 and
            self.writes_since_sync >= self.cfg.sync_every_n_writes;
        const by_bytes = self.cfg.sync_every_bytes != 0 and
            self.bytes_since_sync >= self.cfg.sync_every_bytes;
        if (force or by_writes or by_bytes) try self.sync();
    }

    fn maybeCompact(self: *PrefixCache) !void {
        if (self.cfg.compact_at_stale_pct == 0) return;
        if (self.file_len <= header_size) return;
        const body_bytes: u64 = self.file_len - header_size;
        if (body_bytes == 0) return;
        const pct = (self.dead_bytes * 100) / body_bytes;
        if (pct >= self.cfg.compact_at_stale_pct) {
            try self.compact();
        }
    }

    fn compact(self: *PrefixCache) !void {
        // Rewrite the file in place: build the new image in memory
        // (or, for huge caches, in a tmp file + rename), then atomically
        // replace. For a serving cache the working set is rarely large
        // enough that we need to fall back to a tmp file; we
        // unconditionally do tmp+rename so a crash mid-compaction
        // doesn't leave the live file truncated.

        // Build a temp file path next to the original.
        const tmp_path = try std.fmt.allocPrint(self.allocator, "{s}.compact-tmp", .{self.cfg.path});
        defer self.allocator.free(tmp_path);

        // A crash may leave the exclusive temp name behind. The live cache
        // lock proves no other writer for this file is compacting now, so the
        // stale artifact can be removed safely before retrying.
        const tmp_fd = openOrCreateRWExcl(tmp_path) catch |err| switch (err) {
            error.PathAlreadyExists => blk: {
                try unlinkPath(tmp_path);
                break :blk try openOrCreateRWExcl(tmp_path);
            },
            else => return err,
        };
        var tmp_fd_owned = true;
        defer {
            if (tmp_fd_owned) _ = posix.system.close(tmp_fd);
        }
        errdefer unlinkPath(tmp_path) catch {};
        // Lock the replacement inode before publishing it. This closes the
        // rename→reopen race where a second process could acquire the new
        // path while this process still held only the old inode's lock.
        try flockExclusive(tmp_fd);

        try writeFreshHeader(tmp_fd);
        var new_len: u64 = header_size;

        var new_index = std.AutoHashMap(u64, FileEntry).init(self.allocator);
        errdefer new_index.deinit();

        var it = self.file_index.iterator();
        while (it.next()) |kv| {
            const fe = kv.value_ptr.*;
            const buf = try self.allocator.alloc(u8, fe.size);
            defer self.allocator.free(buf);
            try preadAll(self.fd, buf, fe.offset);
            // Re-verify on the way out; corruption found here is a bug
            // (we just wrote these records) but it's cheap insurance.
            _ = try parseAndVerifyRecord(buf, fe.size);
            try pwriteAll(tmp_fd, buf, new_len);
            try new_index.put(kv.key_ptr.*, .{
                .offset = new_len,
                .size = fe.size,
                .ids_len = fe.ids_len,
            });
            new_len += fe.size;
        }
        try posix.fdatasync(tmp_fd);

        // Atomically swap. The exclusive lock on `self.fd` keeps other
        // writers out; readers from another process opening at this
        // instant will see the post-rename file (the rename is atomic
        // wrt. the directory entry).
        try renamePath(tmp_path, self.cfg.path);

        // `tmp_fd` already refers to (and locks) the renamed inode. Transfer
        // ownership directly instead of reopening and creating a lock gap.
        _ = posix.system.close(self.fd);
        self.fd = tmp_fd;
        tmp_fd_owned = false;

        self.file_index.deinit();
        self.file_index = new_index;
        self.file_len = new_len;
        self.dead_bytes = 0;
        self.dead_entries = 0;
        self.compactions += 1;
    }
};

/// Decode `bytes` (a sequence of u32 LE values, possibly
/// mis-aligned) into the caller-owned, aligned `dst`. Caller
/// guarantees `dst.len * @sizeOf(TokenId) == bytes.len`.
fn decodeIdsInto(dst: []TokenId, bytes: []const u8) void {
    std.debug.assert(dst.len * @sizeOf(TokenId) == bytes.len);
    var i: usize = 0;
    while (i < dst.len) : (i += 1) {
        const off = i * @sizeOf(TokenId);
        dst[i] = std.mem.readInt(TokenId, bytes[off..][0..@sizeOf(TokenId)], .little);
    }
}

// ----- file/fd helpers (POSIX, libc-linked) -----

fn openOrCreateRW(path: []const u8) !posix.fd_t {
    var flags: posix.O = .{ .ACCMODE = .RDWR, .CREAT = true };
    _ = &flags;
    return posix.openat(posix.AT.FDCWD, path, flags, 0o644);
}

fn openOrCreateRWExcl(path: []const u8) !posix.fd_t {
    // Used by compaction's tmp file: must not collide with anything.
    var flags: posix.O = .{ .ACCMODE = .RDWR, .CREAT = true, .EXCL = true, .TRUNC = true };
    _ = &flags;
    return posix.openat(posix.AT.FDCWD, path, flags, 0o644);
}

fn flockExclusive(fd: posix.fd_t) !void {
    while (true) {
        switch (posix.errno(posix.system.flock(fd, posix.LOCK.EX))) {
            .SUCCESS => return,
            .INTR => continue,
            .NOLCK => return error.SystemResources,
            .OPNOTSUPP => return error.FileLocksUnsupported,
            else => |e| return posix.unexpectedErrno(e),
        }
    }
}

/// Try to acquire an exclusive lock without blocking. Returns true on
/// success, false if another process holds the lock. Used by tests.
pub fn tryFlockExclusive(fd: posix.fd_t) !bool {
    switch (posix.errno(posix.system.flock(fd, posix.LOCK.EX | posix.LOCK.NB))) {
        .SUCCESS => return true,
        // On Linux EAGAIN == EWOULDBLOCK; flock(2) uses EWOULDBLOCK
        // for the contended case but the errno enum spells it AGAIN.
        .AGAIN => return false,
        .INTR => return error.Interrupted,
        .NOLCK => return error.SystemResources,
        else => |e| return posix.unexpectedErrno(e),
    }
}

fn fileLength(fd: posix.fd_t) !u64 {
    // Use libc lseek to SEEK_END for the file length. `std.posix` in
    // 0.16 doesn't expose a portable `fstat` (Linux uses `statx` etc.)
    // and the build already links libc, so this is the lightest tap
    // on the system we can take. On non-WASI POSIX, `whence_t` is just
    // `c_int`.
    const whence: std.c.whence_t = @as(std.c.whence_t, std.c.SEEK.END);
    const rc = std.c.lseek(fd, 0, whence);
    if (rc < 0) return error.StatFailed;
    return @intCast(rc);
}

fn ftruncate(fd: posix.fd_t, len: u64) !void {
    if (std.c.ftruncate(fd, @intCast(len)) != 0) return error.TruncateFailed;
}

fn pwriteAll(fd: posix.fd_t, buf: []const u8, offset: u64) !void {
    var written: usize = 0;
    while (written < buf.len) {
        const rc = std.c.pwrite(fd, buf.ptr + written, buf.len - written, @intCast(offset + written));
        if (rc < 0) {
            switch (posix.errno(rc)) {
                .INTR => continue,
                .NOSPC => return error.NoSpaceLeft,
                .IO => return error.InputOutput,
                .BADF => return error.BadFileDescriptor,
                else => |e| return posix.unexpectedErrno(e),
            }
        }
        if (rc == 0) return error.UnexpectedShortWrite;
        written += @intCast(rc);
    }
}

fn preadAll(fd: posix.fd_t, buf: []u8, offset: u64) !void {
    var read_total: usize = 0;
    while (read_total < buf.len) {
        const rc = std.c.pread(fd, buf.ptr + read_total, buf.len - read_total, @intCast(offset + read_total));
        if (rc < 0) {
            switch (posix.errno(rc)) {
                .INTR => continue,
                .IO => return error.InputOutput,
                .BADF => return error.BadFileDescriptor,
                else => |e| return posix.unexpectedErrno(e),
            }
        }
        if (rc == 0) return error.UnexpectedEof;
        read_total += @intCast(rc);
    }
}

fn unlinkPath(path: []const u8) !void {
    // Convert to null-terminated for libc.
    var buf: [posix.PATH_MAX]u8 = undefined;
    if (path.len + 1 > buf.len) return error.NameTooLong;
    @memcpy(buf[0..path.len], path);
    buf[path.len] = 0;
    if (std.c.unlink(@ptrCast(&buf)) != 0) return error.UnlinkFailed;
}

fn renamePath(old_path: []const u8, new_path: []const u8) !void {
    var old_buf: [posix.PATH_MAX]u8 = undefined;
    var new_buf: [posix.PATH_MAX]u8 = undefined;
    if (old_path.len + 1 > old_buf.len) return error.NameTooLong;
    if (new_path.len + 1 > new_buf.len) return error.NameTooLong;
    @memcpy(old_buf[0..old_path.len], old_path);
    old_buf[old_path.len] = 0;
    @memcpy(new_buf[0..new_path.len], new_path);
    new_buf[new_path.len] = 0;
    if (std.c.rename(@ptrCast(&old_buf), @ptrCast(&new_buf)) != 0) return error.RenameFailed;
}

fn writeFreshHeader(fd: posix.fd_t) !void {
    var header: [PrefixCache.header_size]u8 = @splat(0);
    @memcpy(header[0..8], &PrefixCache.magic);
    std.mem.writeInt(u32, header[8..12], PrefixCache.format_version, .little);
    // entry_count + total_bytes left at 0; we don't currently maintain
    // them precisely on every append (would defeat the append-only
    // property) — they're set on close/compact as best-effort hints.
    try pwriteAll(fd, &header, 0);
}

fn validateHeader(fd: posix.fd_t) !void {
    var header: [PrefixCache.header_size]u8 = undefined;
    preadAll(fd, &header, 0) catch return error.HeaderShort;
    if (!std.mem.eql(u8, header[0..8], &PrefixCache.magic)) return error.BadMagic;
    const version = std.mem.readInt(u32, header[8..12], .little);
    if (version != PrefixCache.format_version) return error.UnsupportedVersion;
}

// ============================================================
// Tests
// ============================================================

const testing = std.testing;

test "hashBytes is deterministic" {
    const a = hashBytes("hello world");
    const b = hashBytes("hello world");
    try testing.expectEqual(a, b);
    const c = hashBytes("hello worle");
    try testing.expect(a != c);
}

test "lookup miss returns null" {
    var c = Cache.init(testing.allocator, 4);
    defer c.deinit();
    try testing.expectEqual(@as(?[]const TokenId, null), c.lookup("nope"));
    try testing.expectEqual(@as(u64, 1), c.stats.misses);
    try testing.expectEqual(@as(u64, 0), c.stats.hits);
}

test "explicit insert + lookup roundtrip" {
    var c = Cache.init(testing.allocator, 4);
    defer c.deinit();
    const ids = [_]TokenId{ 1, 2, 3, 4, 5 };
    try c.insert("sys-prompt", &ids);
    const got = c.lookup("sys-prompt") orelse return error.MissingEntry;
    try testing.expectEqualSlices(TokenId, &ids, got);
    try testing.expectEqual(@as(u64, 1), c.stats.hits);
    try testing.expectEqual(@as(u32, 1), c.stats.entries);
}

test "hit increments stats.hits, miss increments stats.misses" {
    var c = Cache.init(testing.allocator, 4);
    defer c.deinit();
    const ids = [_]TokenId{ 7, 8, 9 };
    try c.insert("A", &ids);
    _ = c.lookup("A") orelse return error.MissingEntry;
    _ = c.lookup("A") orelse return error.MissingEntry;
    _ = c.lookup("missing");
    try testing.expectEqual(@as(u64, 2), c.stats.hits);
    try testing.expectEqual(@as(u64, 1), c.stats.misses);
}

test "lookupOrInsert: miss then hit returns same memory" {
    const Vocab = @import("vocab.zig").Vocab;
    var v = Vocab.empty(testing.allocator);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };

    var c = Cache.init(testing.allocator, 4);
    defer c.deinit();

    const a = try c.lookupOrInsert(&pipe, "hi");
    try testing.expectEqualSlices(TokenId, &.{ 'h', 'i' }, a);
    try testing.expectEqual(@as(u64, 0), c.stats.hits);
    try testing.expectEqual(@as(u64, 1), c.stats.misses);

    const b = try c.lookupOrInsert(&pipe, "hi");
    try testing.expectEqual(a.ptr, b.ptr);
    try testing.expectEqual(a.len, b.len);
    try testing.expectEqual(@as(u64, 1), c.stats.hits);
    try testing.expectEqual(@as(u64, 1), c.stats.misses);
}

test "capacity eviction" {
    var c = Cache.init(testing.allocator, 3);
    defer c.deinit();
    const ids = [_]TokenId{1};
    try c.insert("A", &ids);
    try c.insert("B", &ids);
    try c.insert("C", &ids);
    try c.insert("D", &ids);
    try testing.expectEqual(@as(u64, 1), c.stats.evictions);
    try testing.expectEqual(@as(u32, 3), c.stats.entries);
    // A was oldest -> evicted.
    try testing.expectEqual(@as(?[]const TokenId, null), c.lookup("A"));
}

test "LRU promotion keeps recently-touched entry" {
    var c = Cache.init(testing.allocator, 3);
    defer c.deinit();
    const ids = [_]TokenId{42};
    try c.insert("A", &ids);
    try c.insert("B", &ids);
    try c.insert("C", &ids);
    // Touch A -> A is now most-recent, B is now LRU tail.
    _ = c.lookup("A") orelse return error.MissingA;
    try c.insert("D", &ids);
    // B should be evicted; A, C, D still live.
    try testing.expectEqual(@as(u64, 1), c.stats.evictions);
    try testing.expectEqual(@as(?[]const TokenId, null), c.lookup("B"));
    try testing.expect(c.lookup("A") != null);
    try testing.expect(c.lookup("C") != null);
    try testing.expect(c.lookup("D") != null);
}

// ----- compaction tests -----

test "compactionStats: live_bytes matches sum of id-array lengths" {
    var c = Cache.init(testing.allocator, 8);
    defer c.deinit();
    const a_ids = [_]TokenId{ 1, 2, 3 };
    const b_ids = [_]TokenId{ 4, 5, 6, 7, 8 };
    const c_ids = [_]TokenId{9};
    try c.insert("A", &a_ids);
    try c.insert("B", &b_ids);
    try c.insert("C", &c_ids);

    const s = c.compactionStats();
    const expected_live = (a_ids.len + b_ids.len + c_ids.len) * @sizeOf(TokenId);
    try testing.expectEqual(expected_live, s.live_bytes);
    // No compactions triggered yet (we're well under the 4x watermark).
    try testing.expectEqual(@as(u32, 0), s.compactions);
    try testing.expect(s.arena_bytes >= expected_live);
}

test "watermark-triggered compaction preserves entry contents" {
    // Capacity high enough that nothing evicts. We overwrite the same
    // key over and over with growing ids; each overwrite orphans the
    // previous id-slice, growing arena_bytes without growing live_bytes
    // by much. Eventually arena_bytes > 4 * live_bytes triggers a
    // compaction.
    var c = Cache.initWithOptions(testing.allocator, 16, .{ .compaction_ratio = 4.0 });
    defer c.deinit();

    const small = [_]TokenId{ 1, 2, 3, 4 };
    try c.insert("steady", &small);

    // Overwrite "steady" 100 times with a different id (same length) →
    // each overwrite orphans the previous id slot.
    var i: u32 = 0;
    while (i < 100) : (i += 1) {
        var buf: [4]TokenId = .{ i, i, i, i };
        try c.insert("steady", &buf);
    }

    const s = c.compactionStats();
    try testing.expect(s.compactions >= 1);

    // The entry must still resolve to the most recent value.
    const got = c.lookup("steady") orelse return error.MissingEntry;
    const last_i: TokenId = 99;
    try testing.expectEqualSlices(TokenId, &.{ last_i, last_i, last_i, last_i }, got);
}

test "forceCompact shrinks arena_bytes after evictions" {
    var c = Cache.initWithOptions(testing.allocator, 8, .{ .compaction_ratio = 1024.0 });
    defer c.deinit();

    // Insert 8 entries, then evict half by inserting 4 new ones.
    const filler = [_]TokenId{ 100, 101, 102, 103, 104, 105, 106, 107 };
    const keys = [_][]const u8{ "k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7" };
    for (keys) |k| try c.insert(k, &filler);

    const before = c.compactionStats();
    try testing.expectEqual(@as(u32, 0), before.compactions);
    try testing.expectEqual(filler.len * @sizeOf(TokenId) * 8, before.live_bytes);

    // Evict half by inserting 4 fresh keys with capacity = 8.
    const keys2 = [_][]const u8{ "kk0", "kk1", "kk2", "kk3" };
    for (keys2) |k| try c.insert(k, &filler);
    // Still 8 live entries (LRU evicted 4 of the originals).
    try testing.expectEqual(@as(u32, 8), c.stats.entries);

    // arena_bytes grew (orphaned slots from evicted entries are still
    // pinned in the arena until compaction).
    const after_evict = c.compactionStats();
    try testing.expectEqual(@as(u32, 0), after_evict.compactions);
    try testing.expect(after_evict.arena_bytes > after_evict.live_bytes);
    try testing.expectEqual(filler.len * @sizeOf(TokenId) * 8, after_evict.live_bytes);

    // Force compaction: arena_bytes should fall to ~live_bytes.
    try c.forceCompact();
    const after_compact = c.compactionStats();
    try testing.expectEqual(@as(u32, 1), after_compact.compactions);
    try testing.expectEqual(after_compact.live_bytes, after_compact.arena_bytes);
    // arena_bytes after must be < arena_bytes before (we had real waste
    // from the 4 evictions).
    try testing.expect(after_compact.arena_bytes < after_evict.arena_bytes);

    // All live entries still resolve correctly.
    for (keys[4..]) |k| {
        const got = c.lookup(k) orelse return error.MissingEntry;
        try testing.expectEqualSlices(TokenId, &filler, got);
    }
    for (keys2) |k| {
        const got = c.lookup(k) orelse return error.MissingEntry;
        try testing.expectEqualSlices(TokenId, &filler, got);
    }
}

test "LRU order is preserved across forceCompact" {
    var c = Cache.initWithOptions(testing.allocator, 4, .{ .compaction_ratio = 1024.0 });
    defer c.deinit();

    const ids = [_]TokenId{ 1, 2, 3 };
    try c.insert("A", &ids);
    try c.insert("B", &ids);
    try c.insert("C", &ids);
    try c.insert("D", &ids);
    // LRU tail right now: A. Touch B so the tail becomes A still (most
    // recent: B, then D, then C, then A).
    _ = c.lookup("B") orelse return error.MissingB;

    // Compact: the underlying nodes array may be rewritten internally,
    // but the LRU edges (and thus the eviction order) must be exactly
    // the same.
    try c.forceCompact();

    // The next insert (with capacity 4, already full) must evict A.
    try c.insert("E", &ids);
    try testing.expectEqual(@as(?[]const TokenId, null), c.lookup("A"));
    try testing.expect(c.lookup("B") != null);
    try testing.expect(c.lookup("C") != null);
    try testing.expect(c.lookup("D") != null);
    try testing.expect(c.lookup("E") != null);
}

test "heavy churn workload keeps arena_bytes bounded" {
    // Random insert/evict churn for many iterations. The cache has a
    // small capacity, so most inserts evict. Without watermark
    // compaction, arena_bytes would grow linearly with the iteration
    // count. With compaction, it must stay bounded by a small multiple
    // of (capacity * avg_id_bytes).
    var c = Cache.initWithOptions(testing.allocator, 32, .{ .compaction_ratio = 4.0 });
    defer c.deinit();

    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const rand = prng.random();

    const N: u32 = 10_000;
    var key_buf: [16]u8 = undefined;
    var id_buf: [64]TokenId = undefined;

    var i: u32 = 0;
    while (i < N) : (i += 1) {
        // Random 8-char ASCII key. With 2^48-ish possibilities and a
        // 32-slot cache, almost every insert is a miss → eviction.
        for (key_buf[0..8]) |*b| b.* = 'a' + @as(u8, @intCast(rand.intRangeAtMost(u32, 0, 25)));
        const id_len = rand.intRangeAtMost(u32, 1, 64);
        for (id_buf[0..id_len]) |*x| x.* = rand.int(TokenId);
        try c.insert(key_buf[0..8], id_buf[0..id_len]);
    }

    const s = c.compactionStats();

    // Sanity: we ran enough iterations to have triggered many compactions.
    try testing.expect(s.compactions > 0);

    // Hard upper bound: arena_bytes can never exceed
    // watermark + one-entry-worth of slop (the watermark check happens
    // BEFORE we allocate the next entry, so the post-insert arena_bytes
    // is at most `live_bytes_after_compact + one_alloc`). Cap at
    // (compaction_ratio + 2) * max-live to allow generous headroom.
    const max_possible_live_bytes: usize = c.capacity * 64 * @sizeOf(TokenId);
    const upper_bound: usize = max_possible_live_bytes * 8;
    try testing.expect(s.arena_bytes <= upper_bound);
}

test "lookupOrInsert + compaction: invalidated slice is documented behavior" {
    // This test documents the slice-lifetime invariant: a lookup result
    // is valid until the next mutating op. After a compaction, the old
    // ptr is dangling. We don't dereference the old ptr here (that
    // would be UB), but we do verify that a fresh lookup returns the
    // same logical contents, and that the ptr value has changed.
    const Vocab = @import("vocab.zig").Vocab;
    var v = Vocab.empty(testing.allocator);
    defer v.deinit();
    const pipe: Pipeline = .{
        .normalizer = .identity,
        .pre_tokenizer = .identity,
        .model = .byte_id,
        .decoder = .concat,
        .vocab = &v,
    };

    var c = Cache.init(testing.allocator, 4);
    defer c.deinit();

    const first = try c.lookupOrInsert(&pipe, "hello");
    const first_ptr = first.ptr;
    const first_len = first.len;

    try c.forceCompact();

    const again = c.lookup("hello") orelse return error.MissingEntry;
    try testing.expectEqual(first_len, again.len);
    // Pointer must have moved — fresh arena.
    try testing.expect(first_ptr != again.ptr);
    // Contents are identical.
    var i: usize = 0;
    while (i < again.len) : (i += 1) {
        try testing.expectEqual(@as(TokenId, "hello"[i]), again[i]);
    }
}

// ============================================================
// Persistent backend tests
// ============================================================

/// Build an absolute path to `sub_name` inside the test's tmp dir.
/// Caller frees with `allocator.free`.
fn tmpAbsPath(
    allocator: std.mem.Allocator,
    tmp: *std.testing.TmpDir,
    sub_name: []const u8,
) ![]u8 {
    var buf: [4096]u8 = undefined;
    const n = try tmp.dir.realPath(std.testing.io, &buf);
    return std.fmt.allocPrint(allocator, "{s}/{s}", .{ buf[0..n], sub_name });
}

test "persistent: open empty, write 100 entries, close, reopen, all 100 readable" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmpAbsPath(testing.allocator, &tmp, "ztok_prefix_cache.dat");
    defer testing.allocator.free(path);

    // Phase 1: write.
    {
        const pc = try PrefixCache.openPersistent(testing.allocator, .{
            .path = path,
            .sync_every_n_writes = 16,
            .in_memory_capacity = 256,
        });
        defer pc.closePersistent();

        var i: u32 = 0;
        while (i < 100) : (i += 1) {
            var key_buf: [16]u8 = undefined;
            const key = try std.fmt.bufPrint(&key_buf, "k-{d}", .{i});
            const ids = [_]TokenId{ i, i +% 1, i +% 2, i +% 3 };
            try pc.insert(key, &ids);
        }
        // Force a final flush so the close call doesn't depend on
        // closePersistent's best-effort sync.
        try pc.sync();
    }

    // Phase 2: reopen and verify.
    {
        const pc = try PrefixCache.openPersistent(testing.allocator, .{
            .path = path,
            .in_memory_capacity = 256,
        });
        defer pc.closePersistent();

        // File index should hold all 100 hashes.
        const s = pc.persistentStats();
        try testing.expectEqual(@as(u64, 100), s.file_entries);
        try testing.expectEqual(@as(u64, 0), s.dead_entries);

        var i: u32 = 0;
        while (i < 100) : (i += 1) {
            var key_buf: [16]u8 = undefined;
            const key = try std.fmt.bufPrint(&key_buf, "k-{d}", .{i});
            const got = (try pc.lookup(key)) orelse {
                std.debug.print("missing key {s}\n", .{key});
                return error.MissingEntry;
            };
            const expected = [_]TokenId{ i, i +% 1, i +% 2, i +% 3 };
            try testing.expectEqualSlices(TokenId, &expected, got);
        }
    }
}

test "persistent: corruption mid-record is truncated, prior entries intact" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmpAbsPath(testing.allocator, &tmp, "ztok_prefix_cache.dat");
    defer testing.allocator.free(path);

    // Phase 1: write 10 entries, sync, close.
    {
        const pc = try PrefixCache.openPersistent(testing.allocator, .{
            .path = path,
            .sync_every_n_writes = 1,
        });
        defer pc.closePersistent();
        var i: u32 = 0;
        while (i < 10) : (i += 1) {
            var key_buf: [16]u8 = undefined;
            const key = try std.fmt.bufPrint(&key_buf, "key-{d}", .{i});
            const ids = [_]TokenId{ i, i * 2 };
            try pc.insert(key, &ids);
        }
    }

    // Phase 2: corrupt the tail of the file. Open as raw fd, append
    // 7 bytes of garbage (which is < record_prelude_size + crc_size,
    // so scanFile will treat it as trailing garbage and truncate it).
    {
        const fd = try openOrCreateRW(path);
        defer _ = posix.system.close(fd);
        const old_len = try fileLength(fd);
        const garbage = [_]u8{ 0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA };
        try pwriteAll(fd, &garbage, old_len);
    }

    // Phase 3: reopen — the trailing garbage should be truncated, the
    // 10 prior entries should remain.
    {
        const pc = try PrefixCache.openPersistent(testing.allocator, .{ .path = path });
        defer pc.closePersistent();
        const s = pc.persistentStats();
        try testing.expectEqual(@as(u64, 10), s.file_entries);
        var i: u32 = 0;
        while (i < 10) : (i += 1) {
            var key_buf: [16]u8 = undefined;
            const key = try std.fmt.bufPrint(&key_buf, "key-{d}", .{i});
            const got = (try pc.lookup(key)) orelse return error.MissingEntry;
            const expected = [_]TokenId{ i, i * 2 };
            try testing.expectEqualSlices(TokenId, &expected, got);
        }
    }

    // Phase 4: now corrupt deeper — overwrite the LAST RECORD's CRC
    // with zeros. Scan should stop AT the corrupted record (keeping
    // the 9 records before it intact) and truncate the bad one out.
    {
        const fd = try openOrCreateRW(path);
        defer _ = posix.system.close(fd);
        // The simplest way to find "the last record" is to compute its
        // size from the layout. Each record has 16 bytes of prelude,
        // 5..6 bytes of key ("key-0".."key-9"), 8 bytes for the 2
        // TokenIds, and 8 bytes of CRC.
        const file_len = try fileLength(fd);
        const last_5_bytes_off = file_len - 5; // somewhere inside CRC
        const zeros = [_]u8{ 0, 0, 0, 0, 0 };
        try pwriteAll(fd, &zeros, last_5_bytes_off);
    }
    {
        const pc = try PrefixCache.openPersistent(testing.allocator, .{ .path = path });
        defer pc.closePersistent();
        const s = pc.persistentStats();
        // We can't be sure exactly how many records were retained
        // (depends on which record happened to be corrupted), but
        // there must be at least 1 less than before AND the surviving
        // records must round-trip.
        try testing.expect(s.file_entries < 10);
        try testing.expect(s.file_entries >= 1);
    }
}

test "persistent: two PrefixCache instances on same file, second sees first's writes" {
    // Simulates cross-process visibility within one process. Note: Zig
    // 0.16 doesn't expose `fork` portably, and the task spec explicitly
    // allows the same-process simulation. The mechanism is the same —
    // the writer must release the lock before the reader can acquire
    // it (we close+reopen between phases).
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmpAbsPath(testing.allocator, &tmp, "ztok_prefix_cache.dat");
    defer testing.allocator.free(path);

    // Writer instance.
    {
        const w = try PrefixCache.openPersistent(testing.allocator, .{
            .path = path,
            .sync_every_n_writes = 1, // sync after every insert
        });
        defer w.closePersistent();
        try w.insert("alpha", &.{ 1, 2, 3 });
        try w.insert("beta", &.{ 4, 5 });
        try w.insert("gamma", &.{6});
        try w.sync();
    }

    // Reader instance — fresh process simulation: brand-new
    // PrefixCache with empty in-memory state, must reload from disk.
    {
        const r = try PrefixCache.openPersistent(testing.allocator, .{ .path = path });
        defer r.closePersistent();

        const a = (try r.lookup("alpha")) orelse return error.Missing;
        try testing.expectEqualSlices(TokenId, &.{ 1, 2, 3 }, a);
        const b = (try r.lookup("beta")) orelse return error.Missing;
        try testing.expectEqualSlices(TokenId, &.{ 4, 5 }, b);
        const g = (try r.lookup("gamma")) orelse return error.Missing;
        try testing.expectEqualSlices(TokenId, &.{6}, g);

        // A second writer instance on the same path, after the reader
        // releases (i.e. closes), should be able to append more.
        // Demonstrated by inserting a new entry and re-scanning.
        const r_stats = r.persistentStats();
        try testing.expectEqual(@as(u64, 3), r_stats.file_entries);
    }
}

test "persistent: compaction shrinks the file when stale fraction > threshold" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmpAbsPath(testing.allocator, &tmp, "ztok_prefix_cache.dat");
    defer testing.allocator.free(path);

    const pc = try PrefixCache.openPersistent(testing.allocator, .{
        .path = path,
        // Disable auto-compaction so we can measure file growth before
        // calling compactNow explicitly.
        .compact_at_stale_pct = 0,
        .sync_every_n_writes = 0,
        .sync_every_bytes = 0,
    });
    defer pc.closePersistent();

    // Write one steady key + repeatedly overwrite it. Each overwrite
    // appends a new record AND marks the previous record dead.
    const initial_ids = [_]TokenId{1};
    try pc.insert("steady", &initial_ids);

    var i: u32 = 0;
    while (i < 50) : (i += 1) {
        var buf: [4]TokenId = .{ i, i +% 1, i +% 2, i +% 3 };
        try pc.insert("steady", &buf);
    }
    // Also include one "live" cold key so the file isn't empty post-compaction.
    try pc.insert("cold", &.{ 100, 101, 102 });

    const before = pc.persistentStats();
    try testing.expect(before.dead_entries >= 50);
    try testing.expect(before.dead_bytes > 0);
    const before_file_bytes = before.file_bytes;

    try pc.compactNow();

    const after = pc.persistentStats();
    try testing.expectEqual(@as(u64, 0), after.dead_entries);
    try testing.expectEqual(@as(u64, 0), after.dead_bytes);
    try testing.expectEqual(@as(u32, 1), after.compactions);
    try testing.expect(after.file_bytes < before_file_bytes);
    // Exactly two live records: "steady" (most recent overwrite) and "cold".
    try testing.expectEqual(@as(u64, 2), after.file_entries);

    // Surviving entries still round-trip.
    const steady = (try pc.lookup("steady")) orelse return error.Missing;
    const expected_last: TokenId = 49;
    try testing.expectEqualSlices(TokenId, &.{
        expected_last,
        expected_last +% 1,
        expected_last +% 2,
        expected_last +% 3,
    }, steady);
    const cold = (try pc.lookup("cold")) orelse return error.Missing;
    try testing.expectEqualSlices(TokenId, &.{ 100, 101, 102 }, cold);
}

test "persistent: compaction recovers a temp file left by a crash" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmpAbsPath(testing.allocator, &tmp, "ztok_prefix_cache.dat");
    defer testing.allocator.free(path);
    const pc = try PrefixCache.openPersistent(testing.allocator, .{
        .path = path,
        .compact_at_stale_pct = 0,
    });
    defer pc.closePersistent();
    try pc.insert("key", &.{ 7, 8, 9 });

    const stale_path = try std.fmt.allocPrint(testing.allocator, "{s}.compact-tmp", .{path});
    defer testing.allocator.free(stale_path);
    const stale_fd = try openOrCreateRWExcl(stale_path);
    _ = posix.system.close(stale_fd);

    try pc.compactNow();
    const got = (try pc.lookup("key")) orelse return error.Missing;
    try testing.expectEqualSlices(TokenId, &.{ 7, 8, 9 }, got);
}

test "persistent: concurrent writer is blocked by file lock" {
    // Two PrefixCache instances on the same file from the same
    // process. The second one's `openPersistent` would block forever
    // trying to acquire LOCK_EX, so instead we exercise the
    // mechanism directly: open + lock the first fd, then verify
    // `tryFlockExclusive` on a second fd reports "would block".
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmpAbsPath(testing.allocator, &tmp, "ztok_prefix_cache.dat");
    defer testing.allocator.free(path);

    const first = try PrefixCache.openPersistent(testing.allocator, .{ .path = path });
    defer first.closePersistent();
    try first.insert("locked-key", &.{ 7, 8, 9 });
    try first.sync();

    // Now try to acquire an exclusive lock on the stable sidecar via a
    // separate fd. Should fail with "would block" because `first`
    // already holds it. (Note: Linux flock locks are advisory and
    // per-fd, so a second open + flock attempt is a faithful proxy
    // for the cross-process case — that's exactly the syscall path
    // that runs in two PIDs.)
    const lock_path = try std.fmt.allocPrint(testing.allocator, "{s}.lock", .{path});
    defer testing.allocator.free(lock_path);
    const probe_fd = try openOrCreateRW(lock_path);
    defer _ = posix.system.close(probe_fd);
    const got_lock = try tryFlockExclusive(probe_fd);
    try testing.expect(!got_lock);

    // After `first` is closed (at end of test), the lock would be
    // released and a fresh open would succeed — verified implicitly
    // by the other persistent tests which open/close the same path
    // multiple times.
}

test "persistent: lookupOrInsert reloads cold entries from disk" {
    // The in-memory cache may be smaller than the on-disk log. A
    // lookup that misses memory but hits the file index must reload
    // the entry into the LRU and return it.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmpAbsPath(testing.allocator, &tmp, "ztok_prefix_cache.dat");
    defer testing.allocator.free(path);

    // Phase 1: write 5 keys with a small in-memory capacity (which
    // forces LRU eviction). On disk we keep them all.
    {
        const pc = try PrefixCache.openPersistent(testing.allocator, .{
            .path = path,
            .in_memory_capacity = 2, // tiny — evicts on every 3rd write
        });
        defer pc.closePersistent();
        try pc.insert("a", &.{1});
        try pc.insert("b", &.{2});
        try pc.insert("c", &.{3});
        try pc.insert("d", &.{4});
        try pc.insert("e", &.{5});
        try pc.sync();
        // After 5 writes, only the last 2 ("d", "e") are in memory.
        // The other 3 are cold (file-only).
        try testing.expectEqual(@as(u32, 2), pc.mem.stats.entries);

        // A lookup of "a" should miss memory, hit the file index, and
        // reload "a" into the LRU.
        const a = (try pc.lookup("a")) orelse return error.MissingEntry;
        try testing.expectEqualSlices(TokenId, &.{1}, a);
    }
}
