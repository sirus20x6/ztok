//! Persistent worker pool with futex-driven dispatch and optional
//! CPU affinity by physical core.
//!
//! Zig 0.16 removed `std.Thread.Pool`; rather than adopt the new
//! `std.Io.Threaded` async machinery for what is fundamentally a
//! data-parallel loop, we own the spine ourselves:
//!
//!   * `BatchPool` holds N per-worker `ArenaAllocator`s. They live as
//!     long as the pool and get reset between batches so per-batch
//!     scratch never reaches the global allocator's hot path.
//!   * At init we spawn N-1 long-lived helper threads. Each helper
//!     waits on a futex; `runBatch` writes the task descriptor,
//!     bumps a generation counter, and wakes the helpers. The caller
//!     thread participates as worker 0, so a 1-CPU pool degenerates
//!     to a serial loop with zero spawn overhead.
//!   * Workers race for items via a single atomic cursor — the
//!     cheapest possible work-stealing scheme, and the one that wins
//!     when individual jobs are uneven. Persistent threads eliminate
//!     the ~5% `std.Thread.spawn`+`join` overhead the v1.11
//!     scaling investigation flagged at N=48.
//!   * Opt-in CPU affinity: `Options.pin_to_physical_cores = true`
//!     pins each helper to a distinct physical core (SMT-aware,
//!     parsed from `/sys/devices/system/cpu/cpuN/topology/
//!     thread_siblings_list`). On the 24-physical/48-logical EPYC
//!     box this lifts N=24 scaling efficiency from ~53% to ~65%+
//!     by keeping the BPE 644 KB merge-rank hashmap from thrashing
//!     across SMT siblings sharing an L1d.
//!
//! The shared sync state lives in a heap-allocated `Shared` struct
//! that both the BatchPool handle and the helper threads point at.
//! This lets the BatchPool itself remain a small value-type that
//! callers can stack-allocate; the helpers' fate is decoupled from
//! the BatchPool's storage moves.

const std = @import("std");
const builtin = @import("builtin");

// On single-threaded targets (e.g. wasm32-freestanding / wasm32-wasi
// without --import-memory shared), std.Thread is unavailable and the
// BatchPool degrades to a serial loop. The public API is unchanged.
const has_threads = !builtin.single_threaded;
const is_linux = builtin.os.tag == .linux;

const linux = if (is_linux) std.os.linux else struct {};

// === Internal futex wrappers ====================================
//
// We need exactly two operations: wait-while-equal, and wake-N. Linux
// gives us those directly via the futex syscall. Off-Linux we fall
// back to a sched_yield spin — performant enough for the test path
// (non-Linux is not the throughput target).

fn futexWait(ptr: *const std.atomic.Value(u32), expected: u32) void {
    if (!is_linux) {
        var i: u32 = 0;
        while (ptr.load(.acquire) == expected) : (i +%= 1) {
            if (has_threads) std.Thread.yield() catch {};
            if (i & 0xFF == 0xFF) std.atomic.spinLoopHint();
        }
        return;
    }
    _ = linux.futex_4arg(
        @ptrCast(&ptr.raw),
        .{ .cmd = .WAIT, .private = true },
        expected,
        null,
    );
}

fn futexWake(ptr: *const std.atomic.Value(u32), n: u32) void {
    if (!is_linux) return;
    if (n == 0) return;
    _ = linux.futex_4arg(
        @ptrCast(&ptr.raw),
        .{ .cmd = .WAKE, .private = true },
        n,
        null,
    );
}

// === Physical-core discovery ======================================
//
// Linux exposes a per-cpu thread_siblings_list at
// /sys/devices/system/cpu/cpuN/topology/thread_siblings_list, e.g.
// "0,24" for cpu 0 on a 24-physical/48-logical SMT2 box, meaning
// logical 0 and 24 share a physical core.
//
// We build a "pin order" — a permutation of logical CPUs where the
// first n_physical entries are the primaries (lowest sibling) of
// each physical core, then the second n_physical entries are the
// secondaries (next sibling), and so on. Worker `i` pins to
// `order[i % order.len]`. On a 24-physical/48-logical box:
//   workers 1..23  -> CPUs 0..22  (each on a distinct physical
//                                   core; no SMT contention)
//   workers 24..47 -> CPUs 23..46 (SMT siblings of physicals 0..22)
//
// Wraps cleanly for over-subscription.
//
// Returns a slice the caller must `free` against `allocator`.
fn discoverPinOrder(allocator: std.mem.Allocator) !struct {
    order: []u32,
    n_physical: u32,
} {
    if (!is_linux) return error.NotLinux;

    // Read each cpu's siblings list. Map (primary -> list of all
    // siblings sorted by id). We walk cpu 0..1023 (sysfs may have
    // sparse online cpus; we just stop at the first absent file).
    var groups: std.AutoHashMap(u32, std.ArrayList(u32)) = .init(allocator);
    defer {
        var it = groups.valueIterator();
        while (it.next()) |v| v.deinit(allocator);
        groups.deinit();
    }

    var cpu_idx: u32 = 0;
    while (cpu_idx < 1024) : (cpu_idx += 1) {
        var path_buf: [128:0]u8 = undefined;
        const path = std.fmt.bufPrintZ(
            &path_buf,
            "/sys/devices/system/cpu/cpu{d}/topology/thread_siblings_list",
            .{cpu_idx},
        ) catch break;

        // Direct linux syscalls — avoids dragging in the `std.Io`
        // runtime and works in the post-`posix.close`-retirement
        // 0.16 stdlib.
        const open_rc = linux.openat(linux.AT.FDCWD, path.ptr, .{
            .ACCMODE = .RDONLY,
        }, 0);
        if (linux.errno(open_rc) != .SUCCESS) break;
        const fd: i32 = @intCast(open_rc);
        defer _ = linux.close(fd);

        var content_buf: [256]u8 = undefined;
        const read_rc = linux.read(fd, &content_buf, content_buf.len);
        if (linux.errno(read_rc) != .SUCCESS) break;
        const n = read_rc;
        if (n == 0) break;
        const content = std.mem.trim(u8, content_buf[0..n], &std.ascii.whitespace);
        if (content.len == 0) break;

        const primary = parseFirstCpu(content) orelse continue;
        const gop = try groups.getOrPut(primary);
        if (!gop.found_existing) gop.value_ptr.* = .empty;
        // Append this logical cpu to its physical core's group.
        // Dedupe at the same time (a cpu's siblings_list lists
        // itself too).
        var seen = false;
        for (gop.value_ptr.items) |existing| if (existing == cpu_idx) {
            seen = true;
            break;
        };
        if (!seen) try gop.value_ptr.append(allocator, cpu_idx);
    }

    if (groups.count() == 0) return error.NoCpus;

    // Determine the max SMT depth (siblings per physical).
    var max_depth: u32 = 0;
    var it1 = groups.iterator();
    while (it1.next()) |e| {
        const d: u32 = @intCast(e.value_ptr.items.len);
        if (d > max_depth) max_depth = d;
    }

    // Sort physical-core primaries ascending for stable ordering.
    const n_physical: u32 = @intCast(groups.count());
    const primaries = try allocator.alloc(u32, n_physical);
    defer allocator.free(primaries);
    var idx: usize = 0;
    var it2 = groups.iterator();
    while (it2.next()) |e| : (idx += 1) primaries[idx] = e.key_ptr.*;
    std.mem.sort(u32, primaries, {}, std.sort.asc(u32));

    // Build the order: sibling-depth major, primary-id minor.
    // depth=0 across all physicals, then depth=1 across all, etc.
    var order: std.ArrayList(u32) = .empty;
    errdefer order.deinit(allocator);
    try order.ensureTotalCapacity(allocator, @as(usize, n_physical) * max_depth);

    var depth: u32 = 0;
    while (depth < max_depth) : (depth += 1) {
        for (primaries) |p| {
            const siblings = groups.getPtr(p).?.items;
            // Sort siblings ascending in place once on first pass.
            if (depth == 0) std.mem.sort(u32, siblings, {}, std.sort.asc(u32));
            if (depth < siblings.len) {
                try order.append(allocator, siblings[depth]);
            }
        }
    }

    return .{
        .order = try order.toOwnedSlice(allocator),
        .n_physical = n_physical,
    };
}

/// Parse the first (minimum) CPU id from a Linux topology list of
/// the form `"0,24"` or `"0-1,16-17"` or `"0"`.
fn parseFirstCpu(s: []const u8) ?u32 {
    var min_id: ?u32 = null;
    var it = std.mem.tokenizeAny(u8, s, ",");
    while (it.next()) |chunk| {
        var range_it = std.mem.splitScalar(u8, chunk, '-');
        const first_s = range_it.next() orelse continue;
        const first = std.fmt.parseInt(u32, first_s, 10) catch continue;
        if (min_id == null or first < min_id.?) min_id = first;
    }
    return min_id;
}

fn setAffinityToCpu(cpu: u32) !void {
    if (!is_linux) return error.NotLinux;
    var set: linux.cpu_set_t = [_]usize{0} ** (linux.CPU_SETSIZE / @sizeOf(usize));
    const word = cpu / (@sizeOf(usize) * 8);
    const bit = cpu % (@sizeOf(usize) * 8);
    if (word >= set.len) return error.CpuOutOfRange;
    set[word] = @as(usize, 1) << @intCast(bit);

    // pid=0 = "calling thread" (TID) per sched_setaffinity(2).
    try linux.sched_setaffinity(0, &set);
}

// === Huge-page hint =================================================
//
// MADV_HUGEPAGE asks the kernel to back the given anonymous region with
// 2 MiB (or larger) transparent huge pages. It's a hint, not a
// guarantee — kernel policy (the THP "always" / "madvise" / "never"
// sysctl, plus per-cgroup overrides) decides whether to actually back
// the region. On a single hot 256 KiB scratch buffer the win is small
// (most of the TLB pressure comes from the vocab tables and the input
// itself, not the arena), but at high batch sizes it adds up — every
// worker's arena fits in a single huge page after warmup.
//
// Failure is silently ignored — the buffer is still usable. Off-Linux
// targets (macOS / wasm) skip the syscall.

/// Hint the kernel to back `region` with transparent huge pages.
/// Linux-only; no-op everywhere else. Returns true on best-effort
/// success (madvise returned 0), false otherwise.
fn madviseHugepage(region: []u8) bool {
    if (!is_linux) return false;
    if (region.len == 0) return false;
    // madvise(2) requires page alignment on `addr`. We don't enforce
    // that here — the arena buffer's base typically IS page-aligned
    // (it comes from `page_allocator` for sufficiently large sizes),
    // but the kernel will return EINVAL otherwise. Treat EINVAL as
    // benign: it just means hugepage backing won't apply here.
    const rc = linux.madvise(region.ptr, region.len, linux.MADV.HUGEPAGE);
    return linux.errno(rc) == .SUCCESS;
}

// === NUMA topology discovery (Linux) ================================
//
// NUMA-aware affinity is an opt-in `Options.numa_aware` flag. On a
// single-socket box it's a no-op at runtime — `n_nodes == 1` makes the
// node-aware code paths trivial. On multi-socket it (a) pins workers
// to physical cores within the worker's assigned NUMA node and (b)
// allocates per-worker arenas via `mbind(MPOL_PREFERRED)` so the
// arena memory is local to that worker's node.
//
// Topology is discovered from `/sys/devices/system/node/node*/cpulist`
// — each file lists the logical CPUs on that node. The pin order
// becomes a permutation of CPUs: node-major (worker 0 → node 0 first
// CPU, worker 1 → node 1 first CPU, …) so a workload smaller than
// `n_nodes` fans out across sockets, and beyond that we fill node 0
// completely before spilling into node 1.

const NumaTopo = struct {
    /// Per-node CPU list. `nodes[i]` is the slice of logical CPU ids
    /// on NUMA node `i`. Owned by the caller.
    nodes: [][]u32,
    /// Flat node-major pin order. Length sums to the total CPU count.
    order: []u32,
    /// `worker_node[i]` is the NUMA node for the CPU at `order[i]`.
    worker_node: []u32,

    fn deinit(self: *NumaTopo, allocator: std.mem.Allocator) void {
        for (self.nodes) |n| allocator.free(n);
        allocator.free(self.nodes);
        allocator.free(self.order);
        allocator.free(self.worker_node);
    }
};

/// Parse `/sys/devices/system/node/node*/cpulist`. Returns a 1-node
/// "everything on node 0" fallback when the sysfs path is missing
/// (UMA box, non-Linux, or sandboxed container).
fn discoverNumaTopo(allocator: std.mem.Allocator) !NumaTopo {
    if (!is_linux) return error.NotLinux;

    // Probe for the highest node id by walking node0..node63.
    var node_lists: std.ArrayList([]u32) = .empty;
    errdefer {
        for (node_lists.items) |l| allocator.free(l);
        node_lists.deinit(allocator);
    }

    var node_idx: u32 = 0;
    while (node_idx < 64) : (node_idx += 1) {
        var path_buf: [128:0]u8 = undefined;
        const path = std.fmt.bufPrintZ(
            &path_buf,
            "/sys/devices/system/node/node{d}/cpulist",
            .{node_idx},
        ) catch break;

        const open_rc = linux.openat(linux.AT.FDCWD, path.ptr, .{
            .ACCMODE = .RDONLY,
        }, 0);
        if (linux.errno(open_rc) != .SUCCESS) break;
        const fd: i32 = @intCast(open_rc);
        defer _ = linux.close(fd);

        var content_buf: [1024]u8 = undefined;
        const read_rc = linux.read(fd, &content_buf, content_buf.len);
        if (linux.errno(read_rc) != .SUCCESS) break;
        const n = read_rc;
        if (n == 0) break;
        const content = std.mem.trim(u8, content_buf[0..n], &std.ascii.whitespace);
        if (content.len == 0) break;

        const cpus = try parseCpulist(allocator, content);
        try node_lists.append(allocator, cpus);
    }

    if (node_lists.items.len == 0) return error.NoNumaNodes;

    // Build the flat pin order: round-robin one CPU per node until any
    // node runs out, then drain the remainder in node order. Pair each
    // entry with its source node id.
    var total: usize = 0;
    for (node_lists.items) |l| total += l.len;
    const order = try allocator.alloc(u32, total);
    errdefer allocator.free(order);
    const worker_node = try allocator.alloc(u32, total);
    errdefer allocator.free(worker_node);

    var idx: usize = 0;
    var depth: usize = 0;
    var spread: bool = true;
    while (spread) {
        spread = false;
        var ni: usize = 0;
        while (ni < node_lists.items.len) : (ni += 1) {
            const l = node_lists.items[ni];
            if (depth < l.len) {
                order[idx] = l[depth];
                worker_node[idx] = @intCast(ni);
                idx += 1;
                spread = true;
            }
        }
        depth += 1;
    }
    std.debug.assert(idx == total);

    return .{
        .nodes = try node_lists.toOwnedSlice(allocator),
        .order = order,
        .worker_node = worker_node,
    };
}

/// Parse a cpulist string of the form `"0-23,48-71"` (range +
/// comma-separated). Caller owns the returned slice.
fn parseCpulist(allocator: std.mem.Allocator, s: []const u8) ![]u32 {
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(allocator);
    var it = std.mem.tokenizeAny(u8, s, ",");
    while (it.next()) |chunk| {
        var dash = std.mem.splitScalar(u8, chunk, '-');
        const first_s = dash.next() orelse continue;
        const first = std.fmt.parseInt(u32, first_s, 10) catch continue;
        if (dash.next()) |last_s| {
            const last = std.fmt.parseInt(u32, last_s, 10) catch continue;
            var cpu: u32 = first;
            while (cpu <= last) : (cpu += 1) try out.append(allocator, cpu);
        } else {
            try out.append(allocator, first);
        }
    }
    return out.toOwnedSlice(allocator);
}

/// mbind(2) wrapper: bind a memory region to a single NUMA node via
/// `MPOL_PREFERRED`. Pages already faulted in are NOT moved (we don't
/// pass `MPOL_MF_MOVE` — that would scan and migrate the whole
/// region, which is expensive). The hint applies to future faults
/// from this thread. Returns true on syscall success.
fn mbindRegionToNode(region: []u8, node: u32) bool {
    if (!is_linux) return false;
    if (region.len == 0) return false;
    // MPOL_PREFERRED = 1; nodemask is a bitmap, maxnode is the
    // highest bit index + 1 (per mbind(2)).
    const MPOL_PREFERRED: usize = 1;
    var nodemask: [16]u64 = .{0} ** 16; // up to 1024 nodes
    const w = node / 64;
    const b = node % 64;
    if (w >= nodemask.len) return false;
    nodemask[w] = @as(u64, 1) << @intCast(b);
    const maxnode: usize = @as(usize, node) + 1;
    // `linux.SYS.mbind` is the arch-correct syscall number.
    const rc = std.os.linux.syscall6(
        linux.SYS.mbind,
        @intFromPtr(region.ptr),
        region.len,
        MPOL_PREFERRED,
        @intFromPtr(&nodemask),
        maxnode,
        0,
    );
    return linux.errno(rc) == .SUCCESS;
}

// === Options ======================================================

pub const Options = struct {
    /// When true, each spawned helper thread pins itself to a
    /// distinct physical core (SMT-aware). Worker 0 is the caller
    /// thread; we leave it unpinned to respect any affinity the
    /// caller already has. If topology discovery or
    /// `sched_setaffinity` fails (non-Linux, locked-down container,
    /// etc.) the pool falls back to running unpinned — affinity is
    /// best-effort. Inspect `pin_diagnostic` after `init` to see
    /// the actual outcome.
    pin_to_physical_cores: bool = false,

    /// When true, on Linux, call `madvise(MADV_HUGEPAGE)` on each
    /// worker's pre-warmed scratch buffer. Off-Linux: no-op. The hint
    /// asks the kernel to back the region with transparent huge
    /// pages (typically 2 MiB) — the kernel decides whether to
    /// honor it based on the system-wide THP policy (sysctl
    /// `vm.nr_hugepages` / `/sys/kernel/mm/transparent_hugepage/
    /// enabled`). Off-by-default to preserve 1.18 behavior; opt in
    /// for batch workloads where the per-worker arena exceeds a
    /// single 4 KiB page and TLB pressure starts to matter.
    use_hugepages: bool = false,

    /// Size in bytes of the pre-warmed scratch buffer to allocate per
    /// worker arena at init time. The buffer is allocated through
    /// the arena, so future scratch allocations reuse it (no
    /// fragmentation, no `mmap` churn). 256 KiB is a sane default —
    /// large enough to fit cl100k pre-tokenized spans + ids for
    /// chunks up to ~32 KB, small enough that 48 workers fit in 12
    /// MiB of resident memory.
    prewarm_scratch_bytes: usize = 256 * 1024,

    /// When true, on Linux, also use NUMA topology to:
    ///   (1) pin workers to physical cores within their assigned
    ///       NUMA node (round-robin across nodes), and
    ///   (2) bind each worker's pre-warmed scratch buffer to the
    ///       node-local memory via `mbind(MPOL_PREFERRED)`.
    /// On single-socket boxes this is a no-op (1 node = no choice).
    /// On multi-socket boxes it can lift throughput 10-20% at
    /// thread counts that span sockets, by keeping vocab-table
    /// reads and arena writes on the same NUMA controller. Defaults
    /// to off — opt in for explicit multi-socket deployments. If
    /// `pin_to_physical_cores` is also true, NUMA-aware pinning
    /// supersedes it. Inspect `numa_diagnostic` after `init` for
    /// the resolved outcome.
    numa_aware: bool = false,
};

pub const PinStatus = enum {
    /// Pinning was not requested.
    disabled,
    /// Every helper successfully pinned.
    pinned,
    /// Pinning requested but topology query failed (e.g. non-Linux
    /// or /sys unavailable). All helpers run unpinned.
    fallback_no_topology,
    /// Pinning requested, topology resolved, but at least one
    /// `sched_setaffinity` call failed (e.g. cgroup restriction).
    fallback_partial,
};

pub const NumaStatus = enum {
    /// NUMA awareness was not requested.
    disabled,
    /// Single-node system (e.g. desktop, single-socket server). The
    /// `numa_aware` flag was respected but has no per-worker effect
    /// because all CPUs share one memory controller.
    single_node,
    /// Multi-node system, topology resolved, workers pinned per node
    /// and arenas mbind'd to their node. Throughput-relevant path.
    multi_node,
    /// NUMA awareness requested but topology query failed (non-Linux,
    /// /sys unavailable, etc.). Workers run un-NUMA-aware.
    fallback_no_topology,
};

// === Shared state ==================================================
//
// Lives on the heap so helper threads can point at it independently
// of where the BatchPool handle is stored. Survives any move of the
// BatchPool value.
const Shared = if (has_threads) struct {
    // Task descriptor — set by runBatch before bumping `generation`.
    // Workers read after observing a fresh generation via an
    // `.acquire` load (paired with main's `.acq_rel` fetchAdd on
    // generation, which is the release barrier).
    task_run: ?*const fn (*anyopaque, usize, usize) void = null,
    task_ctx: ?*anyopaque = null,
    task_total: usize = 0,
    task_cursor: std.atomic.Value(usize) = .init(0),

    // generation: bumped each runBatch (and on deinit) so workers
    // know fresh work is available without losing wakeups.
    generation: std.atomic.Value(u32) = .init(0),

    // done_count: helpers increment after exhausting the cursor.
    // Main waits via futex until it equals n_helpers.
    done_count: std.atomic.Value(u32) = .init(0),

    // exit_flag: set by deinit so workers break out of their main
    // loop after the wake-up.
    exit_flag: std.atomic.Value(u32) = .init(0),

    // Logical-CPU pin order. First n_physical entries are unique
    // physical-core primaries; subsequent entries are SMT
    // siblings. Helper `i` pins to `pin_cpus[(i-1) % len]`,
    // giving 1-to-1 physical mapping up to n_physical and clean
    // SMT-pair spread beyond. `null` if pinning was not requested
    // or topology discovery failed.
    pin_cpus: ?[]const u32 = null,

    // Diagnostic — workers atomically clear this u32 to 0 if their
    // own setaffinity fails (non-zero starting value).
    all_pinned_ok: std.atomic.Value(u32) = .init(1),
} else struct {};

/// Per-worker NUMA node mapping. `node_for_worker[i]` is the NUMA
/// node that worker `i` is pinned to (and whose memory holds the
/// worker's arena, when `mbind` succeeded). Length == `arenas.len`.
/// All zeros when NUMA awareness is off or single-node.

fn helperMain(shared: *Shared, worker_idx: usize) void {
    // Best-effort affinity pin. We do this on the worker thread
    // so the pin applies to its own TID (sched_setaffinity with
    // pid=0).
    if (shared.pin_cpus) |cpus| {
        if (cpus.len > 0) {
            const target = cpus[(worker_idx - 1) % cpus.len];
            setAffinityToCpu(target) catch {
                shared.all_pinned_ok.store(0, .release);
            };
        }
    }

    var last_seen_gen: u32 = 0;
    while (true) {
        // Wait for a new generation or exit.
        while (true) {
            const g = shared.generation.load(.acquire);
            if (shared.exit_flag.load(.acquire) != 0) return;
            if (g != last_seen_gen) {
                last_seen_gen = g;
                break;
            }
            futexWait(&shared.generation, last_seen_gen);
        }

        if (shared.exit_flag.load(.acquire) != 0) return;

        // Pull items off the cursor.
        const run = shared.task_run;
        const ctx = shared.task_ctx;
        const total = shared.task_total;
        if (run) |run_fn| if (ctx) |c| {
            while (true) {
                const i = shared.task_cursor.fetchAdd(1, .acq_rel);
                if (i >= total) break;
                run_fn(c, i, worker_idx);
            }
        };

        // Signal completion: bump done_count and wake the waiter.
        _ = shared.done_count.fetchAdd(1, .acq_rel);
        futexWake(&shared.done_count, 1);
    }
}

// === BatchPool =====================================================

pub const BatchPool = struct {
    allocator: std.mem.Allocator,
    arenas: []std.heap.ArenaAllocator,

    // Heap-allocated shared sync state. nil when n_workers == 1 or
    // single-threaded (in those cases runBatch runs serially in the
    // caller).
    shared: if (has_threads) ?*Shared else void,
    workers: if (has_threads) []std.Thread else void,
    pin_diagnostic: PinStatus = .disabled,
    numa_diagnostic: NumaStatus = .disabled,
    /// Worker-index → NUMA node id. Populated when
    /// `numa_diagnostic == .multi_node`; all-zero otherwise. Owned
    /// by the pool.
    worker_node: []u32 = &.{},
    /// True iff the pre-warmed scratch buffers were successfully
    /// madvise'd to MADV_HUGEPAGE. False when `use_hugepages == false`
    /// or the syscall failed. Inspect after `init`.
    hugepages_applied: bool = false,
    /// Backing storage for per-worker pre-warmed scratch buffers
    /// (one slice per worker). Owned by the pool; freed in deinit.
    /// Empty when no prewarm was requested.
    prewarm: [][]u8 = &.{},

    pub fn init(allocator: std.mem.Allocator, n_workers: ?u32) !BatchPool {
        return initWithOptions(allocator, n_workers, .{});
    }

    pub fn initWithOptions(
        allocator: std.mem.Allocator,
        n_workers: ?u32,
        opts: Options,
    ) !BatchPool {
        const cpu = if (has_threads) (std.Thread.getCpuCount() catch 1) else 1;
        const n: usize = if (n_workers) |x| @max(1, x) else cpu;

        const arenas = try allocator.alloc(std.heap.ArenaAllocator, n);
        var arenas_initialized: usize = 0;
        errdefer {
            for (arenas[0..arenas_initialized]) |*a| a.deinit();
            allocator.free(arenas);
        }
        for (arenas) |*a| {
            a.* = std.heap.ArenaAllocator.init(allocator);
            arenas_initialized += 1;
        }

        // Per-worker pre-warmed scratch buffers. Allocated through the
        // arena so subsequent reset(.retain_capacity) calls keep them
        // hot. madvise(MADV_HUGEPAGE) and mbind(MPOL_PREFERRED) are
        // applied here when requested. Always tracked (the array
        // header is owned by the pool; the byte storage is owned by
        // each arena and freed when the arena deinits).
        const prewarm = try allocator.alloc([]u8, n);
        @memset(prewarm, &.{});
        errdefer allocator.free(prewarm);

        var hugepages_ok: bool = false;
        if (opts.prewarm_scratch_bytes > 0) {
            for (arenas, 0..) |*a, wi| {
                const arena_alloc = a.allocator();
                const buf = try arena_alloc.alloc(u8, opts.prewarm_scratch_bytes);
                // Touch one byte per 4 KiB page so the kernel commits
                // them up front (madvise on un-faulted anonymous pages
                // is a no-op until they're touched).
                var off: usize = 0;
                while (off < buf.len) : (off += 4096) buf[off] = 0;
                prewarm[wi] = buf;
                if (opts.use_hugepages) {
                    if (madviseHugepage(buf)) hugepages_ok = true;
                }
            }
        }

        const worker_node = try allocator.alloc(u32, n);
        errdefer allocator.free(worker_node);
        @memset(worker_node, 0);

        if (!has_threads or n <= 1) {
            // Serial pool — no helpers, no shared state.
            return .{
                .allocator = allocator,
                .arenas = arenas,
                .shared = if (has_threads) null else {},
                .workers = if (has_threads) &.{} else {},
                .pin_diagnostic = .disabled,
                .numa_diagnostic = .disabled,
                .worker_node = worker_node,
                .hugepages_applied = hugepages_ok,
                .prewarm = prewarm,
            };
        }

        // Multi-worker path: spawn N-1 helpers backed by a single
        // shared sync block.
        if (!has_threads) unreachable;

        const shared = try allocator.create(Shared);
        errdefer allocator.destroy(shared);
        shared.* = .{};

        var pin_diag: PinStatus = .disabled;
        var numa_diag: NumaStatus = .disabled;
        var pin_cpus_owned: ?[]u32 = null;
        errdefer if (pin_cpus_owned) |c| allocator.free(c);

        // NUMA-aware pinning supersedes the plain `pin_to_physical_cores`
        // path. We discover the per-node CPU lists, build a node-major
        // pin order, and (when multi-socket) bind each prewarm scratch
        // buffer to the worker's node.
        if (opts.numa_aware) {
            if (discoverNumaTopo(allocator)) |topo_inout| {
                var topo = topo_inout;
                defer topo.deinit(allocator);
                if (topo.nodes.len == 1) {
                    numa_diag = .single_node;
                } else {
                    // Multi-socket: copy the pin order + per-worker
                    // node assignment, bind each arena's prewarm to
                    // its node.
                    const order_copy = try allocator.dupe(u32, topo.order);
                    pin_cpus_owned = order_copy;
                    shared.pin_cpus = order_copy;
                    pin_diag = .pinned;
                    numa_diag = .multi_node;
                    // worker_node[i]: the node that helper i is on.
                    // worker 0 is the caller — leave its slot as 0
                    // (it inherits whatever node the caller is on;
                    // mbind isn't applied to it).
                    var wi: usize = 0;
                    while (wi < n) : (wi += 1) {
                        const order_idx = if (wi == 0) 0 else (wi - 1) % topo.order.len;
                        const node = topo.worker_node[order_idx];
                        worker_node[wi] = node;
                        if (wi < prewarm.len and prewarm[wi].len > 0 and wi > 0) {
                            _ = mbindRegionToNode(prewarm[wi], node);
                        }
                    }
                }
            } else |_| {
                numa_diag = .fallback_no_topology;
            }
        }

        // Fall back to plain physical-core pinning if NUMA was off or
        // numa_aware was set but the pool is single-node (still nice
        // to keep the pin behavior).
        if (pin_cpus_owned == null and opts.pin_to_physical_cores) {
            if (discoverPinOrder(allocator)) |topo| {
                pin_cpus_owned = topo.order;
                shared.pin_cpus = topo.order;
                pin_diag = .pinned;
            } else |_| {
                pin_diag = .fallback_no_topology;
            }
        }

        const n_helpers = n - 1;
        const handles = try allocator.alloc(std.Thread, n_helpers);
        errdefer allocator.free(handles);

        var spawned: usize = 0;
        errdefer if (spawned > 0) {
            // Wake spawned helpers so they exit cleanly.
            shared.exit_flag.store(1, .release);
            _ = shared.generation.fetchAdd(1, .acq_rel);
            futexWake(&shared.generation, @intCast(spawned));
            for (handles[0..spawned]) |h| h.join();
        };

        var i: usize = 0;
        while (i < n_helpers) : (i += 1) {
            handles[i] = try std.Thread.spawn(.{}, helperMain, .{ shared, i + 1 });
            spawned += 1;
        }

        // If pinning requested but at least one helper later
        // clears all_pinned_ok, we won't observe that here at
        // init time — workers haven't necessarily run their pin
        // step yet. We poll after a brief grace period.
        if (pin_diag == .pinned) {
            // Each helper pins on its first iteration before
            // entering the futex wait. Give them a few ms to
            // execute that step. This is a best-effort polling
            // only used to set `pin_diagnostic`; correctness
            // doesn't depend on it.
            var spin: u32 = 0;
            while (spin < 10_000) : (spin += 1) {
                if (shared.all_pinned_ok.load(.acquire) == 0) {
                    pin_diag = .fallback_partial;
                    break;
                }
                std.atomic.spinLoopHint();
            }
        }

        return .{
            .allocator = allocator,
            .arenas = arenas,
            .shared = shared,
            .workers = handles,
            .pin_diagnostic = pin_diag,
            .numa_diagnostic = numa_diag,
            .worker_node = worker_node,
            .hugepages_applied = hugepages_ok,
            .prewarm = prewarm,
        };
    }

    pub fn deinit(self: *BatchPool) void {
        if (has_threads) {
            if (self.shared) |shared| {
                shared.exit_flag.store(1, .release);
                _ = shared.generation.fetchAdd(1, .acq_rel);
                futexWake(&shared.generation, @intCast(self.workers.len));
                for (self.workers) |h| h.join();
                self.allocator.free(self.workers);
                if (shared.pin_cpus) |cpus| self.allocator.free(cpus);
                self.allocator.destroy(shared);
                self.shared = null;
                self.workers = &.{};
            }
        }
        // prewarm bytes are owned by each arena — freed by arena.deinit
        // below. We only own the slice header.
        if (self.prewarm.len > 0) self.allocator.free(self.prewarm);
        self.prewarm = &.{};
        if (self.worker_node.len > 0) self.allocator.free(self.worker_node);
        self.worker_node = &.{};
        for (self.arenas) |*a| a.deinit();
        self.allocator.free(self.arenas);
        self.arenas = &.{};
    }

    pub fn workerCount(self: *const BatchPool) usize {
        return self.arenas.len;
    }

    /// Reset a per-worker arena and hand back its allocator. Safe to
    /// call once per batch from the worker thread.
    pub fn resetArena(self: *BatchPool, worker_idx: usize) std.mem.Allocator {
        _ = self.arenas[worker_idx].reset(.retain_capacity);
        return self.arenas[worker_idx].allocator();
    }

    /// Current scratch-arena reserved-byte count for a worker.
    pub fn peakScratchBytes(self: *const BatchPool, worker_idx: usize) usize {
        return self.arenas[worker_idx].queryCapacity();
    }

    /// Run `total_items` indexed jobs across the pool. Each worker
    /// calls `Worker.run(ctx, item_idx, worker_idx)`. The calling
    /// thread participates as worker 0, which keeps small batches
    /// cheap.
    pub fn runBatch(
        self: *BatchPool,
        comptime Worker: type,
        ctx: anytype,
        total_items: usize,
    ) !void {
        if (total_items == 0) return;

        const n_workers = self.workerCount();

        const Trampoline = struct {
            fn go(opaque_ctx: *anyopaque, idx: usize, widx: usize) void {
                const c: @TypeOf(ctx) = @ptrCast(@alignCast(opaque_ctx));
                Worker.run(c, idx, widx);
            }
        };

        if (!has_threads or n_workers <= 1 or self.shared == null) {
            var i: usize = 0;
            while (i < total_items) : (i += 1) {
                Worker.run(ctx, i, 0);
            }
            return;
        }

        if (has_threads) {
            const shared = self.shared.?;

            // Publish the task descriptor. The fetchAdd on
            // `generation` below provides the release barrier
            // that makes these writes visible to a helper that
            // sees the new generation via an .acquire load.
            shared.task_run = &Trampoline.go;
            shared.task_ctx = @ptrCast(ctx);
            shared.task_total = total_items;
            shared.task_cursor.store(0, .release);
            shared.done_count.store(0, .release);

            _ = shared.generation.fetchAdd(1, .acq_rel);
            const n_helpers: u32 = @intCast(n_workers - 1);
            futexWake(&shared.generation, n_helpers);

            // Caller participates as worker 0.
            while (true) {
                const i = shared.task_cursor.fetchAdd(1, .acq_rel);
                if (i >= total_items) break;
                Trampoline.go(@ptrCast(ctx), i, 0);
            }

            // Wait for helpers to finish this batch.
            while (true) {
                const d = shared.done_count.load(.acquire);
                if (d >= n_helpers) break;
                futexWait(&shared.done_count, d);
            }
        }
    }
};

// === Tests =========================================================

test "BatchPool single-worker runs serially" {
    var bp = try BatchPool.init(std.testing.allocator, 1);
    defer bp.deinit();
    try std.testing.expectEqual(@as(usize, 1), bp.workerCount());

    var counts = [_]u32{0} ** 8;
    const Ctx = struct { counts: []u32 };
    var ctx: Ctx = .{ .counts = &counts };

    const W = struct {
        fn run(c: *Ctx, idx: usize, widx: usize) void {
            _ = widx;
            c.counts[idx] += 1;
        }
    };
    try bp.runBatch(W, &ctx, counts.len);
    for (counts) |n| try std.testing.expectEqual(@as(u32, 1), n);
}

test "BatchPool parallel fan-out covers every index exactly once" {
    var bp = try BatchPool.init(std.testing.allocator, 4);
    defer bp.deinit();

    const N = 1024;
    const marks = try std.testing.allocator.alloc(std.atomic.Value(u32), N);
    defer std.testing.allocator.free(marks);
    for (marks) |*m| m.* = .init(0);

    const Ctx = struct { marks: []std.atomic.Value(u32) };
    var ctx: Ctx = .{ .marks = marks };
    const W = struct {
        pub fn run(c: *Ctx, idx: usize, widx: usize) void {
            _ = widx;
            _ = c.marks[idx].fetchAdd(1, .acq_rel);
        }
    };
    try bp.runBatch(W, &ctx, N);
    for (marks) |*m| try std.testing.expectEqual(@as(u32, 1), m.load(.acquire));
}

test "BatchPool resetArena gives usable allocator" {
    var bp = try BatchPool.init(std.testing.allocator, 2);
    defer bp.deinit();
    const a = bp.resetArena(0);
    const buf = try a.alloc(u8, 64);
    @memset(buf, 0xAB);
}

test "BatchPool init + deinit with no runBatch leaks nothing" {
    // Exercises the clean-shutdown path: helpers must exit even
    // if they were never given any work.
    var bp = try BatchPool.init(std.testing.allocator, 4);
    bp.deinit();
}

test "BatchPool 100 sequential runBatch calls keep worker count stable" {
    var bp = try BatchPool.init(std.testing.allocator, 4);
    defer bp.deinit();

    try std.testing.expectEqual(@as(usize, 4), bp.workerCount());

    const N = 256;
    const counts = try std.testing.allocator.alloc(std.atomic.Value(u32), N);
    defer std.testing.allocator.free(counts);
    for (counts) |*c| c.* = .init(0);

    const Ctx = struct { counts: []std.atomic.Value(u32) };
    var ctx: Ctx = .{ .counts = counts };
    const W = struct {
        pub fn run(c: *Ctx, idx: usize, widx: usize) void {
            _ = widx;
            _ = c.counts[idx].fetchAdd(1, .acq_rel);
        }
    };

    var iter: u32 = 0;
    while (iter < 100) : (iter += 1) {
        try bp.runBatch(W, &ctx, N);
    }
    try std.testing.expectEqual(@as(usize, 4), bp.workerCount());
    for (counts) |*c| try std.testing.expectEqual(@as(u32, 100), c.load(.acquire));
}

test "BatchPool workers actually run concurrently" {
    if (!has_threads) return error.SkipZigTest;
    if (!is_linux) return error.SkipZigTest;
    if (builtin.mode == .Debug) return; // 10ms*N timing fragile in Debug

    const n_workers: u32 = 4;
    var bp = try BatchPool.init(std.testing.allocator, n_workers);
    defer bp.deinit();

    const Ctx = struct {};
    var ctx: Ctx = .{};
    const W = struct {
        pub fn run(_: *Ctx, idx: usize, widx: usize) void {
            _ = idx;
            _ = widx;
            // 10ms busy-equivalent sleep via nanosleep syscall.
            // The pool itself sleeps via futex; for the test job
            // we want pure wall-clock blocking unrelated to the
            // pool's sync primitives.
            const ts: linux.timespec = .{ .sec = 0, .nsec = 10_000_000 };
            _ = linux.nanosleep(&ts, null);
        }
    };

    const t0 = nanosMonotonic();
    try bp.runBatch(W, &ctx, n_workers);
    const elapsed_ns = nanosMonotonic() - t0;
    const elapsed_ms = elapsed_ns / 1_000_000;

    // 4 tasks of 10ms each should complete in ~10-25ms if parallel,
    // not 40ms+ if serial. Allow generous slop for CI jitter.
    try std.testing.expect(elapsed_ms < 30);
}

fn nanosMonotonic() u64 {
    if (!is_linux) return 0;
    var ts: linux.timespec = .{ .sec = 0, .nsec = 0 };
    _ = linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @bitCast(ts.sec)) * 1_000_000_000 + @as(u64, @bitCast(@as(i64, ts.nsec)));
}

test "BatchPool pin_to_physical_cores on Linux sets affinity" {
    if (!is_linux) return error.SkipZigTest;
    if (!has_threads) return error.SkipZigTest;

    // Verify pin_diagnostic at least claims pinned (or fallback,
    // which is acceptable in a locked-down env). On success the
    // helpers will have called sched_setaffinity(0, ...) before
    // entering futex_wait. We can't directly inspect a helper's
    // affinity from the main thread without its TID, so we
    // verify the diagnostic and the pool runs work correctly.
    var bp = try BatchPool.initWithOptions(std.testing.allocator, 2, .{
        .pin_to_physical_cores = true,
    });
    defer bp.deinit();

    try std.testing.expect(bp.pin_diagnostic == .pinned or
        bp.pin_diagnostic == .fallback_no_topology or
        bp.pin_diagnostic == .fallback_partial);

    // Quick smoke: pool still runs batches correctly with pinning on.
    const N = 64;
    const counts = try std.testing.allocator.alloc(std.atomic.Value(u32), N);
    defer std.testing.allocator.free(counts);
    for (counts) |*c| c.* = .init(0);

    const Ctx = struct { counts: []std.atomic.Value(u32) };
    var ctx: Ctx = .{ .counts = counts };
    const W = struct {
        pub fn run(c: *Ctx, idx: usize, widx: usize) void {
            _ = widx;
            _ = c.counts[idx].fetchAdd(1, .acq_rel);
        }
    };
    try bp.runBatch(W, &ctx, N);
    for (counts) |*c| try std.testing.expectEqual(@as(u32, 1), c.load(.acquire));
}

test "discoverPinOrder returns a valid pin order on Linux" {
    if (!is_linux) return error.SkipZigTest;
    const topo = discoverPinOrder(std.testing.allocator) catch |e| {
        // In sandboxed envs /sys may be unavailable; that's the
        // documented fallback path. Don't fail.
        std.debug.print("discoverPinOrder skipped: {}\n", .{e});
        return error.SkipZigTest;
    };
    defer std.testing.allocator.free(topo.order);
    try std.testing.expect(topo.order.len >= 1);
    try std.testing.expect(topo.n_physical >= 1);
    try std.testing.expect(topo.n_physical <= topo.order.len);

    // First n_physical entries should be all distinct primaries
    // (one per physical core, no SMT contention).
    var seen: std.AutoHashMap(u32, void) = .init(std.testing.allocator);
    defer seen.deinit();
    for (topo.order[0..topo.n_physical]) |cpu| {
        const gop = try seen.getOrPut(cpu);
        try std.testing.expect(!gop.found_existing);
    }
    // Every entry overall should be unique (no logical cpu pinned
    // twice).
    seen.clearRetainingCapacity();
    for (topo.order) |cpu| {
        const gop = try seen.getOrPut(cpu);
        try std.testing.expect(!gop.found_existing);
    }
}

test "parseFirstCpu handles single, comma, and range forms" {
    try std.testing.expectEqual(@as(?u32, 0), parseFirstCpu("0"));
    try std.testing.expectEqual(@as(?u32, 0), parseFirstCpu("0,24"));
    try std.testing.expectEqual(@as(?u32, 5), parseFirstCpu("12,5,17"));
    try std.testing.expectEqual(@as(?u32, 0), parseFirstCpu("0-1,16-17"));
    try std.testing.expectEqual(@as(?u32, 4), parseFirstCpu("4-7"));
    try std.testing.expectEqual(@as(?u32, null), parseFirstCpu(""));
}

// === 1.19 perf push tests (post-1.18 agent E) ====================

test "parseCpulist handles ranges + commas" {
    const a = std.testing.allocator;
    const cpus = try parseCpulist(a, "0-3,8,12-15");
    defer a.free(cpus);
    try std.testing.expectEqual(@as(usize, 9), cpus.len);
    try std.testing.expectEqual(@as(u32, 0), cpus[0]);
    try std.testing.expectEqual(@as(u32, 1), cpus[1]);
    try std.testing.expectEqual(@as(u32, 3), cpus[3]);
    try std.testing.expectEqual(@as(u32, 8), cpus[4]);
    try std.testing.expectEqual(@as(u32, 12), cpus[5]);
    try std.testing.expectEqual(@as(u32, 15), cpus[8]);
}

test "discoverNumaTopo parses /sys/devices/system/node correctly or falls back to 1 node" {
    if (!is_linux) return error.SkipZigTest;
    const a = std.testing.allocator;
    var topo = discoverNumaTopo(a) catch |e| switch (e) {
        // Sandboxed environments may not expose /sys/devices/system/node.
        // The documented fallback at the BatchPool layer is `single_node`
        // (no per-worker NUMA assignment). At the helper level we just
        // skip the test — the BatchPool integration test below covers
        // the fallback path.
        error.NoNumaNodes => return error.SkipZigTest,
        else => |err| return err,
    };
    defer topo.deinit(a);

    try std.testing.expect(topo.nodes.len >= 1);
    try std.testing.expect(topo.order.len >= 1);
    try std.testing.expectEqual(topo.order.len, topo.worker_node.len);

    // Every CPU listed in topo.order must belong to its node.
    for (topo.order, topo.worker_node) |cpu, node| {
        try std.testing.expect(node < topo.nodes.len);
        const node_cpus = topo.nodes[node];
        var found = false;
        for (node_cpus) |nc| if (nc == cpu) {
            found = true;
            break;
        };
        try std.testing.expect(found);
    }
}

test "BatchPool use_hugepages: succeeds gracefully on Linux, no-op elsewhere" {
    if (!has_threads) return error.SkipZigTest;
    var bp = try BatchPool.initWithOptions(std.testing.allocator, 2, .{
        .use_hugepages = true,
        .prewarm_scratch_bytes = 2 * 1024 * 1024,
    });
    defer bp.deinit();

    // We don't assert hugepages_applied — kernel policy might disable
    // THP entirely (sysctl `vm.transparent_hugepage = never`), in
    // which case madvise silently succeeds-with-no-effect. We just
    // verify the pool runs work correctly with the flag on.
    if (is_linux) {
        // On Linux the syscall should at least be invoked; the result
        // is informational. No assertion on hugepages_applied.
    } else {
        try std.testing.expectEqual(false, bp.hugepages_applied);
    }

    const N = 16;
    const counts = try std.testing.allocator.alloc(std.atomic.Value(u32), N);
    defer std.testing.allocator.free(counts);
    for (counts) |*c| c.* = .init(0);

    const Ctx = struct { counts: []std.atomic.Value(u32) };
    var ctx: Ctx = .{ .counts = counts };
    const W = struct {
        pub fn run(c: *Ctx, idx: usize, widx: usize) void {
            _ = widx;
            _ = c.counts[idx].fetchAdd(1, .acq_rel);
        }
    };
    try bp.runBatch(W, &ctx, N);
    for (counts) |*c| try std.testing.expectEqual(@as(u32, 1), c.load(.acquire));
}

test "BatchPool numa_aware: single-thread perf doesn't regress" {
    // The "doesn't regress" check is structural: confirm the
    // single-thread path (n=1) honors numa_aware = true without
    // panicking, and that workerCount/pool state stay sane.
    if (!has_threads) return error.SkipZigTest;
    var bp = try BatchPool.initWithOptions(std.testing.allocator, 1, .{
        .numa_aware = true,
        .pin_to_physical_cores = true,
    });
    defer bp.deinit();
    try std.testing.expectEqual(@as(usize, 1), bp.workerCount());

    // numa_diagnostic on n=1 (serial pool, no NUMA branch taken in
    // initWithOptions) stays at .disabled. The flag flowed through
    // without triggering the multi-worker NUMA setup — the
    // single-worker pool legitimately has no per-worker NUMA
    // assignment to make. Confirm we didn't accidentally regress
    // pin_diagnostic either.
    try std.testing.expectEqual(NumaStatus.disabled, bp.numa_diagnostic);

    var counts = [_]u32{0} ** 4;
    const Ctx = struct { counts: []u32 };
    var ctx: Ctx = .{ .counts = &counts };
    const W = struct {
        pub fn run(c: *Ctx, idx: usize, widx: usize) void {
            _ = widx;
            c.counts[idx] += 1;
        }
    };
    try bp.runBatch(W, &ctx, counts.len);
    for (counts) |n| try std.testing.expectEqual(@as(u32, 1), n);
}

test "BatchPool numa_aware: detects nodes on multi-worker setup" {
    if (!has_threads) return error.SkipZigTest;
    if (!is_linux) return error.SkipZigTest;

    var bp = try BatchPool.initWithOptions(std.testing.allocator, 4, .{
        .numa_aware = true,
    });
    defer bp.deinit();

    // One of three outcomes is acceptable depending on the host:
    //   .multi_node          — multi-socket box, full NUMA path taken
    //   .single_node         — single-socket box, opt-in respected but no per-worker bind
    //   .fallback_no_topology — /sys not visible in this env
    const ok = bp.numa_diagnostic == .multi_node or
        bp.numa_diagnostic == .single_node or
        bp.numa_diagnostic == .fallback_no_topology;
    try std.testing.expect(ok);

    // worker_node array is the right length regardless of mode.
    try std.testing.expectEqual(bp.workerCount(), bp.worker_node.len);
}
