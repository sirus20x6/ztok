//! Small bounded persistent worker pool used by the HTTP frontends.
//!
//! Unlike `thread_pool.BatchPool`, which fans one tokenization job out
//! across CPU workers, this pool schedules independent connections.  A
//! bounded ring provides backpressure to the accept loop instead of
//! spawning one operating-system thread per client.

const std = @import("std");

pub fn BoundedPool(comptime Job: type) type {
    return struct {
        const Self = @This();
        const Handler = *const fn (*anyopaque, Job, usize) void;

        allocator: std.mem.Allocator,
        io: std.Io,
        jobs: []Job,
        workers: []std.Thread,
        handler: Handler,
        handler_ctx: *anyopaque,

        mutex: std.Io.Mutex = .init,
        not_empty: std.Io.Condition = .init,
        not_full: std.Io.Condition = .init,
        idle: std.Io.Condition = .init,
        head: usize = 0,
        tail: usize = 0,
        count: usize = 0,
        active: usize = 0,
        stopping: bool = false,

        /// Allocate the pool at a stable address before starting workers.
        /// `queue_capacity` is clamped to at least `worker_count` so every
        /// worker can have one queued successor without blocking accept.
        pub fn init(
            allocator: std.mem.Allocator,
            io: std.Io,
            worker_count: usize,
            queue_capacity: usize,
            handler_ctx: *anyopaque,
            handler: Handler,
        ) !*Self {
            const n_workers = @max(@as(usize, 1), worker_count);
            const cap = @max(n_workers, queue_capacity);
            const self = try allocator.create(Self);
            errdefer allocator.destroy(self);
            self.* = .{
                .allocator = allocator,
                .io = io,
                .jobs = try allocator.alloc(Job, cap),
                .workers = &.{},
                .handler = handler,
                .handler_ctx = handler_ctx,
            };
            errdefer allocator.free(self.jobs);

            self.workers = try allocator.alloc(std.Thread, n_workers);
            errdefer allocator.free(self.workers);
            var spawned: usize = 0;
            errdefer {
                self.mutex.lockUncancelable(io);
                self.stopping = true;
                self.not_empty.broadcast(io);
                self.mutex.unlock(io);
                for (self.workers[0..spawned]) |thread| thread.join();
            }
            for (self.workers, 0..) |*thread, worker_index| {
                thread.* = try std.Thread.spawn(.{}, workerMain, .{ self, worker_index });
                spawned += 1;
            }
            return self;
        }

        /// Queue a job, blocking when the bounded ring is full. This is
        /// deliberate accept-side backpressure: established work finishes
        /// before more sockets consume unbounded memory or threads.
        pub fn submit(self: *Self, job: Job) !void {
            self.mutex.lockUncancelable(self.io);
            defer self.mutex.unlock(self.io);
            while (self.count == self.jobs.len and !self.stopping) {
                self.not_full.waitUncancelable(self.io, &self.mutex);
            }
            if (self.stopping) return error.PoolStopping;
            self.jobs[self.tail] = job;
            self.tail = (self.tail + 1) % self.jobs.len;
            self.count += 1;
            self.not_empty.signal(self.io);
        }

        /// Wait until all queued and active jobs have completed.
        pub fn waitIdle(self: *Self) void {
            self.mutex.lockUncancelable(self.io);
            defer self.mutex.unlock(self.io);
            while (self.count != 0 or self.active != 0) {
                self.idle.waitUncancelable(self.io, &self.mutex);
            }
        }

        /// Drain queued work, stop the persistent workers, and release the
        /// stable pool allocation.
        pub fn deinit(self: *Self) void {
            self.waitIdle();
            self.mutex.lockUncancelable(self.io);
            self.stopping = true;
            self.not_empty.broadcast(self.io);
            self.mutex.unlock(self.io);
            for (self.workers) |thread| thread.join();

            const allocator = self.allocator;
            allocator.free(self.workers);
            allocator.free(self.jobs);
            allocator.destroy(self);
        }

        fn workerMain(self: *Self, worker_index: usize) void {
            while (true) {
                self.mutex.lockUncancelable(self.io);
                while (self.count == 0 and !self.stopping) {
                    self.not_empty.waitUncancelable(self.io, &self.mutex);
                }
                if (self.count == 0 and self.stopping) {
                    self.mutex.unlock(self.io);
                    return;
                }

                const job = self.jobs[self.head];
                self.head = (self.head + 1) % self.jobs.len;
                self.count -= 1;
                self.active += 1;
                self.not_full.signal(self.io);
                self.mutex.unlock(self.io);

                self.handler(self.handler_ctx, job, worker_index);

                self.mutex.lockUncancelable(self.io);
                self.active -= 1;
                if (self.count == 0 and self.active == 0) self.idle.broadcast(self.io);
                self.mutex.unlock(self.io);
            }
        }
    };
}

test "bounded pool drains every job and reuses fixed workers" {
    const testing = std.testing;
    const Pool = BoundedPool(usize);
    const Ctx = struct {
        seen: []std.atomic.Value(u32),
        fn run(opaque_ctx: *anyopaque, job: usize, worker_index: usize) void {
            _ = worker_index;
            const self: *@This() = @ptrCast(@alignCast(opaque_ctx));
            _ = self.seen[job].fetchAdd(1, .acq_rel);
        }
    };

    var seen: [128]std.atomic.Value(u32) = undefined;
    for (&seen) |*value| value.* = .init(0);
    var ctx: Ctx = .{ .seen = &seen };
    const pool = try Pool.init(testing.allocator, testing.io, 4, 8, &ctx, Ctx.run);
    defer pool.deinit();
    for (0..seen.len) |job| try pool.submit(job);
    pool.waitIdle();
    for (seen) |value| try testing.expectEqual(@as(u32, 1), value.load(.acquire));
}
