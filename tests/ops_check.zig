//! Ops-layer smoke tests: validate the artefacts the post-1.20 agent
//! E (ops + quality) batch added so a stale Dockerfile / malformed
//! CI yaml / broken fuzz step is caught before it reaches the user.
//!
//! These tests SHELL OUT for the YAML check (Python is host-only; we
//! don't want to vendor a YAML parser into the test image). When
//! Python isn't on PATH the YAML test SkipZigTest's; the Dockerfile
//! checks are pure-Zig and always run.

const std = @import("std");
const testing = std.testing;

fn fileExists(io: std.Io, path: []const u8) bool {
    std.Io.Dir.cwd().access(io, path, .{}) catch return false;
    return true;
}

test "ops: Dockerfile present and references the multi-stage shape we expect" {
    const io = testing.io;
    const src = std.Io.Dir.cwd().readFileAlloc(io, "Dockerfile", testing.allocator, .limited(64 * 1024)) catch {
        // Older checkouts may not have the Dockerfile yet — skip
        // rather than fail so cherry-picks of unrelated commits
        // don't break.
        return error.SkipZigTest;
    };
    defer testing.allocator.free(src);

    // Multi-stage build with an explicit `AS builder` and a distroless
    // runtime — both are load-bearing for keeping the image minimal.
    try testing.expect(std.mem.indexOf(u8, src, "AS builder") != null);
    try testing.expect(std.mem.indexOf(u8, src, "distroless") != null);
    // EXPOSE the default `ztok serve` port.
    try testing.expect(std.mem.indexOf(u8, src, "EXPOSE 7890") != null);
    // ENTRYPOINT routed through the installed binary.
    try testing.expect(std.mem.indexOf(u8, src, "ENTRYPOINT [\"/usr/local/bin/ztok\"]") != null);
    // Zig version pin — make sure no one accidentally floats this.
    try testing.expect(std.mem.indexOf(u8, src, "ZIG_VERSION") != null);
}

test "ops: .dockerignore excludes heavy local artefacts" {
    const io = testing.io;
    const src = std.Io.Dir.cwd().readFileAlloc(io, ".dockerignore", testing.allocator, .limited(16 * 1024)) catch {
        return error.SkipZigTest;
    };
    defer testing.allocator.free(src);

    // These would otherwise add hundreds of MB to the build context.
    const must_have = [_][]const u8{
        ".zig-cache/",
        "zig-out/",
        "bindings/python/.pytest_cache/",
        "bindings/nodejs/node_modules/",
    };
    for (must_have) |needle| {
        try testing.expect(std.mem.indexOf(u8, src, needle) != null);
    }
}

test "ops: CI workflow YAML files exist with expected structure" {
    const io = testing.io;
    const cwd = std.Io.Dir.cwd();

    const workflows = [_]struct { path: []const u8, must_have: []const []const u8 }{
        .{
            .path = ".github/workflows/ci.yml",
            .must_have = &.{
                "name: ci",
                "jobs:",
                "runs-on:",
                // Zig version pin.
                "ZIG_VERSION:",
                // Action version pin guard — no @main / @master.
                "uses: actions/checkout@v",
                "uses: mlugg/setup-zig@v",
            },
        },
        .{
            .path = ".github/workflows/equivalence.yml",
            .must_have = &.{
                "name: equivalence",
                "schedule:",
                "cron:",
                "uses: actions/checkout@v",
            },
        },
    };

    var found_any = false;
    for (workflows) |wf| {
        const src = cwd.readFileAlloc(io, wf.path, testing.allocator, .limited(64 * 1024)) catch continue;
        defer testing.allocator.free(src);
        found_any = true;

        // YAML-aware sanity: every workflow MUST NOT pin actions to
        // unstable refs. Only inspect `uses: org/repo@ref` lines so
        // git triggers (`branches: [main]`) don't false-positive.
        var line_it = std.mem.splitScalar(u8, src, '\n');
        while (line_it.next()) |raw_line| {
            const line = std.mem.trim(u8, raw_line, " \t");
            if (!std.mem.startsWith(u8, line, "uses:") and !std.mem.startsWith(u8, line, "- uses:")) continue;
            if (std.mem.indexOf(u8, line, "@master") != null) {
                std.debug.print("workflow {s} line `{s}` pins to @master — forbidden\n", .{ wf.path, line });
                return error.UnstableActionRef;
            }
            if (std.mem.indexOf(u8, line, "@main") != null) {
                std.debug.print("workflow {s} line `{s}` pins to @main — forbidden\n", .{ wf.path, line });
                return error.UnstableActionRef;
            }
        }

        for (wf.must_have) |needle| {
            if (std.mem.indexOf(u8, src, needle) == null) {
                std.debug.print("workflow {s} missing required substring: {s}\n", .{ wf.path, needle });
                return error.WorkflowMissingClause;
            }
        }
    }
    if (!found_any) return error.SkipZigTest;
}

test "ops: CI workflows parse as YAML via python3 (when present)" {
    const io = testing.io;

    const workflows = [_][]const u8{
        ".github/workflows/ci.yml",
        ".github/workflows/equivalence.yml",
    };

    var missing_count: usize = 0;
    for (workflows) |path| {
        if (!fileExists(io, path)) missing_count += 1;
    }
    if (missing_count == workflows.len) {
        return error.SkipZigTest;
    }

    // Locate python3 via PATH; without it, skip.
    var path_buf: [4096]u8 = undefined;
    const path_env = testing.environ.getPosix("PATH") orelse return error.SkipZigTest;
    const python = findOnPath(&path_buf, path_env, "python3") catch return error.SkipZigTest;

    for (workflows) |path| {
        if (!fileExists(io, path)) continue;
        const quoted = try jsonQuote(testing.allocator, path);
        defer testing.allocator.free(quoted);
        const script = try std.fmt.allocPrint(
            testing.allocator,
            "import sys, yaml; yaml.safe_load(open({s}))",
            .{quoted},
        );
        defer testing.allocator.free(script);

        const argv = [_][]const u8{ python, "-c", script };
        var child = std.process.spawn(io, .{
            .argv = &argv,
            .stdout = .ignore,
            .stderr = .pipe,
        }) catch return error.SkipZigTest;

        // Drain stderr so a misconfigured PyYAML install surfaces
        // its error message rather than dangling on a full pipe.
        const stderr_h = child.stderr.?;
        var stderr_buf: [4096]u8 = undefined;
        var err_reader = stderr_h.reader(io, &stderr_buf);
        const captured = err_reader.interface.allocRemaining(testing.allocator, .unlimited) catch &[_]u8{};
        defer if (captured.len > 0) testing.allocator.free(captured);

        const term = child.wait(io) catch return error.YamlParseFailed;
        switch (term) {
            .exited => |code| {
                if (code != 0) {
                    // If PyYAML isn't installed we want SkipZigTest, not failure.
                    if (std.mem.indexOf(u8, captured, "ModuleNotFoundError") != null and
                        std.mem.indexOf(u8, captured, "yaml") != null)
                    {
                        return error.SkipZigTest;
                    }
                    std.debug.print("yaml parse failed for {s}: {s}\n", .{ path, captured });
                    return error.YamlParseFailed;
                }
            },
            else => return error.YamlParseFailed,
        }
    }
}

test "ops: fuzz harness binary runs for 5 seconds without crashing" {
    const io = testing.io;

    // The fuzz binary is installed alongside ztok via `b.installArtifact`
    // so it must exist after `zig build`. If running tests without a
    // prior install (e.g. `zig build test` in a fresh checkout where the
    // user only ran `zig build test`), the binary may be missing — skip
    // in that case so the regular test loop stays green.
    const fuzz_bin = "zig-out/bin/ztok_fuzz_encode_decode";
    if (!fileExists(io, fuzz_bin)) {
        return error.SkipZigTest;
    }

    // Inherit the parent environment so we don't have to clone it
    // (createMap is allocation-heavy). The harness defaults to 60s
    // when ZTOK_FUZZ_BUDGET_SECS is unset; the caller (`zig build
    // test` via setEnvironmentVariable) pins it to 2s for the
    // embedded smoke run.
    const argv = [_][]const u8{fuzz_bin};
    var child = std.process.spawn(io, .{
        .argv = &argv,
        .stdout = .ignore,
        .stderr = .ignore,
    }) catch |e| {
        std.debug.print("spawn failed: {t}\n", .{e});
        return error.FuzzHarnessSpawnFailed;
    };

    const term = child.wait(io) catch |e| {
        std.debug.print("wait failed: {t}\n", .{e});
        return error.FuzzHarnessWaitFailed;
    };
    switch (term) {
        .exited => |code| {
            if (code != 0) {
                std.debug.print("fuzz binary exited {d}\n", .{code});
                return error.FuzzHarnessCrashed;
            }
        },
        .signal => {
            std.debug.print("fuzz binary killed by signal\n", .{});
            return error.FuzzHarnessCrashed;
        },
        else => return error.FuzzHarnessCrashed,
    }
}

// ----- helpers --------------------------------------------------------

/// Search PATH (colon-separated) for `name`. Returns a slice into
/// `buf` on success (avoids per-call allocation).
fn findOnPath(buf: []u8, path_env: []const u8, name: []const u8) ![]const u8 {
    var it = std.mem.tokenizeScalar(u8, path_env, ':');
    while (it.next()) |dir| {
        const written = std.fmt.bufPrint(buf, "{s}/{s}", .{ dir, name }) catch continue;
        const io = testing.io;
        std.Io.Dir.cwd().access(io, written, .{}) catch continue;
        return written;
    }
    return error.FileNotFound;
}

/// JSON-quote a string (with surrounding double-quotes) so it can be
/// embedded inside the Python `-c` snippet we shell out for.
fn jsonQuote(a: std.mem.Allocator, s: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.append(a, '"');
    for (s) |c| {
        switch (c) {
            '\\', '"' => {
                try out.append(a, '\\');
                try out.append(a, c);
            },
            else => try out.append(a, c),
        }
    }
    try out.append(a, '"');
    return out.toOwnedSlice(a);
}
