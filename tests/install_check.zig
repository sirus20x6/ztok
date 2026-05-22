//! Verifies the install tree contains every file consumers depend on.
//!
//! Reads `ZTOK_INSTALL_PREFIX` from the environment (the build step
//! sets it to the prefix passed via `zig build -p PREFIX`) and checks
//! each expected install artefact exists. Intentionally a pure smoke
//! test — content-level validation lives in the example builds.
//!
//! Used by `zig build test-install` after the install step completes.

const std = @import("std");
const testing = std.testing;

/// Look up `ZTOK_INSTALL_PREFIX` via the test runner's `Environ`. Falls
/// through to `error.SkipZigTest` so this file can also be folded into
/// `zig build test` without forcing the install step.
fn installPrefix() ![]u8 {
    return testing.environ.getAlloc(testing.allocator, "ZTOK_INSTALL_PREFIX") catch |e| switch (e) {
        error.EnvironmentVariableMissing => return error.SkipZigTest,
        else => return e,
    };
}

test "install tree contains expected artefacts" {
    const prefix = try installPrefix();
    defer testing.allocator.free(prefix);

    const io = std.Io.Threaded.global_single_threaded.io();
    const cwd = std.Io.Dir.cwd();

    const expected = [_][]const u8{
        "lib/libztok.a",
        "lib/libztok.so",
        "include/ztok.h",
        "lib/cmake/ztok/ztokConfig.cmake",
        "lib/cmake/ztok/ztokTargets.cmake",
        "lib/cmake/ztok/ztokConfigVersion.cmake",
        "lib/pkgconfig/ztok.pc",
    };

    for (expected) |rel| {
        const full = try std.fmt.allocPrint(testing.allocator, "{s}/{s}", .{ prefix, rel });
        defer testing.allocator.free(full);

        cwd.access(io, full, .{}) catch |e| {
            std.debug.print("missing install artefact: {s} ({s})\n", .{ full, @errorName(e) });
            return error.MissingInstallArtefact;
        };
    }
}

test "ztok.pc declares the right prefix" {
    const prefix = try installPrefix();
    defer testing.allocator.free(prefix);

    const io = std.Io.Threaded.global_single_threaded.io();
    const cwd = std.Io.Dir.cwd();

    const pc_path = try std.fmt.allocPrint(testing.allocator, "{s}/lib/pkgconfig/ztok.pc", .{prefix});
    defer testing.allocator.free(pc_path);

    const contents = try cwd.readFileAlloc(io, pc_path, testing.allocator, .unlimited);
    defer testing.allocator.free(contents);

    const expected_prefix_line = try std.fmt.allocPrint(testing.allocator, "prefix={s}\n", .{prefix});
    defer testing.allocator.free(expected_prefix_line);

    if (std.mem.indexOf(u8, contents, expected_prefix_line) == null) {
        std.debug.print(
            "ztok.pc prefix mismatch.\n  want: {s}  contents:\n{s}\n",
            .{ expected_prefix_line, contents },
        );
        return error.PcPrefixMismatch;
    }
}

test "ztokTargets.cmake references the libraries we install" {
    const prefix = try installPrefix();
    defer testing.allocator.free(prefix);

    const io = std.Io.Threaded.global_single_threaded.io();
    const cwd = std.Io.Dir.cwd();

    const cmake_path = try std.fmt.allocPrint(
        testing.allocator,
        "{s}/lib/cmake/ztok/ztokTargets.cmake",
        .{prefix},
    );
    defer testing.allocator.free(cmake_path);

    const contents = try cwd.readFileAlloc(io, cmake_path, testing.allocator, .unlimited);
    defer testing.allocator.free(contents);

    const must_contain = [_][]const u8{
        "ztok::ztok",
        "ztok::ztok_static",
        "libztok.so",
        "libztok.a",
    };
    for (must_contain) |needle| {
        if (std.mem.indexOf(u8, contents, needle) == null) {
            std.debug.print("ztokTargets.cmake missing: {s}\n", .{needle});
            return error.CmakeTargetsMissing;
        }
    }
}
