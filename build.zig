const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const is_wasm = target.result.cpu.arch.isWasm();

    const root_mod = b.addModule("ztok", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        // c_api uses c_allocator (libc malloc) for stable cross-.so
        // semantics; the rest of the library is libc-free.
        .link_libc = true,
    });

    // Single source of truth for the project version: parse build.zig.zon
    // once here and expose to Zig code as @import("build_options").version.
    // Prevents drift between build.zig.zon, c_api.zig, README, and CHANGELOG
    // that bit us across the 1.18-1.21 multi-agent waves.
    const build_opts = b.addOptions();
    build_opts.addOption([]const u8, "version", projectVersion(b));

    // Optional TLS backend for `ztok serve`. The default is `none` so
    // `zig build` works without an mbedtls install. Set
    // `-Dtls=mbedtls` to enable server-side TLS termination; that
    // requires the system to expose libmbedtls / libmbedx509 /
    // libmbedcrypto via the linker (`pkg-config --libs mbedtls`).
    //
    // We expose the choice to Zig code as a build_options enum so
    // src/cli_serve.zig can compile out the TLS wire-up entirely in
    // the default build — no mbedtls headers are referenced unless
    // `-Dtls=mbedtls` is passed.
    const TlsBackend = enum { none, mbedtls };
    const tls_backend = b.option(TlsBackend, "tls", "Optional TLS backend for `ztok serve` (none|mbedtls)") orelse .none;
    build_opts.addOption(TlsBackend, "tls_backend", tls_backend);

    root_mod.addOptions("build_options", build_opts);

    // Link the TLS backend's system libraries onto the root module so
    // any executable that pulls in cli_serve.zig (the `ztok` CLI, the
    // test binary, etc.) gets the symbols at link time.
    if (tls_backend == .mbedtls) {
        root_mod.linkSystemLibrary("mbedtls", .{});
        root_mod.linkSystemLibrary("mbedx509", .{});
        root_mod.linkSystemLibrary("mbedcrypto", .{});
    }

    // Static library — always built.
    const lib_static = b.addLibrary(.{
        .name = "ztok",
        .root_module = root_mod,
        .linkage = .static,
    });
    b.installArtifact(lib_static);

    // Shared library — Linux/macOS only. WASM doesn't have a runtime
    // dynamic-linker model.
    if (!is_wasm) {
        const lib_shared = b.addLibrary(.{
            .name = "ztok",
            .root_module = root_mod,
            .linkage = .dynamic,
        });
        b.installArtifact(lib_shared);
    }

    // C header
    b.installFile("include/ztok.h", "include/ztok.h");

    // CMake + pkg-config integration files. Templated against the
    // install prefix so consumers can do `find_package(ztok)` or
    // `pkg-config --cflags --libs ztok` without manually wiring include
    // paths and link flags. Skipped on WASM where dynamic libs don't
    // exist in the usual sense.
    if (!is_wasm) {
        emitConsumerConfigs(b);
    }

    // CLI and bench rely on stdin/stdout + clock_gettime which work on
    // wasm32-wasi but not on wasm32-freestanding. Skip them on
    // non-WASI WASM targets.
    const build_cli_bench = !is_wasm or target.result.os.tag == .wasi;

    if (build_cli_bench) {
        const cli_mod = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        });
        cli_mod.addImport("ztok", root_mod);

        const cli = b.addExecutable(.{
            .name = "ztok",
            .root_module = cli_mod,
        });
        b.installArtifact(cli);

        const bench_mod = b.createModule(.{
            .root_source_file = b.path("bench/bench_ztok.zig"),
            .target = target,
            .optimize = .ReleaseFast,
        });
        bench_mod.addImport("ztok", root_mod);
        const bench = b.addExecutable(.{
            .name = "bench_ztok",
            .root_module = bench_mod,
        });
        b.installArtifact(bench);
        const run_bench = b.addRunArtifact(bench);
        if (b.args) |args| run_bench.addArgs(args);
        const bench_step = b.step("bench", "Run the ztok benchmark");
        bench_step.dependOn(&run_bench.step);

        // Marginal-value scoring microbench (v2 rebuild vs v3 mask).
        // Uses `optimize` (caller-selected) rather than hard-coding
        // ReleaseFast — both v2 and v3 paths get the same optimization
        // level so the ratio remains meaningful regardless.
        const bench_mv_mod = b.createModule(.{
            .root_source_file = b.path("bench/bench_marginal_value.zig"),
            .target = target,
            .optimize = optimize,
        });
        bench_mv_mod.addImport("ztok", root_mod);
        const bench_mv = b.addExecutable(.{
            .name = "bench_marginal_value",
            .root_module = bench_mv_mod,
        });
        b.installArtifact(bench_mv);
        const run_bench_mv = b.addRunArtifact(bench_mv);
        if (b.args) |args| run_bench_mv.addArgs(args);
        const bench_mv_step = b.step("bench-marginal-value", "v2 rebuild vs v3 mask marginal-value scoring");
        bench_mv_step.dependOn(&run_bench_mv.step);

        // Zig-side mirror of bench_c_api.c (single_small / single_large /
        // batch_pooled) for the C-vs-Zig ratio.
        const bench_zigpath_mod = b.createModule(.{
            .root_source_file = b.path("bench/bench_zig_path.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .link_libc = true,
        });
        bench_zigpath_mod.addImport("ztok", root_mod);
        const bench_zigpath = b.addExecutable(.{
            .name = "bench_zig_path",
            .root_module = bench_zigpath_mod,
        });
        b.installArtifact(bench_zigpath);
        const run_bench_zp = b.addRunArtifact(bench_zigpath);
        if (b.args) |args| run_bench_zp.addArgs(args);
        const bench_zp_step = b.step("bench-zig-path", "Run the Zig in-process mirror of bench_c_api");
        bench_zp_step.dependOn(&run_bench_zp.step);

        // C ABI bench harness. Compiled with the system C toolchain via
        // Zig's `addCSourceFile`, linked against the just-built static
        // libztok.a so we exercise the same code the .so does (libc
        // allocator, exported symbols only) without an rpath dance.
        const bench_c_mod = b.createModule(.{
            .target = target,
            .optimize = .ReleaseFast,
            .link_libc = true,
        });
        bench_c_mod.addIncludePath(b.path("include"));
        bench_c_mod.addCSourceFile(.{
            .file = b.path("bench/bench_c_api.c"),
            .flags = &.{ "-std=c11", "-O3", "-Wall", "-Wextra" },
        });
        const bench_c = b.addExecutable(.{
            .name = "bench_c_api",
            .root_module = bench_c_mod,
        });
        bench_c_mod.linkLibrary(lib_static);
        b.installArtifact(bench_c);
        const run_bench_c = b.addRunArtifact(bench_c);
        if (b.args) |args| run_bench_c.addArgs(args);
        const bench_c_step = b.step("bench-c", "Run the C ABI bench harness");
        bench_c_step.dependOn(&run_bench_c.step);

        // SIMD min-scan microbench (scalar vs narrow vs wide path).
        // Always ReleaseFast — point is to compare generated vector
        // code, which is meaningless at -O0.
        const bench_smin_mod = b.createModule(.{
            .root_source_file = b.path("bench/bench_simd_min.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .link_libc = true,
        });
        bench_smin_mod.addImport("ztok", root_mod);
        const bench_smin = b.addExecutable(.{
            .name = "bench_simd_min",
            .root_module = bench_smin_mod,
        });
        b.installArtifact(bench_smin);
        const run_bench_smin = b.addRunArtifact(bench_smin);
        if (b.args) |args| run_bench_smin.addArgs(args);
        const bench_smin_step = b.step("bench-simd-min", "Microbench scanMin scalar vs narrow vs wide");
        bench_smin_step.dependOn(&run_bench_smin.step);

        // Cross-tokenizer benchmark: ztok loading a TokenMonster .ztm
        // vocab or a SentencePiece .model, encoding the same corpus a
        // reference Python harness (bench/bench_competitors.py
        // --lib tokenmonster|sentencepiece) encodes through the
        // native tool. Run via `zig build bench-cross -- ...`.
        const bench_cross_mod = b.createModule(.{
            .root_source_file = b.path("bench/bench_cross.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .link_libc = true,
        });
        bench_cross_mod.addImport("ztok", root_mod);
        const bench_cross = b.addExecutable(.{
            .name = "bench_cross",
            .root_module = bench_cross_mod,
        });
        b.installArtifact(bench_cross);
        const run_bench_cross = b.addRunArtifact(bench_cross);
        if (b.args) |args| run_bench_cross.addArgs(args);
        const bench_cross_step = b.step("bench-cross", "Cross-tokenizer benchmark (ztok vs TokenMonster, ztok vs SentencePiece)");
        bench_cross_step.dependOn(&run_bench_cross.step);

        // Monster encoder profiling harness (post-1.18 agent E).
        const bench_mp_mod = b.createModule(.{
            .root_source_file = b.path("bench/bench_monster_profile.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .link_libc = true,
        });
        bench_mp_mod.addImport("ztok", root_mod);
        const bench_mp = b.addExecutable(.{
            .name = "bench_monster_profile",
            .root_module = bench_mp_mod,
        });
        b.installArtifact(bench_mp);
        const run_bench_mp = b.addRunArtifact(bench_mp);
        if (b.args) |args| run_bench_mp.addArgs(args);
        const bench_mp_step = b.step("bench-monster-profile", "Profile Monster.encodeChunk hot path");
        bench_mp_step.dependOn(&run_bench_mp.step);

        // Capcode normalizer profiling harness (post-1.20 agent C).
        const bench_cap_mod = b.createModule(.{
            .root_source_file = b.path("bench/bench_capcode_profile.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .link_libc = true,
        });
        bench_cap_mod.addImport("ztok", root_mod);
        const bench_cap = b.addExecutable(.{
            .name = "bench_capcode_profile",
            .root_module = bench_cap_mod,
        });
        b.installArtifact(bench_cap);
        const run_bench_cap = b.addRunArtifact(bench_cap);
        if (b.args) |args| run_bench_cap.addArgs(args);
        const bench_cap_step = b.step("bench-capcode-profile", "Profile capcode/nocapcode normalizer hot paths");
        bench_cap_step.dependOn(&run_bench_cap.step);

        // Capcode pipeline-split profile (post-1.20 agent C).
        const bench_capp_mod = b.createModule(.{
            .root_source_file = b.path("bench/bench_capcode_pipeline.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .link_libc = true,
        });
        bench_capp_mod.addImport("ztok", root_mod);
        const bench_capp = b.addExecutable(.{
            .name = "bench_capcode_pipeline",
            .root_module = bench_capp_mod,
        });
        b.installArtifact(bench_capp);
        const run_bench_capp = b.addRunArtifact(bench_capp);
        if (b.args) |args| run_bench_capp.addArgs(args);
        const bench_capp_step = b.step("bench-capcode-pipeline", "Profile capcode normalizer + encoder pipeline split");
        bench_capp_step.dependOn(&run_bench_capp.step);

        // Vocab-extend microbench: naive O(V·L²) LCS scan vs q-gram
        // pre-filter, on synthetic V=10K/50K/100K vocabs.
        const bench_vx_mod = b.createModule(.{
            .root_source_file = b.path("bench/bench_vocab_extend.zig"),
            .target = target,
            .optimize = .ReleaseSafe,
            .link_libc = true,
        });
        bench_vx_mod.addImport("ztok", root_mod);
        const bench_vx = b.addExecutable(.{
            .name = "bench_vocab_extend",
            .root_module = bench_vx_mod,
        });
        b.installArtifact(bench_vx);
        const run_bench_vx = b.addRunArtifact(bench_vx);
        if (b.args) |args| run_bench_vx.addArgs(args);
        const bench_vx_step = b.step("bench-vocab-extend", "naive vs q-gram weighted_similar microbench");
        bench_vx_step.dependOn(&run_bench_vx.step);

        // Convenience: `zig build run -- ...`
        const run_cli = b.addRunArtifact(cli);
        if (b.args) |args| run_cli.addArgs(args);
        const run_step = b.step("run", "Run the ztok CLI");
        run_step.dependOn(&run_cli.step);

        // ---- Fuzz harness (post-1.20 agent E) -----------------------
        //
        // Standalone binary that drives `fuzz/encode_decode.zig`. The
        // harness runs for `ZTOK_FUZZ_BUDGET_SECS` seconds (default 60)
        // and reports panics / round-trip mismatches by exiting with a
        // non-zero status; `zig build fuzz` then propagates that
        // failure to the caller / CI. Pinned to ReleaseSafe so
        // assertions fire — ReleaseFast would silently skip the
        // `unreachable`/`debug.assert` arms the harness is trying to
        // tickle.
        const fuzz_mod = b.createModule(.{
            .root_source_file = b.path("fuzz/encode_decode.zig"),
            .target = target,
            .optimize = .ReleaseSafe,
            .link_libc = true,
        });
        fuzz_mod.addImport("ztok", root_mod);
        const fuzz_exe = b.addExecutable(.{
            .name = "ztok_fuzz_encode_decode",
            .root_module = fuzz_mod,
        });
        b.installArtifact(fuzz_exe);
        const run_fuzz = b.addRunArtifact(fuzz_exe);
        run_fuzz.has_side_effects = true; // never cached
        if (b.args) |args| run_fuzz.addArgs(args);
        const fuzz_step = b.step("fuzz", "Run the encode/decode fuzz harness (ZTOK_FUZZ_BUDGET_SECS=60)");
        fuzz_step.dependOn(&run_fuzz.step);
    }

    // Tests — host only (tests use clock_gettime, stdin, threads, etc.).
    if (!is_wasm) {
        const tests = b.addTest(.{ .root_module = root_mod });
        const run_tests = b.addRunArtifact(tests);
        const test_step = b.step("test", "Run ztok tests");
        test_step.dependOn(&run_tests.step);

        // Smoke test for the install tree. Depends on the install step
        // so the artefacts it checks actually exist by the time the
        // test runs; reads `ZTOK_INSTALL_PREFIX` from the env (which we
        // forward from `b.install_path`) to know where to look. Skipped
        // (via `SkipZigTest`) when the env var is absent so the regular
        // `zig build test` still passes if a user clears `zig-out/`.
        const install_check_mod = b.createModule(.{
            .root_source_file = b.path("tests/install_check.zig"),
            .target = target,
            .optimize = optimize,
        });
        const install_check = b.addTest(.{ .root_module = install_check_mod });
        const run_install_check = b.addRunArtifact(install_check);
        run_install_check.step.dependOn(b.getInstallStep());
        run_install_check.setEnvironmentVariable("ZTOK_INSTALL_PREFIX", b.install_path);
        run_install_check.has_side_effects = true;
        const install_check_step = b.step("test-install", "Verify the install tree has all consumer-facing files");
        install_check_step.dependOn(&run_install_check.step);

        // Ops smoke tests (post-1.20 agent E): Dockerfile shape,
        // GitHub Actions YAML parses, fuzz harness binary runs. These
        // tests SkipZigTest when the corresponding artefact isn't
        // present so a fresh `zig build test` on a sparse checkout
        // doesn't fail; CI's `zig build` step always satisfies them.
        const ops_check_mod = b.createModule(.{
            .root_source_file = b.path("tests/ops_check.zig"),
            .target = target,
            .optimize = optimize,
        });
        const ops_check = b.addTest(.{ .root_module = ops_check_mod });
        const run_ops_check = b.addRunArtifact(ops_check);
        run_ops_check.step.dependOn(b.getInstallStep());
        run_ops_check.has_side_effects = true;
        // Cap the embedded fuzz smoke run at 2 seconds so `zig build
        // test` stays fast. The 60-second sweep runs in CI's
        // dedicated fuzz job (see .github/workflows/ci.yml).
        run_ops_check.setEnvironmentVariable("ZTOK_FUZZ_BUDGET_SECS", "2");
        run_ops_check.setEnvironmentVariable("ZTOK_FUZZ_SEED", "0xC0FFEE");
        const ops_check_step = b.step("test-ops", "Verify CI / Docker / fuzz harness artefacts");
        ops_check_step.dependOn(&run_ops_check.step);
        // Roll the ops checks into the default `test` step so a
        // regular `zig build test` includes them.
        test_step.dependOn(&run_ops_check.step);

        // Downstream-consumer test (CMake find_package + pkg-config).
        //
        // tests/consumer/run.sh installs ztok to a temp ABSOLUTE prefix
        // and a temp RELATIVE prefix, then for each prefix builds a tiny
        // C program (tests/consumer/consumer.c) against the installed
        // package once via `find_package(ztok)` and once via
        // `pkg-config --cflags --libs ztok`, runs it, and asserts the
        // baked-in paths are absolute even for the relative-prefix
        // install (the regression guard for the relative-prefix bug
        // fixed in 02f6d9f). The script self-skips (exit 0 with a clear
        // message) when cmake / pkg-config / a C compiler are absent, so
        // it's safe in the default `test` step on minimal hosts.
        //
        // We pass ZTOK_REPO_ROOT explicitly so the script installs *this*
        // checkout regardless of the runner's cwd, and forward `zig`'s
        // own path via ZIG so an out-of-PATH toolchain still works.
        const consumer_check = b.addSystemCommand(&.{"sh"});
        consumer_check.addFileArg(b.path("tests/consumer/run.sh"));
        consumer_check.setEnvironmentVariable("ZTOK_REPO_ROOT", b.build_root.path orelse ".");
        consumer_check.setEnvironmentVariable("ZIG", b.graph.zig_exe);
        consumer_check.has_side_effects = true; // installs + compiles; never cache
        const consumer_check_step = b.step("test-cmake", "CMake find_package + pkg-config consumer test (absolute & relative prefix)");
        consumer_check_step.dependOn(&consumer_check.step);
        // Intentionally NOT folded into the default `test` step: it shells
        // out to cmake/pkg-config and reinstalls ztok twice, which is
        // heavier than the in-process smoke tests. Run it explicitly via
        // `zig build test-cmake` (CI does this in a dedicated job).
    }

    // ---- Browser WASM target ---------------------------------------
    //
    // `zig build ztok-wasm-browser` cross-compiles a minimal
    // wasm32-freestanding shared object suitable for loading from a
    // JS Page. It exports a tiny subset of the C ABI on a bytes-in /
    // bytes-out surface (no filesystem) — see
    // `src/wasm_browser_root.zig` for the export list.
    //
    // This is independent of the host build: it forces target +
    // optimization mode so `zig build ztok-wasm-browser` Just Works
    // regardless of what `-Dtarget` the user set. It also doesn't
    // link libc — wasm32-freestanding has none.
    //
    // We use `addExecutable` with `entry = .disabled` because Zig's
    // `addLibrary` for wasm doesn't expose all the exports we need
    // (the freestanding library output omits the `export fn`
    // wrappers we declare). The "executable" path with no entry
    // point + `rdynamic = true` gives us a proper wasm with our
    // exports visible to JS.
    // Enable the WebAssembly SIMD128 feature so `@Vector(N, T)` lowers to
    // `v128.*` opcodes instead of scalar fallbacks. SIMD128 has been
    // baseline-supported by every major browser since ~2021 (Chrome 91,
    // Firefox 89, Safari 16.4); we feature-detect on the JS side via
    // `WebAssembly.validate` and refuse to load if it's missing. This is
    // the only build-time SIMD knob we touch — the host build's
    // `simd_min.zig` continues to use AVX-2/AVX-512 vectors.
    const wasm_browser_features = std.Target.wasm.featureSet(&[_]std.Target.wasm.Feature{.simd128});
    const wasm_browser_target = b.resolveTargetQuery(.{
        .cpu_arch = .wasm32,
        .os_tag = .freestanding,
        .abi = .none,
        .cpu_features_add = wasm_browser_features,
    });
    const wasm_browser_mod = b.createModule(.{
        .root_source_file = b.path("src/wasm_browser_root.zig"),
        .target = wasm_browser_target,
        .optimize = .ReleaseSmall,
        // wasm32-freestanding is single-threaded by default in 0.16;
        // be explicit so a future toolchain change doesn't bite.
        .single_threaded = true,
        // Strip symbols for smaller browser download.
        .strip = true,
    });
    const wasm_browser = b.addExecutable(.{
        .name = "ztok_browser",
        .root_module = wasm_browser_mod,
    });
    // No `_start` entry. The browser loader instantiates the module
    // and calls exported functions directly.
    wasm_browser.entry = .disabled;
    // Mark every `export` decl as visible to JS.
    wasm_browser.rdynamic = true;
    // Don't bring in any libc shims.
    wasm_browser.import_memory = false;
    // Default initial memory is 16 pages (1 MiB). cl100k_base.tiktoken
    // is ~1.7 MB on its own and BPE's hashmap doubles that; budget
    // 32 MiB max so we don't OOM under a typical browser corpus.
    wasm_browser.initial_memory = 4 * 1024 * 1024; // 4 MiB
    wasm_browser.max_memory = 256 * 1024 * 1024; // 256 MiB cap
    const install_wasm = b.addInstallArtifact(wasm_browser, .{});

    const wasm_step = b.step("ztok-wasm-browser", "Build browser-friendly wasm32-freestanding ztok module");
    wasm_step.dependOn(&install_wasm.step);

    // --- Scalar (no SIMD128) wasm build, for benchmarking control and
    // for fallback in browsers that lack SIMD128. Same source, same
    // module — only the CPU feature set differs. Emitted as
    // `ztok_browser_scalar.wasm` next to the SIMD build. Not part of
    // the default `ztok-wasm-browser` step; invoke explicitly via
    // `zig build ztok-wasm-browser-scalar`.
    const wasm_browser_scalar_target = b.resolveTargetQuery(.{
        .cpu_arch = .wasm32,
        .os_tag = .freestanding,
        .abi = .none,
    });
    const wasm_browser_scalar_mod = b.createModule(.{
        .root_source_file = b.path("src/wasm_browser_root.zig"),
        .target = wasm_browser_scalar_target,
        .optimize = .ReleaseSmall,
        .single_threaded = true,
        .strip = true,
    });
    const wasm_browser_scalar = b.addExecutable(.{
        .name = "ztok_browser_scalar",
        .root_module = wasm_browser_scalar_mod,
    });
    wasm_browser_scalar.entry = .disabled;
    wasm_browser_scalar.rdynamic = true;
    wasm_browser_scalar.import_memory = false;
    wasm_browser_scalar.initial_memory = 4 * 1024 * 1024;
    wasm_browser_scalar.max_memory = 256 * 1024 * 1024;
    const install_wasm_scalar = b.addInstallArtifact(wasm_browser_scalar, .{});
    const wasm_scalar_step = b.step("ztok-wasm-browser-scalar", "Build scalar (no SIMD128) wasm32-freestanding ztok module");
    wasm_scalar_step.dependOn(&install_wasm_scalar.step);
    // Note: NOT registered under the default `install` step — we don't
    // want a plain `zig build` (which is the host build) to also
    // cross-compile wasm for everyone every time.

    // Sanity test: after building the wasm, parse the binary to
    // confirm the exports we promise to JS are actually present.
    // Test runs on host (Linux) but reads the wasm-out artifact, so
    // it serves as a smoke-test that the build produced something
    // the loader will actually accept.
    if (!is_wasm) {
        const wasm_check_mod = b.createModule(.{
            .root_source_file = b.path("examples/wasm/check_exports.zig"),
            .target = target,
            .optimize = .Debug,
        });
        const wasm_check = b.addTest(.{ .root_module = wasm_check_mod });
        const run_wasm_check = b.addRunArtifact(wasm_check);
        run_wasm_check.step.dependOn(&wasm_browser.step);
        // Also depend on the scalar wasm build so the scalar control
        // test (asserting the no-SIMD binary contains far fewer 0xFD
        // bytes than the SIMD build) has both artefacts available.
        run_wasm_check.step.dependOn(&wasm_browser_scalar.step);
        run_wasm_check.has_side_effects = true;
        const wasm_check_step = b.step("test-wasm-browser", "Verify the browser wasm exports + SIMD128 opcodes are present");
        wasm_check_step.dependOn(&run_wasm_check.step);
    }
}

/// Read the version string from `build.zig.zon` so the cmake/pkg-config
/// templates emit the canonical project version. Falls back to "0.0.0"
/// (rather than panicking) if the zon file gets reshuffled — the install
/// tree is still usable in that case, the version number just looks off.
fn projectVersion(b: *std.Build) []const u8 {
    // Zig 0.16: `std.fs.cwd()` was retired in favor of `std.Io.Dir.cwd()`
    // (the rest of the codebase already uses the new spelling — see
    // src/bpe.zig:367 etc.). Wrap the IO handle the same way here.
    // We read relative to `b.build_root.handle` rather than process cwd
    // because the user may have invoked `zig build --build-file <path>`
    // from a different directory.
    const io = std.Io.Threaded.global_single_threaded.io();
    const zon = b.build_root.handle.readFileAlloc(io, "build.zig.zon", b.allocator, .unlimited) catch return "0.0.0";
    // Tiny hand-roll. `build.zig.zon` is a fixed structure, not arbitrary
    // user input — the std.zon parser is overkill and brings in extra
    // build-time deps.
    const needle = ".version = \"";
    const start = std.mem.indexOf(u8, zon, needle) orelse return "0.0.0";
    const after = start + needle.len;
    const end_rel = std.mem.indexOfScalarPos(u8, zon, after, '"') orelse return "0.0.0";
    return b.dupe(zon[after..end_rel]);
}

/// Generate `ztokConfig.cmake`, `ztokTargets.cmake`,
/// `ztokConfigVersion.cmake`, and `ztok.pc` from the in-tree templates,
/// substituting the install prefix at build time, and install them under
/// `<prefix>/lib/cmake/ztok/` and `<prefix>/lib/pkgconfig/`.
///
/// We pre-compute the absolute install prefix here so the emitted files
/// contain plain absolute paths — CMake's `find_package(ztok)` then
/// works either by `CMAKE_PREFIX_PATH=<prefix>` or by ztok being on
/// the system search path. The same approach gives pkg-config the
/// `prefix=` line it needs.
fn emitConsumerConfigs(b: *std.Build) void {
    // `b.install_path` echoes `-p`/`--prefix` verbatim, so a relative
    // prefix (`zig build -p prefix`) would bake `prefix=prefix` into
    // ztok.pc — yielding relative `-Lprefix/lib -Iprefix/include` that
    // only resolve when the compiler runs from the one directory holding
    // `prefix/`. pkg-config consumers (cgo via PKG_CONFIG_PATH, CMake)
    // invoke from elsewhere and break. Absolutize against the build
    // runner's *process cwd* — that is what zig itself resolves a
    // relative `-p` against when it places the install tree, so the
    // baked-in paths land on the same files zig actually wrote. (An
    // earlier fix resolved against `b.build_root` instead, which is only
    // the same directory when you run `zig build` from the repo root;
    // `zig build -p out --build-file <repo>/build.zig` from a separate
    // build dir baked in <repo>/out while the files landed in <cwd>/out.
    // tests/consumer/run.sh guards both spellings.)
    const prefix = if (std.fs.path.isAbsolute(b.install_path))
        b.install_path
    else blk: {
        const io = std.Io.Threaded.global_single_threaded.io();
        const cwd = std.process.currentPathAlloc(io, b.allocator) catch @panic("emitConsumerConfigs: getcwd failed");
        break :blk std.fs.path.join(b.allocator, &.{ cwd, b.install_path }) catch @panic("emitConsumerConfigs: resolve prefix failed");
    };
    const lib_dir = b.fmt("{s}/lib", .{prefix});
    const include_dir = b.fmt("{s}/include", .{prefix});
    const version = projectVersion(b);

    // Read raw templates and do trivial @TOKEN@ substitution. CMake's
    // configure_package_config_file is the "right" tool but we don't
    // want to depend on a host cmake at build time — the templates are
    // short, the substitution is mechanical.
    const cfg_tmpl = readTemplate(b, "cmake/ztokConfig.cmake.in");
    const targets_tmpl = readTemplate(b, "cmake/ztokTargets.cmake.in");
    const cfgver_tmpl = readTemplate(b, "cmake/ztokConfigVersion.cmake.in");
    const pc_tmpl = readTemplate(b, "pkgconfig/ztok.pc.in");

    // For `@PACKAGE_INIT@` we expand to the CMake-standard set of
    // helper macros that `configure_package_config_file` would normally
    // emit. They guard against double-inclusion and provide the
    // `set_and_check` / `check_required_components` builtins.
    const package_init =
        \\# Generated by ztok's build.zig (hand-rolled stand-in for
        \\# CMake's configure_package_config_file expansion).
        \\
        \\get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)
        \\
        \\macro(set_and_check _var _file)
        \\    set(${_var} "${_file}")
        \\    if(NOT EXISTS "${_file}")
        \\        message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
        \\    endif()
        \\endmacro()
        \\
        \\macro(check_required_components _NAME)
        \\    foreach(comp ${${_NAME}_FIND_COMPONENTS})
        \\        if(NOT ${_NAME}_${comp}_FOUND)
        \\            if(${_NAME}_FIND_REQUIRED_${comp})
        \\                set(${_NAME}_FOUND FALSE)
        \\            endif()
        \\        endif()
        \\    endforeach()
        \\endmacro()
    ;

    const cfg = expandTemplate(b, cfg_tmpl, &.{
        .{ .from = "@PACKAGE_INIT@", .to = package_init },
        .{ .from = "@PACKAGE_VERSION@", .to = version },
    });
    const targets = expandTemplate(b, targets_tmpl, &.{
        .{ .from = "@PACKAGE_LIB_DIR@", .to = lib_dir },
        .{ .from = "@PACKAGE_INCLUDE_DIR@", .to = include_dir },
    });
    const cfgver = expandTemplate(b, cfgver_tmpl, &.{
        .{ .from = "@PACKAGE_VERSION@", .to = version },
    });
    const pc = expandTemplate(b, pc_tmpl, &.{
        .{ .from = "@PREFIX@", .to = prefix },
        .{ .from = "@PACKAGE_VERSION@", .to = version },
    });

    const wf = b.addWriteFiles();
    const cfg_lp = wf.add("ztokConfig.cmake", cfg);
    const targets_lp = wf.add("ztokTargets.cmake", targets);
    const cfgver_lp = wf.add("ztokConfigVersion.cmake", cfgver);
    const pc_lp = wf.add("ztok.pc", pc);

    b.getInstallStep().dependOn(&b.addInstallFile(cfg_lp, "lib/cmake/ztok/ztokConfig.cmake").step);
    b.getInstallStep().dependOn(&b.addInstallFile(targets_lp, "lib/cmake/ztok/ztokTargets.cmake").step);
    b.getInstallStep().dependOn(&b.addInstallFile(cfgver_lp, "lib/cmake/ztok/ztokConfigVersion.cmake").step);
    b.getInstallStep().dependOn(&b.addInstallFile(pc_lp, "lib/pkgconfig/ztok.pc").step);
}

fn readTemplate(b: *std.Build, sub_path: []const u8) []const u8 {
    // Same 0.16 stdlib note as projectVersion above. Read relative to the
    // build root so out-of-tree `zig build --build-file` invocations work.
    const io = std.Io.Threaded.global_single_threaded.io();
    return b.build_root.handle.readFileAlloc(io, sub_path, b.allocator, .unlimited) catch |e| {
        std.debug.panic("ztok build.zig: failed to read template {s}: {s}", .{ sub_path, @errorName(e) });
    };
}

const Substitution = struct { from: []const u8, to: []const u8 };

/// Trivial @TOKEN@ → value substitution. Keeps the templates readable
/// and avoids dragging in a real templating engine at build time.
fn expandTemplate(b: *std.Build, tmpl: []const u8, subs: []const Substitution) []const u8 {
    var current: []const u8 = tmpl;
    for (subs) |sub| {
        current = std.mem.replaceOwned(u8, b.allocator, current, sub.from, sub.to) catch |e| {
            std.debug.panic("ztok build.zig: template substitution failed: {s}", .{@errorName(e)});
        };
    }
    return current;
}
