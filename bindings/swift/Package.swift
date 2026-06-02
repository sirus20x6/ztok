// swift-tools-version: 5.9
//
// ztok — Swift bindings for libztok.
//
// Idiomatic Swift wrapper over the C ABI declared in `include/ztok.h`,
// mirroring the surface of the Python / Node / Ruby / Go / Rust / .NET /
// Java bindings. Targets libztok 1.28.
//
// Layout:
//   Sources/CZtok   — C interop shim. The module.modulemap exposes
//                     <ztok.h> to Swift as `import CZtok`, and links
//                     libztok (`-lztok`) at build time.
//   Sources/Ztok    — high-level Swift API (Pipeline, BatchPool,
//                     StreamEncoder, Fingerprint, ZtokError).
//   Tests/ZtokTests — XCTest smoke + 1000-iter PRNG fuzz suite. Tests
//                     skip cleanly via XCTSkipUnless when libztok is
//                     not loadable.
//
// Linking libztok:
//   After `zig build -p prefix` you'll have prefix/{include,lib}.
//   - Compile-time:  pkg-config drops `-I prefix/include -L prefix/lib`
//                    if `prefix/lib/pkgconfig` is on PKG_CONFIG_PATH.
//                    Otherwise pass them via `-Xcc -I... -Xlinker -L...`
//                    on the `swift build` command line.
//   - Run-time:      LD_LIBRARY_PATH (Linux) or DYLD_LIBRARY_PATH
//                    (macOS) must include the directory containing
//                    `libztok.{so,dylib}`. See README for details.

import PackageDescription

let package = Package(
    name: "Ztok",
    platforms: [
        // libztok runs anywhere Zig 0.16 can target; SPM needs explicit
        // floors for the Apple platforms. Linux is unconstrained.
        .macOS(.v11),
    ],
    products: [
        .library(name: "Ztok", targets: ["Ztok"]),
    ],
    targets: [
        // Low-level C interop shim. `systemLibrary` reads the
        // module.modulemap and (optionally) consults pkg-config.
        // .linkedLibrary("ztok") emits `-lztok` so the linker resolves
        // against libztok.{so,dylib} on the standard library path.
        .systemLibrary(
            name: "CZtok",
            path: "Sources/CZtok",
            pkgConfig: "ztok",
            providers: [
                // No system package manager ships libztok yet — these
                // hints are aspirational and never trigger on stock
                // brew/apt. Build libztok yourself via `zig build`.
                .brew(["ztok"]),
                .apt(["libztok-dev"]),
            ]
        ),
        // High-level Swift wrapper.
        .target(
            name: "Ztok",
            dependencies: ["CZtok"],
            path: "Sources/Ztok"
        ),
        // XCTest smoke + fuzz suite.
        .testTarget(
            name: "ZtokTests",
            dependencies: ["Ztok"],
            path: "Tests/ZtokTests"
        ),
    ]
)
