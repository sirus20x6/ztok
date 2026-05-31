#!/usr/bin/env bash
#
# CMake + pkg-config consumer test for ztok.
#
# Installs ztok to a temporary ABSOLUTE prefix and a RELATIVE prefix, then
# for each prefix builds a tiny C program (tests/consumer/consumer.c) that
# #includes <ztok.h> and links libztok, once via CMake find_package(ztok)
# and once via `pkg-config --cflags --libs ztok`, runs it, and asserts it
# exits 0.
#
# The relative-prefix case is the regression guard for the bug where
# `zig build -p prefix` baked a *relative* prefix= into ztok.pc and
# relative IMPORTED_LOCATION paths into the CMake targets, so consumers
# invoking from another directory failed to find the headers/libs. We
# additionally inspect the generated ztok.pc / ztokTargets.cmake for the
# relative prefix and assert the paths inside are absolute.
#
# Skip-clean: if cmake or pkg-config (or zig) are missing, print a clear
# SKIP message and exit 0 so this is safe to wire into CI / `zig build`
# on hosts that lack the consumer toolchains.
#
# Usage:
#   tests/consumer/run.sh
#
# Honors:
#   ZTOK_REPO_ROOT  - repo root (default: two levels up from this script)
#   ZIG             - zig binary (default: zig)
#   CC              - C compiler for the pkg-config build (default: cc)

set -u

# ---- locate ourselves / the repo ----------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${ZTOK_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
ZIG="${ZIG:-zig}"
CC="${CC:-cc}"

note() { printf '[consumer-test] %s\n' "$*"; }
skip() { printf '[consumer-test] SKIP: %s\n' "$*"; exit 0; }
fail() { printf '[consumer-test] FAIL: %s\n' "$*" >&2; exit 1; }

# ---- skip-clean if the toolchains aren't present ------------------------
command -v "$ZIG"        >/dev/null 2>&1 || skip "zig not found (need it to install ztok)"
command -v cmake         >/dev/null 2>&1 || skip "cmake not found; skipping CMake + pkg-config consumer test"
command -v pkg-config    >/dev/null 2>&1 || skip "pkg-config not found; skipping CMake + pkg-config consumer test"
command -v "$CC"         >/dev/null 2>&1 || skip "C compiler '$CC' not found; skipping consumer test"

note "repo root: $REPO_ROOT"
note "zig:       $($ZIG version 2>/dev/null || echo '?')"
note "cmake:     $(cmake --version 2>/dev/null | head -n1)"
note "pkgconfig: $(pkg-config --version 2>/dev/null)"

# ---- scratch dir, cleaned on exit ---------------------------------------
WORK="$(mktemp -d "${TMPDIR:-/tmp}/ztok-consumer.XXXXXX")" || fail "mktemp failed"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

# =========================================================================
# Helpers
# =========================================================================

# install_ztok <prefix-as-passed-to-zig> <cwd-for-zig>
# Runs `zig build -p <prefix>` from <cwd>. <prefix> may be relative; in
# that case <cwd> is what it resolves against (mirrors how a user would
# invoke `zig build -p prefix` from a build directory).
install_ztok() {
    local prefix="$1" cwd="$2"
    note "installing ztok: prefix='$prefix' (cwd=$cwd)"
    ( cd "$cwd" && "$ZIG" build -p "$prefix" --build-file "$REPO_ROOT/build.zig" ) \
        || fail "zig build -p '$prefix' failed"
}

# build_with_cmake <abs-install-prefix> <tag>
# Configures + builds tests/consumer via find_package(ztok), runs the
# binary, asserts exit 0.
build_with_cmake() {
    local install_abs="$1" tag="$2"
    local bdir="$WORK/cmake-build-$tag"
    rm -rf "$bdir"; mkdir -p "$bdir"
    note "[$tag] cmake configure (CMAKE_PREFIX_PATH=$install_abs)"
    cmake -S "$SCRIPT_DIR" -B "$bdir" \
          -DCMAKE_PREFIX_PATH="$install_abs" \
          >"$bdir/configure.log" 2>&1 \
        || { cat "$bdir/configure.log" >&2; fail "[$tag] cmake configure failed"; }
    note "[$tag] cmake build"
    cmake --build "$bdir" >"$bdir/build.log" 2>&1 \
        || { cat "$bdir/build.log" >&2; fail "[$tag] cmake build failed"; }
    note "[$tag] run cmake-built consumer"
    "$bdir/consumer" || fail "[$tag] cmake-built consumer exited non-zero"
    note "[$tag] CMake consumer OK"
}

# build_with_pkgconfig <abs-install-prefix> <tag>
# Compiles consumer.c using `pkg-config --cflags --libs ztok` resolved via
# PKG_CONFIG_PATH, runs the binary, asserts exit 0.
build_with_pkgconfig() {
    local install_abs="$1" tag="$2"
    local pcdir="$install_abs/lib/pkgconfig"
    local bin="$WORK/pc-consumer-$tag"

    PKG_CONFIG_PATH="$pcdir" pkg-config --exists ztok \
        || fail "[$tag] pkg-config could not find ztok in $pcdir"

    local cflags libs libdir
    cflags="$(PKG_CONFIG_PATH="$pcdir" pkg-config --cflags ztok)" \
        || fail "[$tag] pkg-config --cflags failed"
    libs="$(PKG_CONFIG_PATH="$pcdir" pkg-config --libs ztok)" \
        || fail "[$tag] pkg-config --libs failed"
    libdir="$(PKG_CONFIG_PATH="$pcdir" pkg-config --variable=libdir ztok)" \
        || fail "[$tag] pkg-config --variable=libdir failed"

    note "[$tag] pkg-config --cflags: $cflags"
    note "[$tag] pkg-config --libs:   $libs"

    # Compile from a neutral cwd (not where any 'prefix/' dir lives) so a
    # relative path leaking out of pkg-config would actually break the
    # build — that's the regression we guard against.
    ( cd "$WORK" && $CC -std=c11 -O2 -Wall -Wextra \
        $cflags -o "$bin" "$SCRIPT_DIR/consumer.c" $libs \
        -Wl,-rpath,"$libdir" ) \
        || fail "[$tag] pkg-config compile failed"

    note "[$tag] run pkg-config-built consumer"
    "$bin" || fail "[$tag] pkg-config-built consumer exited non-zero"
    note "[$tag] pkg-config consumer OK"
}

# assert_paths_absolute <abs-install-prefix> <tag>
# The core regression guard: inspect the generated ztok.pc and
# ztokTargets.cmake and assert the embedded paths are ABSOLUTE, even when
# ztok was installed via a relative prefix.
assert_paths_absolute() {
    local install_abs="$1" tag="$2"
    local pc="$install_abs/lib/pkgconfig/ztok.pc"
    local targets="$install_abs/lib/cmake/ztok/ztokTargets.cmake"

    [ -f "$pc" ]      || fail "[$tag] generated ztok.pc missing at $pc"
    [ -f "$targets" ] || fail "[$tag] generated ztokTargets.cmake missing at $targets"

    # ztok.pc: prefix= line must start with '/'.
    local prefix_line
    prefix_line="$(grep -E '^prefix=' "$pc" | head -n1)"
    note "[$tag] ztok.pc: $prefix_line"
    case "$prefix_line" in
        prefix=/*) : ;; # absolute, good
        *) fail "[$tag] ztok.pc prefix is not absolute: '$prefix_line' (relative-prefix regression)";;
    esac

    # ztokTargets.cmake: every IMPORTED_LOCATION must be an absolute path.
    local loc
    while IFS= read -r loc; do
        # Extract the quoted path after IMPORTED_LOCATION.
        local p
        p="$(printf '%s' "$loc" | sed -nE 's/.*IMPORTED_LOCATION[[:space:]]+"([^"]+)".*/\1/p')"
        [ -n "$p" ] || continue
        note "[$tag] cmake IMPORTED_LOCATION: $p"
        case "$p" in
            /*) : ;; # absolute, good
            *) fail "[$tag] cmake IMPORTED_LOCATION not absolute: '$p' (relative-prefix regression)";;
        esac
        # And the file it points at must actually exist.
        [ -f "$p" ] || fail "[$tag] cmake IMPORTED_LOCATION points at a missing file: $p"
    done < <(grep -E 'IMPORTED_LOCATION' "$targets")

    note "[$tag] embedded paths are absolute and resolve — regression guard passed"
}

# =========================================================================
# 1. ABSOLUTE prefix
# =========================================================================
note "=== absolute-prefix case ==="
ABS_PREFIX="$WORK/install-abs"
install_ztok "$ABS_PREFIX" "$WORK"
assert_paths_absolute "$ABS_PREFIX" "abs"
build_with_cmake      "$ABS_PREFIX" "abs"
build_with_pkgconfig  "$ABS_PREFIX" "abs"

# =========================================================================
# 2. RELATIVE prefix  (the regression case)
# =========================================================================
# Install via a *relative* prefix from a dedicated build dir. The absolute
# location of the install tree is "$REL_BASE/$REL_NAME"; we hand that
# absolute path to the consumers, but the install itself was driven by a
# relative `-p`. The fixed build.zig must absolutize the prefix before
# baking it into ztok.pc / ztokTargets.cmake.
note "=== relative-prefix case (regression guard) ==="
REL_BASE="$WORK/relbuild"
REL_NAME="install-rel"
mkdir -p "$REL_BASE"
install_ztok "$REL_NAME" "$REL_BASE"
REL_ABS="$REL_BASE/$REL_NAME"
[ -d "$REL_ABS" ] || fail "relative install did not land at $REL_ABS"
assert_paths_absolute "$REL_ABS" "rel"
build_with_cmake      "$REL_ABS" "rel"
build_with_pkgconfig  "$REL_ABS" "rel"

note "ALL CONSUMER TESTS PASSED (cmake + pkg-config, absolute + relative prefix)"
exit 0
