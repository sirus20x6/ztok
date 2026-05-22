// Build script for the ztok Rust binding.
//
// Resolution order for the `libztok` link path + include dir:
//
//   1. `ZTOK_LIB_DIR` and/or `ZTOK_INCLUDE_DIR` env vars (explicit
//      override — useful in CI when pkg-config isn't on the box).
//   2. `pkg-config --libs --cflags ztok` (default; available when the
//      crate was built with `--features pkg-config`, which is the
//      default). We shell out to the system `pkg-config` rather than
//      pulling in the `pkg-config` crate so the binding has *zero* build
//      dependencies — keeps `cargo build` fast and avoids a transitive
//      MSRV creep.
//   3. Bare `-lztok` and trust the system loader. This is what you'll
//      hit on minimal Docker images that ship neither pkg-config nor a
//      `ZTOK_LIB_DIR` override.
//
// Link mode:
//   - default: shared (`cargo:rustc-link-lib=ztok`).
//   - `--features link-static`: static (`cargo:rustc-link-lib=static=ztok`)
//     plus libc++/libm/libpthread for the Zig stdlib's typical deps.

use std::env;
use std::process::Command;

fn main() {
    // Re-run if the user's environment changes.
    println!("cargo:rerun-if-env-changed=ZTOK_LIB_DIR");
    println!("cargo:rerun-if-env-changed=ZTOK_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
    println!("cargo:rerun-if-changed=build.rs");

    let want_static = env::var_os("CARGO_FEATURE_LINK_STATIC").is_some();
    let want_pkgconfig = env::var_os("CARGO_FEATURE_PKG_CONFIG").is_some();

    let kind = if want_static { "static" } else { "dylib" };

    // 1. Explicit env-var override.
    let mut found_via_env = false;
    if let Some(lib_dir) = env::var_os("ZTOK_LIB_DIR") {
        println!("cargo:rustc-link-search=native={}", lib_dir.to_string_lossy());
        found_via_env = true;
    }
    if let Some(include_dir) = env::var_os("ZTOK_INCLUDE_DIR") {
        // We don't compile C ourselves, but downstream consumers (e.g.
        // bindgen runs against this crate) can find the header here.
        println!("cargo:include={}", include_dir.to_string_lossy());
    }

    // 2. pkg-config fallback (if the feature is on and env didn't already
    // pin the path). Shelled out — see header comment for why.
    if !found_via_env && want_pkgconfig {
        if let Some((lib_dirs, include_dir)) = query_pkg_config() {
            for d in lib_dirs {
                println!("cargo:rustc-link-search=native={}", d);
            }
            if let Some(inc) = include_dir {
                println!("cargo:include={}", inc);
            }
        }
        // If pkg-config fails we fall through to the bare -l directive —
        // the system loader may still find libztok in /usr/lib.
    }

    // 3. The link directive itself.
    println!("cargo:rustc-link-lib={}=ztok", kind);

    // Static link drags in the Zig stdlib's C deps. Best-effort: only
    // emit on Unix-like targets where these names actually resolve.
    if want_static {
        let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
        match target_os.as_str() {
            "linux" => {
                println!("cargo:rustc-link-lib=dylib=m");
                println!("cargo:rustc-link-lib=dylib=pthread");
                println!("cargo:rustc-link-lib=dylib=dl");
            }
            "macos" => {
                println!("cargo:rustc-link-lib=dylib=System");
            }
            _ => {}
        }
    }
}

/// Run `pkg-config --libs-only-L --cflags-only-I ztok` and parse the
/// `-L` / `-I` entries out of stdout. Returns `None` if pkg-config is
/// missing or the `ztok.pc` file can't be located.
fn query_pkg_config() -> Option<(Vec<String>, Option<String>)> {
    // Try `--libs-only-L` first so we get just the search dirs, no
    // `-lztok` (we emit that ourselves so we control static vs dylib).
    let libs_out = Command::new("pkg-config")
        .args(["--libs-only-L", "ztok"])
        .output()
        .ok()?;
    if !libs_out.status.success() {
        return None;
    }
    let libs_str = String::from_utf8(libs_out.stdout).ok()?;
    let lib_dirs: Vec<String> = libs_str
        .split_whitespace()
        .filter_map(|tok| tok.strip_prefix("-L").map(str::to_owned))
        .collect();

    let cflags_out = Command::new("pkg-config")
        .args(["--cflags-only-I", "ztok"])
        .output()
        .ok()?;
    let include_dir = if cflags_out.status.success() {
        String::from_utf8(cflags_out.stdout)
            .ok()
            .and_then(|s| {
                s.split_whitespace()
                    .find_map(|tok| tok.strip_prefix("-I").map(str::to_owned))
            })
    } else {
        None
    };

    Some((lib_dirs, include_dir))
}
