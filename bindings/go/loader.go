package ztok

// Go's cgo statically links the libztok name into the binary at build
// time via `#cgo LDFLAGS: -lztok` (or `#cgo pkg-config: ztok` when
// ztok.pc is on PKG_CONFIG_PATH). At runtime the dynamic loader
// resolves the library through the system's normal search algorithm:
//
//   - $LD_LIBRARY_PATH (Linux) / $DYLD_LIBRARY_PATH (macOS).
//   - The rpath baked into the binary at link time.
//   - System directories (/lib, /usr/lib, /usr/local/lib).
//
// We don't paper over that with a dlopen fallback — cgo's whole value
// proposition is that the linker, not the binding, manages library
// resolution. If you need a non-system install, set LD_LIBRARY_PATH
// (Linux) or DYLD_LIBRARY_PATH (macOS) or pass `-extldflags '-Wl,-rpath,/abs/path'`
// to `go build`.
//
// Build with `-tags nopkgconfig` to skip pkg-config and use the bare
// `-lztok` link directive in `cgo_nopkgconfig.go` instead. See that
// file + `cgo_pkgconfig.go` for the build-tag split.
