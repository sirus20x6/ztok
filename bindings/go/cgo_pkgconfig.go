//go:build !nopkgconfig
// +build !nopkgconfig

// Default build: resolve include + link flags through pkg-config. After
// `zig build -p prefix`, dropping `prefix/lib/pkgconfig` on
// `PKG_CONFIG_PATH` lets cgo pick everything up automatically.
//
// If pkg-config isn't available on the system (e.g. minimal Docker
// builds), build with `-tags nopkgconfig` to switch to the bare
// `-lztok` directive and supply paths via `CGO_CFLAGS` /
// `CGO_LDFLAGS` instead.

package ztok

/*
#cgo pkg-config: ztok
*/
import "C"
