//go:build nopkgconfig
// +build nopkgconfig

// Fallback build (`go build -tags nopkgconfig`): bare `-lztok` link
// directive, no pkg-config. Set `CGO_CFLAGS=-I/path/to/ztok/include`
// and `CGO_LDFLAGS=-L/path/to/ztok/lib` so the toolchain can find
// `ztok.h` and `libztok.{so,dylib,dll}`.

package ztok

/*
#cgo LDFLAGS: -lztok
*/
import "C"
