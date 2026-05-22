// quickstart: load a tokenizer, encode, decode.
//
// Run from the repo root after `zig build` succeeds:
//
//	LD_LIBRARY_PATH=zig-out/lib \
//	  PKG_CONFIG_PATH=zig-out/lib/pkgconfig \
//	  CGO_CFLAGS=-Izig-out/include \
//	  go run ./bindings/go/examples/quickstart path/to/tokenizer.{tiktoken,json,model,ztm}
package main

import (
	"fmt"
	"log"
	"os"
	"time"

	ztok "github.com/sirus20x6/ztok-go"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: quickstart <tokenizer-file>")
		os.Exit(2)
	}

	pipe, err := ztok.Open(os.Args[1])
	if err != nil {
		log.Fatalf("ztok.Open(%q): %v", os.Args[1], err)
	}
	defer pipe.Close()

	fmt.Printf("ztok %s\n", ztok.Version())
	fmt.Printf("detected format: %s\n", ztok.DetectFormat(os.Args[1]))

	const text = "hello world"
	ids, err := pipe.Encode(text)
	if err != nil {
		log.Fatalf("encode: %v", err)
	}
	out, err := pipe.Decode(ids)
	if err != nil {
		log.Fatalf("decode: %v", err)
	}
	fmt.Printf("encode(%q) -> %d ids: %v\n", text, len(ids), ids[:min(10, len(ids))])
	fmt.Printf("decode -> %q\n", out)

	// Tiny cgo overhead probe — ~1M Version() calls, divide for ns/call.
	const probes = 1_000_000
	start := time.Now()
	for i := 0; i < probes; i++ {
		_ = ztok.Version()
	}
	elapsed := time.Since(start)
	fmt.Printf("ztok.Version() cgo overhead: %v / %d calls = %.1f ns/call\n",
		elapsed, probes, float64(elapsed.Nanoseconds())/float64(probes))
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
