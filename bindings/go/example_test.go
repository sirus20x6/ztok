package ztok_test

// Doc examples — rendered into `go doc` output via the godoc convention.
// These compile-and-run (via `go test`), so they double as regression
// coverage for the public API shape.

import (
	"fmt"
	"log"

	ztok "github.com/sirus20x6/ztok-go"
)

func ExampleVersion() {
	v := ztok.Version()
	if v == "" {
		log.Fatal("ztok version unavailable")
	}
	// Output is non-deterministic across releases, so just check shape:
	fmt.Println(len(v) > 0)
	// Output: true
}

func ExamplePipeline_Encode() {
	// Real callers would point at e.g. cl100k_base.tiktoken. Here we
	// build a baseline byte_id pipeline so the example runs anywhere.
	pipe, err := ztok.NewByteID(nil)
	if err != nil {
		log.Fatal(err)
	}
	defer pipe.Close()

	ids, _ := pipe.Encode("hi")
	out, _ := pipe.Decode(ids)
	fmt.Printf("%d ids, decode=%q\n", len(ids), out)
	// Output: 2 ids, decode="hi"
}

func ExampleBatchPool() {
	pipe, _ := ztok.NewByteID(nil)
	defer pipe.Close()

	pool, _ := ztok.NewBatchPool(&ztok.BatchPoolOptions{Workers: 2})
	defer pool.Close()

	results, _ := pipe.EncodeBatch(pool, []string{"foo", "bar", "baz"})
	fmt.Println(len(results))
	// Output: 3
}
