package ztok

// Shared test fixture: writes a tiny synthetic .tiktoken vocab covering
// all 256 single bytes plus a handful of merges, into a per-package
// temp dir. Mirrors bindings/python/tests/conftest.py and
// bindings/nodejs/test/fixture.js so all language bindings stress the
// same surface.

import (
	"encoding/base64"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

var extras = []string{
	"he", "hel", "hell", "hello",
	" w", " wo", " wor", " worl", " world",
	"th", "the", " th", " the",
	"fo", "foo", "bar", "baz",
	" quick", " brown", " fox",
}

var (
	fixtureOnce sync.Once
	fixturePath string
	fixtureErr  error
)

// tiktokenFixture returns the path to the synthetic vocab, building it
// once per test binary invocation.
func tiktokenFixture(t testing.TB) string {
	t.Helper()
	fixtureOnce.Do(func() {
		dir, err := os.MkdirTemp("", "ztok-go-")
		if err != nil {
			fixtureErr = err
			return
		}
		p := filepath.Join(dir, "synthetic_cl100k.tiktoken")
		var b strings.Builder
		rank := 0
		for i := 0; i < 256; i++ {
			fmt.Fprintf(&b, "%s %d\n",
				base64.StdEncoding.EncodeToString([]byte{byte(i)}), rank)
			rank++
		}
		for _, extra := range extras {
			fmt.Fprintf(&b, "%s %d\n",
				base64.StdEncoding.EncodeToString([]byte(extra)), rank)
			rank++
		}
		if err := os.WriteFile(p, []byte(b.String()), 0o644); err != nil {
			fixtureErr = err
			return
		}
		fixturePath = p
	})
	if fixtureErr != nil {
		t.Fatalf("build tiktoken fixture: %v", fixtureErr)
	}
	return fixturePath
}
