package ztok

// Smoke test for OpenRWKVWorld: builds a synthetic RWKV "World"-style
// vocab (rwkv_vocab_v20230424.txt format: `<id> <python-repr> <len>`)
// covering all 256 single bytes plus a few multi-byte merges, then
// checks greedy longest-match encoding and lossless round-trip.

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// rwkvFixture writes a tiny World vocab. Every byte 0..255 is present
// (ids 0..255 as single-byte `\xHH` reprs) so encode never hits
// NoTokenMatch, followed by a handful of ASCII merges that exercise the
// longest-wins trie walk.
func rwkvFixture(t testing.TB) string {
	t.Helper()
	dir := t.TempDir()
	p := filepath.Join(dir, "rwkv_vocab_test.txt")

	var b strings.Builder
	// Single bytes as bytes-form `\xHH` reprs (length 1 each).
	for i := 0; i < 256; i++ {
		fmt.Fprintf(&b, "%d b'\\x%02x' 1\n", i, i)
	}
	// Multi-byte merges, ids continue after the byte block.
	merges := []string{"ab", "abc", "hello", " world"}
	id := 256
	for _, m := range merges {
		fmt.Fprintf(&b, "%d %q %d\n", id, m, len(m))
		id++
	}

	if err := os.WriteFile(p, []byte(b.String()), 0o644); err != nil {
		t.Fatalf("write rwkv fixture: %v", err)
	}
	return p
}

func TestOpenRWKVWorldGreedyMatch(t *testing.T) {
	path := rwkvFixture(t)
	p, err := OpenRWKVWorld(path, nil)
	if err != nil {
		t.Fatalf("OpenRWKVWorld: %v", err)
	}
	defer p.Close()

	// "abc" must take the id-257 longest match (256="ab", 257="abc"),
	// not "ab"+"c" or three single bytes.
	ids, err := p.Encode("abc")
	if err != nil {
		t.Fatalf("encode abc: %v", err)
	}
	if len(ids) != 1 || ids[0] != 257 {
		t.Fatalf("greedy longest-match failed: got %v, want [257]", ids)
	}
}

func TestOpenRWKVWorldRoundTrip(t *testing.T) {
	path := rwkvFixture(t)
	p, err := OpenRWKVWorld(path, nil)
	if err != nil {
		t.Fatalf("OpenRWKVWorld: %v", err)
	}
	defer p.Close()

	// Includes a merge, raw bytes, and multi-byte UTF-8 (covered by the
	// single-byte block, so it round-trips byte-for-byte).
	for _, in := range []string{"abc", "hello world", "你好 abc", "\x00\x01\xff"} {
		ids, err := p.Encode(in)
		if err != nil {
			t.Fatalf("encode %q: %v", in, err)
		}
		out, err := p.Decode(ids)
		if err != nil {
			t.Fatalf("decode %q: %v", in, err)
		}
		if out != in {
			t.Fatalf("round-trip mismatch: in=%q out=%q ids=%v", in, out, ids)
		}
	}
}

func TestRWKVAutoDetect(t *testing.T) {
	path := rwkvFixture(t)
	if got := DetectFormat(path); got != FormatRWKV {
		t.Fatalf("DetectFormat = %v, want FormatRWKV", got)
	}
	// Open should dispatch to OpenRWKVWorld via auto-detect.
	p, err := Open(path)
	if err != nil {
		t.Fatalf("Open auto-detect: %v", err)
	}
	defer p.Close()
	if _, err := p.Encode("ab"); err != nil {
		t.Fatalf("encode after auto-detect Open: %v", err)
	}
}
