package ztok

// PRNG-driven round-trip fuzz harness for the ztok Go binding.
//
// Mirrors fuzz/encode_decode.zig in shape: each fuzz invocation
// encode-then-decodes a byte input through the byte_id pipeline
// (identity normalizer + identity pre-tok + byte_id model + concat
// decoder). By construction every input byte maps to exactly one id
// and decoding concatenates them back, so for ANY byte sequence we
// must have DecodeBytes(EncodeBytes(x)) == x.
//
// Uses Go's native testing.F fuzz API so `go test -fuzz=FuzzRoundTrip
// -fuzztime=10s ./...` drives it with the runtime's coverage-guided
// mutator on top of a hand-picked seed corpus. Skips cleanly if
// libztok isn't available (the loader will surface that as an error
// from NewByteID).

import (
	"bytes"
	"testing"
)

// FuzzRoundTrip seeds the libFuzzer with a handful of corpus entries
// — empty, ASCII, UTF-8 with combining marks, the full 0..255 byte
// range, and a multi-codepoint emoji — then lets the runtime mutate
// the inputs. The byte_id pipeline guarantees a 1:1 byte<->id mapping
// so any divergence is a bug in encode bookkeeping or decode's
// byte-reassembly path (same property the Zig harness checks).
//
// Run as:
//
//	go test -run='^$' -fuzz=FuzzRoundTrip -fuzztime=10s ./...
//
// The plain `go test ./...` invocation just runs the seed corpus once
// each (it's a built-in correctness pass before the mutator kicks in).
func FuzzRoundTrip(f *testing.F) {
	// Seed corpus: cover the cases the Zig harness biases toward.
	seeds := [][]byte{
		// 1. Empty.
		nil,
		// 2. ASCII printable.
		[]byte("the quick brown fox jumps over the lazy dog"),
		// 3. UTF-8 with combining marks (NFD-style decomposed e-acute).
		//    é hits the cl100k carry path; here we just round-trip
		//    the raw bytes through byte_id.
		[]byte("e\xCC\x81 cafe\xCC\x81"),
		// 4. All 256 byte values 0x00..0xFF — the hardest case for any
		//    encoder that special-cases NUL or non-UTF-8 sequences.
		allBytes(),
		// 5. Multi-codepoint emoji (4-byte UTF-8 lead).
		[]byte("\xF0\x9F\x98\x80\xF0\x9F\x91\x8B\xF0\x9F\x8C\x88"),
	}
	for _, s := range seeds {
		f.Add(s)
	}

	// One pipeline shared across every fuzz call — libztok pipelines
	// are thread-safe for encode/decode (see TestConcurrentEncodeIsSafe
	// in ztok_test.go) and re-creating one per iteration would dominate
	// the wall clock at high mutation rates.
	pipe, err := NewByteID(nil)
	if err != nil {
		f.Skipf("libztok not available: %v", err)
	}
	f.Cleanup(func() {
		// Deterministic release; finalizer would also catch it but
		// Cleanup keeps the fuzz cache directory free of leaked handles.
		_ = pipe.Close()
	})

	f.Fuzz(func(t *testing.T, data []byte) {
		ids, err := pipe.EncodeBytes(data)
		if err != nil {
			t.Fatalf("EncodeBytes(%d bytes): %v", len(data), err)
		}
		// byte_id maps 1:1 — id count must equal input byte length.
		if len(ids) != len(data) {
			t.Fatalf("byte_id produced %d ids for %d-byte input", len(ids), len(data))
		}

		out, err := pipe.DecodeBytes(ids)
		if err != nil {
			t.Fatalf("DecodeBytes (input=%d bytes, ids=%d): %v",
				len(data), len(ids), err)
		}

		// Normalize the nil/empty distinction: Encode returns nil for
		// an empty input, and DecodeBytes returns nil for empty ids.
		// bytes.Equal handles nil==[]byte{} correctly already.
		if !bytes.Equal(out, data) {
			t.Fatalf("round-trip mismatch\n  in (%d bytes):  %x\n  out (%d bytes): %x",
				len(data), data, len(out), out)
		}
	})
}

// allBytes returns a slice containing every byte value 0x00..0xFF.
func allBytes() []byte {
	out := make([]byte, 256)
	for i := range out {
		out[i] = byte(i)
	}
	return out
}
