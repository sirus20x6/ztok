package ztok

// Tokenizer-fingerprint tests over a byte_id pipeline. Mirror the
// rust/dotnet/java fingerprint tests: 32-byte length + determinism, plus
// a cross-binding golden value shared with the
// python/nodejs/ruby/rust/dotnet/java bindings.

import (
	"testing"
)

// goldenByteIDFingerprint is the fingerprint of the default byte_id
// pipeline. Computed directly from libztok's ztok_fingerprint and shared
// across all bindings to confirm cross-binding agreement.
const goldenByteIDFingerprint = "201ecf86554b5a970471e0189d7e78dc2c3df24519f7d5ebb0caacc86701e77c"

func TestFingerprintIs32Bytes(t *testing.T) {
	p, err := NewByteID(nil)
	if err != nil {
		t.Fatalf("NewByteID: %v", err)
	}
	defer p.Close()

	fp, err := p.Fingerprint()
	if err != nil {
		t.Fatalf("Fingerprint: %v", err)
	}
	if len(fp) != FingerprintSize {
		t.Fatalf("fingerprint length = %d, want %d", len(fp), FingerprintSize)
	}
	if len(fp.Hex()) != 64 {
		t.Fatalf("hex length = %d, want 64", len(fp.Hex()))
	}
	allZero := true
	for _, b := range fp {
		if b != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Fatal("fingerprint is all zeros")
	}
}

func TestFingerprintIsDeterministic(t *testing.T) {
	a, err := NewByteID(nil)
	if err != nil {
		t.Fatalf("NewByteID a: %v", err)
	}
	defer a.Close()
	b, err := NewByteID(nil)
	if err != nil {
		t.Fatalf("NewByteID b: %v", err)
	}
	defer b.Close()

	fa, err := a.Fingerprint()
	if err != nil {
		t.Fatalf("Fingerprint a: %v", err)
	}
	fb, err := b.Fingerprint()
	if err != nil {
		t.Fatalf("Fingerprint b: %v", err)
	}
	if fa != fb {
		t.Fatalf("fingerprints differ: %s vs %s", fa.Hex(), fb.Hex())
	}
}

func TestFingerprintGoldenByteID(t *testing.T) {
	p, err := NewByteID(nil)
	if err != nil {
		t.Fatalf("NewByteID: %v", err)
	}
	defer p.Close()

	fp, err := p.Fingerprint()
	if err != nil {
		t.Fatalf("Fingerprint: %v", err)
	}
	if got := fp.Hex(); got != goldenByteIDFingerprint {
		t.Fatalf("fingerprint = %s, want %s", got, goldenByteIDFingerprint)
	}
}
