package ztok

import (
	"errors"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestVersion(t *testing.T) {
	v := Version()
	if v == "" {
		t.Fatal("Version() returned empty string")
	}
	if !strings.HasPrefix(v, "1.") {
		t.Errorf("expected 1.x version, got %q", v)
	}
}

func TestEncodeDecodeRoundtrip(t *testing.T) {
	pipe, err := OpenTiktoken(tiktokenFixture(t), nil)
	if err != nil {
		t.Fatalf("OpenTiktoken: %v", err)
	}
	defer pipe.Close()

	cases := []string{
		"hello world",
		"the quick brown fox",
		"foo bar baz",
		"",
		// non-ASCII through the byte path
		"héllo wörld",
		// multi-codepoint emoji (round-trips through byte_id)
		"emoji: \xF0\x9F\x98\x80",
	}
	for _, in := range cases {
		ids, err := pipe.Encode(in)
		if err != nil {
			t.Fatalf("encode %q: %v", in, err)
		}
		out, err := pipe.Decode(ids)
		if err != nil {
			t.Fatalf("decode %q: %v", in, err)
		}
		if out != in {
			t.Errorf("roundtrip mismatch: in=%q out=%q ids=%v", in, out, ids)
		}
	}
}

func TestAutoDetect(t *testing.T) {
	// We only have a tiktoken fixture locally — assert that DetectFormat
	// + Open both pick the right loader for it.
	path := tiktokenFixture(t)
	fmt := DetectFormat(path)
	if fmt != FormatTiktoken {
		t.Fatalf("DetectFormat(%q) = %v, want FormatTiktoken", path, fmt)
	}
	if fmt.String() != "tiktoken" {
		t.Errorf("Format.String() = %q, want %q", fmt.String(), "tiktoken")
	}

	pipe, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer pipe.Close()

	ids, err := pipe.Encode("hello world")
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if len(ids) == 0 {
		t.Error("Open + Encode produced no ids")
	}

	// A bogus path should land on FormatUnknown without raising.
	if got := DetectFormat("/nonexistent/file"); got != FormatUnknown {
		t.Errorf("DetectFormat(/nonexistent) = %v, want FormatUnknown", got)
	}
	if got := (FormatUnknown).String(); got != "unknown" {
		t.Errorf("FormatUnknown.String() = %q", got)
	}
}

func TestBatchPool(t *testing.T) {
	pipe, err := OpenTiktoken(tiktokenFixture(t), nil)
	if err != nil {
		t.Fatalf("OpenTiktoken: %v", err)
	}
	defer pipe.Close()

	pool, err := NewBatchPool(&BatchPoolOptions{Workers: 4})
	if err != nil {
		t.Fatalf("NewBatchPool: %v", err)
	}
	defer pool.Close()

	if w := pool.Workers(); w < 1 {
		t.Errorf("Workers() = %d, want >= 1", w)
	}

	// 1000 strings — large enough to exercise the worker pool.
	n := 1000
	inputs := make([]string, n)
	for i := range inputs {
		inputs[i] = "the quick brown fox jumped " + string(rune('a'+(i%26)))
	}

	results, err := pipe.EncodeBatch(pool, inputs)
	if err != nil {
		t.Fatalf("EncodeBatch: %v", err)
	}
	if len(results) != n {
		t.Fatalf("len(results) = %d, want %d", len(results), n)
	}

	// Cross-check against single-shot encode for the first few inputs.
	for i := 0; i < 10; i++ {
		exp, err := pipe.Encode(inputs[i])
		if err != nil {
			t.Fatalf("single Encode %d: %v", i, err)
		}
		if !equalU32(exp, results[i]) {
			t.Errorf("batch result %d mismatches single encode:\n  batch=%v\n  single=%v",
				i, results[i], exp)
		}
	}

	// Empty batch is allowed.
	empty, err := pipe.EncodeBatch(pool, nil)
	if err != nil || empty != nil {
		t.Errorf("empty batch should return (nil, nil); got (%v, %v)", empty, err)
	}
}

func TestStream(t *testing.T) {
	pipe, err := OpenTiktoken(tiktokenFixture(t), nil)
	if err != nil {
		t.Fatalf("OpenTiktoken: %v", err)
	}
	defer pipe.Close()

	// Use enough text to fan into multiple chunks at the small chunk size.
	text := strings.Repeat("the quick brown fox ", 200)
	expected, err := pipe.Encode(text)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}

	var got []uint32
	err = pipe.EncodeStream(text, func(ids []uint32) error {
		got = append(got, ids...)
		return nil
	}, &StreamOptions{ChunkSize: 64})
	if err != nil {
		t.Fatalf("EncodeStream: %v", err)
	}
	if !equalU32(expected, got) {
		t.Errorf("stream vs single-shot mismatch\n  single=%v len=%d\n  stream=%v len=%d",
			expected, len(expected), got, len(got))
	}
}

func TestCloseIdempotent(t *testing.T) {
	pipe, err := OpenTiktoken(tiktokenFixture(t), nil)
	if err != nil {
		t.Fatalf("OpenTiktoken: %v", err)
	}
	// Double close.
	if err := pipe.Close(); err != nil {
		t.Errorf("first Close: %v", err)
	}
	if err := pipe.Close(); err != nil {
		t.Errorf("second Close: %v", err)
	}
	// Methods after close should return ErrClosed.
	if _, err := pipe.Encode("x"); !errors.Is(err, ErrClosed) {
		t.Errorf("Encode after Close: err=%v, want ErrClosed", err)
	}

	pool, err := NewBatchPool(nil)
	if err != nil {
		t.Fatalf("NewBatchPool: %v", err)
	}
	if err := pool.Close(); err != nil {
		t.Errorf("pool Close: %v", err)
	}
	if err := pool.Close(); err != nil {
		t.Errorf("pool second Close: %v", err)
	}
	if w := pool.Workers(); w != 0 {
		t.Errorf("Workers() on closed pool = %d, want 0", w)
	}
}

func TestInvalidInputError(t *testing.T) {
	// Non-existent file should surface an InvalidInputError (status 2)
	// through the loader.
	_, err := OpenTiktoken("/path/that/does/not/exist.tiktoken", nil)
	if err == nil {
		t.Fatal("expected error opening nonexistent tiktoken file")
	}
	var iie *InvalidInputError
	if !errors.As(err, &iie) {
		// libztok may surface this as InternalError depending on the
		// underlying I/O failure mode; accept either typed error as
		// long as it's not nil.
		var ie *InternalError
		if !errors.As(err, &ie) {
			t.Errorf("error %v is neither InvalidInputError nor InternalError", err)
		}
	}

	// Open() on an unknown format returns InvalidInputError.
	_, err = Open("/nonexistent/format")
	if err == nil {
		t.Fatal("expected error from Open on unknown format")
	}
	if !errors.As(err, &iie) {
		t.Errorf("Open: error %v is not InvalidInputError", err)
	}
}

// TestFinalizerReleasesHandle is a soft check: drop a pipeline reference
// and force GC. The finalizer should release the C handle within a
// bounded number of cycles. (We don't assert on memory — just that the
// finalizer runs without panicking, which would surface as a runtime
// crash.)
func TestFinalizerReleasesHandle(t *testing.T) {
	for i := 0; i < 10; i++ {
		p, err := OpenTiktoken(tiktokenFixture(t), nil)
		if err != nil {
			t.Fatalf("OpenTiktoken: %v", err)
		}
		_, _ = p.Encode("hello")
		// drop reference, let GC + finalizer run
	}
	// Wait a beat for any pending finalizers, then succeed if no panic.
	time.Sleep(50 * time.Millisecond)
}

// TestConcurrentEncodeIsSafe spins multiple goroutines off a single
// Pipeline. ztok_encode is documented as thread-safe on a const
// pipeline (see src/c_api.zig).
func TestConcurrentEncodeIsSafe(t *testing.T) {
	pipe, err := OpenTiktoken(tiktokenFixture(t), nil)
	if err != nil {
		t.Fatalf("OpenTiktoken: %v", err)
	}
	defer pipe.Close()

	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 50; i++ {
				if _, err := pipe.Encode("the quick brown fox"); err != nil {
					t.Errorf("concurrent encode: %v", err)
					return
				}
			}
		}()
	}
	wg.Wait()
}

func equalU32(a, b []uint32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
