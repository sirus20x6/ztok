package ztok

// Tekken (`tekken.json`) tokenizer tests for the Go binding.
//
// Loads the real mistral_nemo_tekken.json fixture (skipped when absent)
// and asserts ztok reproduces the canonical reference encodings, which
// were verified against mistral_common 1.8.6. Also checks that
// auto-detect routes Tekken vocabs to OpenTekken.

import (
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"
)

// fileExists reports whether path names an existing file.
func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

// tekkenVocabPath returns the path to the real Tekken fixture under
// bench/vocabs/. bindings/go/ -> repo root -> bench/vocabs/...
func tekkenVocabPath() string {
	_, thisFile, _, _ := runtime.Caller(0)
	root := filepath.Dir(filepath.Dir(filepath.Dir(thisFile)))
	return filepath.Join(root, "bench", "vocabs", "mistral_nemo_tekken.json")
}

func openTekkenFixture(t testing.TB) *Pipeline {
	t.Helper()
	path := tekkenVocabPath()
	if !fileExists(path) {
		t.Skipf("Tekken vocab fixture not present at %s", path)
	}
	p, err := OpenTekken(path, nil)
	if err != nil {
		t.Fatalf("OpenTekken(%q): %v", path, err)
	}
	return p
}

// goldenTekken mirrors the cross-binding parity vectors: ids verified
// against mistral_common 1.8.6 on mistral_nemo_tekken.json.
var goldenTekken = []struct {
	text string
	want []uint32
}{
	{"Hello, world!", []uint32{22177, 1044, 4304, 1033}},
	{"The quick brown fox", []uint32{1784, 7586, 22980, 94137}},
	{" and the", []uint32{1321, 1278}},
}

func TestOpenTekkenGolden(t *testing.T) {
	p := openTekkenFixture(t)
	defer p.Close()

	for _, tc := range goldenTekken {
		ids, err := p.Encode(tc.text)
		if err != nil {
			t.Fatalf("encode %q: %v", tc.text, err)
		}
		if !reflect.DeepEqual(ids, tc.want) {
			t.Errorf("Encode(%q) = %v, want %v", tc.text, ids, tc.want)
		}
	}
}

func TestTekkenAutoDetect(t *testing.T) {
	path := tekkenVocabPath()
	if !fileExists(path) {
		t.Skipf("Tekken vocab fixture not present at %s", path)
	}
	if got := DetectFormat(path); got != FormatTekken {
		t.Fatalf("DetectFormat = %v, want FormatTekken", got)
	}
	// Open should dispatch to OpenTekken via auto-detect and match the
	// golden ids.
	p, err := Open(path)
	if err != nil {
		t.Fatalf("Open auto-detect: %v", err)
	}
	defer p.Close()

	ids, err := p.Encode("Hello, world!")
	if err != nil {
		t.Fatalf("encode after auto-detect Open: %v", err)
	}
	want := []uint32{22177, 1044, 4304, 1033}
	if !reflect.DeepEqual(ids, want) {
		t.Errorf("auto-detect Encode = %v, want %v", ids, want)
	}
}
