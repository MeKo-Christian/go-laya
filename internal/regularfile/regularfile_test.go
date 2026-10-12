package regularfile

import (
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A regular file opens and reads as os.Open's would.
func TestOpenRegular(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte("{}"), 0o600); err != nil {
		t.Fatal(err)
	}
	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open = %v", err)
	}
	defer f.Close()
	if b, err := io.ReadAll(f); err != nil || string(b) != "{}" {
		t.Fatalf("read %q, %v; want {}", b, err)
	}
}

// A directory opens on every platform, and must still be refused, as
// ErrNotRegular so that callers can tell it from a failed open.
func TestOpenDirectory(t *testing.T) {
	f, err := Open(t.TempDir())
	if err == nil {
		f.Close()
		t.Fatal("Open accepted a directory")
	}
	if !errors.Is(err, ErrNotRegular) || !strings.Contains(err.Error(), "not a regular file") {
		t.Fatalf("Open = %v, want ErrNotRegular", err)
	}
}

// A missing file is os.OpenFile's error.
func TestOpenMissing(t *testing.T) {
	if _, err := Open(filepath.Join(t.TempDir(), "absent")); !os.IsNotExist(err) || errors.Is(err, ErrNotRegular) {
		t.Fatalf("Open = %v, want a not-exist error", err)
	}
}
