//go:build unix

package regularfile

import (
	"errors"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"
)

// Opening a named pipe without a writer must neither block nor succeed. The
// type check runs on the opened descriptor, with no path-based check before
// it, so a regular file swapped for a FIFO between a stat and the open is
// refused too: Open is called on the FIFO directly.
func TestOpenFIFO(t *testing.T) {
	path := filepath.Join(t.TempDir(), "fifo")
	if err := syscall.Mkfifo(path, 0o600); err != nil {
		t.Skipf("mkfifo: %v", err)
	}

	done := make(chan error, 1)
	go func() {
		f, err := Open(path)
		if err == nil {
			f.Close()
		}
		done <- err
	}()
	select {
	case err := <-done:
		if !errors.Is(err, ErrNotRegular) || !strings.Contains(err.Error(), "not a regular file") {
			t.Fatalf("Open(FIFO) = %v, want ErrNotRegular", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Open blocked on a FIFO with no writer")
	}
}
