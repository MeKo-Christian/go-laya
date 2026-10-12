//go:build unix

package ortlib

import (
	"context"
	"errors"
	"os"
	"strings"
	"syscall"
	"testing"
	"time"
)

// TestCachedFIFO: a library that becomes a named pipe with no writer after
// Cached's Lstat is ErrHashMismatch, as a pipe the Lstat sees is, instead of
// an open that blocks with no context to end it. The Lstat refuses a pipe
// that is there before the call, so the pipe is swapped in through
// afterLstat, between the check and the open.
func TestCachedFIFO(t *testing.T) {
	archive := tgz(t, entry{name: member, body: libBytes})
	rel := release(archive)
	srv, _ := server(t, rel.Archive, archive)
	c := &Client{BaseURL: srv.URL, Dir: t.TempDir()}
	lib, err := c.Download(context.Background(), rel)
	if err != nil {
		t.Fatal(err)
	}

	swapped := false
	afterLstat = func(p string) {
		if p != lib {
			return
		}
		swapped = true
		if err := os.Remove(p); err != nil {
			t.Error(err)
			return
		}
		if err := syscall.Mkfifo(p, 0o600); err != nil {
			t.Errorf("mkfifo: %v", err)
		}
	}
	t.Cleanup(func() { afterLstat = nil })
	// Should the open block, a writer opening the pipe releases it, so the
	// goroutine does not outlive the test.
	t.Cleanup(func() {
		if w, err := os.OpenFile(lib, os.O_WRONLY|syscall.O_NONBLOCK, 0); err == nil {
			w.Close()
		}
	})

	done := make(chan error, 1)
	go func() {
		_, err := c.Cached(rel)
		done <- err
	}()
	select {
	case err := <-done:
		if !swapped {
			t.Fatalf("Cached never reached afterLstat for %s", lib)
		}
		if !errors.Is(err, ErrHashMismatch) || !strings.Contains(err.Error(), lib) ||
			!strings.Contains(err.Error(), "not a regular file") {
			t.Fatalf("Cached = %v, want ErrHashMismatch naming the library as not a regular file", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Cached blocked on a FIFO with no writer")
	}
}
