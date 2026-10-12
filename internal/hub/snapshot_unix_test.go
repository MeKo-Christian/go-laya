//go:build unix

package hub

import (
	"context"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"syscall"
	"testing"
	"time"
)

// TestReadCachedFIFO: a ref or listing that becomes a named pipe with no
// writer after cached accepted it is a miss, as if cached had refused it, so
// an offline Snapshot fails with ErrNotCached instead of blocking in an open
// no context can interrupt. cached already refuses a pipe that is there
// before the call, so the pipe is swapped in through afterCached, between the
// check and the open.
func TestReadCachedFIFO(t *testing.T) {
	for _, c := range []struct {
		name, rev, rel string
	}{
		{"ref", "main", "refs/main"},
		{"listing", commit, "listings/" + commit + ".json"},
	} {
		t.Run(c.name, func(t *testing.T) {
			f := newFakeRepo(t)
			online := f.client(t, "")
			if _, err := online.Snapshot(context.Background(), "org/model", "main", []string{"multilingual/*"}); err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(online.Dir, "org", "model", filepath.FromSlash(c.rel))
			swapped := false
			afterCached = func(p string) {
				if p != path {
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
			t.Cleanup(func() { afterCached = nil })
			// Should the open block, a writer opening the pipe releases it,
			// so the goroutine does not outlive the test.
			t.Cleanup(func() {
				if w, err := os.OpenFile(path, os.O_WRONLY|syscall.O_NONBLOCK, 0); err == nil {
					w.Close()
				}
			})

			offline := &Client{
				Endpoint: f.srv.URL, Dir: online.Dir, Offline: true,
				HTTP: &http.Client{Transport: failTransport{t}},
			}
			done := make(chan error, 1)
			go func() {
				_, err := offline.Snapshot(context.Background(), "org/model", c.rev, []string{"multilingual/*"})
				done <- err
			}()
			select {
			case err := <-done:
				if !swapped {
					t.Fatalf("readCached never reached afterCached for %s", path)
				}
				if !errors.Is(err, ErrNotCached) {
					t.Fatalf("Snapshot = %v, want ErrNotCached", err)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("Snapshot blocked on a FIFO with no writer")
			}
		})
	}
}
