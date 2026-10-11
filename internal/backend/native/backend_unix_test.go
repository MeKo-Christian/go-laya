//go:build unix

package native

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/MeKo-Christian/go-laya/backend"
)

// TestOpenFIFO: a named pipe in place of either file, with no writer, must
// be refused at once rather than block Open, which ctx cannot interrupt
// while the open itself waits.
func TestOpenFIFO(t *testing.T) {
	for _, file := range []string{"model.safetensors", filepath.Join("encoder", "config.json")} {
		t.Run(file, func(t *testing.T) {
			dir := newSynth(4).write(t)
			path := filepath.Join(dir, file)
			if err := os.Remove(path); err != nil {
				t.Fatal(err)
			}
			if err := syscall.Mkfifo(path, 0o600); err != nil {
				t.Skipf("mkfifo: %v", err)
			}
			// Should Open block, a writer opening the pipe releases it, so
			// the goroutine does not outlive the test.
			t.Cleanup(func() {
				if w, err := os.OpenFile(path, os.O_WRONLY|syscall.O_NONBLOCK, 0); err == nil {
					w.Close()
				}
			})

			done := make(chan error, 1)
			go func() {
				b, err := Open(t.Context(), dir, Options{})
				if err == nil {
					b.Close()
				}
				done <- err
			}()
			select {
			case err := <-done:
				if !errors.Is(err, backend.ErrIncompatibleCheckpoint) ||
					!strings.Contains(err.Error(), filepath.Base(file)) ||
					!strings.Contains(err.Error(), "not a regular file") {
					t.Fatalf("Open = %v, want ErrIncompatibleCheckpoint naming %s as not a regular file", err, filepath.Base(file))
				}
			case <-time.After(5 * time.Second):
				t.Fatalf("Open blocked on a FIFO at %s with no writer", file)
			}
		})
	}
}
