//go:build unix

package onnxheader

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

// TestReadFIFO: a named pipe with no writer at the graph path is refused at
// once, by the Lstat before anything opens it, and at the open should the
// path become one after that Lstat. Either way Read must not block in an open
// no context can interrupt.
func TestReadFIFO(t *testing.T) {
	for _, c := range []struct {
		name string
		// swap replaces the graph after the Lstat when true, at once
		// otherwise.
		swap bool
	}{
		{"at the path", false},
		{"swapped in after the stat", true},
	} {
		t.Run(c.name, func(t *testing.T) {
			path := writeModel(t, model(10, 18, exportIO()))
			mkfifo := func() {
				if err := os.Remove(path); err != nil {
					t.Error(err)
					return
				}
				if err := syscall.Mkfifo(path, 0o600); err != nil {
					t.Errorf("mkfifo: %v", err)
				}
			}
			if c.swap {
				afterStat = func(string) { mkfifo() }
				t.Cleanup(func() { afterStat = nil })
			} else {
				mkfifo()
			}
			// Should Read block, a writer opening the pipe releases it, so
			// the goroutine does not outlive the test.
			t.Cleanup(func() {
				if w, err := os.OpenFile(path, os.O_WRONLY|syscall.O_NONBLOCK, 0); err == nil {
					w.Close()
				}
			})

			done := make(chan error, 1)
			go func() {
				_, err := Read(path)
				done <- err
			}()
			select {
			case err := <-done:
				if !errors.Is(err, backend.ErrIncompatibleCheckpoint) ||
					!strings.Contains(err.Error(), filepath.Base(path)) ||
					!strings.Contains(err.Error(), "not a regular file") {
					t.Fatalf("Read = %v, want ErrIncompatibleCheckpoint naming the graph as not a regular file", err)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("Read blocked on a FIFO with no writer")
			}
		})
	}
}
