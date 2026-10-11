//go:build unix

package safetensors

import (
	"os"
	"path/filepath"
	"syscall"
	"testing"
	"time"
)

// TestReadHeaderFIFO: a named pipe at the weights path, with no writer, must
// be refused at once rather than block ReadHeader in an open nothing can
// interrupt.
func TestReadHeaderFIFO(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.safetensors")
	if err := syscall.Mkfifo(path, 0o600); err != nil {
		t.Skipf("mkfifo: %v", err)
	}
	// Should ReadHeader block, a writer opening the pipe releases it, so the
	// goroutine does not outlive the test.
	t.Cleanup(func() {
		if w, err := os.OpenFile(path, os.O_WRONLY|syscall.O_NONBLOCK, 0); err == nil {
			w.Close()
		}
	})

	done := make(chan error, 1)
	go func() {
		_, err := ReadHeader(path)
		done <- err
	}()
	select {
	case err := <-done:
		wantIncompatible(t, err, path, "not a regular file")
	case <-time.After(5 * time.Second):
		t.Fatal("ReadHeader blocked on a FIFO with no writer")
	}
}
