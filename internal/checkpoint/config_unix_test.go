//go:build unix

package checkpoint

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

// TestLoadConfigFIFO: a named pipe at rl_agent_config.json, with no writer,
// must be refused at once rather than block LoadConfig, and with it Open on
// either backend, in an open no context can interrupt.
func TestLoadConfigFIFO(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, ConfigFile)
	if err := syscall.Mkfifo(path, 0o600); err != nil {
		t.Skipf("mkfifo: %v", err)
	}
	// Should LoadConfig block, a writer opening the pipe releases it, so the
	// goroutine does not outlive the test.
	t.Cleanup(func() {
		if w, err := os.OpenFile(path, os.O_WRONLY|syscall.O_NONBLOCK, 0); err == nil {
			w.Close()
		}
	})

	done := make(chan error, 1)
	go func() {
		_, err := LoadConfig(dir)
		done <- err
	}()
	select {
	case err := <-done:
		if !errors.Is(err, backend.ErrIncompatibleCheckpoint) ||
			!strings.Contains(err.Error(), ConfigFile) ||
			!strings.Contains(err.Error(), "not a regular file") {
			t.Fatalf("LoadConfig = %v, want ErrIncompatibleCheckpoint naming %s as not a regular file", err, ConfigFile)
		}
	case <-time.After(5 * time.Second):
		t.Fatalf("LoadConfig blocked on a FIFO at %s with no writer", ConfigFile)
	}
}
