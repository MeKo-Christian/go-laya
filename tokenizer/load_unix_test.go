//go:build unix

package tokenizer

import (
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"
)

// TestOpenFIFO: a named pipe with no writer at either tokenizer file must be
// refused at once rather than block Open, and with it laya.Open, in an open
// no context can interrupt. The other file is a valid fixture, so the pipe is
// the only thing wrong.
func TestOpenFIFO(t *testing.T) {
	for _, name := range []string{fileTokenizerJSON, fileTokenizerConfig} {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			for _, f := range []string{fileTokenizerJSON, fileTokenizerConfig} {
				raw, err := os.ReadFile(filepath.Join("testdata", "mini_en", f))
				if err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(dir, f), raw, 0o600); err != nil {
					t.Fatal(err)
				}
			}
			path := filepath.Join(dir, name)
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
				_, err := Open(dir)
				done <- err
			}()
			select {
			case err := <-done:
				if err == nil ||
					!strings.Contains(err.Error(), "tokenizer: read "+name) ||
					!strings.Contains(err.Error(), "not a regular file") {
					t.Fatalf("Open = %v, want a read error naming %s as not a regular file", err, name)
				}
			case <-time.After(5 * time.Second):
				t.Fatalf("Open blocked on a FIFO at %s with no writer", name)
			}
		})
	}
}
