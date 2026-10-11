//go:build unix

package laya

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"
)

// Backlog B.14: a named pipe at rl_agent_config.json, with no writer, is
// refused at once on either runtime, before any backend is opened, rather
// than block Open in an open its context cannot end.
func TestOpenConfigFIFO(t *testing.T) {
	testOpenFIFO(t, "rl_agent_config.json")
}

// Backlog B.17: the same for each tokenizer file, which readCheckpoint reads
// before either runtime's backend is opened.
func TestOpenTokenizerFIFO(t *testing.T) {
	for _, name := range []string{"tokenizer.json", "tokenizer_config.json"} {
		t.Run(name, func(t *testing.T) {
			testOpenFIFO(t, filepath.Join("tokenizer", name))
		})
	}
}

// testOpenFIFO replaces rel, a file of a valid checkpoint, with a named pipe
// that has no writer, and requires Open to refuse it at once on either runtime
// with ErrIncompatibleCheckpoint naming the file, having opened no backend.
func testOpenFIFO(t *testing.T, rel string) {
	t.Helper()
	name := filepath.Base(rel)
	graph := writeGraph(t, t.TempDir(), "x")
	for _, tc := range []struct {
		name string
		opts []Option
	}{
		{"onnx", []Option{WithGraph(graph)}},
		{"native", []Option{WithRuntime(RuntimeNative)}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ck := t.TempDir()
			writeNativeCheckpoint(t, ck)
			path := filepath.Join(ck, rel)
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

			// ck as the snapshot root, so the stub never calls t.Fatalf from
			// the goroutine below.
			rec := &runtimeRec{}
			done := make(chan error, 1)
			go func() {
				a, err := openAgent(context.Background(), ck, tc.opts, stubRuntimes(t, ck, rec))
				if err == nil {
					a.Close()
				}
				done <- err
			}()
			select {
			case err := <-done:
				if !errors.Is(err, ErrIncompatibleCheckpoint) ||
					!strings.Contains(err.Error(), name) ||
					!strings.Contains(err.Error(), "not a regular file") {
					t.Fatalf("Open = %v, want ErrIncompatibleCheckpoint naming %s as not a regular file", err, name)
				}
				if len(rec.snapshots)+len(rec.onnxOpens)+len(rec.nativeOpens) != 0 {
					t.Errorf("snapshots %v, ONNX opens %v, native opens %v; want none",
						rec.snapshots, rec.onnxOpens, rec.nativeOpens)
				}
			case <-time.After(5 * time.Second):
				t.Fatalf("Open blocked on a FIFO at %s with no writer", name)
			}
		})
	}
}
