//go:build !android && !ios && ((darwin && (amd64 || arm64)) || (linux && (amd64 || arm64 || loong64)) || (netbsd && (amd64 || arm64)))

package onnx

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"google.golang.org/protobuf/encoding/protowire"

	"github.com/MeKo-Christian/go-laya/internal/ortlib"
)

// headerOnlyGraph writes the smallest file internal/onnxheader accepts, an IR
// 8 model with an empty graph and an ai.onnx opset 17 import, so that Open
// gets as far as resolving the library without ONNX Runtime.
func headerOnlyGraph(t *testing.T) string {
	t.Helper()
	var opset []byte
	opset = protowire.AppendTag(opset, 2, protowire.VarintType)
	opset = protowire.AppendVarint(opset, 17)
	var m []byte
	m = protowire.AppendTag(m, 1, protowire.VarintType)
	m = protowire.AppendVarint(m, 8)
	m = protowire.AppendTag(m, 7, protowire.BytesType)
	m = protowire.AppendBytes(m, nil)
	m = protowire.AppendTag(m, 8, protowire.BytesType)
	m = protowire.AppendBytes(m, opset)
	p := filepath.Join(t.TempDir(), "model.onnx")
	if err := os.WriteFile(p, m, 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

// Task 7.7.7 (D29): Options.CacheDir reaches the library lookup Open runs.
// The pinned download there fails its hash, so Open fails with
// ortlib.ErrHashMismatch before loading anything; ignoring CacheDir would
// look in $LAYA_CACHE, find nothing and load the empty candidate instead.
func TestOpenCacheDir(t *testing.T) {
	cache := t.TempDir()
	pinnedLibrary(t, cache)
	t.Setenv("LAYA_CACHE", t.TempDir())
	t.Setenv("LAYA_ORT_LIB", "")
	t.Setenv("ORT_LIBRARY_PATH", "")
	prev := ortLibraryCandidates
	ortLibraryCandidates = []string{touch(t, filepath.Join(t.TempDir(), "candidate.so"))}
	t.Cleanup(func() { ortLibraryCandidates = prev })

	b, err := Open(headerOnlyGraph(t), Options{Device: "cpu", CacheDir: cache})
	if err == nil {
		_ = b.Close()
	}
	if !errors.Is(err, ortlib.ErrHashMismatch) {
		t.Fatalf("Open(CacheDir %q) = %v, want ortlib.ErrHashMismatch", cache, err)
	}
}
