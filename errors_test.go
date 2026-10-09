package laya

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/MeKo-Christian/go-laya/backend"
	"github.com/MeKo-Christian/go-laya/internal/backend/onnx"
)

// A backend wraps backend.ErrIncompatibleCheckpoint because it cannot import
// the root package; a caller holding only the documented root name must still
// match it.
func TestErrIncompatibleCheckpoint(t *testing.T) {
	wrapped := fmt.Errorf("onnx backend: model.onnx: missing input qtype: %w", backend.ErrIncompatibleCheckpoint)
	if !errors.Is(wrapped, ErrIncompatibleCheckpoint) {
		t.Fatalf("errors.Is(%v, laya.ErrIncompatibleCheckpoint) = false", wrapped)
	}
	if got, want := ErrIncompatibleCheckpoint.Error(), "laya: incompatible checkpoint"; got != want {
		t.Fatalf("message = %q, want %q (docs/API.md)", got, want)
	}
}

// Task B.9: the ONNX backend's platform and device errors live in an internal
// package, which a caller outside this module cannot import. The root names
// are the same values, so errors.Is matches either.
func TestBackendErrorsReexported(t *testing.T) {
	for _, tc := range []struct {
		name       string
		root, leaf error
	}{
		{"ErrUnsupportedPlatform", ErrUnsupportedPlatform, onnx.ErrUnsupportedPlatform},
		{"ErrUnknownDevice", ErrUnknownDevice, onnx.ErrUnknownDevice},
	} {
		if !errors.Is(fmt.Errorf("onnx backend: %w", tc.leaf), tc.root) {
			t.Errorf("errors.Is(wrapped onnx.%s, laya.%s) = false", tc.name, tc.name)
		}
	}
}

// Task B.9: a device the backend does not know, such as Python's "mps", fails
// Open with an error a caller can match by its root name. The default loader
// opens the real backend, which rejects the device before it reads the graph
// or loads the runtime, so this runs on every platform without ONNX Runtime.
func TestOpenUnknownDevice(t *testing.T) {
	ck := t.TempDir()
	writeCheckpoint(t, ck)
	graph := writeGraph(t, t.TempDir(), "x")

	_, err := Open(context.Background(), ck, WithGraph(graph), WithDevice("mps"))
	if !errors.Is(err, ErrUnknownDevice) {
		t.Fatalf("Open(WithDevice(\"mps\")) = %v, want laya.ErrUnknownDevice", err)
	}
}
