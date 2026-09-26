package laya

import (
	"errors"
	"fmt"
	"testing"

	"github.com/MeKo-Christian/go-laya/backend"
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
