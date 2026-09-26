//go:build windows || js || wasm

package onnx

import (
	"context"
	"errors"
	"testing"

	"github.com/MeKo-Christian/go-laya/backend"
)

func TestUnsupportedPlatform(t *testing.T) {
	if _, err := Open("model.onnx", Options{}); !errors.Is(err, ErrUnsupportedPlatform) {
		t.Fatalf("Open = %v, want ErrUnsupportedPlatform", err)
	}
	var b Backend
	if _, _, err := b.Forward(context.Background(), backend.Batch{}); !errors.Is(err, ErrUnsupportedPlatform) {
		t.Fatalf("Forward = %v, want ErrUnsupportedPlatform", err)
	}
}
