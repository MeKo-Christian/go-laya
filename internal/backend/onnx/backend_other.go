//go:build windows || js || wasm

package onnx

import (
	"context"

	"github.com/MeKo-Christian/go-laya/backend"
)

// Backend is the ONNX backend's placeholder on platforms the binding cannot
// load a library on. Open never returns one; the type exists so code naming it
// builds everywhere, which is what release.yml's `GOOS=windows go build ./...`
// checks.
type Backend struct{}

var _ backend.Backend = (*Backend)(nil)

// Open fails with ErrUnsupportedPlatform. purego v0.9.0 defines Dlopen only on
// darwin, freebsd, linux and netbsd, and nobody has run the binding on Windows
// (PLAN.md Task 6.7.1), so claiming support here would be a claim no command
// backs.
func Open(string, Options) (*Backend, error) {
	return nil, ErrUnsupportedPlatform
}

// Forward fails with ErrUnsupportedPlatform.
func (*Backend) Forward(context.Context, backend.Batch) (logits, act [][]float32, err error) {
	return nil, nil, ErrUnsupportedPlatform
}

// Close does nothing.
func (*Backend) Close() error { return nil }
