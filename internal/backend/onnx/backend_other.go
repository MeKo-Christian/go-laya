//go:build !(!android && !ios && ((darwin && (amd64 || arm64)) || (linux && (amd64 || arm64 || loong64)) || (netbsd && (amd64 || arm64))))

package onnx

import (
	"context"

	"github.com/MeKo-Christian/go-laya/backend"
)

// Backend is the ONNX backend's placeholder on every target the binding does
// not compile for (see backend.go's build constraint). Open never returns one.
// The type exists so code naming it builds everywhere, which is what
// release.yml's `GOOS=windows go build ./...` checks.
type Backend struct{}

var _ backend.Backend = (*Backend)(nil)

// Open fails with ErrUnsupportedPlatform. The binding does not compile here:
// purego v0.9.0 has no Dlopen on Windows, js/wasm or the other BSDs, and its
// fakecgo fails CGO-free builds on freebsd and several linux architectures.
// Nobody has run the binding on Windows either (PLAN.md Task 6.7.1).
//
// An unknown Options.Device is ErrUnknownDevice here too, as on supported
// platforms, so a typo is reported the same way everywhere.
func Open(_ string, opts Options) (*Backend, error) {
	if _, err := parseDevice(opts.Device); err != nil {
		return nil, err
	}
	return nil, ErrUnsupportedPlatform
}

// Forward fails with ErrUnsupportedPlatform.
func (*Backend) Forward(context.Context, backend.Batch) (logits, act [][]float32, err error) {
	return nil, nil, ErrUnsupportedPlatform
}

// Device returns "": there is no session to run anywhere.
func (*Backend) Device() string { return "" }

// Close does nothing.
func (*Backend) Close() error { return nil }
