//go:build !android && !ios && ((darwin && (amd64 || arm64)) || (linux && (amd64 || arm64 || loong64)) || (netbsd && (amd64 || arm64)))

// Same constraint as backend.go: purego's Dlopen is only reachable where the
// binding itself builds.

package onnx

import (
	"fmt"

	"github.com/ebitengine/purego"
)

// ortAPIBase mirrors the C struct OrtApiBase: two function pointers, GetApi
// then GetVersionString. Both have been its whole layout since ORT 1.0, which
// is what makes the version readable from a library too old for the binding.
type ortAPIBase struct {
	_                uintptr // GetApi, which the binding calls itself
	getVersionString uintptr
}

// runtimeVersion reads the library's self-reported version through
// OrtGetApiBase, before the binding sees it. The binding reads the same string
// but only after asking for C API 23, so a pre-1.23 library fails there with
// an unnamed "failed to get OrtAPI"; reading it first is what lets Open name
// the version instead (Task 6.6.2).
//
// The handle is closed again; dlopen reference-counts, so the binding's own
// Dlopen of the same path gets a fresh reference.
func runtimeVersion(lib string) (string, error) {
	h, err := purego.Dlopen(lib, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return "", fmt.Errorf("load %s: %w", lib, err)
	}
	defer func() { _ = purego.Dlclose(h) }()

	sym, err := purego.Dlsym(h, "OrtGetApiBase")
	if err != nil {
		return "", fmt.Errorf("%s: no OrtGetApiBase, not an ONNX Runtime library: %w", lib, err)
	}
	var getBase func() *ortAPIBase
	purego.RegisterFunc(&getBase, sym)
	base := getBase()
	if base == nil || base.getVersionString == 0 {
		return "", fmt.Errorf("%s: OrtGetApiBase returned no GetVersionString", lib)
	}
	var version func() string
	purego.RegisterFunc(&version, base.getVersionString)
	return version(), nil
}
