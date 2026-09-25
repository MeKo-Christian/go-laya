// Package onnx runs the exported DecisionModel graph through ONNX Runtime, via the
// CGO-free binding github.com/shota3506/onnxruntime-purego (PLAN.md D5).
//
// It absorbed internal/onnxspike (PLAN.md Task 6.10): the library-resolution chain,
// the external-data check, the fixture schema, the finalizer regression and the S3
// benchmarks came from Spikes S2 and S3 unchanged.
//
// This file is deliberately pure stdlib and carries no build constraints. Everything
// that imports the binding is constrained to the platforms purego can dlopen on, so
// `GOOS=windows go build ./...` (release.yml) keeps working.
package onnx

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

// errNotFound is returned when neither the environment nor the fallback
// candidates point at an existing file. Callers turn it into a test skip.
var errNotFound = errors.New("not found")

// ortLibraryEnv lists the environment variables consulted for the ONNX Runtime
// shared library, in order. The chain mirrors go-pocket-tts
// (internal/onnx/runtime.go:71-121) so a developer with one repo set up already
// has the other working; LAYA_ORT_LIB is this repo's own name and wins.
var ortLibraryEnv = []string{"LAYA_ORT_LIB", "ORT_LIBRARY_PATH"}

// ortLibraryCandidates are the default install locations checked when no
// environment variable is set.
var ortLibraryCandidates = []string{
	"/usr/local/lib/libonnxruntime.so",
	"/usr/lib/libonnxruntime.so",
	"/usr/lib/x86_64-linux-gnu/libonnxruntime.so",
	"/opt/homebrew/lib/libonnxruntime.dylib",
	"/usr/local/lib/libonnxruntime.dylib",
}

// findORTLibrary resolves the ONNX Runtime shared library.
//
// A variable that is set but points nowhere is an error rather than a fallback:
// silently ignoring it would run against a different runtime than the caller
// asked for, and the whole point of pinning a runtime (R7) is knowing which library answered.
func findORTLibrary() (string, error) {
	for _, key := range ortLibraryEnv {
		path := os.Getenv(key)
		if path == "" {
			continue
		}
		// #nosec G703 -- the path is the developer's own environment variable naming a
		// local shared library, and it is only stat'ed. This holds only while every path
		// here is developer-supplied: Hub-derived paths (internal/hub) must never be routed
		// through it.
		if _, err := os.Stat(path); err != nil {
			return "", fmt.Errorf("%s=%q: %w", key, path, err)
		}
		return path, nil
	}

	for _, path := range ortLibraryCandidates {
		if _, err := os.Stat(path); err == nil {
			return path, nil
		}
	}

	return "", fmt.Errorf("ONNX Runtime shared library: %w (set %s)", errNotFound, ortLibraryEnv[0])
}

// findModel resolves the .onnx file named by the fixture.
//
// LAYA_ONNX_DIR must be set explicitly; there is no implicit default. The
// exports are gitignored build artefacts of scripts/export_onnx.py, and a
// default pointing at build/onnx meant that any developer with an export on
// disk pulled 1.7 GB into every `go test ./...`. The `just test-onnx` and
// `just bench-onnx` recipes set the variable to build/onnx themselves.
func findModel(name string) (string, error) {
	dir := os.Getenv("LAYA_ONNX_DIR")
	if dir == "" {
		return "", fmt.Errorf("model %s: %w (set LAYA_ONNX_DIR)", name, errNotFound)
	}

	path := filepath.Join(dir, name)

	// #nosec G703 -- dir is a developer environment variable and name comes from a
	// checked-in fixture; both are stat'ed, never opened. Same caveat as above.
	if _, err := os.Stat(path); err != nil {
		return "", fmt.Errorf("%s: %w", path, err)
	}

	// The exports exceed the 2 GB protobuf limit and are split into a graph file
	// plus an external-data blob. ORT resolves that blob relative to the model
	// path, which is why the session must be created from a path and never from
	// a reader -- checking for it here turns a confusing ORT error into a skip.
	// #nosec G703 -- same path as above, with the external-data suffix.
	if _, err := os.Stat(path + ".data"); err != nil {
		return "", fmt.Errorf("external data %s.data: %w", path, err)
	}

	return path, nil
}
