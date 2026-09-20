// Package onnxspike is Spike S2 (PLAN.md §3): it proves the S1 ONNX export runs
// through a CGO-free ONNX Runtime binding, and that the runtime.AddCleanup defect
// go-pocket-tts hit (risk R5) is gone.
//
// The package is throwaway by design. Milestone M6 replaces it with
// internal/backend and deletes it; nothing but its own tests may import it.
//
// This file is deliberately pure stdlib and carries no build constraints, so
// `GOOS=windows go build ./...` (release.yml) keeps working. The binding is
// imported only from spike_test.go, which is constrained to the platforms purego
// can dlopen on.
package onnxspike

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
// asked for, and the whole point of S2.4 is knowing which library answered.
func findORTLibrary() (string, error) {
	for _, key := range ortLibraryEnv {
		path := os.Getenv(key)
		if path == "" {
			continue
		}
		// #nosec G703 -- the path is the developer's own environment variable naming a
		// local shared library, and it is only stat'ed. Nothing downloaded reaches here.
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
// The exports are gitignored build artefacts of scripts/export_onnx.py, so the
// default is where that script writes them, two levels up from this package.
// LAYA_ONNX_DIR overrides the directory.
func findModel(name string) (string, error) {
	dir := os.Getenv("LAYA_ONNX_DIR")
	if dir == "" {
		dir = filepath.Join("..", "..", "build", "onnx")
	}

	path := filepath.Join(dir, name)

	// #nosec G703 -- dir is a developer environment variable or a fixed relative default,
	// and name comes from a checked-in fixture; both are stat'ed, never opened.
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
