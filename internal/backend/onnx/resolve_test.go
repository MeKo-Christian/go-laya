package onnx

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/MeKo-Christian/go-laya/internal/ortlib"
)

// touch creates an empty file standing in for a shared library; resolution
// only ever stats it.
func touch(t *testing.T, path string) string {
	t.Helper()
	if err := os.WriteFile(path, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// withCandidates swaps the platform fallback list and the download cache for
// the duration of a test, so the result does not depend on what the machine
// running it has installed or downloaded. cached is what the cache answers:
// a path, or an error such as ortlib.ErrNotCached.
func withCandidates(t *testing.T, cached string, cachedErr error, paths ...string) {
	t.Helper()
	prevPaths, prevCache := ortLibraryCandidates, cachedORTLibrary
	ortLibraryCandidates = paths
	cachedORTLibrary = func(string) (string, error) { return cached, cachedErr }
	t.Cleanup(func() { ortLibraryCandidates, cachedORTLibrary = prevPaths, prevCache })
}

// TestFindORTLibrary is Task 6.6.3: LAYA_ORT_LIB, then ORT_LIBRARY_PATH, then
// the verified download (Task 6.6.1), then the platform candidates, and a
// variable that is set but points nowhere is an error rather than a fallback.
func TestFindORTLibrary(t *testing.T) {
	dir := t.TempDir()
	laya := touch(t, filepath.Join(dir, "laya.so"))
	generic := touch(t, filepath.Join(dir, "generic.so"))
	candidate := touch(t, filepath.Join(dir, "candidate.so"))
	downloaded := touch(t, filepath.Join(dir, "downloaded.so"))
	missing := filepath.Join(dir, "missing.so")

	cases := []struct {
		name          string
		layaEnv, ort  string
		cached        string
		cachedErr     error
		candidates    []string
		want, wantErr string
		wantNotFound  bool
		wantIs        error
	}{
		{name: "LAYA_ORT_LIB wins", layaEnv: laya, ort: generic, candidates: []string{candidate}, want: laya},
		{name: "ORT_LIBRARY_PATH next", ort: generic, candidates: []string{candidate}, want: generic},
		{name: "first existing candidate", candidates: []string{missing, candidate}, want: candidate},
		{name: "download before candidates", cached: downloaded, candidates: []string{candidate}, want: downloaded},
		{name: "variables before download", ort: generic, cached: downloaded, want: generic},
		{
			name: "no download on this platform", cachedErr: ortlib.ErrUnsupportedPlatform,
			candidates: []string{candidate}, want: candidate,
		},
		{
			name: "tampered download is an error", cachedErr: fmt.Errorf("%w: x", ortlib.ErrHashMismatch),
			candidates: []string{candidate}, wantIs: ortlib.ErrHashMismatch,
		},
		{
			name: "unreadable download is an error", cachedErr: fmt.Errorf("open x: %w", fs.ErrPermission),
			candidates: []string{candidate}, wantIs: fs.ErrPermission,
		},
		{
			name: "LAYA_ORT_LIB set but missing", layaEnv: missing, ort: generic, candidates: []string{candidate},
			wantErr: "LAYA_ORT_LIB=",
		},
		{
			name: "ORT_LIBRARY_PATH set but missing", ort: missing, candidates: []string{candidate},
			wantErr: "ORT_LIBRARY_PATH=",
		},
		{name: "nothing anywhere", candidates: []string{missing}, wantNotFound: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("LAYA_ORT_LIB", tc.layaEnv)
			t.Setenv("ORT_LIBRARY_PATH", tc.ort)
			cachedErr := tc.cachedErr
			if tc.cached == "" && cachedErr == nil {
				cachedErr = ortlib.ErrNotCached
			}
			withCandidates(t, tc.cached, cachedErr, tc.candidates...)

			got, err := findORTLibrary("")
			switch {
			case tc.wantIs != nil:
				if !errors.Is(err, tc.wantIs) {
					t.Fatalf("got %q, %v; want %v", got, err, tc.wantIs)
				}
			case tc.wantErr != "":
				if err == nil {
					t.Fatalf("got %q, want an error naming %s", got, tc.wantErr)
				}
				if !strings.Contains(err.Error(), tc.wantErr) || !errors.Is(err, fs.ErrNotExist) {
					t.Fatalf("error %q: want it to name %s and wrap fs.ErrNotExist", err, tc.wantErr)
				}
			case tc.wantNotFound:
				if !errors.Is(err, errNotFound) {
					t.Fatalf("got %q, %v; want errNotFound", got, err)
				}
				if !strings.Contains(err.Error(), "LAYA_ORT_LIB") || !strings.Contains(err.Error(), "cmd/laya-ort") {
					t.Errorf("error %q does not say which variable to set or how to download", err)
				}
			default:
				if err != nil || got != tc.want {
					t.Fatalf("got %q, %v; want %q", got, err, tc.want)
				}
			}
		})
	}
}

// pinnedLibrary puts a file where cmd/laya-ort puts this platform's pinned
// library under the cache root dir, with bytes that fail its hash. The test
// is skipped on a platform with no pinned download.
func pinnedLibrary(t *testing.T, dir string) {
	t.Helper()
	rel, err := ortlib.Pinned(runtime.GOOS, runtime.GOARCH)
	if err != nil {
		t.Skipf("no pinned ONNX Runtime download here: %v", err)
	}
	lib := filepath.Join(dir, "onnxruntime", strings.TrimSuffix(rel.Archive, ".tgz"), path.Base(rel.Member))
	if err := os.MkdirAll(filepath.Dir(lib), 0o750); err != nil {
		t.Fatal(err)
	}
	touch(t, lib)
}

// Task 7.7.7 (D29): a cache directory moves the download lookup there from
// $LAYA_CACHE. The copy there fails its pinned hash, which is an error; the
// one $LAYA_CACHE names does not exist, so looking there would fall through
// to the candidate instead. Runs the real cache lookup, not a stand-in.
func TestFindORTLibraryCacheDir(t *testing.T) {
	cache := t.TempDir()
	pinnedLibrary(t, cache)
	t.Setenv("LAYA_CACHE", t.TempDir())
	t.Setenv("LAYA_ORT_LIB", "")
	t.Setenv("ORT_LIBRARY_PATH", "")
	prev := ortLibraryCandidates
	ortLibraryCandidates = []string{touch(t, filepath.Join(t.TempDir(), "candidate.so"))}
	t.Cleanup(func() { ortLibraryCandidates = prev })

	if got, err := findORTLibrary(cache); !errors.Is(err, ortlib.ErrHashMismatch) {
		t.Fatalf("findORTLibrary(%q) = %q, %v; want ortlib.ErrHashMismatch from the copy there", cache, got, err)
	}
	if got, err := findORTLibrary(""); err != nil || got != ortLibraryCandidates[0] {
		t.Errorf("findORTLibrary(\"\") = %q, %v; want the candidate, $LAYA_CACHE having no download", got, err)
	}
}
