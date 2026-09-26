package laya

import (
	"os/exec"
	"slices"
	"strings"
	"testing"
)

// runtimeModules are the ML runtime's modules: the ONNX binding and the
// dlopen layer beneath it. PLAN.md D9 names the first; purego is what the
// binding pulls in, and is no more acceptable in a zero-ML-dependency package.
var runtimeModules = []string{
	"github.com/shota3506/onnxruntime-purego",
	"github.com/ebitengine/purego",
}

// leafPackages are importable with no ML dependency (PLAN.md §8, D9).
var leafPackages = []string{"./lang", "./mailtext", "./presets", "./backend"}

// goList runs `go list` in the module root and returns its non-empty lines.
func goList(t *testing.T, args ...string) []string {
	t.Helper()
	// #nosec G204 -- fixed arguments; go list over this module's own packages.
	out, err := exec.CommandContext(t.Context(), "go", append([]string{"list"}, args...)...).CombinedOutput()
	if err != nil {
		t.Fatalf("go list %v: %v\n%s", args, err, out)
	}
	return slices.DeleteFunc(strings.Split(string(out), "\n"), func(s string) bool { return s == "" })
}

func isRuntime(path string) bool {
	return slices.ContainsFunc(runtimeModules, func(m string) bool {
		return path == m || strings.HasPrefix(path, m+"/")
	})
}

// TestNoMLDependency is §8's D9 check, run by every `go test`: lang, mailtext,
// presets and backend reach no ML runtime through any import chain. Under
// purego nothing links ORT symbols, so the import graph is what can be
// checked. GOOS is pinned to linux because the binding is build-constrained
// away elsewhere, which would let a darwin or windows run pass vacuously.
func TestNoMLDependency(t *testing.T) {
	t.Setenv("GOOS", "linux")
	t.Setenv("GOARCH", "amd64")
	for _, dep := range goList(t, append([]string{"-deps"}, leafPackages...)...) {
		if isRuntime(dep) {
			t.Errorf("%v depend on %s; they must stay free of the ML runtime (PLAN.md D9)", leafPackages, dep)
		}
	}
}

// TestRuntimeImportedOnlyByBackend: the binding is imported directly by
// internal/backend/onnx and nothing else, so D9's seam — swap the binding,
// touch one package — stays true (PLAN.md Task 6.3.4).
func TestRuntimeImportedOnlyByBackend(t *testing.T) {
	t.Setenv("GOOS", "linux")
	t.Setenv("GOARCH", "amd64")
	const owner = "github.com/MeKo-Christian/go-laya/internal/backend/onnx"

	var importers []string
	for _, line := range goList(t, "-f", "{{.ImportPath}} {{join .Imports \" \"}}", "./...") {
		pkg, imports, _ := strings.Cut(line, " ")
		if slices.ContainsFunc(strings.Fields(imports), isRuntime) {
			importers = append(importers, pkg)
		}
	}
	if !slices.Equal(importers, []string{owner}) {
		t.Errorf("packages importing the ML runtime = %v, want only %s", importers, owner)
	}
}
