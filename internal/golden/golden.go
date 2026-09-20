// Package golden loads the checked-in parity corpus under testdata/.
//
// The corpus lives once at the repo root, not once per package: it is generated
// as a set by scripts/dump_python_parity.py and regenerating it is a single
// reviewed diff (PLAN.md R6). Several packages assert against the same file --
// render.jsonl backs both jsonx and internal/prompt -- so the loader lives here
// rather than being copied into each _test.go and drifting.
package golden

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// maxLine is generous because logits.jsonl carries whole tensors on one line,
// well past bufio's 64 KiB default.
const maxLine = 16 * 1024 * 1024

// Case is one record of a fixture. Raw is kept so a caller can decode the
// per-fixture fields itself, with an order-preserving decoder where key order
// is the assertion.
type Case struct {
	Kind string
	Name string
	Fn   string
	Raw  json.RawMessage
}

// Unmarshal decodes the whole record into v.
func (c Case) Unmarshal(tb testing.TB, v any) {
	tb.Helper()
	if err := json.Unmarshal(c.Raw, v); err != nil {
		tb.Fatalf("%s: decode case: %v", c.Name, err)
	}
}

// Load returns the case records of testdata/<fixture>.jsonl, header excluded.
// The header is TestGoldenProvenance's business; callers want the cases.
func Load(tb testing.TB, fixture string) []Case {
	tb.Helper()

	path := filepath.Join(Root(tb), "testdata", fixture+".jsonl")
	f, err := os.Open(path)
	if err != nil {
		tb.Fatalf("open %s: %v", path, err)
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), maxLine)

	var cases []Case
	for line := 0; sc.Scan(); line++ {
		var c Case
		raw := append(json.RawMessage(nil), sc.Bytes()...)
		var head struct {
			Kind string `json:"kind"`
			Name string `json:"name"`
			Fn   string `json:"fn"`
		}
		if err := json.Unmarshal(raw, &head); err != nil {
			tb.Fatalf("%s line %d: %v", path, line+1, err)
		}
		if head.Kind != "case" {
			continue
		}
		c = Case{Kind: head.Kind, Name: head.Name, Fn: head.Fn, Raw: raw}
		cases = append(cases, c)
	}
	if err := sc.Err(); err != nil {
		tb.Fatalf("read %s: %v", path, err)
	}
	if len(cases) == 0 {
		tb.Fatalf("%s has no cases; an empty corpus is a broken checkout, not nothing to assert", path)
	}
	return cases
}

// ByFn returns the cases whose "fn" field is one of fns, failing if none match
// -- a filter that silently selects nothing is a test that silently passes.
func ByFn(tb testing.TB, fixture string, fns ...string) []Case {
	tb.Helper()

	want := make(map[string]bool, len(fns))
	for _, fn := range fns {
		want[fn] = true
	}

	var out []Case
	for _, c := range Load(tb, fixture) {
		if want[c.Fn] {
			out = append(out, c)
		}
	}
	if len(out) == 0 {
		tb.Fatalf("%s.jsonl has no cases with fn in %v", fixture, fns)
	}
	return out
}

// Root returns the repository root, located from this file rather than from the
// working directory, so the loader works from any package's test.
func Root(tb testing.TB) string {
	tb.Helper()

	_, self, _, ok := runtime.Caller(0)
	if !ok {
		tb.Fatal("runtime.Caller failed; cannot locate the repo root")
	}
	root := filepath.Join(filepath.Dir(self), "..", "..")
	abs, err := filepath.Abs(root)
	if err != nil {
		tb.Fatalf("resolve repo root: %v", err)
	}
	if _, err := os.Stat(filepath.Join(abs, "go.mod")); err != nil {
		tb.Fatalf("%s is not the repo root: %v", abs, err)
	}
	return abs
}
