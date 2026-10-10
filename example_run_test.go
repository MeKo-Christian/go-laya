package laya_test

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/MeKo-Christian/go-laya"
)

const (
	// exampleEnv names the Example a child process of TestREADMEExamples runs.
	exampleEnv = "LAYA_README_EXAMPLE"
	// exampleDone, followed by the Example's name, is what a child prints
	// once the Example has returned.
	exampleDone = "README example done: "
	// childTimeoutMargin is how much earlier than the parent a child times
	// out, so that the child's goroutine dump is the one that is reported.
	childTimeoutMargin = 30 * time.Second
)

// readmeExamples are the seven README Examples that need weights, so `go test`
// compiles them but never runs them (they carry no Output comment). want lists
// what the README's trailing comments promise each one prints, in the order it
// prints them; an Example the README promises nothing for only has to finish.
var readmeExamples = []struct {
	name string
	run  func()
	want []string
}{
	{"ExampleRouter", ExampleRouter, []string{
		"Department: billing",
		"Routing   : english",
		"Department: billing",
		"Routing   : multilingual",
		"Repo      : convaiinnovations/laya/multilingual",
		"Reason    : non-Latin script (devanagari",
	}},
	{"ExampleRouter_Preload", ExampleRouter_Preload, []string{"[english multilingual]"}},
	{"ExampleRouter_Attach", ExampleRouter_Attach, nil},
	{"ExampleOpen", ExampleOpen, []string{"Department: billing"}},
	{"ExampleAnswer_confidence", ExampleAnswer_confidence, nil},
	{"ExampleAgent_presets", ExampleAgent_presets, nil},
	{"ExampleAgent_SetLimits", ExampleAgent_SetLimits, []string{"1024 512"}},
}

// TestREADMEExamples runs every weight-bound README Example (PLAN.md section
// 8, "Every README example compiles and runs"). CI only compiles them, so this
// is a local gate, like the tokenizer corpora.
//
// Gated: it skips under -short, without $LAYA_MODELS and without
// $LAYA_ONNX_DIR, a directory holding the exports of all three checkpoints
// (laya-<name>.onnx, or the -dynamo ones scripts/export_onnx.py --dynamo
// writes). $LAYA_MODELS is the opt-in AGENTS.md names for weight-bound tests;
// the Examples do not read it, but $LAYA_ONNX_DIR alone is a library setting a
// shell may export for other reasons. It also needs ONNX Runtime and network:
// the Examples open the Hub repo by name, so the config and the tokenizer come
// from the Hub at the pinned revision, or from the laya cache.
//
// Each Example runs in a child process of this test binary. The Examples end
// on log.Fatal and log.Panic as the README does, and an os.Exit cannot be
// recovered in-process; a child's exit status can, so a failing Example fails
// its own subtest and the rest still run.
func TestREADMEExamples(t *testing.T) {
	if testing.Short() {
		t.Skip("-short: needs the ONNX exports, ONNX Runtime and the Hub")
	}
	if name := os.Getenv(exampleEnv); name != "" {
		runExampleChild(t, name)
		return
	}
	if os.Getenv("LAYA_MODELS") == "" {
		t.Skip("weight-bound tests are not opted in (set LAYA_MODELS)")
	}
	exports := os.Getenv("LAYA_ONNX_DIR")
	if exports == "" {
		t.Skip("no exports (set LAYA_ONNX_DIR)")
	}
	exports = stageExports(t, exports)

	for _, ex := range readmeExamples {
		t.Run(ex.name, func(t *testing.T) {
			args := []string{"-test.run=^TestREADMEExamples$", "-test.paniconexit0"}
			// The child gets the parent's deadline less a margin: a parent
			// that times out panics without cancelling t.Context(), which
			// would leave a child holding a checkpoint running, and a child
			// that times out first dumps the goroutines of the hung Example.
			if d, ok := t.Deadline(); ok {
				left := time.Until(d) - childTimeoutMargin
				if left <= 0 {
					t.Fatalf("%s: less than %v left before the test deadline", ex.name, childTimeoutMargin)
				}
				args = append(args, "-test.timeout="+left.String())
			}
			// #nosec G204 -- re-runs this test binary with its own arguments.
			cmd := exec.CommandContext(t.Context(), os.Args[0], args...)
			cmd.Env = append(os.Environ(), exampleEnv+"="+ex.name, "LAYA_ONNX_DIR="+exports)
			out, err := cmd.CombinedOutput()
			t.Logf("%s:\n%s", ex.name, out)
			if err != nil {
				t.Fatalf("%s failed: %v", ex.name, err)
			}
			// Only a child that returned from the Example prints this, so
			// an early exit, a skip or a run that matched no test fails here.
			if !strings.Contains(string(out), exampleDone+ex.name) {
				t.Fatalf("%s did not run to its end", ex.name)
			}
			rest := string(out)
			for _, w := range ex.want {
				_, after, found := strings.Cut(rest, w)
				if !found {
					t.Errorf("%s does not print %q where the README promises it", ex.name, w)
					break
				}
				rest = after
			}
		})
	}
}

// runExampleChild runs one Example in the child process. A log.Fatal or
// log.Panic in it ends the process with a non-zero status, which the parent
// reports against the Example's name.
func runExampleChild(t *testing.T, name string) {
	for _, ex := range readmeExamples {
		if ex.name == name {
			ex.run()
			fmt.Fprintln(os.Stdout, exampleDone+name)
			return
		}
	}
	t.Fatalf("%s=%q names no README Example", exampleEnv, name)
}

// stageExports returns a directory holding laya-<name>.onnx for all three
// checkpoints, the name Open looks for. That is dir itself when it already
// does. Otherwise each name is hard-linked into a temporary directory inside
// dir, removed again by the cleanup, from its canonical export, else its
// -dynamo one, together with the external data under the name the graph
// refers to; no existing file in dir is written. A symlink would not do: the
// header check requires the external data to be a regular file. Like
// linkExports, it skips without an export or without hard links.
func stageExports(t *testing.T, dir string) string {
	t.Helper()
	names := []string{laya.ModelEnglish, laya.ModelMultilingual, laya.ModelTypedDecisions}
	canonical := true
	for _, name := range names {
		if _, err := os.Stat(filepath.Join(dir, "laya-"+name+".onnx")); err != nil {
			canonical = false
		}
	}
	if canonical {
		return dir
	}

	// Beside the exports, as linkExports does: a hard link cannot cross
	// filesystems, and $TMPDIR is often on another one.
	staged, err := os.MkdirTemp(dir, "readme-examples-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(staged) })
	for _, name := range names {
		src := filepath.Join(dir, "laya-"+name+".onnx")
		if _, err := os.Stat(src); err != nil {
			src = filepath.Join(dir, "laya-"+name+"-dynamo.onnx")
		}
		if _, err := os.Stat(src); err != nil {
			t.Skipf("no %s export in %s (run scripts/export_onnx.py --all --dynamo)", name, dir)
		}
		links := map[string]string{"laya-" + name + ".onnx": src}
		data, err := filepath.Glob(src + "*.data")
		if err != nil {
			t.Fatal(err)
		}
		for _, d := range data {
			links[filepath.Base(d)] = d
		}
		for dst, from := range links {
			if err := os.Link(from, filepath.Join(staged, dst)); err != nil {
				t.Skipf("cannot hard-link the %s export: %v", name, err)
			}
		}
	}
	return staged
}
