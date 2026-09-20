package golden

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// modelsEnv overrides where the checkpoints live. scripts/export_onnx.py:477
// reads the same variable, so one export is enough for both languages.
const modelsEnv = "LAYA_MODELS"

// The three checkpoint names, spelled as scripts/dump_python_parity.py:56-61
// and the fixture headers spell them.
const (
	English        = "english"
	Multilingual   = "multilingual"
	TypedDecisions = "typed-decisions"
)

// checkpointDirs maps a checkpoint name to its subfolder inside the bundle
// repo, mirroring CHECKPOINT_DIRS in scripts/dump_python_parity.py:56-61.
// English is the repo root; the other two are subfolders of it.
var checkpointDirs = map[string]string{
	English:        "",
	Multilingual:   Multilingual,
	TypedDecisions: TypedDecisions,
}

// ModelsRoot locates the downloaded checkpoint tree, reporting whether one is
// usable. It is $LAYA_MODELS when set, otherwise <repo>/models.
//
// "Usable" means the tree actually contains laya/, not merely that the path
// exists. That distinction is what makes the gate honest in a git worktree:
// models/ is gitignored, so `git worktree add` yields a checkout with no models
// directory, and a gate keyed on the default path alone would skip every
// checkpoint-backed test while the suite reported green. Run a worktree with
// LAYA_MODELS pointed at the main checkout.
func ModelsRoot() (string, bool) {
	root := os.Getenv(modelsEnv)
	if root == "" {
		// Located from this file rather than the working directory, the same
		// way Root does, so the gate answers identically from any package.
		_, self, _, ok := runtime.Caller(0)
		if !ok {
			return "", false
		}
		root = filepath.Join(filepath.Dir(self), "..", "..", "models")
	}
	root = filepath.Clean(root)
	// #nosec G703 -- root is a developer-supplied path to their own checkpoint
	// download, and this only stats it to decide whether to skip a test. There
	// is no untrusted input and nothing is read.
	if fi, err := os.Stat(filepath.Join(root, "laya")); err != nil || !fi.IsDir() {
		return "", false
	}
	return root, true
}

// CheckpointDir is the directory holding one checkpoint's tokenizer/ and
// encoder/ subfolders.
func CheckpointDir(root, checkpoint string) string {
	return filepath.Join(root, "laya", checkpointDirs[checkpoint])
}

// SkipWithoutModels returns the models root, skipping the test when there is
// none. The message is actionable on purpose: a silent skip on a machine that
// does have the checkpoints is how a parity suite stops testing parity.
func SkipWithoutModels(tb testing.TB) string {
	tb.Helper()

	if testing.Short() {
		tb.Skip("-short: needs the downloaded checkpoints")
	}
	root, ok := ModelsRoot()
	if !ok {
		tb.Skipf("no checkpoints: set %s to a tree containing laya/ "+
			"(see scripts/README.md; a git worktree has no models/ of its own)", modelsEnv)
	}
	return root
}
