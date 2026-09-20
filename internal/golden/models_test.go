package golden

import (
	"os"
	"path/filepath"
	"testing"
)

func TestModelsRootPrefersTheEnvironment(t *testing.T) {
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "laya", "tokenizer"), 0o750); err != nil {
		t.Fatalf("seed a fake checkpoint tree: %v", err)
	}
	t.Setenv(modelsEnv, dir)

	got, ok := ModelsRoot()
	if !ok || got != dir {
		t.Errorf("ModelsRoot() = %q, %v; want %q, true", got, ok, dir)
	}
}

// A directory that exists but holds no checkpoint is not a models root. The
// worktree case makes this worth asserting: `git worktree add` produces a
// checkout with no models/ at all (it is gitignored), and an empty or absent
// directory that reported ok would skip every gated test while looking green.
func TestModelsRootRejectsATreeWithoutCheckpoints(t *testing.T) {
	t.Setenv(modelsEnv, t.TempDir())

	if got, ok := ModelsRoot(); ok {
		t.Errorf("ModelsRoot() = %q, true; want ok=false for a tree with no laya/ in it", got)
	}
}

func TestModelsRootRejectsAMissingDirectory(t *testing.T) {
	t.Setenv(modelsEnv, filepath.Join(t.TempDir(), "nope"))

	if got, ok := ModelsRoot(); ok {
		t.Errorf("ModelsRoot() = %q, true; want ok=false for a missing directory", got)
	}
}

func TestCheckpointDirNamesTheThreeSubfolders(t *testing.T) {
	for _, c := range []struct{ name, want string }{
		{English, "laya"},
		{Multilingual, filepath.Join("laya", Multilingual)},
		{TypedDecisions, filepath.Join("laya", TypedDecisions)},
	} {
		if got := CheckpointDir("/m", c.name); got != filepath.Join("/m", c.want) {
			t.Errorf("CheckpointDir(/m, %q) = %q, want %q", c.name, got, filepath.Join("/m", c.want))
		}
	}
}
