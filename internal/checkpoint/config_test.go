package checkpoint

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/MeKo-Christian/go-laya/backend"
	"github.com/MeKo-Christian/go-laya/internal/golden"
)

// writeConfig puts body where LoadConfig looks for it and returns the
// checkpoint directory.
func writeConfig(t *testing.T, body string) string {
	t.Helper()

	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, ConfigFile), []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return dir
}

// TestLoadConfig pins _verify_compatibility's first check (agent.py:51-57):
// the keys "encoder" and "head_layers" must be present, whatever their value,
// and a config without them is ErrIncompatibleCheckpoint naming exactly the
// missing ones. Anything that is not a JSON object fails the same way.
func TestLoadConfig(t *testing.T) {
	tests := []struct {
		name string
		body string
		want []string // substrings of the error; nil means no error
		not  []string // substrings the error must not contain
	}{
		{name: "minimal", body: `{"encoder":"answerdotai/ModernBERT-large","head_layers":2}`},
		// `k not in cfg` tests presence, so null passes upstream too.
		{name: "null values", body: `{"encoder":null,"head_layers":null}`},
		{name: "extra keys", body: `{"encoder":"x","head_layers":2,"act_costs":{"escalate":0.5}}`},
		{
			name: "no encoder", body: `{"head_layers":2}`,
			want: []string{"missing keys encoder"}, not: []string{"head_layers"},
		},
		{
			name: "no head_layers", body: `{"encoder":"x"}`,
			want: []string{"missing keys head_layers"}, not: []string{"encoder"},
		},
		{name: "empty object", body: `{}`, want: []string{"missing keys encoder, head_layers"}},
		{
			name: "nested only", body: `{"training":{"encoder":"x","head_layers":2}}`,
			want: []string{"missing keys encoder, head_layers"},
		},
		{name: "array", body: `[]`, want: []string{"not a JSON object"}},
		{name: "string", body: `"encoder"`, want: []string{"not a JSON object"}},
		{name: "null", body: `null`, want: []string{"not a JSON object"}},
		{name: "invalid JSON", body: `{"encoder":`, want: []string{"config"}},
		{
			name: "oversized", body: `{"encoder":"` + strings.Repeat("x", maxConfigSize) + `","head_layers":2}`,
			want: []string{"larger than"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := writeConfig(t, tt.body)
			cfg, err := LoadConfig(dir)
			if tt.want == nil {
				if err != nil {
					t.Fatalf("LoadConfig = %v, want nil", err)
				}
				if cfg == nil {
					t.Fatal("LoadConfig returned a nil *Config and no error")
				}
				return
			}
			if !errors.Is(err, backend.ErrIncompatibleCheckpoint) {
				t.Fatalf("LoadConfig = %v, want ErrIncompatibleCheckpoint", err)
			}
			for _, w := range tt.want {
				if !strings.Contains(err.Error(), w) {
					t.Errorf("error %q does not mention %q", err, w)
				}
			}
			// The temp dir is named after the subtest, so leave it out.
			msg := strings.ReplaceAll(err.Error(), dir, "<dir>")
			for _, n := range tt.not {
				if strings.Contains(msg, n) {
					t.Errorf("error %q names %q, which is present", err, n)
				}
			}
		})
	}
}

// A directory without rl_agent_config.json is not a laya checkpoint
// (agent.py:139-143), and a caller may also want to tell that apart from a
// broken one.
func TestLoadConfigMissingFile(t *testing.T) {
	_, err := LoadConfig(t.TempDir())
	if !errors.Is(err, backend.ErrIncompatibleCheckpoint) {
		t.Errorf("LoadConfig = %v, want ErrIncompatibleCheckpoint", err)
	}
	if !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("LoadConfig = %v, want fs.ErrNotExist", err)
	}
	if err != nil && !strings.Contains(err.Error(), ConfigFile) {
		t.Errorf("error %q does not name %s", err, ConfigFile)
	}
}

// The three shipped checkpoints all pass.
func TestLoadConfigShipped(t *testing.T) {
	root := golden.SkipWithoutModels(t)
	for _, ck := range []string{golden.English, golden.Multilingual, golden.TypedDecisions} {
		t.Run(ck, func(t *testing.T) {
			if _, err := LoadConfig(golden.CheckpointDir(root, ck)); err != nil {
				t.Fatalf("LoadConfig: %v", err)
			}
		})
	}
}
