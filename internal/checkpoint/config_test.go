package checkpoint

import (
	"errors"
	"io/fs"
	"maps"
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

// Field hands a caller the raw JSON of one top-level key, so later checks
// (act_costs for 6.4.5, the temperature tables for M7) need not re-read the
// file, and it cannot be used to change what the Config holds.
func TestConfigField(t *testing.T) {
	cfg, err := LoadConfig(writeConfig(t, `{"encoder":null,"head_layers":2,"act_costs":{"escalate":0.5}}`))
	if err != nil {
		t.Fatal(err)
	}
	for key, want := range map[string]string{
		"head_layers": `2`,
		"act_costs":   `{"escalate":0.5}`,
		"encoder":     `null`,
	} {
		got, ok := cfg.Field(key)
		if !ok || string(got) != want {
			t.Errorf("Field(%q) = %s, %v; want %s, true", key, got, ok, want)
		}
	}
	if got, ok := cfg.Field("training"); ok {
		t.Errorf("Field(%q) = %s, true; want absent", "training", got)
	}

	got, _ := cfg.Field("head_layers")
	got[0] = '9'
	if again, _ := cfg.Field("head_layers"); string(again) != `2` {
		t.Errorf("writing to Field's result changed the Config: now %s", again)
	}
}

// TestConfigActWidth pins the act head's width as common.py:137 derives it:
// len(cfg.get("act_costs", {})) + 1, with Python's len, so an absent key is
// width 1 and a value len() rejects is ErrIncompatibleCheckpoint.
func TestConfigActWidth(t *testing.T) {
	tests := []struct {
		name  string
		costs string // the act_costs value; empty leaves the key out
		want  int
		err   bool
	}{
		{name: "shipped", costs: `{"escalate":0.5}`, want: 2},
		{name: "empty object", costs: `{}`, want: 1},
		{name: "absent", want: 1},
		{name: "two keys", costs: `{"escalate":0.5,"defer":0.25}`, want: 3},
		// json.loads keeps the last of a duplicated key.
		{name: "duplicate key", costs: `{"escalate":0.5,"escalate":1}`, want: 2},
		{name: "array", costs: `[0.5,1,2]`, want: 4},
		// len() of a str counts code points.
		{name: "string", costs: `"hé"`, want: 3},
		{name: "null", costs: `null`, err: true},
		{name: "number", costs: `0.5`, err: true},
		{name: "bool", costs: `true`, err: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := `{"encoder":"x","head_layers":2}`
			if tt.costs != "" {
				body = `{"encoder":"x","head_layers":2,"act_costs":` + tt.costs + `}`
			}
			cfg, err := LoadConfig(writeConfig(t, body))
			if err != nil {
				t.Fatal(err)
			}
			got, err := cfg.ActWidth()
			if tt.err {
				if !errors.Is(err, backend.ErrIncompatibleCheckpoint) || !strings.Contains(err.Error(), "act_costs") {
					t.Fatalf("ActWidth = %d, %v; want ErrIncompatibleCheckpoint naming act_costs", got, err)
				}
				return
			}
			if err != nil || got != tt.want {
				t.Fatalf("ActWidth = %d, %v; want %d", got, err, tt.want)
			}
		})
	}
}

// All three shipped checkpoints have the two-way act head.
func TestConfigActWidthShipped(t *testing.T) {
	root := golden.SkipWithoutModels(t)
	for _, ck := range []string{golden.English, golden.Multilingual, golden.TypedDecisions} {
		t.Run(ck, func(t *testing.T) {
			cfg, err := LoadConfig(golden.CheckpointDir(root, ck))
			if err != nil {
				t.Fatal(err)
			}
			if w, err := cfg.ActWidth(); err != nil || w != 2 {
				t.Fatalf("ActWidth = %d, %v; want 2", w, err)
			}
		})
	}
}

// TestConfigTemperatures pins the two temperature tables agent.py:194-195
// reads: cfg.get("temperature", [1.0, 1.0, 1.0]) and
// cfg.get("temperature_by_options", {}). Values are decoded as JSON float64
// without rounding, since agent.py:304-305 divides the logits by them. A
// present null is a value, not an absence: Python indexes (or .get()s) it and
// raises, so it fails here with ErrIncompatibleCheckpoint.
func TestConfigTemperatures(t *testing.T) {
	tests := []struct {
		name      string
		temp      string // the temperature value; empty leaves the key out
		byOptions string // the temperature_by_options value; empty leaves the key out
		want      [3]float64
		wantBy    map[string]float64
		errKey    string // the key the error must name; empty means no error
	}{
		{name: "absent", want: [3]float64{1, 1, 1}, wantBy: map[string]float64{}},
		{
			name: "shipped",
			temp: `[1.6369030475616455, 1.2514300346374512, 1.983399510383606]`,
			byOptions: `{"choice:3-5": 1.7601518630981445, "choice:11+": 0.10058280825614929,` +
				` "noul:2": 1.983399510383606}`,
			want: [3]float64{1.6369030475616455, 1.2514300346374512, 1.983399510383606},
			wantBy: map[string]float64{
				"choice:3-5": 1.7601518630981445, "choice:11+": 0.10058280825614929,
				"noul:2": 1.983399510383606,
			},
		},
		{name: "integers", temp: `[1, 2, 3]`, want: [3]float64{1, 2, 3}, wantBy: map[string]float64{}},
		{name: "only by_options", byOptions: `{"choice:2": 0.5}`, want: [3]float64{1, 1, 1}, wantBy: map[string]float64{"choice:2": 0.5}},
		{name: "empty by_options", temp: `[2, 2, 2]`, byOptions: `{}`, want: [3]float64{2, 2, 2}, wantBy: map[string]float64{}},
		{name: "temperature null", temp: `null`, errKey: "temperature"},
		{name: "temperature short", temp: `[1, 2]`, errKey: "temperature"},
		{name: "temperature long", temp: `[1, 2, 3, 4]`, errKey: "temperature"},
		{name: "temperature string element", temp: `[1, "2", 3]`, errKey: "temperature"},
		{name: "temperature bool element", temp: `[1, true, 3]`, errKey: "temperature"},
		{name: "temperature null element", temp: `[1, null, 3]`, errKey: "temperature"},
		{name: "temperature object", temp: `{"0": 1}`, errKey: "temperature"},
		{name: "temperature number", temp: `1.5`, errKey: "temperature"},
		{name: "by_options null", byOptions: `null`, errKey: "temperature_by_options"},
		{name: "by_options array", byOptions: `[1]`, errKey: "temperature_by_options"},
		{name: "by_options string value", byOptions: `{"choice:2": "1.5"}`, errKey: "temperature_by_options"},
		{name: "by_options null value", byOptions: `{"choice:2": null}`, errKey: "temperature_by_options"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := `{"encoder":"x","head_layers":2`
			if tt.temp != "" {
				body += `,"temperature":` + tt.temp
			}
			if tt.byOptions != "" {
				body += `,"temperature_by_options":` + tt.byOptions
			}
			cfg, err := LoadConfig(writeConfig(t, body+`}`))
			if err != nil {
				t.Fatal(err)
			}
			got, gotBy, err := cfg.Temperatures()
			if tt.errKey != "" {
				if !errors.Is(err, backend.ErrIncompatibleCheckpoint) || !strings.Contains(err.Error(), tt.errKey+":") {
					t.Fatalf("Temperatures = %v, %v, %v; want ErrIncompatibleCheckpoint naming %s", got, gotBy, err, tt.errKey)
				}
				return
			}
			if err != nil {
				t.Fatalf("Temperatures: %v", err)
			}
			if got != tt.want {
				t.Errorf("byQType = %v, want %v", got, tt.want)
			}
			if gotBy == nil || !maps.Equal(gotBy, tt.wantBy) {
				t.Errorf("byOptions = %v, want %v", gotBy, tt.wantBy)
			}
		})
	}
}

// TestConfigMaxLen pins the two sequence budgets agent.py:256-257 reads,
// cfg.get("max_len", 512) and cfg.get("head_max_len", 192). Python needs an
// int (common.py:82-86 slices with them), so a float literal or any other
// type fails; a non-positive int, which Python would accept and then truncate
// every sequence with, fails too (a deliberate loud failure, D23).
func TestConfigMaxLen(t *testing.T) {
	type accessor struct {
		key  string
		get  func(*Config) (int, error)
		dflt int
	}
	accessors := []accessor{
		{key: "max_len", get: (*Config).MaxLen, dflt: 512},
		{key: "head_max_len", get: (*Config).HeadMaxLen, dflt: 192},
	}
	tests := []struct {
		name  string
		value string // empty leaves the key out
		want  int    // 0 with value set means an error; with value empty, the default
	}{
		{name: "absent"},
		{name: "shipped 512", value: `512`, want: 512},
		{name: "shipped 1024", value: `1024`, want: 1024},
		{name: "one", value: `1`, want: 1},
		{name: "zero", value: `0`},
		{name: "negative", value: `-5`},
		{name: "float literal", value: `512.0`},
		{name: "exponent", value: `5e2`},
		{name: "string", value: `"512"`},
		{name: "null", value: `null`},
		{name: "bool", value: `true`},
		{name: "array", value: `[512]`},
		{name: "overflow", value: `99999999999999999999999`},
	}
	for _, a := range accessors {
		for _, tt := range tests {
			t.Run(a.key+"/"+tt.name, func(t *testing.T) {
				body := `{"encoder":"x","head_layers":2`
				if tt.value != "" {
					body += `,"` + a.key + `":` + tt.value
				}
				cfg, err := LoadConfig(writeConfig(t, body+`}`))
				if err != nil {
					t.Fatal(err)
				}
				got, err := a.get(cfg)
				want := tt.want
				if tt.value == "" {
					want = a.dflt
				}
				if want == 0 {
					if !errors.Is(err, backend.ErrIncompatibleCheckpoint) || !strings.Contains(err.Error(), a.key+":") {
						t.Fatalf("%s = %d, %v; want ErrIncompatibleCheckpoint naming %s", a.key, got, err, a.key)
					}
					return
				}
				if err != nil || got != want {
					t.Fatalf("%s = %d, %v; want %d", a.key, got, err, want)
				}
			})
		}
	}
}

// The shipped checkpoints decode: three temperatures each, and budgets of
// 512/192 or 1024/256.
func TestConfigBudgetsShipped(t *testing.T) {
	root := golden.SkipWithoutModels(t)
	for _, ck := range []string{golden.English, golden.Multilingual, golden.TypedDecisions} {
		t.Run(ck, func(t *testing.T) {
			cfg, err := LoadConfig(golden.CheckpointDir(root, ck))
			if err != nil {
				t.Fatal(err)
			}
			if _, _, err := cfg.Temperatures(); err != nil {
				t.Errorf("Temperatures: %v", err)
			}
			ml, err := cfg.MaxLen()
			if err != nil || (ml != 512 && ml != 1024) {
				t.Errorf("MaxLen = %d, %v", ml, err)
			}
			hml, err := cfg.HeadMaxLen()
			if err != nil || (hml != 192 && hml != 256) {
				t.Errorf("HeadMaxLen = %d, %v", hml, err)
			}
		})
	}
}
