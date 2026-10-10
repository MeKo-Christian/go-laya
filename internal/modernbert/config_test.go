package modernbert

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/MeKo-Christian/go-laya/internal/golden"
)

// The keys of the two checkpoints' encoder config.json that Config reads,
// copied verbatim (models/laya/encoder/config.json and
// models/laya/multilingual/encoder/config.json, transformers_version 5.0.0),
// so that CI checks them without the download. TestCheckpointConfigs checks
// the copies against the real files where they exist.
const (
	modernBERTLargeConfig = `{
  "global_attn_every_n_layers": 3,
  "global_rope_theta": null,
  "layer_types": ["full_attention", "sliding_attention", "sliding_attention", "full_attention",
    "sliding_attention", "sliding_attention", "full_attention", "sliding_attention",
    "sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
    "full_attention", "sliding_attention", "sliding_attention", "full_attention",
    "sliding_attention", "sliding_attention", "full_attention", "sliding_attention",
    "sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
    "full_attention", "sliding_attention", "sliding_attention", "full_attention"],
  "local_attention": 128,
  "local_rope_theta": null,
  "num_attention_heads": 16,
  "num_hidden_layers": 28,
  "rope_parameters": {
    "full_attention": {"rope_theta": 160000.0, "rope_type": "default"},
    "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
  },
  "rope_theta": null
}`
	mmBERTBaseConfig = `{
  "global_attn_every_n_layers": 3,
  "global_rope_theta": null,
  "layer_types": ["full_attention", "sliding_attention", "sliding_attention", "full_attention",
    "sliding_attention", "sliding_attention", "full_attention", "sliding_attention",
    "sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
    "full_attention", "sliding_attention", "sliding_attention", "full_attention",
    "sliding_attention", "sliding_attention", "full_attention", "sliding_attention",
    "sliding_attention", "full_attention"],
  "local_attention": 128,
  "local_rope_theta": null,
  "num_attention_heads": 12,
  "num_hidden_layers": 22,
  "rope_parameters": {
    "full_attention": {"rope_theta": 160000, "rope_type": "default"},
    "sliding_attention": {"rope_theta": 160000, "rope_type": "default"}
  },
  "rope_theta": null
}`
)

// everyThird returns the layer types of n layers with every third one, from
// layer 0, full at theta full and the rest sliding at theta sliding with
// local_attention 128.
func everyThird(n int, full, sliding float64) []LayerType {
	out := make([]LayerType, n)
	for i := range out {
		out[i] = SlidingAttention(sliding, 128)
		if i%3 == 0 {
			out[i] = FullAttention(full)
		}
	}
	return out
}

// TestCheckpointLayerTypes pins both checkpoints' layer plan
// (docs/ARCHITECTURE.md §1.2): ModernBERT-large has 28 layers, 0, 3, ..., 27
// full at theta 160000 and the rest sliding at 10000; mmBERT-base has 22,
// 0, 3, ..., 21 full, and theta 160000 on both types. The window is 64.
func TestCheckpointLayerTypes(t *testing.T) {
	for _, tc := range []struct {
		name, raw string
		heads     int
		want      []LayerType
	}{
		{"modernbert-large", modernBERTLargeConfig, 16, everyThird(28, 160000, 10000)},
		{"mmbert-base", mmBERTBaseConfig, 12, everyThird(22, 160000, 160000)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg, err := ParseConfig([]byte(tc.raw))
			if err != nil {
				t.Fatalf("ParseConfig: %v", err)
			}
			got, err := cfg.Layers()
			if err != nil {
				t.Fatalf("Layers: %v", err)
			}
			if !slices.Equal(got, tc.want) {
				t.Fatalf("Layers() = %+v\nwant %+v", got, tc.want)
			}
			if got[len(got)-1].Sliding || got[0].Sliding || got[1].Window != 64 {
				t.Errorf("first/last layer or window wrong: %+v", got)
			}
			if cfg.NumAttentionHeads != tc.heads {
				t.Errorf("NumAttentionHeads = %d, want %d", cfg.NumAttentionHeads, tc.heads)
			}
		})
	}
}

// TestCheckpointConfigs checks the copies above against the downloaded
// checkpoints, so that the CI test cannot drift from what Task 8.10 will read.
func TestCheckpointConfigs(t *testing.T) {
	root := golden.SkipWithoutModels(t)
	for checkpoint, copied := range map[string]string{
		golden.English:      modernBERTLargeConfig,
		golden.Multilingual: mmBERTBaseConfig,
	} {
		path := filepath.Join(golden.CheckpointDir(root, checkpoint), "encoder", "config.json")
		raw, err := os.ReadFile(path) // #nosec G304 -- the developer's own checkpoint download
		if err != nil {
			t.Fatalf("read %s: %v", path, err)
		}
		onDisk, err := ParseConfig(raw)
		if err != nil {
			t.Fatalf("%s: ParseConfig: %v", path, err)
		}
		want, err := ParseConfig([]byte(copied))
		if err != nil {
			t.Fatalf("%s copy: ParseConfig: %v", checkpoint, err)
		}
		gotLayers, err := onDisk.Layers()
		if err != nil {
			t.Fatalf("%s: Layers: %v", path, err)
		}
		wantLayers, _ := want.Layers()
		if !slices.Equal(gotLayers, wantLayers) || onDisk.NumAttentionHeads != want.NumAttentionHeads {
			t.Errorf("%s: %+v, heads %d; the copy in config_test.go says %+v, heads %d",
				path, gotLayers, onDisk.NumAttentionHeads, wantLayers, want.NumAttentionHeads)
		}
	}
}

// configCase is a "config" record of testdata/ops.json: the keyword arguments
// the dumper passed to the pinned ModernBertConfig, and what it resolved them
// to, or the exception it raised.
type configCase struct {
	Name           string                    `json:"name"`
	Config         json.RawMessage           `json:"config"`
	Error          *string                   `json:"error"`
	NumHidden      int                       `json:"num_hidden_layers"`
	LayerTypes     []string                  `json:"layer_types"`
	Rope           map[string]configCaseRope `json:"rope"`
	LocalAttention int                       `json:"local_attention"`
	SlidingWindow  int                       `json:"sliding_window"`
	HiddenSize     int                       `json:"hidden_size"`
	Heads          int                       `json:"num_attention_heads"`
	HeadDim        *int                      `json:"head_dim"`
}

type configCaseRope struct {
	RopeType  *string  `json:"rope_type"`
	RopeTheta *float64 `json:"rope_theta"`
}

// goRejects returns why Config.Layers must fail on c, or "" when it must
// return what transformers resolved. It fails where transformers raises, and
// where transformers goes on into something Go does not implement or that
// cannot run: rope_scaling, a RoPE other than "default", a theta of None or
// not positive, a negative window, a head_dim other than hidden / heads.
func goRejects(t *testing.T, c configCase) string {
	t.Helper()

	if c.Error != nil {
		return "transformers raises " + *c.Error
	}
	var keys map[string]json.RawMessage
	if err := json.Unmarshal(c.Config, &keys); err != nil {
		t.Fatalf("%s: config: %v", c.Name, err)
	}
	if v, ok := keys["rope_scaling"]; ok && string(v) != "null" {
		return "rope_scaling"
	}
	if c.LocalAttention < 0 {
		return "a negative window"
	}
	for name, r := range c.Rope {
		if r.RopeType == nil || *r.RopeType != "default" {
			return name + ": a RoPE other than default"
		}
		if r.RopeTheta == nil || !(*r.RopeTheta > 0) {
			return name + ": a theta of None or not positive"
		}
	}
	if c.HeadDim != nil && *c.HeadDim*c.Heads != c.HiddenSize {
		return "head_dim other than hidden / heads"
	}
	return ""
}

// TestConfigMatchesTransformers runs Config over each "config" record: the
// layer plan, the thetas, the window and the heads must be what the pinned
// ModernBertConfig resolved (configuration_modernbert.py:113-152), or, where
// goRejects says so, an error. The records cover the derivation from
// global_attn_every_n_layers, explicit layer_types, each theta fallback, null
// against a missing key, case-folded keys and the RoPE settings Go refuses.
func TestConfigMatchesTransformers(t *testing.T) {
	raws := casesOf(t, loadOps(t), "config")
	outcomes := map[string]int{}
	for _, raw := range raws {
		var c configCase
		if err := json.Unmarshal(raw, &c); err != nil {
			t.Fatalf("decode a config case: %v", err)
		}
		t.Run(c.Name, func(t *testing.T) {
			cfg, err := ParseConfig(c.Config)
			var got []LayerType
			if err == nil {
				got, err = cfg.Layers()
			}
			if reason := goRejects(t, c); reason != "" {
				if c.Error != nil {
					outcomes["transformers raises"]++
				} else {
					outcomes["go refuses"]++
				}
				if err == nil {
					t.Fatalf("%s: want an error (%s), got %+v", c.Config, reason, got)
				}
				t.Logf("%s: %v", reason, err)
				return
			}
			outcomes["same plan"]++
			if err != nil {
				t.Fatalf("%s: %v; transformers resolves %v", c.Config, err, c.LayerTypes)
			}
			if len(got) != len(c.LayerTypes) || len(got) != c.NumHidden {
				t.Fatalf("%d layers, transformers %v", len(got), c.LayerTypes)
			}
			for i, l := range got {
				name := fullAttention
				if l.Sliding {
					name = slidingAttention
				}
				if name != c.LayerTypes[i] || l.RopeTheta != *c.Rope[name].RopeTheta ||
					(l.Sliding && l.Window != c.SlidingWindow) {
					t.Errorf("layer %d: %s theta %g window %d; transformers %s theta %g window %d",
						i, name, l.RopeTheta, l.Window, c.LayerTypes[i], *c.Rope[c.LayerTypes[i]].RopeTheta, c.SlidingWindow)
				}
			}
			if cfg.NumAttentionHeads != c.Heads {
				t.Errorf("NumAttentionHeads %d, transformers %d", cfg.NumAttentionHeads, c.Heads)
			}
		})
	}
	for _, o := range []string{"same plan", "transformers raises", "go refuses"} {
		if outcomes[o] == 0 {
			t.Errorf("no config case where %s; the records must cover all three outcomes", o)
		}
	}
	t.Logf("%d config cases: %v", len(raws), outcomes)
}

// TestConfigParseErrors covers the inputs that are not keyword arguments:
// config.json that is not an object, or has a value of the wrong type.
func TestConfigParseErrors(t *testing.T) {
	for _, raw := range []string{
		`null`,
		`[]`,
		`{"num_hidden_layers": "x"}`,
		`{"rope_parameters": {"full_attention": {"rope_theta": "x"}}}`,
		`{"rope_parameters": {"full_attention": []}}`,
	} {
		if _, err := ParseConfig([]byte(raw)); err == nil {
			t.Errorf("ParseConfig(%s) accepted it", raw)
		}
	}
	// A Config built by hand rather than parsed: no heads would divide by zero.
	if _, err := (Config{NumHiddenLayers: 1, NumAttentionHeads: 0, HiddenSize: 8}).Layers(); err == nil {
		t.Error("Layers accepted 0 heads")
	}
}
